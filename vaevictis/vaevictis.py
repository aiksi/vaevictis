import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np
from .tsne_helper_njit import compute_transition_probability
from .cense_helper_njit import dist_to_knn as cense_dtk, remove_asym
from .umap_helper_njit import dist_to_knn as umap_dtk, find_ab_params, smooth_knn_dist, compute_membership_strengths, simplicial_graph_from_dist, euclidean_embedding
from .ivis_helper import input_compute, pn_loss_g, euclidean_distance, cosine_distance
from .knn_annoy import build_annoy_index, extract_knn
from tensorflow.keras.callbacks import EarlyStopping
import os
import json
import ipdb
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
K.set_floatx('float64')

eps_std = tf.constant(1e-2, dtype=tf.float64)
eps_sq = eps_std ** 2
eta = tf.constant(1e-4, dtype=tf.float64)
EPS = 1e-7

@tf.function
def masked_log(x): #applies log to matrix with elements <= 0 while preserving ordering
    return tf.math.log(tf.clip_by_value(x, EPS, 100.))

# @tf.function
# def compute_cross_entropy(
#     probabilities_graph, probabilities_distance, EPS=1e-7, repulsion_strength=1.0
# ):
#     # cross entropy
#     attraction_term = -probabilities_graph * tf.math.log(
#         tf.clip_by_value(probabilities_distance, EPS, 1.0)
#     )
#     repellant_term = (
#         -(1.0 - probabilities_graph)
#         * tf.math.log(tf.clip_by_value(1.0 - probabilities_distance, EPS, 1.0))
#         * repulsion_strength
#     )

#     # balance the expected losses between atrraction and repel
#     CE = attraction_term + repellant_term
#     return attraction_term, repellant_term, CE
    
def nll(y_true, y_pred):
    """ loss """

    return tf.reduce_mean((y_true - y_pred) ** 2)


def nll_builder(ww):
    def nll(y_true, y_pred):
        """ loss """
        return ww[4] * tf.reduce_mean((y_true - y_pred) ** 2)

    def nll_null(y_true, y_pred):
        return tf.cast(0., tf.float64)

    return nll if ww[4] > 0. else nll_null


def tsne_reg_builder(ww, perplexity, latent_dim):
    def tsne_reg(x, z):
        sum_x = tf.reduce_sum(tf.square(x), 1)
        dist = tf.constant(-2.0, tf.float64) * tf.matmul(x,
                                                         x,
                                                         transpose_b=True) + tf.reshape(sum_x, [-1, 1]) + sum_x

        p = tf.numpy_function(compute_transition_probability, [x, dist, perplexity, 1e-4, 50, False], tf.float64)
        # p=compute_transition_probability(x.numpy(),dist.numpy(),perplexity, 1e-4, 50,False) ## for eager dubugging
        nu = tf.constant(latent_dim-1.0, dtype=tf.float64)
        n = tf.shape(x)[0]

        sum_y = tf.reduce_sum(tf.square(z), 1)
        num = tf.constant(-2.0, tf.float64) * tf.matmul(z,
                                                        z,
                                                        transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
        num = num / nu

        p = p + tf.constant(0.1, tf.float64) / tf.cast(n, tf.float64)
        p = p / tf.expand_dims(tf.reduce_sum(p, 1), 1)

        num = tf.math.pow(tf.constant(1.0, tf.float64) + num,
                          -(nu + tf.constant(1.0, tf.float64)) / tf.constant(2.0, tf.float64))
        attraction = tf.multiply(p, tf.math.log(num))
        attraction = -tf.reduce_sum(attraction)

        den = tf.reduce_sum(num, 1) - 1
        repellant = tf.reduce_sum(tf.math.log(den))
        
        # tf.print("tsne loss :", tf.reduce_mean((repellant + attraction) / tf.cast(n, tf.float64)))
        
        return (repellant + attraction) / tf.cast(n, tf.float64)

    def null_reg(x, z):
        return tf.cast(0., tf.float64)

    return null_reg if ww[0] <= 0 else tsne_reg

def cense_reg_builder(ww, kneighs):
    def cense_reg(x):
        n = tf.shape(x)[0]
        sum_x = tf.reduce_sum(tf.square(x), 1)
        dist = tf.constant(-2.0, tf.float64) * tf.matmul(x,
                                                         x,
                                                         transpose_b=True) + tf.reshape(sum_x, [-1, 1]) + sum_x
        
        dist_and_knn = tf.numpy_function(cense_dtk, [dist], tf.float64)
        k = tf.shape(dist)[0] # k is batch size
        # tf.print("dk: ", dist_and_knn, summarize=-1)
        dist = dist_and_knn[:k,1:kneighs+1] # first neighbor is the point itself
        knn = dist_and_knn[k:,1:kneighs+1]
        
        # tf.print("dist: ",dist, summarize=-1)
        # tf.print("knn: ", knn, summarize=-1)
        
        p = tf.math.reciprocal(tf.constant(1, tf.float64) + dist) #+ tf.constant(0.1, tf.float64) / tf.cast(n, tf.float64)
        p = p / tf.expand_dims(tf.reduce_sum(p, 1), 1)
        
        # tf.print("prob: ",p, summarize=-1)
        
        # resymmetrize
        
        p = tf.numpy_function(remove_asym, [p, knn], tf.float64)
        # tf.print("newp: ", p)
            
        z_p = tf.reduce_sum(tf.reduce_sum(p, 1)) # + tf.constant(0.1, tf.float64)

        info_loss = tf.math.log(z_p) - tf.multiply(tf.math.reciprocal(z_p), 
                                                   tf.reduce_sum(tf.reduce_sum(p * masked_log(p), 1)))
        
        # tf.print("cense loss :", tf.reduce_mean(info_loss))
        return info_loss
  
    def null_reg(x):
        return tf.cast(0., tf.float64)
    
    return null_reg if ww[1] <= 0 else cense_reg 

def umap_reg_builder(ww, a, b):
    def umap_reg(x, z):
        sum_x = tf.reduce_sum(tf.square(x), 1)
        dist = tf.constant(-2.0, tf.float64) * tf.matmul(x,
                                                         x,
                                                         transpose_b=True) + tf.reshape(sum_x, [-1, 1]) + sum_x
        n = tf.shape(dist)[0]*tf.shape(dist)[1]
        # dist = K.maximum(dist, K.epsilon())
        # tf.print("dist :", dist, summarize=-1)
        # dist_and_knn = tf.numpy_function(umap_dtk, [dist], tf.float64)
        # k = tf.shape(dist)[0] # k is batch size
        # tf.print("dist_knn: ", dist_and_knn, summarize=-1)
        # dist = dist_and_knn[:k,:] 
        # knn = dist_and_knn[k:,:]
        
        # rhos_and_sigmas = tf.numpy_function(smooth_knn_dist, [dist], tf.float64)
        # tf.print("rhos_sigmas: ", rhos_and_sigmas, summarize=-1)
        # rhos = rhos_and_sigmas[0,:]
        # sigmas = rhos_and_sigmas[1,:]

        # uwg = tf.numpy_function(compute_membership_strengths, [knn, dist, sigmas, rhos], tf.float64)
        # tf.print("Graph: ", uwg, summarize=-1)

        highD_proba = tf.numpy_function(simplicial_graph_from_dist, [dist], tf.float64)
        # tf.print("highD_proba :", highD_proba, summarize=-1)
        
        sum_z = tf.reduce_sum(tf.square(z), 1)
        dist_z = tf.constant(-2.0, tf.float64) * tf.matmul(z,
                                                            z,
                                                            transpose_b=True) + tf.reshape(sum_z, [-1, 1]) + sum_z
        
        # lowD_proba = tf.numpy_function(euclidean_embedding, [dist_z, a, b], tf.float64)
        
        # lowD_proba = tf.math.reciprocal(tf.constant(1, tf.float64)
        #                                 + tf.multiply(tf.constant(1.0, tf.float64), 
        #                                               tf.math.exp(tf.multiply(tf.constant(2.0, tf.float64),
        #                                                                       tf.math.log(dist_z)))))
        
        lowD_proba = tf.math.reciprocal(tf.constant(1, tf.float64)
                                        + tf.multiply(a, tf.math.pow(K.maximum(dist_z, K.epsilon()), 
                                                                     tf.cast(b, tf.float64))))
        
        # tf.print("dist_z :", dist_z, summarize=-1)
        # tf.print("lowD_proba :", lowD_proba, summarize=-1)
        # tf.print("z :", z, summarize=-1)
        attraction = tf.reduce_sum(
            tf.subtract(tf.multiply(highD_proba, masked_log(highD_proba)),
                        tf.multiply(highD_proba, masked_log(lowD_proba))))
        
        negH = tf.subtract(tf.constant(1, tf.float64), highD_proba)
        negL = tf.subtract(tf.constant(1, tf.float64), lowD_proba)
        
        repellant = tf.reduce_sum(
            tf.subtract(tf.multiply(negH, masked_log(negH)),
                        tf.multiply(negH, masked_log(negL))))
        
        cross_entropy = (repellant + attraction) / tf.cast(n, tf.float64)
                        
        # tf.print("umap loss :", cross_entropy)
        
        return tf.reduce_mean(cross_entropy)
    
    def null_reg(x,z):
        return tf.cast(0., tf.float64)

    return null_reg if ww[2] <= 0 else umap_reg

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + eps_std * tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def __init__(self,
                 drate=0.1,
                 encoder_shape=[32, 32],
                 latent_dim=32,
                 activation="relu",
                 name='encoder',
                 dynamic=False,
                 **kwargs):

        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder_shape = encoder_shape
        self.drop0 = layers.Dropout(rate=drate)
        self.alphadrop = layers.AlphaDropout(rate=drate)
        self.dense_proj = [None] * len(encoder_shape)
        for i, v in enumerate(self.encoder_shape):
            self.dense_proj[i] = layers.Dense(v, activation=activation)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)

    def call(self, inputs, training=None):
        x = inputs
        for dl in self.dense_proj: x = self.drop0(dl(x))
        return self.dense_mean(x), self.dense_log_var(x)


class Decoder(layers.Layer):

    def __init__(self,
                 original_dim,
                 # encoder,
                 activation="relu",
                 drate=0.1,
                 decoder_shape=[32, 32],
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_shape = decoder_shape
        self.drop = layers.Dropout(rate=drate)
        self.alphadrop = layers.AlphaDropout(rate=drate)
        self.dense_proj = [None] * len(decoder_shape)
        for i, v in enumerate(self.decoder_shape):
            self.dense_proj[i] = layers.Dense(v, activation=activation)

        self.dense_output = layers.Dense(original_dim)  # ,kernel_regularizer=l1_l2(l1=0.001, l2=0.001))

    def call(self, inputs, training=None):
        x = inputs
        for dl in self.dense_proj: x = dl(x)
        return self.dense_output(x)


class Vaevictis(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim,
                 encoder_shape=[32, 32],
                 decoder_shape=[32, 32],
                 latent_dim=32,
                 perplexity=10.,
                 metric="euclidean",
                 margin=1.,
                 cense_kneighs= 16,
                 ww=[10., 1., 1., 1., 1., 1.],
                 name='Vaevictis',
                 **kwargs):
        super(Vaevictis, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder_shape = encoder_shape
        self.decoder_shape = decoder_shape
        self.latent_dim = latent_dim
        self.perplexity = perplexity
        self.metric = metric
        self.distance = euclidean_distance if metric == "euclidean" else cosine_distance
        self.margin = margin
        self.kneighs = cense_kneighs
        self.ww = ww
        self.encoder = Encoder(latent_dim=latent_dim,
                               encoder_shape=encoder_shape, drate=0.2)
        self.decoder = Decoder(original_dim, decoder_shape=decoder_shape, drate=0.1)
        self.sampling = Sampling()
        self.tsne_reg = tsne_reg_builder(self.ww, self.perplexity, self.latent_dim)
        self.cense_reg = cense_reg_builder(self.ww, self.kneighs)
        self.a, self.b = find_ab_params()
        self.a = tf.constant(self.a, tf.float64)
        self.b = tf.constant(self.b, tf.float64)
        self.umap_reg = umap_reg_builder(self.ww, self.a, self.b)
        self.nll = nll_builder(self.ww)

    def call(self, inputs, training=None):
        # tf.print("inp :", inputs)
        z_mean, z_log_var = self.encoder(inputs[0], training=training)
        pos, _ = self.encoder(inputs[1], training=training)
        neg, _ = self.encoder(inputs[2], training=training)

        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var + tf.math.log(eps_sq) - tf.square(z_mean) - eps_sq * tf.exp(z_log_var))
        # tf.print("kl loss:", kl_loss)
        self.add_loss(self.ww[5] * kl_loss)
        
        b = self.tsne_reg(inputs[0], z_mean)
        # tf.print("tsne loss:", b)
        self.add_loss(self.ww[0] * b)
        
        cen = self.cense_reg(inputs[0]) + EPS
        cenl = self.ww[1] * kl_loss / cen
        # tf.print("cense loss:", cenl)
        self.add_loss(cenl)
        
        umap = self.umap_reg(inputs[0], z_mean)
        self.add_loss(self.ww[2] * umap)
        
        pnl = pn_loss_g((z_mean, pos, neg), self.distance, self.margin)
        # tf.print("pn loss:", pnl)
        self.add_loss(self.ww[3] * pnl)
        
        
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z, training=training)
        # tf.print("rec :", reconstructed)
        # rls=self.nll(inputs[1],reconstructed)
        # self.add_loss(self.ww[4]*rls)
        return reconstructed

    def get_config(self):
        return {'original_dim': self.original_dim, 'encoder_shape': self.encoder_shape.copy(),
                'decoder_shape': self.decoder_shape.copy(), 'latent_dim': self.latent_dim,
                'perplexity': self.perplexity, 'metric': self.metric,
                'margin': self.margin, 'ww': self.ww.copy(), 'name': self.name}

    def save(self, config_file, weights_file):
        json_config = self.get_config()
        json.dump(json_config, open(config_file, 'w'))
        self.save_weights(weights_file)

    def refit(self, x_train, batch_size=512, epochs=100, patience=0, vsplit=0.1, k=30, knn=None):
        triplets = input_compute(x_train, k, self.metric, knn)
        es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience)
        self.fit(triplets, triplets[0], batch_size=batch_size, epochs=epochs, callbacks=[es],
                 validation_split=vsplit, shuffle=True)

    def predict_np(self):
        def predict(data):
            return self.encoder(data)[0].numpy()

        return (predict)
    

def dimred(x_train, dim=2, vsplit=0.1, enc_shape=[128, 128, 128], dec_shape=[128, 128, 128],
           perplexity=10., batch_size=512, epochs=100, patience=0, ivis_pretrain=0, ww=[10., 10., 1., 1., 1., 1.],
           metric="euclidean", margin=1., k=30, cense_kneighs=16, knn=None,batch_size_predict=524288):
    """Wrapper for model build and training

    Parameters
    ----------
    x_train : array, shape (n_samples, n_dims)
              Data to embedd, training dataset
    dim : integer, embedding_dim
    vsplit : float, proportion of data used at validation step - splitted befor shuffling!
    enc_shape : list of integers, shape of the encoder i.e. [128, 128, 128] means 3 dense layers with 128 neurons
    dec_shape : list of integers, shape of the decoder
    perplexity : float, perplexity parameter for tsne regularisation
    batch_size : integer, batch size
    epochs : integer, maximum number of epochs
    patience : integer, callback patience
    ivis_pretrain : integer, number of epochs to run without tsne regularisation as pretraining
    ww : list of floats, weights on losses in this order: tsne regularization, cense regularization, umap regularization, ivis pn loss, reconstruction error, KL divergence
    metric : str, "euclidean" or "angular"
    margin : float, ivis margin
    k : integer, number of nearest neighbors for triplet computation
    cense_kneighs : integer, number of neighbors to keep in a batch for cense computation
    knn : integer array, precomputed knn matrix
    batch_size_predict : batch_size for model prediction

    Returns
    ----------
    X_new : array, shape (n_samples, embedding_dims) embedded data
    predict : projection function
    vae : whole vaevictis model
    """

    triplets = input_compute(x_train, k, metric, knn)
    # triplets = [x_train, x_train, x_train]
    # tf.print("triplets :", triplets)

    optimizer = tf.keras.optimizers.Adam()
    if ivis_pretrain > 0:
        ww1 = ww.copy()
        ww1[0] = -1.
        ##ww1[1]=np.maximum(ww1[1],1.)
        vae = Vaevictis(x_train.shape[1], enc_shape, dec_shape, dim, perplexity, metric, margin, ww1)
        nll_f = nll_builder(ww1)
        vae.compile(optimizer, loss=nll_f)
        vae.fit(triplets, triplets[0], batch_size=batch_size, epochs=ivis_pretrain, validation_split=vsplit,
                shuffle=True)
        pre_weight = vae.get_weights()

    vae = Vaevictis(x_train.shape[1], enc_shape, dec_shape, dim, perplexity, metric, margin, cense_kneighs, ww)
    nll_f = nll_builder(ww)
    vae.compile(optimizer, loss=nll_f)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience)

    if ivis_pretrain > 0:
        aux = vae.predict((x_train[:10, ], x_train[:10, ], x_train[:10, ]))  # instantiate model
        vae.set_weights(pre_weight)

    vae.fit(triplets, triplets[0], batch_size=batch_size, epochs=epochs, callbacks=[es], validation_split=vsplit,
            shuffle=True)

    # eager debugging
    # @tf.function
    # def train_one_step(m1,optimizer,x):
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(m1.trainable_weights)
    #         reconstructed = m1(x)
    #         # Compute reconstruction loss
    #         rec_l = nll(x, reconstructed)
    
            
    #         tsne_l = m1.losses[1]
    #         cense_l = m1.losses[2]
    #         umap_l = m1.losses[3]
    #         ivis_l = m1.losses[4]
    #         kl_l = m1.losses[0]
                        
    #         loss = sum(m1.losses) 
    #         # tf.print("batch_data :", x, summarize=-1)
    #         # tf.print("pred :", reconstructed, summarize=-1)
    
    #     # tf.print("losses :", m1.losses)
    #     # tf.print("weights :", m1.trainable_weights, summarize=-1)
    #     # grads = tape.gradient(loss, m1.trainable_weights)
    #     # optimizer.apply_gradients(zip(grads, m1.trainable_weights))
    #     # gm = [tf.reduce_mean(tf.math.abs(g)) for g in grads]
    #     # tf.print("mean of grads :", sum(gm)/len(gm), summarize=-1)

    #     tsne_g = tape.gradient(tsne_l, m1.trainable_weights)
    #     cense_g = tape.gradient(cense_l, m1.trainable_weights)
    #     umap_g = tape.gradient(umap_l, m1.trainable_weights)
    #     ivis_g = tape.gradient(ivis_l, m1.trainable_weights)
    #     rec_g = tape.gradient(rec_l, m1.trainable_weights)
    #     kl_g = tape.gradient(kl_l, m1.trainable_weights)
    #     # tf.print("testg :", cense_g, summarize=-1)
            
    #     optimizer.apply_gradients(zip(tsne_g, m1.trainable_weights))
    #     optimizer.apply_gradients(zip(cense_g, m1.trainable_weights))
    #     optimizer.apply_gradients(zip(umap_g, m1.trainable_weights))
    #     optimizer.apply_gradients(zip(ivis_g, m1.trainable_weights))
    #     optimizer.apply_gradients(zip(rec_g, m1.trainable_weights))
    #     optimizer.apply_gradients(zip(kl_g, m1.trainable_weights))
        
    #     grads = [tsne_g, cense_g, umap_g, ivis_g, rec_g, kl_g]
    #     processed = []
    #     for grad in grads:
    #         means = []
    #         for g in grad:
    #             if g is not None:
    #                 means.append(tf.reduce_mean(tf.math.abs(g)))
    #         processed.append(sum(means)/len(means))

    #     return loss, processed
    
    
    
    # # # Iterate over epochs.
    # def train():
    #     loss=0.
    #     formatted_data = []
    #     for dataset in triplets:
    #         slices = tf.data.Dataset.from_tensor_slices(dataset)
    #         slices = slices.batch(batch_size)
    #         formatted_data.append(list(slices.as_numpy_iterator()))
    #     anc, pos, neg = formatted_data
    #     for epoch in range(epochs):
    #         print('Start of epoch %d' % (epoch,))
    #         grads = [0]*6
    #         for  b, x_batch_train in enumerate(anc):
    #             # print('\t Batch', b)
    #             obj=train_one_step(vae,optimizer,[x_batch_train, pos[b], neg[b]])
    #             loss = obj[0]
    #             gs = obj[1]
    #             grads = [grads[i]+gs[i] for i in range(len(grads))]
              
    #         grads = [g/len(anc) for g in grads]    
    #         tf.print("mean of tsne grads :", grads[1], summarize=-1)
    #         tf.print("mean of cense grads :", grads[2], summarize=-1)
    #         tf.print("mean of umap grads :", grads[3], summarize=-1)
    #         tf.print("mean of ivis grads :", grads[4], summarize=-1)
    #         tf.print("mean of reconstruction grads :", grads[5], summarize=-1)
    #         tf.print("mean of KL grads :", grads[0], summarize=-1)
    #     return loss
    
    # loss=train()

    inputs=layers.Input(shape=(x_train.shape[1],))
    outputs=vae.encoder(inputs)[0]
    encoder_model=tf.keras.models.Model(inputs,outputs)
                 
    def predict(data,batch_size_predict=batch_size_predict):
        return encoder_model.predict(data,batch_size=batch_size_predict)

    z_test = predict(x_train)
    return z_test, predict, vae


def loadModel(config_file, weights_file):
    config = json.load(open(config_file))
    new_model = Vaevictis(config["original_dim"], config["encoder_shape"],
                          config["decoder_shape"], config["latent_dim"], config["perplexity"],
                          config["metric"], config["margin"], config["ww"])

    optimizer = tf.keras.optimizers.Adam()
    nll_f = nll_builder(config["ww"])
    new_model.compile(optimizer, loss=nll_f)
    x = [np.random.rand(10, config["original_dim"]),
         np.random.rand(10, config["original_dim"]),
         np.random.rand(10, config["original_dim"])]
    new_model.train_on_batch(x, x[0])
    new_model.load_weights(weights_file)


    inputs=layers.Input(shape=(config["original_dim"],))
    outputs=new_model.encoder(inputs)[0]
    encoder_model=tf.keras.models.Model(inputs,outputs)
    
    def predict(data,batch_size_predict=524288):
        return encoder_model.predict(data,batch_size=batch_size_predict)

    return new_model, predict

def cluster_sampling(data, n_clusters, n_samples):
  kmeans = KMeans(n_clusters = n_clusters).fit(data)
  memberships = kmeans.predict(data)
  sample = np.zeros(n_clusters*n_samples)
  for clust in range(n_clusters):
    ids = np.where(memberships==clust)
    s = np.random.choice(ids[0], size = n_samples, replace = True)
    sample[clust*n_samples:(clust+1)*n_samples] = s
    
  return data[sample.astype(int)], memberships, kmeans

# data = np.loadtxt("../../MMDResNet/data/source_pIC_orig.csv", delimiter=',')[:50,:]
# cl = cluster_sampling(data, n_clusters=1, n_samples=12)
# sampled_data = cl[0]
# lab = cl[2].predict(sampled_data)
# dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# # ipdb.set_trace()
# build_annoy_index(sampled_data, "index/ind_" + dt, metric="euclidean", build_index_on_disk=True)
# # ipdb.set_trace()
# knn_matrix = extract_knn(data, "index/ind_" + dt, metric="euclidean")
# # ipdb.set_trace()
# vae = dimred(sampled_data,
#               dim=2,
#               vsplit=0.1,
#               enc_shape=[16,16], 
#               dec_shape=[16,16],
#               perplexity=10., 
#               batch_size=6, 
#               epochs=10, 
#               patience=8, 
#               ww=[1., 1., 1., 1., 1., 2.],
#               metric="euclidean", 
#               margin=1.,
#               k=6,
#               cense_kneighs = 4,
#               knn=None)

# data = np.loadtxt("../../MMDResNet/data/source_pIC_orig.csv", delimiter=',')
# cl = cluster_sampling(data, n_clusters=15, n_samples=500)
# sampled_data = cl[0]
# lab = cl[2].predict(sampled_data)

# nbrs = NearestNeighbors(n_neighbors = 16).fit(sampled_data)
# dst, knn_mat = nbrs.kneighbors(sampled_data)

# vae = dimred(sampled_data,
#               dim=2,
#               vsplit=0.1,
#               enc_shape=[128, 128, 128], 
#               dec_shape=[128, 128, 128],
#               perplexity=10., 
#               batch_size=128, 
#               epochs=15, 
#               patience=10, 
#               ww=[1., 1., 1., 1., 1., 1.],
#               metric="euclidean", 
#               margin=1.,
#               k=16,
#               cense_kneighs = 8,
#               knn=knn_mat)

# layout=vae[0]
# pred=vae[1]

# fig1 = plt.figure(figsize=(8,8), dpi=320)
# plt.scatter(layout[:,0], layout[:,1], c=lab, cmap='viridis', s=0.2)

# new = pred(data)
# newl = cl[2].predict(data)
# fig2 = plt.figure(figsize=(8,8), dpi=320)
# plt.scatter(new[:,0], new[:,1], c=newl, cmap='viridis', s=0.2)
from setuptools import setup

setup(name='vaevictis',
      version='0.3.1',
      description='test',
      install_requires=["annoy","numba",
      "tensorflow","tqdm","scipy","numpy","scikit-learn"],
      packages=['vaevictis'],
      zip_safe=False)

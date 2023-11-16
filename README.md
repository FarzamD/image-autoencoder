# image-autoencoder
repository includes 3 auto-encoder projects:
+ mnist auto-encoder: simple 1D auto-encoder for mnist dataset
+ img_auto_encoder 1D: 1D auto-encoder for a set of common images
+ img_auto_encoder 2D: 2D auto-encoder for a set of common images
## Table of Contents
+ [mnist auto-encoder](#mnist-auto-encoder)
+ [img_auto_encoder 1D](#img_auto_encoder-1d)
+ [img_auto_encoder 2D](#img_auto_encoder-2d)
  ***
# mnist auto-encoder
#### pre-requisites
+ download [mnist dataset](https://github.com/FarzamD/image-autoencoder/blob/main/mnist%20data.zip)
+ install tensorflow
### code blocks
#### load mnist
+ load mnist dataset
    + dataset consists of images and labels of handwritten digits  
        + images are 28⨉28 pixels
        + trainset has 60,000 samples and testset 10,000 samples
    + An alternative way to download mnist dataset is using code bellow
      ```python
      from keras.datasets import mnist
      (train_X, train_y), (test_X, test_y) = mnist.load_data()
+ images are flattened because the model is 1D
#### simple model
***
# img_auto_encoder 1D
***
# img_auto_encoder 2D

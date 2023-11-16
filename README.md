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
+ install tensorflow
+ download [mnist dataset](https://github.com/FarzamD/image-autoencoder/blob/main/mnist%20data.zip)
    + An alternative way to download and load mnist dataset is using code bellow
      ```python
      from keras.datasets import mnist
      (train_X, train_y), (test_X, test_y) = mnist.load_data()
### code blocks
#### load mnist
+ load mnist dataset
    + dataset consists of images and labels of handwritten digits  
        + images are 28⨉28 pixels
        + trainset has 60,000 samples and testset 10,000 samples
+ images are flattened because the model is 1D
#### simple model
+ A single-layer encoder with n=300 neurons is designed
+ 28⨉28 images are compressed(encoded) to k*8 bytes(k=4 and encoder output is float64(8 bytes) ) 
+ A single-layer decoder with m=300 neurons is designed
+ 28⨉28 images are reconstructed(decoded) from k*8 bytes
    + because the images are of type float32(4 bytes) they have a total size of 3136 bytes
+ model parameters n, m, and k can be modified to observe changes
+ adadelta optimizer with learning_rate=1, and rho=.8 is used to train model
##### plot
+ a number of images are selected to compare with reconstructed ones
***
# img_auto_encoder 1D
***
# img_auto_encoder 2D

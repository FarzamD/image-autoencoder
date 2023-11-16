# image-autoencoder
+ This repository includes 3 auto-encoder projects:
    + mnist auto-encoder: simple 1D auto-encoder for mnist dataset
    + img_auto_encoder 1D: 1D auto-encoder for a set of common images
    + img_auto_encoder 2D: 2D auto-encoder for a set of common images
+ Further details for each project are mentioned bellow
## Table of Contents
+ [mnist auto-encoder](#mnist-auto-encoder)
+ [img_auto_encoder 1D](#img_auto_encoder-1d)
+ [img_auto_encoder 2D](#img_auto_encoder-2d)
  ***
# mnist auto-encoder
#### pre-requisites
+ Install tensorflow
+ Download and extract [mnist dataset](https://github.com/FarzamD/image-autoencoder/blob/main/mnist%20data.zip)
    + An alternative way to download and load mnist dataset is using code bellow
      ```python
      from keras.datasets import mnist
      (train_X, train_y), (test_X, test_y) = mnist.load_data()
## code blocks
### load mnist
+ Load mnist dataset
    + Dataset consists of images and labels of handwritten digits  
        + Images are 28⨉28 pixels
        + Trainset has 60,000 samples and testset 10,000 samples
+ Images are turned to float32
+ Images are flattened because the model is 1D
### simple model
+ A single-layer encoder with n=300 neurons is designed
+ 28⨉28 images are compressed(encoded) to k*8 bytes(k=4 and encoder output is float64(8 bytes) ) 
+ A single-layer decoder with m=300 neurons is designed
+ 28⨉28 images are reconstructed(decoded) from k*8 bytes
    + because the images are of type float32(4 bytes) they have a total size of 3136 bytes
+ Model parameters n, m, and k can be modified to observe changes
+ adadelta optimizer with learning_rate=1, and rho=.8 is used to train model
##### plot
+ A number of images are selected to be compared with reconstructed ones
### divide image model
+ images are divided into smaller images of size 28/d⨉28/d (d=2)
+ The model then auto-encodes smaller images
+ The smaller images are then put back together to make full images of size 28⨉28
+ The model structure is the same as before
+ Model parameters are set as below
    + n=m=300
    + k=10
##### plot
+ A number of images are selected to be compared with reconstructed ones
+ Below is the resulting plot 
  
![mnist auto-encoder plot](https://github.com/FarzamD/image-autoencoder/blob/main/readme-files/mnist-ae.PNG "auto-encoder plot")

+ The huge difference between original and reconstructed images is due to sharp edges and high-frequency components of the original images

***

# img_auto_encoder 1D

#### pre-requisites
+ Install tensorflow
+ Download [dataset](https://github.com/FarzamD/image-autoencoder/blob/main/mnist%20data.zip)

## code blocks
### load dataset
+ Load dataset
    + Dataset consists of commonly used images  
        + Images are 256⨉256 pixels
        + Trainset has 91 samples and testset 5 samples
+ Images are turned to float32
+ Images are flattened because the model is 1D
### divide image model
+ images are divided into smaller images of size 16⨉16 (image size divided by d=16)
+ A single-layer encoder with n=64 neurons is designed
+ 16⨉16 images are compressed(encoded) to k*8 bytes(k=16 and encoder output is float64(8 bytes) ) 
+ A single-layer decoder with m=64 neurons is designed
+ 16⨉16 images are reconstructed(decoded) from k*8 bytes
    + because the small images are of type float32(4 bytes) they have a total size of 1024 bytes
+ The smaller images are then put back together to make full images of size 256⨉256
+ Model parameters n, m, and k can be modified to observe changes
+ adadelta optimizer with learning_rate=1, and rho=.8 is used to train model
##### plot
+ A number of images are selected to be compared with reconstructed ones
+ Below is the resulting plot 
  
![1d auto-encoder plot](https://github.com/FarzamD/image-autoencoder/blob/main/readme-files/1d-ae.PNG "1d auto-encoder plot")

+ The reconstructed images are better compared to mnist auto-encoder. this is due to
    +  less sharp edges and high-frequency components compared to mnist auto-encoder
    +  more compression bytes  compared to mnist auto-encoder


***
# img_auto_encoder 2D

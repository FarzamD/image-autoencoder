# %% load images
import os
import numpy as np
from matplotlib import pyplot as plt

trains= os.listdir('TrainSet')
tests= os.listdir('TestSet')

train_set= np.zeros((len(trains),256,256))#initialize train_set
test_set= np.zeros((len(tests),256,256))#initialize test_set

for i,file in enumerate(tests):
    img=plt.imread('TestSet/'+tests[i])
    test_set[i,:,:]=img
for i,file in enumerate(trains):
    img=plt.imread('TrainSet/'+trains[i])
    train_set[i,:,:]=img
del file,i,tests, trains


# %% divide image model
shape=256**2
d=16
shape_d= shape//(d**2)

train_set = train_set.astype("float32") / 255.0           
test_set = test_set.astype("float32") / 255.0

train_set= train_set.reshape((-1,shape_d))
test_set = test_set.reshape((-1,shape_d))
# %%% models
# %%%% encoder
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,LeakyReLU
from tensorflow.keras.models import Model

n=m=64
k=16
i = Input(shape=shape_d, name="encoder_input")

x = Dense(units=n, name="encoder_dense_1")(i)
x = LeakyReLU(name="encoder_leakyrelu_1")(x)

x = Dense(units=k, name="encoder_dense_2")(x)
x = LeakyReLU(name="encoder_output")(x)

encoder = Model(i, x, name="encoder_model")
# %%%% decoder
i = Input(shape=k, name="decoder_input")

x = Dense(units=m, name="decoder_dense_1")(i)
x = LeakyReLU(name="decoder_leakyrelu_1")(x)

x = Dense(units=shape_d, name="decoder_dense_2")(x)
x = LeakyReLU(name="decoder_output")(x)

decoder = Model(i, x, name="decoder_model")
# %%%% auto encoder
ae_input = Input(shape=shape_d, name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = Model(ae_input, ae_decoder_output, name="AE")

# %%% fit
op= tf.keras.optimizers.Adadelta(learning_rate=1, rho= .8)
ae.compile(loss="mse", optimizer=op)

ae.fit(train_set, train_set, epochs=50,  validation_data=(test_set, test_set))
# %%%% pred
N=3

img_p= ae.predict(test_set)
from matplotlib import pyplot as plt
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / mse**.5) 
    return psnr 
img0= test_set.reshape((-1,256,256))
img_p0= img_p.reshape((-1,256,256))

i=np.random.choice(range(5),size=N)
img0= img0[i]
img_p0= img_p0[i]

plt.subplot(N, 2, 1)
plt.title('original images')
plt.subplot(N, 2, 2)
plt.title('reconstructed images')


for i in range(N):
    plot_ind = i*2 + 1
    
    plt.subplot(N, 2, plot_ind)
    plt.imshow(img0[i], cmap="gray")

    plt.subplot(N, 2, plot_ind+1)
    plt.imshow(img_p0[i], cmap="gray")

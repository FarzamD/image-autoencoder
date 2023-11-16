# %% load mnist
import pickle
mnist= pickle.load(open('mnist.pickle','rb'))

[x_train_orig, y_train, x_test_orig, y_test]= mnist
x_train_orig = x_train_orig.astype("float32") / 255.0           
x_test_orig = x_test_orig.astype("float32") / 255.0

shape=28*28#784

x_train= x_train_orig.reshape((-1,shape))
x_test= x_test_orig.reshape((-1,shape))
del x_train_orig, y_train, x_test_orig, y_test,mnist
# %% simple model
# %%% models
# %%%% encoder
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,LeakyReLU
from tensorflow.keras.models import Model

n=m=300
k=4
i = Input(shape=shape, name="encoder_input")

x = Dense(units=n, name="encoder_dense_1")(i)
x = LeakyReLU(name="encoder_leakyrelu_1")(x)

x = Dense(units=k, name="encoder_dense_2")(x)
x = LeakyReLU(name="encoder_output")(x)

encoder = Model(i, x, name="encoder_model")
# %%%% decoder
i = Input(shape=k, name="decoder_input")

x = Dense(units=m, name="decoder_dense_1")(i)
x = LeakyReLU(name="decoder_leakyrelu_1")(x)

x = Dense(units=shape, name="decoder_dense_2")(x)
x = LeakyReLU(name="decoder_output")(x)

decoder = Model(i, x, name="decoder_model")
# %%%% auto encoder
ae_input = Input(shape=shape, name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = Model(ae_input, ae_decoder_output, name="AE")

# %%% fit
op= tf.keras.optimizers.Adadelta(learning_rate=1, rho= .8)
ae.compile(loss="mse", optimizer=op)

ae.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))
# %%% pred
# encoded_images = encoder.predict(x_train)
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)

# %%%% plot
import numpy as np
from matplotlib import pyplot as plt
import pickle
mnist= pickle.load(open('mnist.pickle','rb'))

[x_train_orig, y_train, x_test_orig, y_test]= mnist

encoded_images = encoder.predict(x_train)
decoded_images = decoder.predict(encoded_images)
decoded_images_orig = np.reshape(decoded_images, newshape=(decoded_images.shape[0], 28, 28))

N = 5 #number of images to compare
for i in range(N):
    plot_ind = i*2 + 1
    rand_ind = np.random.randint(low=0, high=x_train.shape[0])
    plt.subplot(N, 2, plot_ind)
    plt.imshow(x_train_orig[rand_ind, :, :], cmap="gray")
    plt.subplot(N, 2, plot_ind+1)
    plt.imshow(decoded_images_orig[rand_ind, :, :], cmap="gray")

# %% divide image model
d=2
shape_d= shape//(d**2)
x_train= x_train.reshape((-1,shape_d))
x_test= x_test.reshape((-1,shape_d))

# %%% models
# %%%% encoder
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,LeakyReLU
from tensorflow.keras.models import Model
n=m=300
k=10
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
ae.compile(loss="mse", optimizer=op,steps_per_execution=4)

ae.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))

# %%%% plot
import numpy as np
from matplotlib import pyplot as plt
import pickle
mnist= pickle.load(open('mnist.pickle','rb'))

[x_train_orig, y_train, x_test_orig, y_test]= mnist

decoded_images = ae.predict(x_train)
decoded_images_orig = decoded_images.reshape( (-1, 28, 28))

N = 5 #number of images to compare
plt.subplot(N, 2, 1)
plt.title('original images')
plt.subplot(N, 2, 2)
plt.title('reconstructed images')
for i in range(N):
    plot_ind = i*2 + 1
    rand_ind = np.random.randint(low=0, high=shape_d)
    plt.subplot(N, 2, plot_ind)
    plt.imshow(x_train_orig[rand_ind], cmap="gray")
    plt.subplot(N, 2, plot_ind+1)
    plt.imshow(decoded_images_orig[rand_ind], cmap="gray")

decoded_images = ae.predict(x_test)
decoded_images_orig = decoded_images.reshape( (-1, 28, 28))

plt.figure()
plt.subplot(N, 2, 1)
plt.title('original images')
plt.subplot(N, 2, 2)
plt.title('reconstructed images')
for i in range(N):
    plot_ind = i*2 + 1
    rand_ind = np.random.randint(low=0, high=shape_d)
    plt.subplot(N, 2, plot_ind)
    plt.imshow(x_test_orig[rand_ind], cmap="gray")
    plt.subplot(N, 2, plot_ind+1)
    plt.imshow(decoded_images_orig[rand_ind], cmap="gray")

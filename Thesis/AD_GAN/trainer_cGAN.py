import os
import nibabel as nib
import numpy as np
import tensorflow
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from numpy import ones, expand_dims
from numpy import zeros
from numpy.random import randint
from numpy.random import randn
from scipy import ndimage

# Filepaths
# Image filepaths

nc_filepath = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
              'AD_GAN/image_files/AD'
ad_filepath = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
              'AD_GAN/image_files/NC'
# Model Filepaths
disc_model_fp = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
                'AD_GAN/models/cgan_discriminator.h5'
gen_model_fp = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
               'AD_GAN/models/cgan_generator.h5'


# Convert 1 channel image to 3 channel
def grey_to_rgb(image):
    rgb_image = np.stack((image,) * 3, axis=-1)
    return rgb_image


# Read nifti image files
def read_nifti_file(filepth):
    scan = nib.load(filepth)
    scan = scan.get_fdata()
    return scan


# Normalize image
def normalize(volume):
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    volume = volume.astype("float32")
    return volume


# Resize function
def resizeVolume(img):
    # desired
    d_width = 224
    d_height = 224

    # current
    c_width = img.shape[0]
    c_height = img.shape[1]

    # factor to change by
    w_factor = d_width / c_width
    h_factor = d_height / c_height

    # Adjust proper rotation
    img = ndimage.rotate(img, 90, reshape=False)

    # apply the factors
    img = ndimage.zoom(img, (w_factor, h_factor), order=1)

    return img


def process_scan(filepath):
    volume = read_nifti_file(filepath)
    volume = normalize(volume)
    volume = resizeVolume(volume)
    # volume = grey_to_rgb(volume)
    return volume


# define the standalone discriminator model
def define_discriminator(in_shape=(224, 224, 1), n_classes=2):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 23x23 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 56x56
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 112x112
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 224x224
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load fashion mnist images
def load_real_samples(x, y):
    # load dataset
    (trainX, trainy) = x, y
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = trainX.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    return [X, trainy]


# # select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save(gen_model_fp)
    d_model.save(disc_model_fp)


# Setting up the filepaths for each file in NC
normal_scan_paths = [
    os.path.join(os.getcwd(), nc_filepath, x)
    for x in os.listdir(nc_filepath)
]

# Setting up the filepaths for each file in AD
alzheimer_scan_paths = [
    os.path.join(os.getcwd(), ad_filepath, x)
    for x in os.listdir(ad_filepath)
]
len(normal_scan_paths)

# Normal Subject files into numpy arrays
normal_volumes = np.array([process_scan(path) for path in normal_scan_paths], dtype=object)
normal_volume_labels = np.array([0 for _ in range(len(normal_volumes))])

print(normal_volumes.shape)
print(normal_volume_labels)

# Alzheimer's Subject files into numpy arrays
alzheimer_volumes = np.array([process_scan(path) for path in alzheimer_scan_paths], dtype=object)
alzheimer_volume_labels = np.array([1 for _ in range(len(alzheimer_volumes))])

print(alzheimer_volumes.shape)
print(alzheimer_volume_labels)

print("MRI scans with normal neuro presentation: " + str(len(normal_scan_paths)))
print("MRI scans with abnormal alzheimer's presentation: " + str(len(alzheimer_scan_paths)))

non_bias_split = 440

# 60/20 Split of volumes
X_train = np.concatenate((alzheimer_volumes[:non_bias_split], normal_volumes[:non_bias_split]), axis=0)
y_train = np.concatenate((alzheimer_volume_labels[:non_bias_split], normal_volume_labels[:non_bias_split]), axis=0)

X_val = np.concatenate((alzheimer_volumes[non_bias_split:], normal_volumes[non_bias_split:]), axis=0)
y_val = np.concatenate((alzheimer_volume_labels[non_bias_split:], normal_volume_labels[non_bias_split:]), axis=0)

# See some of the results and checking sizes

print(
    "Number of samples in train and validation are %d and %d."
    #     % (X_train.shape[0], X_val.shape[0])
)

print(len(y_train))

print(len(y_val))

print(X_train.shape)

print(type(X_train))

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples(X_train, y_train)
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

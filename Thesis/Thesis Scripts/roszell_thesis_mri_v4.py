# -*- coding: utf-8 -*-
# Roszell_Thesis_MRI_V4


# In this copy I will be using part of the ADNI dataset in order to see which is the best base for the model training.

# I will then use the original dataset as my test

# Setting up the Google Drive


# Loading the google drive where I stored the MOSMEDDATA files
import csv
import os
import random
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import visualkeras
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

train_name = 'some/filedir/in/arc/data/csv/train_path.csv'
test_name = 'some/filedir/in/arc/data/csv/test_path.csv'
counter = 0

# Only uncomment if you are adding to the dataset or changing it

paths = ['AD_Update', 'CN_Update']
imgdir = '/some/filedir/in/arc/data/thesis'
file_path = 'some/filedir/in/arc/data/csv/train_path.csv'
outputimgdir = '/some/filedir/in/arc/data/thesis/images/training_dataset_images'

with open(train_name, 'w', newline='') as new_train_file:
    train_w = csv.writer(new_train_file)
    with open(test_name, 'w', newline='') as new_test_file:
        test_w = csv.writer(new_test_file)
        for path in paths:
            img_path = os.path.join(imgdir, path)
            for path, dirs, files, in os.walk(img_path):
                for file in files:
                    if file.endswith('.nii') and not file.startswith('._'):
                        img_path = os.path.join(path, file)
                        if counter % 5 == 0 or 'T2' in path:
                            test_w.writerow([img_path])
                        else:
                            train_w.writerow([img_path])
                        counter += 1

img_paths = pd.read_csv(file_path, header=None)


def read_save_nifti_file(filepath, name):

    scan = nib.load(filepath)
    image = scan.get_fdata()
    image = np.squeeze(image)

    height, width, depth = image.shape

    image_1 = image[round(height / 2) - 10:round(height / 2) + 10, :, :]
    image_2 = image[:, round(width / 2) - 10:round(width / 2) + 10, :]
    image_3 = image[:, :, round(depth / 2) - 10:round(depth / 2) + 10]

    # Save 20 center slices of 3 different views for each subject
    for i in range(20):
        im_1 = image_1[i, :, :]
        im_2 = image_2[:, i, :]
        im_3 = image_3[:, :, i]

        filename_1 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '1' + '.nii'
        filename_2 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '2' + '.nii'
        filename_3 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '3' + '.nii'

        im_1 = nib.Nifti1Image(im_1, scan.affine, scan.header)
        im_2 = nib.Nifti1Image(im_2, scan.affine, scan.header)
        im_3 = nib.Nifti1Image(im_3, scan.affine, scan.header)

        nib.save(im_1, os.path.join(new_image_path, filename_1))
        nib.save(im_2, os.path.join(new_image_path, filename_2))
        nib.save(im_3, os.path.join(new_image_path, filename_3))


# Only uncomment if adding or changing the original dataset

for ind in tqdm(range(len(img_paths))):
    path = img_paths.iloc[ind, 0]
    if '/AD_Update' in path:
        if 'siemens_3' in path.lower():
            read_save_nifti_file(path, 'AD_siemens_3')
        if 'siemens_15' in path.lower():
            read_save_nifti_file(path, 'AD_siemens_15')
        if 'philips_3' in path.lower():
            read_save_nifti_file(path, 'AD_philips_3')
        if 'philips_15' in path.lower():
            read_save_nifti_file(path, 'AD_philips_15')
        if 'ge_3' in path.lower():
            read_save_nifti_file(path, 'AD_GE_3')
        if 'ge_15' in path.lower():
            read_save_nifti_file(path, 'AD_GE_15')

    if '/CN_Update' in path:
        if 'siemens_3' in path.lower():
            read_save_nifti_file(path, 'NC_siemens_3')
        if 'siemens_15' in path.lower():
            read_save_nifti_file(path, 'NC_siemens_15')
        if 'philips_3' in path.lower():
            read_save_nifti_file(path, 'NC_philips_3')
        if 'philips_15' in path.lower():
            read_save_nifti_file(path, 'NC_philips_15')
        if 'ge_3' in path.lower():
            read_save_nifti_file(path, 'NC_GE_3')
        if 'ge_15' in path.lower():
            read_save_nifti_file(path, 'NC_GE_15')

# Now to get them into their own labelled folders

files = os.listdir(new_image_path)
nc_filepath = 'some/file/dir/in/arc/data/NC'
ad_filepath = 'some/file/dir/in/arc/data/AD'

# Only uncomment if you are adding or editing the dataset

for f in files:
    # move if NC
    if 'NC_' in f:
        shutil.move(os.path.join(new_image_path, f), os.path.join(nc_filepath, f))
    # move if AD
    elif 'AD_' in f:
        shutil.move(os.path.join(new_image_path, f), os.path.join(ad_filepath, f))
    # notify if something else
    else:
        print('Could not categorize file with name %s' % f)


# Convert 1 channel image to 3 channel
def grey_to_rgb(image):
    rgb_image = np.repeat(image[np.newaxis, ...], 3, 2)
    return rgb_image


# Read nifti image files
def read_nifti_file(filepath):
    scan = nib.load(filepath)
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
    volume = grey_to_rgb(volume)
    return volume


# Setting up the filepaths for each file in NC
normal_scan_paths = [
    os.path.join(os.getcwd(), nc_filepath, x)
    for x in os.listdir(nc_filepath)
]

# Setting up the filepaths for each file in class 2
alzheimer_scan_paths = [
    os.path.join(os.getcwd(), ad_filepath, x)
    for x in os.listdir(ad_filepath)
]

# Normal Subject files into numpy arrays
normal_volumes = np.array([process_scan(path) for path in normal_scan_paths], dtype=object)
normal_volume_labels = np.array([0 for _ in range(len(normal_volumes))])

print(normal_volumes.shape)
print(normal_volume_labels)

# Alzheimer's Subject files into numpy arrays
alzheimer_volumes = np.array([process_scan(path) for path in alzheimer_scan_paths], dtype=object)
alzheimer_volume_labels = np.array([1 for _ in range(len(alzheimer_volumes))])

print("MRI scans with normal neuro presentation: " + str(len(normal_scan_paths)))
print("MRI scans with abnormal alzheimner's presentation: " + str(len(alzheimer_scan_paths)))

# 60/20 Split of volumes
X_train = np.concatenate((alzheimer_volumes[:1900], normal_volumes[:9000]), axis=0)
y_train = np.concatenate((alzheimer_volume_labels[:1900], normal_volume_labels[:9000]), axis=0)

X_val = np.concatenate((alzheimer_volumes[1900:], normal_volumes[9000:]), axis=0)
y_val = np.concatenate((alzheimer_volume_labels[1900:], normal_volume_labels[9000:]), axis=0)

print(
    "Number of samples in train and validation are %d and %d."
    #     % (X_train.shape[0], X_val.shape[0])
)

print(len(y_train))

print(len(y_val))

print(X_train.shape)

plt.imshow(X_train[1, 10], cmap='bone')

plt.imshow(X_train[1, 21], cmap='bone')

print(X_train.shape)

"""### Now the model"""

data_augmentation = tf.keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), ])


# The model build
def buildModel():
    initial_model = tf.keras.applications.VGG19(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)

    # Freeze the pretrained model parameters
    initial_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    x = initial_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


model = buildModel()
model.summary()

# Setting up the fit parameters
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics=tf.keras.metrics.BinaryAccuracy(),
              )

# Defining checkpoints
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "3D_MRI_classification.h5", save_best_only=True
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

# How man runs
epochs = 50

# Training the model
history = model.fit(x=X_train,
                    y=y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    shuffle=True,
                    verbose='auto',
                    callbacks=[checkpoint_cb, early_stopping_cb],
                    )

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("/content/drive/MyDrive/Colab Notebooks/MRI_model_v4")

"""## Loading the model and predicting here"""

reloaded_model = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/MRI_model_v4")

"""## Here we will take a look at a visualization of the model."""
# Using visualkeras

visualkeras.layered_view(reloaded_model).show()
visualkeras.layered_view(reloaded_model)

# Using the keras visualizer
tf.keras.utils.plot_model(reloaded_model, show_shapes=True)

print(history.history.keys())

# Showing the accuracy as a graph for easier viewing

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
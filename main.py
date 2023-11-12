# -*- coding: utf-8 -*-
import numpy as np
from glob import glob
import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import Dataset

from sklearn.model_selection import train_test_split

from model import *
from augmentation import *

# images directory
IMAGE_DIR = r'C:\Users\guilh\Desktop\image segmentation\INPUTS'
MASK_DIR = r'C:\Users\guilh\Desktop\image segmentation\MASKS'

image_paths = glob(os.path.join(IMAGE_DIR, '*.jpg'))
mask_paths = glob(os.path.join(MASK_DIR, '*.png'))
image_paths.sort()
mask_paths.sort()


########################
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
HEIGHT = 512
WIDTH = 512
num_classes = 33
########################

# split 
X_train, X_test, y_train, y_test = train_test_split(image_paths, mask_paths, 
                                                    test_size=0.2, random_state=0)

# preparing the dataset
train_ds = Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(read_image)
train_ds1 = train_ds.map(augment_image_batch1)
train_ds2 = train_ds.map(augment_image_batch2)
train_ds = train_ds.concatenate(train_ds1.concatenate(train_ds2))
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(read_image)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

#####################################
# unet model
model = UNet(HEIGHT, WIDTH, 33, LR)
#####################################


history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
model.save('unet_trained.h5')


# plotting the predictions of test data
for images, masks in test_ds.take(2):
    pred = model.predict(images)
    
predictions = np.argmax(pred, axis=-1)
labels = np.argmax(masks, axis=-1)

fig, ax = plt.subplots(BATCH_SIZE, 3, figsize=(15, 4*BATCH_SIZE))
for j in range(BATCH_SIZE):
    ax[j, 0].imshow(images[j, ...].numpy())
    ax[j, 1].imshow(predictions[j, ...].astype('uint8'))
    ax[j, 2].imshow(labels[j, ...])
    ax[j, 0].set_title('Original image')
    ax[j, 1].set_title('Prediction')
    ax[j, 2].set_title('Ground truth')
plt.show()
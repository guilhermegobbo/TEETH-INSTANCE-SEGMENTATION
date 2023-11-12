# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def read_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = image / 255.0
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [512, 512])
    mask = tf.cast(tf.squeeze(mask), dtype=tf.int32)
    mask = tf.one_hot(mask, 33, dtype=tf.int32)
    return image, mask

def augment_image_batch1(image, mask):
    new_seed = np.random.randint(5)
    print(image.shape)
    print(mask.shape)
    image = tf.image.resize(image, [int(1.1 * 512), int(1.1 * 512)])
    mask = tf.image.resize(mask, [int(1.1 * 512), int(1.1 * 512)])
    image = tf.image.random_crop(image, (512, 512, 3), seed=new_seed)
    mask = tf.image.random_crop(mask, (512, 512, 33), seed=new_seed)
    image = tf.image.random_flip_left_right(image, seed=new_seed)
    mask = tf.image.random_flip_left_right(mask, seed=new_seed)
    mask = tf.cast(mask, dtype=tf.int32)
    return image, mask

def augment_image_batch2(image, mask):
    new_seed = np.random.randint(5)
    image = tf.image.resize(image, [int(1.1 * 512), int(1.1 * 512)])
    mask = tf.image.resize(mask, [int(1.1 * 512), int(1.1 * 512)])
    image = tf.image.random_crop(image, (512, 512, 3), seed=new_seed)
    mask = tf.image.random_crop(mask, (512, 512, 33), seed=new_seed)
    image = tf.image.random_flip_up_down(image, seed=new_seed)
    mask = tf.image.random_flip_up_down(mask, seed=new_seed)
    mask = tf.cast(mask, dtype=tf.int32)
    return image, mask
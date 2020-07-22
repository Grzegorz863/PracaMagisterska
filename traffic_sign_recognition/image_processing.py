import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import math

from traffic_sign_recognition.utils.read_train_img import read_train_img


def image_processing(batch_size=32, train_data_augmentation=1):
    image_dir = r'F:\do_pracy_mrg\data\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
    train_images, train_labels, val_images, val_labels = read_train_img(image_dir, omission_image_times=0,
                                                                        first_classes_number=0, last_classes_number=42)

    train_images = np.expand_dims(train_images, axis=3)
    val_images = np.expand_dims(val_images, axis=3)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    train_data_gen = ImageDataGenerator(rescale=1. / 255, brightness_range=[0.2, 1.0], width_shift_range=0.1,
                                        height_shift_range=0.1)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_img_gen = train_data_gen.flow(train_images, train_labels, batch_size=batch_size)
    val_img_gen = val_data_gen.flow(val_images, val_labels, batch_size=batch_size)

    steps_per_epoch_train = math.ceil(len(train_images) / batch_size) * train_data_augmentation
    steps_per_epoch_val = math.ceil(len(val_images) / batch_size)

    return train_img_gen, val_img_gen, steps_per_epoch_train, steps_per_epoch_val

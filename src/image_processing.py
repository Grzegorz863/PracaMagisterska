from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import math

from src.utils.read_train_img import read_train_img


def image_processing(batch_size=32):
    image_dir = r'F:\do_pracy_mrg\data\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
    train_images, train_labels, val_images, val_labels = read_train_img(image_dir, omission_image_times=1,
                                                                        first_classes_number=0, last_classes_number=9)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_img_gen = train_data_gen.flow(train_images, train_labels, batch_size=batch_size)
    val_img_gen = val_data_gen.flow(val_images, val_labels, batch_size=batch_size)

    steps_per_epoch_train = math.ceil(len(train_images) / batch_size)
    steps_per_epoch_val = math.ceil(len(val_images) / batch_size)

    return train_img_gen, val_img_gen, steps_per_epoch_train, steps_per_epoch_val


image_processing(34)

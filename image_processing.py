from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from read_train_img import read_train_img


def image_processing():
    image_dir = r'F:\do_pracy_mrg\data\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
    train_images, train_labels, val_images, val_labels = read_train_img(image_dir, omission_image_times=1,
                                                                        classes_number=10)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_img_gen = train_data_gen.flow(train_images, train_labels)
    val_img_gen = val_data_gen.flow(val_images, val_labels)

    return train_img_gen, val_img_gen

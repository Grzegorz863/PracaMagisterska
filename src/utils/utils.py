import os
import cv2
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


# read images to predict
def read_my_images(path):
    my_images = []
    file_names = []
    valid_images = [".jpg", ".png", ".ppm"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = cv2.resize(cv2.imread(join(path, f)), (100, 100))
        file_names.append(f)
        my_images.append(img)

    return np.asarray(my_images, dtype=np.float32), file_names


def generate_model_plots(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
    plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
    plt.title('Dokladnosc trenowania i walidacji')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.legend()

    plt.show()


def generate_save_file_path(root_path=r'F:\PracaMagisterska\saved_models'):
    model_str = 'model_'
    extension_str = '_.h5'
    max = 0
    for file in os.listdir(root_path):
        file = file.lower()
        model_index = file.find(model_str)
        if model_index is not -1 and file.find(extension_str) is not -1:
            x = file[model_index + len(model_str):]
            x = x[:x.find(extension_str)]
            if x.isnumeric():
                x = int(x)
                if x > max:
                    max = x
    return root_path + '\\' + model_str + str(max + 1) + extension_str

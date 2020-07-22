import os
import pickle
import shutil

import cv2
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

from keras.engine.saving import load_model

from traffic_sign_recognition.test import test_model


def read_my_images(path):
    my_images = []
    file_names = []
    valid_images = [".jpg", ".png", ".ppm"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        resized_image = cv2.resize(cv2.imread(join(path, f)), (100, 100))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        file_names.append(f)
        my_images.append(resized_image)

    return np.asarray(my_images, dtype=np.float32), file_names


def generate_model_plots(history, model_name):
    with open('F:\\PracaMagisterska\\saved_models\\history\\' + model_name + '.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model_info_path = 'F:\\PracaMagisterska\\saved_models\\info\\' + model_name + '\\'
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    epochs = epochs[5:]
    acc = acc[5:]
    val_acc = val_acc[5:]
    loss = loss[5:]
    val_loss = val_loss[5:]

    plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
    plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
    plt.title('Dokladnosc trenowania i walidacji')
    plt.legend()
    plt.savefig(model_info_path + 'accuracy.png')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.legend()
    plt.savefig(model_info_path + 'loss.png')
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
    return root_path + '\\' + model_str + str(max + 1) + extension_str, model_str + str(max + 1), model_str + str(
        max + 1) + extension_str, 'F:\\PracaMagisterska\\saved_models\\info\\' + model_str + str(max + 1)


def generate_model_info_before_fit(model_name, model):
    model_info_path = 'F:\\PracaMagisterska\\saved_models\\info\\' + model_name + '\\'
    src_path = '/traffic_sign_recognition\\'
    if not os.path.exists(model_info_path):
        os.mkdir(path=model_info_path)

    shutil.copyfile(src_path + 'model.py', model_info_path + 'model.py')
    shutil.copyfile(src_path + 'image_processing.py', model_info_path + 'image_processing.py')
    shutil.copyfile(src_path + 'utils\\read_train_img.py', model_info_path + 'read_train_img.py')

    with open(model_info_path + 'summary.txt', 'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))


def generate_model_info_after_fit(model_name, model_name_ext):
    model_info_path = 'F:\\PracaMagisterska\\saved_models\\info\\' + model_name + '\\'
    test_loss, test_acc = test_model('F:\\PracaMagisterska\\saved_models\\' + model_name_ext, False)
    with open(model_info_path + 'summary.txt', 'a') as file:
        file.write('test accuracy: ' + str(test_acc) + '\n')
        file.write('test loss: ' + str(test_loss) + '\n')


def show_model_summary(model_name):
    path = 'F:\\PracaMagisterska\\saved_models\\' + model_name
    model = load_model(path)
    print(model.summary())
    print(test_model(model_name, False))

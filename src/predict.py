import os
from keras.engine.saving import load_model
from keras_preprocessing.image import ImageDataGenerator, np

from src.utils.utils import read_my_images


def predict(model_name, use_gpu):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = load_model(r'../saved_models/' + model_name)
    my_images_tab = read_my_images(r'F:\do_pracy_mrg\data\to_predict')
    test_img_gen = ImageDataGenerator(rescale=1. / 255).flow(my_images_tab)

    predictions = []
    remained = len(my_images_tab)
    for my_image in test_img_gen:
        if remained == 0:
            break
        predictions.append(model.predict(my_image, verbose=1))
        remained -= 1

    return predictions


print(predict('model1.h5', False))

import os
from keras.engine.saving import load_model
from keras_preprocessing.image import ImageDataGenerator, np
from tabulate import tabulate

from src.utils.utils import read_my_images


def predict(model_name, use_gpu):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = load_model(r'../saved_models/' + model_name)
    my_images_tab, _file_names = read_my_images(r'F:\do_pracy_mrg\data\to_predict')
    my_images_tab = np.expand_dims(my_images_tab, axis=3)
    my_images_tab /= 255
    predictions = model.predict(my_images_tab, verbose=1)
    _max_values = []
    _predicted_classes = []
    for prediction in predictions:
        _max_value = np.max(prediction)
        _max_values.append(_max_value)
        _predicted_classes.append(np.argmax(prediction))

    show_predict_results(_max_values, _predicted_classes, _file_names)


def show_predict_results(max_values, predicted_classes, file_names):
    max_values_str = []
    for max_value in max_values:
        max_values_str.append(str(round(max_value * 100, 4)))

    print(tabulate(zip(file_names, predicted_classes, max_values_str), headers=['File name', 'Class', 'Percentage [%]'],
                   tablefmt='orgtbl'))


predict('model_45_.h5', False)

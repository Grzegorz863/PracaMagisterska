import os

from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from src.utils.read_test_img import read_test_img


def test_model(model_name, use_gpu):
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    test_images, test_labels = read_test_img(r'F:\do_pracy_mrg\data\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images',
                                             classes_number=10)
    model = load_model('F:\\PracaMagisterska\\saved_models\\' + model_name)

    test_labels = to_categorical(test_labels)
    test_img_gen = ImageDataGenerator(rescale=1. / 255).flow(test_images, test_labels)
    return model.evaluate_generator(test_img_gen)


test_loss, test_acc = test_model('model1.h5', True)
print('dokładnosc podczas testowania:', test_acc)

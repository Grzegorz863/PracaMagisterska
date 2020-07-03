import os

from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator, np

from src.utils.read_test_img import read_test_img
from keras import backend as K

def test_model(path_to_model, use_gpu):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    test_images, test_labels = read_test_img(r'F:\do_pracy_mrg\data\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images',
                                             first_classes_number=0, last_classes_number=42)
    model = load_model(path_to_model)

    test_labels = to_categorical(test_labels)
    test_images = np.expand_dims(test_images, axis=3)
    test_img_gen = ImageDataGenerator(rescale=1. / 255).flow(test_images, test_labels)

    return model.evaluate_generator(test_img_gen)



test_loss, test_acc = test_model('F:\\PracaMagisterska\\saved_models\\info\\model_82\\max_val_acc.h5', True)
print('dokładnosc podczas testowania max_val_acc:', test_acc)

test_loss, test_acc = test_model('F:\\PracaMagisterska\\saved_models\\info\\model_82\\min_val_loss.h5', True)
print('dokładnosc podczas testowania min_val_loss:', test_acc)

from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from read_test_img import read_test_img

test_images, test_labels = read_test_img(r'F:\do_pracy_mrg\data\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images')
model = load_model(r'.\saved_models\model1.h5')

test_labels = to_categorical(test_labels)

test_img_gen = ImageDataGenerator(rescale=1. / 255).flow(test_images, test_labels)

model.evaluate_generator(test_img_gen)

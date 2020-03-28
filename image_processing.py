from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical

from read_train_img import read_train_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
image_dir = r'F:\do_pracy_mrg\data\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
train_images, train_labels, val_images, val_labels = read_train_img(image_dir, omission_image_times=1,
                                                                    classes_number=10)

train_data_gen = ImageDataGenerator(rescale=1. / 255)
val_data_gen = ImageDataGenerator(rescale=1. / 255)

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

train_img_gen = train_data_gen.flow(train_images, train_labels)
val_img_gen = val_data_gen.flow(val_images, val_labels)

# for train_image, train_label in train_img_gen:
#     print(train_label)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
      train_img_gen,
      steps_per_epoch=165,
      epochs=20,
      validation_data=val_img_gen,
      validation_steps=50)


model.save(r'.\saved_models\model1.h5')


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

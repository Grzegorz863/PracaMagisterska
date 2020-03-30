import os
from keras import layers, optimizers
from keras import models

from src.image_processing import image_processing
from src.utils.utils import generate_model_plots, generate_save_file_path


def fit_model(use_gpu, epochs):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    train_img_gen, val_img_gen, steps_per_epoch_train, steps_per_epoch_val = image_processing()

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
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=val_img_gen,
        validation_steps=steps_per_epoch_val)

    model.save(generate_save_file_path())

    generate_model_plots(history)


fit_model(False, 1)

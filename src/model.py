import os
from keras import layers, optimizers
from keras import models

from src.image_processing import image_processing
from src.utils.utils import generate_model_plots, generate_save_file_path, generate_model_info_before_fit, \
    generate_model_info_after_fit


def fit_model(use_gpu, epochs):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    train_img_gen, val_img_gen, steps_per_epoch_train, steps_per_epoch_val = image_processing()
    path_to_model, model_name, model_name_with_ext = generate_save_file_path()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    generate_model_info_before_fit(model_name, model)

    history = model.fit_generator(
        train_img_gen,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=val_img_gen,
        validation_steps=steps_per_epoch_val)

    model.save(path_to_model)
    generate_model_info_after_fit(model_name, model_name_with_ext)
    generate_model_plots(history, model_name)


fit_model(True, 100)

import os
from keras import layers, optimizers, regularizers
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization

from traffic_sign_recognition.image_processing import image_processing
from traffic_sign_recognition.utils.utils import generate_model_plots, generate_save_file_path, generate_model_info_before_fit, \
    generate_model_info_after_fit


def fit_model(use_gpu, epochs):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    train_img_gen, val_img_gen, steps_per_epoch_train, steps_per_epoch_val = image_processing(batch_size=64, train_data_augmentation=6)
    path_to_model, model_name, model_name_with_ext, path_to_model_info = generate_save_file_path()

    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(100, 100, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(43, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    generate_model_info_before_fit(model_name, model)

    max_val_acc_checkpoint = ModelCheckpoint(path_to_model_info + '\\' + 'max_val_acc.h5', monitor='val_acc',
                                             mode='max', save_best_only=True)
    min_val_loss_checkpoint = ModelCheckpoint(path_to_model_info + '\\' + 'min_val_loss.h5', monitor='val_loss',
                                              mode='min', save_best_only=True)
    history = model.fit_generator(
        train_img_gen,
        callbacks=[max_val_acc_checkpoint, min_val_loss_checkpoint],
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=val_img_gen,
        validation_steps=steps_per_epoch_val)

    model.save(path_to_model)
    generate_model_info_after_fit(model_name, model_name_with_ext)
    generate_model_plots(history, model_name)


fit_model(True, 100)

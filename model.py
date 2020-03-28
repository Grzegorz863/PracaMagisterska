from keras import layers, optimizers
from keras import models
import matplotlib.pyplot as plt

from image_processing import image_processing


def fit_model(save_file_name):
    train_img_gen, val_img_gen = image_processing()

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

    model.save('.\\saved_models\\' + save_file_name)

    generate_model_plots(history)


def generate_model_plots(history):
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

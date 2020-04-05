import os

from keras.engine.saving import load_model

files = iter(os.listdir(r'F:\PracaMagisterska\saved_models'))
next(files)
for f in files:
    model = load_model('F:\\PracaMagisterska\\saved_models\\' + f)
    print('\n\n\n\n' + f)
    model.summary()
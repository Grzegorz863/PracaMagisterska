import os

from PIL import Image

valid_images = [".jpg", ".png", ".ppm"]
path = 'F:\\do_pracy_mrg\\data\\FullIJCNN2013z'
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue
    im = Image.open(path + '\\' + f)
    im.save(path + '\\jpg\\' + name + '.jpg')

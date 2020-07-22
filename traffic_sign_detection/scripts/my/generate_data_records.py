import csv
import hashlib
import io
import os
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('train_output_path', r'F:\do_pracy_mrg\data\detection_records_output\train.record',
                    'Path to train output TFRecord')
flags.DEFINE_string('val_output_path', r'F:\do_pracy_mrg\data\detection_records_output\val.record',
                    'Path to val output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(file_name_from_csv, image_info):
    # TODO(user): Populate the following variables from your example.
    height = 800  # Image height
    width = 1360  # Image width
    path = 'F:\\do_pracy_mrg\\data\\FullIJCNN2013z\\jpg\\'
    file_name_png = os.path.splitext(file_name_from_csv)[0] + '.jpg'
    full_file_name = path + file_name_png

    with tf.gfile.GFile(full_file_name, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmins = list(map(int, image_info[0]))  # List of normalized left x coordinates in bounding box (1 per box)
    ymins = list(map(int, image_info[1]))
    xmaxs = list(map(int, image_info[2]))  # List of normalized right x coordinates in bounding box
    ymaxs = list(map(int, image_info[3]))  # List of normalized bottom y coordinates in bounding box
    # normalizacja
    xmins = [x / width for x in xmins]
    ymins = [y / height for y in ymins]
    xmaxs = [x / width for x in xmaxs]
    ymaxs = [y / height for y in ymaxs]
    classes_text = [bytes(str(x), encoding='utf8') for x in image_info[4]]
    classes = image_info[4]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image.height),
        'image/width': dataset_util.int64_feature(image.width),
        'image/filename': dataset_util.bytes_feature(file_name_png.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(file_name_png.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def read_photos_info(csv_file_name, train):
    path = 'F:\\do_pracy_mrg\\data\\FullIJCNN2013z'
    gt_file = open(path + '\\' + csv_file_name)
    gt_reader = csv.reader(gt_file, delimiter=';')

    imgs = {}
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    max_len=0
    max_file = ''
    for row in gt_reader:
        file_name, xmin, ymin, xmax, ymax, class_43 = row
        prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]  # zakazu
        mandatory = [33, 34, 35, 36, 37, 38, 39, 40]  # nakazu
        danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # ostrzegawcze
        other = [6, 12, 13, 14, 17, 32, 41, 42] # inne
        class_43 = int(class_43)
        if class_43 in prohibitory:
            class_3 = 1
        elif class_43 in mandatory:
            class_3 = 2
        elif class_43 in danger:
            class_3 = 3
        elif class_43 in other:
            class_3 = 4
        else:
            class_3 = -1

        if file_name not in imgs:
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            classes = []
            imgs[file_name] = [xmins, ymins, xmaxs, ymaxs, classes]

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)

        ymaxs.append(ymax)
        classes.append(class_3)

    if train:
        print('sss')
    else:
        for x in range(700, 900):
            file_name_form_dir = '00' + str(x) + '.ppm'
            if file_name_form_dir not in imgs:
                imgs[file_name_form_dir] = [[], [], [], [], [-1]]
    print("dsd")
    return imgs


def main(_):
    # train set
    train_writer = tf.python_io.TFRecordWriter(FLAGS.train_output_path)
    images = read_photos_info('gt_full.csv', True)

    for file_name in images:
        tf_example = create_tf_example(file_name, images[file_name])
        train_writer.write(tf_example.SerializeToString())

    train_writer.close()

    # val set
    # val_writer = tf.python_io.TFRecordWriter(FLAGS.val_output_path)
    # images = read_photos_info('gt_val_full.csv', False)
    #
    # for file_name in images:
    #     tf_example = create_tf_example(file_name, images[file_name])
    #     val_writer.write(tf_example.SerializeToString())
    #
    # val_writer.close()


if __name__ == '__main__':
    tf.app.run()

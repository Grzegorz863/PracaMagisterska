import cv2
import csv
import numpy as np


def read_train_img(root_path, omission_image_times=0, first_classes_number=0, last_classes_number=42):
    train_images = []
    val_images = []
    train_labels = []
    val_labels = []
    # loop over all 42 classes
    for c in range(first_classes_number, last_classes_number + 1):
        prefix = root_path + '\\' + format(c, '05d') + '\\'  # subdirectory for class
        gt_file = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gt_file2 = open(prefix + 'GT-' + format(c, '05d') + '.csv')
        gt_reader = csv.reader(gt_file, delimiter=';')  # csv parser for annotations file
        next(gt_reader)  # skip header
        image_number = (sum(1 for _ in gt_file2) - 1)
        different_signs_number = int(image_number / 30)
        train_signs_num = int(round(different_signs_number * 0.7, 0))
        # val_signs_num = different_signs_number - train_signs_num
        for row in gt_reader:
            sign_index = int(row[0][0:5])
            if sign_index <= train_signs_num:
                resized_image = cv2.resize(cv2.imread(prefix + row[0]), (100, 100))
                train_images.append(resized_image)  # the 1th column is the filename
                train_labels.append(str(int(row[7])-first_classes_number))  # the 8th column is the label
            else:
                resized_image = cv2.resize(cv2.imread(prefix + row[0]), (100, 100))
                val_images.append(resized_image)  # the 1th column is the filename
                val_labels.append(str(int(row[7])-first_classes_number))  # the 8th column is the label

            for i in range(0, omission_image_times):
                try:
                    next(gt_reader)
                except StopIteration:
                    print("An StopIteration occurred")
        gt_file.close()
        gt_file2.close()
    return np.asarray(train_images, dtype=np.float32), np.asarray(train_labels, dtype=np.float32), \
           np.asarray(val_images, dtype=np.float32), np.asarray(val_labels, dtype=np.float32),

import cv2
import csv
import numpy as np


def read_test_img(root_path, omission_image_times=0, first_classes_number=0, last_classes_number=43):
    test_images = []
    test_labels = []

    gt_file = open(root_path + '\\' + 'GT-final_test.csv')  # annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # csv parser for annotations file
    next(gt_reader)  # skip header
    for row in gt_reader:
        if first_classes_number <= int(row[7]) <= last_classes_number:
            resized_image = cv2.resize(cv2.imread(root_path + '\\' + row[0]), (100, 100))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            test_images.append(resized_image)  # the 1th column is the filename
            test_labels.append(str(int(row[7])-first_classes_number))  # the 8th column is the label

        for i in range(0, omission_image_times):
            next(gt_reader)
    gt_file.close()

    return np.asarray(test_images, dtype=np.float32), np.asarray(test_labels, dtype=np.float32)

# -*- coding: utf-8 -*-
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)
import os
import h5py
import numpy as np
from sklearn.utils import shuffle


def load_dataset(path, mode):
    texts, labels = [], []
    with open(path + os.sep + mode + ".csv", "rb") as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = "{} {}".format(line[1].lower(), line[2].lower())
            label = int(line[0]) - 1
            texts.append(text)
            labels.append(label)
    np.random.seed(42)
    texts, labels = shuffle(texts, labels)
    return texts, labels


def generator(filename, batch_size=128):
    with h5py.File(filename, 'r', libver='latest') as file:
        n_chunks = len(file.keys()) / 2
        for i in range(n_chunks):
            texts = file["text_{}".format(i)]
            labels = file["label_{}".format(i)]
            for j in range(0, len(texts), batch_size):
                yield texts[j: min(j + batch_size, len(texts))], labels[j: min(j + batch_size, len(texts))]

def get_size(filename):
    with h5py.File(filename, 'r', libver='latest') as file:
        n_chunks = len(file.keys()) / 2
        total_length = sum([len(file["label_{}".format(i)]) for i in range(n_chunks)])
        return total_length

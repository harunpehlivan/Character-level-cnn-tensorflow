import argparse
from utils import load_dataset
import numpy as np
import h5py
import os
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)

def get_args():
    parser = argparse.ArgumentParser("Preprocess data")
    parser.add_argument("--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
                        help="Valid characters used for model")
    parser.add_argument("--length", type=int, default=1014, help="The maximum length of input")
    parser.add_argument("--data_path", type=str, default="../data", help="path to the dataset")
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()
    return args

def create_data(opts, mode):
    raw_text, raw_label = load_dataset(path=opts.data_path, mode=mode)
    identity_mat = np.identity(len(opts.alphabet), dtype=np.int8)
    char_dictionary = list(opts.alphabet)
    with h5py.File(opts.data_path + os.sep + "{}.h5".format(mode), 'w', libver='latest') as file:
        index = 0
        texts, labels = [], []
        for idx, document in enumerate(raw_text):
            text = np.array([identity_mat[char_dictionary.index(i)] for i in list(document) if i in char_dictionary])
            if len(text) > opts.length:
                text = text[:opts.length]
            elif len(text) < opts.length:
                text = np.concatenate((text, np.zeros((opts.length - len(text), len(opts.alphabet)))))
            texts.append(text)
            if (idx + 1) % opts.chunk_size == 0:
                sys.stdout.write("\tWriting chunk %d\r" % (index + 1))
                sys.stdout.flush()
                file.create_dataset('text_{}'.format(index), shape=(opts.chunk_size, opts.length, len(opts.alphabet)),
                                    dtype='f', compression='lzf', data=np.concatenate(texts, axis=0))
                file.create_dataset('label_{}'.format(index), shape=(opts.chunk_size,), dtype='i8', compression='lzf',
                                    data=np.array(raw_label[index * opts.chunk_size:(index + 1) * opts.chunk_size]))
                texts, labels = [], []
                index += 1

        file.create_dataset('text_{}'.format(index),
                            shape=((idx + 1) % opts.chunk_size, opts.length, len(opts.alphabet)),
                            dtype='f', compression='lzf', data=np.concatenate(texts, axis=0))
        file.create_dataset('label_{}'.format(index), shape=((idx + 1) % opts.chunk_size,), dtype='i8',
                            compression='lzf',
                            data=np.array(raw_label[index * opts.chunk_size:]) - 1)




if __name__ == "__main__":
    opts = get_args()
    create_data(opts, mode = 'train')
    create_data(opts, mode = 'test')

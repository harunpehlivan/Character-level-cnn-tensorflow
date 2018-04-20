import argparse
from src.utils import generator, get_size
from src.char_level_cnn import Char_level_cnn
import tensorflow as tf
import os
import sys
import numpy as np


def get_args():
    parser = argparse.ArgumentParser("Character-level convolutional networks for text classification")
    parser.add_argument("--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
                        help="Valid characters used for model")
    parser.add_argument("--length", type=int, default=1014, help="The maximum length of input")
    parser.add_argument("--input_folder", type=str, default="data")
    parser.add_argument("--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("--data_path", type=str, default="data", help="path to the dataset")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=5000, help="Number of iterations for one epoch")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr_halve_interval", type=float, default=100,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--test_interval", type=int, default=5000,
                        help="Number of iterations between 2 consecutive tests")
    parser.add_argument("--checkpoint_path", type=str, default="trained_models", help="path to store trained models")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="Number of iterations between 2 consecutive store")
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    return args


def main(opts):
    model = Char_level_cnn(batch_size=opts.batch_size, learning_rate=opts.lr, num_classes=opts.num_classes,
                           num_characters=len(opts.alphabet))

    if opts.feature == "large":
        num_conv_features = 1024
        kernel_size = [7, 7, 3, 3, 3, 3]
    elif opts.feature == "small":
        num_conv_features = 256
        kernel_size = [7, 7, 3, 3, 3, 3]
    train_set_size = get_size(opts.data_path + os.sep + "train.h5")
    print("There are {} instances in train set".format(train_set_size))
    test_set_size = get_size(opts.data_path + os.sep + "test.h5")
    print("There are {} instances in test set".format(test_set_size))
    with tf.Graph().as_default():
        x = tf.placeholder(shape=[None, opts.length, len(opts.alphabet)], dtype=tf.float32, name="x")
        y = tf.placeholder(shape=[None], dtype=tf.int32, name="y")
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.Variable(0, name="global_step", trainable=False)

        logits = model.forward(features=x, kernel_size=kernel_size, num_filters=num_conv_features, padding='VALID',
                               pool_size=3, keep_prob=keep_prob)

        loss = model.loss(logits=logits, labels=y)
        accuracy = model.accuracy(logits, y)
        train_op = model.train(loss, global_step=global_step)
        print('Initializing the Variables.')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if not os.path.exists(opts.checkpoint_path):
            os.makedirs(opts.checkpoint_path)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        with tf.Session(config=session_conf) as sess:
            sess.run(init)
            tr_gen = generator(opts.data_path + os.sep + "train.h5", batch_size=opts.batch_size)
            for _ in range(opts.num_epochs * opts.epoch_size):
                try:
                    data = tr_gen.next()
                except StopIteration:
                    tr_gen = generator(opts.data_path + os.sep + "train.h5", batch_size=opts.batch_size)
                    data = tr_gen.next()
                texts, labels = data
                _, current_loss, current_accuracy, step = sess.run([train_op, loss, accuracy, global_step],
                                                                   feed_dict={x: texts, y: labels,
                                                                              keep_prob: opts.dropout})
                sys.stdout.write(
                    "\tTraining phase: Step: {} Loss: {} Accuracy: {}\r".format(step, current_loss, current_accuracy))
                sys.stdout.flush()
                current_iter = tf.train.global_step(sess, global_step)
                if current_iter % opts.test_interval == 0:
                    print("Evaluating test set at epoch {}:".format(current_iter / opts.test_interval))
                    num_test_batches = test_set_size / opts.batch_size
                    test_losses = []
                    test_accuracy = []
                    batch_lengths = []
                    te_gen = generator(opts.data_path + os.sep + "test.h5", batch_size=opts.batch_size)
                    for i in range(num_test_batches):
                        test_data = te_gen.next()
                        test_texts, test_labels = test_data
                        current_test_loss, current_test_accuracy = sess.run([loss, accuracy],
                                                                            feed_dict={x: test_texts, y: test_labels,
                                                                                       keep_prob: 1.0})
                        sys.stdout.write(
                            "\tTest phase: Step {} Loss: {} Accuracy: {}\r".format(
                                i, current_test_loss, current_test_accuracy))
                        sys.stdout.flush()
                        test_losses.append(current_test_loss)
                        test_accuracy.append(current_test_accuracy)
                        batch_lengths.append(len(test_labels))

                    mean_losses = np.sum(np.array(test_losses) * np.array(batch_lengths), dtype=np.float32) / np.sum(
                        batch_lengths)
                    mean_accuracy = np.sum(np.array(test_accuracy) * np.array(batch_lengths),
                                           dtype=np.float32) / np.sum(
                        batch_lengths)
                    print("Loss: {} Accuracy: {}".format(mean_losses, mean_accuracy))

                if current_iter % opts.checkpoint_interval == 0:
                    saved_path = saver.save(sess, opts.checkpoint_path + os.sep + "model", global_step=current_iter)
                    print("Saved model checkpoint to {}".format(saved_path))


if __name__ == "__main__":
    opts = get_args()
    main(opts)

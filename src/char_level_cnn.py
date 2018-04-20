import tensorflow as tf


class Char_level_cnn(object):
    def __init__(self, batch_size=128, learning_rate=1e-2, num_classes=14, num_characters=68):
        super(Char_level_cnn, self).__init__()
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_classes = num_classes
        self._num_characters = num_characters

    def forward(self, features, num_filters,
                kernel_size, padding, pool_size, keep_prob):
        if num_filters == 1024:
            stddev_initialization = 0.02
            num_fully_connected_features = 2048
        else:
            stddev_initialization = 0.05
            num_fully_connected_features = 1024

        with tf.name_scope("conv-maxpool-0"):
            weight = self._create_weights([self._num_characters, kernel_size[0], 1, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            input = tf.expand_dims(tf.transpose(features, [0, 2, 1]), -1)
            conv = tf.nn.conv2d(input=input, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            max = tf.nn.max_pool(value=activation, ksize=[1, 1, pool_size, 1], strides=[1, 1, pool_size, 1],
                                 padding=padding, name='maxpool')

        with tf.name_scope("conv-maxpool-1"):
            weight = self._create_weights([1, kernel_size[1], num_filters, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            conv = tf.nn.conv2d(input=max, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            max = tf.nn.max_pool(value=activation, ksize=[1, 1, pool_size, 1], strides=[1, 1, pool_size, 1],
                                 padding=padding, name='maxpool')

        with tf.name_scope("conv-2"):
            weight = self._create_weights([1, kernel_size[2], num_filters, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            conv = tf.nn.conv2d(input=max, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

        with tf.name_scope("conv-3"):
            weight = self._create_weights([1, kernel_size[3], num_filters, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            conv = tf.nn.conv2d(input=activation, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

        with tf.name_scope("conv-4"):
            weight = self._create_weights([1, kernel_size[4], num_filters, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            conv = tf.nn.conv2d(input=activation, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

        with tf.name_scope("conv-5"):
            weight = self._create_weights([1, kernel_size[5], num_filters, num_filters], stddev_initialization)
            bias = self._create_bias([num_filters])

            conv = tf.nn.conv2d(input=activation, filter=weight, strides=[1, 1, 1, 1], padding=padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            max = tf.nn.max_pool(value=activation, ksize=[1, 1, pool_size, 1], strides=[1, 1, pool_size, 1],
                                 padding=padding, name='maxpool')

        with tf.variable_scope('fc-0'):
            new_feature_size = num_filters * ((features.get_shape().as_list()[1] - 96) / 27)
            flatten = tf.reshape(max, [-1, new_feature_size])
            weight = self._create_weights([new_feature_size, num_fully_connected_features], stddev_initialization)
            bias = self._create_bias([num_fully_connected_features])

            dense = tf.nn.bias_add(tf.matmul(flatten, weight), bias)
            drop = tf.nn.dropout(dense, keep_prob)

        with tf.variable_scope('fc-1'):
            weight = self._create_weights(
                [num_fully_connected_features, num_fully_connected_features],
                stddev_initialization)
            bias = self._create_bias([num_fully_connected_features])

            dense = tf.nn.bias_add(tf.matmul(drop, weight), bias)
            drop = tf.nn.dropout(dense, keep_prob)

        with tf.variable_scope('final-fc'):
            weight = self._create_weights(
                [num_fully_connected_features, self._num_classes],
                stddev_initialization)
            bias = self._create_bias([self._num_classes])

            dense = tf.nn.bias_add(tf.matmul(drop, weight), bias)

        return dense

    def _create_weights(self, shape, stddev):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32, name='weight'))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32, name='bias'))

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        return loss

    def accuracy(self, logits, y):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64)), dtype=tf.float32))
        return accuracy

    def train(self, loss, global_step):
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(self._learning_rate,)
        train_op = optimizer.minimize(loss, global_step=global_step)  # 1

        # gvs = optimizer.compute_gradients(loss)    #2
        # capped_gvs = [(tf.clip_by_value(grad, 1e-6, 1.), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        # gradients, variables = zip(*optimizer.compute_gradients(loss))  #3
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        return train_op

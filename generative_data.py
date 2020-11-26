import tensorflow as tf
import tensorflow_datasets as tfds


def pre_process_data(ds, info, px_family):
    """
    :param ds: TensorFlow Dataset object
    :param info: TensorFlow DatasetInfo object
    :param px_family: data distribution assumption (e.g. Bernoulli, Gaussian, etc...)
    :return: the passed in data set with map pre-processing applied
    """
    # support assertions
    assert px_family in {'Bernoulli', 'Beta', 'Kumaraswamy'} or 'Normal' in px_family

    def bernoulli_sample(x):
        orig_shape = tf.shape(x)
        p = tf.reshape(tf.cast(x, dtype=tf.float32) / 255.0, [-1, 1])
        logits = tf.math.log(tf.concat((1 - p, p), axis=1))
        return tf.cast(tf.reshape(tf.random.categorical(logits, 1), orig_shape), dtype=tf.float32)

    # apply pre-processing function for given data set and modelling assumptions
    if px_family == 'Bernoulli':
        return ds.map(lambda d: {'image': bernoulli_sample(d['image']),
                                 'label': d['label']},
                      num_parallel_calls=16)
    if px_family in {'Beta', 'Kumaraswamy'}:
        return ds.map(lambda d: {'image': tf.clip_by_value(tf.cast(d['image'], dtype=tf.float32) / d['image'].dtype.max,
                                                           clip_value_min=1e-2, clip_value_max=1 - 1e-2),
                                 'label': d['label']},
                      num_parallel_calls=16)
    if 'Normal' in px_family:
        return ds.map(lambda d: {'image': tf.cast(d['image'], dtype=tf.float32) / d['image'].dtype.max,
                                 'label': d['label'] if 'label' in d.keys() else tf.random.categorical(tf.ones([1, 10]), 1)},
                      num_parallel_calls=16)


def configure_data_set(ds, info, px_family, batch_size, shuffle):
    """
    :param ds: TensorFlow data set object
    :param info: TensorFlow DatasetInfo object
    :param px_family: data distribution assumption (e.g. Bernoulli, Gaussian, etc...)
    :param batch_size: batch size
    :param shuffle: whether to reshuffle each iteration
    :return: a configured TensorFlow data set object
    """
    # enable shuffling and repeats
    ds = ds.shuffle(1000 * batch_size, reshuffle_each_iteration=shuffle)

    # batch the data before pre-processing
    ds = ds.batch(batch_size)

    # pre-process the data set
    with tf.device('/cpu:0'):
        ds = pre_process_data(ds, info, px_family)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load_data_set(data_set_name, px_family, batch_size=100):
    """
    :param data_set_name: data set name--call tfds.list_builders() for options
    :param px_family: data distribution assumption (e.g. Bernoulli, Gaussian, etc...)
    :param batch_size: training/testing batch size
    :return:
        train_ds: TensorFlow Dataset object for training data
        test_ds: TensorFlow Dataset object for testing data
        info: data set info object
    """
    # load training and test sets
    train_ds, info = tfds.load(name=data_set_name, split=tfds.Split.TRAIN, with_info=True)
    test_ds = tfds.load(name=data_set_name, split=tfds.Split.TEST, with_info=False)

    # create and configure the data sets
    train_ds = configure_data_set(train_ds, info, px_family, batch_size, shuffle=True)
    test_ds = configure_data_set(test_ds, info, px_family, batch_size, shuffle=False)

    return train_ds, test_ds, info

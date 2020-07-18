import os
import librosa
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# 获取浮点数组
def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# 获取整型数据
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 把数据添加到TFRecord中
def data_example(data, label):
    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# 开始创建tfrecord数据
def create_data_tfrecord(data_list_path, save_path):
    with open(data_list_path, 'r') as f:
        data = f.readlines()
    with tf.io.TFRecordWriter(save_path) as writer:
        for d in tqdm(data):
            path, label = d.replace('\n', '').split('\t')
            y1, sr1 = librosa.load(path, duration=4)
            ps = librosa.feature.melspectrogram(y=y1, sr=sr1).reshape(-1).tolist()
            print(len(ps))
            tf_example = data_example(ps, int(label))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    create_data_tfrecord('dataset/train_list.txt', 'dataset/train.tfrecord')
    create_data_tfrecord('dataset/test_list.txt', 'dataset/test.tfrecord')

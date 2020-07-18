import tensorflow as tf
import reader
import numpy as np

class_dim = 3
EPOCHS = 500

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=20, kernel_size=5, activation=tf.nn.relu, input_shape=(128, 173, 1)),
    tf.keras.layers.Conv2D(filters=50, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
])

model.summary()


# 定义优化方法
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer = tf.keras.optimizers.Adam(learning_rate=87e-10)

train_dataset = reader.train_reader_tfrecord('dataset/train.tfrecord', EPOCHS)
test_dataset = reader.test_reader_tfrecord('dataset/test.tfrecord')
for batch_id, data in enumerate(train_dataset):
    sounds = data['data'].numpy().reshape((-1, 128, 173, 1))
    labels = data['label']
    # 执行训练
    with tf.GradientTape() as tape:
        predictions = model(sounds)
        # 获取损失值
        train_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        train_loss = tf.reduce_mean(train_loss)
        # 获取准确率
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, predictions)
        train_accuracy = np.sum(train_accuracy.numpy()) / len(train_accuracy.numpy())

    # 更新梯度
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if batch_id % 20 == 0:
        print("Batch %d, Loss %f, Accuracy %f" % (batch_id, train_loss.numpy(), train_accuracy))

    if batch_id % 200 == 0 and batch_id!=0:
        test_losses = list()
        test_accuracies = list()
        for d in test_dataset:
            test_sounds = d['data'].numpy().reshape((-1, 128, 173, 1))
            test_labels = d['label']

            test_result = model(test_sounds)
            # 获取损失值
            test_loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, test_result)
            test_loss = tf.reduce_mean(test_loss)
            test_losses.append(test_loss)
            # 获取准确率
            test_accuracy = tf.keras.metrics.sparse_categorical_accuracy(test_labels, test_result)
            test_accuracy = np.sum(test_accuracy.numpy()) / len(test_accuracy.numpy())
            test_accuracies.append(test_accuracy)

        print('=================================================')
        print("Test, Loss %f, Accuracy %f" % (
            sum(test_losses) / len(test_losses), sum(test_accuracies) / len(test_accuracies)))
        print('=================================================')

        # 保存模型
        model.save(filepath='models/cnn.h5')

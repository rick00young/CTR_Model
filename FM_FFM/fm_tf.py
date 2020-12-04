
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler


continuous_feature = ['I'] * 13
continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
category_feature = ['C'] * 26
category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]


def data_process():
    ids_file = 'data/ids.np'
    vals_file = 'data/vals.np'
    label_file = 'data/label.np'
    if os.path.exists(ids_file) and os.path.exists(vals_file) and os.path.exists(label_file):
        ids = pickle.load(open(ids_file, 'rb'))
        vals = pickle.load(open(vals_file, 'rb'))
        label = pickle.load(open(label_file, 'rb'))
        return (label, ids, vals, )

    path = '../GBDT_LR/data/'
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    # df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('data/data.csv', index=False)

    # 类别特征one-hot编码
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    useful_data = data['Label'] != -1
    train_data = data[useful_data]
    label_raw = train_data['Label']
    train_data.drop(['Label'], axis=1, inplace=True)
    ids = []
    vals = []
    label = []
    for i in range(len(train_data)):
        line = train_data.loc[i]
        keys = line.keys()
        _ids = []
        _vals = []
        _label = label_raw.loc[i]
        for j, k in enumerate(keys):
            v = line.get(k)
            if k == 'Label':
                _label = v
                continue
            if str(k).startswith('C'):
                if v == 0:
                    continue
                _ids.append(j)
                _vals.append(v)
            else:
                _ids.append(j)
                _vals.append(v)
        label.append(_label)
        ids.append(_ids)
        vals.append(_vals)
    pickle.dump(label, open('data/label.np', 'wb'))
    pickle.dump(ids, open('data/ids.np', 'wb'))
    pickle.dump(vals, open('data/vals.np', 'wb'))
    feature_size = train_data.shape[1]
    return (label, ids, vals, )


label, ids, vals = data_process()

data_len = len(label)
split_num = int(data_len*.2)

train_label = label[0:data_len-split_num]
train_ids = ids[0:data_len-split_num]
train_vals = vals[0:data_len-split_num]

test_label = label[data_len-split_num:]
test_ids = ids[data_len-split_num:]
test_vals = vals[data_len-split_num:]

batch_size = 32
feature_size = 13104
field_size = len(ids[0])
embedding_size = 32
batch_size = 32
num_epochs = 10
epochs = 60



def process(ids, vals, label):
    label = label
    ids = ids
    vals = vals
    return (ids, vals), label


train_dataset = tf.data.Dataset.from_tensor_slices((train_ids, train_vals, train_label)).map(process).batch(batch_size=batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_ids, test_vals, test_label)).map(process).batch(batch_size=batch_size)


def create_network():
    feature_ids_input = layers.Input((field_size,), dtype=tf.int32, name='ids')
    feature_vals_input = layers.Input((field_size,), dtype=tf.float32, name='vals')
    feature_values_input = layers.Reshape(target_shape=(field_size, 1))(feature_vals_input)

    # FM_B = tf.Variable(initial_value=tf.constant_initializer(0.0), name='fm_bias', shape=(1,))
    fm_w = layers.Embedding(input_dim=feature_size, output_dim=1,
                            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                            name='fm_w')
    fm_v = layers.Embedding(input_dim=feature_size, output_dim=embedding_size,
                            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                            name='fm_v')

    # bias = layers.Layer().add_weight(name='fm_bias', initializer='zero', shape=[1])
    bias = tf.Variable(name='fm_bias', initial_value=[0.0], shape=[1], trainable=True)
    # tf.Variable

    # # first order
    feature_weights = fm_w(feature_ids_input)
    y_w = tf.reduce_sum(tf.multiply(feature_weights, feature_values_input, name='fm_1-1_multiply'),
                        1, name='fm_1-2_sum')

    # output = tf.math.sigmoid(y_w)
    # model = tf.keras.Model(inputs=[feature_ids_input, feature_vals_input], outputs=output)
    # model.summary()

    # second order
    embeddings = fm_v(feature_ids_input)
    embeddings = tf.multiply(embeddings, feature_values_input, name='fm_2-1_multiply')
    sum_square = tf.square(tf.reduce_sum(embeddings, axis=1, name='fm_2-2_sum'), name='fm_2-3_square')
    square_sum = tf.reduce_sum(tf.square(embeddings, name='fm_2-4_square'), axis=1, name='fm_2-5_sum')
    y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum, name='fm_2-6_subtract'), axis=1, name='fm_2-7_sum',
                              keepdims=True)

    # y = layers.concatenate([y_w, y_v, FM_B])
    # y_output = layers.Reshape(target_shape=(1,))(y_w + y_v + bias)
    y_output = y_w + y_v + bias

    # output = layers.Dense(1, activation='sigmoid', name='fm_output')(y_output)
    output = tf.math.sigmoid(y_output)
    # loss = tf.losses.binary_crossentropy(label, )

    #
    model = tf.keras.Model(inputs=[feature_ids_input, feature_vals_input], outputs=output)
    model.summary()
    return model

def train():
    model = create_network()
    callbacks = [
        keras.callbacks.ModelCheckpoint("model/save_at_{epoch}.h5"),
    ]

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )
    history = model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, validation_data=test_dataset, batch_size=32
    )


if '__main__' == __name__:
    train()

'''
40/40 [==============================] - 1s 28ms/step - loss: 0.7379 - accuracy: 0.9844 - val_loss: 24.1718 - val_accuracy: 0.7618- accuracy: 0.
Epoch 52/60
40/40 [==============================] - 1s 25ms/step - loss: 1.0630 - accuracy: 0.9883 - val_loss: 23.9790 - val_accuracy: 0.7743s: 1.7019 
Epoch 53/60
40/40 [==============================] - 1s 26ms/step - loss: 0.7249 - accuracy: 0.9914 - val_loss: 25.6265 - val_accuracy: 0.7743 loss: 1.3885 - accu - ETA: 0s - loss: 0.7839 - ac
Epoch 54/60
40/40 [==============================] - 1s 26ms/step - loss: 0.2647 - accuracy: 0.9820 - val_loss: 23.0220 - val_accuracy: 0.7210
Epoch 55/60
40/40 [==============================] - 1s 24ms/step - loss: 0.2750 - accuracy: 0.9898 - val_loss: 22.9936 - val_accuracy: 0.7555
Epoch 56/60
40/40 [==============================] - 1s 26ms/step - loss: 0.0478 - accuracy: 0.9930 - val_loss: 23.1696 - val_accuracy: 0.7524
Epoch 57/60
40/40 [==============================] - 1s 29ms/step - loss: 0.0406 - accuracy: 0.9984 - val_loss: 23.0113 - val_accuracy: 0.7555
Epoch 58/60
40/40 [==============================] - 2s 45ms/step - loss: 0.0195 - accuracy: 0.9969 - val_loss: 22.8983 - val_accuracy: 0.7524
Epoch 59/60
40/40 [==============================] - 2s 61ms/step - loss: 0.0122 - accuracy: 0.9977 - val_loss: 23.7126 - val_accuracy: 0.7555oss: 0.0139 - ac
Epoch 60/60
40/40 [==============================] - 1s 28ms/step - loss: 0.0085 - accuracy: 0.9992 - val_loss: 23.5280 - val_accuracy: 0.7586042 
'''


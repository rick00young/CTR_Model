
import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
num_epochs = 10
epochs = 60

layer = [256, 128, 64]
drop_out = [.3, .3, .3]


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
    bias = tf.Variable(name='fm_bias', initial_value=[0.0], shape=[1], trainable=True, dtype=tf.float32)

    # # first order
    feature_weights = fm_w(feature_ids_input)
    y_w = tf.reduce_sum(tf.multiply(feature_weights, feature_values_input, name='fm_1-1_multiply'),
                        1, name='fm_1-2_sum')
    # second order
    embeddings = fm_v(feature_ids_input)
    embeddings = tf.multiply(embeddings, feature_values_input, name='fm_2-1_multiply')
    sum_square = tf.square(tf.reduce_sum(embeddings, axis=1, name='fm_2-2_sum'), name='fm_2-3_square')
    square_sum = tf.reduce_sum(tf.square(embeddings, name='fm_2-4_square'), axis=1, name='fm_2-5_sum')
    y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum, name='fm_2-6_subtract'), axis=1, name='fm_2-7_sum',
                              keepdims=True)
    # deep part
    deep_inputs = layers.Reshape(target_shape=(field_size*embedding_size,))(embeddings)
    for i in range(len(layer)):
        deep_inputs = layers.Dense(units=layer[i], kernel_regularizer=tf.keras.regularizers.l2(),
                                   name='fm_3_%s_dense' % i)(deep_inputs)
        # batch_norm
        deep_inputs = layers.BatchNormalization(trainable=True)(deep_inputs)
        deep_inputs = layers.Dropout(rate=drop_out[i])(deep_inputs)

    y_deep = layers.Dense(units=1, activation=tf.identity, kernel_regularizer=tf.keras.regularizers.l2(),
                          name='fm_3_output')(deep_inputs)

    y_output = y_w + y_v + bias + y_deep

    output = tf.math.sigmoid(y_output)

    model = tf.keras.Model(inputs=[feature_ids_input, feature_vals_input], outputs=output)
    model.summary()
    return model



def train():
    model = create_network()
    callbacks = [
        keras.callbacks.ModelCheckpoint("model/save_at_{epoch}.h5"),
    ]

    def reg_loss():
        reg_loss_list = []
        for w in model.trainable_variables:
            name = w.name
            if name.split('/')[0] in ['fm_w', 'ff_v']:
                reg_loss_list.append(tf.nn.l2_loss(w))

        return tf.reduce_sum(reg_loss_list)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=['binary_crossentropy', reg_loss],
                  loss_weights=[1, 0.2],
                  metrics=['accuracy']
                  )
    history = model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, validation_data=test_dataset, batch_size=batch_size
    )
    plot_model(history)

def plot_model(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if '__main__' == __name__:
    train()

'''
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
ids (InputLayer)                [(None, 39)]         0                                            
__________________________________________________________________________________________________
vals (InputLayer)               [(None, 39)]         0                                            
__________________________________________________________________________________________________
reshape (Reshape)               (None, 39, 1)        0           vals[0][0]                       
__________________________________________________________________________________________________
fm_v (Embedding)                (None, 39, 32)       419328      ids[0][0]                        
__________________________________________________________________________________________________
tf_op_layer_fm_2-1_multiply (Te [(None, 39, 32)]     0           fm_v[0][0]                       
                                                                 reshape[0][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1248)         0           tf_op_layer_fm_2-1_multiply[0][0]
__________________________________________________________________________________________________
fm_3_0_dense (Dense)            (None, 256)          319744      reshape_1[0][0]                  
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 256)          1024        fm_3_0_dense[0][0]               
__________________________________________________________________________________________________
dropout (Dropout)               (None, 256)          0           batch_normalization[0][0]        
__________________________________________________________________________________________________
tf_op_layer_fm_2-2_sum (TensorF [(None, 32)]         0           tf_op_layer_fm_2-1_multiply[0][0]
__________________________________________________________________________________________________
tf_op_layer_fm_2-4_square (Tens [(None, 39, 32)]     0           tf_op_layer_fm_2-1_multiply[0][0]
__________________________________________________________________________________________________
fm_3_1_dense (Dense)            (None, 128)          32896       dropout[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_fm_2-3_square (Tens [(None, 32)]         0           tf_op_layer_fm_2-2_sum[0][0]     
__________________________________________________________________________________________________
tf_op_layer_fm_2-5_sum (TensorF [(None, 32)]         0           tf_op_layer_fm_2-4_square[0][0]  
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 128)          512         fm_3_1_dense[0][0]               
__________________________________________________________________________________________________
fm_w (Embedding)                (None, 39, 1)        13104       ids[0][0]                        
__________________________________________________________________________________________________
tf_op_layer_fm_2-6_subtract (Te [(None, 32)]         0           tf_op_layer_fm_2-3_square[0][0]  
                                                                 tf_op_layer_fm_2-5_sum[0][0]     
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 128)          0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
tf_op_layer_fm_1-1_multiply (Te [(None, 39, 1)]      0           fm_w[0][0]                       
                                                                 reshape[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_fm_2-7_sum (TensorF [(None, 1)]          0           tf_op_layer_fm_2-6_subtract[0][0]
__________________________________________________________________________________________________
fm_3_2_dense (Dense)            (None, 64)           8256        dropout_1[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_fm_1-2_sum (TensorF [(None, 1)]          0           tf_op_layer_fm_1-1_multiply[0][0]
__________________________________________________________________________________________________
tf_op_layer_Mul (TensorFlowOpLa [(None, 1)]          0           tf_op_layer_fm_2-7_sum[0][0]     
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 64)           256         fm_3_2_dense[0][0]               
__________________________________________________________________________________________________
tf_op_layer_AddV2 (TensorFlowOp [(None, 1)]          0           tf_op_layer_fm_1-2_sum[0][0]     
                                                                 tf_op_layer_Mul[0][0]            
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 64)           0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
tf_op_layer_AddV2_1 (TensorFlow [(None, 1)]          0           tf_op_layer_AddV2[0][0]          
__________________________________________________________________________________________________
fm_3_output (Dense)             (None, 1)            65          dropout_2[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_AddV2_2 (TensorFlow [(None, 1)]          0           tf_op_layer_AddV2_1[0][0]        
                                                                 fm_3_output[0][0]                
__________________________________________________________________________________________________
tf_op_layer_Sigmoid (TensorFlow [(None, 1)]          0           tf_op_layer_AddV2_2[0][0]        
==================================================================================================
Total params: 795,185
Trainable params: 794,289
Non-trainable params: 896


40/40 [==============================] - 2s 42ms/step - loss: 2.3872 - accuracy: 0.9750 - val_loss: 29.2972 - val_accuracy: 0.77745 - accuracy: 0. - ETA: 0s - loss: 3.8306 - accuracy: 0. - ETA: 0s - loss: 3.5442 - accu - ETA: 0s - loss: 2.5767 - accuracy: 
Epoch 51/60
40/40 [==============================] - 2s 39ms/step - loss: 1.1171 - accuracy: 0.9805 - val_loss: 25.6204 - val_accuracy: 0.7555oss: 1.7918 - ac - ETA: 0s - loss: 1.2864 - accuracy: 0. - ETA: 0s - loss: 1.2006 - accuracy
Epoch 52/60
40/40 [==============================] - 2s 39ms/step - loss: 1.2765 - accuracy: 0.9852 - val_loss: 22.5591 - val_accuracy: 0.7586 - loss: 0.035 - ETA: 0s - loss: 2.0
Epoch 53/60
40/40 [==============================] - 1s 37ms/step - loss: 0.4570 - accuracy: 0.9867 - val_loss: 26.7725 - val_accuracy: 0.7680 - loss: - ETA: 0s - loss: 0.5246 - accuracy
Epoch 54/60
40/40 [==============================] - 2s 39ms/step - loss: 3.7317 - accuracy: 0.9859 - val_loss: 19.8140 - val_accuracy: 0.7335 ETA: 0s - loss: 3.9241 - accuracy: 0.
Epoch 55/60
40/40 [==============================] - 2s 39ms/step - loss: 0.1404 - accuracy: 0.9891 - val_loss: 22.2136 - val_accuracy: 0.7492 - loss: - ETA: 0s - loss: 0.1717 - accura
Epoch 56/60
40/40 [==============================] - 2s 42ms/step - loss: 0.0518 - accuracy: 0.9945 - val_loss: 21.5435 - val_accuracy: 0.7367
Epoch 57/60
40/40 [==============================] - 2s 41ms/step - loss: 0.1017 - accuracy: 0.9937 - val_loss: 22.0791 - val_accuracy: 0.7712 0.051 - ETA: 0s - loss: 0.0560 - accu
Epoch 58/60
40/40 [==============================] - 1s 37ms/step - loss: 0.2250 - accuracy: 0.9914 - val_loss: 21.5067 - val_accuracy: 0.7398 - loss: 0.2408 - accuracy: 0.
Epoch 59/60
40/40 [==============================] - ETA: 0s - loss: 0.1366 - accuracy: 0.9928 ETA: 1s - loss: 0.4876 - ac - ETA: 0s - loss: 0.1 - ETA: 0s - loss: 0.1570 - accuracy - 1s 37ms/step - loss: 0.1339 - accuracy: 0.9930 - val_loss: 19.7131 - val_accuracy: 0.7335
Epoch 60/60
40/40 [==============================] - ETA: 0s - loss: 0.0320 - accuracy: 0.9968 ETA: 1s - loss: 0.0242 - accura - ETA: 0s - l - 1s 36ms/step - loss: 0.0318 - accuracy: 0.9969 - val_loss: 19.4057 - val_accuracy: 0.7273
'''


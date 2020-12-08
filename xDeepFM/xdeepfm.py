import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
try:
    from . import config
    from .DataLoader import FeatureDictionary, DataParser
except Exception as e:
    import config
    from DataLoader import FeatureDictionary, DataParser

from tensorflow.keras import layers


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

def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test,


dfTrain, dfTest, X_train, y_train, X_test, ids_test = load_data()
print('load_data_over')



fd = FeatureDictionary(dfTrain, dfTest, numeric_cols=config.NUMERIC_COLS,
                       ignore_cols=config.IGNORE_COLS,
                       cate_cols=config.CATEGORICAL_COLS)

print(fd.feat_dim)
print(fd.feat_dict)

data_parser = DataParser(feat_dict=fd)
cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
# cate_Xi_test, cate_Xv_test, numeric_Xv_test, y_test, ids_test = data_parser.parse(df=dfTest)


def process(cate_ids, cate_vals, y_label):
    # feat_ins
    # print('----',len(cate_ids))
    return (sorted(cate_ids), cate_vals), y_label


train_dataset = tf.data.Dataset.from_tensor_slices((cate_Xi_train, cate_Xv_train, y_train)).\
    map(process).batch(batch_size=32)

for i in train_dataset.take(1):
    print(i)
    pass



data_len = len(cate_Xi_train)
split_num = int(data_len*.2)


batch_size = 32
feature_size = 13104+1
field_size = len(cate_Xi_train[0])
embedding_size = 64
epochs = 60
direct = False
deep_layers = [256, 128, 64]
cross_layer = [256, 128, 64]


feat_ids_input = layers.Input((field_size,), dtype=tf.int64, name='ids_input')
feat_vals_input = layers.Input((field_size,), dtype=tf.float32, name='vals_input')
# feat_values_input = layers.Reshape(target_shape=(field_size, 1))(feat_vals_input)

feat_embedding = layers.Embedding(input_dim=feature_size, output_dim=embedding_size,
                                  embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                  name='feat_embedding')
# fm_sparse_index = tf.sparse.SparseTensor(feat_ids_input, feat_vals_input,
#                                          dense_shape=[batch_size, field_size,])

feat_weight = layers.Embedding(input_dim=feature_size, output_dim=embedding_size,
                                  embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                  name='feat_weight')

embeddings = feat_embedding(feat_ids_input)
# weights = feat_weight(feat_ids_input)

# embeddings_weight = tf.multiply(embeddings, weights)
# embeddings_input = tf.reduce_sum(embeddings, axis=1)
embeddings_output = layers.Reshape(target_shape=(field_size*embedding_size,))(embeddings)

# linear model
linear_output = layers.Dense(units=1, use_bias=True, name='linear_output')(embeddings_output)


# dnn model
deep_layer = embeddings_output
for i, layer_size in enumerate(deep_layers):
    deep_layer = layers.Dense(units=layer_size, use_bias=True, activation='relu', name='deep_output_%s' % i)(deep_layer)

deep_output = layers.Dense(units=1, use_bias=True, name='deep_output')(deep_layer)


class cin(layers.Layer):
    def __init__(self, **kwargs):
        super(cin, self).__init__(**kwargs)
        self.filters = []
        self.bias = []
        self.field_nums = []
        self.field_nums.append(field_size)
        for i, layer_size in enumerate(cross_layer):
            _filter = layers.Layer().add_weight(name='filter_%s' % i,
                                                shape=(1, self.field_nums[0]*self.field_nums[-1], layer_size),
                                                dtype=tf.float32, initializer=tf.keras.initializers.GlorotNormal())
            _bias = layers.Layer().add_weight(name='bias_%s' % i, shape=(layer_size,), dtype=tf.float32,
                                              initializer=tf.keras.initializers.zeros())
            self.filters.append(_filter)
            self.bias.append(_bias)
            self.field_nums.append(layer_size)

    # def build(self, input_shape):
    #     pass
    def call(self, inputs, **kwargs):
        hidden_nn_layers = []
        field_nums = []
        nn_input = inputs
        field_nums.append(field_size)
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], [1]*embedding_size, 2) # (embedding_size, batch_size, field_size, 1)
        for i, layer_size in enumerate(cross_layer):
            split_tensor = tf.split(hidden_nn_layers[-1], [1]*embedding_size, 2) # (embedding_size, batch_size, field_size, 1)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True, name='split_matmul_%s' % i) # (embedding_size, batch_size, field_size, field_size)
            dot_result_o = tf.reshape(dot_result_m, shape=[embedding_size,-1, field_nums[0]*field_nums[-1]]) # （embedding_size, batch_size, field_size^2）
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2]) #(batch_size, embedding_size, field_size^2)
            curr_out = tf.nn.conv1d(dot_result, self.filters[i], stride=1, padding='VALID', name='conv1d_%s' % i) # (batch_size, embedding_size, layer_size)
            curr_out = tf.nn.bias_add(curr_out, self.bias[i])
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # (batch_size, layer_size, embedding_size)
            # if direct:
            direct_connect = curr_out
            next_hidden = curr_out
            field_nums.append(layer_size)
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = tf.concat(final_result, axis=1) # (32, 448, 64)
        result = tf.reduce_sum(result, axis=-1, name='reduce_sum_result') # ((32, 448))
        return result

def create_network():
    result = cin()(embeddings)
    # if res:
    ex_fm_output = layers.Dense(units=1, use_bias=True, name='ex_fm_output')(result)
    y_output = linear_output + deep_output + ex_fm_output
    output = tf.math.sigmoid(y_output)
    model = tf.keras.Model(inputs=[feat_ids_input, feat_vals_input], outputs=output)
    model.summary()
    return model


def train():
    model = create_network()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model/save_at_{epoch}.h5"),
    ]
    def reg_loss():
        reg_loss_list = []
        for w in model.trainable_variables:
            name = w.name
            if name.split('/')[0] in ['fm_w', 'ff_v']:
                reg_loss_list.append(tf.nn.l2_loss(w))
        return tf.reduce_sum(reg_loss_list)


    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=['binary_crossentropy'],
                  loss_weights=[1],
                  metrics=['accuracy']
                  )
    history = model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks)

if '__main__' == __name__:
    train()

'''
Model: "functional_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
ids_input (InputLayer)          [(None, 30)]         0                                            
__________________________________________________________________________________________________
feat_embedding (Embedding)      (None, 30, 64)       838720      ids_input[0][0]                  
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1920)         0           feat_embedding[0][0]             
__________________________________________________________________________________________________
deep_output_0 (Dense)           (None, 256)          491776      reshape_1[0][0]                  
__________________________________________________________________________________________________
deep_output_1 (Dense)           (None, 128)          32896       deep_output_0[0][0]              
__________________________________________________________________________________________________
deep_output_2 (Dense)           (None, 64)           8256        deep_output_1[0][0]              
__________________________________________________________________________________________________
linear_output (Dense)           (None, 1)            1921        reshape_1[0][0]                  
__________________________________________________________________________________________________
deep_output (Dense)             (None, 1)            65          deep_output_2[0][0]              
__________________________________________________________________________________________________
cin_1 (cin)                     (None, 448)          0           feat_embedding[0][0]             
__________________________________________________________________________________________________
tf_op_layer_AddV2_4 (TensorFlow [(None, 1)]          0           linear_output[0][0]              
                                                                 deep_output[0][0]                
__________________________________________________________________________________________________
ex_fm_output (Dense)            (None, 1)            449         cin_1[0][0]                      
__________________________________________________________________________________________________
tf_op_layer_AddV2_5 (TensorFlow [(None, 1)]          0           tf_op_layer_AddV2_4[0][0]        
                                                                 ex_fm_output[0][0]               
__________________________________________________________________________________________________
vals_input (InputLayer)         [(None, 30)]         0                                            
__________________________________________________________________________________________________
tf_op_layer_Sigmoid_2 (TensorFl [(None, 1)]          0           tf_op_layer_AddV2_5[0][0]        
==================================================================================================
Total params: 1,374,083
Trainable params: 1,374,083
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/60
223/313 [====================>.........] - ETA: 1:23 - loss: 0.1656 - accuracy: 0.9624
'''
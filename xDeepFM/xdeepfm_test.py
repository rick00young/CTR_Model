import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle


from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
try:
    from . import config
    from .DataLoader import FeatureDictionary, DataParser
except Exception as e:
    import config
    from DataLoader import FeatureDictionary, DataParser

from tensorflow.keras import layers

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


feat_ids_input = layers.Input((field_size,), dtype=tf.int64, name='ids_input', batch_size=batch_size)
feat_vals_input = layers.Input((field_size,), dtype=tf.float32, name='vals_input', batch_size=batch_size)
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
    #     self.filed_nums = [input_shape.shape[1].value]
    #     self.filters = []
    #     self.bias = []
    #     for i, field_size in enumerate(self.layers):
    #         self.filters.append(self.add_weight(name='filters'+str(i),
    #                                             shape=(1, self.field_nums[-1] * self.filed_nums[0]),
    #                                             dtype=tf.float32,
    #                                             initializer=glorot_normal(seed=self.seed + i),
    #                                             regularizer=l2(self.l2_reg)))
    #         self.bias.append(self.add_weight(name='bias'+str(i), shape=(field_size), dtype=tf.float32,
    #                                          initializer=tf.keras.initializer.Zeros()))
    #         if self.if_direct:
    #             self.filed_nums.append(field_size // 2)
    #         else:
    #             self.filed_nums.append(field_size)
    #     super(cin, self).build(input_shape)

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

            # filters = layers.Layer().add_weight(name='filter_%s' % i, shape=(1, field_nums[0]*field_nums[-1], layer_size),
            #                                     dtype=tf.float32, initializer=tf.keras.initializers.GlorotNormal())
            curr_out = tf.nn.conv1d(dot_result, self.filters[i], stride=1, padding='VALID', name='conv1d_%s' % i) # (batch_size, embedding_size, layer_size)
            # bias = layers.Layer().add_weight(name='bias_%s' % i, shape=(layer_size,), dtype=tf.float32,
            #                                  initializer=tf.keras.initializers.zeros())
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


result = cin()(embeddings)
# if res:
ex_fm_output = layers.Dense(units=1, use_bias=True, name='ex_fm_output')(result)

y_output = linear_output + deep_output + ex_fm_output

output = tf.math.sigmoid(y_output)

model = tf.keras.Model(inputs=[feat_ids_input, feat_vals_input], outputs=output)
model.summary()


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
    train_dataset, epochs=epochs, callbacks=callbacks, batch_size=batch_size
)

# extreme FM
# hidden_nn_layers = []
# field_nums = []
# nn_input = embeddings
# field_nums.append(field_size)
# hidden_nn_layers.append(nn_input)
# final_result = []
# split_tensor0 = tf.split(hidden_nn_layers[0], [1]*embedding_size, 2) # (embedding_size, batch_size, field_size, 1)
# for i, layer_size in enumerate(cross_layer):
#     split_tensor = tf.split(hidden_nn_layers[-1], [1]*embedding_size, 2) # (embedding_size, batch_size, field_size, 1)
#     dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True, name='split_matmul_%s' % i) # (embedding_size, batch_size, field_size, field_size)
#     # dot_result_o = layers.Reshape(target_shape=(field_size, -1, field_nums[0]*field_nums[-1]))(dot_result_m)
#     dot_result_o = tf.reshape(dot_result_m, shape=[embedding_size,-1, field_nums[0]*field_nums[-1]]) # （embedding_size, batch_size, field_size^2）
#     # dot_result = tf.transpose(dot_result_o, perm=[1,2,0])
#     dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2]) #(batch_size, embedding_size, field_size^2)
#
#     filters = layers.Layer().add_weight(name='filter_%s' % i, shape=(1, field_nums[0]*field_nums[-1], layer_size),
#                                         dtype=tf.float32, initializer=tf.keras.initializers.GlorotNormal())
#     curr_out = tf.nn.conv1d(dot_result, filters, stride=1, padding='VALID', name='conv1d_%s' % i) # (batch_size, embedding_size, layer_size)
#     # curr_out = layers.Conv1D(filters=layer_size,kernel_size=field_nums[0]*field_nums[-1], strides=1)(dot_result)
#     # filters = tf.Variable(name="f_" + str(i), initial_value=tf.random_normal_initializer(),
#     #                           dtype=tf.float32, shape=[1, field_nums[-1] * field_nums[0], layer_size])
#     # curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
#
#     bias = layers.Layer().add_weight(name='bias_%s' % i, shape=(layer_size,), dtype=tf.float32,
#                                      initializer=tf.keras.initializers.zeros())
#     curr_out = tf.nn.bias_add(curr_out, bias)
#     curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # (batch_size, layer_size, embedding_size)
#     # if direct:
#     direct_connect = curr_out
#     next_hidden = curr_out
#     field_nums.append(layer_size)
#     final_result.append(direct_connect)
#     hidden_nn_layers.append(next_hidden)
#
# result = tf.concat(final_result, axis=1) # (32, 448, 64)
# result = tf.reduce_sum(result, axis=-1, name='reduce_sum_result') # ((32, 448))
# # if res:
# ex_fm_output = layers.Dense(units=1, use_bias=True, name='ex_fm_output')(result)
#

deep_output = layers.Dense(units=1, use_bias=True, name='deep_output')(deep_layer)



layers.Dot
embedding = tf.nn.embedding_lookup_sparse()








np.spar
tf.keras.layers.Add()

tf.keras.layers.S

tf.sparse.sparse_dense_matmul
tf.sparse.sparse_dense_matmul
tf.keras

tf.matmul
tf.keras.layers.Subtract
tf.nn.embedding_lookup_sparse
tf.nn.embedding_lookup
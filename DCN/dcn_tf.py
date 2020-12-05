'''
Deep&Cross Networ

DCN能够有效地捕获有限度的有效特征的相互作用，学会高度非线性的相互作用，不需要人工特征工程或遍历搜索，并具有较低的计算成本。
论文的主要贡献包括：

1）提出了一种新的交叉网络，在每个层上明确地应用特征交叉，有效地学习有界度的预测交叉特征，并且不需要手工特征工程或穷举搜索。
2）跨网络简单而有效。通过设计，各层的多项式级数最高，并由层深度决定。网络由所有的交叉项组成，它们的系数各不相同。
3）跨网络内存高效，易于实现。
4）实验结果表明，交叉网络（DCN）在LogLoss上与DNN相比少了近一个量级的参数量。
'''

import tensorflow as tf

import pandas as pd
import numpy as np

'https://www.jianshu.com/p/77719fc252fa'

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

# one hot前特征数
field_size = len(cate_Xv_train[0])
# one hot后的特征数
cate_feature_size = fd.feat_dim
numeric_feature_size = len(numeric_Xv_train[0])
embedding_size = 32

total_size = field_size*embedding_size + numeric_feature_size

deep_layers = [256, 62, 32]
dropout_layers_deep = [.5, .5, .5]
cross_layer_num = 3
epochs = 60
batch_size = 32
l2_reg = .01


def process(cate_ids, cate_vals, numeric_vals, y_label):
    return (cate_ids, cate_vals, numeric_vals), y_label


train_dataset = tf.data.Dataset.from_tensor_slices((cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train)).\
    map(process).batch(batch_size=batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((cate_Xi_test, cate_Xv_test, numeric_Xv_test, y_test)).\
#     map(process).batch(batch_size=batch_size)

def create_network():
    feat_index_input = layers.Input((field_size, ), dtype=tf.int32, name='feat_index')
    feat_value_input = layers.Input((field_size,), dtype=tf.float32, name='feat_value')
    feat_value_input_r = layers.Reshape(target_shape=(field_size, 1), name='feat_value_reshape')(feat_value_input)

    numeric_feat_value_input = layers.Input((numeric_feature_size, ), dtype=tf.float32, name='num_value')


    feature_embedding = layers.Embedding(input_dim=cate_feature_size, output_dim=embedding_size,
                                         embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                         name='feature_embedding')
    embeddings = feature_embedding(feat_index_input)
    embeddings = tf.multiply(embeddings, feat_value_input_r, name='feat_index_x_value')

    embeddings = layers.Reshape(target_shape=(field_size*embedding_size,))(embeddings)

    x0 = layers.concatenate([numeric_feat_value_input, embeddings], axis=1)

    # deep part
    y_deep = layers.Dropout(rate=dropout_layers_deep[0])(x0)
    for i in range(len(deep_layers)):
        y_deep = layers.Dense(units=deep_layers[i], activation='relu', use_bias=True, name='deep_%s' % i)(y_deep)
        y_deep = layers.Dropout(rate=dropout_layers_deep[i])(y_deep)

    # cross part
    _x0 = layers.Reshape(target_shape=(total_size, 1))(x0)
    x_l = _x0
    for i in range(cross_layer_num):
        x_l = tf.matmul(_x0, x_l, transpose_b=True, name='cross_matmul_%s' % i)
        # cross_weight = tf.Variable(shape=(total_size, 1), )
        # cross_weight =
        # x_l = tf.tensordot(x_l, )
        x_l = layers.Dense(units=1, use_bias=True, name='cross_dense_%s' % i)(x_l)

    cross_network_out = layers.Reshape(target_shape=(total_size,))(x_l)

    concat_output = layers.concatenate([cross_network_out, y_deep], axis=1)
    y_output = layers.Dense(units=1, use_bias=True, name='concat_projection')(concat_output)

    output = tf.math.sigmoid(y_output)

    model = tf.keras.Model(inputs=[feat_index_input, feat_value_input, numeric_feat_value_input], outputs=output)
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
                  loss_weights=[1, 0.2],
                  metrics=['accuracy']
                  )
    history = model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, batch_size=batch_size
    )

if '__main__' == __name__:
    train()


'''
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
feat_index (InputLayer)         [(None, 30)]         0                                            
__________________________________________________________________________________________________
feat_value (InputLayer)         [(None, 30)]         0                                            
__________________________________________________________________________________________________
feature_embedding (Embedding)   (None, 30, 32)       7904        feat_index[0][0]                 
__________________________________________________________________________________________________
feat_value_reshape (Reshape)    (None, 30, 1)        0           feat_value[0][0]                 
__________________________________________________________________________________________________
tf_op_layer_feat_index_x_value  [(None, 30, 32)]     0           feature_embedding[0][0]          
                                                                 feat_value_reshape[0][0]         
__________________________________________________________________________________________________
num_value (InputLayer)          [(None, 9)]          0                                            
__________________________________________________________________________________________________
reshape (Reshape)               (None, 960)          0           tf_op_layer_feat_index_x_value[0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 969)          0           num_value[0][0]                  
                                                                 reshape[0][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 969, 1)       0           concatenate[0][0]                
__________________________________________________________________________________________________
tf_op_layer_BatchMatMulV2 (Tens [(None, 969, 969)]   0           reshape_1[0][0]                  
                                                                 reshape_1[0][0]                  
__________________________________________________________________________________________________
dropout (Dropout)               (None, 969)          0           concatenate[0][0]                
__________________________________________________________________________________________________
cross_dense_0 (Dense)           (None, 969, 1)       970         tf_op_layer_BatchMatMulV2[0][0]  
__________________________________________________________________________________________________
deep_0 (Dense)                  (None, 256)          248320      dropout[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_BatchMatMulV2_1 (Te [(None, 969, 969)]   0           reshape_1[0][0]                  
                                                                 cross_dense_0[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           deep_0[0][0]                     
__________________________________________________________________________________________________
cross_dense_1 (Dense)           (None, 969, 1)       970         tf_op_layer_BatchMatMulV2_1[0][0]
__________________________________________________________________________________________________
deep_1 (Dense)                  (None, 62)           15934       dropout_1[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_BatchMatMulV2_2 (Te [(None, 969, 969)]   0           reshape_1[0][0]                  
                                                                 cross_dense_1[0][0]              
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 62)           0           deep_1[0][0]                     
__________________________________________________________________________________________________
cross_dense_2 (Dense)           (None, 969, 1)       970         tf_op_layer_BatchMatMulV2_2[0][0]
__________________________________________________________________________________________________
deep_2 (Dense)                  (None, 32)           2016        dropout_2[0][0]                  
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 969)          0           cross_dense_2[0][0]              
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 32)           0           deep_2[0][0]                     
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1001)         0           reshape_2[0][0]                  
                                                                 dropout_3[0][0]                  
__________________________________________________________________________________________________
concat_projection (Dense)       (None, 1)            1002        concatenate_1[0][0]              
__________________________________________________________________________________________________
tf_op_layer_Sigmoid (TensorFlow [(None, 1)]          0           concat_projection[0][0]          
==================================================================================================
Total params: 278,086
Trainable params: 278,086
Non-trainable params: 0



Epoch 1/10
313/313 [==============================] - 166s 532ms/step - loss: 0.1932 - accuracy: 0.9579
Epoch 2/10
313/313 [==============================] - 175s 558ms/step - loss: 0.1610 - accuracy: 0.9621
Epoch 3/10
186/313 [================>.............] - ETA: 1:10 - loss: 0.1470 - accuracy: 0.9649
'''
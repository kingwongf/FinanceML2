import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
np.set_printoptions(threshold=np.inf)
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tools.featGen import get_norm_side



pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199


class nn(object):

    def __init__(self, Xy_loc, cat, drop_col, val_date_split, test_date_split, batch_size=32, side=True):
        self.Xy = pd.read_pickle(Xy_loc)
        self.cat = cat ## sould be ['Date', 'target', 'ticker']
        self.drop_cat_leakage_col = drop_col
        self.val_date_split = val_date_split
        self.test_date_split = test_date_split
        self.numeric = [e for e in self.Xy.columns.tolist() if e not in self.drop_cat_leakage_col]
        self.batch_size = batch_size
        self.side = side

    def set_target(self, col):
        if self.side:
            self.Xy['target'] = get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645))
            self.Xy['target'] = self.Xy['target'].astype('category')
            # self.Xy['target'] = to_categorical(get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645)).astype('category'),3)
            # self.Xy = self.Xy[self.Xy["target"].notnull()]
        else: self.Xy['target'] = self.Xy[col]
        self.Xy = self.Xy.drop([col], axis=1)


    def min_max(self):
        '''
        ## TODO change to train min/ max only to prevent leakage
        :return: scaled features and target
        '''
        self.Xy[self.numeric] = (self.Xy[self.numeric] - self.Xy[self.numeric].min(axis=0)) / (self.Xy[self.numeric].max(axis=0) - self.Xy[self.numeric].min(axis=0))


    def train_val_test_split(self):
        return self.Xy[:self.val_date_split], \
               self.Xy.loc[self.val_date_split: self.test_date_split], \
               self.Xy[self.test_date_split:]



    def df_to_dataset(self,dataframe, shuffle=True, cat_no=3):
        dataframe = dataframe.copy()
        if cat_no==None: labels = dataframe.pop('target')
        else: labels = to_categorical(dataframe.pop('target'),cat_no)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(self.batch_size)
        return ds


    def to_ds(self):
        train, val, test = self.train_val_test_split()
        if self.side:
            return self.df_to_dataset(train), \
               self.df_to_dataset(val, shuffle=False), \
               self.df_to_dataset(test, shuffle=False)
        else:
            return self.df_to_dataset(train, cat_no=None), \
                   self.df_to_dataset(val, shuffle=False, cat_no=None), \
                   self.df_to_dataset(test, shuffle=False, cat_no=None)


    def gen_feature_columns(self):
        '''
        generate feature columns (limited to X features only)
        ## TODO recode to dynamic list of cat, not just "ticker"
        :return: features columns
        '''
        feature_columns = []
        # numeric cols, make sure everything is float
        for header in self.numeric:
            # feature_columns.append(feature_column.numeric_column(header, normalizer_fn=lambda x: (x - 3.0) / 4.2))
            feature_columns.append(feature_column.numeric_column(header))

        # indicator cols
        thal = feature_column.categorical_column_with_vocabulary_list('ticker', self.Xy.ticker.unique().tolist())
        thal_one_hot = feature_column.indicator_column(thal)
        feature_columns.append(thal_one_hot)

        # embedding cols
        # thal_embedding = feature_column.embedding_column(thal, dimension=8)
        # feature_columns.append(thal_embedding)
        return feature_columns


    def build_nn(self):

        feature_layer = tf.keras.layers.DenseFeatures(self.gen_feature_columns())

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        def coeff_determination(y_true, y_pred):
            SS_res = K.sum(K.square(y_true - y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res / (SS_tot + K.epsilon()))

        model = tf.keras.Sequential([
            feature_layer,
            layers.Conv1D(64, kernel_size=5),
            layers.Dense(64, activation='relu',
                         activity_regularizer=regularizers.l1(0.0001)),
            #  layers.Dropout(0.5),
            # layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            #  layers.BatchNormalization(),
            #  layers.Dropout(0.1),
            #  layers.Dense(8, activation='relu'),
            #layers.BatchNormalization(),
            layers.Dense(3, activation='softmax')
            # layers.LeakyReLU(alpha=0.3)
        ])

        if self.side:   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else: model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[coeff_determination])

        return model, callback

    def run_nn(self):

        self.set_target('fwdret')
        # print(self.Xy[self.Xy['target'].isna()]['target'])
        # exit()
        self.min_max()
        # print(self.Xy[self.Xy.isna().any(axis=1)])
        # self.Xy['na_bool'] = self.Xy['price'].isna()
        # print(self.Xy.shape)
        # print(self.Xy[self.Xy['na_bool']].groupby(['ticker']).agg('count'))
        # exit()
        train_ds, val_ds, test_ds = self.to_ds()
        model, callback = self.build_nn()
        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=10, callbacks=[callback])

        # print(model.layers[0].output)

        loss, accuracy = model.evaluate(test_ds)
        # print("R^2", coeff_determination)
        print("Accuracy", accuracy)

test_nn = nn("pre_data/feat_useod_daily.pkl", ['ticker'], ['fwdret', 'ticker'], '2018-01-10', '2018-01-15', 32, True)
test_nn.run_nn()

'''
Xy.dayofweek = Xy.dayofweek.astype(np.float64)
# print(Xy.info(verbose=True))

train, test = train_test_split(Xy, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 1
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# for feature_batch, label_batch in train_ds.take(1):
#  # print('Every feature:', list(feature_batch.keys()))
#  numeric_cols = list(feature_batch.keys())
#  print(numeric_cols)


## TODO customize below
numeric_cols = Xy.columns.tolist()
numeric_cols = [e for e in numeric_cols if e not in ('ticker', 'target')]

feature_columns = []

# numeric cols, make sure everything is float
for header in numeric_cols:
  # feature_columns.append(feature_column.numeric_column(header, normalizer_fn=lambda x: (x - 3.0) / 4.2))
  feature_columns.append(feature_column.numeric_column(header))

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list('ticker', Xy.ticker.unique().tolist())
# thal_one_hot = feature_column.indicator_column(thal)
# feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model = tf.keras.Sequential([
  feature_layer,
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(0.0001)),
  #  layers.Dropout(0.5),
   layers.BatchNormalization(),
   layers.Dense(64, activation='relu'),
  #  layers.BatchNormalization(),
  #  layers.Dropout(0.1),
  #  layers.Dense(8, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='linear')
    # layers.LeakyReLU(alpha=0.3)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[coeff_determination, 'accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5, callbacks=[callback])

# print(model.layers[0].output)

loss, coeff_determination, accuracy = model.evaluate(test_ds)
print("R^2", coeff_determination)
print("Accuracy", accuracy)

'''
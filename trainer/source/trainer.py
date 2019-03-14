
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import backend as K
import keras
import math
from sklearn.preprocessing import RobustScaler
# import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# import seaborn as sns
# get_ipython().magic(u'matplotlib inline')
from datetime import datetime
from keras import backend as K
from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from utils_v3 import get_data_v2 as get_data
from keras.layers.cudnn_recurrent import CuDNNGRU as GRU
K.clear_session()


# In[2]:
time_val = 30
time_freq_str = 'S'
time_freq = time_freq_str
time = '%d%s' % (time_val, time_freq)
seq = 20
ahead = '5Min'
behind = '10Min'
name = '%s-%sx%s' % (ahead, behind, time)

# def get_data(csv_path, time_val, time_freq_str, series_count, nrows=None, test_count=15000, test_only=False, resample=False):

(x_train, y_train, x_test, y_test, test_scalers) = get_data('/floyd/input/data/prepped_5Min-10Minx30S.npz', time_val, time_freq_str, seq, None, 15000, False, False, no_high_low=True, npz=True, ahead=ahead, behind=behind) 

print('x_train: %d' % x_train.shape[0])
print('y_train: %d' % y_train.shape[0])
print('x_test: %d' % x_test.shape[0])
print('y_test: %d' % y_test.shape[0])

# In[3]:


K.get_session().list_devices()


# In[4]:


x = x_train
y = y_train


# In[9]:


checkpointer = ModelCheckpoint(filepath='sp%s.hdf5' % name, verbose=1, save_best_only=True)
csv_logger = CSVLogger('logs/sp%s.log' % name, append=False)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

callbacks_list = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.00001, verbose=1), checkpointer, csv_logger]


# In[10]:


def build_model(layers, d):
    model = Sequential()
    
    model.add(GRU(512, input_shape=(layers[0], layers[1]), return_sequences=True))
    model.add(Dropout(d))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128, return_sequences=False))
    model.add(BatchNormalization())       
    model.add(Dense(128, activation='relu'))#,kernel_regularizer=regularizers.l2()))
    model.add(Dropout(d))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse',optimizer='adam')
    return model


# In[11]:


model = build_model([seq+1, x.shape[2]], 0.3)


# In[ ]:


model.fit(
    x_train,
    y_train,
    batch_size=512, # Using mini-batch gradient descent
    epochs=100, # Doesn't matter because we are using early stopping
    validation_data=(x_test, y_test),
    verbose=1,callbacks = callbacks_list)


# In[ ]:


K.clear_session()


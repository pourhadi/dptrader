# from utils import get_data
#from utils import get_data_v2 as get_data
#floyd run --cpu2 --env default --data pourhadi/datasets/historical/4:data 'python dataprep.py'
from utils_v3 import get_data_v2
import pandas as pd
import numpy as np




time_val = 2
time_freq_str = 'S'
time_freq = time_freq_str
time = '%d%s' % (time_val, time_freq)
seq = 60
ahead = '10S'
behind = '2Min'
name = '%s-%sx%s' % (ahead, behind, time)


(x_train, y_train, x_test, y_test, test_scalers) = get_data_v2('/floyd/input/data/historical.csv', time_val, time_freq_str, seq, None, 15000, False, resample=True, preprocessed=False, npz=False, no_high_low=True, ahead=ahead, behind=behind, categorical=False) 


np.savez_compressed(('prepped_%s' % name), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
import utils
import os
import pandas as pd
import numpy as np

datadir='data/'
feature='fbank'

#xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])

#merged = xtrain.merge(y_train, how='left')
steps = 6
#utils.get_sequence(merged, steps, feature)

#x_test = utils.load_data(os.path.join(datadir, feature), flag='test')
#utils.get_test_sequence(x_test, steps, feature)
datadir='data/'
feature='mfcc'

xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
merged = xtrain.merge(y_train, how='left')
steps = 6
utils.get_sequence(merged, steps, feature)
x_test = utils.load_data(os.path.join(datadir, feature), flag='test')
utils.get_test_sequence(x_test, steps, feature)


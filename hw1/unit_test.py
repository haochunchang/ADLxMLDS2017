import utils
import os
import pandas as pd
import numpy as np

datadir='data/'
feature='fbank'

xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])

merged = xtrain.merge(y_train, how='left')
steps = 30
x_train = utils.get_sequence(merged, steps)
np.save('./data/{}_step{}'.format(feature, steps), x_train)

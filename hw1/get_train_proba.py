import utils
import os, sys, pickle
import pandas as pd
import numpy as np
import editdistance
from bayes_opt import BayesianOptimization
from keras.models import model_from_json

path=os.path.join('.', 'models')
train = utils.load_data(os.path.join('./data', 'fbank'))
x_test = np.load('./data/sents.npy')
idx = train['id']
idx.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in idx])
new_idx = []
for person, new_df in idx.groupby(level=0):
    for sentence, fea_id in new_df.groupby(level=1):
        fea = list(fea_id)
        fea += [person+'_'+sentence+'_'+str(i) for i in range(len(fea)+1, 778)]
        new_idx += fea
    
idx = pd.Series(new_idx)
#====================================================================
with open(os.path.join(path, 'best_model.json'), "r") as json_file:
     best = model_from_json(json_file.read())
best.load_weights(os.path.join(path, 'best.h5'))

best_pred = best.predict(x_test, batch_size=64, verbose=1)
with open('best_train_proba.pkl', 'wb') as f:
    pickle.dump((best_pred, idx), f)
#=====================================================================

with open(os.path.join(path, 'cnn_model.json'), "r") as json_file:
     cnn = model_from_json(json_file.read())
cnn.load_weights(os.path.join(path, 'cnn.h5'))

cnn_pred = cnn.predict(x_test, batch_size=64, verbose=1)
with open('cnn_train_proba.pkl', 'wb') as f:
    pickle.dump((cnn_pred, idx), f)
#=====================================================================
with open(os.path.join(path, 'rnn_model.json'), "r") as json_file:
     rnn = model_from_json(json_file.read())
rnn.load_weights(os.path.join(path, 'rnn.h5'))

rnn_pred = rnn.predict(x_test, batch_size=64, verbose=1)
with open('rnn_train_proba.pkl', 'wb') as f:
    pickle.dump((rnn_pred, idx), f)

import utils
import os, sys, pickle
import pandas as pd
import numpy as np
import editdistance
from bayes_opt import BayesianOptimization
from keras.models import model_from_json

def train_distance(threshold):

    model_name = 'best'  
    (y_pred, idx) = pd.read_pickle('{}_train_proba.pkl'.format(model_name)) 
    with open('label_map.pkl', 'rb') as lm:
        label_map = pickle.load(lm)

    # Hand cut threshold
    result = pd.DataFrame()
    result['id'] = idx
    result['pred'] = y_pred.reshape((y_pred.shape[0]*y_pred.shape[1], 48))
    result['pred'][result['pred'] < threshold] = 0

    result['pred'] = label_map.inverse_transform(result['pred'].values, 0.5)

    result = utils.combine_phone_seq(result)
    result = utils.trim(result, datadir)
    
    new_result = pd.DataFrame()
    new_result['id'] = result.index
    new_result['phone_sequence'] = result.values
    #new_result.to_csv('prediction_result_for_tuning_threshold.csv', index=False)
    
    truth = pd.read_csv('truth.csv')
    compare = truth.merge(new_result, on='id', how='left', suffixes=('_truth', '_pred'))
    
    distance = []
    for i in range(result.shape[0]):
        distance.append(editdistance.eval(compare['phone_sequence_truth'].values[i], compare['phone_sequence_pred'].values[i]))
    return sum(distance)/len(distance)
    
def find_threshold():
    #with open('predict_proba.pkl', 'rb') as p:

    bo = BayesianOptimization(train_distance, {'threshold': [0.1, 1.0]})
    bo.explore({'threshold': [0.1, 1.0]})
    bo.maximize(init_points=5, n_iter=15, kappa=2)
    print(bo.res['max'])

if __name__ == "__main__":
    print('Finding best.h5 best threshold')
    #train_distance(0.5)
    find_threshold()


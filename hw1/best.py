import utils
import os, sys, pickle
import pandas as pd
import numpy as np
import argparse

def main(datadir, outfilepath, model='best'):

    import model_cnn as md 

    # Load in training data
    #y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'),
	#    header=None, names=['id', 'label'])

    clf = md.load_pretrained(model_name=model)
    # Testing
    x_test_f = utils.load_data(os.path.join(datadir, 'fbank'), flag='test')
    x_test_m = utils.load_data(os.path.join(datadir, 'mfcc'), flag='test')
    utils.get_test_sequence(x_test_f, x_test_m, save_all=False)
    y_pred, idx = md.test(clf, x_test_f, model_name=model)
   
    #with open('predict_proba.pkl', 'wb') as p:
    #    pickle.dump((y_pred, idx), p)
 
    threshold = 0.6
    with open('label_map.pkl', 'rb') as lm:
        label_map = pickle.load(lm)

    new_pred = []
    for label in y_pred:
        tmp = label_map.inverse_transform(label, threshold)
        new_pred.append(tmp)

    new_pred = np.array(new_pred)
    result = pd.DataFrame()
    result['id'] = idx
    result['pred'] = new_pred.reshape((new_pred.shape[0]*new_pred.shape[1], 1))
    #result.to_csv('{}prime.csv'.format(model), index=False)
    
    # Post-processing for submission 
    result = utils.combine_phone_seq(result)
    result = utils.trim(result, datadir)
    sub = pd.DataFrame()
    sub['id'] = result.index
    sub['phone_sequence'] = result.values
    sub.to_csv(outfilepath, index=False)  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Data Directory")   
    parser.add_argument("outpath", help="output file name / path") 
    #parser.add_argument("-t", "--train", help="Train model from scratch", action='store_true')
    #parser.add_argument("model", type=str, choices=['rnn', 'cnn', 'concat'], help="Select model: rnn or +cnn", default='rnn')
    args = parser.parse_args()

    main(args.dir, args.outpath)

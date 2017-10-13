import utils
import os, sys
import pandas as pd
import argparse


## Load best.h5, best_model.json, best_(f)label_map.pkl to produce best submission
def main(datadir, outfilepath, model='rnn', feature='fbank'):

    if model == 'rnn':
        import model_rnn as md
    elif model == 'cnn':
        import model_cnn as md 

    # Load in training data
    y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])
    # Use both kinds of features, averaging their model predictions
    if feature == 'both':          
        clf_f = md.load_pretrained(model_name='{}_f'.format(model))
        clf_m = md.load_pretrained(model_name='{}_m'.format(model))
        # Testing
        x_test_f = utils.load_data(os.path.join(datadir, 'fbank'), flag='test')
        x_test_m = utils.load_data(os.path.join(datadir, 'mfcc'), flag='test')
        result_f = md.primary_test(clf_f, x_test_f, model_name='{}_f'.format(model))
        result_m = md.primary_test(clf_m, x_test_m, model_name='{}_m'.format(model))
        y_pred = (result_f + result_m) / 2
        with open('{}_flabel_map.pkl'.format(model), 'rb') as lm:
            label_map = pickle.load(lm)
    
        pred = label_map.inverse_transform(y_pred, 0.5)
        result = pd.DataFrame()
        result['id'] = idx
        result['pred'] = pred

    # using only one of either features
    else:
        clf = md.load_pretrained(model_name=model)
        # Testing
        x_test = utils.load_data(os.path.join(datadir, '{}'.format(feature)), flag='test')
        result = md.test(clf, x_test, model_name=model)
   
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
    parser.add_argument("feature", type=str, choices=['fbank', 'mfcc', 'both'], help="Select input feature", default='fbank')
    args = parser.parse_args()

    main(args.dir, args.outpath, feature=args.feature, model='best')

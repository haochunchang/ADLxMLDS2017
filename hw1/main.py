import utils
import os, sys
import pandas as pd
import argparse

def main(datadir, outfilepath, flag='train', model='rnn', feature='fbank'):

    if model == 'rnn':
        import model_rnn as md
    elif model == 'cnn':
        import model_cnn as md 

    # Train RNN model
    if flag == 'train':
        # Load in training data
        if feature == 'both':
            train_f = utils.load_data(os.path.join(datadir, 'fbank'))
            train_m = utils.load_data(os.path.join(datadir, 'mfcc'))
            x_train = pd.concat([train_f, train_m], axis=1)
        else:
            x_train = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
        y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                                header=None, names=['id', 'label'])
        clf = md.train(x_train, y_train)

    else:
        clf = md.load_pretrained()

    # Load in testing data
    x_test = utils.load_data(os.path.join(datadir, '{}'.format(feature)), flag='test')

    # Test RNN model
    result = clf.test(model, x_test)
   
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
    parser.add_argument("-t", "--train", help="Train model from scratch", action='store_true')
    parser.add_argument("feature", type=str, choices=['fbank', 'mfcc', 'both'], help="Select input feature", default='fbank')
    parser.add_argument("model", type=str, choices=['rnn', 'cnn'], help="Select model: rnn or +cnn", default='rnn')
    args = parser.parse_args()

    if args.train:
        main(args.dir, args.outpath, flag='train', feature=args.feature, model=args.model)
    else:
        main(args.dir, args.outpath, flag='', feature=args.feature, model=args.model)

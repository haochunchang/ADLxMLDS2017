import utils
import model_rnn
import os, sys
import pandas as pd

def main(datadir, outfilepath, flag='train', feature='fbank'):
    assert(feature == 'fbank' or feature == 'mfcc')   
 
    # Train RNN model
    if flag == 'train':
        # Load in training data
        x_train = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
        y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                                header=None, names=['id', 'label'])
        print('Start Training...')
        model.train_rnn(x_train, y_train)

    else:
        model = model_rnn.load_pretrained()

    # Load in testing data
    test_f = utils.load_data(os.path.join(datadir, 'fbank'), flag='test')
    #test_m = utils.load_data(os.path.join(datadir, 'mfcc'), flag='test')

    # Test RNN model
    result = model_rnn.test(model, test_f)
   
    # Post-processing for submission 
    result = utils.combine_phone_seq(result)
    result = utils.trim(result, datadir)
    sub = pd.DataFrame()
    sub['id'] = result.index
    sub['phone_sequence'] = result.values
    sub.to_csv(outfilepath, index=False)  
    
if __name__ == "__main__":
    datadir = sys.argv[1]
    outfilepath = sys.argv[2]
    main(datadir, outfilepath, flag='ha')

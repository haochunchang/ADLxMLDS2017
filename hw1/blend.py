import pandas as pd
import numpy as np
import os, pickle, sys
import utils
def blend(x_test, path=os.path.join('.', 'models')):

    from keras.utils import plot_model
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Dropout, Input, Flatten, Reshape
    from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
    from keras.layers import Conv2D, Conv1D, MaxPooling2D, BatchNormalization, merge
    from keras.callbacks import ModelCheckpoint, EarlyStopping 
    from keras.callbacks import TensorBoard, Callback
    from sklearn.model_selection import train_test_split

    # Load model components
    with open(os.path.join(path, 'rnn_model.json'), "r") as json_file:
        rnn = model_from_json(json_file.read())
    rnn.load_weights(os.path.join(path, 'rnn.h5'))
    with open(os.path.join(path, 'cnn_model.json'), "r") as json_file:
        cnn = model_from_json(json_file.read())
    cnn.load_weights(os.path.join(path, 'cnn.h5'))
    with open(os.path.join(path, 'best_model.json'), "r") as json_file:
        best = model_from_json(json_file.read())
    best.load_weights(os.path.join(path, 'best.h5'))

    # Pad frames id to match with padded prediction
    idx = x_test['id']
    idx.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in idx])
    new_idx = []
    for person, new_df in idx.groupby(level=0):
        for sentence, fea_id in new_df.groupby(level=1):
            fea = list(fea_id)
            fea += [person+'_'+sentence+'_'+str(i) for i in range(len(fea)+1, 778)]
            new_idx += fea
    
    idx = pd.Series(new_idx)
    x_test = np.load('./test_sents.npy')
    
    rnn_pred = rnn.predict(x_test, batch_size=64, verbose=1)
    cnn_pred = cnn.predict(x_test, batch_size=64, verbose=1)
    best_pred = best.predict(x_test, batch_size=64, verbose=1)

    y_pred = rnn_pred * 0.25 + cnn_pred * 0.375 + best_pred * 0.375

    return y_pred, idx 

if __name__ == "__main__":
    datadir = sys.argv[1]
    outfilepath = sys.argv[2]
    
    # Testing
    x_test_f = utils.load_data(os.path.join(datadir, 'fbank'), flag='test')
    x_test_m = utils.load_data(os.path.join(datadir, 'mfcc'), flag='test')
    utils.get_test_sequence(x_test_f, x_test_m, save_all=False)
    y_pred, idx = blend(x_test_f)
   
    #with open('{}_predict_proba.pkl'.format(model), 'wb') as p:
    #    pickle.dump((y_pred, idx), p)
 
    with open('label_map.pkl', 'rb') as lm:
        label_map = pickle.load(lm)

    new_pred = []
    for label in y_pred:
        tmp = label_map.inverse_transform(label, 0.5)
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


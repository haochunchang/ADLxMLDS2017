import pandas as pd
import numpy as np
import os, pickle

def train(xtrain, ytrain, batch_size=256, epochs=100, model_name='rnn'):

    from keras.utils import plot_model
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Dropout, Input, Flatten
    from keras.layers import LSTM, GRU, TimeDistributed
    from keras.callbacks import ModelCheckpoint, EarlyStopping 
    from keras.callbacks import TensorBoard, Callback
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer    

    # Preprocessing
    merged = xtrain.merge(ytrain, how='left')
    new_label = []
    y_train = merged['label'].values
    
    # Save labelBinarizer
    if model_name.split('_')[-1] == 'm':
        x_train = np.load('./data/mfcc/sents.npy')
        labels = np.load('./data/mfcc/sents_labels.npy')
        with open('{}_flabel_map.pkl'.format(model_name.split('_')[0]), 'rb') as f:
            lb = pickle.load(f)
        for label in labels:
            tmp = lb.transform(label.flatten())
            new_label.append(tmp)

        y_train = np.array(new_label)
 
    else:
        x_train = np.load('./data/fbank/sents.npy')
        labels = np.load('./data/fbank/sents_labels.npy')
        lb = LabelBinarizer()
        lb.fit(merged['label'].values)

        for label in labels:
            tmp = lb.transform(label.flatten())
            new_label.append(tmp)

        y_train = np.array(new_label)
 
        with open('{}label_map.pkl'.format(model_name), 'wb') as f:
            pickle.dump(lb, f)

    print(x_train.shape, y_train.shape)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=6)
 
    # Define RNN model
    rnn = Sequential()
    rnn.add(GRU(500, input_shape=(None, x_train.shape[2]), return_sequences=True))
    #rnn.add(GRU(128, dropout=0.2))
    #rnn.add(TimeDistributed(Dense(256, activation='relu')))
    #rnn.add(TimeDistributed(Dropout(0.2)))
    #rnn.add(TimeDistributed(Dense(256, activation='relu')))
    #rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(TimeDistributed(Dense(y_train.shape[2], activation='softmax')))
    #rnn.add(TimeDistributed(Dropout(0.2)))
    #rnn.add(Flatten())
    #rnn.add(Dense(y_train.shape[1], activation='softmax'))
 
    # Compile & print model summary
    rnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    print(rnn.summary())
 
    # Save model definition  
    model_json = rnn.to_json()
    with open("models/{}_model.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    
    plot_model(rnn, to_file='{}_model.png'.format(model_name), show_shapes=True)
 
    # Checkpoints
    checkpointer = ModelCheckpoint(filepath="./models/{}.h5".format(model_name), 
                    verbose=1, save_best_only=True, monitor='val_acc', mode='max')  
    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
 
    # Train model
    rnn.fit(x_train, y_train, batch_size=batch_size,
            verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            callbacks=[earlystopping, checkpointer])
    return rnn

def load_pretrained(path=os.path.join('.', 'models'), model_name='rnn'):

    from keras.models import model_from_json

    with open(os.path.join(path, '{}_model.json'.format(model_name)), "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(os.path.join(path, '{}.h5'.format(model_name)))

    return model

def test(model, x_test, model_name=''):
    
    idx = x_test['id']
    steps = 6

    if moedel_name.split('_')[-1] == 'm':
        x_test = np.load('data/mfcc/mfcc_test_all_steps{}.npy'.format(steps))  
    else:
        x_test = np.load('data/fbank/fbank_test_all_steps{}.npy'.format(steps))  
    y_pred = model.predict(x_test, batch_size=256, verbose=1)

    with open('{}label_map.pkl'.format(model_name), 'rb') as lm:
        label_map = pickle.load(lm)
    
    pred = label_map.inverse_transform(y_pred, 0.5)
    result = pd.DataFrame()
    result['id'] = idx
    result['pred'] = pred

    return result

def primary_test(model, x_test, model_name=''):
    
    if model_name.split('_')[-1] == 'f':
        feature = 'fbank'
    else:
        feature = 'mfcc'

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
    #frames = np.array([i for i in x_test['feature'].values])
    # Pad zero
    #padding = np.zeros((steps//2, len(x_test['feature'].values[0])))
    #frames = np.append(frames, padding, axis=0)
    #frames = np.append(padding, frames, axis=0)
    #x_test = np.array([frames[i-steps//2:i+steps//2, :] for i in range(steps//2, frames.shape[0]-steps//2)])
    x_test = np.load('./data/{}/test_sents.npy'.format(feature))
    y_pred = model.predict(x_test, batch_size=256, verbose=1)
    return y_pred, idx

if __name__ == "__main__":
    pass

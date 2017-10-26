import pandas as pd
import numpy as np
import os, pickle

def train(x_train, ytrain, batch_size=64, epochs=100, model_name='rnn'):

    from keras.utils import plot_model
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Dropout, Input, Flatten, Reshape
    from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
    from keras.layers import Conv2D, Conv1D, MaxPooling2D, BatchNormalization
    from keras.callbacks import ModelCheckpoint, EarlyStopping 
    from keras.callbacks import TensorBoard, Callback
    from sklearn.model_selection import train_test_split

    # Preprocessing
    y_train = np.load('./data/sents_labels.npy')
	
    print(x_train.shape, y_train.shape)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=6)
    # Define RNN model
    rnn = Sequential()    
    rnn.add(Conv1D(128, kernel_size=7, padding='same', input_shape=(None, x_train.shape[2])))
    rnn.add(BatchNormalization())
    rnn.add(Conv1D(256, kernel_size=5, padding='same'))
    rnn.add(BatchNormalization())
    rnn.add(Conv1D(256, kernel_size=5, padding='same'))
    rnn.add(BatchNormalization())
    rnn.add(Bidirectional(GRU(128, dropout=0.2, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.2, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.2, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.2, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.2, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.4, return_sequences=True)))
    rnn.add(Bidirectional(GRU(128, dropout=0.4, return_sequences=True)))
    rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(TimeDistributed(Dense(128, activation='relu')))
    rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(TimeDistributed(Dense(y_train.shape[2], activation='softmax')))
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
    earlystopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=1, mode='max')
    print(x_train.shape) 
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
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    
    return y_pred, idx

def test_train(model, model_name=''):
    x_test1 = np.load('./data/fbank/sents.npy')
    x_test2 = np.load('./data/mfcc/sents.npy')
    x_test = np.append(x_test1, x_test2, axis=2)
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    
    return y_pred


if __name__ == "__main__":
    pass

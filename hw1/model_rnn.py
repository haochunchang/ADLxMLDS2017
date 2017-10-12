import pandas as pd
import numpy as np
import os, pickle

def train(xtrain, ytrain, batch_size=128, epochs=100):

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
    x_train = np.array([i for i in merged['feature'].values])
    y_train = merged['label'].values
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
  
    # Save labelBinarizer
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(lb, f)

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    print(x_train.shape, y_train.shape)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=6)
 
    # Define RNN model
    rnn = Sequential()
    rnn.add(LSTM(128, input_shape=(1, x_train.shape[2]), return_sequences=True))
    rnn.add(LSTM(128, return_sequences=True))
    rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(Flatten())
    rnn.add(Dense(y_train.shape[1], activation='softmax'))
 
    # Compile & print model summary
    rnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    print(rnn.summary())
 
    # Save model definition  
    model_json = rnn.to_json()
    with open("models/rnn_model.json", "w") as json_file:
        json_file.write(model_json)
    
    plot_model(rnn, to_file='rnn_model.png', show_shapes=True)
 
    # Checkpoints
    checkpointer = ModelCheckpoint(filepath="./models/rnn.h5", 
                    verbose=1, save_best_only=True, monitor='val_acc', mode='max')  
    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
 
    # Train model
    rnn.fit(x_train, y_train, batch_size=batch_size,
            verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            callbacks=[earlystopping, checkpointer])
    return rnn

def load_pretrained(path=os.path.join('.', 'models')):

    from keras.models import model_from_json

    with open(os.path.join(path, 'rnn_model.json'), "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(os.path.join(path, 'rnn.h5'))

    return model

def test(model, x_test):
    
    idx = x_test['id']
    x_test = np.array([i for i in x_test['feature'].values])
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    
    y_pred = model.predict(x_test, batch_size=128, verbose=1)

    with open('label_map.pkl', 'rb') as lm:
        label_map = pickle.load(lm)
    
    pred = label_map.inverse_transform(y_pred, 0.5)
    result = pd.DataFrame()
    result['id'] = idx
    result['pred'] = pred

    return result

if __name__ == "__main__":
    pass

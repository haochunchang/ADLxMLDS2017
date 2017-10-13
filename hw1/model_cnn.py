import pandas as pd
import numpy as np
import os, pickle

def train(xtrain, ytrain, batch_size=128, epochs=100, model_name='cnn'):

    from keras.utils import plot_model
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Dropout, Input, Reshape, Flatten
    from keras.layers import LSTM, GRU, TimeDistributed
    from keras.layers import Conv1D, MaxPooling1D, Permute
    from keras.callbacks import ModelCheckpoint, EarlyStopping 
    from keras.callbacks import TensorBoard, Callback
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer    

    # Preprocessing
    merged = xtrain.merge(ytrain, how='left')
    
    steps = 30
    frames = np.array([i for i in merged['feature'].values])
    frames = np.append(np.array([[0 for i in range(frames.shape[1])] for j in range(steps)]), frames, axis=0)
    x_train = np.array([frames[i:i+steps, :] for i in range(frames.shape[0]-steps)])
 
    y_train = merged['label'].values
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
  
    # Save labelBinarizer
    with open('{}label_map.pkl'.format(model_name), 'wb') as f:
        pickle.dump(lb, f)

    print(x_train.shape, y_train.shape)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=6)
 
    # Define RNN model
    rnn = Sequential()
    rnn.add(Conv1D(128, kernel_size=3, input_shape=(steps, x_train.shape[2])))
    rnn.add(MaxPooling1D())
    rnn.add(Conv1D(64, kernel_size=2))
    rnn.add(MaxPooling1D())
    rnn.add(Permute((1, 2)))
    rnn.add(LSTM(128, return_sequences=True))
    #rnn.add(LSTM(128, return_sequences=True))
    rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(TimeDistributed(Dropout(0.2)))
    #rnn.add(TimeDistributed(Dense(256, activation='relu')))
    rnn.add(Flatten())
    rnn.add(Dense(y_train.shape[1], activation='softmax'))
 
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
    #rnn.fit(x_train, y_train, batch_size=batch_size,
    #        verbose=1, epochs=epochs, validation_data=(x_val, y_val),
    #        callbacks=[earlystopping, checkpointer])
    return rnn

def load_pretrained(path=os.path.join('.', 'models'), model_name='cnn'):

    from keras.models import model_from_json

    with open(os.path.join(path, '{}_model.json'.format(model_name)), "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(os.path.join(path, '{}.h5'.format(model_name)))

    return model

def test(model, x_test, model_name='cnn'):
    
    idx = x_test['id']
    steps = 30
    frames = np.array([i for i in x_test['feature'].values])
    # Pad zero
    frames = np.append(frames, np.array([[0 for i in range(frames.shape[1])] for j in range(steps)]), axis=0)
    x_test = np.array([frames[i:i+steps, :] for i in range(frames.shape[0]-steps)])
 
    y_pred = model.predict(x_test, batch_size=128, verbose=1)

    with open('{}label_map.pkl'.format(model_name), 'rb') as lm:
        label_map = pickle.load(lm)
    
    pred = label_map.inverse_transform(y_pred, 0.5)
    result = pd.DataFrame()
    result['id'] = idx
    result['pred'] = pred

    return result

if __name__ == "__main__":
    pass

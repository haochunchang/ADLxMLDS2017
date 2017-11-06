# Training code of seq2seq model
import tensorflow as tf
import pandas as pd
import numpy as np
import os, sys, time
import utils
import VCG_model as VCG

def train(datadir):

    # Declare some parameters for tuning and experiment
    isAtten = False # True for attention-based
    dim_image = 4096
    dim_hidden = 512
    batch_size = 50
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    n_epochs = 500
    learning_rate = 0.001

    # get training and testing data
    x_train, y_train = utils.load_data(datadir, flag='train')
    x_test, all_test_caps = utils.load_data(datadir, flag='test')
    
    # Preprocess captions
    wordtoix, ixtoword, bias_init_vec = utils.preprocess_caps(y_train, all_test_caps, 1) 

    # Build S2VT model
    model = VCG.Video_Caption_Generator(
                dim_image = dim_image,
                n_words = len(wordtoix),
                dim_hidden = dim_hidden,
                batch_size = batch_size,
                n_lstm_steps = n_frame_step,
                n_video_lstm_step = n_video_lstm_step,
                n_caption_lstm_step = n_caption_lstm_step,
                bias_init_vector = None)

    if not isAtten:
        tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    else:
        # Attention-based not yet developed
        pass

    # Start session
    sess = tf.InteractiveSession()
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run() 
    saver = tf.train.Saver()
    
    n_sample = x_train.shape[0]
    index = np.arange(n_sample)
    # For each epoch, get data batches
    for epoch in range(0, n_epochs):

        # Suffle index before getting training batches
        np.random.shuffle(index)
        x_train = x_train[index, :, :]
        y_train = [y_train[i] for i in index]

        start_time = time.time()
        # and for each batch...
        for start, end in zip(
            range(0, n_sample, batch_size), 
            range(batch_size, n_sample, batch_size)
        ):
            
            # preprocessing features
            current_feats = x_train[start:end]
            current_video_masks = np.ones((batch_size, n_video_lstm_step))

            # preprocessing captions...
            # Filter out other symbols
            train_caps = [y_train[i] for i in range(start, end)]
            train_caps = map(lambda x: '<bos> '+x, train_caps)
            train_caps = map(lambda x: x.replace('.', ''), train_caps)
            train_caps = map(lambda x: x.replace(',', ''), train_caps)
            train_caps = map(lambda x: x.replace('"', ''), train_caps)
            train_caps = map(lambda x: x.replace('\n', ''), train_caps)
            train_caps = map(lambda x: x.replace('?', ''), train_caps)
            train_caps = map(lambda x: x.replace('!', ''), train_caps)
            train_caps = map(lambda x: x.replace('\\', ''), train_caps)
            train_caps = map(lambda x: x.replace('/', ''), train_caps)

            # truncate caption and add <eos>
            train_caps_list = list(train_caps)
            for ind, cap in enumerate(train_caps):
                words = cap.lower().split(' ')
                if len(words) >= n_caption_lstm_step:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word += words[i] + ' '
                    train_caps_list[ind] = new_word + '<eos>'
                else:
                    train_caps_list[ind] = train_caps_list[ind] + ' <eos>'

            # convert caption into word_index or <unk>
            train_caps_index = []
            for cap in train_caps_list:
                current_word_idx = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_idx.append(wordtoix[word])
                    else:
                        current_word_idx.append(wordtoix['<unk>'])
                train_caps_index.append(current_word_idx)

            # pad sequences
            train_caps_matrix = utils.pad_sequences(train_caps_index, maxlen=n_caption_lstm_step)
            train_caps_matrix = np.hstack([train_caps_matrix, np.zeros((len(train_caps_matrix), 1))]).astype(int)

            # get caption_mask where nonzero is 1
            train_caps_masks = np.zeros((train_caps_matrix.shape[0], train_caps_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, train_caps_matrix)))
            for ind, row in enumerate(train_caps_masks):
                row[:nonzeros[ind]] = 1

            # Run session!!
            probs_val = sess.run(tf_probs, feed_dict={
                            tf_video: current_feats,
                            tf_caption: train_caps_matrix
                        })

            _, loss_val = sess.run([train_op, tf_loss],
                            feed_dict={
                                tf_video: current_feats,
                                tf_video_mask : current_video_masks,
                                tf_caption: train_caps_matrix,
                                tf_caption_mask: train_caps_masks
                            })

        # Print out loss and time
        print(" Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
    
        # save model every k epochs
        if np.mod(epoch, 10) == 0:
            print("Saving the model of epoch{}...".format(epoch))
            saver.save(sess, './models', global_step=epoch)
    print("Saving the model of epoch{}...".format(epoch))
    saver.save(sess, './models', global_step=epoch)


if __name__ == "__main__":
    train(sys.argv[1])

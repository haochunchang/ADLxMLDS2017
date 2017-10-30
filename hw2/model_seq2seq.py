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
    dim_hidden = 500
    batch_size = 128
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    n_epochs = 200
    learning_rate = 0.001

    # get training and testing data
    x_train, train_caps = utils.load_data(datadir, flag='train')
    x_test, test_caps = utils.load_data(datadir, flag='test')

    # Preprocess captions
    wordtoix, ixtoword, bias_init_vec = utils.preprocess_caps(train_caps, test_caps) 

    # Build S2VT model
    model = Video_Caption_Generator(
                dim_image = dim_image,
                n_words = len(wordtoix),
                dim_hidden = dim_hidden,
                batch_size = batch_size,
                n_lstm_steps = n_frame_step,
                n_video_lstm_step = n_video_lstm_step,
                n_caption_lstm_step = n_caption_lstm_step,
                bias_init_vector = bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = VCG.build_model(isAtten)
    # Start session


    # For each epoch, get data batches

        # and for each batch...
            start_time = time.time()
            # preprocessing features
            
            # preprocessing captions
                # add <bos>, filter out other symbols
                # truncate caption and add <eos>
                # convert caption into word_index or <unk>
                # pad sequences
                # get caption_mask where nonzero is 1
            # Run session

            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
    # save model every k epochs
        if np.mod(epoch, 10) == 0:
            print("Saving the model of epoch{}...".format(epoch))
            saver.save(sess, './models', global_step=epoch)

if __name__ == "__main__":
    train(sys.argv[1])

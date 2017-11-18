import tensorflow as tf
import pandas as pd
import numpy as np
import sys, json
import utils
import VCG_model_gru as VCG
from os import listdir
from os.path import join, isfile

def test(model_path='./', datadir='./data', outfilepath='captions.csv', peer='peer.csv'):

    dim_image = 4096
    dim_hidden = 512
    batch_size = 50
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    #===================================================
    # Testing data set
    video_path = join(datadir, 'testing_data', 'feat')
    test_videos = [f for f in listdir(video_path) if isfile(join(video_path, f))]

    ixtoword = pd.Series(np.load(join(datadir, 'gru_ixtoword.npy')).tolist())
    
    bias_init_vector = np.load(join(datadir, 'gru_bias_init_vector.npy'))

    model = VCG.Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(ixtoword),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step,
                bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    print('\n=== Use model', model_path, '===\n')
    saver.restore(sess, model_path)

    generated_sentences = []
    index = []
    for idx, video in enumerate(test_videos):
        print('video =>', video)

        video_feat_path = join(datadir, 'testing_data', 'feat', video)
        video_feat = np.load(video_feat_path)[None,...]
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
        generated_words = ixtoword[generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '<pad>')
        if punctuation > 0:
            generated_words = generated_words[:punctuation] 
        
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<pad>', '')        
        generated_sentence = generated_sentence.replace('<unk>', '')       
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')

        print('generated_sentence =>', generated_sentence)
 
        generated_sentences.append(generated_sentence)
        index.append('.'.join(video.split('.')[:-1]))

    captions = pd.DataFrame()
    captions['id'] = index
    captions['captions'] = generated_sentences
    captions.to_csv(outfilepath, index=False, header=False)
    #=====================================================================================
    # Testing peer review
    video_path = join(datadir, 'peer_review', 'feat')
    test_videos = [f for f in listdir(video_path) if isfile(join(video_path, f))]
    
    generated_sentences = []
    index = []
    for idx, video in enumerate(test_videos):
        print('video =>', video)

        video_feat_path = join(datadir, 'peer_review', 'feat', video)
        video_feat = np.load(video_feat_path)[None,...]
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
        generated_words = ixtoword[generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '<pad>')
        if punctuation > 0:
            generated_words = generated_words[:punctuation] 
        
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<pad>', '')        
        generated_sentence = generated_sentence.replace('<unk>', '')       
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')

        print('generated_sentence =>', generated_sentence)
 
        generated_sentences.append(generated_sentence)
        index.append('.'.join(video.split('.')[:-1]))

    captions = pd.DataFrame()
    captions['id'] = index
    captions['captions'] = generated_sentences
    captions.to_csv(peer, index=False, header=False)
 
if __name__ == "__main__":
    test(model_path=sys.argv[2], datadir=sys.argv[1], outfilepath=sys.argv[3], peer=sys.argv[4])


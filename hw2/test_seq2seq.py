import tensorflow as tf
import pandas as pd
import numpy as np
import os, sys
import utils
import VCG_model as VCG
import json

def test(model_path='./', datadir='./data'):

    dim_image = 4096
    dim_hidden = 1000

    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80

    batch_size = 128

    test_videos = open(os.path.join(datadir, 'testing_id.txt'), 'r').read().split('\n')[:-1]
    with open(os.path.join(datadir, 'testing_label.json')) as data_file:
        test_labels = json.load(data_file)

    ixtoword = pd.Series(np.load(os.path.join(datadir, 'ixtoword.npy')).tolist())

    bias_init_vector = np.load(os.path.join(datadir, 'bias_init_vector.npy'))

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

        video_feat_path = os.path.join(datadir, 'testing_data', 'feat', video) + '.npy'
        video_feat = np.load(video_feat_path)[None,...]
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation] 
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')

        print('generated_sentence =>', generated_sentence)

        
        generated_sentences.append(generated_sentence)
        index.append(video)

    captions = pd.DataFrame()
    captions['id'] = index
    captions['captions'] = generated_sentences
    captions.to_csv('captions.csv', index=False, header=False)

if __name__ == "__main__":
    test(model_path='./models-490', datadir=sys.argv[1])


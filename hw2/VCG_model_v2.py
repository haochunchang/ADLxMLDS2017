import tensorflow as tf
import numpy as np

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step
    
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm1_back = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2_back = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([2*dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        h_state1 = tf.zeros([self.batch_size, self.lstm1.state_size[0]])
        c_state1 = tf.zeros([self.batch_size, self.lstm1.state_size[1]])
        h_state1_back = tf.zeros([self.batch_size, self.lstm1_back.state_size[0]])
        c_state1_back = tf.zeros([self.batch_size, self.lstm1_back.state_size[1]])

        h_state2 = tf.zeros([self.batch_size, self.lstm2.state_size[0]])
        c_state2 = tf.zeros([self.batch_size, self.lstm2.state_size[1]])
        h_state2_back = tf.zeros([self.batch_size, self.lstm2_back.state_size[0]])
        c_state2_back = tf.zeros([self.batch_size, self.lstm2_back.state_size[1]])

        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        probs_back = []
        loss = 0.0

        ######  Encoding Stage ############
        ## Forward pass
        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1"):
                if i > 0:
                   tf.get_variable_scope().reuse_variables()
                output1, (h_state1, c_state1) = self.lstm1(image_emb[:,i,:], (h_state1, c_state1))

            with tf.variable_scope("LSTM3"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                output2, (h_state2, c_state2) = self.lstm2(tf.concat([padding, output1], 1), (h_state2, c_state2))
        
        ## Backward pass
        for i in range(self.n_video_lstm_step-1, -1, -1):
            with tf.variable_scope("LSTM2"):
                if i < self.n_video_lstm_step-1:
                   tf.get_variable_scope().reuse_variables()
                output1_back, (h_state1_back, c_state1_back) = self.lstm1_back(image_emb[:,i,:], (h_state1_back, c_state1_back))

            with tf.variable_scope("LSTM4"):
                if i < self.n_video_lstm_step-1:
                    tf.get_variable_scope().reuse_variables()
                output2_back, (h_state2_back, c_state2_back) = self.lstm2_back(tf.concat([padding, output1_back], 1), 
                                                                                    (h_state2_back, c_state2_back))

        ####### Decoding Stage #############
        for i in range(0, self.n_caption_lstm_step): 
            ## Phase 2 => only generate captions
            if i == 0:
                current_embed = tf.zeros([self.batch_size, self.dim_hidden])
                embed_back = tf.zeros([self.batch_size, self.dim_hidden])
            else:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                    embed_back = tf.nn.embedding_lookup(self.Wemb, caption[:, -1-i])

            with tf.variable_scope("LSTM1"):
                tf.get_variable_scope().reuse_variables()
                output1, (h_state1, c_state1) = self.lstm1(padding, (h_state1, c_state1))
           
            with tf.variable_scope("LSTM2"):
                tf.get_variable_scope().reuse_variables()
                output1_back, (h_state1_back, c_state1_back) = self.lstm1_back(padding, (h_state1_back, c_state1_back))

            with tf.variable_scope("LSTM3"):
                tf.get_variable_scope().reuse_variables()
                output2, (h_state2, c_state2) = self.lstm2(tf.concat([current_embed, output1], 1), (h_state2, c_state2))
            
            with tf.variable_scope("LSTM4"):
                tf.get_variable_scope().reuse_variables()
                output2_back, (h_state2_back, c_state2_back) = self.lstm2_back(tf.concat([embed_back, output1_back], 1), 
                                                                                    (h_state2_back, c_state2_back))
            probs.append(output2)
            probs_back.append(output2_back)
            
        for i in range(len(probs)):
            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            new_output = tf.concat([probs[i], probs_back[-1-i]], axis=1) # b x 2h x 1
            logit_words = tf.nn.xw_plus_b(new_output, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss += current_loss

        return loss, video, video_mask, caption, caption_mask, probs

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        h_state1 = tf.zeros([1, self.lstm1.state_size[0]])
        c_state1 = tf.zeros([1, self.lstm1.state_size[1]])
        h_state1_back = tf.zeros([1, self.lstm1_back.state_size[0]])
        c_state1_back = tf.zeros([1, self.lstm1_back.state_size[1]])

        h_state2 = tf.zeros([1, self.lstm2.state_size[0]])
        c_state2 = tf.zeros([1, self.lstm2.state_size[1]])
        h_state2_back = tf.zeros([1, self.lstm2_back.state_size[0]])
        c_state2_back = tf.zeros([1, self.lstm2_back.state_size[1]])

        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        probs_back = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, (h_state1, c_state1) = self.lstm1(image_emb[:,i,:], (h_state1, c_state1))

            with tf.variable_scope("LSTM3"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output2, (h_state2, c_state2) = self.lstm2(tf.concat([padding, output1], 1), (h_state2, c_state2))

        ## Backward pass
        for i in range(self.n_video_lstm_step-1, -1, -1):
            with tf.variable_scope("LSTM2"):
                if i < self.n_video_lstm_step-1:
                   tf.get_variable_scope().reuse_variables()
                output1_back, (h_state1_back, c_state1_back) = self.lstm1_back(image_emb[:,i,:], (h_state1_back, c_state1_back))

            with tf.variable_scope("LSTM4"):
                if i < self.n_video_lstm_step-1:
                    tf.get_variable_scope().reuse_variables()
                output2_back, (h_state2_back, c_state2_back) = self.lstm2_back(tf.concat([padding, output1_back], 1), 
                                                                                    (h_state2_back, c_state2_back))

        for i in range(0, self.n_caption_lstm_step):

            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
                    embed_back = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                tf.get_variable_scope().reuse_variables()
                output1, (h_state1, c_state1) = self.lstm1(padding, (h_state1, c_state1))
            with tf.variable_scope("LSTM2"):
                tf.get_variable_scope().reuse_variables()
                output1_back, (h_state1_back, c_state1_back) = self.lstm1_back(padding, (h_state1_back, c_state1_back))

            with tf.variable_scope("LSTM3"):
                tf.get_variable_scope().reuse_variables()
                output2, (h_state2, c_state2) = self.lstm2(tf.concat([current_embed, output1], 1), (h_state2, c_state2))
            with tf.variable_scope("LSTM4"):
                tf.get_variable_scope().reuse_variables()
                output2_back, (h_state2_back, c_state2_back) = self.lstm2_back(tf.concat([embed_back, output1_back], 1), 
                                                                                    (h_state2_back, c_state2_back))

            probs.append(output2)
            probs_back.append(output2_back)

        for i in range(len(probs)):
            new_output = tf.concat([probs[i], probs_back[-1-i]], axis=1)
            logit_words = tf.nn.xw_plus_b(new_output, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


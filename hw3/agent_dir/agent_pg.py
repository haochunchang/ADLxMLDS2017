from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import pickle, os

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.env = env

        # Define Agent Model...
        # Hyper-parameters
        self.lr = args.lr # learning rate
        self.bz = args.bz # batch size
        self.episodes = args.eps # total episodes(epochs)
        self.gamma = args.gamma
        self.freq = args.freq

        self.action_size = 3#env.get_action_space().n
        self.hidden_dim = 256

        self.model = tf.Graph()
        with self.model.as_default():
            # Network Architecture
            self.state_in = tf.placeholder(shape=[None, 105, 80, 1], dtype=tf.float32, name='state_in')
            
            # Standardization
            #red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.state_in)
            #self.state_in = tf.concat(axis=3, values=[ blue - tf.reduce_mean(blue, axis=(1,2)),
                                                        #green - tf.reduce_mean(green, axis=(1,2)),
                                                        #red - tf.reduce_mean(red, axis=(1,2))])
            init = tf.contrib.layers.xavier_initializer()
            self.conv = tf.layers.conv2d(self.state_in, 8, kernel_size=2, padding='same', kernel_initializer=init, activation=tf.nn.relu) 
            self.conv = tf.layers.conv2d(self.conv, 16, kernel_size=2, padding='same', kernel_initializer=init, activation=tf.nn.relu)
            #self.maxpool = tf.layers.max_pooling2d(self.conv, 2, strides=2, padding='valid')
            #print(self.maxpool.get_shape())
            
            self.hidden = tf.contrib.layers.flatten(self.state_in)
            self.hidden = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                            activation=tf.nn.relu)
            self.hidden2 = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                            activation=tf.nn.relu) 
            self.output = tf.layers.dense(self.hidden2, self.action_size, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.softmax)
            #self.chosen_action = tf.argmax(self.output, 1)

            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name='action')

            # Get action indexes
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
            #self.loss = tf.losses.log_loss(
            #                    labels = self.action_holder,
            #                    predictions = self.responsible_outputs,
            #                    weights = self.reward_holder)
            self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward_holder)
        
            tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        
            if args.test_pg: 
                model_path = os.path.join('models', 'pg-30')
            
                print('loading trained model from {}'.format(model_path))
                self.sess = tf.InteractiveSession()
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_path)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0 # Pong-specific
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r
    
    def prepro(self, s):
        s = s[35:195]
        s = s[::2,::2,0]
        s[s==144] = 0
        s[s==109] = 0
        s[s!=0] = 1
        s = s.reshape((s.shape[0], s.shape[1], 1))
        return s

    def train(self):
        """
        Implement your training algorithm here
        """
        # Launch session
        with tf.Session(graph=self.model) as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            i = 0
            total_reward = []
            total_length = []
            self.total_r_per_eps = [] # for plotting learning curve

            gradBuffer = sess.run(tf.trainable_variables())
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

            while i < self.episodes:
                s = (self.env).reset()
                s = self.prepro(s)
                running_reward = 0
                episode_his = []
                done = False
                while not done:
                    action_dist = sess.run(self.output, 
                                            feed_dict={self.state_in: [s]}) 
                    action = np.random.choice(np.arange(1, 4), p=action_dist[0])
                    #print(action_dist[0], action)
                    #action = np.argmax(action_dist == action)
                    s1, r, done, _ = (self.env).step(action) # Get reward for taking action
                    s1 = self.prepro(s1)
                 
                    episode_his.append([s, action, r])
                    s = s1 - s
                    running_reward += r
                    if done: # Update policy network
                        episode_his = np.array(episode_his)
                        episode_his[:,2] = self.discount_rewards(episode_his[:,2])
                        
                        feed_dict={self.reward_holder: episode_his[:,2],
                                    self.action_holder: episode_his[:,1],
                                    self.state_in: np.array([i for i in episode_his[:,0]])}
                        grads = sess.run(self.gradients, feed_dict=feed_dict)
                        for idx, grad in enumerate(grads):
                            gradBuffer[idx] += grad
                        
                        if i % self.freq == 0 and i != 0: 
                            feed_dict = dictionary = dict(zip(self.gradient_holders, gradBuffer))
                            _ = sess.run(self.update_batch, feed_dict=feed_dict)
                            for ix, grad in enumerate(gradBuffer):
                                gradBuffer[ix] = grad * 0

                        total_length.append(episode_his[:,2].shape[0])
                        total_reward.append(running_reward)
                        break
                # Update running tally of rewards
                if i % 1 == 0:
                    print(np.sum(total_reward))
                    #print(total_length)
                    self.total_r_per_eps.append(total_reward)
                
                # save model every k epochs
                epoch = i
                if np.mod(epoch, 100) == 0:
                    print("Saving the model of epoch{}...".format(epoch))
                    saver.save(sess, './models', global_step=epoch)
                    with open('total_reward.pkl', 'wb') as p:
                        pickle.dump(self.total_r_per_eps, p)
                i += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        action_dist = self.sess.run(self.output, 
                                feed_dict={self.state_in: [observation]})
        print(action_dist[0])
        action = np.argmax(action_dist[0])
        
        return action#self.env.get_random_action()



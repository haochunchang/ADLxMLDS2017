from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import pickle, os
import random

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.env = env

        # Define Agent Model...
        # Hyper-parameters
        self.lr = args.lr # learning rate
        self.bz = args.bz # batch size
        self.episodes = args.eps # total episodes(epochs)
        self.gamma = args.gamma
        self.freq = args.freq

        # Exploration
        self.explore_rate = 1.0
        self.explore_min = 0.01
        self.explore_decay = 0.995

        self.action_size = env.get_action_space().n
        self.hidden_dim = 256
        self.memory = []
        self.learned_eps = 0

        self.model = tf.Graph()
        with self.model.as_default():
            # Network Architecture
            self.state_in = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state_in')
            self.reward_holder = tf.placeholder(shape=[None,4], dtype=tf.float32, name='reward')
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name='action')

            init = tf.contrib.layers.xavier_initializer()

            self.conv = tf.layers.max_pooling2d(self.state_in, 2, strides=2) 
            self.hidden = tf.contrib.layers.flatten(self.conv)
            
            self.hidden = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=init, 
                                            activation=tf.nn.relu)
            self.hidden = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=init, 
                                            activation=tf.nn.relu)
            self.output = tf.layers.dense(self.hidden, self.action_size, kernel_initializer=init,
                                            activation=None)
            
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
 
            pred_Q = tf.reduce_sum(self.responsible_outputs*self.reward_holder)
            self.loss = tf.reduce_mean(tf.square(self.reward_holder - pred_Q))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.optim = optimizer.minimize(self.loss)
        
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

    def act(self, s, sess):

        action_dist = sess.run(self.output, feed_dict={self.state_in: [s]}) 
        if np.random.rand() <= self.explore_rate:
            return random.randrange(self.action_size)
        return np.argmax(action_dist[0])
    
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

            while self.learned_eps < self.episodes:
                s = (self.env).reset()
                done = False
                running_reward = 0
                while not done:
                    action = self.act(s, sess)
                    s1, r, done, _ = (self.env).step(action) # Get reward for taking action
                 
                    self.memory.append([s, action, r, s1, done])
                    s = s1 - s
                    running_reward += r
                    if done:
                        total_length.append(len(self.memory))
                        total_reward.append(running_reward)
                        break
   
                    if len(self.memory) > self.bz: 
                        # sample mini-batch from memory
                        minibatch = random.sample(self.memory, self.bz)
                        for s, a, r, s1, done in minibatch:
                            s1 = s1.reshape((1, s1.shape[0], s1.shape[1], s1.shape[2]))
                            predict = sess.run(self.output, feed_dict={self.state_in: s1})
                            target = r
                            if not done:
                                target = r + self.gamma * np.amax(predict[0])
                            
                            target_f = predict
                            target_f[0][a] = target
                            _ = sess.run(self.optim, feed_dict={self.state_in: [s], 
                                                                self.reward_holder: target_f,
                                                                self.action_holder: [a]})
                            if self.explore_rate > self.explore_min:
                                self.explore_rate *= self.explore_decay

                # Update running tally of rewards
                if self.learned_eps % 10 == 0:
                    print(np.sum(total_reward[-10:]))
                    self.total_r_per_eps.append(total_reward)
                
                # save model every k epochs
                epoch = self.learned_eps
                if np.mod(epoch, 100) == 0:
                    print("Saving the model of epoch{}...".format(epoch))
                    saver.save(sess, './dqn_models', global_step=epoch)
                    with open('dqn_total_reward.pkl', 'wb') as p:
                        pickle.dump(self.total_r_per_eps, p)
                
                self.learned_eps += 1

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



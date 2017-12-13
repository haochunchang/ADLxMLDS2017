from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import pickle, os, random

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
        self.hidden_dim = 200

        self.model = tf.Graph()
        with self.model.as_default():
            # Network Architecture
            self.state_in = tf.placeholder(shape=[None, 80, 80, 1], dtype=tf.float32, name='state_in')
            
            #init1 = tf.truncated_normal_initializer(0, stddev=1./np.sqrt(80*80), dtype=tf.float32)
            #init2 = tf.truncated_normal_initializer(0, stddev=1./np.sqrt(self.hidden_dim), dtype=tf.float32)
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            
            self.conv = tf.layers.conv2d(self.state_in, 32, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=init,
                                        activation=tf.nn.relu)
            self.conv = tf.layers.conv2d(self.conv, 64, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=init,
                                        activation=tf.nn.relu)
            self.conv = tf.layers.conv2d(self.conv, 64, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=init, 
                                        activation=tf.nn.relu)
            #self.conv = tf.layers.max_pooling2d(self.conv, 2, strides=2)
            #print(self.conv.get_shape())

            self.hidden = tf.contrib.layers.flatten(self.conv)
            self.hidden = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=init, 
                                            activation=tf.nn.relu)
            self.hidden = tf.layers.dense(self.hidden, self.hidden_dim, kernel_initializer=init, 
                                            activation=tf.nn.relu)
            self.output = tf.layers.dense(self.hidden, self.action_size, kernel_initializer=init,
                                            activation=tf.nn.softmax)
            
            #self.chosen_action = tf.argmax(self.output, 1)

            self.reward_holder = tf.placeholder(shape=[None,1], dtype=tf.float32, name='reward')
            self.action_holder = tf.placeholder(shape=[None,self.action_size], dtype=tf.float32, name='action')
            
            self.loss = tf.nn.l2_loss(self.action_holder - self.output)
            #tf.nn.softmax_cross_entropy_with_logits(
            #                            labels=self.action_onehot, logits=self.output, name="cross_entropy")
            #self.loss = -tf.reduce_sum(tf.multiply(self.reward_holder, self.cross_entropy, name="rewards"))
            #self.loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.action_dist, 1e-10, 1.0))*self.reward_holder)
            
            tvars = tf.trainable_variables()
            #self.gradient_holders = []
            #for idx, var in enumerate(tvars):
            #    placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            #    self.gradient_holders.append(placeholder)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99) 
            self.gradients = optimizer.compute_gradients(self.loss, tvars, grad_loss=self.reward_holder)
            self.optim = optimizer.apply_gradients(self.gradients)
            #self.optim = optimizer.minimize(self.loss)
            
            if args.test_pg: 
                self.model_path = os.path.join('./models/pg_models-1200')
                self.saver = tf.train.Saver()

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        np.random.seed(1) 
        self.s_prev = np.zeros((80, 80, 1))
        print('loading trained model from {}'.format(self.model_path))
        self.sess = tf.InteractiveSession(graph=self.model)
        self.saver.restore(self.sess, self.model_path)


    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0 # Pong-specific
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        
        #print("Mean reward before normalized: {}".format(np.mean(discounted_r)))
        mu = np.mean(discounted_r)
        var = np.var(discounted_r)
        discounted_r -= mu 
        discounted_r /= np.sqrt(var+1e-6)
        return discounted_r
    
    def prepro(self, s):
        s = s[35:195]
        s = s[::2,::2, 0]
        #s[:,:,1] = s[:,:,1] / 255.0
        #s[:,:,2] = s[:,:,2] / 255.0
        s[s==144] = 0
        s[s==109] = 0
        s[s!=0] = 1
        #s = s / 255
        s = s.reshape((s.shape[0], s.shape[1], 1))
        return s


    def train(self):
        
        config = tf.ConfigProto(
                    device_count = {'GPU': 1}
                )
  
        # Launch session
        with tf.Session(graph=self.model, config=config) as sess:
            saver = tf.train.Saver()
            #init = tf.global_variables_initializer()
            #sess.run(init)
            model_path = os.path.join('pg_models-5000')
            
            print('loading trained model from {}'.format(model_path))
            saver.restore(sess, model_path)
            i = 0
            total_reward = []
            total_length = []
            self.total_r_per_eps = [] # for plotting learning curve

            #gradBuffer = sess.run(tf.trainable_variables())
            #for ix, grad in enumerate(gradBuffer):
            #    gradBuffer[ix] = grad * 0
            s_prev = np.zeros((80,80,1))
            while i < self.episodes:
                s = (self.env).reset()
                running_reward = 0
                episode_his = []
                done = False
                while not done:
                    
                    s_cur = self.prepro(s)
                    x = s_cur - s_prev 
                    s_prev = s_cur
                    
                    action_dist = sess.run(self.output, feed_dict={self.state_in: [x]}) 
                    action = np.random.choice(self.action_size, p=action_dist[0])
                    
                    s, r, done, _ = (self.env).step(action+1) # Get reward for taking action
                
                    label = np.zeros_like(action_dist[0])
                    label[action] = 1
                    episode_his.append([x, label, r])
                    running_reward += r
                    
                    if done: # Update policy network
                        episode_his = np.array(episode_his)
                        episode_his[:,2] = self.discount_rewards(episode_his[:,2])
                        feed_dict={self.reward_holder: np.vstack(episode_his[:,2]),
                                    self.action_holder: np.vstack(episode_his[:,1]),
                                    self.state_in: np.array([i for i in episode_his[:,0]])}
                        
                        #grads = sess.run(self.gradients, feed_dict=feed_dict)
                        #for idx, grad in enumerate(grads):
                        #    gradBuffer[idx] += grad
                        
                        if i % self.freq == 0 and i != 0:
                        #    feed_dict = dictionary = dict(zip(self.gradient_holders, gradBuffer))
                            _ = sess.run(self.optim, feed_dict=feed_dict)
                        #    for ix, grad in enumerate(gradBuffer):
                        #        gradBuffer[ix] = grad * 0
                        print("Episode: {}, Total reward: {}".format(i, running_reward))
                        total_length.append(episode_his[:,2].shape[0])
                        total_reward.append(running_reward)
                        break
                # Update running tally of rewards
                if i % 30 == 0:
                    print("Average reward of last 30 episodes: {}\n".format(np.mean(total_reward[-30:])))
                    print("Average # of actions of last 30 episode: {}\n".format(np.mean(total_length[-30:])))
                    self.total_r_per_eps.append(total_reward)
                
                # save model every k epochs
                epoch = i
                if np.mod(epoch, 100) == 0:
                    print("Saving the model of epoch{}...\n".format(epoch))
                    saver.save(sess, './pg_models', global_step=epoch)
                    with open('pg_total_reward.pkl', 'wb') as p:
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
        self.s_cur = self.prepro(observation)
        s = self.s_cur - self.s_prev
        self.s_prev = self.s_cur

        action_dist = self.output.eval(feed_dict={self.state_in: [s]})
        action = np.random.choice(self.action_size, p=action_dist[0])
        #action = np.argmax(action_dist[0])
        #print(action)        
        return action+1#self.env.get_random_action()



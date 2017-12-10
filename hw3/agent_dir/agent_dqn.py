from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import pickle, os
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class Agent_DQN(Agent):
    def __init__(self, env, args):

        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.ENV_NAME = 'Breakout-v0'

        # Define Agent Model...
        # Hyper-parameters
        self.lr = args.lr # learning rate
        self.bz = args.bz # batch size
        self.episodes = args.eps # total episodes(epochs)
        self.gamma = args.gamma
        self.freq = 1000 # Update Frequency of Target Q network
        self.init_replay = 10000

        # Exploration
        self.explore_initial = 1.0
        self.explore_step = 1000000
        self.explore_final = 0.05
        self.epsilon = self.explore_initial
        self.epsilon_step = (self.explore_initial - self.explore_final) / self.explore_step

        self.action_size = env.get_action_space().n
        self.num_actions = env.get_action_space().n
        self.hidden_dim = 512
        self.t = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights, max_to_keep=1)
        
        ###### Summary #######
        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('./summary/', self.sess.graph)
        ######################
        if not os.path.exists('./models'):
            os.makedirs('./models')
        
        self.sess.run(tf.global_variables_initializer())

        # Load network
        if args.test_dqn:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)
        self.running_reward = []

    def build_network(self):
        
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))

        s = tf.placeholder(tf.float32, [None, 84, 84, 4])
        q_values = model(s)

        return s, q_values, model
 
    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.action_size, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        #loss = tf.reduce_mean(tf.square(error))

        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_action(self, state):
        
        if self.epsilon >= random.random() or self.t < self.init_replay:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        if self.epsilon > self.explore_final and self.t >= self.init_replay:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = observation

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > 10000:
            self.replay_memory.popleft()

        if self.t >= self.init_replay:
            # Train network
            if self.t % 4 == 0:
                self.train_network()

            # Update target network
            if self.t % self.freq == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % 30000 == 0:
                save_path = self.saver.save(self.sess, './models')#, global_step=self.t, max_to_keep=1)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))
        self.duration += 1

        if terminal:
            self.running_reward.append(self.total_reward)
            # Write summary
            if self.t >= self.init_replay:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                self.duration, self.total_loss / (float(self.duration) / float(100))]
                if self.episode % 100 == 0:
                    print("Episode: {}, Total reward: {}\n".format(self.episode, np.mean(self.running_reward[-100:])))
                    print("Total duration: {}".format(stats[2]))
                    with open("dqn_rewards.pkl", 'wb') as p:
                        pickle.dump(self.running_reward, p)
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.bz)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})
        y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
                self.s: np.float32(np.array(state_batch)),
                self.a: action_batch,
                self.y: y_batch
                })


    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(self.ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(self.ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(self.ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(self.ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        #checkpoint = tf.train.get_checkpoint_state("./models-")
        #if checkpoint and checkpoint.model_checkpoint_path:
        self.saver.restore(self.sess, './models')
        print('Successfully loaded')
        #else:
            #print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        self.t += 1

        return action
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        config = tf.ConfigProto(
                    device_count = {'GPU': 1}
                )
        
        for _ in range(self.episodes):
            terminal = False
            state = self.env.reset()
            
            while not terminal:
                action = self.get_action(state)
                observation, reward, terminal, _ = self.env.step(action)
                state = self.run(state, action, reward, terminal, observation)

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
        action = self.get_action_at_test(observation)
        
        return action#self.env.get_random_action()



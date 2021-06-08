import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import time
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))
    
    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, epsilon, copy_step, show = False):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        if show:
            env.render()
    return rewards

def main():
    print("Opening gym_environement :", 'Pacman-v6')
    env = PacmanEnv()
    print("Successfully open the environement :", 'Pacman-v6')
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    print(env.observation_space.sample().shape)
    print("num_states :", num_states)
    num_actions = env.action_space.n
    print("num_actions :", num_actions)
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    print("Creation of DQN : TrainNet & TargetNet")
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    print("Successfully created DQN : TrainNet")
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    print("Successfully created DQN : TargetNet")
    N = 10000
    N_avg = 100
    N_info = 500
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    start_time = time.time()
    total_times = np.empty(N_info)
    print("Training begin")
    for n in range(N):
         
        iter_time = time.time()
        epsilon = max(min_epsilon, epsilon * decay)

        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)

        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - N_avg):(n + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward('+str(N_avg)+')', avg_rewards, step=n)
        total_times[n%N_info] = time.time() - iter_time

        if (n+1) % N_info == 0:
            print("episode: {0}/{1}, episode reward: {2:.1f}, eps: {3:.4f}, avg reward (last {4}): {5:.4f}, avg time: {6:.4f}, time: {7:.4f}".format(
                  n+1, N, total_reward, epsilon, N_avg, avg_rewards, total_times.mean(), time.time()-start_time))
    play_game(env, TrainNet, TargetNet, epsilon, copy_step, show=True)
    env.close()
    print("Successfully trained")
    return total_rewards

total_rewards = main()

reward = [np.mean(total_rewards[-400 + i:i]) for i in range(len(total_rewards))]
print("Ploting reward...")
plt.plot(reward)
plt.show()


print("Successfully reached the end !")
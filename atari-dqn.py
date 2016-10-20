# -*- coding: utf-8 -*-

import gym
import numpy as np
import itertools as it
from time import time, sleep
import pickle
from random import sample, randint, random
import theano
import theano.tensor as T
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.init import HeUniform, Constant
from lasagne.nonlinearities import tanh, rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from tqdm import trange
import skimage.color, skimage.transform
from matplotlib import pyplot as plt
from skimage import data

class ReplayMemory:
    def __init__(self, capacity, resolution, channels):
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Agent(object):
    """
    Reinforcement Learning Agent

    This agent can learn to solve reinforcement learning tasks from
    OpenAI Gym by applying the policy gradient method.
    """

    def __init__(self, env, colors=True, scale=1, discount_factor=0.99, learning_rate=0.00025, \
                 replay_memory_size=100000, batch_size=64, cropping=(0, 0, 0, 0)):

        # Create the input variables
        s1 = T.tensor4("States")
        a = T.vector("Actions", dtype="int32")
        q2 = T.vector("Next State's best Q-Value")
        r = T.vector("Rewards")
        isterminal = T.vector("IsTerminal", dtype="int8")

        # Set field values
        if colors:
            self.channels = 3
        else:
            self.channels = 1
        self.resolution = ((env.observation_space.shape[0] - cropping[0] - cropping[1]) * scale, \
                           (env.observation_space.shape[1] - cropping[2] - cropping[3]) * scale)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.actions = env.action_space
        self.scale = scale
        self.cropping = cropping

        print("Resolution = " + str(self.resolution))
        print("Channels = " + str(self.channels))

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=replay_memory_size, resolution=self.resolution, channels=self.channels)

        # policy network
        l_in = InputLayer(shape=(None, self.channels, self.resolution[0], self.resolution[1]), input_var=s1)
        l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=[8, 8], nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=4)
        l_conv2 = Conv2DLayer(l_conv1, num_filters=64, filter_size=[4, 4], nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=2)
        l_conv3 = Conv2DLayer(l_conv2, num_filters=64, filter_size=[3, 3], nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=1)
        l_hid1 = DenseLayer(l_conv3, num_units=512, nonlinearity=rectify, W=HeUniform("relu"), b=Constant(.1))
        self.dqn = DenseLayer(l_hid1, num_units=self.actions.n, nonlinearity=None)

        # Define the loss function
        q = get_output(self.dqn)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q = T.set_subtensor(q[T.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
        loss = squared_error(q, target_q).mean()

        # Update the parameters according to the computed gradient using RMSProp.
        params = get_all_params(self.dqn, trainable=True)
        updates = rmsprop(loss, params, learning_rate)

        # Compile the theano functions
        print "Compiling the network ..."
        self.fn_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
        self.fn_get_q_values = theano.function([s1], q, name="eval_fn")
        self.fn_get_best_action = theano.function([s1], T.argmax(q), name="test_fn")
        print "Network compiled."

    def learn_from_transition(self, s1, a, s2, s2_isterminal, r):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, s2_isterminal, r)

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            q2 = np.max(self.fn_get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.fn_learn(s1, q2, a, r, isterminal)

    def perform_learning_step(self, epoch, epochs, s1):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps
        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, self.actions.n - 1)
        else:
            # Choose the best action according to the network.
            a = self.fn_get_best_action(s1)
        (s2, reward, isterminal, _) = env.step(a)  # TODO: Check a
        s2 = self.preprocess(s2)
        s3 = s2 if not isterminal else None
        if isterminal:
            x = 2
        self.learn_from_transition(s1, a, s3, isterminal, reward)

        return s2, reward, isterminal

    def preprocess(self, img):

        #plt.imshow(img)

        # Crop
        img = img[self.cropping[0]:len(img)-self.cropping[1], self.cropping[2]:len(img[0])-self.cropping[3], 0:]

        #plt.imshow(img)

        # Scaling
        if self.scale != 1:
            img = skimage.transform.rescale(img, self.scale)

        # Grayscale
        if self.channels == 1:
            #plt.imshow(img)
            img = skimage.color.rgb2gray(img)
            #plt.imshow(img, cmap=plt.cm.gray)
            img = img[np.newaxis, ...]
        else:
            img = img.reshape(self.channels, self.resolution[0], self.resolution[1])
        img = img.astype(np.float32)

        return img

    def learn(self, render_training=False, render_test=False, learning_steps_per_epoch=2000, \
              test_episodes_per_epoch=1000, epochs=20):

        print "Starting the training!"

        time_start = time()
        for epoch in range(epochs):
            print "\nEpoch %d\n-------" % (epoch + 1)
            train_episodes_finished = 0
            train_scores = []

            print "Training..."
            s1 = env.reset()
            s1 = self.preprocess(s1)
            score = 0
            for learning_step in trange(learning_steps_per_epoch):
                s2, reward, isterminal = self.perform_learning_step(epoch, epochs, s1)
                score += reward
                s1 = s2
                if (render_training):
                    env.render()
                if isterminal:
                    train_scores.append(score)
                    s1 = env.reset()
                    s1 = self.preprocess(s1)
                    train_episodes_finished += 1

            print "%d training episodes played." % train_episodes_finished

            train_scores = np.array(train_scores)

            print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

            print "\nTesting..."
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch):
                s1 = env.reset()
                s1 = self.preprocess(s1)
                score = 0
                isterminal = False
                while not isterminal:
                    a = self.fn_get_best_action(s1)
                    (s2, reward, isterminal, _) = env.step(a)  # TODO: Check a
                    s2 = self.preprocess(s2) if not isterminal else None
                    score += reward
                    s1 = s2
                    if (render_test):
                        env.render()
                test_scores.append(score)

            test_scores = np.array(test_scores)
            print "Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()

            print "Saving the network weigths..."
            pickle.dump(get_all_param_values(self.dqn), open('weights.dump', "w"))

            print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)

        env.render(close=True)
        print "======================================"
        print "Training finished. It's time to watch!"


# init environment
env = gym.make('Breakout-v0')

# init agent
agent = Agent(env, colors=False, scale=1, cropping=(30, 10, 6, 6))
#agent = Agent(env, colors=False, scale=.5, cropping=(30, 30, 20, 20))
# train agent on the environment
#agent.learn(render_training=True, render_test=True, learning_steps_per_epoch=10)
agent.learn(render_training=False, render_test=False)
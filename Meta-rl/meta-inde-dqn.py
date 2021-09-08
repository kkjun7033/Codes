import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque
import random

#%%
class DQN(Model):
    def __init__(self, nstate, naction):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(150, activation='relu')
        self.layer2 = tf.keras.layers.Dense(100, activation='relu')
        self.layer3 = tf.keras.layers.Dense(70, activation='tanh')        
        self.value = tf.keras.layers.Dense(naction)

    def call(self, state, z):
        state = tf.keras.layers.concatenate([state, z])
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        value = self.value(layer3)
        return value

class ENCODER(Model):
    def __init__(self):
        super(ENCODER, self).__init__()
        self.layer1 = tf.keras.layers.Dense(300, activation='relu')  #150
        self.layer2 = tf.keras.layers.Dense(150, activation='relu')  #100
        self.layer3 = tf.keras.layers.Dense(100, activation='relu')      #50  
        self.value = tf.keras.layers.Dense(10)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        value = self.value(layer3)
        return value

class Agent:
    def __init__(self, nstate, naction):
        # hyper parameters
        self.lr =0.001
        self.gamma = 0.99

        self.dqn_model = DQN(nstate,naction)
        self.dqn_target = DQN(nstate,naction)
        self.encoder = ENCODER()
        self.opt = optimizers.Adam(lr=self.lr, )
        self.opt2 = optimizers.Adam(lr=self.lr, )

        self.batch_size = 64
        self.state_size = nstate
        self.action_size = naction

        self.memory = deque(maxlen=2000)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value) 
        return action, q_value

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def product_of_gaussian(self, means, std_sqs):
        sigmas_squared = tf.clip_by_value(std_sqs, clip_value_min=1e-7, clip_value_max=1e+7)
        sigma_squared = 1. / tf.math.reduce_sum(tf.math.reciprocal(sigmas_squared), axis=0)
        mu = sigma_squared * tf.math.reduce_sum(means / sigmas_squared, axis=0)
        return mu, sigma_squared
    
    def reparameterize(self, mean, std):
        eps = tf.random.normal(shape=tf.shape(mean))
        eps = tf.cast(eps, dtype = tf.float32)
        mean = tf.cast(mean, dtype = tf.float32)
        std = tf.cast(std, dtype = tf.float32)
        return eps * std + mean    
    
    def infer_posterior(self, context):
        latent = self.encoder(context)
        mu = latent[:, :5]
        sigma_sq = tf.math.softplus(latent[:, 5:])
        N_m, N_s = self.product_of_gaussian(mu, sigma_sq)
        sample_z = self.reparameterize(N_m, tf.math.sqrt(N_s))
        return sample_z, N_m, N_s
    
    def update(self, replay):

        states = tf.convert_to_tensor(np.asarray(replay[0]), dtype=tf.float32)
        actions = np.asarray(replay[1])
        rewards = tf.convert_to_tensor(np.asarray(replay[2]), dtype=tf.float32)#np.asarray(replay[2])
        next_states = tf.convert_to_tensor(np.asarray(replay[3]), dtype=tf.float32)#np.asarray(replay[3])
        St = tf.convert_to_tensor(np.asarray(replay[4]), dtype=tf.float32)
        context = np.asarray(replay[5])

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            Z, _, _ = self.infer_posterior(context) #np.ones((32,17)))
            task_Z = tf.repeat(tf.expand_dims(Z,0), 32, axis=0)
            bar_Z = tf.stop_gradient(task_Z)
            
            target_q = self.dqn_target(tf.convert_to_tensor(next_states, dtype=tf.float32), bar_Z)
            next_action = tf.argmax(target_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)

            target_value = self.gamma * target_value + rewards

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32), task_Z)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

        encoder_variable = self.encoder.trainable_variables
        with tf.GradientTape() as tape2:
            tape2.watch(encoder_variable)
            
            Z, m, v = self.infer_posterior(context)
            prior = tfp.distributions.Normal(tf.zeros(tf.shape(m)), tf.ones(tf.shape(v)))
            posteriors = tfp.distributions.Normal(m, tf.math.sqrt(v))# for mu, v in zip(m, v)]
            kl_divs = tfp.distributions.kl_divergence(posteriors, prior)# for post in posteriors]
            kl_loss = tf.reduce_sum(kl_divs, -1)
    
        enc_grads = tape2.gradient(kl_loss, encoder_variable)
        self.opt2.apply_gradients(zip(enc_grads, encoder_variable))
        
        SSa=np.repeat(np.expand_dims(St,0), 32, axis=0)
        q_value = self.dqn_model(tf.convert_to_tensor(SSa, dtype=tf.float32),bar_Z)
        return q_value[0]
    
    def predict(self, replay):
        St = tf.convert_to_tensor(np.asarray(replay[0]), dtype=tf.float32)
        context = np.asarray(replay[1])
            
        Z, _, _ = self.infer_posterior(context) #np.ones((32,17)))
        task_Z = tf.repeat(tf.expand_dims(Z,0), 32, axis=0)        
        SSa=np.repeat(np.expand_dims(St,0), 32, axis=0)
        q_value = self.dqn_model(tf.convert_to_tensor(SSa, dtype=tf.float32), task_Z)
        return q_value[0]
        
        

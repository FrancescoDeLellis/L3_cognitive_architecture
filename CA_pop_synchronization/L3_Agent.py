import networkx as nx
import tensorflow as tf
import cmath
import math
import numpy as np
from sklearn.decomposition import PCA  # TODO remove?
from util import wrap_angle, compute_average_phasor, kuramoto_dynamics
from Phase_estimator_pca_online import Phase_estimator_pca_online  # TODO remove?


# ---------------------------------------------------------------- DQN BASED L3 AGENT
class L3_Agent:
    def __init__(self, n_nodes, omega_vals, coupling_strenght, model_path, dt=0.01, virtual_agent = 0):

        self.graph_nx = nx.complete_graph(n_nodes)
        self.n_nodes = n_nodes
        self.omega_vals = omega_vals
        self.dt = dt
        self.coupling_vals = nx.to_numpy_array(self.graph_nx) * coupling_strenght
        # self.model = keras.models.load_model(model_path, compile=False)
        loaded_model = tf.saved_model.load(model_path)
        self.model = loaded_model.signatures['serving_default']
        self.actions = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.action_length = len(self.actions)
        self.virtual_agent = virtual_agent
        self.wrapping_domain = '-pi to pi'

    def l3_update(self, theta_old):
        sum_coupling = 0
        for ii in range(self.n_nodes): 
            if ii != self.virtual_agent: sum_coupling += self.coupling_vals[self.virtual_agent, ii] * np.sin(theta_old[self.virtual_agent] - theta_old[ii])
        theta_new = theta_old[self.virtual_agent] + self.dt * (self.omega_vals[self.virtual_agent] - sum_coupling)
        return theta_new

    def observations_input(self, theta_obs):
        obs_pos = np.zeros(len(theta_obs))
        theta_a = theta_obs[self.virtual_agent]
        for i in range(1, len(theta_obs)):
            theta_i = theta_obs[i]
            obs_pos[i] = wrap_angle(theta_a - theta_i, self.wrapping_domain)
        average_phasor = compute_average_phasor(obs_pos[1:])
        mean_obs_pos = cmath.phase(average_phasor)
        var_obs_pos = 1 - np.abs(average_phasor)

        return mean_obs_pos, var_obs_pos, self.omega_vals[self.virtual_agent]

    def get_greedy_action(self, obs_input, action_length): # Take the greedy action for deployment
        state_tensor = tf.convert_to_tensor(obs_input)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor)
        return np.argmax(action_probs['dense_2'].numpy()[0])

    def get_epsilon_greedy_action(self, obs_input, action_length, epsilon): # Take the epsilon-greedy action for learning
        if np.random.random() > epsilon:
            return self.get_greedy_action(obs_input, action_length)
        else:
            return np.random.randint(0, action_length)

    def take_action(self, a):
        d_omega = self.actions[a]
        return d_omega

    def update_phases(self, theta):
        theta = kuramoto_dynamics(theta, self.n_nodes, self.omega_vals, self.dt, self.coupling_vals, self.wrapping_domain)
        obs_input = self.observations_input(theta)
        action = self.get_greedy_action(obs_input, self.action_length)
        omega_delta = self.take_action(action)
        self.omega_vals[self.virtual_agent] += omega_delta

        return theta

    def l3_update_phase(self, theta):
        obs_input = self.observations_input(theta)
        action = self.get_greedy_action(obs_input, self.action_length)
        omega_delta = self.take_action(action)
        self.omega_vals[self.virtual_agent] += omega_delta

        return self.l3_update(theta)
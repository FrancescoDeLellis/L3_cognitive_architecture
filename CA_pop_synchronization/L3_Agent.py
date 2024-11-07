import networkx as nx
import tensorflow as tf
import cmath
import math
import numpy as np
from sklearn.decomposition import PCA  # TODO remove?
from util import *

from CA_pop_synchronization.util import compute_average_phasor
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
        self.virtual_agent = virtual_agent

    def wrapped_difference(self, angle1, angle2):
        diff = angle1 - angle2

        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi

        return diff

    def wrapping(self, theta):
        if theta > np.pi:
            theta -= 2 * np.pi
        elif theta < -np.pi:
            theta += 2 * np.pi

        return theta

    def kuramoto_dynamics(self, theta_old, num_nodes, natural_frequency, dt, coupling):
        theta_new = np.zeros(num_nodes)
        for i in range(num_nodes):
            sum_coupling = 0
            for ii in range(num_nodes):
                sum_coupling += coupling[i, ii] * np.sin(theta_old[i] - theta_old[ii])
            theta_new[i] = theta_old[i] + dt * (natural_frequency[i] - sum_coupling)
        return [self.wrapping(theta) for theta in theta_new]
    
    def l3_update(self, theta_old):
        sum_coupling = 0
        for ii in range(self.n_nodes): 
            if ii != self.virtual_agent: sum_coupling += self.coupling_vals[self.virtual_agent, ii] * np.sin(theta_old[self.virtual_agent] - theta_old[ii])
        theta_new = theta_old[self.virtual_agent] + self.dt * (self.omega_vals[self.virtual_agent] - sum_coupling)
        return theta_new

    def observations_input(self, theta_obs, virtual_agent, omega):
        obs_pos = np.zeros(len(theta_obs))
        theta_a = theta_obs[virtual_agent]
        for i in range(1, len(theta_obs)):
            theta_i = theta_obs[i]
            obs_pos[i] = self.wrapped_difference(theta_a, theta_i)
        average_phasor = compute_average_phasor(obs_pos[1:])
        mean_obs_pos = cmath.phase(average_phasor)
        var_obs_pos = 1 - np.abs(average_phasor)

        return mean_obs_pos, var_obs_pos, omega[virtual_agent]

    def get_action_key(self, obs_input, nn_model, action_length, epsilon, is_learning):
        if is_learning:
            if np.random.random() > epsilon:
                state_tensor = tf.convert_to_tensor(obs_input)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = nn_model(state_tensor, training=False)
                return tf.argmax(action_probs[0]).numpy()
            else:
                return np.random.randint(0, action_length)
        else:
            state_tensor = tf.convert_to_tensor(obs_input)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = nn_model(state_tensor)
            return np.argmax(action_probs['dense_2'].numpy()[0])

    def take_action(self, a, actions):
        d_omega = actions[a]
        return d_omega

    def update_phases(self, theta):
        action_length = len(self.actions)
        theta = self.kuramoto_dynamics(theta, self.n_nodes, self.omega_vals, self.dt, self.coupling_vals)
        obs_input = self.observations_input(theta, self.virtual_agent, self.omega_vals)
        action = self.get_action_key(obs_input, self.model, action_length, self.virtual_agent, is_learning=False)
        omega_delta = self.take_action(action, self.actions)
        self.omega_vals[self.virtual_agent] += omega_delta

        return theta

    def l3_update_phase(self, theta):
        action_length = len(self.actions)
        obs_input = self.observations_input(theta, self.virtual_agent, self.omega_vals)
        action = self.get_action_key(obs_input, self.model, action_length, self.virtual_agent, is_learning=False)
        omega_delta = self.take_action(action, self.actions)
        self.omega_vals[self.virtual_agent] += omega_delta

        return self.l3_update(theta)
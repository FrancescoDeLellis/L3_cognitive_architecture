import numpy as np
import networkx as nx
import tensorflow as tf
import cmath
import math
import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------- DQN BASED L3 AGENT
class L3Agent:
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

    def order_parameter(self, theta, number_nodes):
        z = 1 / number_nodes * sum([cmath.exp(complex(0, theta[node])) for node in range(number_nodes)])
        return z

    def wrapped_difference(self, angle1, angle2):
        diff = angle1 - angle2

        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi

        return diff

    def wrapping(self, theta):
        if theta > 2 * np.pi:
            theta -= 2 * np.pi
        elif theta < 0:
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
        order_parameter_complex = self.order_parameter(obs_pos[1:], len(obs_pos[1:]))
        mean_obs_pos = cmath.phase(order_parameter_complex)
        var_obs_pos = 1 - np.abs(order_parameter_complex)

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

# ---------------------------------------------------------------- PHASE ESTIMATION CLASS
class Phase_estimator_pca_online:
    def __init__(self, window_pca, interval_between_pca):
        self.trajectory    = [] 
        self.time_instants = []    

        self.pca                  = PCA(n_components=1)
        self.pca_direction        = 1                     # 1 or -1; used to preserve existing direction of the pca when recomputed
        self.window_pca           = window_pca            # [s]
        self.interval_between_pca = interval_between_pca  # [s]

        self.pos_princ_comp        = 0  
        self.pos_princ_comp_prev   = 0 
        self.vel_princ_comp        = 0  
        self.vel_princ_comp_prev   = 0 
        self.acc_princ_comp        = 0    
        self.pos_princ_comp_offset = 0

        self.phase = 0                
        self.is_first_estimation   = True  
        self.time_last_pca         = None  
        self.is_first_pca_computed = False
        
        self.amplitude_vel_p = 1
        self.amplitude_vel_n = 1
        self.amplitude_pos_n = 1
        self.amplitude_pos_p = 1
        
        
    def estimate_phase(self, position, current_time):
        self.time_instants.append(current_time)
        current_time = self.time_instants[-1]
        self.trajectory.append(position)

        if self.is_first_estimation: 
            self.time_last_pca = current_time
            self.is_first_estimation = False
        
        if not self.is_first_pca_computed:
            if current_time - self.time_last_pca >= self.window_pca:
                self.time_last_pca = current_time
                self.compute_PCA()
                self.is_first_pca_computed = True
        else:
            if current_time - self.time_last_pca >= self.interval_between_pca:
                self.time_last_pca = current_time
                self.compute_PCA()
        
        if self.is_first_pca_computed:  self.update_phase()
        
        return self.phase


    def compute_PCA(self):
        idx = 1
        is_found = False
        while not is_found:
            if self.time_instants[-1] - self.time_instants[-idx] >= self.window_pca:
                is_found = True
            else:  idx += 1
        
        if self.is_first_pca_computed:  prev_pca_vec = self.pca.components_[0]

        self.pca.fit(np.array(self.trajectory)[-idx:,:])

        if self.is_first_pca_computed:
            pca_vec = self.pca.components_[0]
            if np.dot(pca_vec, prev_pca_vec) < 0:  self.pca_direction = -1 * self.pca_direction
        
        score = self.pca_direction * np.reshape(self.pca.transform(np.array(self.trajectory)[-idx:,:]),-1)  
        max_score = max(score)
        min_score = min(score)
        self.pos_princ_comp_offset = (max_score + min_score)/2

        array_1d = np.array(np.array(self.trajectory)[-4,:])        
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset

        array_1d = np.array(np.array(self.trajectory)[-5,:])        
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp_prev = score[0, 0] - self.pos_princ_comp_offset
        
        sampling_time = self.time_instants[-5] - self.time_instants[-4]
        self.vel_princ_comp_prev = (self.pos_princ_comp-self.pos_princ_comp_prev) / sampling_time

        array_1d = np.array(np.array(self.trajectory)[-2,:])        
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset

        array_1d = np.array(np.array(self.trajectory)[-3,:])        
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp_prev = score[0, 0] - self.pos_princ_comp_offset

        sampling_time = self.time_instants[-2] - self.time_instants[-3]
        self.vel_princ_comp = (self.pos_princ_comp-self.pos_princ_comp_prev) / sampling_time

    def update_phase(self):   
        array_1d = np.array(np.array(self.trajectory)[-1,:])        
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        
        self.pos_princ_comp_prev = self.pos_princ_comp
        self.vel_princ_comp_prev = self.vel_princ_comp
        
        sampling_time = self.time_instants[-1] - self.time_instants[-2]

        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset
        self.vel_princ_comp = (self.pos_princ_comp-self.pos_princ_comp_prev) / sampling_time
        self.acc_princ_comp    = (self.vel_princ_comp-self.vel_princ_comp_prev) / sampling_time

        if self.pos_princ_comp_prev < 0 and self.pos_princ_comp >= 0 and self.vel_princ_comp > 0:
            self.amplitude_vel_p = abs(self.vel_princ_comp)
        if self.pos_princ_comp_prev >= 0 and self.pos_princ_comp < 0 and self.vel_princ_comp < 0:
            self.amplitude_vel_n = abs(self.vel_princ_comp)
        if self.vel_princ_comp_prev >= 0 and self.vel_princ_comp < 0 and self.acc_princ_comp < 0:
            self.amplitude_pos_p = abs(self.pos_princ_comp)
        if self.vel_princ_comp_prev < 0 and self.vel_princ_comp >= 0 and self.acc_princ_comp > 0:
            self.amplitude_pos_n = abs(self.pos_princ_comp)
        if self.pos_princ_comp >= 0:
            position_normalized = self.pos_princ_comp / self.amplitude_pos_p
        else:
            position_normalized = self.pos_princ_comp / self.amplitude_pos_n
        if self.vel_princ_comp >= 0:
            velocity_normalized = self.vel_princ_comp / self.amplitude_vel_p
        else:
            velocity_normalized = self.vel_princ_comp / self.amplitude_vel_n

        self.phase = math.atan2(-velocity_normalized, position_normalized)
        
        self.phase = np.mod(self.phase, 2*np.pi)  # wrap to [0, 2pi)
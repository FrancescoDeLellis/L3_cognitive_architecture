import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_Agent import L3_Agent
from Phase_estimator_pca_online import Phase_estimator_pca_online
from util import create_folder_if_not_exists


class L3_Wrapper():

    def __init__(self, model_path, save_path, ID=0, amplitude=15, omega=2, n_participants=3,
                 omega_parts=np.array([0, 3.4, 4.6]), c_strenght=0.25):
        self.ID = ID  # Python CA instance ID
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega  # Movement frequency
        self.x = 0
        self.y = 0
        self.z = 0
        self.z_amp_ratio = 0.1
        self.initial_position = 0
        self.initial_phase = 0
        self.n_participants = n_participants

        self.l3_phase = []
        self.l3_agent = L3_Agent(n_participants, omega_parts, c_strenght, model_path)

        self.window_pca = 4  # duration of the time window [seconds] in which the PCA is operatedf
        self.interval_between_pca = 1  # time interval [seconds] separating consecutive computations of the PCA

        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca,
                                                                   self.interval_between_pca))  # One estimator for each participant

        self.kuramoto_phases = [np.zeros(self.n_participants)]

        self.time_history = [0]
        self.phases_history = [np.zeros(self.n_participants)]
        self.positions_history = []
        self.save_path = save_path
        create_folder_if_not_exists(save_path)

    def reset_CA(self):
        # STORE DATA
        self.save_data()
        self.plot_phases(np.stack(self.phases_history))

        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))

        self.kuramoto_phases = [np.zeros(self.n_participants)]

        self.phases_history = [np.zeros(self.n_participants)]
        self.time_history = [0]
        self.positions_history = []

    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string):
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3 * self.l3_agent.n_nodes + 1
        return flag, numbers[0:-1], numbers[-1]

    def set_intial_position(self, position):
        self.initial_position = position[:, self.l3_agent.virtual_agent]
        self.initial_phase = 0
        self.positions_history.append(position.T)

    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions, delta_t, time):
        # positions contains the neighbors 3D end effectors
        self.positions_history.append(positions.T)
        theta = np.arctan2(self.z, self.y)
        # theta = np.mod(theta, 2*np.pi)  # wrap to [0, 2pi)
        ic(theta)
        self.l3_phase.append(theta)

        ic(time)

        phases = []
        for i in range(
                self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if i != self.l3_agent.virtual_agent:
                phases.append(self.estimators_live[i].estimate_phase(positions[:, i], time))
            else:
                phases.append(theta - self.initial_phase)

        ic(phases)

        self.l3_agent.dt = delta_t
        self.time_history.append(self.time_history[-1] + delta_t)
        self.phases_history.append(np.array(phases))
        theta_next = self.l3_agent.l3_update_phase(
            np.array(phases))  # This function changes the omega of the avatar and outputs its new phase
        self.kuramoto_phases.append(self.l3_agent.update_phases(self.kuramoto_phases[
                                                                    -1]))  # comment this and next line and uncomment upper line if L3 is connected with VR agents
        # theta_next = np.mod(self.kuramoto_phases[-1][self.l3_agent.virtual_agent], 2*np.pi)

        ic(theta_next)

        self.y = self.amplitude * np.cos(theta_next)
        self.z = self.amplitude * np.sin(theta_next)

        y_k1 = self.amplitude * np.cos(self.kuramoto_phases[-1][1])
        z_k1 = self.amplitude * np.sin(self.kuramoto_phases[-1][1])

        y_k2 = self.amplitude * np.cos(self.kuramoto_phases[-1][2])
        z_k2 = self.amplitude * np.sin(self.kuramoto_phases[-1][2])

        message = 'X=' + str(self.initial_position[0]) + ' Y=' + str(self.initial_position[1] + self.y) + ' Z=' + str(
            self.initial_position[2] + self.z_amp_ratio * np.abs(self.z))  # Format data as UE Vector
        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k1) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k1))
        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k2) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k2))

        return message

    def plot_phases(self, phases):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.n_participants):
            if i == self.l3_agent.virtual_agent:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i + 1}')
            else:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i + 1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\phases_plot.png')
        plt.close()

        # PLOT ESTIMATION ERROR
        plt.figure()
        for i in range(1, self.n_participants): plt.plot(self.time_history[:-1],
                                                         np.abs(phases[1:, i] - np.array(self.kuramoto_phases)[:-1, i]),
                                                         color=colors[i], label=f'VH {i + 1}')

        plt.title('Phase Estimation error')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Absolute error (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\phases_estimation_error.png')
        plt.close()

    def save_data(self):
        np.save(f'{self.save_path}/phases_history.npy', np.stack(self.phases_history))
        np.save(f'{self.save_path}/postions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/time_history.npy', np.stack(self.time_history))

    @staticmethod
    def start_connection(address, port):
        # Create a TCP sockets
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Bind the socket to the server address and port
            server_socket.bind((address, port))
        except socket.error as e:
            print("Connection error: %s" % e)

        # Listen for incoming connections
        server_socket.listen(1)  # Limit number of connections to L3 socket
        print(f'Server listening on {address}:{port}')

        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        return connection, client_address

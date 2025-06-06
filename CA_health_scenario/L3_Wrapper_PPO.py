import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_Agent_PPO import L3, Kuramoto, Network
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from ListeningModeManager import ListeningModeManager
from Phase_estimator_pca_online import Phase_estimator_pca_online
from util import create_folder_if_not_exists, l3_update_theta, get_time_string
from wrap_functions import wrap_to_pi
from typing import Sequence # For type hinting numpy array

class L3_Wrapper():

    def __init__(self, model_path : str, save_path : str, exercise_ID: int = 0, ID : int =0, n_dims_estimand_pos : int = 3, listening_time : float =10.0,  amplitude : float =15, omega : float =2, n_participants : int =3,
                 omega_parts=np.array([0, 3.4, 4.6]), c_strength : float =1.25, omega_sat : float = 15):
        self.ID = ID  # Python CA instance ID
        self.exercise_ID = exercise_ID # Exercise ID for the trial
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega  # Movement frequency
        self.n_dims_estimand_pos = n_dims_estimand_pos  # Number of dimensions of the position estimand
        self.listening_time = listening_time  # Time to listen for the phase estimators
        self.y = 0 # TODO: this will become the output of the phase indexer. Maybe you don't need it 
        self.z = 0
        self.z_amp_ratio = 0.1
        self.initial_position = 0 # TODO: don't need this
        self.initial_phase = 0
        self.n_participants = n_participants
        self.omega_sat = omega_sat
        self.c_strength = c_strength
        self.listening_manager = ListeningModeManager()

        self.l3_agent = L3(0,omega_parts[0],0,n_actions=1,input_dims=(3,), omega_sat=omega_sat, model_path=model_path)
        self.l3_agent.load_model()
        # TODO: Instantiate a phase indexer for the agent

        self.AGENTS = []
        self.AGENTS.append(self.l3_agent)
        for i in range(1, self.n_participants):
            self.AGENTS.append(Kuramoto(i, omega_parts[i], 0))
        
        self.A = np.ones((self.n_participants, self.n_participants))
        for i in range(self.n_participants): # Create the all-to-all adjacency matrix
            self.A[i, i] = 0
        self.net = Network(A=self.A, agents=self.AGENTS, c=c_strength) # Delete 40-43 when you don't use Kuramotos

        self.estimators_live = []
        for _ in range(self.n_participants):
            # self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time))
            self.estimators_live.append(Phase_estimator_pca_online(4, 1)) # Old phase estimator

        self.time_history = [0]
        self.phases_history = [np.zeros(self.n_participants)]
        self.kuramoto_phases = [np.zeros(self.n_participants)] # Delete when you don't use Kuramotos
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
             # self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time))
            self.estimators_live.append(Phase_estimator_pca_online(4, 1)) # Old phase estimator
            
        for i in range(self.n_participants):
            self.AGENTS[i].theta = 0
            self.net.thetas = 0

        self.phases_history = [np.zeros(self.n_participants)]
        self.kuramoto_phases = [np.zeros(self.n_participants)] # Delete when you don't use Kuramotos
        self.time_history = [0]
        self.positions_history = []

     # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string : str) -> tuple[bool, Sequence[float]]:
        # TODO: Check how unreal engine sends the data and change the regex accordingly
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3 * self.n_participants + 1 # TODO: change to number_of_coordinates * n_participants + 1. The number of coordinates is 3*4=12 because we have 3D positions for 4 end effectors (left hand, right hand, left foot, right foot)
        return flag, numbers[0:-1], numbers[-1] # Use flag for debugging.

    def set_initial_position(self, position : list[list[float]]):
        # TODO: Use the phase indexer to set the initial position
        self.initial_position = position[:, 0] # 0 because it is always the index of L3
        self.initial_phase = 0
        self.positions_history.append(position.T)
    
    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions : list[list[float]], delta_t : float, time : float) -> str:
        # positions contains the neighbors 3D end effectors
        self.positions_history.append(positions.T)
        ic(self.positions_history[-1].shape)
        theta = np.arctan2(self.z, self.y) # TODO: remove this recomputation and use the L3_current_phase as an attribute that is stored
        ic(theta)

        ic(time)

        phases = [] # Vector of the real phases
        for i in range(self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if self.AGENTS[i].is_virtual == False: 
                # phases.append(self.estimators_live[i].update_estimator(positions[:, i], time))
                phases.append(self.estimators_live[i].estimate_phase(positions[:, i], time))
                phases[-1] = self.listening_manager.mask_none(phases[-1])
                phases[-1] = wrap_to_pi(phases[-1])
            else:
                phases.append(theta - self.initial_phase)

        ic(phases[-1])

        self.net.dt = delta_t # Delete this line when you don't use Kuramotos
        self.time_history.append(self.time_history[-1] + delta_t)
        self.phases_history.append(np.array(phases))

        observation = self.l3_agent.get_state(np.array(phases), self.n_participants)
        self.l3_agent.omega = self.l3_agent.choose_action_mean(observation)  # Compute the new omega for the virtual agent
        self.net.omegas[0] = self.AGENTS[0].omega = self.l3_agent.omega  # Update the vector of omegas for simulations
        self.net.step()
        theta_next_kuramoto = self.net.thetas # Compute theta of Kuramoto agents at time t
        self.kuramoto_phases.append(theta_next_kuramoto) # Delete 113-116 when you don't use Kuramotos

        ic(theta_next_kuramoto)[-1]

        # l3_theta_next = l3_update_theta(np.array(phases), self.l3_agent.omega, coupling=self.c_strength, dt=delta_t)

        self.y = self.amplitude * np.cos(self.AGENTS[0].theta) # TODO: Use phase indexer to compute the new positions
        self.z = self.amplitude * np.sin(self.AGENTS[0].theta)

        y_k1 = self.amplitude * np.cos(self.AGENTS[1].theta)
        z_k1 = self.amplitude * np.sin(self.AGENTS[1].theta)

        y_k2 = self.amplitude * np.cos(self.AGENTS[2].theta)
        z_k2 = self.amplitude * np.sin(self.AGENTS[2].theta) # Delete 123-125 when you don't use Kuramotos 

        message = 'X=' + str(self.initial_position[0]) + ' Y=' + str(self.initial_position[1] + self.y) + ' Z=' + str(
            self.initial_position[2] + self.z_amp_ratio * np.abs(self.z))  # Format data as UE Vector TODO: Change to the format that UE expects

        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k1) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k1))
        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k2) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k2)) # Delete 130 - 133 when you don't use Kuramotos

        return message
    
    def plot_phases(self, phases: list[list[float]]):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.n_participants):
            if self.AGENTS[i].is_virtual == True: 
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i + 1}')
            else:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i + 1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        time_string = get_time_string()

        plt.savefig(f'{self.save_path}\\'+time_string+'phases_plot.png')
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

        plt.savefig(f'{self.save_path}\\'+time_string+'phases_estimation_error.png') 
        plt.close() # Delete 163 - 171 when you don't use Kuramoto

    def save_data(self):
        time_string = get_time_string()
        np.save(f'{self.save_path}/'+time_string+'phases_history.npy',   np.stack(self.phases_history))
        np.save(f'{self.save_path}/'+time_string+'positions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/'+time_string+'time_history.npy',     np.stack(self.time_history))

    @staticmethod
    def start_connection(address : str, port : int) -> tuple[str, int]:
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
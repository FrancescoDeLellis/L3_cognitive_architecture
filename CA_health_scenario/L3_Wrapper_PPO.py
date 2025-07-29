import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_Agent_PPO import L3, Kuramoto, Network
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from ListeningModeManager import ListeningModeManager
from Phase_estimator_pca_online import Phase_estimator_pca_online
from PhaseIndexer import PhaseIndexer
from util import create_folder_if_not_exists, l3_update_theta, get_time_string
from wrap_functions import wrap_to_pi, wrap_to_2pi
from typing import Sequence # For type hinting numpy array
import pandas as pd

class L3_Wrapper():

    def __init__(self, model_path : str, save_path : str, exercise_ID: int = 0, ID : int =0, n_dims_estimand_pos : int = 12, listening_time : float = 10.0,  amplitude : float =15, omega : float =2, n_participants : int =3,
                 omega_parts=np.array([0, 1.0, 1.2]), c_strength : float = 1.25, omega_sat : float = 15):
        self.ID = ID  # Python CA instance ID
        self.exercise_ID = exercise_ID # Exercise ID for the trial
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega  # Movement frequency
        self.n_dims_estimand_pos = n_dims_estimand_pos  # Number of dimensions of the position estimand
        self.listening_time = listening_time  # Time to listen for the phase estimators
        self.initial_phase = 0
        self.n_participants = n_participants
        self.omega_sat = omega_sat
        self.c_strength = c_strength
        self.listening_manager = ListeningModeManager()

        self.l3_agent = L3(0,omega_parts[0],0,n_actions=1,input_dims=(5,), omega_sat=omega_sat, model_path=model_path)
        self.l3_agent.load_model()

        match exercise_ID:
            case 0: 
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex02_baseline.csv", index_col='phase')
                end_effector_inputs = 4 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.discarded_time = 0
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 30
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 0.81  # Natural frequency of the L3 agent during the listening phase
                columns_in = [
                    "RHeel.X","RHeel.Y","RHeel.Z",
                    "LHeel.X","LHeel.Y","LHeel.Z",
                    'LWrist.X', 'LWrist.Y', 'LWrist.Z',
                    'RWrist.X', 'RWrist.Y', 'RWrist.Z',
                ]
            case 1:
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                end_effector_inputs = 3 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.discarded_time = 2
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 12
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 1.29  # Natural frequency of the L3 agent during the listening phase
                columns_in = [
                    'Hip.X', 'Hip.Y', 'Hip.Z',
                    'RHip.X', 'RHip.Y', 'RHip.Z',
                    'LHip.X', 'LHip.Y', 'LHip.Z',
                ]
            case _: 
                print(f'Exercise ID {exercise_ID} not recognized, using default baseline.')
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                end_effector_inputs = 5 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
        
        self.n_dims_estimand_pos = 3 * end_effector_inputs  # Number of dimensions of the position estimand
        columns = [
            'LAnkle.X', 'LAnkle.Y', 'LAnkle.Z',
            'RAnkle.X', 'RAnkle.Y', 'RAnkle.Z',
            'LWrist.X', 'LWrist.Y', 'LWrist.Z',
            'RWrist.X', 'RWrist.Y', 'RWrist.Z',
            'Hip.X', 'Hip.Y', 'Hip.Z'
        ]
        self.indexer = PhaseIndexer(self.baseline_dataframe, columns_names=columns)  # Create the phase indexer with the baseline data
        self.indexer_in = PhaseIndexer(self.baseline_dataframe, columns_names=columns_in)  # Create the phase indexer with the baseline data for input
        
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
            self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time, discarded_time=self.discarded_time, min_duration_first_pseudoperiod=0, look_behind_pcent=self.look_behind_pcent, look_ahead_pcent=self.look_ahead_pcent, time_const_lowpass_filter_phase=self.time_const_lowpass_filter_phase, time_const_lowpass_filter_pos=self.time_const_lowpass_filter_estimand_pos, is_use_elapsed_time=self.is_use_elapsed_time))

        self.n_end_effectors = 5 # Number of end effectors available (hands, feet, hip)
        self.time_history = [0]
        self.current_phase = 0
        self.phases_history = [np.zeros(self.n_participants)]
        self.kuramoto_phases = [np.zeros(self.n_participants)]  # Delete when you don't use Kuramotos
        self.positions_history = []
        self.positions_sent = []  # Store the positions sent to UE
        self.save_path = save_path
        create_folder_if_not_exists(save_path)

    def reset_CA(self):
        # STORE DATA
        self.save_data()
        self.plot_phases(np.stack(self.phases_history))

        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time, discarded_time=1, min_duration_first_pseudoperiod=0, look_behind_pcent=self.look_behind_pcent, look_ahead_pcent=self.look_ahead_pcent, time_const_lowpass_filter_phase=self.time_const_lowpass_filter_phase, time_const_lowpass_filter_pos=self.time_const_lowpass_filter_estimand_pos, is_use_elapsed_time=self.is_use_elapsed_time))
            
        for i in range(self.n_participants):
            self.AGENTS[i].theta = 0
            self.net.thetas = 0

        self.phases_history = [np.zeros(self.n_participants)]
        self.kuramoto_phases = [np.zeros(self.n_participants)] # Delete when you don't use Kuramotos
        self.time_history = [0]
        self.positions_history = []
     
    def parse_TCP_string(self, string : str) -> tuple[bool, Sequence[float]]:
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == self.n_dims_estimand_pos * self.n_participants + 1 
        return flag, numbers[0:-1], numbers[-1]  # Use flag for debugging. The last number is the time in seconds
    

    def set_initial_position(self, position : list[list[float]]):
        # position = np.reshape(position, (self.n_participants, self.n_dims_estimand_pos)).T
        self.initial_phase = 0
    
    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions : list[list[float]], delta_t : float, time : float) -> str:
        # positions contains the neighbors 3D end effectors
        # positions = np.reshape(positions, (self.n_participants, self.n_dims_estimand_pos)).T
        positions = np.zeros((self.n_participants, self.n_dims_estimand_pos)).T
        for i in range(self.n_participants):
            positions[:, i] =  self.indexer_in.get_values_at_phase(wrap_to_2pi(self.kuramoto_phases[-1][i]))  # Get the positions of the end effectors at the current phase

        self.positions_history.append(positions)
        self.current_phase = self.l3_agent.theta  # Get the current phase of the L3 agent
    
        ic(time)

        ic(positions)
        phases = [] # Vector of the real phases
        for i in range(self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if self.AGENTS[i].is_virtual == False:
                phases.append(self.estimators_live[i].update_estimator(positions[:, i], time))  # Update the phase estimator with the new position
                phases[-1] = self.listening_manager.mask_none(phases[-1])
                phases[-1] = wrap_to_pi(phases[-1])  # Wrap the phase to [-pi, pi]
            else:
                phases.append(self.current_phase - self.initial_phase)


        self.net.dt = delta_t # Delete this line when you don't use Kuramotos
        self.time_history.append(time)  # Store the time in the history)
        self.phases_history.append(np.array(phases))
        
        # observation = self.l3_agent.get_state(np.array(phases), self.n_participants)
        phases_dots = wrap_to_2pi(self.phases_history[-1] - self.phases_history[-2])/delta_t  # Compute the phase dots
        observation = self.l3_agent.get_state_with_theta_dots(np.array(phases), phases_dots, self.n_participants)
        if time > (self.estimators_live[0].listening_time + self.estimators_live[0].discarded_time):  # If the listening time is over, use the phase estimator to compute the omega
            self.l3_agent.omega = self.l3_agent.choose_action_mean(observation)  # Compute the new omega for the virtual agent
        else:
            self.l3_agent.omega = self.omega_listening
        self.net.omegas[0] = self.AGENTS[0].omega = self.l3_agent.omega  # Update the vector of omegas for simulations
        
        self.net.step()
        theta_next_kuramoto = self.net.thetas # Compute theta of Kuramoto agents at time t
        self.kuramoto_phases.append(np.copy(theta_next_kuramoto)) # Delete 113-116 when you don't use Kuramotos.
        ic(theta_next_kuramoto)
        # l3_theta_next = l3_update_theta(np.array(phases), self.l3_agent.omega, coupling=self.c_strength, dt=delta_t)

        end_effectors_positions = np.zeros((3*self.n_end_effectors*self.n_participants))
        for i in range(self.n_participants):
            end_effectors_positions[3*i*self.n_end_effectors:3*(i+1)*self.n_end_effectors] = self.indexer.get_values_at_phase(wrap_to_2pi(theta_next_kuramoto[i])) # Uncomment when you use phase indexer
        message = self.write_message(end_effectors_positions)  # Format the message to be sent to UE
        self.positions_sent.append(end_effectors_positions.reshape((self.n_participants, 3*self.n_end_effectors)).T)  # Store the positions sent to UE

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
        for i in range(1, self.n_participants):
            angle_error = np.abs(np.angle(np.exp(1j*(phases[1:, i] - np.array(self.kuramoto_phases)[:-1, i]))))
            plt.plot(self.time_history[:-1], angle_error, color=colors[i], label=f'VH {i + 1}')
        '''
        for i in range(1, self.n_participants):
            plt.plot(self.time_history[:-1],
                        np.abs(phases[1:, i] - np.array(self.kuramoto_phases)[:-1, i]),
                        color=colors[i], label=f'VH {i + 1}')
        '''
        plt.title('Phase Estimation error')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Absolute error (radiants)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_path}\\'+time_string+'phases_estimation_error.png') 
        plt.close()

         # PLOT KURAMOTO PHASES HISTORY
        plt.figure()
        for i in range(1, self.n_participants): plt.plot(self.time_history[:-1], np.array(self.kuramoto_phases)[:-1, i],
                                                         color=colors[i], label=f'VH {i + 1}')

        plt.title('Kuramoto phases')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\\'+time_string+'phases_history.png') 
        plt.close() # Delete 163 - 171 when you don't use Kuramoto

         # PLOT POSITIONS HISTORY
        plt.figure()
        plt.plot(self.time_history[:-1], np.array(self.positions_history)[:, 5, 1],
                    color='red', label=f'VH 1')
        plt.plot(self.time_history[:-1], np.array(self.positions_sent)[:, 5, 1], color='blue', label=f'VH 1 sent')

        plt.title('Positions of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\\'+time_string+'position_history.png') 
        plt.close() # Delete 163 - 171 when you don't use Kuramoto

    def save_data(self):
        time_string = get_time_string()
        np.save(f'{self.save_path}/'+time_string+'phases_history.npy',   np.stack(self.phases_history))
        #np.save(f'{self.save_path}/'+time_string+'kuramoto_phases_history.npy',   np.stack(self.kuramoto_phases_history))
        np.save(f'{self.save_path}/'+time_string+'positions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/'+time_string+'time_history.npy',     np.stack(self.time_history))
    '''
    def write_message(self, end_effectors_positions : list[list[float]]) -> str:
        message = ''
        for i in range(self.n_participants):
            participant_end_effectors = end_effectors_positions[3*i*self.n_end_effectors:3*(i+1)*self.n_end_effectors]
            for j in range(self.n_end_effectors):
                message += f'X={participant_end_effectors[3*j]} Y={participant_end_effectors[3*j+1]} Z={participant_end_effectors[3*j+2]}/'
            message = message[:-1]  # Remove the last slash
            message += ';'  # End of participant's end effectors
        message = message[:-1]  # Remove the last semicolon
        ic(message)
        return message
    '''

    def write_message(self, end_effectors_positions: list[float]) -> str:
        message = ''
        for i in range(self.n_participants):
            participant_end_effectors = end_effectors_positions[3 * i * self.n_end_effectors : 3 * (i + 1) * self.n_end_effectors]
            for j in range(self.n_end_effectors):
                x = participant_end_effectors[3 * j]
                y = participant_end_effectors[3 * j + 1]
                z = participant_end_effectors[3 * j + 2]
                message += f'X={x:.3f} Y={y:.3f} Z={z:.3f}/'
            message = message[:-1]  # Remove the last slash
            message += ';'  # End of participant's end effectors
        message = message[:-1]  # Remove the last semicolon
        ic(message)
        return message
    
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
import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_engine import L3Agent, Phase_estimator_pca_online

ic.configureOutput(prefix='DEBUG | ')
# ic.disable()                              # Uncomment to stop debugging messages

class L3_Wrapper():

    def __init__(self, model_path, save_path, ID = 0, amplitude = 15, omega = 2, participants = 3, omega_parts = np.array([0, 3.4, 4.6]), c_strenght = 0.25):
        self.ID = ID                # Python CA instance ID
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega          # Movement frequency                
        self.x = 0 
        self.y = 0  
        self.z = 0  
        self.z_amp_ratio = 0.1    
        self.intial_position = 0
        self.intial_phase = 0
        self.participants = participants

        self.l3_phase = []
        self.l3_agent = L3Agent(participants, omega_parts, c_strenght, model_path)

        self.window_pca           = 4     # duration of the time window [seconds] in which the PCA is operatedf
        self.interval_between_pca = 1     # time interval [seconds] separating consecutive computations of the PCA

        self.estimators_live = []
        for _ in range(self.participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))     # One estimator for each participant

        self.kuramoto_phases = [np.zeros(self.participants)]

        self.time_history = [0]
        self.phases_history = [np.zeros(self.participants)]
        self.positions_history = []
        self.save_path = save_path
        self.create_folder_if_not_exists(save_path)

    def reset_CA(self):
        # STORE DATA
        self.save_data()
        self.plot_phases(np.stack(self.phases_history))

        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))

        self.kuramoto_phases = [np.zeros(self.participants)]

        self.phases_history = [np.zeros(self.participants)]
        self.time_history = [0]
        self.positions_history = []

    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string):
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3*self.l3_agent.n_nodes + 1
        return flag, numbers[0:-1], numbers[-1]
    
    def set_intial_position(self, position):
        self.intial_position = position[:, self.l3_agent.virtual_agent]
        self.intial_phase = 0
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
        for i in range(self.participants): # Collect phases of all participants. The ones from other participants are estimated
            if i != self.l3_agent.virtual_agent: phases.append(self.estimators_live[i].estimate_phase(positions[:, i], time))
            else: phases.append(theta - self.intial_phase)

        ic(phases)

        self.l3_agent.dt = delta_t
        self.time_history.append(self.time_history[-1] + delta_t)
        self.phases_history.append(np.array(phases))
        theta_next = self.l3_agent.l3_update_phase(np.array(phases))    # This function changes the omega of the avatar and outputs its new phase    
        self.kuramoto_phases.append(self.l3_agent.update_phases(self.kuramoto_phases[-1]))          # comment this and next line and uncomment upper line if L3 is connected with VR agents 
        # theta_next = np.mod(self.kuramoto_phases[-1][self.l3_agent.virtual_agent], 2*np.pi)

        ic(theta_next)

        self.y = self.amplitude * np.cos(theta_next)
        self.z = self.amplitude * np.sin(theta_next)

        y_k1 = self.amplitude * np.cos(self.kuramoto_phases[-1][1])
        z_k1 = self.amplitude * np.sin(self.kuramoto_phases[-1][1])

        y_k2 = self.amplitude * np.cos(self.kuramoto_phases[-1][2])
        z_k2 = self.amplitude * np.sin(self.kuramoto_phases[-1][2])

        message = 'X=' + str(self.intial_position[0]) + ' Y=' + str(self.intial_position[1] + self.y) + ' Z=' + str(self.intial_position[2] + self.z_amp_ratio * np.abs(self.z))    # Format data as UE Vector
        message = message + ';X=' + str(self.intial_position[0]) + ' Y=' + str(self.intial_position[1] + y_k1) + ' Z=' + str(self.intial_position[2] + self.z_amp_ratio * np.abs(z_k1))
        message = message + ';X=' + str(self.intial_position[0]) + ' Y=' + str(self.intial_position[1] + y_k2) + ' Z=' + str(self.intial_position[2] + self.z_amp_ratio * np.abs(z_k2))
        
        return message
    
    def plot_phases(self, phases):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.participants):
            if i == self.l3_agent.virtual_agent: plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i+1}')
            else: plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i+1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\phases_plot.png')
        plt.close()

        # PLOT ESTIMATION ERROR
        plt.figure()
        for i in range(1, self.participants): plt.plot(self.time_history[:-1], np.abs(phases[1:, i] - np.array(self.kuramoto_phases)[:-1, i]), color=colors[i], label=f'VH {i+1}')

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
        server_socket.listen(1)         # Limit number of connections to L3 socket
        print(f'Server listening on {address}:{port}')

        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        return connection, client_address

    @staticmethod
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")


if __name__ == "__main__":

    parameters = sys.argv[1:] # Takes the parameters (number of participants) from command line

    # Validate inputs to ensure they are numbers
    n_participants = 0
    error = False

    try:
        n_participants=(int(parameters[0]))  # try converting to float
    except ValueError:
        print(f'Invalid input: {parameters[0]} is not a number, please insert the number of participants connected to the L3 CA.')
        error = True
        sys.exit(1) 

    # Implement additional error handling
    if n_participants <= 0 or len(parameters) < 2:
        print('Error: Input is not valid, please insert the number of participants connected to the L3 CA and a valid path to store simlation data')
        sys.exit(1)
    elif not error:
        n_participants = n_participants + 1            # participant number is inteded as the number that the L3 is connected to
        path_to_data = parameters[1]

    # n_participants = 3
    # path_to_data = 'simulation_data'

    agent = L3_Wrapper('model', participants = n_participants, save_path = path_to_data)

    # Set the server address and port (must match with socket in UE)
    SERVER_ADDRESS = 'localhost'
    SERVER_PORT = 12345

    connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)

    def signal_handler(sig, frame):
        message = 'quit'
        connection.send(message.encode('utf-8'))
        print('Control + C pressed, closing socket...')
        agent.reset_CA()
        connection.close()
        sys.exit(0)

    # Register the signal handler for SIGINT (Control + C)
    signal.signal(signal.SIGINT, signal_handler)     # Ctrl + C to interrupt the game

    time = 0

    # Receive data stream from UE
    while True:    
        try:
            ic(f'waiting for data')
            ready_to_read, ready_to_write, exception = select.select([connection], [], [], 5) # 5 is the timeout time (in seconds)
            if ready_to_read: # Wait for the connection to start reading
                data = connection.recv(1024).decode() # Receive data for a maximum of 1024 bytes
                ic(f'Received data: {data}')

            if data == '' or not(ready_to_read or ready_to_write or exception): 
                print('Connection with client terminated')
                connection.close()
                
                agent.reset_CA()
                print('Cognitive Architecture reset completed')
                connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)
    
                time = 0

            else:
                _, position, delta_t = agent.parse_TCP_string(data) # Extract data coming from Unreal Engine
                position = np.reshape(position, (agent.participants, 3)).T

                if time == 0: agent.set_intial_position(position)
                
                time += delta_t  # Update time
                ic(position, delta_t)
                message = agent.update_position(position, delta_t, time) # Update position

                _, ready_to_write, _ = select.select([], [connection], [])
                if ready_to_write: connection.send(message.encode('utf-8')) # Send the new positions and time to Unreal Engine
                ic(f'Message sent: {message}')

        except KeyboardInterrupt:
            # Handle the Control + C key press gracefully
            signal_handler(None, None)    

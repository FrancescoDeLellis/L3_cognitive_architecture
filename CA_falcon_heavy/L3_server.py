import socket
import select
import re
import numpy as np
import sys
import signal
from icecream import ic

sys.path.append('L3_Engine')
from CA_falcon_heavy import CA_falcon_heavy
from fun_sync_indices import *

ic.configureOutput(prefix='DEBUG | ')
# ic.disable()

class L3_Wrapper():

    def __init__(self, acc = False, ID = 0, participants = 3):
        self.ID = ID                # Python CA instance ID             

        self.window_pca = 4     # duration of the time window [seconds] in which the PCA is operated
        self.interval_between_pca = 1     # time interval [seconds] separating consecutive computations of the PCA
        self.is_antiphase_ok = True
        self.acc = acc
        self.participants = participants

        self.ca_falcon = CA_falcon_heavy(is_acceleration=self.acc, is_antiphase_ok=self.is_antiphase_ok, n_participants=self.participants, window_pca=self.window_pca, interval_between_pca=self.interval_between_pca)
        self.sync_indices = [np.zeros(self.participants+1)]  # [global_sync, invidual_sync_1, invidual_sync_2, ....., invidual_sync_n_participants]

    def reset_CA(self):
        self.ca_falcon = CA_falcon_heavy(is_acceleration=self.acc, is_antiphase_ok=self.is_antiphase_ok, n_participants=self.participants, window_pca=self.window_pca, interval_between_pca=self.interval_between_pca)
        self.sync_indices = [np.zeros(self.participants+1)]

    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string):
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        ic(numbers)
        flag = len(numbers) == 3*self.ca_falcon.n_participants + 1
        return flag, numbers[0:-1], numbers[-1]

    # Calculates the synchronization indexes and formats the message to be sent to UE for animation
    def update_synch_indexes(self, positions, time):
        ic(time)
        self.sync_indices.append(self.ca_falcon.compute_sync_indices(current_state=positions, current_time=time))
        message = ';'.join([str(item) for item in self.sync_indices[-1]]) # 'global_sync;invidual_sync_1;invidual_sync_2: .....;invidual_sync_n_participants'
        return message
    
    @staticmethod
    def start_connection(address, port):
        # Create a TCP socket
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
    

if __name__ == "__main__":
    agent = L3_Wrapper(participants=3)

    # Set the server address and port (must match with socket in UE)
    SERVER_ADDRESS = 'localhost'
    SERVER_PORT = 12345

    connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)

    def signal_handler(sig, frame):
        message = 'quit'
        connection.send(message.encode('utf-8'))
        print('Control + C pressed, closing socket...')
        connection.close()
        sys.exit(0)

    # Register the signal handler for SIGINT (Control + C)
    signal.signal(signal.SIGINT, signal_handler)

    time = 0

    # Receive data stream from UE
    while True:    
        try:
            ic(f'waiting for data')
            ready_to_read, ready_to_write, exception = select.select([connection], [], [], 5)
            ic(ready_to_read, ready_to_write, exception)

            if ready_to_read: 
                data = connection.recv(1024).decode()
                ic(f'Received data: {data}')

            if data == '' or not(ready_to_read or ready_to_write or exception): 
                print('Connection with client terminated')
                connection.close()
                
                agent.reset_CA()
                print('Cognitive Architecture reset completed')
                connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)
    
                time = 0

            else:
                _, position, delta_t = agent.parse_TCP_string(data)
                time += delta_t
                ic(position, delta_t)
                message = agent.update_synch_indexes(position, time)

                _, ready_to_write, _ = select.select([], [connection], [])
                if ready_to_write: connection.send(message.encode('utf-8'))
                ic(f'Message sent: {message}')
        except KeyboardInterrupt:
            # Handle the Control + C key press gracefully
            signal_handler(None, None)
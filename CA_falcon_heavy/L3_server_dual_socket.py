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
# ic.disable()                              # Uncomment to get useful debugging messages

class L3_Wrapper():
    def __init__(self, acc = False, ID = 0, parts = 3):
        self.ID = ID   # Python CA instance ID             

        self.window_pca = 4     # duration of the time window [seconds] in which the PCA is operated
        self.interval_between_pca = 1     # time interval [seconds] separating consecutive computations of the PCA
        self.is_antiphase_ok = True

        self.ca_falcon = CA_falcon_heavy(is_acceleration=acc, is_antiphase_ok=self.is_antiphase_ok, n_participants=parts, window_pca=self.window_pca, interval_between_pca=self.interval_between_pca)
        self.sync_indices = [np.zeros(parts+1)]  # [global_sync, invidual_sync_1, invidual_sync_2, ....., invidual_sync_n_participants]

    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string):
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        ic(numbers)
        flag = len(numbers) == 3*self.ca_falcon.n_participants + 1
        return flag, numbers[0:-1], numbers[-1]
    
    def set_intial_position(self, position):
        self.intial_position = position
        self.y = -self.amplitude
        self.z = 0
        self.intial_phase = 0

    # Calculates the synchronization indexes and formats the message to be sent to UE for animation
    def update_synch_indexes(self, positions, time):
        ic(time)
        self.sync_indices.append(self.ca_falcon.compute_sync_indices(current_state=positions, current_time=time))
        message = ';'.join([str(item) for item in self.sync_indices[-1]]) # 'global_sync;invidual_sync_1;invidual_sync_2: .....;invidual_sync_n_participants'
        return message

if __name__ == "__main__":
    agent = L3_Wrapper(parts=2)

    # IP and port for incoming connections
    incoming_ip = '0.0.0.0'  # Listen on all available interfaces
    incoming_port = 12345

    # IP and port for outgoing connections
    outgoing_ip = 'localhost'  # Replace with the actual destination IP
    outgoing_port = 54321


    print("Arguments:", sys.argv)

    if len(sys.argv) > 1:
        inport = sys.argv[1]
        print("In Port =", inport)
        incoming_port = int(inport)
    
    if len(sys.argv) > 2:
        outport = sys.argv[2]
        print("Out Port =", outport)
        outgoing_port = int(outport)
        
    if len(sys.argv) > 3:
        outgoing_ip = sys.argv[3]
        print("Out IP =", outgoing_ip)



    # setup the input data stream
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Bind the socket to the server address and port
        server_socket.bind((incoming_ip, incoming_port))
    except socket.error as e:
        print("Connection error: %s" % e)
    
    # Listen for incoming connections
    server_socket.listen(1)         # Limit number of connections to L3 socket
    print(f"Listening for incoming connections on {incoming_ip}:{incoming_port}")

    # Wait for a client connection
    print('Waiting for a connection...')
    incoming_conn, incoming_addr = server_socket.accept()
    print(f"Accepted connection from {incoming_addr}")

    # Set the output data stream
    outgoing_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        outgoing_conn.connect((outgoing_ip, outgoing_port))
    except socket.error as e:
        print("Connection error: %s" % e)
    print(f"Connected to outgoing address {outgoing_ip}:{outgoing_port}")

    # Register the signal handler for SIGINT (Control + C)
    def signal_handler(sig, frame):
        message = 'quit'
        outgoing_conn.send(message.encode('utf-8'))
        print('Control + C pressed, closing socket...')
        outgoing_conn.close()
        incoming_conn.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    time = 0

    # Receive data stream and reply
    while True:    
        try:
            # ic(f'waiting for data')
            ready_to_read, _, _ = select.select([incoming_conn], [], [])
            if ready_to_read:
                data = incoming_conn.recv(1024).decode()
                _, position, delta_t = agent.parse_TCP_string(data)
                ic(f'Received data: {data}')
        
            time += delta_t
            ic(position, delta_t)
            message = agent.update_synch_indexes(position, time)

            _, ready_to_write, _ = select.select([], [outgoing_conn], [])
            if ready_to_write:
                outgoing_conn.send(message.encode('utf-8'))
                ic(f'Message sent: {message}')
        except KeyboardInterrupt:
            # Handle the Control + C key press gracefully
            signal_handler(None, None)
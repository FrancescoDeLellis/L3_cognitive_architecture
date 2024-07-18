import socket
import re
import numpy as np
from icecream import ic

ic.configureOutput(prefix='DEBUG | ')
ic.disable()                              # Uncomment to get useful debugging messages

class L3_Wrapper():

    def __init__(self, ID = 0, amp = 20, omega = 2):
        self.ID = ID                # Python CA instance ID
        self.amplitude = amp        # Movement amplitude
        self.omega = omega          # Movement frequency                
        self.x = 0   
        self.z = 0  
        self.z_amp_ratio = 0.3    
        self.intial_position = 0

        self.thetas = []

    # This function extracts the 3D data position coming from UE
    @staticmethod
    def parse_TCP_string(string):
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        return numbers[0:3], numbers[3]

    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, position, delta_t):
        theta = np.arctan2(self.z, self.x)
        self.thetas.append(theta)
        theta_next = theta + delta_t * self.omega           # Forward euler integration to obtain next position

        self.x = self.amplitude * np.cos(theta_next)
        self.z = self.amplitude * np.sin(theta_next)

        message = 'X=' + str(self.intial_position[0] + self.x) + ' Y=' + str(position[1]) + ' Z=' + str(self.intial_position[2] + self.z_amp_ratio * np.abs(agent.z))    # Format data as UE Vector
        return message

if __name__ == "__main__":
    agent = L3_Wrapper()

    # Set the server address and port (must match with socket in UE)
    SERVER_ADDRESS = 'localhost'
    SERVER_PORT = 12345

    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Bind the socket to the server address and port
        server_socket.bind((SERVER_ADDRESS, SERVER_PORT))
    except socket.error as e:
        print("Connection error: %s" % e)

    # Listen for incoming connections
    server_socket.listen(1)         # Limit number of connections to L3 socket
    print(f'Server listening on {SERVER_ADDRESS}:{SERVER_PORT}')

    timeout = 10
    server_socket.settimeout(timeout)

    while True:
        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        try:
            print(f'Connection from {client_address}')

            while np.sum(agent.intial_position) == 0:
                data = connection.recv(1024).decode()
                agent.intial_position, delta_t = agent.parse_TCP_string(data)           # Store intial position of L3 end-effector
                ic(f'Received data: {agent.intial_position}')
                position = agent.intial_position

            # Receive data stream from UE
            while True:
                data = connection.recv(1024).decode()
                if not data:
                    print(f'Connection from {client_address} has failed')
                    break
                ic(f'Received data: {data}')
                position, delta_t = agent.parse_TCP_string(data)
                ic(position, delta_t)
                message = agent.update_position(position, delta_t)
                connection.send(message.encode('utf-8'))
                ic(f'Message sent: {message}')
            

        finally:
            print("Closing connection")
            connection.close()    

            print('Waiting for a connection...')
            connection, client_address = server_socket.accept()
            # Handle the timeout (e.g., close the connection)     
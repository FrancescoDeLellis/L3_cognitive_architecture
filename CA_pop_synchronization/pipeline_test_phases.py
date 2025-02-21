from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
import os

print('PATH: ', os.path.abspath("."))
path = 'C:\\Users\\Angelo\\Desktop\\Codici\\branch_con_Marco\\CA_pop_synchronization\\simulation_data'
os.chdir(path)
print('PATH: ', os.path.abspath("."))
print(os.listdir())  # Mostra i file nella cartella corrente
savepath = path+'\\testing_results'

phases_history = np.load('.\phases_history.npy')
positions_history = np.load('.\postions_history.npy')
positions_history = positions_history - np.mean(positions_history, axis=0)  # Remove the mean of the positions to avoid the influence of the center of mass
time_history = np.load('.\\time_history.npy')


real_phases = np.angle(hilbert(positions_history[:, :, 1], axis=0))  # Transform the y-axis signals of the participants to phases through the Hilbert transform


for i in range(1, phases_history.shape[1]): # Start from 1 because the first one is always the L3 CA
    plt.plot(time_history, real_phases[:, i], label=f'Participant {i+1}')
    plt.plot(time_history, phases_history[:, i], label=f'Participant {i+1} estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Phases (rad)')
    plt.title('Comparison between real and estimated phases')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}\\comparison_phases_participant_{i+1}.png')
    plt.close()

    plt.plot(time_history[:-1], np.abs(real_phases[1:, i] - phases_history[:-1, i]))
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Error between real and estimated phases')
    plt.grid(True)
    plt.savefig(f'{path}\\error_phases_participant_{i+1}.png')
    plt.close()
    
import numpy as np
import matplotlib.pyplot as plt
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from scipy.signal import hilbert

# Load the numpy array from file
data = np.load(r"2025_6_11_h17_m36_s57_positions_history.npy")
time = np.load(r"2025_6_11_h17_m36_s57_time_history.npy")
time = time[:-1] # Adjust time to match the data length

Estimator = RecursiveOnlinePhaseEstimator(n_dims_estimand_pos=3, listening_time=5, discarded_time=5, min_duration_first_pseudoperiod=0, look_behind_pcent=5, look_ahead_pcent=15, time_const_lowpass_filter_phase=0.1, time_const_lowpass_filter_pos=0.01)
estimated_phase = np.zeros((data.shape[0], 1))

data = data - np.mean(data, axis=0)  # Remove the mean of the positions to avoid the influence of the center of mass
hilbert_phase = np.angle(hilbert(data[:, 1], axis=0))+np.pi  # Compute the Hilbert phase for each position dimension

for i in range(len(time)):
    estimated_phase[i] = Estimator.update_estimator(data[i, :], time[i])

print(hilbert_phase.shape)

# Plot the data
'''
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Positions History')
plt.xlabel('Time Step')
plt.ylabel('Position Value')
plt.grid(True)
plt.show()
'''
plt.figure(figsize=(10, 6))
plt.plot(time, estimated_phase, label='Estimated Phase', color='orange')
plt.plot(time, hilbert_phase, label='Hilbert Phase', color='blue')
plt.title('Phase Estimation')
plt.xlabel('Time (s)')
plt.ylabel('Phase')
plt.legend()
plt.grid(True)
plt.show()
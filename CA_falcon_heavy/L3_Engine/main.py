import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from CA_falcon_heavy import CA_falcon_heavy
from fun_sync_indices import *


# Parameters to set
# -----------------

window_pca           = 4     # duration of the time window [seconds] in which the PCA is computed
interval_between_pca = 1     # time interval [seconds] separating consecutive computations of the PCA
is_antiphase_ok      = True  # True: consider antiphase as synchronized condition. False: Do not

file_path = r".\UM_data\BLOC3.csv"
n_participants  = 5          # number of participants
is_acceleration = False      # True: motion data are accelerations; False: positions


# Data load
# ---------


first_row = 0    # first row to read data from in csv file
t_column  = 0    # column of time in csv file
x_column  = []
y_column  = []
z_column  = []
for i in range(0, n_participants):  # assign column numbers for all participants
    x_column.append(1 + 3*i)
    y_column.append(2 + 3*i)
    z_column.append(3 + 3*i)

dtype_spec  = {0: 'float64',  1: 'float64'}
motion_data = pd.read_csv(file_path, skiprows=first_row, dtype=dtype_spec)

time_vec = np.array(motion_data.iloc[:,t_column]).T
dt       = np.diff(time_vec, axis=0) 


# Set trajectory vector
# ---------------------
n_time_instants = motion_data.shape[0]

if is_acceleration:
    trajectory     = []   # (position)
    trajectory_vel = []
    trajectory_acc = []

    for i_p in range(0, n_participants):
        trajectory_acc.append( np.array([motion_data.iloc[:,x_column[i_p]], motion_data.iloc[:,y_column[i_p]], motion_data.iloc[:,z_column[i_p]]]).T )
        trajectory_vel.append( np.zeros((n_time_instants, 3)) )
        trajectory.append(     np.zeros((n_time_instants, 3)) )
        
        for i_t in range(1, n_time_instants):
            trajectory_vel[i_p][i_t] = trajectory_vel[i_p][i_t - 1] + trajectory_acc[i_p][i_t - 1] * dt[i_t-1]
            trajectory[i_p][i_t]     = trajectory[i_p][i_t - 1]     + trajectory_vel[i_p][i_t - 1] * dt[i_t-1]
else: 
    trajectory = []
    for i in range(0,n_participants):
        trajectory.append( np.array([motion_data.iloc[:,x_column[i]], motion_data.iloc[:,y_column[i]], motion_data.iloc[:,z_column[i]]]).T )

print('Computation started...')


# Computation of Hilbert phase and synchronization index
# ----------------------------
global_sync_index_offline       = []
individual_sync_indices_offline = []
phases_hilbert                  = []

for i in range(0, n_participants):
    centroid_trajectory = np.mean(trajectory[i], axis=0)
    centered_trajectory = trajectory[i] - centroid_trajectory

    pca = PCA(n_components=1)
    pca.fit(centered_trajectory)
    principal_component = np.reshape(pca.transform(centered_trajectory), -1)
    phases_hilbert.append( np.angle(hilbert(principal_component)) )
    individual_sync_indices_offline.append( np.zeros((n_time_instants, 1)) )

for i_t in range(n_time_instants-1):
	phases_current = [sublist[i_t] for sublist in phases_hilbert]
	global_sync_index_offline.append( compute_global_sync_index(phases_current, is_antiphase_ok) )
	individual_sync_indices_temp = compute_local_sync_indices(phases_current, is_antiphase_ok)
	for i_p in range(0, n_participants):
		individual_sync_indices_offline[i_p][i_t] = individual_sync_indices_temp[i_p]
        
print('Computation of offline sync. indices completed...')


# Online phase estimation and synchronization index
# -------------------------------------------------

motion_data = pd.read_csv(file_path, skiprows=first_row, dtype=dtype_spec)
time_vec    = np.array(motion_data.iloc[:,t_column]).T
traj        = np.zeros((motion_data.shape[0], n_participants * 3)) 

for i in range(n_participants):
    traj[:, i * 3] = motion_data.iloc[:, x_column[i]]
    traj[:, i * 3 + 1] = motion_data.iloc[:, y_column[i]]
    traj[:, i * 3 + 2] = motion_data.iloc[:, z_column[i]]

ca_falcon = CA_falcon_heavy(
    is_acceleration=is_acceleration, 
    is_antiphase_ok=is_antiphase_ok, 
    n_participants=n_participants, 
    window_pca=window_pca,
    interval_between_pca=interval_between_pca)

sync_indices = [np.zeros((n_participants+1))]  # [global_sync, invidual_sync_1, invidual_sync_2, ....., invidual_sync_n_participants]
for i_t in range(n_time_instants-1):
    print(f'Computation of online sync. indices: {i_t}/{n_time_instants-1}.')
    sync_indices[i_t,:] = ca_falcon.compute_sync_indices(current_state=traj[i_t,:], current_time=time_vec[i_t])
    
print('Computation of online sync. indices completed.')


# Plots
# -----

plt.figure(figsize = (10, 5))
plt.plot(time_vec[0:len(global_sync_index_offline)], global_sync_index_offline, label='global sync offline')
plt.plot(time_vec[0:len(sync_indices[:,0])],  sync_indices[:,0],  label='global sync online', linestyle='--')

plt.title('Group synchronization index')
plt.xlabel('Time [s]')
plt.ylabel('Sync. index')
plt.legend()
plt.grid(True)

for i in range(n_participants):
    plt.figure(figsize = (10, 5))
    plt.plot(time_vec[0:len(individual_sync_indices_offline[i])], individual_sync_indices_offline[i], label='Individual synchronization index offline')
    plt.plot(time_vec[0:len(sync_indices[:,i+1])],  sync_indices[:,i+1],  label='Individual synchronization index online', linestyle='--')

    plt.title(f'Individual synchronization index {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Sync. index')
    plt.legend()
    plt.grid(True)

plt.show()
import cmath
import warnings
import numpy as np
from Phase_estimator_pca_online import Phase_estimator_pca_online
from fun_sync_indices import *


class CA_falcon_heavy:
	def __init__(self, is_acceleration, is_antiphase_ok, n_participants, window_pca, interval_between_pca):
		self.is_acceleration      = is_acceleration
		self.is_antiphase_ok      = is_antiphase_ok
		self.n_participants       = n_participants
		self.window_pca           = window_pca
		self.interval_between_pca = interval_between_pca
		self.time_prev            = 0
		self.velocity             = np.zeros(self.n_participants*3)
		self.position             = np.zeros(self.n_participants*3)
		self.phases               = np.zeros(self.n_participants)
		self.sync_indices         = np.zeros(self.n_participants+1)
		self.estimator_online     = {}
		for i in range(self.n_participants):
			self.estimator_online[i] = Phase_estimator_pca_online(self.window_pca, self.interval_between_pca)

	
	def update_phases(self, current_state,current_time):
		if self.is_acceleration:
			for i in range(self.n_participants*3):
				self.velocity[i] += current_state[i] * (current_time-self.time_prev )
				self.position[i] += self.velocity[i] * (current_time-self.time_prev )
				self.time_prev = current_time
		else:
			self.position = current_state
		for i in range(self.n_participants):
			self.phases[i]=self.estimator_online[i].estimate_phase([self.position[3*i],self.position[(3*i)+1],self.position[(3*i)+2]], current_time)
		return self.phases


	def compute_sync_indices(self, current_time, current_state):
		if self.n_participants > 1:
			self.update_phases(current_state,current_time)
			self.sync_indices[0]  = compute_global_sync_index( self.phases, self.is_antiphase_ok)
			self.sync_indices[1:] = compute_local_sync_indices(self.phases, self.is_antiphase_ok)
			return self.sync_indices
		else:
			warnings.warn("Sync indices not defined when number of participant < 2.")
			return None
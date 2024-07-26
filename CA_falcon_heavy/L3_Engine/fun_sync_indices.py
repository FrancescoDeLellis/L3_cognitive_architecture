import cmath
import numpy as np

def compute_global_sync_index(phases: list[float], is_antiphase_ok: bool) -> float:
	if is_antiphase_ok:  phases = 2*np.mod(phases, np.pi)
	return np.abs(np.average(np.exp(1j * phases)))

def compute_local_sync_indices(phases: list[float], is_antiphase_ok: bool) -> np.array:
	if is_antiphase_ok:  phases = 2*np.mod(phases, np.pi)
	n_oscillators = len(phases)
	sum_phases = np.sum(np.exp(1j * phases))
	local_sync_indices = []
	for i_osc in range(n_oscillators):
		mean_phases_others = cmath.phase(( sum_phases - np.exp(1j * phases[i_osc]) ) / (n_oscillators-1))
		local_sync_indices.append( abs(np.exp(1j * phases[i_osc]) + np.exp(1j * mean_phases_others) ) / 2 )
	return local_sync_indices
import warnings
import numpy as np
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt
from low_pass_filters import LowPassFilter, LowPassFilterPhase


##################################################
# Estimator class
##################################################

class RecursiveOnlinePhaseEstimator:
    def __init__(self,
                 n_dims_estimand_pos: int,
                 listening_time,
                 discarded_time                  = 0,
                 min_duration_first_pseudoperiod  = 1,
                 look_behind_pcent               = 0,
                 look_ahead_pcent                = 10,
                 time_const_lowpass_filter_phase = None,
                 time_const_lowpass_filter_pos   = None,
                 is_use_baseline                 = False,
                 baseline_pos_loop               = None,
                 time_step_baseline              = 0.01,
                 ref_frame_estimand_points       = None,
                 ref_frame_baseline_points       = None,
                 is_use_elapsed_time             = False):

        self.is_first_loop_estimated = False
        assert look_ahead_pcent + look_behind_pcent <= 100, "look_ahead_pcent + look_behind_pcent must not exceed 100"

        # Initialization from arguments
        self.n_dims                          = n_dims_estimand_pos
        self.discarded_time                  = discarded_time       # [s] discarded at the beginning before estimation
        self.listening_time                  = listening_time       # [s] waits this time before estimating first loop must contain 2 pseudoperiods
        self.look_ahead_pcent                = look_ahead_pcent     # % of last completed loop before last nearest point on which estimate the new phase
        self.look_behind_pcent               = look_behind_pcent    # % of last completed loop after last nearest point on which estimate the new phase
        self.time_const_lowpass_filter_phase = time_const_lowpass_filter_phase
        self.time_const_lowpass_filter_pos   = time_const_lowpass_filter_pos
        self.is_use_baseline                 = is_use_baseline
        self.min_duration_pseudoperiod        = min_duration_first_pseudoperiod # [s]
        self.time_step_baseline              = time_step_baseline
        self.is_use_elapsed_time             = is_use_elapsed_time   # True: also uses elapsed time to determine period completion

        if is_use_baseline:
            assert n_dims_estimand_pos == 3,      "Baseline mode can be used only with n_dim = 3"
            assert baseline_pos_loop is not None, "Baseline mode was required but baseline_pos_loop was not provided"
            assert ref_frame_estimand_points is not None, "Baseline mode was required but ref_frame_points was not provided"

            self.baseline_pos_loop = baseline_pos_loop.copy()
            self.ref_frame_estimand_points = ref_frame_estimand_points.copy()
            self.ref_frame_baseline_points = ref_frame_baseline_points.copy()

        if time_const_lowpass_filter_phase in {None, 0, -1}:
            self.is_use_lowpass_filter_phase = False
        else:  self.is_use_lowpass_filter_phase = True
        if time_const_lowpass_filter_pos in {None, 0, -1}:
            self.is_use_lowpass_filter_pos = False
        else:  self.is_use_lowpass_filter_pos = True

        # Attributes not tunable by caller
        self.phase_jump_for_loop_detection = np.pi

        # Initial values
        self.active_mode             = "sleeping"
        self.phase_offset            = 0
        self.loop_end_time           = 0
        self.pos_signal              = []  #                  [0=start_listening ...]
        self.vel_signal              = []
        self.local_time_signal       = []  # [0=initial_time ... start_listening ...]
        self.delimiter_time_instants = []  # framed in the external time
        self.delimiter_idxs          = []  # index in which each loop start (relative to pos_signal)
        self.local_phase_signal      = []  # local:  without offset
        self.global_phase_signal     = []  # global: with offset
        self.new_loop                = []
        self.idx_curr_phase_in_latest_loop = 0


    def update_pos_vel(self, curr_pos) -> None:
        self.pos_signal.append(curr_pos)
        if len(self.pos_signal) < 2:  # first instant
            if self.is_use_lowpass_filter_pos:
                self.lowpass_filter_pos = LowPassFilter(init_state=curr_pos,
                                                        time_step=0,
                                                        time_const=self.time_const_lowpass_filter_pos,)
            self.vel_signal.append(np.zeros(self.n_dims))
        else:
            curr_step_time = self.local_time_signal[-1] - self.local_time_signal[-2]
            if self.is_use_lowpass_filter_pos:
                self.lowpass_filter_pos.change_time_step(curr_step_time)
                self.pos_signal[-1] = self.lowpass_filter_pos.update_state(self.pos_signal[-1])
            self.vel_signal.append((self.pos_signal[-1] - self.pos_signal[-2]) / curr_step_time)


    def get_kinematics(self, idx:int) -> np.ndarray:
        return np.concatenate((self.pos_signal[idx], self.vel_signal[idx]))


    def update_look_ranges(self) -> None:
        self.look_ahead_range  = int(len(self.latest_loop) * self.look_ahead_pcent / 100)
        self.look_behind_range = int(len(self.latest_loop) * self.look_behind_pcent / 100)


    # This is the main cycle performed at each time instant
    def update_estimator(self, curr_pos, curr_external_time) -> float:

        # Time frames and vectors:
        # external time:  0---E------->
        # local time:         0---D--->
        # pos/vel/phase:          0--->
        #   E: estimator started; D: discarded time/listening starts

        if not self.local_time_signal:  self.time_estimator_started = curr_external_time  # initialize initial_time
        self.local_time_signal.append(curr_external_time - self.time_estimator_started)

        # Update active mode
        if self.active_mode == "sleeping" and self.local_time_signal[-1] >= self.discarded_time:
            self.active_mode = "listening"
            self.idx_time_start_listening = len(self.local_time_signal)-1
        if self.active_mode == "listening" and self.local_time_signal[-1] > self.discarded_time + self.listening_time:
            self.active_mode = "estimating"
            # self.idx_time_start_estimating = len(self.local_time_signal)

        # Update estimator according to active mode
        match self.active_mode:
            case "sleeping":
                return None

            case "listening":
                self.update_pos_vel(curr_pos)
                return None

            case "estimating":
                self.update_pos_vel(curr_pos)

                if not self.is_first_loop_estimated:
                    self.latest_loop = compute_loop_with_autocorrelation(
                        pos_signal=self.pos_signal,
                        vel_signal=self.vel_signal,
                        local_time_vec=self.local_time_signal,
                        min_duration_pseudoperiod=self.min_duration_pseudoperiod)
                    self.update_look_ranges()

                    self.delimiter_time_instants.append(float(self.local_time_signal[self.idx_time_start_listening] + self.time_estimator_started))
                    self.delimiter_idxs.append(0)
                    self.delimiter_time_instants.append(float(self.local_time_signal[len(self.latest_loop) + self.idx_time_start_listening] + self.time_estimator_started))
                    self.delimiter_idxs.append(len(self.latest_loop))
                    self.is_first_loop_estimated = True

                    if self.is_use_baseline:  self.compute_phase_offset()

                    # Compute phases in first loop
                    local_phases_first_loop  = np.linspace(0, 2 * np.pi, len(self.latest_loop))
                    global_phases_first_loop = np.mod(local_phases_first_loop + self.phase_offset, 2 * np.pi)
                    self.local_phase_signal  = self.local_phase_signal + local_phases_first_loop.tolist()
                    self.global_phase_signal = self.global_phase_signal + (global_phases_first_loop + self.phase_offset).tolist()

                    # Estimate phases between first loop and current time
                    idx_first_instant_of_second_loop = len(self.latest_loop)
                    idx_prev_instant = len(self.pos_signal) - 2
                    if self.is_use_lowpass_filter_phase:
                        self.lowpass_filter_phase = LowPassFilterPhase(init_state=self.local_phase_signal[-1],
                                                                       time_step=self.local_time_signal[idx_first_instant_of_second_loop] - self.local_time_signal[idx_first_instant_of_second_loop-1],
                                                                       time_const=self.time_const_lowpass_filter_phase,
                                                                       wrap_interv="0_to_2pi")
                    for idx_scanning in range(idx_first_instant_of_second_loop, idx_prev_instant+1):   # recall: range excludes end point
                        curr_time_step = self.local_time_signal[idx_scanning] - self.local_time_signal[idx_scanning-1]
                        scanning_time = self.local_time_signal[idx_scanning + self.idx_time_start_listening] + self.time_estimator_started
                        self.compute_phase(self.get_kinematics(idx_scanning), curr_time_step, scanning_time)
                        if idx_scanning > idx_first_instant_of_second_loop:  # at first instant of second loop don't check, because new loop is already loaded and besides new_loop is empty
                            self.possibly_update_latest_loop(scanning_time, idx_scanning)
                        self.new_loop.append(self.get_kinematics(idx_scanning))

                # Estimate phase at current time
                curr_idx       = len(self.pos_signal)
                curr_time_step = self.local_time_signal[-1] - self.local_time_signal[-2]
                self.compute_phase(self.get_kinematics(-1), curr_time_step, curr_external_time)
                self.possibly_update_latest_loop(curr_external_time, curr_idx)
                self.new_loop.append(self.get_kinematics(-1))
                return self.global_phase_signal[-1]

            case _:  raise ValueError(f"No match found for value: {self.active_mode}")


    def possibly_update_latest_loop(self, curr_external_time, curr_idx):
            if self.local_phase_signal[-1] - self.local_phase_signal[-2] < - self.phase_jump_for_loop_detection:   # a pseudoperiodicity window ended
                self.delimiter_time_instants.append(float(curr_external_time))
                self.delimiter_idxs.append(curr_idx)
                self.latest_loop = np.vstack(self.new_loop)
                self.update_look_ranges()
                self.new_loop = []         # reinitialize new_loop
                self.idx_curr_phase_in_latest_loop = 0


    def compute_phase(self, curr_kinematics, curr_time_step, curr_external_time):
        len_latest_loop = len(self.latest_loop)
        if self.idx_curr_phase_in_latest_loop + self.look_ahead_range <= len_latest_loop:
            #   loop: [- - - - - - single part - - - - -]
            idxs_loop_for_search = np.arange( max(0, self.idx_curr_phase_in_latest_loop - self.look_behind_range),
                                              self.idx_curr_phase_in_latest_loop + self.look_ahead_range)
            loop_for_search = self.latest_loop[idxs_loop_for_search]
        else:
            #   loop: [part_2 - - - - - - - - part_1]
            idxs_end_part_2 = self.idx_curr_phase_in_latest_loop + self.look_ahead_range - len_latest_loop
            idxs_part_1 = np.arange(max(idxs_end_part_2, self.idx_curr_phase_in_latest_loop - self.look_behind_range), len_latest_loop)
            idxs_part_2 = np.arange(0, idxs_end_part_2)
            idxs_loop_for_search = np.concatenate((idxs_part_1, idxs_part_2))
            loop_for_search = self.latest_loop[idxs_loop_for_search]
        if not self.is_use_elapsed_time:
            idx_min_distance = compute_idx_min_distance(pos_signal = loop_for_search[:, 0:self.n_dims].copy(),
                                                        vel_signal = loop_for_search[:, self.n_dims:].copy(),
                                                        curr_pos   = curr_kinematics[0:self.n_dims],
                                                        curr_vel   = curr_kinematics[self.n_dims:])
        else:
            latest_loop_time = np.array(self.local_time_signal[self.delimiter_idxs[-2]:self.delimiter_idxs[-1]]) - self.local_time_signal[self.delimiter_idxs[-2]]
            time_loop_for_search = latest_loop_time[idxs_loop_for_search]
            idx_min_distance = compute_idx_min_distance_with_time(pos_signal           = loop_for_search[:, 0:self.n_dims].copy(),
                                                                  vel_signal           = loop_for_search[:, self.n_dims:].copy(),
                                                                  curr_pos             = curr_kinematics[0:self.n_dims],
                                                                  curr_vel             = curr_kinematics[self.n_dims:],
                                                                  curr_elapsed_time    = curr_external_time - self.delimiter_time_instants[-1],
                                                                  elapsed_time_signal= time_loop_for_search,
                                                                  duration_latest_loop = self.delimiter_time_instants[-1] - self.delimiter_time_instants[-2])
        self.idx_curr_phase_in_latest_loop = idxs_loop_for_search[idx_min_distance]
        self.local_phase_signal.append((2 * np.pi * self.idx_curr_phase_in_latest_loop) / len_latest_loop)
        if self.is_use_lowpass_filter_phase:
            self.lowpass_filter_phase.change_time_step(curr_time_step)
            self.local_phase_signal[-1] = self.lowpass_filter_phase.update_state(self.local_phase_signal[-1])
        self.global_phase_signal.append(np.mod(self.local_phase_signal[-1] + self.phase_offset, 2 * np.pi))


    def compute_phase_offset(self) -> None:
        # TODO edit point. path A: comment next five lines. path B: uncoment
        x_axis_baseline, y_axis_baseline, z_axis_baseline = calculate_axes(self.ref_frame_baseline_points)
        rotat_matrix_global_to_ego_baseline = np.vstack([x_axis_baseline, y_axis_baseline, z_axis_baseline]).T
        self.baseline_pos_loop =  self.baseline_pos_loop[:, 0:3] @ rotat_matrix_global_to_ego_baseline
        centroid_baseline = np.mean(self.baseline_pos_loop, axis=0)
        self.baseline_pos_loop = self.baseline_pos_loop - centroid_baseline

        x_axis_estimand, y_axis_estimand, z_axis_estimand = calculate_axes(self.ref_frame_estimand_points)
        rotat_matrix_global_to_ego_estimand = np.vstack([x_axis_estimand, y_axis_estimand, z_axis_estimand]).T
        rotated_loop = self.latest_loop[:, 0:3] @ rotat_matrix_global_to_ego_estimand

        centroid_estimand = np.mean(rotated_loop, axis=0)
        rotated_centered_loop = rotated_loop - centroid_estimand

        scale_factors = np.std(self.baseline_pos_loop, axis=0) / np.std(rotated_centered_loop, axis=0)
        scale_factors[np.isnan(scale_factors)] = 1
        scaled_rotated_centered_loop = rotated_centered_loop * scale_factors

        curr_pos_ = scaled_rotated_centered_loop[0, :]
        curr_step_time = self.local_time_signal[-1] - self.local_time_signal[-2]
        curr_vel_ = (scaled_rotated_centered_loop[1, :] - scaled_rotated_centered_loop[0, :]) / curr_step_time
        baseline_vel_loop = np.gradient(self.baseline_pos_loop, np.arange(0, len(self.baseline_pos_loop) * self.time_step_baseline, self.time_step_baseline), axis=0)
        index = compute_idx_min_distance(pos_signal = self.baseline_pos_loop.copy(),
                                         vel_signal = baseline_vel_loop.copy(),
                                         curr_pos   = curr_pos_,
                                         curr_vel   = curr_vel_)
        self.phase_offset = (2 * np.pi * index) / len(self.baseline_pos_loop)



##################################################
# Helper functions
##################################################

threshold_acceptable_peaks_wrt_maximum_pcent = 40  # Acceptance range of autocorrelation peaks defined as a percentage of the maximum autocorrelation value.


def compute_loop_with_autocorrelation(pos_signal, vel_signal, local_time_vec, min_duration_pseudoperiod) -> np.ndarray:
    n_dim = len(pos_signal[0])
    pos_signal_stacked = np.vstack(pos_signal)  # time flows vertically
    vel_signal_stacked = np.vstack(vel_signal)

    pos_signal_unstacked = np.full(n_dim, None)
    autocorr_vecs_per_dim = np.full(n_dim, None)
    for i in range(n_dim):
        pos_signal_unstacked[i] = pos_signal_stacked[:, i]
        autocorr_vecs_per_dim[i] = compute_autocorr_vec(detrend(pos_signal_unstacked[i]))
    autocorr_vec_tot = np.sum(np.array(autocorr_vecs_per_dim), axis=0)

    idx_min_length = np.argmax(np.array(local_time_vec) > min_duration_pseudoperiod)  # argmax finds the first True
    autocorr_vec_tot[:idx_min_length] = autocorr_vec_tot[idx_min_length]

    autocorr_vec_tot = autocorr_vec_tot / np.max(np.abs(autocorr_vec_tot))

    peaks, _ = find_peaks(autocorr_vec_tot)
    peaks_values = autocorr_vec_tot[peaks[np.where(peaks > idx_min_length)]]
    assert max(peaks_values) > 0, "No valid first loop was found, try increasing the listening time."
    lower_bound_acceptable_peaks_values = max(peaks_values) - (max(peaks_values) * threshold_acceptable_peaks_wrt_maximum_pcent / 100)
    idxs_possible_period = np.where(peaks_values > lower_bound_acceptable_peaks_values)[0]  # indexing necessary because np.where returns a tuple containing an array
    idx_start_loop = 0
    idx_stop_loop = peaks[idxs_possible_period][0]

    pos_loop = pos_signal_stacked[idx_start_loop:idx_stop_loop, :]
    vel_loop = vel_signal_stacked[idx_start_loop:idx_stop_loop, :]
    return np.column_stack((pos_loop, vel_loop))


def compute_autocorr_vec(signal: np.ndarray) -> np.ndarray:
    max_lag = len(signal) // 2
    autocorr_vec = np.zeros(max_lag + 1)

    for lag in range(1, len(autocorr_vec)):
        var = np.var(signal[0:2 * lag])
        if var == 0:
            autocorr_vec[lag] = 0
        else:
            mean = np.mean(signal[0:2 * lag])
            sum_ = 0
            for t in range(lag):
                sum_ += (signal[t] - mean) * (signal[t + lag] - mean)
            autocorr_vec[lag] = sum_ / (lag * var)
    return autocorr_vec


def compute_idx_min_distance(pos_signal, vel_signal, curr_pos, curr_vel) -> int:
    distances_pos = np.sqrt(np.sum((pos_signal - curr_pos) ** 2, axis=1))
    distances_vel = np.sqrt(np.sum((vel_signal - curr_vel) ** 2, axis=1))
    distances_pos = distances_pos / max(distances_pos, default=1)  # avoids dividing by zero
    distances_vel = distances_vel / max(distances_vel, default=1)
    return np.argmin(distances_pos + distances_vel)

def compute_idx_min_distance_with_time(pos_signal, vel_signal, curr_pos, curr_vel, curr_elapsed_time, elapsed_time_signal, duration_latest_loop) -> int:
    distances_pos = np.sqrt(np.sum((pos_signal - curr_pos) ** 2, axis=1))
    distances_vel = np.sqrt(np.sum((vel_signal - curr_vel) ** 2, axis=1))
    distances_time_no_shift = np.abs(elapsed_time_signal - curr_elapsed_time)
    distances_time_shift    = np.abs(elapsed_time_signal + duration_latest_loop - curr_elapsed_time)
    distances_time          = np.minimum(distances_time_no_shift, distances_time_shift)
    distances_pos  = distances_pos  / max(distances_pos,  default=1)
    distances_vel  = distances_vel  / max(distances_vel,  default=1)
    distances_time = distances_time / max(distances_time, default=1)
    return np.argmin(distances_pos + distances_vel + distances_time)


def calculate_axes(points):
    """Computes a frame of reference, given 3 non collinear points"""
    x_axis = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
    z_vector = points[2] - points[1]
    z_axis = z_vector - np.dot(z_vector, x_axis) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis

def angle_distance_radians(alpha, beta):
    diff = np.abs(np.angle(np.exp(1j * (alpha - beta))))
    return diff
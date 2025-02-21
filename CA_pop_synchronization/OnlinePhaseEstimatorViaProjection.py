import math
import numpy as np
from sklearn.decomposition import PCA
from LowPassFilters import LowPassFilterPhase, wrap_to_2pi, wrap_to_pi

class OnlinePhaseEstimatorViaProjection:

    def __init__(self,
                 window_for_offset_update,
                 interv_betw_offset_updates,
                 projection_axis                 = np.array([0, 1, 0]),
                 time_const_lowpass_filter_phase = None,
                 wrap_interv                     = "-pi_to_pi"):

        self.window_for_offset_update        = window_for_offset_update    # [s]
        self.interv_betw_offset_updates      = interv_betw_offset_updates  # [s]
        self.projection_axis                 = projection_axis  # onto which motion is projected
        self.time_const_lowpass_filter_phase = time_const_lowpass_filter_phase
        self.wrap_interv                     = wrap_interv

        self.trajectory    = []
        self.time_instants = []

        self.pos_projected        = 0
        self.pos_projected_prev   = 0
        self.vel_projected        = 0
        self.vel_projected_prev   = 0
        self.acc_projected        = 0
        self.pos_projected_offset = 0

        self.phase = 0

        self.time_last_offset_update  = None
        self.is_first_offset_computed = False

        self.amplitude_vel_p = 1
        self.amplitude_vel_n = 1
        self.amplitude_pos_n = 1
        self.amplitude_pos_p = 1

        if   self.wrap_interv == "0_to_2pi":   self.wrap_fun = wrap_to_2pi
        elif self.wrap_interv == "-pi_to_pi":  self.wrap_fun = wrap_to_pi
        else:                             raise ValueError("wrap_interv must be in {'-pi_to_pi', '0_to_2pi'}")

        if self.time_const_lowpass_filter_phase in {None, 0, -1}:
            self.is_use_lowpass_filter_phase = False
        else:  self.is_use_lowpass_filter_phase = True
        self.is_lowpass_filter_phase_initialized = False


    def estimate_phase(self, position, curr_time):
        self.time_instants.append(curr_time)
        curr_time = self.time_instants[-1]
        self.trajectory.append(position)

        if self.time_last_offset_update is None:  self.time_last_offset_update = curr_time  # this occurs on first call for estimation only

        if not self.is_first_offset_computed:
            if curr_time - self.time_last_offset_update >= self.window_for_offset_update:
                self.time_last_offset_update = curr_time
                self.update_offset()
                self.is_first_offset_computed = True
        else:
            if curr_time - self.time_last_offset_update >= self.interv_betw_offset_updates:
                self.time_last_offset_update = curr_time
                self.update_offset()

        if self.is_first_offset_computed:
            self.update_phase()

            if self.is_use_lowpass_filter_phase:
                if not self.is_lowpass_filter_phase_initialized:
                    self.lowpass_filter_phase = LowPassFilterPhase(init_state=self.phase,
                                                                   time_step=0,
                                                                   time_const=self.time_const_lowpass_filter_phase,
                                                                   wrap_interv=self.wrap_interv)
                else:
                    time_step = self.time_instants[-1] - self.time_instants[-2]
                    self.lowpass_filter_phase.change_time_step(time_step)
                    self.lowpass_filter_phase.set_time_const(self.time_const_lowpass_filter_phase)
                    self.phase = self.lowpass_filter_phase.update_state(self.phase)

        return self.phase


    def update_offset(self):
        idx = 1
        while True:
            if self.time_instants[-1] - self.time_instants[-idx] >= self.window_for_offset_update:
                break
            idx += 1

        score = (project_onto(np.array(self.trajectory)[-idx:, :], self.projection_axis)).reshape(-1)
        self.pos_projected_offset = (max(score) + min(score)) / 2

        # Update positions and velocities using new offset
        score = project_onto(np.array(self.trajectory)[-3, :], self.projection_axis)
        self.pos_projected = score - self.pos_projected_offset

        score = project_onto(np.array(self.trajectory)[-4, :], self.projection_axis)
        self.pos_projected_prev = score - self.pos_projected_offset

        sampling_time = self.time_instants[-4] - self.time_instants[-3]
        self.vel_projected_prev = (self.pos_projected - self.pos_projected_prev) / sampling_time

        score = project_onto(np.array(self.trajectory)[-2, :], self.projection_axis)
        self.pos_projected = score - self.pos_projected_offset

        score = project_onto(np.array(self.trajectory)[-3, :], self.projection_axis)
        self.pos_projected_prev = score - self.pos_projected_offset

        sampling_time = self.time_instants[-2] - self.time_instants[-3]
        self.vel_projected = (self.pos_projected - self.pos_projected_prev) / sampling_time


    def update_phase(self):
        score = project_onto(np.array(self.trajectory)[-1, :], self.projection_axis)

        self.pos_projected_prev = self.pos_projected
        self.vel_projected_prev = self.vel_projected

        sampling_time = self.time_instants[-1] - self.time_instants[-2]

        self.pos_projected = score - self.pos_projected_offset
        self.vel_projected = (self.pos_projected - self.pos_projected_prev) / sampling_time
        self.acc_projected = (self.vel_projected - self.vel_projected_prev) / sampling_time

        if self.pos_projected_prev < 0 and self.pos_projected >= 0 and self.vel_projected > 0:
            self.amplitude_vel_p = abs(self.vel_projected)
        if self.pos_projected_prev >= 0 and self.pos_projected < 0 and self.vel_projected < 0:
            self.amplitude_vel_n = abs(self.vel_projected)
        if self.vel_projected_prev >= 0 and self.vel_projected < 0 and self.acc_projected < 0:
            self.amplitude_pos_p = abs(self.pos_projected)
        if self.vel_projected_prev < 0 and self.vel_projected >= 0 and self.acc_projected > 0:
            self.amplitude_pos_n = abs(self.pos_projected)
        if self.pos_projected >= 0:
            position_normalized = self.pos_projected / self.amplitude_pos_p
        else:
            position_normalized = self.pos_projected / self.amplitude_pos_n
        if self.vel_projected >= 0:
            velocity_normalized = self.vel_projected / self.amplitude_vel_p
        else:
            velocity_normalized = self.vel_projected / self.amplitude_vel_n

        self.phase = self.wrap_fun(math.atan2(-velocity_normalized, position_normalized))


def project_onto(input_, vec_onto):
    """Projects each row of 'input_' onto vector 'vec_onto'. input_ can be a 1d array, or a matrix. Returns weights"""
    n_dims = vec_onto.shape[0]
    input_.reshape(-1, n_dims)
    vec_onto.reshape((1,-1))
    return (input_ @ vec_onto) / (vec_onto @ vec_onto.T)  # weights
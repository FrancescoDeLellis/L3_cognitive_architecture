import warnings
import numpy as np
from wrap_functions import wrap_to_pi, wrap_to_2pi


class LowPassFilter:
    def __init__(self, init_state:np.ndarray, time_step:float, time_const:float) -> None:
        self.set_curr_state(init_state)
        self.time_step = time_step
        self.set_time_const(time_const)

    def set_curr_state(self, new_state:np.ndarray) -> None:
        self.curr_state = new_state

    def set_time_const(self, new_time_const:float) -> None:
        if new_time_const < self.time_step:
            warnings.warn("new_time_const must be >= time_step. I'm setting time_const = time_step", UserWarning)
            self.time_const = self.time_step
        else:
            self.time_const = new_time_const

    def change_time_step(self, new_time_step:float) -> None:
        self.time_step = new_time_step
        self.set_time_const(self.time_const)

    def update_state(self, input_) -> np.ndarray:
        gain = self.time_step / self.time_const
        self.curr_state = (1 - gain) * self.curr_state + gain * input_
        return self.curr_state


class LowPassFilterPhase(LowPassFilter):
    def __init__(self, init_state:np.ndarray, time_step:float, time_const:float, wrap_interv:str) -> None:
        if   wrap_interv == "0_to_2pi":   self.wrap_fun = wrap_to_2pi
        elif wrap_interv == "-pi_to_pi":  self.wrap_fun = wrap_to_pi
        else:                             raise ValueError("wrap_interv must be in {'-pi_to_pi', '0_to_2pi'}")
        LowPassFilter.__init__(self, init_state, time_step, time_const)

    def set_curr_state(self, new_state:np.ndarray) -> None:
        self.curr_state = self.wrap_fun(new_state)

    def update_state(self, input_) -> np.ndarray:
        if self.curr_state - input_ > np.pi:  self.curr_state = input_  # no filtering if a jump is detected
        else:                                 LowPassFilter.update_state(self, input_)
        self.set_curr_state(self.curr_state)  # apply wrapping
        return self.curr_state


def filter_signal(signal_to_filter:np.ndarray, time_signal:np.ndarray, time_const:float) -> np.ndarray:
    signal_filtered = np.full_like(signal_to_filter, np.nan)
    signal_filtered[0, :] = signal_to_filter[0, :]
    low_pass_filter_estimand = LowPassFilter(signal_to_filter[0, :], float(time_signal[1] - time_signal[0]), time_const=time_const)
    for i_t in range(1, len(signal_to_filter)):
        low_pass_filter_estimand.change_time_step(float(time_signal[i_t] - time_signal[i_t-1]))
        signal_filtered[i_t, :] = low_pass_filter_estimand.update_state(signal_to_filter[i_t, :])
    return signal_filtered
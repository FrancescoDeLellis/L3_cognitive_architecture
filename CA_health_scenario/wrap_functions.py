import numpy as np

def wrap_to_2pi(x): return np.mod(x, 2 * np.pi)
def wrap_to_pi(x):  return np.mod(x + np.pi, 2 * np.pi) - np.pi
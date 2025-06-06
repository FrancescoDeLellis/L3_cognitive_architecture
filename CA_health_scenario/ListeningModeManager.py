import numpy as np

##################################################
# Listening mode manager class
##################################################

class ListeningModeManager:
    def __init__(self):
        self.default_value = 0
    
    def mask_none(self, phase):
        if phase == None: return 0
        else: return phase
from dataclasses import dataclass

import numpy as np

@dataclass
class AFMParameters:    
    sampling_rate:  float = 5e4
    k:              float = 0.07
    poission_ratio: float = 0.5
    bead_radias: float = 5e-6
    theta:          float = 17.5
    
    def __post_init__(self):
        self.tan_theta: float = np.tan(self.theta)
        self.t_dash:    float = 1 / self.sampling_rate

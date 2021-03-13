
############################## INCLUDE RELEVANT PYTHON MODULES ##############################

import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import interp1d as interp
from scipy.integrate import quadrature as quad

GAUSS_CONST = 1.0 / np.sqrt(2.0 * np.pi)

############################## DEFINE PARAMETERS CLASS FOR LINEAR MODEL ##############################

class Parameters:
    def __init__(self, time_steps, space_steps, total_time, upper_space_limit, sigma, rho, gauss_steps, lamb, mu_jump, sigma_jump, alpha):
        self.time_steps = time_steps
        self.space_steps = space_steps
        self.total_time = total_time
        self.upper_space_limit = upper_space_limit
        self.sigma = sigma
        self.rho = rho
        self.gauss_steps = gauss_steps
        self.lamb = lamb
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.time_delta = self.total_time / self.time_steps
        self.space_delta = self.upper_space_limit / self.space_steps
        self.alpha = alpha
        
############################## DEFINE CLASS TO RUN SIMULATION OF LINEAR MODEL ##############################
        
class LinearModel:
    def __init__(self, V0, parameters):
        BM = generate_BM(parameters.total_time, parameters.time_steps)
        shifts = [parameters.rho * parameters.sigma * (BM[i+1] - BM[i]) for i in range(parameters.time_steps)]
        jumps = generate_Poisson(parameters.total_time, parameters.time_steps, parameters.lamb)
        self.para = parameters
        self.jumps = jumps
        self.space_grid = np.array([i * self.para.space_delta for i in range(self.para.space_steps + 1)])
        self.time_grid = np.array([i * self.para.time_delta for i in range(self.para.time_steps + 1)])
        self.data = [np.array([V0(x) for x in self.space_grid]) / quad(V0,0.0,self.para.upper_space_limit)[0]]
        self.loss = [0.0]
        self.shifts = shifts
        self.jumps = jumps
        
    def simulate(self):
        for i in range(self.para.time_steps):
            f = self.data[i]
            kernel_dis = np.zeros_like(self.time_grid)
            kernel_dis[1] = 2.0 / self.para.time_delta
            kernel = interp(self.time_grid, kernel_dis, bounds_error = False, fill_value = 0.0)
            no_losses = len(self.loss)
            extended_losses = np.zeros_like(self.time_grid)
            extended_losses[:no_losses] = self.loss
            L = interp(self.time_grid, extended_losses, bounds_error = False, fill_value = 0.0)
            d_L = L(self.time_grid[i]) - L(self.time_grid[i - 1])
            cont = self.para.alpha * d_L
            f_next = push_forward(f, self.space_grid, self.para.time_delta, self.shifts[i], self.para.sigma, self.para.rho, self.para.mu_jump, self.para.sigma_jump, self.jumps, i, cont)
            self.data.append(f_next)
            self.loss.append(1.0 - integrate.trapz(f_next, self.space_grid))
            
 ############################## DEFINE HELPING FUNCTIONS FOR SPDE SIMULATION ##############################
            
def push_forward(fvals, grid, dt, shift, sigma, rho, mu_jump, sigma_jump, jumps, i, cont) :
   f = interp(grid, fvals, bounds_error=False, fill_value=0.0)  
   x0 = np.linspace(-5, 5, 400)                              
   gauss = GAUSS_CONST * np.exp( -0.5 * x0 * x0 )            
   height = np.random.normal(mu_jump, sigma_jump)
   SHIFT = shift + sigma * np.sqrt(1 - rho ** 2) * np.sqrt(dt) * x0 + height * jumps[i] - cont
   shifted_f = f(grid[:,None] - SHIFT) 
   f_conv = integrate.trapz(shifted_f * gauss, x0, axis = 1)  
   return f_conv

def generate_BM(time_horizon, no_steps):
    dt = time_horizon / no_steps
    discrete_BM = np.random.normal(0.0, np.sqrt(dt), no_steps + 1)
    discrete_BM[0] = 0.0
    discrete_BM = np.cumsum(discrete_BM)
    return discrete_BM


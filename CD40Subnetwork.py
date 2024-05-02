import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

def model(r0, t, k_r, sigma_r, mu_r, lambda_r, cdt):

    # BCL6 is downregulated and thus repression is abrogated
    CD40 = cdt(t)
    drdt = mu_r + sigma_r * ((r0**2) / (k_r**2 + r0**2)) + CD40 - (lambda_r * r0)

    return drdt

def solve_model(r0, t, k_r, sigma_r, mu_r, lambda_r, cdt): 
    return odeint(model, r0, t, args=(k_r, sigma_r, mu_r, lambda_r, cdt))

def plot_model(ax, t, sol):
    ax.plot(t, sol[:, 0], 'blue', label='IRF4', lw=2.0)
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Expression level $[C_{0}=10^{-8}]$M")
    ax.set_title("Model evolution in time")
    ax.set_xlim(0,)
    ax.set_ylim(0,)

    ax.legend(loc='best')
    ax.grid()

def plot_signals(ax, t, cdt):
    CD40 = cdt(t)
    ax.plot(t, CD40, 'blue', label='CD40')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Signal strength")
    ax.set_title("Signal evolution in time")

    ax.legend(loc='best')
    ax.grid()

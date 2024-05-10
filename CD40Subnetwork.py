import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

default_params = {
    "mu_r" : 0.1,
    "sigma_r" : 2.6,
    "k_r" : 1,
    "lambda_r" : 1
}

def model(r0, t, params):
    """Returns the ODE for CD40 subnetwork.

    Since the model includes many parameters, they were collapsed into a single dictionary argument 'params'.

    Args:
        r0 (float): initial IRF4 value
        t (np.array[float]): time steps
        params (dict): dictionary with parameters
    
    'params' dictionary contains the following entries:
        k_r: dissociation constant
        sigma_r: maximum transcription rate
        mu_r: basal production rate
        lambda_r: degradation rate
        cdt: time-dependent functions for calculating cd0(t)
    """

    # Checking validity of input parameters
    acceptable_names = [ "k_r", "sigma_r", "mu_r", "lambda_r", "cdt" ]
    
    for param in params.keys():
        assert param in acceptable_names

    # UNpack params dictionary
    k_r = params["k_r"]
    sigma_r = params["sigma_r"]
    mu_r = params["mu_r"]
    lambda_r = params["lambda_r"]
    cdt = params["cdt"]

    # BCL6 is downregulated and thus repression is abrogated
    CD40 = cdt(t)
    drdt = mu_r + sigma_r * ((r0**2) / (k_r**2 + r0**2)) + CD40 - (lambda_r * r0)

    return drdt

def solve_model(r0, t, params):
    """Solves the model in time given the set of parameters and inital values.

    Args:
        r0 (float): initial IRF4 value
        t (np.array[float]): time steps
        params (dict): dictionary with parameters

    Returns:
        np.array[np.array[float]]: solutions to the model
    """  
    return odeint(model, r0, t, args=(params,))

def plot_model(ax, t, sol):
    """Plots the model solution.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        sol (np.array[np.array[float]]): solution returned by solve_model
    """
    # Plot model solution over time
    ax.plot(t, sol[:, 0], 'blue', label='IRF4', lw=2.0)
    
    # Make the plot pretty
    ax.set_xlabel('t')
    ax.set_ylabel(r"Expression level $[C_{0}=10^{-8}]$M")
    ax.set_title("Model evolution in time")
    ax.set_xlim(0,)
    ax.set_ylim(0,)

    ax.legend(loc='best')
    ax.grid()

def plot_signals(ax, t, params):
    """Plots the CD40 signal over time, corresponding to model solution (as there
    is no repression term, the signal is only depending on crt parameter).

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        params (dict): dictionary with parameters for which the solutions were computed
    """

    cdt = params["cdt"]
    CD40 = cdt(t)
    ax.plot(t, CD40, 'blue', label='CD40')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Signal strength")
    ax.set_title("Signal evolution in time")

    ax.legend(loc='best')
    ax.grid()

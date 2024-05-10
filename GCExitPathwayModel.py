import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

default_params = {
    "mu_p" : 1e-6,
    "mu_b" : 2,
    "mu_r" : 0.1,

    "sigma_p" : 9,
    "sigma_b" : 100,
    "sigma_r" : 2.6,

    "k_p" : 1,
    "k_b" : 1,
    "k_r" : 1,

    "lambda_p" : 1,
    "lambda_b" : 1,
    "lambda_r" : 1
}

def model(y0, t, params):
    """Returns the system of ODEs corresponding to change of
    transcriptional activity in time [dpdt, dbdt, drdt].

    Since the model includes many parameters, they were collapsed into a single dictionary argument 'params'.

    Args:
        y0 (np.array[float]): initial factor values of p (BLIMP1), b (BCL6) and r (IRF4)
        t (np.array[float]): time steps
        params (dict): dictionary with parameters
    
    'params' dictionary contains the following entries:
        k_p, k_b, k_r: dissociation constants 
        sigma_p, sigma_b, sigma_r: maximum transcription rates
        mu_p, mu_b, mu_r: basal production rates
        lambda_p, lambda_b, lambda_r: degradation rates
        bcrt, cdt: time-dependent functions for calculating bcr0(t) and cd0(t)
    """

    # Checking validity of input parameters
    acceptable_names = [
            "k_p", "k_b", "k_r",
            "sigma_p", "sigma_b", "sigma_r",
            "mu_p", "mu_b", "mu_r",
            "lambda_p", "lambda_b", "lambda_r",
            "bcrt", "cdt"
        ]
    
    for param in params.keys():
        assert param in acceptable_names

    # Initial values
    p, b, r = y0

    # UNpack params dictionary
    kp, kb, kr = params["k_p"], params["k_b"], params["k_r"]
    sp, sb, sr = params["sigma_p"], params["sigma_b"], params["sigma_r"]
    mp, mb, mr = params["mu_p"], params["mu_b"], params["mu_r"]
    lp, lb, lr = params["lambda_p"], params["lambda_b"], params["lambda_r"]
    bcrt, cdt = params["bcrt"], params["cdt"]

    # Calculate unrepressed BCR and CD40 signal
    bcr0 = bcrt(t)
    cd0 = cdt(t)

    # Calculate repressed BCR and CD40 signal
    BCR = bcr0 * ((kb**2) / (kb**2 + b**2))
    CD40 = cd0 * ((kb**2) / (kb**2 + b**2))

    # ODE system
    dpdt = mp + sp * ((kb**2) / (kb**2 + b**2)) + sp * ((r**2) / (kr**2 + r**2)) - (lp * p)
    dbdt = mb + sb * ((kp**2) / (kp**2 + p**2)) * ((kb**2) / (kb**2 + b**2)) * ((kr**2) / (kr**2 + r**2)) - (lb + BCR) * b
    drdt = mr + sr * ((r**2) / (kr**2 + r**2)) + CD40 - (lr * r)
    dydt = [dpdt, dbdt, drdt]

    return dydt

def solve_model(y0, t, params):
    """Solves the model in time given the set of parameters and inital values.

    Args:
        y0 (np.array[float]): initial factor values of p (BLIMP1), b (BCL6) and r (IRF4)
        t (np.array[float]): time steps
        params (dict): dictionary with parameters

    Returns:
        np.array[np.array[float]]: solutions to the model
    """    
    return odeint(model, y0, t, args=(params,))

def plot_model(ax, t, sol):
    """Plots the model solutions.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        sol (np.array[np.array[float]]): solutions returned by solve_model
    """
    # Plot model solutions over time
    ax.plot(t, sol[:, 0], 'blue', label='BLIMP1', lw=2.0)
    ax.plot(t, sol[:, 1], 'green', label='BCL6', lw=2.0)
    ax.plot(t, sol[:, 2], 'red', label='IRF4', lw=2.0)
    
    # Make the plot pretty
    ax.set_xlabel('t')
    ax.set_ylabel(r"Expression level $[C_{0}=10^{-8}]$M")
    ax.set_title("Model evolution in time")
    ax.set_xlim(0,)
    ax.set_ylim(0,)

    ax.legend(loc='best')
    ax.grid()

def plot_signals(ax, t, params, sol):
    """Plots the CD40 and BCR signals over time, corresponding to model solutions.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        params (dict): dictionary with parameters for which the solutions were computed
        sol (np.array[np.array[float]]): solutions returned by scipy.integrate.odeint
    """
    # Unpack the parameters
    kb = params["k_b"]
    bcrt, cdt = params["bcrt"], params["cdt"]

    # Calculate input (non-repressed) signals
    bcr0 = bcrt(t)
    cd0 = cdt(t)

    # Obtain BCL6 to calculate repressed signals
    b = sol[:,1]

    # Calculate output (repressed) signals
    BCR = bcr0 * ((kb**2) / (kb**2 + b**2))
    CD40 = cd0 * ((kb**2) / (kb**2 + b**2))

    # Plotting
    ax.plot(t, bcr0, 'red', ls="--", label=r'bcr$_0$')
    ax.plot(t, BCR, 'red', label='BCR')
    ax.plot(t, cd0, 'blue', ls="--", label=r'cd$_0$')
    ax.plot(t, CD40, 'blue', label='CD40')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Signal strength")
    ax.set_title("Signal evolution in time")

    ax.legend(loc='best')
    ax.grid()

def bell_curve_signal(t, strength, tpeak, scale):
    """Generates a bell curve shaped signal.

    Args:
        t (np.array[float]): time steps
        strength (float): signal value at the peak
        tpeak (float): location parameter of normal distribution
        scale (float): scale parameter of normal distribution

    Returns:
        np.array[float]: signal value at the given time steps
    """    
    return norm.pdf(t, loc=tpeak, scale=scale) / norm.pdf(tpeak, loc=tpeak, scale=scale) * strength

def rectangle_signal(t, strength, tstart, tend):
    """Generates a rectangle shaped signal.

    Args:
        t (np.array[float]): time steps
        strength (float): signal value at the peak
        tstart (float): time when the signal starts
        tend (float): time when the signal end

    Returns:
        np.array[float]: signal value at the given time steps
    """
    return (np.heaviside(t - tstart, 1) - np.heaviside(t - tend, 1)) * strength
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

def model(y0, t, k, sigma, mu, lam, bcrt, cdt):
    """Returns the system of ODEs corresponding to change of
    transcriptional activity in time [dpdt, dbdt, drdt].

    Args:
        y0 (np.array[float]): initial factor values of p (BLIMP1), b (BCL6) and r (IRF4)
        t (np.array[float]): time steps
        k (np.array[float]): dissociation constants 
        sigma (np.array[float]): maximum transcription rates
        mu (np.array[float]): basal production rates
        lam (np.array[float]): degradation rates
        bcrt ((float) -> (float)): time-dependent function for calculating bcr0(t)
        cdt ((float) -> (float)): time-dependent function for calculating cd0(t)
    """

    p, b, r, x = y0
    kp, kb, kr, kx = k
    sp, sb_1, sb_2, sr, sx = sigma
    mp, mb, mr, mx = mu
    lp, lb, lr, lx = lam

    bcr0 = bcrt(t)
    cd0 = cdt(t)

    BCR = bcr0 * ((kb**2) / (kb**2 + b**2))
    CD40 = cd0 * ((kb**2) / (kb**2 + b**2))

    dpdt = mp + sp * ((kb**2) / (kb**2 + b**2)) + sp * ((r**2) / (kr**2 + r**2)) - (lp * p)
    dbdt = mb + sb_1  * ((kx**2) / (kx**2 + x**2)) + sb_2 * ((kp**2) / (kp**2 + p**2)) * ((kb**2) / (kb**2 + b**2)) * ((kr**2) / (kr**2 + r**2)) - (lb + BCR) * b
    drdt = mr + sr * ((r**2) / (kr**2 + r**2)) + CD40 - (lr * r)
    dxdt = mx + sx * ((kp**2) / (kp**2 + p**2)) - (lx * x)

    dydt = [dpdt, dbdt, drdt, dxdt]

    return dydt

def solve_model(y0, t, k, sigma, mu, lam, bcrt, cdt):
    """Solves the model in time given the set of parameters and inital values.

    Args:
        y0 (np.array[float]): initial values
        t (np.array[float]): time steps
        k (np.array[float]): dissociation constants 
        sigma (np.array[float]): maximum transcription rates
        mu (np.array[float]): basal production rates
        lam (np.array[float]): degradation rates
        bcrt ((float) -> (float)): time-dependent function for calculating bcr0(t)
        cdt ((float) -> (float)): time-dependent function for calculating cd0(t)

    Returns:
        np.array[np.array[float]]: solutions to the model
    """    
    return odeint(model, y0, t, args=(k, sigma, mu, lam, bcrt, cdt))

def plot_model(ax, t, sol):
    """Plots the model solutions.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        sol (np.array[np.array[float]]): solutions returned by solve_model
    """    
    ax.plot(t, sol[:, 0], 'blue', label='BLIMP1', lw=2.0)
    ax.plot(t, sol[:, 1], 'green', label='BCL6', lw=2.0)
    ax.plot(t, sol[:, 2], 'red', label='IRF4', lw=2.0)
    ax.plot(t, sol[:, 3], 'cyan', label='PAX5', lw=2.0)
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Expression level $[C_{0}=10^{-8}]$M")
    ax.set_title("Model evolution in time")
    ax.set_xlim(0,)
    ax.set_ylim(0,)

    ax.legend(loc='best')
    ax.grid()

def plot_singals(ax, t, sol, kb, bcrt, cdt):
    """Plots the CD40 and BCR signals over time, corresponding to model solutions.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        sol (np.array[np.array[float]]): =solutions returned by scipy.integrate.odeint
        kb (float): BCL6 dissociation constant used in solution
        bcrt ((float) -> (float)): time-dependent function for calculating bcr0(t)
        cdt ((float) -> (float)): time-dependent function for calculating cd0(t)
    """
    bcr0 = bcrt(t)
    cd0 = cdt(t)

    b = sol[:, 1]
    BCR = bcr0 * ((kb**2) / (kb**2 + b**2))
    CD40 = cd0 * ((kb**2) / (kb**2 + b**2))

    ax.plot(t, bcr0, 'red', ls="--", label=r'bcr$_0$')
    ax.plot(t, BCR, 'red', label='BCR')
    ax.plot(t, cd0, 'blue', ls="--", label=r'cd$_0$')
    ax.plot(t, CD40, 'blue', label='CD40')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Signal strength")
    ax.set_title("Signal evolution in time")

    ax.legend(loc='best')
    ax.grid()

def bell_curve_signal(t, strength, loc, scale):
    """Generates a bell curve shaped signal.

    Args:
        t (np.array[float]): time steps
        strength (float): signal value at the peak
        loc (float): location parameter of normal distribution
        scale (float): scale parameter of normal distribution

    Returns:
        np.array[float]: signal value at the given time steps
    """    
    return norm.pdf(t, loc=loc, scale=scale) / norm.pdf(loc, loc=loc, scale=scale) * strength

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
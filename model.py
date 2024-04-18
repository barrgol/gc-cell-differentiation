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

    p, b, r = y0
    kp, kb, kr = k
    sp, sb, sr = sigma
    mp, mb, mr = mu
    lp, lb, lr = lam

    bcr0 = bcrt(t)
    cd0 = cdt(t)

    BCR = bcr0 * ((kb**2) / (kb**2 + b**2))
    CD40 = cd0 * ((kb**2) / (kb**2 + b**2))

    dpdt = mp + sp * ((kb**2) / (kb**2 + b**2)) + sp * ((r**2) / (kr**2 + r**2)) - (lp * p)
    dbdt = mb + sb * ((kp**2) / (kp**2 + p**2)) * ((kb**2) / (kb**2 + b**2)) * ((kr**2) / (kr**2 + r**2)) - (lb + BCR) * b
    drdt = mr + sr * ((r**2) / (kr**2 + r**2)) + CD40 - (lr * r)
    dydt = [dpdt, dbdt, drdt]

    return dydt

def solve_model(y0, t, k, sigma, mu, lam, bcr, cd40):
    """Solves the model in time given the set of parameters and inital values.

    Args:
        y0 (np.array[float]): initial values
        t (np.array[float]): time steps
        k (np.array[float]): dissociation constants 
        sigma (np.array[float]): maximum transcription rates
        mu (np.array[float]): basal production rates
        lam (np.array[float]): degradation rates
        bcr (tuple[float]): triplet (peak, mu, sigma) for BCR bell curve signal with a certain peak value
        cd40 (tuple[float]): pair (peak, mu, sigma) for CD40 bell curve signal with a certain peak value

    Returns:
        np.array[np.array[float]]: solutions to the model
    """    
    return odeint(model, y0, t, args=(k, sigma, mu, lam, bcr, cd40))

def plot_model(ax, t, sol):
    """Plots the model solutions.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib's ax to plot the model to
        t (np.array[float]): time steps
        sol (np.array[np.array[float]]): solutions returned by scipy.integrate.odeint
    """    
    ax.plot(t, sol[:, 0], 'blue', label='BLIMP1')
    ax.plot(t, sol[:, 1], 'green', label='BCL6')
    ax.plot(t, sol[:, 2], 'red', label='IRF4')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Expression level $[C_{0}=10^{-8}]$M")
    ax.set_title("Model evolution in time")

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

    ax.plot(t, bcr0, 'red', ls="--", label='bcr0')
    ax.plot(t, BCR, 'red', label='BCR')
    ax.plot(t, cd0, 'blue', ls="--", label='cd0')
    ax.plot(t, CD40, 'blue', label='CD40')
    
    ax.set_xlabel('t')
    ax.set_ylabel(r"Signal strength")
    ax.set_title("Signal evolution in time")

    ax.legend(loc='best')
    ax.grid()

def bell_curve_signal(t, strength, loc, scale):
    return norm.pdf(t, loc=loc, scale=scale) / norm.pdf(loc, loc=loc, scale=scale) * strength

def heaviside_signal(t, strength, tstart, tend):
    return (np.heaviside(t - tstart, 1) - np.heaviside(t - tend, 1)) * strength

# Model parameters as given in table S1 of the Martinez paper
mu_p = 1e-6
mu_b = 2
mu_r = 0.1

sigma_p = 9
sigma_b = 100
sigma_r = 2.6

k_p = 1
k_b = 1
k_r = 1

lam_p = 1
lam_b = 1
lam_r = 1

bcrt = lambda t : heaviside_signal(t, strength=1.35, tstart=25, tend=55)
cdt = lambda t : heaviside_signal(t, strength=0.75, tstart=40, tend=60)

mu = np.array([mu_p, mu_b, mu_r])
sigma = np.array([sigma_p, sigma_b, sigma_r])
k = np.array([k_p, k_b, k_r])
lam = np.array([lam_p, lam_b, lam_r])

# Time steps
t = np.linspace(0, 100, 10000)

# Initial conditions
p0 = 0.0  # BLIMP1
b0 = 4.5  # BCL6
r0 = 0.0  # IRF4

y0 = np.array([p0, b0, r0])

# Solutions and plotting
sol = solve_model(y0, t, k, sigma, mu, lam, bcrt, cdt)

fig, ax = plt.subplots(1,2,figsize=(15,4))
plot_model(ax[0], t, sol)
plot_singals(ax[1], t, sol, k_b, bcrt, cdt)
plt.show()

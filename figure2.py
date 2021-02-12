import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from plot_settings import *


#define the parameters from table 1
secs_in_year = 365*24*60*60.0
Q10 = 2.5
alpha = 0.1 * np.log(Q10)
Lam = 10.0 #lambda
H = 0.4
A = 3.9e7
npp = 0.5 / secs_in_year
nppc = Lam/(alpha * A)
mu = 1.0e6

N = 100 #number of equally spaced vertical levels, used when descritizing

def respiration(T):
    # Calculate the respiration occuring for input values of temperature, T, given
    # the global variables npp,nppc,alpha.
    return npp * np.exp(-npp/nppc) * np.exp(alpha * T)


# for given values of kappa and air temperature (warming)
# determine if a solution to the PDE exists, given an initial profile T0 returns True or False
def is_sol(kappa,warming,T0):
    sol = scipy.integrate.solve_ivp(continuum_kappa_critical,
                                    (0.0,20.0*secs_in_year),
                                    T0,
                                    method="BDF",
                                    args=(warming,kappa))
    return sol.success


# find the critical warming for a given value of kappa.
# Starting with an initial guess for the critical warming (upper_guess)
# we keep doublign it until we find an unstable value.
# By comparing with a lower guess we know is stable, we repeatedly bisect the interval until we reach
# the unstable warming
def find_warming(kappa):
    print(f"Trying kappa = {kappa}")
    #begin in equilibrium
    equil = scipy.integrate.solve_ivp(continuum_kappa_critical,
                                      (0.0,20.0*secs_in_year),
                                      np.zeros(N),
                                      method="BDF",
                                      args=(0.0,kappa))
    T0 = equil.y[:,-1]
    lower_guess = 0.0
    upper_guess = 1.0
    while True:
        if not is_sol(kappa,upper_guess,T0):
            break
        else:
            upper_guess *= 2
            if upper_guess >= 10000:
                return np.inf
    while True:
        mid_guess = 0.5 * (upper_guess + lower_guess)
        if is_sol(kappa,mid_guess,T0):
            lower_guess = mid_guess
        else:
            upper_guess = mid_guess
        if (upper_guess - lower_guess) < 0.1:
            return mid_guess


# move through an array of kappas and find the corresponding critical warming
def scan(kappas):
    return np.asarray([find_warming(kappa) for kappa in kappas])

def continuum_kappa(t,y,amplitude,kappa):
    # see documentation in figure1.py
    delta = 10*H / N
    tau_t = mu / kappa
    B = 1 / (tau_t * delta**2)
    main_diag = -2 * np.ones(N)
    upper_diag = np.ones(N-1)
    lower_diag = np.ones(N-1)

    main_diag[0] = -2 * (delta *Lam/kappa + 1.0)
    upper_diag[0] = 2
    lower_diag[-1] = 2.0

    L = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)
    T = y.reshape((N,1))
    b = np.zeros_like(T)
    b[0,0] = 2 * delta * Lam/kappa * B * amplitude * np.sin(2*np.pi*t/secs_in_year)

    return (B * np.dot(L,T) + b + A/H*respiration(T)*np.exp(np.linspace(0.0,-10*H,N).reshape((N,1)) / H) /mu).flatten()

def continuum_kappa_critical(t,y,warming,kappa):
    # see documentation in figure1.py, except rather than time dependent forcing, the air temperature is held constant at `warming'
    delta = 10*H / N
    tau_t = mu / kappa
    B = 1 / (tau_t * delta**2)
    main_diag = -2 * np.ones(N)
    upper_diag = np.ones(N-1)
    lower_diag = np.ones(N-1)

    main_diag[0] = -2 * (delta *Lam/kappa + 1.0)
    upper_diag[0] = 2
    lower_diag[-1] = 2.0

    L = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)
    T = y.reshape((N,1))
    b = np.zeros_like(T)
    b[0,0] = 2 * delta * Lam/kappa * B * warming

    return (B * np.dot(L,T) + b + A/H*respiration(T)*np.exp(np.linspace(0.0,-10*H,N).reshape((N,1)) / H) /mu).flatten()

def is_sol_cont(T0,kappa,amplitude):
    #like is_sol, except the air temperature varies sinusoidally
    sol = scipy.integrate.solve_ivp(continuum_kappa,
                                    (0.0,20.0*365*24*60*60.0),
                                    T0,
                                    method="BDF",
                                    args=(amplitude,kappa))
    return sol.success

def find_continuum_unstable_amplitude(kappa):
    #like find_warming except now the air temperature varies with the seasonal cycle
    equil = scipy.integrate.solve_ivp(continuum_kappa,
                                      (0.0,20.0*365*24*60*60.0),
                                      np.zeros(N),
                                      method="BDF",
                                      args=(0.0,kappa))
    T0 = equil.y[:,-1]
    upper_guess = 1.0
    lower_guess = 0.0

    while True:
        print(kappa,upper_guess)
        if not is_sol_cont(T0,kappa,upper_guess):
            break
        else:
            upper_guess *= 2
            if upper_guess >= 10000:
                return np.inf
    while True:
        mid_guess = 0.5 * (upper_guess + lower_guess)
        print(kappa,mid_guess)
        if is_sol_cont(T0,kappa,mid_guess):
            lower_guess = mid_guess
        else:
            upper_guess = mid_guess
        if (upper_guess - lower_guess) < 0.1:
            guess = np.linspace(lower_guess - 1.0,upper_guess+1.0,100)
            return mid_guess




#geometrically spaced kappas to test
kappas = np.geomspace(0.1,500.0,100)
#find the critical air temperature for compost bomb as function of kappa
warmings = scan(kappas)
#find critical amplitude for seasonal cycle as function of kappa
seasonal_cycle = np.asarray([find_continuum_unstable_amplitude(kappa) for kappa in kappas])
#plot these
plt.plot(kappas,warmings,color="black",label="Continuum Model")
plt.plot(kappas,seasonal_cycle,color="red",label="Seasonal Cycle")
#plot on the critical warming for the LC10 model.
plt.axhline(1/alpha * (npp/nppc - np.log(npp/nppc) -1),color="black",linestyle="--",label="LC10")
plt.xscale("log")
plt.xlabel(r"$\kappa$ $\mathrm{Wm^{-1}} ^\circ\mathrm{C}$")
plt.ylabel(r"Critical Warming $^\circ\mathrm{C}$")
plt.legend()
plt.xlim(kappas.min(),kappas.max())
plt.tight_layout()
plt.savefig("dimensional_continuum_vs_lc10.pdf")
plt.close()


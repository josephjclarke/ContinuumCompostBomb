import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np

from plot_settings import *

secs_in_year = 365*24*60*60.0


N = 100
q10 = 2.5
kappa = 0.16
H = 0.5
N = 100
Lam = 10.0
A = 3.9e7
secs_in_year = 365*24*60*60.0
alpha = 0.1 * np.log(q10)
npp = 0.5 / secs_in_year
nppc = Lam/(alpha * A)


#see figure1.py
def respiration(T):
    return npp * np.exp(-npp/nppc) * np.exp(alpha * T)


#see figure1.py, but now we also let the frequency of Ta forcing vary
def continuum_kappa(t,y,amplitude,kappa,omega):
    mu = 1.0e6
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
    b[0,0] = 2 * delta * Lam/kappa * B * amplitude * np.sin(omega*t)

    return (B * np.dot(L,T) + b + A/H*respiration(T)*np.exp(np.linspace(0.0,-10*H,N).reshape((N,1)) / H) /mu).flatten()



#initialise in equilibrium
equil = scipy.integrate.solve_ivp(continuum_kappa,
                                      (0.0,20.0*secs_in_year),
                                      np.zeros(N),
                                      method="BDF",
                                      args=(0.0,kappa,0.0))
T0 = equil.y[:,-1]
z = np.linspace(0.0,-10*H,N)

#see figure2.py
def is_stable(amplitude,omega):
    return scipy.integrate.solve_ivp(continuum_kappa,
                                     (0.0,20.0*2*np.pi/omega),
                                     T0,
                                     method="BDF",
                                     args=(amplitude,kappa,omega)).success

#see figure2.py
def find_unstable_amplitude(omega):
    upper_guess = 1.0
    lower_guess = 0.0
    while True:
        print(omega,upper_guess)
        if not is_stable(upper_guess,omega):
            break
        else:
            upper_guess *= 2
            if upper_guess >= 10000:
                return np.inf
    while True:
        mid_guess = 0.5 * (upper_guess + lower_guess)
        print(omega,mid_guess)
        if is_stable(mid_guess,omega):
            lower_guess = mid_guess
        else:
            upper_guess = mid_guess
        if (upper_guess - lower_guess) < 0.1:
            guess = np.linspace(lower_guess - 1.0,upper_guess+1.0,100)
            return mid_guess


#scan over frequency range
omegas = np.geomspace(2*np.pi/(5.0*secs_in_year),2*np.pi/(24*60*60.0))
#work out critical amplitudes
amps = np.asarray([find_unstable_amplitude(omega) for omega in omegas])
fig,ax1 = plt.subplots()
ax1.set_xlabel(r"Forcing Period ($\mathrm{yr}$)")
ax1.set_ylabel(r"Critical Amplitude ($^\circ\mathrm{C}$)")
ax1.plot(2*np.pi/omegas / secs_in_year,amps,color="black")
ax1.set_xscale("log")
ax1.set_xlim(2*np.pi/omegas[-1]/secs_in_year,5)
plt.tight_layout()
plt.savefig("critical_amplitude_vs_period.pdf")
plt.close()

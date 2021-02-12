import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np

from plot_settings import *

secs_in_year = 365*24*60*60.0
#approximate vale of nondimensional frequency corresponding to the seasonal_cycle
ndss = 2 * np.pi * 0.5 * 1.0e6 / 100.0 / secs_in_year  


#like continuum_kappa from figure1.py, but now npp, D,N, cycle amplitude
#and frequency can be controlled. We are also working nondimensionally
def f_sine(t,theta,npp,D,N,amplitude,omega):
    W = npp*np.exp(-npp)/D
    depth = D * 10.0
    dz = depth / N

    main_diag = -2 * np.ones(N)
    main_diag[0] = -2 * (dz + 1.0)

    upper_diag = np.ones(N - 1)
    upper_diag[0] = 2

    lower_diag = np.ones(N - 1)
    lower_diag[-1] = 2.0

    A = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)

    T = theta.reshape((N,1))
    b = np.zeros_like(T)

    density = np.exp(np.linspace(0.0,-depth,N) / D).reshape((N,1))

    b[0,0] = 2 * (amplitude * np.sin(omega * t)) / dz
    return (np.dot(A,T) + b + W * np.exp(T)*density ).flatten()


N = 100

npps = np.linspace(0.1,0.5,20) / 100.0
Ds = np.linspace(4/3.0,10.0,20)


#see figure2.py
def is_stable(amplitude,npp,D):
    sol = scipy.integrate.solve_ivp(f_sine,(0.0,20.0*1000.0),np.zeros(N),method="BDF",args=(npp,D,N,0.0,0.0))
    equil = sol.y[:,-1]
    return scipy.integrate.solve_ivp(
        f_sine,
        (0.0,20.0/ndss),
        equil,
        method="BDF",
        args=(npp,D,N,amplitude,ndss)).success

#see figure2.py
def find_unstable_amplitude(D,npp):
    print(f"Trying D,npp = {D},{npp}")
    lower_guess = 0.0
    upper_guess = 1.0
    while True:
        if not is_stable(upper_guess,npp,D):
            break
        else:
            upper_guess *= 2
            if upper_guess >= 10000:
                return np.inf
    while True:
        mid_guess = 0.5 * (upper_guess + lower_guess)
        if is_stable(mid_guess,npp,D):
            lower_guess = mid_guess
        else:
            upper_guess = mid_guess
        if (upper_guess - lower_guess) < 0.1:
            return mid_guess


#calculate the critical seasonal cycle
crit_thetas = np.empty((npps.size,Ds.size))
for idx,_ in np.ndenumerate(crit_thetas):
    crit_thetas[idx] = find_unstable_amplitude(Ds[idx[1]],npps[idx[0]])


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.text(0.05, 0.925, "a", fontsize=14, transform=plt.gcf().transFigure)
#plot nondimensional values
plt.contourf(npps,Ds,crit_thetas.T,cmap="YlGnBu",levels=np.arange(0.0,6.25,0.25))
plt.xlabel(r"$\widetilde{\Pi}$")
plt.ylabel(r"$D$")
cbar = plt.colorbar(ticks=np.arange(0.0,6.25,1.0))
cbar.ax.set_ylabel(r"Critical Amplitude $\Delta\theta_a$")
plt.subplot(122)
#plot dimensional values
plt.text(0.55, 0.925, "b", fontsize=14, transform=plt.gcf().transFigure)
alpha = 0.1 * np.log(2.5)
pic = 10.0 /(alpha * 3.9e7) * 365*24*60*60.0
plt.contourf(pic*npps,0.5*Ds/10.0,crit_thetas.T/alpha,cmap="YlGnBu",levels=np.arange(0.0,65.0,3.0))
plt.xlabel(r"NPP ($\mathrm{kgC}\mathrm{m}^{-2}\mathrm{yr}^{-1}$)")
plt.ylabel(r"$H$ ($\mathrm{m}$)")
cbar = plt.colorbar(ticks=np.arange(0.0,65.0,6.0))
cbar.ax.set_ylabel(r"Critical Amplitude $^\circ\mathrm{C}$")
plt.tight_layout()
plt.savefig("seasonal_dim_and_nondim.pdf")
plt.close()

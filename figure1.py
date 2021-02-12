import matplotlib.pyplot as plt

from plot_settings import *

import numpy as np
import scipy.integrate


secs_in_year = 365*24*60*60.0


#define the parameters from table 1
q10 = 2.5
kappa = 0.16
H = 0.4
Lam = 10.0 #lambda
A = 3.9e7
npp = 0.5 / secs_in_year
mu = 1.0e6 

alpha = 0.1 * np.log(q10)
nppc = Lam/(alpha * A) #critical npp


N = 100 #number of equally spaced vertical levels, used when descritizing

def respiration(T):
    # Calculate the respiration occuring for input values of temperature, T, given
    # the global variables npp,nppc,alpha.
    return npp * np.exp(-npp/nppc) * np.exp(alpha * T)


# We convert the PDE into an ODE of the form y' = f(y,t,params)
# which can then be integrated. The function `continuum_kappa'  is that function
# f, in the case where kappa is an inputted soil heat conductivity and the air
# temperature varies sinusoidally with period one year and amplitude amplitude.
def continuum_kappa(t,y,amplitude,kappa):
    # We solve the equation on 100 equally spaced levels, over a domain 10 times the length of the characteristic soil depth.
    # This length was chosen because deep in the soil the temperature profile becomes `flat' and therefore making the soil deeper has
    # little impact on the compost bomb effect

    delta = 10*H / N #thickness of one level

    #define some parameters
    tau_t = mu / kappa 
    B = 1 / (tau_t * delta**2)

    #construct the matrix which represents the second derivative 
    main_diag = -2 * np.ones(N)
    upper_diag = np.ones(N-1)
    lower_diag = np.ones(N-1)

    #Take into account boundary conditions:
    main_diag[0] = -2 * (delta *Lam/kappa + 1.0) 
    upper_diag[0] = 2
    lower_diag[-1] = 2.0

    #Second derivative matrix:
    L = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)

    #Temperature should have shape of column vector
    T = y.reshape((N,1))

    #b is a vector which applies the seasonal cycle forcing
    b = np.zeros_like(T)
    b[0,0] = 2 * delta * Lam/kappa * B * amplitude * np.sin(2*np.pi *t/secs_in_year)

    #The equation has been discretized to the form
    # T' = (BL)T + b + (A/(mu H))(respiration)(soil carbon vertical profile)
    
    return (B * np.dot(L,T) + b + A/H*respiration(T)*np.exp(np.linspace(0.0,-10*H,N).reshape((N,1)) / H) /mu).flatten()



# Find an equilibrium profile by integrating for 20 years (long compared to soil thermal timescale)
# without atmospheric forcing
equil = scipy.integrate.solve_ivp(continuum_kappa,
                                      (0.0,20.0*secs_in_year),
                                      np.zeros(N),
                                      method="BDF",
                                      args=(0.0,kappa))

# Spun up initial equilibrium temperature:
T0 = equil.y[:,-1]
# Vertical coordinate:
z = np.linspace(0.0,-10*H,N)

#Seasonal Cycle amplitude
amplitude = 32.5
#integrate for one year
sol = scipy.integrate.solve_ivp(continuum_kappa,
                                    (0.0, secs_in_year),
                                    T0,
                                    method="BDF",
                                    args=(amplitude,kappa))


#define the colours used in the plot
colors = ["#a6cee3","#1f78b4","#b2df8a","#33a02c"]
labels=["Initial","1 Months","3 Months","5 Months"]

#plot the profiles, #44,52,-1 correspond to 1 month,3 months, five months 
for i,c,l in zip([0,44,52,-1],colors,labels):
    T = sol.y[:,i] if i != 0 else T0 # make sure initial plot is of the initial equilibrium
    plt.plot(T,z,label=l,linewidth=3,color=c)

plt.xlabel(r"Soil Temperature ($^\circ\mathrm{C}$)")
plt.ylabel(r"Depth ($\mathrm{m}$)")
plt.legend()
plt.tight_layout()
plt.savefig("seasonal_cycle_profiles.pdf")
plt.close()

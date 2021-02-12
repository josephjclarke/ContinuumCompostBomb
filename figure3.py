import matplotlib.pyplot as plt
import numpy as np

from plot_settings import *


#nondimensional parameters
npps = np.linspace(0.1,0.5) / 100.0
Ds = np.linspace(4/3.0,100.0)

#find c2 as defined in eq 13
def find_constant(W,D,warming):
    return scipy.optimize.root(lambda c: 1/D * np.tanh(c/2) -2 * np.log(1/np.cosh(c/2)) +
                                       1/D + np.log(2*W*D**2) + warming,x0=0.0).x.item()
# calculate theta using eq 12
def profile(x,W,D,warming):
    c = find_constant(W,D,warming)
    return np.log(1/(2*W*D**2)) + 2 * np.log(1/np.cosh(c/2 + x/(2*D))) - x/D

# calculate theta_a^crit using eq 14
def critical_theta_a_with_carbon(W, D):
    p1 = np.log(2 * D * np.sqrt(D**2 + 1) - 2 * D**2)
    p2 = -1 + 1 / D * np.sqrt(D**2 + 1) - 1 / D
    p3 = -np.log(2 * W * D**2)
    return p1 + p2 + p3

# calculate theta_a^crit using eq 14 using npp rather than W
def critical_theta(npp,D):
    return critical_theta_a_with_carbon(npp * np.exp(-npp)/D, D)


#calculate theta_a^crit
crit_thetas = critical_theta(npps[:,np.newaxis],Ds[np.newaxis,:])
crit_thetas[np.where(crit_thetas < 0)] = 0.0


#plot nondimensional and dimensional
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.text(0.05, 0.925, "a", fontsize=14, transform=plt.gcf().transFigure,fontweight='bold')
#plt.title("Nondimensional Parameters")
plt.contourf(npps,Ds,crit_thetas,cmap="YlGnBu",levels=np.arange(0.0,6.0,0.25))
plt.xlabel(r"$\widetilde{\Pi}$")
plt.ylabel(r"$D$")
cbar = plt.colorbar(ticks=np.arange(0.0,6.0,0.5))
cbar.ax.set_ylabel(r"Critical Warming $\theta_a$")
plt.subplot(122)
#plt.title("Dimensional Parameters")
plt.text(0.55, 0.925, "b", fontsize=14, transform=plt.gcf().transFigure,fontweight='bold')
alpha = 0.1 * np.log(2.5)
pic = 10.0 /(alpha * 3.9e7) * 365*24*60*60.0
plt.contourf(pic*npps,0.16*Ds/10.0,crit_thetas/alpha,cmap="YlGnBu",levels=np.arange(0.0,65.0,2.5))
plt.xlabel(r"NPP ($\mathrm{kgC}\mathrm{m}^{-2}\mathrm{yr}^{-1}$)")
plt.ylabel(r"$H$ ($\mathrm{m}$)")
cbar = plt.colorbar(ticks=np.arange(0.0,65.0,5.0))
cbar.ax.set_ylabel(r"Critical Warming $^\circ\mathrm{C}$")
plt.tight_layout()
plt.savefig("static_dim_and_nondim.pdf")
plt.close()

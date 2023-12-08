import numpy as np
from matplotlib import pyplot as plt


UNITS = "cm-1" # "meV" or "cm-1"

TIME = np.loadtxt("Dipole.dat")[:,0] # fs
MU   = np.loadtxt("Dipole.dat")[:,1:] # Debye
dt   = TIME[1] - TIME[0]
T      = TIME[-1] - TIME[0]

f_w  = np.fft.fft( MU[:,-1], n=2**15, norm='ortho' )
w    = np.fft.fftfreq( len(f_w) ) * (2*np.pi)/dt

if ( UNITS == "cm-1" ):
    w   *= 33356.683011 # fs/cm^-1
    #w   *= (1/41.341) * 220000 #* 2 * np.pi
    plt.plot( w, np.abs(f_w) )
    plt.xlim(0,4000)
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
elif ( UNITS == "meV" ):
    w   *= 4.135668*1000 # fs/meV
    plt.plot( w, np.abs(f_w) )
    plt.xlim(0,500)
    plt.xlabel("Energy (meV)", fontsize=15)
else:
    print("Warning. UNITS not recognized.")
    exit()

plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
plt.savefig("IR_SPEC.jpg", dpi=300)
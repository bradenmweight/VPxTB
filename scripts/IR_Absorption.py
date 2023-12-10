import numpy as np
from matplotlib import pyplot as plt

from emcee.autocorr import function_1d as get_ACF

UNITS = "cm-1" # "meV" or "cm-1"

TIME = np.loadtxt("Dipole.dat")[:,0] # fs
#MU   = np.loadtxt("Dipole.dat")[:,-1] # Debye
MU   = np.sum(np.loadtxt("Dipole.dat")[:,1:],axis=-1)/np.sqrt(3) # Debye
dt   = TIME[1]  - TIME[0]
T    = TIME[-1] - TIME[0]


# PLOT DIPOLE TIME SERIES
plt.plot( TIME, MU )
plt.xlabel("Time (fs)", fontsize=15)
plt.ylabel("Dipole (Debye)", fontsize=15)
plt.xlim( 0, 500 )
plt.tight_layout()
plt.savefig("Dipole.jpg", dpi=300)
plt.clf()

# GET AUTO-CORRELATION FUNCTION OF DIPOLE
MU = get_ACF( MU )
plt.plot( TIME, MU )
plt.xlabel("Lag Time (fs)", fontsize=15)
plt.ylabel("Auto-correlation Function", fontsize=15)
plt.xlim( 0, 500 )
plt.tight_layout()
plt.savefig("Dipole_ACF.jpg", dpi=300)
plt.clf()

#### NUMERICAL FOURIER KERNEL ####
Nw       = 401
wmin     = 0.    # 0.    # meV
wmax     = 400   # 400.  # meV
MU_w     = np.zeros( (Nw), dtype=complex )
w_meV    = np.linspace(wmin,wmax,Nw)
dw       = w_meV[1] - w_meV[0]

TIME_AU = TIME * 41.341 # au/fs
w_AU    = w_meV / 1000 / 27.2114 # meV --> au energy
cos_fac = np.cos( np.pi * TIME_AU / (2 * np.max(TIME_AU)) ) # Smooth Function
FUNC    = MU * cos_fac
for wi in range(Nw):
    if ( wi%500==0 ):
        print(w_meV[wi], "meV")
    exp_fact = np.exp( 1j * w_AU[wi] * TIME_AU )
    int_func = exp_fact * FUNC
    MU_w[wi] = 2*np.sum(int_func)
MU_w *= dw / np.sqrt(2 * np.pi)


if ( UNITS == "cm-1" ):
    w_cm = w_meV * 8.065610 # cm^-1 / meV
    #plt.plot( w_cm, np.abs(np.real(MU_w)), "-", c='black', label="RE" )
    #plt.plot( w_cm, np.abs(np.imag(MU_w)), "-", c='red',   label="IM" )
    plt.plot( w_cm, np.real(MU_w), "-", c='black', label="RE" )
    plt.plot( w_cm, np.imag(MU_w), "-", c='red',   label="IM" )
    plt.xlim(w_cm[0],w_cm[-1])
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
elif ( UNITS == "meV" ):
    plt.plot( w_meV, np.abs(np.real(MU_w)), "-",  c='black', label="RE" )
    plt.plot( w_meV, np.abs(np.imag(MU_w)), "-",  c='red',   label="IM" )
    plt.xlim(w_meV[0],w_meV[-1])
    plt.xlabel("Energy (meV)", fontsize=15)
else:
    print("Warning. UNITS not recognized.")
    exit()

plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig("IR_SPEC.jpg", dpi=300)
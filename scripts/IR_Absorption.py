import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

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
#plt.xlim( 0, 2000 )
plt.xlim( 0 )
plt.tight_layout()
plt.savefig("Dipole_ACF.jpg", dpi=300)
plt.clf()

#### NUMERICAL FOURIER KERNEL ####
Nw       = 10001
wmin     = 2000 / 8.065610 # 0 # 310.    # 0.    # meV
wmax     = 3000 / 8.065610 # 400 # 325   # 400.  # meV
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

# FIND THE PEAKS
THRESHOLD = np.max( np.real(MU_w) ) / 50
PEAKS, _  = find_peaks(np.real(MU_w), height=THRESHOLD)
w_cm = w_meV * 8.065610 # cm^-1 / meV
np.savetxt( f"IR_SPEC.dat", np.c_[w_meV, w_cm, np.real(MU_w)], fmt="%1.5f", header="w(meV)    w(cm-1)    IR" )
np.savetxt( f"IR_SPEC_PEAKS.dat", np.c_[w_meV[PEAKS], w_cm[PEAKS], np.real(MU_w)[PEAKS] ], fmt="%1.5f", header="w_max(meV)    w_max(cm-1)    IR_max" )


if ( UNITS == "cm-1" ):
    w_cm = w_meV * 8.065610 # cm^-1 / meV
    plt.plot( w_cm, np.real(MU_w), "-", c='black', label="RE" )
    #plt.plot( w_cm, np.imag(MU_w), "-", c='red',   label="IM" )
    plt.xlim(w_cm[0],w_cm[-1])
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
elif ( UNITS == "meV" ):
    plt.plot( w_meV, np.real(MU_w), "-",  c='black', label="RE" )
    #plt.plot( w_meV, np.imag(MU_w), "-",  c='red',   label="IM" )
    plt.xlim(w_meV[0],w_meV[-1])
    plt.xlabel("Energy (meV)", fontsize=15)
else:
    print("Warning. UNITS not recognized.")
    exit()

plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
#plt.legend()
plt.tight_layout()
plt.savefig(f"IR_SPEC_{UNITS}.jpg", dpi=300)
plt.clf()














TIME = np.loadtxt("QC.dat")[:,0] # fs
QC   = np.loadtxt("QC.dat")[:,1] # a.u.
dt   = TIME[1]  - TIME[0]
T    = TIME[-1] - TIME[0]

# PLOT DIPOLE TIME SERIES
plt.plot( TIME, QC )
plt.xlabel("Time (fs)", fontsize=15)
plt.ylabel("Dipole (Debye)", fontsize=15)
plt.xlim( 0, 500 )
plt.tight_layout()
plt.savefig("QC.jpg", dpi=300)
plt.clf()

# GET AUTO-CORRELATION FUNCTION OF QC
QC = get_ACF( QC )
plt.plot( TIME, QC )
plt.xlabel("Lag Time (fs)", fontsize=15)
plt.ylabel("Auto-correlation Function", fontsize=15)
#plt.xlim( 0, 2000 )
plt.xlim( 0 )
plt.tight_layout()
plt.savefig("QC_ACF.jpg", dpi=300)
plt.clf()


QC_w     = np.zeros( (Nw), dtype=complex )

TIME_AU = TIME * 41.341 # au/fs
w_AU    = w_meV / 1000 / 27.2114 # meV --> au energy
cos_fac = np.cos( np.pi * TIME_AU / (2 * np.max(TIME_AU)) ) # Smooth Function
FUNC    = QC * cos_fac
for wi in range(Nw):
    if ( wi%500==0 ):
        print(w_meV[wi], "meV")
    exp_fact = np.exp( 1j * w_AU[wi] * TIME_AU )
    int_func = exp_fact * FUNC
    QC_w[wi] = 2*np.sum(int_func)
QC_w *= dw / np.sqrt(2 * np.pi)

# FIND THE PEAKS
THRESHOLD = np.max( np.real(QC_w) ) / 50
PEAKS, _  = find_peaks(np.real(QC_w), height=THRESHOLD)
w_cm = w_meV * 8.065610 # cm^-1 / meV
np.savetxt( f"QC_SPEC.dat", np.c_[w_meV, w_cm, np.real(QC_w)], fmt="%1.5f", header="w(meV)    w(cm-1)    IR" )
np.savetxt( f"QC_SPEC_PEAKS.dat", np.c_[w_meV[PEAKS], w_cm[PEAKS], np.real(QC_w)[PEAKS] ], fmt="%1.5f", header="w_max(meV)    w_max(cm-1)    IR_max" )


if ( UNITS == "cm-1" ):
    w_cm = w_meV * 8.065610 # cm^-1 / meV
    plt.plot( w_cm, np.real(QC_w), "-", c='black', label="RE" )
    #plt.plot( w_cm, np.imag(QC_w), "-", c='red',   label="IM" )
    plt.xlim(w_cm[0],w_cm[-1])
    plt.xlabel("Energy (cm$^{-1}$)", fontsize=15)
elif ( UNITS == "meV" ):
    plt.plot( w_meV, np.real(QC_w), "-",  c='black', label="RE" )
    #plt.plot( w_meV, np.imag(QC_w), "-",  c='red',   label="IM" )
    plt.xlim(w_meV[0],w_meV[-1])
    plt.xlabel("Energy (meV)", fontsize=15)
else:
    print("Warning. UNITS not recognized.")
    exit()

plt.ylabel("Absoprtion (Arb. Units)", fontsize=15)
#plt.legend()
plt.tight_layout()
plt.savefig(f"QC_SPEC_{UNITS}.jpg", dpi=300)
plt.clf()




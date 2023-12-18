import numpy as np
import random

def get_Polariton_Energy( DYN_PROPERTIES ):
    """
    Classical cavity DOFs
    H_PF         = Eg + 0.5 WC^2 qc^2 + sqrt(2 WC^3) A0 MU_gg qc + WC A0^2 MU_00^2
    """
    QC       = DYN_PROPERTIES["QC"]
    PC       = DYN_PROPERTIES["PC"]
    A0       = DYN_PROPERTIES["A0"]
    WC       = DYN_PROPERTIES["WC_AU"]
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3
    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL )

    E  = 0.0
    E += 0.500 * WC**2 * QC**2 # This term dies becuse <0|qc|0> = 0
    E += np.sqrt(2 * WC**3) * A0 * DIP * QC # This term dies becuse <0|qc|0> = 0
    E += WC * A0**2 * DIP**2

    return E

def get_Polariton_Kinetic_Energy( DYN_PROPERTIES ):
    PC = DYN_PROPERTIES["PC"]
    return 0.500 * PC**2

def get_Polariton_Force( DYN_PROPERTIES ):
    """
    Classical Cavity
    H_PF         = Eg + 0.5 WC^2 qc^2 + sqrt(2 WC^3) A0 MU_gg qc + WC A0^2 MU_00^2
    \\nabla H_PF = (\\nabla E_g) + sqrt(2 WC^3) A0 (\\nabla MU_gg) qc + 2 * WC A0^2 MU_00 * (\\nabla MU_00)
    """

    QC       = DYN_PROPERTIES["QC"]
    PC       = DYN_PROPERTIES["PC"]
    A0       = DYN_PROPERTIES["A0"]
    WC       = DYN_PROPERTIES["WC_AU"]
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    DIP_GRAD = DYN_PROPERTIES["DIP_GRAD"] # N,3,3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3
    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL ) # 1
    DIP_GRAD = np.einsum( "Nde,e->Nd", DIP_GRAD, EPOL ) # N,3

    # Cavity Born-Oppenheimer Gradient
    FORCE    = np.zeros( (len(DIP_GRAD),3) ) # N,3
    FORCE   += -1 * np.sqrt(2 * WC**3) * A0 * DIP_GRAD[:,:] * QC # The direct coupling term dies because <0|qc|0> = 0, unless we do cavity BO approximation
    FORCE   += -1 * WC * A0**2 * (DIP_GRAD[:,:] * DIP + DIP * DIP_GRAD[:,:] ) # Chain Rule
    return FORCE

def propagate_Polariton( DYN_PROPERTIES ):
    """
    Classical cavity coordinates.
    H_PF           = Eg + 0.5 PC^2 +  0.5 WC^2 qc_min^2 + sqrt(2 WC^3) A0 MU_gg qc_MIN + WC A0^2 MU_00^2
    \nabla_q H_PF  = WC^2 QC + sqrt(2 WC^3) A0 MU_gg
    -\nabla_p H_PF = PC
    """
    dtI      = DYN_PROPERTIES["dtI"]
    QC       = DYN_PROPERTIES["QC"]
    PC       = DYN_PROPERTIES["PC"]
    A0       = DYN_PROPERTIES["A0"]
    WC       = DYN_PROPERTIES["WC_AU"]
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3
    # Project along cavity polarization
    DIP      = np.einsum( "e,e->", DIP, EPOL ) # 1

    ESTEPS = 1000
    dtE    = dtI / ESTEPS
    for step in range( ESTEPS ):
        QC       += 0.500 * dtE * PC
        FORCE     = -1 * ( WC**2 * QC + np.sqrt(2 * WC**3) * A0 * DIP )
        PC       +=         dtE * FORCE
        QC       += 0.500 * dtE * PC

    DYN_PROPERTIES["QC"] = QC
    DYN_PROPERTIES["PC"] = PC

    return DYN_PROPERTIES



def initialize_Cavity(DYN_PROPERTIES):
    # Sample QC and PC from Gaussian
    A0   = DYN_PROPERTIES["A0"]
    WC   = DYN_PROPERTIES["WC_AU"]
    DIP  = DYN_PROPERTIES["DIPOLE"] # 3
    EPOL = DYN_PROPERTIES["EPOL"] # 3
    DIP  = np.einsum( "e,e->", DIP, EPOL ) # 1
    
    T    = 300               # K
    kBT  = T * (0.025 / 300) # K --> eV
    kBT /= 27.2114           # eV --> a.u.
    #beta = 315774 / T # 1/K --> 1/a.u. --- from Sebastian
    beta = 1/kBT
    QC0  = -np.sqrt(2/WC) * A0 * DIP
    DYN_PROPERTIES["QC"] = random.gauss( QC0, np.sqrt(1/beta * WC**2) )
    DYN_PROPERTIES["PC"] = random.gauss( 0, np.sqrt(1/beta) )

    #print("WC, QC0, QC, PC, MU:", WC, QC0, DYN_PROPERTIES["QC"], DYN_PROPERTIES["PC"], DIP )
    #print("Photon Energy:", 0.500 * DYN_PROPERTIES["QC"]**2 * WC**2 + 0.500 * DYN_PROPERTIES["PC"]**2 )
    #exit()
    return DYN_PROPERTIES


import numpy as np

def get_Polariton_Energy( DYN_PROPERTIES ):
    """
    Cavity Born-Oppenheimer Approximation
    qc_min       = -sqrt(2/wc) A0 MU_gg
    H_PF         = Eg + 0.5 WC^2 qc_min^2 + sqrt(2 WC^3) A0 MU_gg qc_MIN + WC A0^2 MU_00^2
    """
    A0       = DYN_PROPERTIES["A0"] # 3
    WC       = DYN_PROPERTIES["WC_AU"] # 3
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3

    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL )

    QC_MIN   = -np.sqrt( 2/WC ) * A0 * DIP

    # Cavity Born-Oppenheimer Energy
    E  = 0.0
    E += 0.500 * WC**2 * QC_MIN**2 # This term dies becuse <0|qc|0> = 0
    E += np.sqrt(2 * WC**3) * A0 * DIP * QC_MIN    # This term dies becuse <0|qc|0> = 0
    E += WC * A0**2 * DIP**2

    return E

def get_Polariton_Force( DYN_PROPERTIES ):
    """
    Cavity Born-Oppenheimer Approximation
    qc_min       = -sqrt(2/wc) A0 MU_gg
    H_PF         = Eg + 0.5 WC^2 qc_min^2 + sqrt(2 WC^3) A0 MU_gg qc_MIN + WC A0^2 MU_00^2
    \\nabla H_PF = (\\nabla E_g) + sqrt(2 WC^3) A0 (\\nabla MU_gg) qc_MIN + 2 * WC A0^2 MU_00 * (\\nabla MU_00)
    """

    A0       = DYN_PROPERTIES["A0"] # 3
    WC       = DYN_PROPERTIES["WC_AU"] # 3
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    DIP_GRAD = DYN_PROPERTIES["DIP_GRAD"] # N,3,3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3

    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL ) # 1
    DIP_GRAD = np.einsum( "Nde,e->Nd", DIP_GRAD, EPOL ) # N,3

    QC_MIN   = -np.sqrt( 2/WC ) * A0 * DIP

    # Cavity Born-Oppenheimer Gradient
    FORCE    = np.zeros( (len(DIP_GRAD),3) ) # N,3
    FORCE   += -1 * np.sqrt(2 * WC**3) * A0 * DIP_GRAD[:,:] * QC_MIN # The direct coupling term dies because <0|qc|0> = 0, unless we do cavity BO approximation
    FORCE   += -1 * WC * A0**2 * (DIP_GRAD[:,:] * DIP + DIP * DIP_GRAD[:,:] ) # Chain Rule
    return FORCE



import numpy as np

def get_Polariton_Energy( DYN_PROPERTIES ):
    A0       = DYN_PROPERTIES["A0"] # 3
    WC       = DYN_PROPERTIES["WC_AU"] # 3
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3

    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL )

    E  = 0.0
    #E += WC * A0 * DIP # This term dies becuse <0|qc|0> = 0
    E += WC * A0**2 * DIP**2

    return E

def get_Polariton_Force( DYN_PROPERTIES ):
    A0       = DYN_PROPERTIES["A0"] # 3
    WC       = DYN_PROPERTIES["WC_AU"] # 3
    DIP      = DYN_PROPERTIES["DIPOLE"] # 3
    DIP_GRAD = DYN_PROPERTIES["DIP_GRAD"] # N,3,3
    EPOL     = DYN_PROPERTIES["EPOL"] # 3

    # Project along cavity polarizations
    DIP      = np.einsum( "e,e->", DIP, EPOL ) # 1
    DIP_GRAD = np.einsum( "Nde,e->Nd", DIP_GRAD, EPOL ) # N,3

    FORCE    = np.zeros( (len(DIP_GRAD),3) ) # N,3
    #FORCE   += -1 * WC * A0    * DIP_GRAD[:,:] # The direct coupling term dies because <0|qc|0> = 0
    FORCE   += -1 * WC * A0**2 * (DIP_GRAD[:,:] * DIP + DIP * DIP_GRAD[:,:] ) # Chain Rule
    return FORCE



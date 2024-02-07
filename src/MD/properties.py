import numpy as np

from polariton import get_Polariton_Energy, get_Polariton_Kinetic_Energy

def compute_KE(DYN_PROPERTIES):
    KE = 0.0
    for at in range( DYN_PROPERTIES["NAtoms"] ):
        KE += 0.50000000 * DYN_PROPERTIES["MASSES"][at] * np.linalg.norm(DYN_PROPERTIES["Atom_velocs_new"][at,:])**2
    if ( DYN_PROPERTIES["do_POLARITON"] ):
        KE += get_Polariton_Kinetic_Energy(DYN_PROPERTIES)

    DYN_PROPERTIES["KE"] = KE
    return DYN_PROPERTIES

def compute_PE(DYN_PROPERTIES):
    DYN_PROPERTIES["PE"] = DYN_PROPERTIES["ENERGY_NEW"]
    if ( DYN_PROPERTIES["do_POLARITON"] ):
        DYN_PROPERTIES["PE"] += get_Polariton_Energy(DYN_PROPERTIES)
    return DYN_PROPERTIES

def compute_Temperature(DYN_PROPERTIES):
    DYN_PROPERTIES = compute_KE(DYN_PROPERTIES)
    NAtoms = DYN_PROPERTIES['NAtoms']

    KE = DYN_PROPERTIES["KE"] * 27.2114 # a.u. --> eV

    T = (2/3) * KE / NAtoms * (300 / 0.025) # eV --> K

    return T




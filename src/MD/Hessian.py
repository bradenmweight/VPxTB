import numpy as np
import subprocess as sp
import os
import multiprocessing as mp

import output

def make_XYZ( LABELS, POS ):
    FILE01 = open("geometry.xyz","w")
    FILE01.write("%1.0f\n" % len(LABELS))
    FILE01.write("Title Line\n")
    for at in range( len(LABELS) ):
        FILE01.write( "%s %1.5f %1.5f %1.5f" % (LABELS[at], POS[at,0]*0.529, POS[at,1]*0.529, POS[at,2]*0.529) ) # Already in Bohr
        if ( not at == len(LABELS)-1 ):
            FILE01.write("\n")
    FILE01.close()


def get_Hessian( DYN_PROPERTIES ):
    """
    TODO -- Add flag for do_POLARITON to get normal modes and frequencies while coupling to cavity
    """

    LABELS = DYN_PROPERTIES["Atom_labels"]
    COORDS = DYN_PROPERTIES["Atom_coords_new"]
    M      = DYN_PROPERTIES["MASSES"]
    NATOMS = len(LABELS)
    dR_num = 0.001

    # Get Energy at Reference Geometry
    make_XYZ(LABELS, COORDS)
    sp.call("xtb geometry.xyz > xtb.out", shell=True)
    ER = float( sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True) )

    # Manually compute the Hessian
    HESSIAN = np.zeros((NATOMS, 3, NATOMS, 3))
    
    # Compute the diagonal parts first
    for at1 in range(NATOMS):
        for x1 in range(3):
            # Get forward displacement
            COORDS_NUM = COORDS.copy()
            COORDS_NUM[at1,x1] += dR_num
            make_XYZ(LABELS, COORDS_NUM)
            sp.call("xtb geometry.xyz > xtb.out", shell=True)
            E_FORWARD = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
            E_FORWARD = float(E_FORWARD.strip())
            # Get backward displacement
            COORDS_NUM = COORDS.copy()
            COORDS_NUM[at1,x1] += -dR_num
            make_XYZ(LABELS, COORDS_NUM)
            sp.call("xtb geometry.xyz > xtb.out", shell=True)
            E_BACKWARD = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
            E_BACKWARD = float(E_BACKWARD.strip())
            # Compute d^2E/dx^2 using central difference
            HESSIAN[at1,x1,at1,x1] = (E_FORWARD - 2*ER + E_BACKWARD) / dR_num**2
    
    # Compute the off-diagonal parts
    for at1 in range(NATOMS):
        for x1 in range(3):
            for at2 in range(NATOMS):
                for x2 in range(3):
                    if ( at1 == at2 and x1 == x2 ): continue
                    # Get two forward displacements
                    COORDS_NUM = COORDS.copy()
                    COORDS_NUM[at1,x1] += dR_num
                    COORDS_NUM[at2,x2] += dR_num
                    make_XYZ(LABELS, COORDS_NUM)
                    sp.call("xtb geometry.xyz > xtb.out", shell=True)
                    E_FF = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                    E_FF = float(E_FF.strip())

                    # Get one forward displacement and one backward displacement
                    COORDS_NUM = COORDS.copy()
                    COORDS_NUM[at1,x1] += dR_num
                    COORDS_NUM[at2,x2] -= dR_num
                    make_XYZ(LABELS, COORDS_NUM)
                    sp.call("xtb geometry.xyz > xtb.out", shell=True)
                    E_FB = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                    E_FB = float(E_FB.strip())

                    # Get one backward displacement and one forward displacement
                    COORDS_NUM = COORDS.copy()
                    COORDS_NUM[at1,x1] -= dR_num
                    COORDS_NUM[at2,x2] += dR_num
                    make_XYZ(LABELS, COORDS_NUM)
                    sp.call("xtb geometry.xyz > xtb.out", shell=True)
                    E_BF = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                    E_BF = float(E_BF.strip())

                    # Get two backward displacements
                    COORDS_NUM = COORDS.copy()
                    COORDS_NUM[at1,x1] -= dR_num
                    COORDS_NUM[at2,x2] -= dR_num
                    make_XYZ(LABELS, COORDS_NUM)
                    sp.call("xtb geometry.xyz > xtb.out", shell=True)
                    E_BB = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                    E_BB = float(E_BB.strip())

                    # Compute d^2E/dx^2 using central difference
                    HESSIAN[at1,x1,at2,x2] = (E_FF - E_FB - E_BF + E_BB) / (4*dR_num**2) 
                    # Weight Hessian by masses
                    #HESSIAN[at1,x1,at2,x2] = HESSIAN[at1,x1,at2,x2] / np.sqrt(M[at1]*M[at2])

    DYN_PROPERTIES["HESSIAN"] = HESSIAN

    return DYN_PROPERTIES


def get_Normal_Modes( DYN_PROPERTIES ):
    # Compute the mass-weighted Hessian
    # H = m * w^2 --> Hjk = 1/sqrt(mk) * w * 1/sqrt(mj)
    HESSIAN = DYN_PROPERTIES["HESSIAN"] # SHAPE = (N,3,N,3)
    NATOMS  = len(HESSIAN)
    M      = DYN_PROPERTIES["MASSES"]
    for at1 in range(NATOMS):
        for x1 in range(3):
            for at2 in range(NATOMS):
                for x2 in range(3):
                    HESSIAN[at1,x1,at2,x2] = HESSIAN[at1,x1,at2,x2] / np.sqrt( M[at1]*M[at2] )
    H       = HESSIAN.reshape( 3*NATOMS, 3*NATOMS ) # Convert to square matrix
    w2, U   = np.linalg.eigh( H )
    U       = U.reshape( (NATOMS,3,3*NATOMS) ) # Turn each mode into an xyz-vector per atom
    w       = np.sqrt(w2)

    DYN_PROPERTIES["NM_FREQUENCIES"] = w
    DYN_PROPERTIES["NM_WAVEFUNCTIONS"] = U # Should we un-mass weight these modes ?
    return DYN_PROPERTIES




def main( DYN_PROPERTIES ):

    if ( not os.path.exists(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE") ):
        sp.call(f"mkdir {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)
    os.chdir(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE")

    DYN_PROPERTIES = get_Hessian( DYN_PROPERTIES )
    DYN_PROPERTIES = get_Normal_Modes( DYN_PROPERTIES )

    os.chdir(f"{DYN_PROPERTIES['VPxTB_RUNNING_DIR']}")
    output.saveNM( DYN_PROPERTIES )
    exit()


    return DYN_PROPERTIES


if ( __name__ == "__main__" ):
    main()
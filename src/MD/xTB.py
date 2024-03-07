import numpy as np
import subprocess as sp
import os
import multiprocessing as mp

def make_XYZ( LABELS, POS ):
    FILE01 = open("geometry.xyz","w")
    FILE01.write("%1.0f\n" % len(LABELS))
    FILE01.write("Title Line\n")
    for at in range( len(LABELS) ):
        FILE01.write( "%s %1.5f %1.5f %1.5f" % (LABELS[at], POS[at,0]*0.529, POS[at,1]*0.529, POS[at,2]*0.529) ) # Already in Bohr
        if ( not at == len(LABELS)-1 ):
            FILE01.write("\n")
    FILE01.close()

def run_xTB_SinglePoint( DYN_PROPERTIES ):
    
    Atom_labels      = DYN_PROPERTIES["Atom_labels"]
    Atom_coords_new  = DYN_PROPERTIES["Atom_coords_new"]
    
    make_XYZ( Atom_labels, Atom_coords_new )
    sp.call("xtb geometry.xyz --grad > xtb.out", shell=True)


def get_numerical_gradients( DYN_PROPERTIES ):
    #do_HESSIAN = DYN_PROPERTIES["do_HESSIAN"]
    E0         = DYN_PROPERTIES["ENERGY_NEW"]
    MU0        = DYN_PROPERTIES["DIPOLE"]
    LABELS     = DYN_PROPERTIES["Atom_labels"]
    COORDS     = DYN_PROPERTIES["Atom_coords_new"]
    NATOMS   = len(LABELS)
    dR_num   = 0.01
    E0       = 0.0
    MU0      = np.zeros( (3) )
    E_NUM    = np.zeros( (NATOMS,3,2) )   # N, xyz, Forward/backward
    E_GRAD   = np.zeros( (NATOMS,3) )     # N, xyz
    DIP_NUM  = np.zeros( (NATOMS,3,2,3) ) # N, xyz, Forward/backward, (MUx,MUy,MUz)
    DIP_GRAD = np.zeros( (NATOMS,3,3) )   # N, xyz, (MUx,MUy,MUz)
    # if ( do_HESSIAN == True ):
    #     HESS      = np.zeros( (NATOMS,NATOMS,3,3) ) # N, N, xyz, xyz -- Only need E(x + h, y + h) and E(x - h, y - h) terms in addition to E_GRAD
    #     E_NUM_NUM = np.zeros( (NATOMS,NATOMS,3,3,2) ) # N, N, xyz, xyz, FF/BB -- Only need E(x + h, y + h) and E(x - h, y - h) terms in addition to E_GRAD
    
    # This set of loops in all we need for E_GRAD and MU_GRAD
    # All we do here is E(x+h) and MU(x+h)
    for at in range( NATOMS ):
        for d in range( 3 ):
            for pm in range( 2 ):
                # Shift single DOF
                COORDS_NUM  = COORDS
                COORDS_NUM[at,d] += -dR_num * (pm==0) + dR_num * (pm==1)
                # Make XYZ file and run xTB
                make_XYZ( LABELS, COORDS_NUM )
                sp.call("xtb geometry.xyz > xtb.out", shell=True)
                # Extract new energy
                E = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                E_NUM[at,d,pm] = E
                # Extract new dipole
                sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
                DIP_NUM[at,d,pm,:] = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.
                # if ( do_HESSIAN == True ):
                #     # E(x + h, y + h) and E(x - h, y - h) terms
                #     for at_2 in range( NATOMS ):
                #         for d_2 in range( 3 ):
                #             # Extract new energy
                #             COORDS_NUM[at_2,d_2] = -dR_num * (pm==0) + dR_num * (pm==1) # Do same shift as above current pm
                #             make_XYZ( LABELS, COORDS_NUM )
                #             sp.call("xtb geometry.xyz > xtb.out", shell=True)
                #             E = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
                #             E_NUM_NUM[at,at_2,d,d_2,pm] = E
            # Central difference
            E_GRAD[at,d] = (E_NUM[at,d,1] - E_NUM[at,d,0]) / 2 / dR_num
            DIP_GRAD[at,d,:] = (DIP_NUM[at,d,1,:] - DIP_NUM[at,d,0,:]) / 2 / dR_num # (dx,dy,dz)

    return DIP_GRAD

def get_numerical_gradients_parallel( at, Atom_labels, COORDS, do_HESSIAN ):
    sp.call(f"rm -r TMP_{at}", shell=True)
    sp.call(f"mkdir TMP_{at}", shell=True)
    os.chdir(f"TMP_{at}")

    dR_num = 0.01
    DIP_NUM  = np.zeros( (3,2,3) ) # Forward/backward, (dx,dy,dz)
    DIP_GRAD = np.zeros( (3,3) )
    for d in range( 3 ):
        for pm in range( 2 ):
            # Shift single DOF
            COORDS_NUM  = COORDS * 1.0
            COORDS_NUM[at,d] += -dR_num * (pm==0) + dR_num * (pm==1)
            # Make XYZ file and run xTB
            make_XYZ( Atom_labels, COORDS_NUM )
            sp.call("xtb geometry.xyz > xtb.out", shell=True)
            # Extract new dipole
            sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
            DIP_NUM[d,pm,:] = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.
        # Central difference
        DIP_GRAD[d,:] = (DIP_NUM[d,1,:] - DIP_NUM[d,0,:]) / 2 / dR_num # (dx,dy,dz)
    
    os.chdir(f"../")

    return DIP_GRAD

def get_Properties( DYN_PROPERTIES ):

    # Get Energy
    sp.call("grep 'TOTAL ENERGY' xtb.out | awk '{print $4}' > GS_ENERGY.dat", shell=True)
    DYN_PROPERTIES["ENERGY_NEW"] = np.loadtxt("GS_ENERGY.dat")

    # Get electronic gradient
    NATOMS = len(DYN_PROPERTIES["Atom_labels"])
    lines = open("gradient", "r").readlines()
    NATOMS = (len(lines) - 3) // 2
    GRAD = np.zeros( (NATOMS,3) )
    for at in range( NATOMS ):
        t = lines[2+NATOMS+at].split()
        GRAD[at,:] = np.array( t[:] )
    DYN_PROPERTIES["GRAD_NEW"] = GRAD

    # Get GS dipole
    sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
    DYN_PROPERTIES["DIPOLE"] = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.


    # Get GS dipole gradient -- numerical gradient is expensive
    if ( DYN_PROPERTIES["do_POLARITON"] == True ): # Are we even including polaritonic effects ?
        if ( DYN_PROPERTIES["PARALLEL_GRADIENT"] == True ): # Can we do it in parallel ?
            DYN_PROPERTIES["DIP_GRAD"] = np.zeros( (NATOMS,3,3) )
            LIST = [ [at, DYN_PROPERTIES["Atom_labels"], DYN_PROPERTIES["Atom_coords_new"], False] for at in range( NATOMS ) ]
            with mp.Pool(processes=DYN_PROPERTIES["NCPUS"]) as pool:
                DIP_GRAD = pool.starmap(get_numerical_gradients_parallel, LIST )
            DYN_PROPERTIES["DIP_GRAD"] += np.array( DIP_GRAD )
        else:
            DIP_GRAD = get_numerical_gradients( DYN_PROPERTIES )
            DYN_PROPERTIES["DIP_GRAD"] = DIP_GRAD
    else:
        DYN_PROPERTIES["DIP_GRAD"] = np.zeros( (NATOMS,3,3) )

    return DYN_PROPERTIES

def main( DYN_PROPERTIES ):

    if ( not os.path.exists(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE") ):
        sp.call(f"mkdir {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)
    os.chdir(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE")

    # Do single-point calculation
        # Get GS energy
        # Get GS electronic forces
    run_xTB_SinglePoint( DYN_PROPERTIES )
    DYN_PROPERTIES = get_Properties( DYN_PROPERTIES )

    os.chdir(f"{DYN_PROPERTIES['VPxTB_RUNNING_DIR']}")

    return DYN_PROPERTIES


if ( __name__ == "__main__" ):
    main()
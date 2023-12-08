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


def get_numerical_gradient( LABELS, COORDS ):
    NATOMS = len(LABELS)
    dR_num = 0.01
    DIP_NUM  = np.zeros( (NATOMS,3,2,3) ) # Forward/backward, (dx,dy,dz)
    DIP_GRAD = np.zeros( (NATOMS,3,3) )
    for at in range( NATOMS ):
        for d in range( 3 ):
            for pm in range( 2 ):
                # Shift single DOF
                COORDS_NUM  = COORDS
                Atom_labels = LABELS
                COORDS_NUM[at,d] += dR_num * (pm==0) - dR_num * (pm==1)
                # Make XYZ file and run xTB
                make_XYZ( Atom_labels, COORDS_NUM )
                sp.call("xtb geometry.xyz > xtb.out", shell=True)
                # Extract new dipole
                sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
                DIP_NUM[at,d,pm,:] = np.loadtxt("DIPOLE.dat") / 2.5 # (dx,dy,dz) # Debye --> a.u.
            # Central difference
            DIP_GRAD[at,d,:] = (DIP_NUM[at,d,1,:] - DIP_NUM[at,d,0,:]) / 2 / dR_num # (dx,dy,dz)
    return DIP_GRAD

def get_numerical_gradient_parallel( at, Atom_labels, COORDS ):
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
            COORDS_NUM[at,d] += dR_num * (pm==0) - dR_num * (pm==1)
            # Make XYZ file and run xTB
            make_XYZ( Atom_labels, COORDS_NUM )
            sp.call("xtb geometry.xyz > xtb.out", shell=True)
            # Extract new dipole
            sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
            DIP_NUM[d,pm,:] = np.loadtxt("DIPOLE.dat") / 2.5 # (dx,dy,dz) # Debye --> a.u.
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
    DYN_PROPERTIES["DIPOLE"] = np.loadtxt("DIPOLE.dat") / 2.5 # Debye --> a.u.


    # Get GS dipole gradient -- numerical gradient is expensive
    if ( DYN_PROPERTIES["do_POLARITON"] == True ): # Are we even including polaritonic effects ?
        if ( DYN_PROPERTIES["PARALLEL_GRADIENT"] == True ):
            DYN_PROPERTIES["DIP_GRAD"] = np.zeros( (NATOMS,3,3) )
            LIST = [ [at, DYN_PROPERTIES["Atom_labels"], DYN_PROPERTIES["Atom_coords_new"]] for at in range( NATOMS ) ]
            with mp.Pool(processes=DYN_PROPERTIES["NCPUS"]) as pool:
                DIP_GRAD = pool.starmap(get_numerical_gradient_parallel, LIST )
            DYN_PROPERTIES["DIP_GRAD"] += np.array( DIP_GRAD )
        else:
            DIP_GRAD = get_numerical_gradient( DYN_PROPERTIES["Atom_labels"], DYN_PROPERTIES["Atom_coords_new"] )
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
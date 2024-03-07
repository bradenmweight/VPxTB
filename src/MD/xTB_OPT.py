import numpy as np
import subprocess as sp
import os
import multiprocessing as mp

import polariton
import output
import rotation

def make_XYZ( LABELS, POS ):
    FILE01 = open("geometry.xyz","w")
    FILE01.write("%1.0f\n" % len(LABELS))
    FILE01.write("Title Line\n")
    for at in range( len(LABELS) ):
        FILE01.write( "%s %1.5f %1.5f %1.5f" % (LABELS[at], POS[at,0]*0.529, POS[at,1]*0.529, POS[at,2]*0.529) ) # Already in Bohr
        if ( not at == len(LABELS)-1 ):
            FILE01.write("\n")
    FILE01.close()

def read_Dipole( LABELS, COORDS, QN ):
    # OLD WAY -- ONLY PROVIDES 0.001 a.u. accuracy
    sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
    DIPOLE_OLD = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.   

    # NEW WAY -- COMPUTE DIPOLE MANUALLY WITH CHARGES AND COORDINATES
    CHARGES = np.loadtxt("charges")
    DIPOLE = np.zeros(3)
    for at in range( len(LABELS) ):
        print("q, Q, R", -CHARGES[at], QN[at], COORDS[at,:])
        DIPOLE += (CHARGES[at]-1) * COORDS[at,:]

    print("DIPOLE (OLD)", DIPOLE_OLD)
    print("DIPOLE (NEW)", DIPOLE)
    exit()
    return DIPOLE

def get_numerical_gradient( DYN_PROPERTIES ):
    QN         = DYN_PROPERTIES["QN"]
    LABELS     = DYN_PROPERTIES["Atom_labels"]
    COORDS     = DYN_PROPERTIES["Atom_coords_new"]
    NATOMS   = len(LABELS)
    dR_num   = 1e-4
    E_NUM    = np.zeros( (NATOMS,3,2) )   # N, xyz, Forward/backward
    E_GRAD   = np.zeros( (NATOMS,3) )     # N, xyz

    # Get reference geometry energy and dipole

    if ( DYN_PROPERTIES["do_POLARITON"] == True ):
        make_XYZ( LABELS, COORDS )
        sp.call("xtb geometry.xyz > xtb.out", shell=True)        
        #sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
        #DIPOLE                   = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.
        DIPOLE = read_Dipole( LABELS, COORDS, QN )
        DYN_PROPERTIES["DIPOLE"] = DIPOLE
        A0                       = DYN_PROPERTIES["A0"]
        WC                       = DYN_PROPERTIES["WC_AU"]
        DIP_e                    = np.einsum( "e,e->", DIPOLE, DYN_PROPERTIES["EPOL"] )
        DYN_PROPERTIES["QC"]     = -np.sqrt(2/WC) * A0 * DIP_e # Choose minimum of CBO potential at reference geometry

    # This set of loops is all we need for E_GRAD
    # All we do here is E(x+h) and E(x-h)
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
                E = float( sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True) )
                if ( DYN_PROPERTIES["do_POLARITON"] == True ):
                    sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
                    DYN_PROPERTIES["DIPOLE"] = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.
                    Eph = polariton.get_Polariton_Energy( DYN_PROPERTIES )
                    E += Eph
                E_NUM[at,d,pm] = E
            # Central difference
            E_GRAD[at,d] = (E_NUM[at,d,1] - E_NUM[at,d,0]) / 2 / dR_num


    # if ( DYN_PROPERTIES["do_POLARITON"] == False or DYN_PROPERTIES["A0"] == 0.0 ):
        # Compare numerical gradient against the xTB analytical gradient
        # make_XYZ( LABELS, COORDS )
        # sp.call("xtb geometry.xyz --grad > xtb.out", shell=True)
        # lines = open("gradient", "r").readlines()
        # xTB_GRAD = np.zeros( (NATOMS,3) )
        # for at in range( NATOMS ):
        #     t = lines[2+NATOMS+at].split()
        #     xTB_GRAD[at,:] = np.array( t[:] )
    #     if ( not np.allclose(xTB_GRAD, E_GRAD) ):
    #         print("Analytical xTB gradient does not match numerical gradient. Exiting.")
    #         print("Analytical\n", xTB_GRAD)
    #         print("Numerical\n", E_GRAD)
    #         print("Error\n", xTB_GRAD-E_GRAD)
    #         exit()
    #     else:
    #         print("Analytical xTB gradient matches numerical gradient.")
    #         print("Analytical\n", xTB_GRAD)
    #         print("Numerical\n", E_GRAD)
    #         print("Error\n", xTB_GRAD-E_GRAD)
    return E_GRAD


def get_Energy( DYN_PROPERTIES ):
    make_XYZ( DYN_PROPERTIES["Atom_labels"], DYN_PROPERTIES["Atom_coords_new"] )
    sp.call("xtb geometry.xyz > xtb.out", shell=True)
    E = sp.check_output("grep 'TOTAL ENERGY' xtb.out | tail -n 1 | awk '{print $4}'", shell=True)
    E = float( E )
    if ( DYN_PROPERTIES["do_POLARITON"] == True ):
        sp.call("grep 'molecular dipole' xtb.out -A 3 | tail -n 1 | awk '{print $2, $3, $4}' > DIPOLE.dat", shell=True)
        DIPOLE = np.loadtxt("DIPOLE.dat") #/ 2.5 # (dx,dy,dz) # ALREADY IN a.u.
        DYN_PROPERTIES["DIPOLE"] = DIPOLE
        A0 = DYN_PROPERTIES["A0"]
        WC = DYN_PROPERTIES["WC_AU"]
        DIP_e = np.einsum( "e,e->", DIPOLE, DYN_PROPERTIES["EPOL"] )
        DYN_PROPERTIES["QC"] = -np.sqrt(2/WC) * A0 * DIP_e # Choose minimum of CBO potential
        Eph = polariton.get_Polariton_Energy( DYN_PROPERTIES )
        #print( "Photon Energy Contribution:", E, Eph, Eph/(E+Eph) )
        E += Eph
    return E

def optimize( DYN_PROPERTIES ):

    LABELS       = DYN_PROPERTIES["Atom_labels"]
    COORDS       = DYN_PROPERTIES["Atom_coords_new"]
    M            = DYN_PROPERTIES["MASSES"]
    NATOMS       = len(LABELS)

    NSTEPS       = 5000
    dE_THRESHOLD = 1e-6
    dF_THRESHOLD = 1e-3

    COORDS_OLD  = DYN_PROPERTIES["Atom_coords_new"]
    E_GRAD_OLD  = DYN_PROPERTIES["Atom_coords_new"] * 0.0
    E_LIST      = [ get_Energy( DYN_PROPERTIES ) ]
    GAMMA       = 0.1
    PERP_GRAD   = []
    for step in range( NSTEPS ):
        E_GRAD_NEW = get_numerical_gradient( DYN_PROPERTIES )
        COORDS_NEW = COORDS_OLD - GAMMA * E_GRAD_NEW
        print( "dR1 =\n", -GAMMA * E_GRAD_NEW * 0.529 )
        print( "dR2 =\n", COORDS_NEW * 0.529 - COORDS_OLD * 0.529 )
        #COORDS_NEW = COORDS_OLD - (NSTEPS-step)/NSTEPS * E_GRAD_NEW
        DYN_PROPERTIES["Atom_coords_new"] = COORDS_NEW
        #DYN_PROPERTIES = rotation.shift_COM(DYN_PROPERTIES)
        #DYN_PROPERTIES = rotation.remove_rotations(DYN_PROPERTIES)
        print ("GAMMA\n", GAMMA )
        print ("GRAD\n", E_GRAD_NEW )
        print("ATOMS 1:\n", 0.529 * (COORDS_OLD - COORDS_OLD[1,:] ) )
        print("ATOMS 2:\n", 0.529 * (COORDS_NEW - COORDS_NEW[1,:] ) )
        E_LIST.append( get_Energy( DYN_PROPERTIES ) )
        if ( step >= 1 ):
            print("!!!!!!! step, E, |GRAD| =", step, E_LIST[-1], np.sum( np.abs(E_GRAD_NEW) ) )
            print ("!!!!!!! dE = ", E_LIST[-1] - E_LIST[-2] )
            #print( "NORM GEOM DIFF ( (1/3N) * \SUM_a^(3N) (x'_a - x_a) ):", np.sum( np.abs(COORDS_NEW - COORDS_OLD) ) / NATOMS / 3 )
            #print( "NORM GRAD DIFF ( (1/3N) * \SUM_a^(3N) (dE'_a - dE_a) ):", np.sum( np.abs(E_GRAD_NEW - E_GRAD_OLD) ) / NATOMS / 3 )
            #if ( abs(E_LIST[-1] - E_LIST[-2]) < dE_THRESHOLD ):
            if ( E_LIST[-1] - E_LIST[-2] < dE_THRESHOLD and np.sum( np.abs(E_GRAD_NEW) ) < dF_THRESHOLD ):
                print("I HAVE OPTIMIZED THE STRUCTURE AFTER %d STEPS.\n\tFinal energy difference: dE = %1.6f a.u." % (step+1, E_LIST[-1] - E_LIST[-2]))
                break
            elif ( step == NSTEPS-1 ):
                print("Optimization Failed. Exiting.")
                break
        COORDS_OLD = COORDS_NEW * 1.0
        E_GRAD_OLD = E_GRAD_NEW * 1.0

    print("Final Coordinates (Angstroms)\n", 0.529 * (DYN_PROPERTIES["Atom_coords_new"] - DYN_PROPERTIES["Atom_coords_new"][1,:])  )
    for step in range(len(E_LIST)):
        print(step, E_LIST[step] - E_LIST[-1])
    exit()

    return DYN_PROPERTIES






def main( DYN_PROPERTIES ):

    if ( not os.path.exists(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE") ):
        sp.call(f"mkdir {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)

    DYN_PROPERTIES = optimize( DYN_PROPERTIES )

    os.chdir(f"{DYN_PROPERTIES['VPxTB_RUNNING_DIR']}")
    #output.saveNM( DYN_PROPERTIES )
    exit()


    return DYN_PROPERTIES


if ( __name__ == "__main__" ):
    main()
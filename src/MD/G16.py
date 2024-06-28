import numpy as np
import subprocess as sp
import os
import multiprocessing as mp

def make_COM( DYN_PROPERTIES, doFORCE=False ):
    GEOM         = DYN_PROPERTIES["Atom_coords_new"]
    LABELS       = DYN_PROPERTIES["Atom_labels"]
    MEMORY       = DYN_PROPERTIES["MEMORY"]
    CHARGE       = DYN_PROPERTIES["CHARGE"]
    MULTIPLICITY = DYN_PROPERTIES["MULTIPLICITY"]

    FILE01 = open("geometry.com","w")
    if ( not doFORCE and DYN_PROPERTIES["MD_STEP"] > 0):
        FILE01.write("%oldchk=../geometry.chk\n")
    FILE01.write("%chk=geometry.chk\n")
    FILE01.write(f"%mem={MEMORY}GB\n")
    FILE01.write("%nprocshared=1\n\n")
    if ( DYN_PROPERTIES["MD_STEP"] > 0 ):
        if ( doFORCE ): 
            FILE01.write("#P AM1/STO-5G guess=read SCF=XQC NoSymm FORCE\n\n")
        else:
            FILE01.write("#P AM1/STO-5G guess=read SCF=XQC NoSymm\n\n")
    else:
        if ( doFORCE ): 
            FILE01.write("#P AM1/STO-5G SCF=XQC NoSymm FORCE\n\n")
        else:
            FILE01.write("#P AM1/STO-5G SCF=XQC NoSymm\n\n")
    FILE01.write("Title\n\n")
    FILE01.write(f"{CHARGE} {MULTIPLICITY}\n")
    for at in range( len(LABELS) ):
        FILE01.write( "%s %1.5f %1.5f %1.5f" % (LABELS[at], GEOM[at,0]*0.529, GEOM[at,1]*0.529, GEOM[at,2]*0.529) ) # Already in Bohr
        if ( not at == len(LABELS)-1 ):
            FILE01.write("\n")
    FILE01.write("\n\n\n\n\n\n\n")
    FILE01.close()

def run_SinglePoint( DYN_PROPERTIES, doFORCE=False ):
    
    make_COM( DYN_PROPERTIES, doFORCE=True )
    sp.call("g16 < geometry.com > geometry.out", shell=True)
    sp.call("formchk geometry.chk", shell=True)

def extract_dipole( PATH="geometry.out" ):
    DIPOLE = np.zeros( 3 )
    LINES = open(PATH,"r").readlines()
    for count,line in enumerate( LINES ):
        t = line.split()
        if ( t == "Dipole moment (field-independent basis, Debye):".split() ):
            s = LINES[count+1].split()
            DIPOLE[0] = float( s[1] )
            DIPOLE[1] = float( s[3] )
            DIPOLE[2] = float( s[5] )
    return DIPOLE * 0.393456 # Convert to a.u. from Debye

def extract_gradient( NATOMS, PATH="geometry.out" ):
    GRAD   = np.zeros( (NATOMS,3) )
    LINES = open(PATH,"r").readlines()
    for count,line in enumerate( LINES ):
        t = line.split()
        if ( "Forces" in t and "(Hartrees/Bohr)" in t ):
            for at in range(NATOMS):
                s = LINES[count+3+at].split()
                GRAD[at,0] = float( s[2] )
                GRAD[at,1] = float( s[3] )
                GRAD[at,2] = float( s[4] )
    return GRAD

def get_numerical_gradient_parallel( at, DYN_PROPERTIES ):
    if ( os.path.isdir(f"TMP_{at}") ): sp.call(f"rm -r TMP_{at}", shell=True)
    sp.call(f"mkdir TMP_{at}", shell=True)
    os.chdir(f"TMP_{at}")

    dR_num   = 0.01
    DIPOLE   = np.zeros( (3,2,3) ) # Forward/backward, (dx,dy,dz)
    DIP_GRAD = np.zeros( (3,3) )
    for d in range( 3 ):
        for pm in range( 2 ):
            # Shift single DOF
            DYN_PROPERTIES["Atom_coords_new"][at,d] += dR_num * (pm==0) - dR_num * (pm==1)
            # Make COM file and run G16
            run_SinglePoint( DYN_PROPERTIES, doFORCE=False )
            # Extract new dipole
            DIPOLE[d,pm,:] = extract_dipole()
        # Central difference
        DIP_GRAD[d,:] = (DIPOLE[d,0,:] - DIPOLE[d,1,:]) / 2 / dR_num # (dx,dy,dz)
    os.chdir(f"../")
    return DIP_GRAD

def get_numerical_gradient( DYN_PROPERTIES ):
    LABELS = DYN_PROPERTIES["Atom_labels"]
    NATOMS = len(LABELS)
    dR_num   = 0.01 # Bohr
    DIPOLE   = np.zeros( (NATOMS,3,2,3) ) # Forward/backward, (MUx,MUy,MUz)
    DIP_GRAD = np.zeros( (NATOMS,3,3) )
    for at in range( NATOMS ):
        for d in range( 3 ): # Dimension of gradient
            for pm in range( 2 ): # Forward/backward for gradient in direction d
                # Shift single DOF
                DYN_PROPERTIES["Atom_coords_new"][at,d] += dR_num * (pm==0) - dR_num * (pm==1)
                # Make COM file and run G16
                run_SinglePoint( DYN_PROPERTIES, doFORCE=False )
                # Extract new dipole
                DIPOLE[at,d,pm,:] = extract_dipole()
            # Central difference ( pm=0 is forward, pm=1 is backward )
            DIP_GRAD[at,d,:] = (DIPOLE[at,d,0,:] - DIPOLE[at,d,1,:]) / 2 / dR_num # (dx,dy,dz)
    return DIP_GRAD

def get_dipole_gradient( DYN_PROPERTIES ):
    COORDS = DYN_PROPERTIES["Atom_coords_new"]
    LABELS = DYN_PROPERTIES["Atom_labels"]
    NATOMS = len(LABELS)
    if ( not DYN_PROPERTIES["do_POLARITON"] ): return np.zeros( (NATOMS,3,3) )

    # Get GS dipole gradient -- numerical gradient is expensive
    if ( DYN_PROPERTIES["do_POLARITON"] == True ): # Are we even including polaritonic effects ?
        if ( DYN_PROPERTIES["PARALLEL_GRADIENT"] == True ):
            DYN_PROPERTIES["DIP_GRAD"] = np.zeros( (NATOMS,3,3) )
            LIST = [ [at, DYN_PROPERTIES] for at in range( NATOMS ) ]
            with mp.Pool(processes=DYN_PROPERTIES["NCPUS"]) as pool:
                DIP_GRAD = pool.starmap(get_numerical_gradient_parallel, LIST )
            return np.array( DIP_GRAD )
        else:
            return get_numerical_gradient( DYN_PROPERTIES )
    else:
        return np.zeros( (len(DYN_PROPERTIES["Atom_labels"]),3,3) )

def get_Energy_Gradient( DYN_PROPERTIES ):

    # Run calculation
    run_SinglePoint( DYN_PROPERTIES, doFORCE=True )

    # Get Energy
    command = "grep 'SCF Done' geometry.out | tail -n 1 | awk '{print $5}'"
    DYN_PROPERTIES["ENERGY_NEW"] = float( sp.check_output(command, shell=True).decode() )

    # Get electronic gradient
    NATOMS = len(DYN_PROPERTIES["Atom_labels"])
    DYN_PROPERTIES["GRAD_NEW"] = -1 * extract_gradient( NATOMS ) # Already in a.u.
    DYN_PROPERTIES["DIPOLE"]   = extract_dipole()           # Already converted to a.u.

    print( "GRADIENT\n", DYN_PROPERTIES["GRAD_NEW"] )

    # Get dipole gradient if we are doing polaritons (also benchmark gradient with energy gradient from previous)
    
    DYN_PROPERTIES["DIPOLE_GRAD_NEW"] = get_dipole_gradient( DYN_PROPERTIES )

    #print( np.einsum("Nde,e->Nd", DYN_PROPERTIES["DIPOLE_GRAD_NEW"] , np.array([0,0,1])) )
    #print( DYN_PROPERTIES["DIPOLE_GRAD_NEW"] )

    return DYN_PROPERTIES

def main( DYN_PROPERTIES ):

    if ( not os.path.exists(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE") ):
        sp.call(f"mkdir {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)
    os.chdir(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE")

    # Do single-point calculation with bare electronic GS force
    DYN_PROPERTIES = get_Energy_Gradient( DYN_PROPERTIES )

    os.chdir(f"{DYN_PROPERTIES['VPxTB_RUNNING_DIR']}")

    return DYN_PROPERTIES

if ( __name__ == "__main__" ):
    main()
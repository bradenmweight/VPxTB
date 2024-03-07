import numpy as np
import subprocess as sp
import os

import polariton

def read():

    DYN_PROPERTIES = {} # Everything will come from here

    # Read input file
    input_lines = open('input.in','r').readlines()
    for count, line in enumerate(input_lines):
        ### Clean line and Check for comments ###
        t = line.split()
        if ( len(t) == 0 or line.split()[0] in ["#","!"] ): continue # Check for comment line
        t = [ j.strip() for j in line.split("=") ]
        tnew = []
        for tj in t:
            if ( "#" in tj.split() ):
                tnew.append( tj.split()[:tj.split().index("#")][0] )
                break
            if ( "!" in tj.split() ):
                tnew.append( tj.split()[:tj.split().index("!")][0] )
                break
            else:
                tnew.append(tj)
        t = tnew

        if ( len(t) == 2 ):

            # Look for NSteps
            if ( t[0].upper() == "do_HESSIAN".upper() ):
                try:
                    if ( t[1].upper() in ["TRUE", "1"] ):
                        DYN_PROPERTIES["do_HESSIAN"] = True
                    else:
                        DYN_PROPERTIES["do_HESSIAN"] = False
                except ValueError:
                    print(f"\t'HESSIAN' must be either (1,0) or (True, False): '{t[1]}'")
                    exit()

            # Look for NSteps
            if ( t[0].upper() == "nsteps".upper() ):
                try:
                    DYN_PROPERTIES["NSteps"] = int( t[1] )
                except ValueError:
                    print(f"\t'NSteps' must be an integer: '{t[1]}'")
                    exit()

            # Look for dtI
            if ( t[0].upper() == "dtI".upper() ):
                try:
                    DYN_PROPERTIES["dtI"] = float( t[1] )*41.341
                except:# ValueError:
                    print(f"\t'dtI' must be supplied:")
                    exit()

            # Look for NCPUS
            if ( t[0].upper() == "NCPUS".upper() ):
                try:
                    DYN_PROPERTIES["NCPUS"] = int( t[1] )
                except ValueError:
                    print(f"\t'NCPUS' must be an integer: '{t[1]}'")
                    exit()

            # Look for CHARGE
            if ( t[0].upper() == "CHARGE".upper() ):
                try:
                    DYN_PROPERTIES["CHARGE"] = int( t[1] )
                except ValueError:
                    print(f"\t'CHARGE' must be an integer: '{t[1]}'")
                    exit()

            # Look for VELOC
            if ( t[0].upper() == "VELOC".upper() ):
                DYN_PROPERTIES["VELOC"] = t[1].upper()
                # Later, we will check this input.

            # Look for PARALLEL_GRADIENT
            if ( t[0].upper() == "PARALLEL_GRADIENT".upper() ):
                try:
                    if ( t[1].lower() in ["true", "1"] ):
                        DYN_PROPERTIES["PARALLEL_GRADIENT"] = True
                    else:
                        DYN_PROPERTIES["PARALLEL_GRADIENT"] = False
                except ValueError:
                    print("Input for 'PARALLEL_GRADIENT' must be a boolean. (True or False)")
                    exit()

            # Look for MD_ENSEMBLE
            if ( t[0].upper() == "MD_ENSEMBLE".upper() ):
                DYN_PROPERTIES["MD_ENSEMBLE"] = t[1].upper()
                if ( DYN_PROPERTIES["MD_ENSEMBLE"] not in ["NVT","NVE"] ):
                    print("Input for 'MD_ENSEMBLE' must be either 'NVT' or 'NVE'.")
                    exit()

            # Look for NVT_TYPE
            if ( t[0].upper() == "NVT_TYPE".upper() ):
                DYN_PROPERTIES["NVT_TYPE"] = t[1].upper()
                if ( DYN_PROPERTIES["NVT_TYPE"] not in ["LANGEVIN", "RESCALE"] ):
                    print("Input for 'NVT_TYPE' must be 'LANGEVIN' or 'RESCALE'.")
                    exit()

            # Look for LANGEVIN_LAMBDA
            if ( t[0].upper() == "LANGEVIN_LAMBDA".upper() ):
                try:
                    DYN_PROPERTIES["LANGEVIN_LAMBDA"] = float( t[1] )
                except ValueError:
                    print(f"\t'LANGEVIN_LAMBDA' must be a float: '{t[1]}'")
                    exit()
                assert( DYN_PROPERTIES["LANGEVIN_LAMBDA"] >= 0 ), f"'LANGEVIN_LAMBDA' must be greater than or equal to 0.0: {DYN_PROPERTIES['LANGEVIN_LAMBDA']}"

            # Look for RESCALE_FREQ
            if ( t[0].upper() == "RESCALE_FREQ".upper() ):
                try:
                    DYN_PROPERTIES["RESCALE_FREQ"] = int( t[1] )
                except ValueError:
                    print(f"\t'RESCALE_FREQ' must be an integer: '{t[1]}'")
                    exit()
                assert( DYN_PROPERTIES["RESCALE_FREQ"] >= 0 ), f"'RESCALE_FREQ' must be greater than or equal to 0: {DYN_PROPERTIES['RESCALE_FREQ']}"


            # Look for TEMP
            if ( t[0].upper() == "TEMP".upper() ):
                try:
                    DYN_PROPERTIES["TEMP"] = float( t[1] )
                except ValueError:
                    print(f"\t'TEMP' must be a float: '{t[1]}'")
                    exit()
                assert( DYN_PROPERTIES["TEMP"] >= 0 ), f"'TEMP' must be greater than or equal to 0.0: {DYN_PROPERTIES['TEMP']}"


            # Look for DATA_SAVE_FREQ
            if ( t[0].upper() == "DATA_SAVE_FREQ".upper() ):
                try:
                    DYN_PROPERTIES["DATA_SAVE_FREQ"] = int( t[1] )
                except ValueError:
                    print(f"\t'DATA_SAVE_FREQ' must be an integer: '{t[1]}'")
                    exit()


            # Look for do_POLARITON
            if ( t[0].upper() == "do_POLARITON".upper() ):
                if ( t[1].lower() in ["true", "1"] ):
                    DYN_PROPERTIES["do_POLARITON"] = True
                else:
                    DYN_PROPERTIES["do_POLARITON"] = False
            
            # Look for A0
            if ( t[0].upper() == "A0".upper() ):
                try:
                    DYN_PROPERTIES["A0"] = float( t[1] )
                except:
                    print(f"\t'A0' must be provided.")
                    exit()

            # Look for WC
            if ( t[0].upper() == "WC".upper() ):
                try:
                    DYN_PROPERTIES["WC_eV"] = float( t[1] )
                    DYN_PROPERTIES["WC_AU"] = DYN_PROPERTIES["WC_eV"] / 27.2114
                except:
                    print(f"\t'WC' must be provided.")
                    exit()

            # Look for EPOL
            if ( t[0].upper() == "EPOL".upper() ):
                try:
                    TMP = t[1]
                    DYN_PROPERTIES["EPOL"] = np.array([TMP[0],TMP[1],TMP[2]]).astype(float)
                    DYN_PROPERTIES["EPOL"] = DYN_PROPERTIES["EPOL"] / np.linalg.norm(DYN_PROPERTIES["EPOL"])
                except ValueError:
                    print(f"\t'EPOL' must be provided.")
                    exit()


        else:
            print( f"Error: Input is wrong at line {count+1}: {line}" )
            print( f"\tToo many '='" )
            exit()

    try:
        print("MANDATORY INPUT VARIABLES:")
        print( "  NSteps ="); print("\t\t", DYN_PROPERTIES["NSteps"] )
        print( "  dtI ="); print("\t\t", DYN_PROPERTIES["dtI"]/41.341, "(fs)" )
        print( "  CHARGE ="); print("\t\t", DYN_PROPERTIES["CHARGE"] )
        print( "  MD_ENSEMBLE ="); print("\t\t", DYN_PROPERTIES["MD_ENSEMBLE"] )
        print( "  VELOC ="); print("\t\t", DYN_PROPERTIES["VELOC"] )
        print( "  do_POLARITON ="); print("\t\t", DYN_PROPERTIES["do_POLARITON"] )
        print( "  A0 ="); print("\t\t", DYN_PROPERTIES["A0"], "a.u." )
        print( "  WC ="); print("\t\t", DYN_PROPERTIES["WC_eV"], "eV" )
    except KeyError:
        print("Input file is missing a mandatory entry (see above). Check it.")
        exit()

    #assert( DYN_PROPERTIES["ISTATE"] <= DYN_PROPERTIES["NStates"]-1 ), "ISTATE must be less than the total number of states."

    # Try to read the VPxTB_HOME path
    VPxTB_HOME_PATH = sp.check_output("echo $VPxTB_HOME",shell=True).decode().strip("\n")
    if ( VPxTB_HOME_PATH == "" ):
        print("SQD home path is not set.\n\texport VPxTB_HOME=/absolute/path/to/VPxTB/")
        exit()
    else:
        DYN_PROPERTIES["VPxTB_HOME_PATH"] = VPxTB_HOME_PATH

    # Try to read the VPxTB_SCRATCH path
    VPxTB_SCRATCH = sp.check_output("echo $VPxTB_SCRATCH",shell=True).decode().strip("\n")
    if ( VPxTB_SCRATCH == "" ):
        print("SQD scratch path is not set.\n\texport VPxTB_SCRATCH=/absolute/path/to/VPxTB_SCRATCH/")
        exit()
    else:
        DYN_PROPERTIES["VPxTB_SCRATCH_PATH"] = VPxTB_SCRATCH
    
    # Save running directory
    DYN_PROPERTIES["VPxTB_RUNNING_DIR"] = os.getcwd()
    
    
    return DYN_PROPERTIES



def read_geom():
    XYZ_File = open("geometry_input.xyz","r").readlines()
    NAtoms = int(XYZ_File[0])
    Atom_labels = []
    Atom_coords_new = np.zeros(( NAtoms, 3 ))
    for count, line in enumerate(XYZ_File[2:]):
        t = line.split()
        Atom_labels.append( t[0] )
        Atom_coords_new[count,:] = np.array([ float(t[1]), float(t[2]), float(t[3]) ]) / 0.529 # Ang -> a.u.

    assert( int(XYZ_File[0]) == len(Atom_labels) ), "Number of atoms incorrect in input XYZ file."
    assert( int(XYZ_File[0]) == len(Atom_coords_new) ), "Number of atoms incorrect in input XYZ file."

    return Atom_labels, Atom_coords_new

def read_veloc():
    """
    TODO Add checks for XYZ user input
    """
    XYZ_File = open("velocity_input.xyz","r").readlines()
    NAtoms = int(XYZ_File[0])
    Atom_velocs_new = np.zeros(( NAtoms, 3 ))
    for count, line in enumerate(XYZ_File[2:]):
        t = line.split()
        Atom_velocs_new[count,:] = np.array([ float(t[1]), float(t[2]), float(t[3]) ]) / 0.529 / 41.341 # Ang -> a.u.

    return Atom_velocs_new

def set_masses(Atom_labels):
    mass_amu_to_au = 1837/1.007 # au / amu
    masses_amu = \
{"H":   1.00797,
"He":	4.00260,
"Li":	6.941,
"Be":	9.01218,
"B":    10.81,
"C":    12.011,
"N":    14.0067,
"O":    15.9994,
"F":    18.998403,
"Ne":	20.179,
"Na":	22.98977,
"Mg":	24.305,
"Al":	26.98154,
"Si":	28.0855,
"P":    30.97376,
"S":    32.06,
"Cl":	35.453,
"K":    39.0983,
"Ar":	39.948,
"Ca":	40.08,
"Sc":	44.9559,
"Ti":	47.90,
"V":    50.9415,
"Cr":	51.996,
"Mn":	54.9380,
"Fe":	55.847,
"Ni":	58.70,
"Co":	58.9332,
"Cu":	63.546,
"Zn":	65.38,
"Ga":	69.72,
"Ge":	72.59,
"As":	74.9216,
"Se":	78.96,
"Br":	79.904,
"Kr":	83.80,
"Rb":	85.4678,
"Sr":	87.62,
"Y":    88.9059,
"Zr":	91.22,
"Nb":	92.9064,
"Mo":	95.94,
"Ru":	101.07,
"Rh":	102.9055,
"Pd":	106.4,
"Ag":	107.868,
"Cd":	112.41,
"In":	114.82,
"Sn":	18.69,
"Sb":	121.75,
"I":    126.9045,
"Te":	127.60,
"Xe":	131.30,
"Cs":	132.9054,
"Ba":	137.33,
"La":	138.9055,
"Ce":	140.12,
"Pr":	140.9077,
"Nd":	144.24,
"Sm":	150.4,
"Eu":	151.96,
"Gd":	157.25,
"Tb":	158.9254,
"Dy":	162.50,
"Ho":	164.9304,
"Er":	167.26,
"Tm":	168.9342,
"Yb":	173.04,
"Lu":	174.967,
"Hf":	178.49,
"Ta":	180.9479,
"W":    183.85,
"Re":	186.207,
"Os":	190.2,
"Ir":	192.22,
"Pt":	195.09,
"Au":	196.9665,
"Hg":	200.59,
"Tl":	204.37,
"Pb":	207.2,
"Bi":	208.9804,
"Ra":	226.0254}


    masses = []
    for at in Atom_labels:
        masses.append( masses_amu[at] )
    return np.array(masses) * mass_amu_to_au

def get_initial_velocs(DYN_PROPERTIES):
    
    Atom_labels = DYN_PROPERTIES["Atom_labels"]
    masses      = DYN_PROPERTIES["MASSES"]

    # TODO Get Wigner distribution for initial velocities

    if ( DYN_PROPERTIES["VELOC"] == "MB" ):
        import random
        velocs = np.zeros(( len(Atom_labels), 3 ))
        T = 300 # K
        kT  = T * (0.025/300) / 27.2114 # K -> eV -> au
        V0  = np.sqrt( kT / masses )
        SIG = kT / masses
        for at,atom in enumerate(Atom_labels):
            for d in range(3):
                velocs[at,d] = random.gauss( V0[at], SIG[at] )
    
    elif (DYN_PROPERTIES["VELOC"] == "ZERO"):
        velocs = np.zeros(( len(Atom_labels), 3 ))
    
    elif ( DYN_PROPERTIES["VELOC"] == "READ" ): # This will be usual way to perform with Wigner for now.
        velocs = read_veloc() # Reads "velocity_input.xyz"

    else:
        assert(False), "Initial velocities not specified properly.\t Must be 'ZERO', 'MB', or 'RED'."

    return velocs

def initialize_MD_variables(DYN_PROPERTIES):
    
    DYN_PROPERTIES["MD_STEP"] = 0    
    DYN_PROPERTIES["Atom_labels"], DYN_PROPERTIES["Atom_coords_new"] = read_geom()
    DYN_PROPERTIES["NAtoms"] = len( DYN_PROPERTIES["Atom_labels"] )
    DYN_PROPERTIES["MASSES"] = set_masses(DYN_PROPERTIES["Atom_labels"])
    DYN_PROPERTIES["Atom_velocs_new"] = get_initial_velocs(DYN_PROPERTIES)

    try:
        tmp = DYN_PROPERTIES["NCPUS"]
    except KeyError:
        DYN_PROPERTIES["NCPUS"] = 1

    try:
        tmp = DYN_PROPERTIES["REMOVE_COM_MOTION"]
    except KeyError:
        DYN_PROPERTIES["REMOVE_COM_MOTION"] = True # Default is to remove COM motion

    try:
        tmp = DYN_PROPERTIES["REMOVE_ANGULAR_VELOCITY"]
    except KeyError:
        DYN_PROPERTIES["REMOVE_ANGULAR_VELOCITY"] = True # Default is to remove angular velocity

    try:
        tmp = DYN_PROPERTIES["do_POLARITON"]
    except KeyError:
        DYN_PROPERTIES["do_POLARITON"] = False # Set to False by default
    ### MOVED THIS TO SOMEWHERE ELSE ###
    #if ( DYN_PROPERTIES["do_POLARITON"] == True ):
    #    DYN_PROPERTIES = polariton.initialize_Cavity( DYN_PROPERTIES )


    try:
        tmp = DYN_PROPERTIES["PARALLEL_GRADIENT"]
    except KeyError:
        DYN_PROPERTIES["PARALLEL_GRADIENT"] = False # Set to False by default


    if ( DYN_PROPERTIES["MD_ENSEMBLE"] == "NVT" ):
        try:
            tmp = DYN_PROPERTIES["NVT_TYPE"]
        except KeyError:
            assert(False), f"\t'NVT_TYPE' needs to be defined if 'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']}"
        
        try:
            tmp = DYN_PROPERTIES["LANGEVIN_LAMBDA"]
        except KeyError:
            if( DYN_PROPERTIES["NVT_TYPE"] == "LANGEVIN" ):
                assert(False), f"\t'LANGEVIN_LAMBDA' needs to be defined if 'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']}"
            elif( DYN_PROPERTIES["NVT_TYPE"] == "RESCALE" ):
                DYN_PROPERTIES["LANGEVIN_LAMBDA"] = None
            else:
                assert(False), f"\t'MD_ENSEMBLE', 'NVT_TYPE', and 'RESCALE_FREQ' need to be consistent."
        
        try:
            tmp = DYN_PROPERTIES["RESCALE_FREQ"]
        except KeyError:
            if( DYN_PROPERTIES["NVT_TYPE"] == "LANGEVIN" ):
                DYN_PROPERTIES["RESCALE_FREQ"] = None
            elif( DYN_PROPERTIES["NVT_TYPE"] == "RESCALE" ):
                assert(False), f"\t'RESCALE_FREQ' needs to be defined if 'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']}"
            else:
                assert(False), f"\t'MD_ENSEMBLE', 'NVT_TYPE', and 'RESCALE_FREQ' need to match."

        try:
            tmp = DYN_PROPERTIES["TEMP"]
        except KeyError:
            assert(False), f"\t'TEMP' needs to be defined if 'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']}"

    elif ( DYN_PROPERTIES["MD_ENSEMBLE"] == "NVE" ):
        try:
            tmp = DYN_PROPERTIES["NVT_TYPE"]
        except KeyError:
            DYN_PROPERTIES["NVT_TYPE"] = None
        assert(DYN_PROPERTIES["NVT_TYPE"] == None ), f"\n\t'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']} and 'NVT_TYPE' = {DYN_PROPERTIES['NVT_TYPE']} are not compatible.\n\t 'NVT_TYPE' should not appear as we are trying to do NVE dynamics !"
        
        try:
            tmp = DYN_PROPERTIES["LANGEVIN_LAMBDA"]
        except KeyError:
            DYN_PROPERTIES["LANGEVIN_LAMBDA"] = None
        assert(DYN_PROPERTIES["LANGEVIN_LAMBDA"] == None ), f"\n\t'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']} and 'LANGEVIN_LAMBDA' = {DYN_PROPERTIES['LANGEVIN_LAMBDA']} are not compatible.\n\t'LANGEVIN_LAMBDA' should not appear as we are trying to do NVE dynamics !"
        
        try:
            tmp = DYN_PROPERTIES["TEMP"]
        except KeyError:
            DYN_PROPERTIES["TEMP"] = None
        assert(DYN_PROPERTIES["TEMP"] == None ), f"\n\t'MD_ENSEMBLE' = {DYN_PROPERTIES['MD_ENSEMBLE']} and 'TEMP' = {DYN_PROPERTIES['TEMP']} are not compatible.\n\t'TEMP' should not appear as we are trying to do NVE dynamics !"
    
    else:
        assert(False), "'MD_ENSEMBLE' needs to be defined as either 'NVE' or 'NVT'."
    


    try:
        tmp = DYN_PROPERTIES["DATA_SAVE_FREQ"]
    except KeyError:
        DYN_PROPERTIES["DATA_SAVE_FREQ"] = 1 # Default is to save every step. Might make large output files for NVT

    try:
        tmp = DYN_PROPERTIES["do_HESSIAN"]
    except KeyError:
        DYN_PROPERTIES["do_HESSIAN"] = False # Default is not to compute the Hessian at each step



    print("Input looks good.")
    return DYN_PROPERTIES
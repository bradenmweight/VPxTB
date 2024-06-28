import numpy as np
import sys
import subprocess as sp
from time import time

import read_input
import nuclear_propagation
import polariton
import output
import rotation
import xTB

def main( ):
    DYN_PROPERTIES = read_input.read()
    DYN_PROPERTIES = read_input.initialize_MD_variables(DYN_PROPERTIES)

    # Remove COM motion and angular velocity
    # Do we need to do this at every step. Probably should at least remove COM.
    #if ( DYN_PROPERTIES["REMOVE_COM_MOTION"] == True ):
    #    DYN_PROPERTIES = rotation.shift_COM(DYN_PROPERTIES)
    #if ( DYN_PROPERTIES["REMOVE_ANGULAR_VELOCITY"] == True ):
    #    DYN_PROPERTIES = rotation.remove_rotations(DYN_PROPERTIES)

    # Perform first electronic structure calculation
        # Get diagonal energies and gradients
    DYN_PROPERTIES = xTB.main(DYN_PROPERTIES)
    # Initialize photon based on moolecular dipole. We have dipole here.
    if ( DYN_PROPERTIES["do_POLARITON"] == True ):
        DYN_PROPERTIES = polariton.initialize_Cavity( DYN_PROPERTIES )
    
    output.save_data(DYN_PROPERTIES)

    # Start main MD loop
    for step in range( DYN_PROPERTIES["NSteps"] ):
        T_STEP_START = time()
        #print(f"Working on step {step} of { DYN_PROPERTIES['NSteps'] }")

        DYN_PROPERTIES = nuclear_propagation.Nuclear_X_Step(DYN_PROPERTIES) # Propagate nuclear coordinates
        if ( DYN_PROPERTIES["do_POLARITON"] ): 
            DYN_PROPERTIES = polariton.propagate_Polariton( DYN_PROPERTIES ) # Propagate polariton coordinates (QC and PC)

        # Perform jth electronic structure calculation
            # Get diagonal energies and grad
        DYN_PROPERTIES["MD_STEP"] += 1 # This needs to be exactly here for technical reasons.
        T0 = time()
        DYN_PROPERTIES = xTB.main(DYN_PROPERTIES)
        print( "Total QM took %2.2f s." % (time() - T0) )

        # Propagate nuclear momenta
        DYN_PROPERTIES = nuclear_propagation.Nuclear_V_Step(DYN_PROPERTIES)

        # Remove COM motion and angular velocity
        # Do we need to do this at every step ? Probably should at least remove COM.
        # if ( DYN_PROPERTIES["REMOVE_COM_MOTION"] == True ):
        #     DYN_PROPERTIES = rotation.shift_COM(DYN_PROPERTIES)
        # if ( DYN_PROPERTIES["REMOVE_ANGULAR_VELOCITY"] == True ):
        #     DYN_PROPERTIES = rotation.remove_rotations(DYN_PROPERTIES)

        if ( DYN_PROPERTIES["MD_STEP"] % DYN_PROPERTIES["DATA_SAVE_FREQ"]  == 0 ):
            output.save_data(DYN_PROPERTIES)


        print( "Total MD Step took %2.2f s." % (time() - T_STEP_START) )

    # Remove VPxTB_SCRATCH_PATH
    print("\nRemoving scratch files.")
    sp.call(f"rm -r {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)

if ( __name__ == "__main__" ):
    main()
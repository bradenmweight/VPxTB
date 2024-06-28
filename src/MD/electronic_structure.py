import numpy as np
import subprocess as sp
import os
import multiprocessing as mp

import xTB, G16


def main( DYN_PROPERTIES ):

    if ( not os.path.exists(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE") ):
        sp.call(f"mkdir {DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE", shell=True)
    os.chdir(f"{DYN_PROPERTIES['VPxTB_SCRATCH_PATH']}/EL_STRUCTURE")

    # Do single-point calculation
        # Get GS energy
        # Get GS electronic forces
    if ( DYN_PROPERTIES["EL_PACKAGE"] == "XTB" ):
        DYN_PROPERTIES = xTB.get_Energy_Gradient( DYN_PROPERTIES )
    else:
        DYN_PROPERTIES = G16.get_Energy_Gradient( DYN_PROPERTIES )

    os.chdir(f"{DYN_PROPERTIES['VPxTB_RUNNING_DIR']}")

    return DYN_PROPERTIES


if ( __name__ == "__main__" ):
    main()
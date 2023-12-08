import numpy as np
import subprocess as sp

import properties

def save_data(DYN_PROPERTIES):

    TIME    = round(DYN_PROPERTIES["MD_STEP"] * DYN_PROPERTIES["dtI"] / 41.341 ,4) # Print in fs

    if ( DYN_PROPERTIES["MD_STEP"] == 0 ):
        sp.call("rm -r MD_OUTPUT", shell=True)
        sp.call("mkdir MD_OUTPUT", shell=True)

    with open("MD_OUTPUT/trajectory.xyz","a") as file01:
        file01.write(f"{DYN_PROPERTIES['NAtoms']}\n")
        file01.write(f"MD Step {DYN_PROPERTIES['MD_STEP']} Units = [Angstroms]\n")
        Atom_labels = DYN_PROPERTIES["Atom_labels"]
        Atom_coords = DYN_PROPERTIES["Atom_coords_new"] * 0.529 # Convert to Angstroms
        for count, atom in enumerate( Atom_labels ):
            #file01.write(f"{atom}\t{Atom_coords[count,0]*0.529}\t{Atom_coords[count,1]*0.529}\t{Atom_coords[count,2]*0.529}\n")
            file01.write(f"{atom}  " + " ".join(map("{:2.8f}".format,Atom_coords[count,:]))  + "\n")

    with open("MD_OUTPUT/velocity.xyz","a") as file01:
        file01.write(f"{DYN_PROPERTIES['NAtoms']}\n")
        file01.write(f"MD Step {DYN_PROPERTIES['MD_STEP']} Units = [Angstroms / fs]\n")
        Atom_labels = DYN_PROPERTIES["Atom_labels"]
        Atom_velocs = DYN_PROPERTIES["Atom_velocs_new"] * 0.529 * 41.341 # 0.529 Ang./Bohr, 41.341 a.u. / fs
        for count, atom in enumerate( Atom_labels ):
            file01.write(f"{atom}  " + " ".join(map("{:2.8f}".format,Atom_velocs[count,:]))  + "\n") # Ang / fs

    with open("MD_OUTPUT/forces.xyz","a") as file01:
        file01.write(f"{DYN_PROPERTIES['NAtoms']}\n")
        file01.write(f"MD Step {TIME} Units = [eV / Ang.]\n")
        Atom_labels = DYN_PROPERTIES["Atom_labels"]
        Atom_forces = -1 * DYN_PROPERTIES["GRAD_NEW"] / 0.529 * 27.2114 # 0.529 Ang./Bohr, 27.2114 eV / Hartree
        for count, atom in enumerate( Atom_labels ):
            file01.write(f"{atom}  " + " ".join(map("{:2.8f}".format,Atom_forces[count,:]))  + "\n") # eV / Ang.


    with open("MD_OUTPUT/PES.dat","a") as file01:
        file01.write( f"{TIME}  {np.round(DYN_PROPERTIES['ENERGY_NEW']*27.2114,8)}\n" )

    with open("MD_OUTPUT/Energy.dat","a") as file01:
        
        DYN_PROPERTIES = properties.compute_KE(DYN_PROPERTIES)
        DYN_PROPERTIES = properties.compute_PE(DYN_PROPERTIES)

        KE = DYN_PROPERTIES["KE"] * 27.2114
        PE = DYN_PROPERTIES["PE"] * 27.2114
        TE = KE + PE

        file01.write(f"{TIME}  " + "%2.6f  %2.6f  %2.6f\n" % (KE,PE,TE))

    with open("MD_OUTPUT/Temperature.dat","a") as file01:
        T = properties.compute_Temperature(DYN_PROPERTIES) # k
        file01.write(f"{TIME}  " + "%2.4f\n" % (T))




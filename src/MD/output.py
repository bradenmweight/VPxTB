import numpy as np
import subprocess as sp
import os

import properties

def save_data(DYN_PROPERTIES):

    TIME    = round(DYN_PROPERTIES["MD_STEP"] * DYN_PROPERTIES["dtI"] / 41.341 ,4) # Print in fs

    if ( DYN_PROPERTIES["MD_STEP"] == 0 ):
        if ( os.path.exists("MD_OUTPUT") ): 
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

    with open("MD_OUTPUT/Dipole.dat","a") as file01:
        MU = DYN_PROPERTIES["DIPOLE"]
        file01.write(f"{TIME}  " + "%2.6f %2.6f %2.6f\n" % (MU[0], MU[1], MU[2]))

    if ( DYN_PROPERTIES["do_POLARITON"] == True ):
        with open("MD_OUTPUT/QC.dat","a") as file01:
            QC = DYN_PROPERTIES["QC"]
            file01.write(f"{TIME}  " + "%2.6f\n" % (QC))
        
        with open("MD_OUTPUT/PC.dat","a") as file01:
            PC = DYN_PROPERTIES["PC"]
            file01.write(f"{TIME}  " + "%2.6f\n" % (PC))

def saveNM( DYN_PROPERTIES ):
    if ( DYN_PROPERTIES["MD_STEP"] == 0 ):
        if ( os.path.exists("MD_OUTPUT") ): 
            sp.call("rm -r MD_OUTPUT", shell=True)
        sp.call("mkdir MD_OUTPUT", shell=True)
    w = DYN_PROPERTIES["NM_FREQUENCIES"]
    OUTPUT = np.array([w, w * 27.2114 * 1000, w * 27.2114 * 1000 / 0.123983])
    np.savetxt("MD_OUTPUT/Normal_Mode_Frequencies.dat", OUTPUT.T, fmt='%2.6f', header="a.u., meV, cm^-1")

    U      = DYN_PROPERTIES["NM_WAVEFUNCTIONS"]
    COORDS = DYN_PROPERTIES["Atom_coords_new"] * 0.529 # Convert to Angstroms
    with open("MD_OUTPUT/Normal_Mode_Wavefunctions.dat","w") as file01:
        for m in range( len(U[0,0,:]) ):
            file01.write("Mode %d FREQ = %1.4f cm^-1\n" % (m+1, w[m]* 27.2114 * 1000 / 0.123983))
            for at in range( len(U[:]) ):
                file01.write(f"{COORDS[at,0]:2.5f} {COORDS[at,1]:2.5f} {COORDS[at,2]:2.5f}  {U[at,0,m]:2.5f} {U[at,1,m]:2.5f} {U[at,2,m]:2.5f}\n")
            file01.write("\n")


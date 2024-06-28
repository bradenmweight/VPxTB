Setting paths in slurm:
  VPxTB_HOME      = /scratch/bweight/software/VPxTB/
  VPxTB_SCRATCH   = /local_scratch/23803866/
Traceback (most recent call last):
  File "/scratch/bweight/software/VPxTB//src/MD/main.py", line 80, in <module>
    main()
  File "/scratch/bweight/software/VPxTB//src/MD/main.py", line 17, in main
    DYN_PROPERTIES = read_input.initialize_MD_variables(DYN_PROPERTIES)
  File "/gpfs/fs2/scratch/bweight/software/VPxTB/src/MD/read_input.py", line 487, in initialize_MD_variables
    DYN_PROPERTIES["QN"] = set_Nuclear_Charges( DYN_PROPERTIES["Atom_labels"] )
  File "/gpfs/fs2/scratch/bweight/software/VPxTB/src/MD/read_input.py", line 444, in set_Nuclear_Charges
    QN[Atom_labels.index(key)] = counter
ValueError: 'H' is not in list

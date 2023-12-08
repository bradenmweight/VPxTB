# VPxTB
# Vibro-Polariton Ab Initio Dynamics with Extended Tight Binding

## Installation
### VPxTB
git clone https://github.com/bradenmweight/VPxTB
### Extended Tight Binding (S. Grimme Group)
conda install -c conda-forge xtb

## Examples
### Bare Electronic Ground State Dynamics
```
### FILE: input.in ###
NSTEPS           = 1000 # Total MD steps [int]
dtI              = 0.10 # Nuclear time-step (fs) [float]
CHARGE           = 0 # System's net charge [int]
MD_ENSEMBLE      = NVE
VELOC            = READ # "READ" -- from 'velocity_input.xyz' "ZERO" "MB" -- Maxwell-Boltzmann
DATA_SAVE_FREQ   = 10

do_POLARITON      = False
A0                = 0.0
WC                = 0.0 # eV
EPOL              = 111 # Three positive integers
```
### Polariton Dynamics
```
### FILE: input.in ###
NSTEPS           = 1000 # Total MD steps [int]
dtI              = 0.10 # Nuclear time-step (fs) [float]
CHARGE           = 0 # System's net charge [int]
MD_ENSEMBLE      = NVE
VELOC            = READ # "READ" -- from 'velocity_input.xyz' "ZERO" "MB" -- Maxwell-Boltzmann
DATA_SAVE_FREQ   = 10

do_POLARITON      = True
A0                = 0.1
WC                = 0.1 # eV
EPOL              = 111 # Three positive integers
```
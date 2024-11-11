#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '11-11-2024'
__date__ = '11-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
from pathlib import Path

#%% Functions

def WriteMain(FileName):

    L1, L2, L3 = 5.28, 5.28, 5.28
    Mesh = FileName + '.inp'
    BCs = FileName + '_KUBC.inp'

    Text = f"""
**********************************************
**
**         Main Abaqus input File
**
**       Homogenization of cubic ROI
**     
**    Mathieu Simon, ARTORG Center, 2024 
**
**********************************************
** Paramter Definition for Steps
**  (Unit Cell Dimensions l1,l2,l3=h)
**********************************************
*PARAMETER
l1  = {L1}
l2  = {L2}
l3  = {L3}
**********
l1l2=l1*l2
l2l3=l2*l3
l1l3=l1*l3
**
** Node, Element, and Material Definitons 
**********************************************
*INCLUDE, INPUT={Mesh}
**
** Interactions (*Equation and *Nset)
**********************************************
*INCLUDE, INPUT={BCs}
**
** Boundary Conditions
**********************************************
*BOUNDARY, TYPE=DISPLACEMENT
SWB, 1, 3, 0
SEB, 3, 3, 0
NWB, 3, 3, 0
NWB, 1, 1, 0
**
** Steps Definitions
***************** Tensile 1 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
SEB, 1,  <l2l3>
*NODE FILE
U
*EL FILE
S
*EL PRINT
EVOL, S, E
*END STEP
"""

    with open(FileName + '_Main.inp','w') as File:
        File.write(Text)

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/Abaqus'
        AbaqusInps = [F.name[:10] for F in Path.iterdir(DataPath) if F.name.endswith('temp.inp')]

    # Create output directory if necessary
    Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)

    for Input in AbaqusInps:
        WriteMain(Input)



if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--Sample', help='Sample main file name', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI and png image of the plot', default=Path(__file__).parents[1] / '02_Results/Scans')
    Parser.add_argument('--NROIs', help='Number of region of interests to extract', type=int, default=3)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

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
    Shift = 10
    Mesh = FileName.name[:Shift] + '_Mesh.inp'
    BCs = FileName.name[:Shift] + '_KUBC.inp'

    Text = f"""**********************************************
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
u1  = {L1/100}
u2  = {L2/100}
u3  = {L3/100}
**
** Node, Element, and Material Definitons 
**********************************************
*INCLUDE, INPUT=/home/ms20s284/FABTIB2/02_Results/Abaqus/{Mesh}
**
** Interactions (*Equation and *Nset)
**********************************************
*INCLUDE, INPUT=/home/ms20s284/FABTIB2/02_Results/Abaqus/{BCs}
**
** Steps Definitions
***************** Tensile 1 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 1, 1, <u1>
ALL_NODE_W, 1, 1, 0
ALL_NODE_S, 2, 2, 0
ALL_NODE_N, 2, 2, 0
ALL_NODE_B, 3, 3, 0
ALL_NODE_T, 3, 3, 0
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
***************** Tensile 2 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 1, 1, 0
ALL_NODE_W, 1, 1, 0
ALL_NODE_S, 2, 2, <u2>
ALL_NODE_N, 2, 2, 0
ALL_NODE_B, 3, 3, 0
ALL_NODE_T, 3, 3, 0
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
***************** Tensile 3 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 1, 1, 0
ALL_NODE_W, 1, 1, 0
ALL_NODE_S, 2, 2, 0
ALL_NODE_N, 2, 2, 0
ALL_NODE_B, 3, 3, <u3>
ALL_NODE_T, 3, 3, 0
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
****************** Shear 23 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 1, 1, 0
ALL_NODE_W, 1, 1, 0
ALL_NODE_S, 3, 3, 0
ALL_NODE_N, 3, 3, <u3>
ALL_NODE_B, 2, 2, 0
ALL_NODE_T, 2, 2, <u2>
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
****************** Shear 13 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 3, 3, 0
ALL_NODE_W, 3, 3, <u3>
ALL_NODE_S, 2, 2, 0
ALL_NODE_N, 2, 2, 0
ALL_NODE_B, 1, 1, 0
ALL_NODE_T, 1, 1, <u1>
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
****************** Shear 21 ******************
*STEP
*STATIC
*BOUNDARY, OP=NEW
ALL_NODE_E, 2, 2, 0
ALL_NODE_W, 2, 2, <u2>
ALL_NODE_S, 1, 1, 0
ALL_NODE_N, 1, 1, <u1>
ALL_NODE_B, 3, 3, 0
ALL_NODE_T, 3, 3, 0
** Element Output 
*OUTPUT, FIELD
*ELEMENT OUTPUT
IVOL, S, E
*END STEP
"""



    with open(FileName.parent / (FileName.name[:Shift] + '_Main.inp'),'w') as File:
        File.write(Text)


    # Check that no step is written in the mesh file
    with open(FileName.parent / Mesh, 'r') as File:
        Text = File.read()
    StepStart = Text.find('*STEP')

    # If yes, remove it
    if StepStart > 0:
        with open(FileName.parent / Mesh, 'w') as File:
            File.write(Text[:StepStart])

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/Abaqus/'
        AbaqusInps = [F for F in Path.iterdir(DataPath) if F.name.endswith('temp.inp')]

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

#%%
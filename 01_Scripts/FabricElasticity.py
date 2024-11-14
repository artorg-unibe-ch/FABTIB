#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '12-11-2024'
__date__ = '12-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

#%% Functions

def Engineering2MandelNotation(A):

    B = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            if i < 3 and j >= 3:
                B[i,j] = A[i,j] * np.sqrt(2)
            elif i >= 3 and j < 3:
                B[i,j] = A[i,j] * np.sqrt(2)
            elif i >= 3 and j >= 3:
                B[i,j] = A[i,j] * 2
            else:
                B[i, j] = A[i, j]

    return B

#%% Main

def Main(Arguments):

    MorphoPath = Path(__file__).parents[1] / '02_Results/Morphometry'
    AbaqusPath = Path(__file__).parents[1] / '02_Results/Abaqus'

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        Samples = sorted([F.name[:-4] for F in Path.iterdir(MorphoPath) if F.name.endswith('.csv')])


    Strain = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    
    Stress = np.array([[0.15775,  0.076343, 0.098583,  0.0026706, -0.013437, 0.013415],
                       [0.076343, 0.17785,  0.12364, -0.0047940, -0.010732,  0.019610],
                       [0.098583, 0.12364,  0.69002, -0.0095967, -0.030878,  0.018034],
                       [0.023547, 0.0064413, 0.0099294, 0.090003, 0.010028, -0.0020072],
                       [-0.0031354, -0.00081684, -0.011264,  0.0025182, 0.047505, 0.0025298],
                       [0.0026288, -0.0011353, 0.0025323, -0.0054995, -0.00085062, 0.028082]])
        
    for s, Sample in enumerate(Samples):

        Morpho = pd.read_csv(MorphoPath / (Sample + '.csv'), delimiter=';')
        Abaqus = open(AbaqusPath / (Sample + '.out'), 'r').readlines()

        Stress = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stress[i,j] = float(Abaqus[i+4].split()[j+1])

        Stiffness = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stiffness[i,j] = Stress[j,i] / Strain[j]

        # Symetrize matrix
        Stiffness = 1/2 * (Stiffness + Stiffness.T)

        # Transform into mandel notation
        Mandel = Engineering2MandelNotation(Stiffness)
        
        S11 = Stiffness[0,0]
        S22 = Stiffness[1,1]
        S33 = Stiffness[2,2]
        S32 = Stiffness[2,1]
        S13 = Stiffness[0,2]
        S21 = Stiffness[1,0]
        S44 = Stiffness[3,3]
        S55 = Stiffness[4,4]
        S66 = Stiffness[5,5]
        S23 = Stiffness[1,2]
        S31 = Stiffness[2,0]
        S12 = Stiffness[0,1]



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

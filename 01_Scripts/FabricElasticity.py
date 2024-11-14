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

def IsoMorphism66_3333(A):

    # Check symmetry
    Symmetry = True
    for i in range(6):
        for j in range(6):
            if not A[i,j] == A[j,i]:
                Symmetry = False
                break
    if Symmetry == False:
        print('Matrix is not symmetric!')
        return

    B = np.zeros((3,3,3,3))

    # Build 4th tensor
    B[0, 0, 0, 0] = A[0, 0]
    B[1, 1, 0, 0] = A[1, 0]
    B[2, 2, 0, 0] = A[2, 0]
    B[1, 2, 0, 0] = A[3, 0] / np.sqrt(2)
    B[2, 0, 0, 0] = A[4, 0] / np.sqrt(2)
    B[0, 1, 0, 0] = A[5, 0] / np.sqrt(2)

    B[0, 0, 1, 1] = A[0, 1]
    B[1, 1, 1, 1] = A[1, 1]
    B[2, 2, 1, 1] = A[2, 1]
    B[1, 2, 1, 1] = A[3, 1] / np.sqrt(2)
    B[2, 0, 2, 1] = A[4, 1] / np.sqrt(2)
    B[0, 1, 2, 1] = A[5, 1] / np.sqrt(2)

    B[0, 0, 2, 2] = A[0, 2]
    B[1, 1, 2, 2] = A[1, 2]
    B[2, 2, 2, 2] = A[2, 2]
    B[1, 2, 2, 2] = A[3, 2] / np.sqrt(2)
    B[2, 0, 2, 2] = A[4, 2] / np.sqrt(2)
    B[0, 1, 2, 2] = A[5, 2] / np.sqrt(2)

    B[0, 0, 1, 2] = A[0, 3] / np.sqrt(2)
    B[1, 1, 1, 2] = A[1, 3] / np.sqrt(2)
    B[2, 2, 1, 2] = A[2, 3] / np.sqrt(2)
    B[1, 2, 1, 2] = A[3, 3] / 2
    B[2, 0, 1, 2] = A[4, 3] / 2
    B[0, 1, 1, 2] = A[5, 3] / 2

    B[0, 0, 2, 0] = A[0, 4] / np.sqrt(2)
    B[1, 1, 2, 0] = A[1, 4] / np.sqrt(2)
    B[2, 2, 2, 0] = A[2, 4] / np.sqrt(2)
    B[1, 2, 2, 0] = A[3, 4] / 2
    B[2, 0, 2, 0] = A[4, 4] / 2
    B[0, 1, 2, 0] = A[5, 4] / 2

    B[0, 0, 0, 1] = A[0, 5] / np.sqrt(2)
    B[1, 1, 0, 1] = A[1, 5] / np.sqrt(2)
    B[2, 2, 0, 1] = A[2, 5] / np.sqrt(2)
    B[1, 2, 0, 1] = A[3, 5] / 2
    B[2, 0, 0, 1] = A[4, 5] / 2
    B[0, 1, 0, 1] = A[5, 5] / 2



    # Add minor symmetries ijkl = ijlk and ijkl = jikl

    B[0, 0, 0, 0] = B[0, 0, 0, 0]
    B[0, 0, 0, 0] = B[0, 0, 0, 0]

    B[0, 0, 1, 0] = B[0, 0, 0, 1]
    B[0, 0, 0, 1] = B[0, 0, 0, 1]

    B[0, 0, 1, 1] = B[0, 0, 1, 1]
    B[0, 0, 1, 1] = B[0, 0, 1, 1]

    B[0, 0, 2, 1] = B[0, 0, 1, 2]
    B[0, 0, 1, 2] = B[0, 0, 1, 2]

    B[0, 0, 2, 2] = B[0, 0, 2, 2]
    B[0, 0, 2, 2] = B[0, 0, 2, 2]

    B[0, 0, 0, 2] = B[0, 0, 2, 0]
    B[0, 0, 2, 0] = B[0, 0, 2, 0]



    B[0, 1, 0, 0] = B[0, 1, 0, 0]
    B[1, 0, 0, 0] = B[0, 1, 0, 0]

    B[0, 1, 1, 0] = B[0, 1, 0, 1]
    B[1, 0, 0, 1] = B[0, 1, 0, 1]

    B[0, 1, 1, 1] = B[0, 1, 1, 1]
    B[1, 0, 1, 1] = B[0, 1, 1, 1]

    B[0, 1, 2, 1] = B[0, 1, 1, 2]
    B[1, 0, 1, 2] = B[0, 1, 1, 2]

    B[0, 1, 2, 2] = B[0, 1, 2, 2]
    B[1, 0, 2, 2] = B[0, 1, 2, 2]

    B[0, 1, 0, 2] = B[0, 1, 2, 0]
    B[1, 0, 2, 0] = B[0, 1, 2, 0]



    B[1, 1, 0, 0] = B[1, 1, 0, 0]
    B[1, 1, 0, 0] = B[1, 1, 0, 0]

    B[1, 1, 1, 0] = B[1, 1, 0, 1]
    B[1, 1, 0, 1] = B[1, 1, 0, 1]

    B[1, 1, 1, 1] = B[1, 1, 1, 1]
    B[1, 1, 1, 1] = B[1, 1, 1, 1]

    B[1, 1, 2, 1] = B[1, 1, 1, 2]
    B[1, 1, 1, 2] = B[1, 1, 1, 2]

    B[1, 1, 2, 2] = B[1, 1, 2, 2]
    B[1, 1, 2, 2] = B[1, 1, 2, 2]

    B[1, 1, 0, 2] = B[1, 1, 2, 0]
    B[1, 1, 2, 0] = B[1, 1, 2, 0]



    B[1, 2, 0, 0] = B[1, 2, 0, 0]
    B[2, 1, 0, 0] = B[1, 2, 0, 0]

    B[1, 2, 1, 0] = B[1, 2, 0, 1]
    B[2, 1, 0, 1] = B[1, 2, 0, 1]

    B[1, 2, 1, 1] = B[1, 2, 1, 1]
    B[2, 1, 1, 1] = B[1, 2, 1, 1]

    B[1, 2, 2, 1] = B[1, 2, 1, 2]
    B[2, 1, 1, 2] = B[1, 2, 1, 2]

    B[1, 2, 2, 2] = B[1, 2, 2, 2]
    B[2, 1, 2, 2] = B[1, 2, 2, 2]

    B[1, 2, 0, 2] = B[1, 2, 2, 0]
    B[2, 1, 2, 0] = B[1, 2, 2, 0]



    B[2, 2, 0, 0] = B[2, 2, 0, 0]
    B[2, 2, 0, 0] = B[2, 2, 0, 0]

    B[2, 2, 1, 0] = B[2, 2, 0, 1]
    B[2, 2, 0, 1] = B[2, 2, 0, 1]

    B[2, 2, 1, 1] = B[2, 2, 1, 1]
    B[2, 2, 1, 1] = B[2, 2, 1, 1]

    B[2, 2, 2, 1] = B[2, 2, 1, 2]
    B[2, 2, 1, 2] = B[2, 2, 1, 2]

    B[2, 2, 2, 2] = B[2, 2, 2, 2]
    B[2, 2, 2, 2] = B[2, 2, 2, 2]

    B[2, 2, 0, 2] = B[2, 2, 2, 0]
    B[2, 2, 2, 0] = B[2, 2, 2, 0]



    B[2, 0, 0, 0] = B[2, 0, 0, 0]
    B[0, 2, 0, 0] = B[2, 0, 0, 0]

    B[2, 0, 1, 0] = B[2, 0, 0, 1]
    B[0, 2, 0, 1] = B[2, 0, 0, 1]

    B[2, 0, 1, 1] = B[2, 0, 1, 1]
    B[0, 2, 1, 1] = B[2, 0, 1, 1]

    B[2, 0, 2, 1] = B[2, 0, 1, 2]
    B[0, 2, 1, 2] = B[2, 0, 1, 2]

    B[2, 0, 2, 2] = B[2, 0, 2, 2]
    B[0, 2, 2, 2] = B[2, 0, 2, 2]

    B[2, 0, 0, 2] = B[2, 0, 2, 0]
    B[0, 2, 2, 0] = B[2, 0, 2, 0]


    # Complete minor symmetries
    B[0, 2, 1, 0] = B[0, 2, 0, 1]
    B[0, 2, 0, 2] = B[0, 2, 2, 0]
    B[0, 2, 2, 1] = B[0, 2, 1, 2]

    B[1, 0, 1, 0] = B[1, 0, 0, 1]
    B[1, 0, 0, 2] = B[1, 0, 2, 0]
    B[1, 0, 2, 1] = B[1, 0, 1, 2]

    B[2, 1, 1, 0] = B[2, 1, 0, 1]
    B[2, 1, 0, 2] = B[2, 1, 2, 0]
    B[2, 1, 2, 1] = B[2, 1, 1, 2]


    # Add major symmetries ijkl = klij
    B[0, 1, 1, 1] = B[1, 1, 0, 1]
    B[1, 0, 1, 1] = B[1, 1, 1, 0]

    B[0, 2, 1, 1] = B[1, 1, 0, 2]
    B[2, 0, 1, 1] = B[1, 1, 2, 0]


    return B

def CheckMinorSymmetry(A):
    MinorSymmetry = True
    for i in range(3):
        for j in range(3):
            PartialTensor = A[:,:, i, j]
            if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                MinorSymmetry = True
            else:
                MinorSymmetry = False
                break

    if MinorSymmetry == True:
        for i in range(3):
            for j in range(3):
                PartialTensor = np.squeeze(A[i, j,:,:])
                if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                    MinorSymmetry = True
                else:
                    MinorSymmetry = False
                    break

    return MinorSymmetry

def IsoMorphism3333_66(A):

    if CheckMinorSymmetry == False:
        print('Tensor does not present minor symmetry')
    else:

        B = np.zeros((6,6))

        B[0, 0] = A[0, 0, 0, 0]
        B[0, 1] = A[0, 0, 1, 1]
        B[0, 2] = A[0, 0, 2, 2]
        B[0, 3] = np.sqrt(2) * A[0, 0, 1, 2]
        B[0, 4] = np.sqrt(2) * A[0, 0, 2, 0]
        B[0, 5] = np.sqrt(2) * A[0, 0, 0, 1]

        B[1, 0] = A[1, 1, 0, 0]
        B[1, 1] = A[1, 1, 1, 1]
        B[1, 2] = A[1, 1, 2, 2]
        B[1, 3] = np.sqrt(2) * A[1, 1, 1, 2]
        B[1, 4] = np.sqrt(2) * A[1, 1, 2, 0]
        B[1, 5] = np.sqrt(2) * A[1, 1, 0, 1]

        B[2, 0] = A[2, 2, 0, 0]
        B[2, 1] = A[2, 2, 1, 1]
        B[2, 2] = A[2, 2, 2, 2]
        B[2, 3] = np.sqrt(2) * A[2, 2, 1, 2]
        B[2, 4] = np.sqrt(2) * A[2, 2, 2, 0]
        B[2, 5] = np.sqrt(2) * A[2, 2, 0, 1]

        B[3, 0] = np.sqrt(2) * A[1, 2, 0, 0]
        B[3, 1] = np.sqrt(2) * A[1, 2, 1, 1]
        B[3, 2] = np.sqrt(2) * A[1, 2, 2, 2]
        B[3, 3] = 2 * A[1, 2, 1, 2]
        B[3, 4] = 2 * A[1, 2, 2, 0]
        B[3, 5] = 2 * A[1, 2, 0, 1]

        B[4, 0] = np.sqrt(2) * A[2, 0, 0, 0]
        B[4, 1] = np.sqrt(2) * A[2, 0, 1, 1]
        B[4, 2] = np.sqrt(2) * A[2, 0, 2, 2]
        B[4, 3] = 2 * A[2, 0, 1, 2]
        B[4, 4] = 2 * A[2, 0, 2, 0]
        B[4, 5] = 2 * A[2, 0, 0, 1]

        B[5, 0] = np.sqrt(2) * A[0, 1, 0, 0]
        B[5, 1] = np.sqrt(2) * A[0, 1, 1, 1]
        B[5, 2] = np.sqrt(2) * A[0, 1, 2, 2]
        B[5, 3] = 2 * A[0, 1, 1, 2]
        B[5, 4] = 2 * A[0, 1, 2, 0]
        B[5, 5] = 2 * A[0, 1, 0, 1]

        return B
    
def TransformTensor(A,OriginalBasis,NewBasis):

    # Build change of coordinate matrix
    O = OriginalBasis
    N = NewBasis

    Q = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Q[i,j] = np.dot(O[i,:],N[j,:])

    if A.size == 36:
        A4 = IsoMorphism66_3333(A)

    elif A.size == 81 and A.shape == (3,3,3,3):
        A4 = A

    TransformedA = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    TransformedA[i, j, k, l] += Q[m,i]*Q[n,j]*Q[o,k]*Q[p,l] * A4[m, n, o, p]
    if A.size == 36:
        TransformedA = IsoMorphism3333_66(TransformedA)

    return TransformedA

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
    
    # Stress = np.array([[0.15775,  0.076343, 0.098583,  0.0026706, -0.013437, 0.013415],
    #                    [0.076343, 0.17785,  0.12364, -0.0047940, -0.010732,  0.019610],
    #                    [0.098583, 0.12364,  0.69002, -0.0095967, -0.030878,  0.018034],
    #                    [0.023547, 0.0064413, 0.0099294, 0.090003, 0.010028, -0.0020072],
    #                    [-0.0031354, -0.00081684, -0.011264,  0.0025182, 0.047505, 0.0025298],
    #                    [0.0026288, -0.0011353, 0.0025323, -0.0054995, -0.00085062, 0.028082]])
        
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

        # Write tensor into mandel notation
        Mandel = Engineering2MandelNotation(Stiffness)

        # Transform tensor into fabric coordinate system
        I = np.eye(3)
        Q = np.array(EigenVectors)
        TS4 = TransformTensor(IsoMorphism66_3333(Mandel), I, Q)
        TransformedStiffness = IsoMorphism3333_66(TS4)

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

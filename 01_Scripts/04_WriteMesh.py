#%% !/usr/bin/env python3

Description = """
Script used to write hexahedral elements using gmsh
"""
__author__ = ['Mathieu Simon']
__date_created__ = '06-11-2024'
__date__ = '06-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
from numba import njit
from Utils import Time
from pathlib import Path

#%% Functions

@njit
def Mapping(Array):

    X, Y, Z = Array.T.shape

    # Generate nodes map
    Index = 0
    Nodes = np.zeros((Z+1,Y+1,X+1),'int')
    Coords = np.zeros((Z+1,Y+1,X+1,3),'int')
    for Zn in range(Z + 1):
        for Yn in range(Y + 1):
            for Xn in range(X + 1):
                Index += 1
                Nodes[Zn,Yn,Xn] = Index
                Coords[Zn,Yn,Xn] = [Zn, Yn, Xn]

    # Generate elements map
    Index = 0
    Elements = np.zeros((Z, Y, X),'int')
    ElementsNodes = np.zeros((Z, Y, X, 8), 'int')
    for Xn in range(X):
            for Yn in range(Y):
                for Zn in range(Z):
                    Index += 1
                    Elements[Zn, Yn, Xn] = Index
                    ElementsNodes[Zn, Yn, Xn, 0] = Nodes[Zn, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 1] = Nodes[Zn, Yn, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 2] = Nodes[Zn, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 3] = Nodes[Zn, Yn+1, Xn]
                    ElementsNodes[Zn, Yn, Xn, 4] = Nodes[Zn+1, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 5] = Nodes[Zn+1, Yn, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 6] = Nodes[Zn+1, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 7] = Nodes[Zn+1, Yn+1, Xn]

    return Nodes, Coords, Elements, ElementsNodes

# @njit
def NodesText(NCS):
    Nodes, Coords, Sep = NCS
    NText = [''.join([N, Sep[0],
                      C[0], Sep[0],
                      C[1], Sep[0],
                      C[2], Sep[1]]) for N, C in zip(Nodes, Coords)]
    NText = ''.join(NText)
    return NText

# @njit
def ElementsText(ESS):
    ElementsNumber, SortedElementsNodes, Sep = ESS
    EText = [''.join([N, Sep[0],
                      E[0], Sep[1],
                      E[1], Sep[1],
                      E[2], Sep[1],
                      E[3], Sep[1],
                      E[4], Sep[1],
                      E[5], Sep[1],
                      E[6], Sep[1],
                      E[7], Sep[2]]) for N, E in zip(ElementsNumber, SortedElementsNodes)]
    EText = ''.join(EText)

    return EText

# Main
def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])
    
    if Arguments.OutputPath:
        Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)
    else:
        Path.mkdir(Path('FabricResults'), exist_ok=True)
        

    for i, ROI in enumerate([InputROIs[0]]):

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = np.load(ROI)

        if len(np.unique(Array)) == 2:
            Array = Array.astype(int)
        else:
            print('Image is not binary')
            return
        
        # Perform mapping
        Time.Update(1/6,'Perform mapping')
        Nodes, Coords, Elements, ElementsNodes = Mapping(Array)
        NodesNeeded = np.unique(ElementsNodes[Array.astype(bool)])

        ElementsNodes = ElementsNodes[Array.astype(bool)]
        Elements = Elements[Array.astype(bool)]
        Coords = Coords[np.isin(Nodes,NodesNeeded)]
        Nodes = Nodes[np.isin(Nodes,NodesNeeded)]

        # Sort nodes according to coordinates
        Indices = np.lexsort((Coords[:,2],Coords[:,1],Coords[:,0]))
        Coords = np.round(Coords[Indices],3)[:,::-1]
        Nodes = Nodes[Indices]

        # Sort elements according to their number
        Indices = np.argsort(Elements)
        Elements = Elements[Indices]
        ElementsNodes = ElementsNodes[Indices]

        # Generate nodes text
        Time.Update(3/6,'Write nodes text')
        NodesStr = (np.arange(len(Nodes)) + 1).astype('<U32')
        CoordsStr = Coords.astype('<U32')
        Sep = np.array([' ', '\n']).astype('<U32')

        NText = NodesText([NodesStr, CoordsStr, Sep])
        NText = ''.join(NText)

        # Generate element text
        Time.Update(4/6,'Write elem. text')
        EN = (np.arange(len(Elements)) + 1).astype('<U32')
        NS = np.argsort(Nodes)
        SM = np.searchsorted(Nodes[NS], ElementsNodes)
        SEN = (NS[SM] + 1).astype('<U32')
        Sep = np.array([' 5 2 1 1 ', ' ', '\n']).astype('<U32')

        EText = ElementsText([EN, SEN, Sep])
        EText = ''.join(EText)

        # Write file
        Time.Update(5/6,'Write mesh file')
        FName = Path(Arguments.OutputPath) / (ROI.name[:-4] + '.msh')
        with open(FName,'w') as File:

            # Write heading
            File.write('$MeshFormat\n')
            File.write('2.0 0 8\n')
            File.write('$EndMeshFormat\n\n')

            # Write nodes
            File.write('$Nodes\n')
            File.write(str(len(Nodes)) + '\n')
            File.write(NText)
            File.write('$EndNodes\n\n')

            # Write elements
            File.write('$Elements\n')
            File.write(str(len(Elements)) + '\n')
            File.write(EText)
            File.write('$EndElements')

        # Print time
        Time.Process(0,f'ROI {i+1}/{len(InputROIs)} done')

    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputROI', help='File name of the binary ROI', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI fabric and png image of the plot', type=str, default='02_Results/Mesh')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)
#%%

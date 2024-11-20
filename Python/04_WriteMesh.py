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

import gmsh
import argparse
import numpy as np
from numba import njit
from Utils import Time
from pathlib import Path
import SimpleITK as sitk

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

def CleanAndSort(Array, Nodes, Coords, Elements, ElementsNodes):

    NodesNeeded = np.unique(ElementsNodes[Array.astype(bool)])

    ElementsNodes = ElementsNodes[Array.astype(bool)]
    Elements = Elements[Array.astype(bool)]
    Coords = Coords[np.isin(Nodes,NodesNeeded)]
    Nodes = Nodes[np.isin(Nodes,NodesNeeded)]

    return ElementsNodes, Elements, Coords, Nodes

#%% Main
def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.mhd')])
        
    Path.mkdir(Arguments.OutputPath, exist_ok=True)

    for i, ROI in enumerate(InputROIs):

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = sitk.GetArrayFromImage(sitk.ReadImage(ROI)).T
        Array = np.array(Array - 1,int)

        # Perform mapping
        Time.Update(1/5,'Get Nodes Map')
        Nodes, Coords, Elements, ElementsNodes = Mapping(Array)
        Time.Update(2/5,'Remove 0 voxels')
        ElementsNodes, Elements, Coords, Nodes = CleanAndSort(Array, Nodes, Coords, Elements, ElementsNodes)

        NodeTags = np.arange(len(Nodes)) + 1
        ElementsTags = np.arange(len(Elements)) + 1
        NodesArgSorted = np.argsort(Nodes)
        ElementsNodes = np.searchsorted(Nodes[NodesArgSorted], ElementsNodes)

        # Generate mesh 
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.model.addDiscreteEntity(3, 1)

        # Nodes
        Time.Update(3/5,'Add Nodes')
        NodesCoords = [C for C in Coords.ravel()]
        gmsh.model.mesh.addNodes(3, 1, list(NodeTags), NodesCoords)

        # Element
        Time.Update(4/5,'Add Elements')
        ElementType = list(np.zeros(len(ElementsTags),int)+5)
        TagList = [[E] for E in ElementsTags]
        NodeList = [list(N+1) for N in ElementsNodes]
        gmsh.model.mesh.addElements(3, 1, ElementType, TagList, NodeList)

        # Physical group
        gmsh.model.addPhysicalGroup(3, [1], 1)

        # Write mesh
        Time.Update(5/5,'Write Mesh')
        FName = Path(__file__).parent / 'Mesh' / (ROI.name[:-4] + '.msh')
        gmsh.write(str(FName))
        gmsh.finalize()

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
    Parser.add_argument('--OutputPath', help='Output path for the ROI mesh', type=str, default=Path(__file__).parents[1] / '02_Results/Mesh')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)
#%%
#%% #!/usr/bin/env python3

"""
Script used to write .msh files according to gmsh standars
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import os
import numpy as np
from Time import Time
from numba import njit
import SimpleITK as sitk
from multiprocessing import Pool

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
def Write(BinArray, FileName, Spacing=np.ones(3)):
    
    Text = 'Write FE mesh'
    Time.Process(1,Text)

    if len(np.unique(BinArray)) == 2:
        BinArray = BinArray.astype(int)
    else:
        print('Image is not binary')
        return
    
    # Define input file name and create it
    if os.name == 'nt':
        NProc = 4
    else:
        NProc = 8
    
    # Perform mapping
    Time.Update(1/6,'Perform mapping')
    Nodes, Coords, Elements, ElementsNodes = Mapping(BinArray)
    NodesNeeded = np.unique(ElementsNodes[BinArray.astype(bool)])

    ElementsNodes = ElementsNodes[BinArray.astype(bool)]
    Elements = Elements[BinArray.astype(bool)]
    Coords = Coords[np.isin(Nodes,NodesNeeded)]
    Nodes = Nodes[np.isin(Nodes,NodesNeeded)]

    # Sort nodes according to coordinates
    Indices = np.lexsort((Coords[:,2],Coords[:,1],Coords[:,0]))
    Coords = np.round(Coords[Indices] * Spacing,3)[:,::-1]
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

    # Ns = np.array_split(NodesStr,NProc)
    # Cs = np.array_split(CoordsStr, NProc)
    # Processes = [[N, C, Sep] for N, C in zip(Ns, Cs)]
    # NText = []
    # with Pool(processes=NProc) as P:
    #     for TaskResult in P.map(NodesText, Processes):
    #         NText.append(TaskResult)
    NText = ''.join(NText)

    # Generate element text
    Time.Update(4/6,'Write elem. text')
    EN = (np.arange(len(Elements)) + 1).astype('<U32')
    NS = np.argsort(Nodes)
    SM = np.searchsorted(Nodes[NS], ElementsNodes)
    SEN = (NS[SM] + 1).astype('<U32')
    Sep = np.array([' 5 2 1 1 ', ' ', '\n']).astype('<U32')

    EText = ElementsText([EN, SEN, Sep])

    # ENs = np.array_split(EN,NProc)
    # SENs = np.array_split(SEN, NProc)
    # Processes = [[E, S, Sep] for E, S in zip(ENs, SENs)]
    # EText = []
    # with Pool(processes=NProc) as P:
    #     for TaskResult in P.map(ElementsText, Processes):
    #         EText.append(TaskResult)
    EText = ''.join(EText)

    # Write file
    Time.Update(5/6,'Write mesh file')
    with open(FileName,'w') as File:

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

    Time.Process(0,Text)

    return

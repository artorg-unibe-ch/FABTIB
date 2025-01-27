#%% !/usr/bin/env python3

"""
Script description
"""
__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import os
import numpy as np
import pandas as pd
import GmshMesh as GM
import SimpleITK as sitk
from pathlib import Path
import FabricAnalysis as FA
import Homogenization as FE

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#%% Main

FilePath = Path.cwd().parent / '01_Data/'
Files = sorted([F for F in FilePath.iterdir() if F.name.endswith('.mhd')])

try:
    Data = pd.read_csv(Path.cwd().parent / '03_Results/Morphometry.csv', index_col=0)
except:
    Cols = ['Tb.Th.','Tb.Sp.','Tb.N.','CV','DA','BVTV']
    Data = pd.DataFrame(columns=Cols, index=[F.name[:-4] for F in Files])


#%% Step 1: Image Processing

# import TestMIL as TM
# Directions = TM.FibonacciSphere(N=128)

# Read image
for File in Files:
    print('\n' + File.name + ' - Fabric-Elasticity Analysis')
    print('Read Image')
    Binary = sitk.ReadImage(File)
    BinArray = sitk.GetArrayFromImage(Binary) - 1

    # #%% Step 2: Fabric Analysis
    print('Compute Fabric')
    # MIL = TM.MIL(BinArray, Directions, Binary.GetSpacing()[0])
    # H = TM.FitFabric(MIL, Directions)
    # eValues, eVectors = np.linalg.eig(H)
    # eValues = 1 / np.sqrt(eValues)

    # # Normalize values and sort
    # eValues = 3 * eValues / sum(eValues)
    # eVectors = eVectors[np.argsort(eValues)]
    # eValues = np.sort(eValues)

    eValues, eVectors = FA.MILEigenValuesAndVectors(BinArray, NDir=128, NRays=50)
    # eValues, eVectors = TM.MSLEigenValuesAndVectors(BinArray,N=128)

    FabricPath = Path.cwd().parent / '03_Results/02_Fabric'
    os.makedirs(FabricPath, exist_ok=True)

    np.save(FabricPath / (File.name[:-4] + '_eValues.npy'), eValues)
    np.save(FabricPath / (File.name[:-4] + '_eVectors.npy'), eVectors)
    Data.loc[File.name[:-4], 'DA'] = max(eValues) / min(eValues)
    # Data.loc[File.name[:-4], 'BVTV'] = BinArray.sum() / BinArray.size
    Data.to_csv(Path.cwd().parent / '03_Results/Morphometry.csv')

    # #%% Step 3: Homogenization
    # HomPath = Path.cwd().parent / '03_Results/03_Homogenization'
    # os.makedirs(HomPath, exist_ok=True)
    # MeshFile = HomPath / (File.name[:-4] + '.msh')
    # GM.Write(BinArray, MeshFile)

    # # Perform homogenization with FEniCS
    # E, Nu = 1e4, 0.3
    # Stiffness = FE.Homogenize(str(MeshFile), (E,Nu), 'KUBC')
    # np.save(HomPath / (File.name[:-4] + '.npy'), Stiffness)

#%%

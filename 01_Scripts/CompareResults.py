#%% !/usr/bin/env python3

Description = """
Script used to compare results from custom analyses
to the medtool software
"""
__author__ = ['Mathieu Simon']
__date_created__ = '07-11-2024'
__date__ = '07-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#%% Functions

def Main():

    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/Fabric'
        ValidationPath = Path(__file__).parents[1] / '02_Results/Medtool'
        InputFabric = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])
    
    if Arguments.OutputPath:
        Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)
    else:
        Path.mkdir(Path('Validation'), exist_ok=True)


    FabValues = np.empty((len(InputFabric),3))
    MedValues = np.empty((len(InputFabric),3))
    for i, FabricFile in enumerate(InputFabric):

        Fab = np.load(FabricFile)
        Med = np.load(ValidationPath / FabricFile.name)

        eVal, eVec = np.linalg.eig(Fab)
        FabValues[i] = sorted(eVal)

        eVal, eVec = np.linalg.eig(Med)
        MedValues[i] = sorted(eVal)

    Min = np.min([MedValues, FabValues])
    Max = np.max([MedValues, FabValues])

    Figure, Axis = plt.subplots(1,1)
    Axis.plot(MedValues[:,0], FabValues[:,0], label='$m_1$',
              linestyle='none', color=(1,0,0), marker='o')
    Axis.plot(MedValues[:,1], FabValues[:,1], label='$m_2$',
              linestyle='none', color=(0,1,0), marker='o')
    Axis.plot(MedValues[:,2], FabValues[:,2], label='$m_3$',
              linestyle='none', color=(0,0,1), marker='o')
    Axis.plot([Min, Max], [Min, Max], linestyle='--', color=(0,0,0))
    Axis.set_xlabel('Medtool values')
    Axis.set_ylabel('Custom values')
    plt.legend(loc='upper left')
    plt.show(Figure)

    DAMin = np.min([MedValues[:,2] / MedValues[:,0],
                    FabValues[:,2] / FabValues[:,0]])
    DAMax = np.max([MedValues[:,2] / MedValues[:,0],
                    FabValues[:,2] / FabValues[:,0]])
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(MedValues[:,2] / MedValues[:,0], 
              FabValues[:,2] / FabValues[:,0],
              linestyle='none', color=(1,0,0), marker='o')
    Axis.plot([DAMin, DAMax], [DAMin, DAMax], linestyle='--', color=(0,0,0))
    Axis.set_xlabel('Medtool values')
    Axis.set_ylabel('Custom values')
    plt.show(Figure)

    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputROI', help='File name of the ROI fabric', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the results plots', type=str, default='02_Results/Validation')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)
#%%

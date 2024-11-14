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

def WriteBash(Sample, BVTV):

    Job = Sample[:10]
    Main = Sample[:10] + '_Main.inp'
    ODBFile = Sample[:10] + '.odb'
    OutFile = Sample[:10] + '.out'
    BVTV = round(BVTV,3)

    Text = f"""abaqus interactive job={Job} inp="/home/ms20s284/FABTIB2/02_Results/Abaqus/{Main}" cpus=24

abaqus python "/home/ms20s284/FABTIB2/01_Scripts/ReadAbaqus.py" in="/home/ms20s284/FABTIB2/02_Results/Abaqus/{ODBFile}"  out="/home/ms20s284/FABTIB2/02_Results/Abaqus/{OutFile}"  BVTV={BVTV}

rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.com
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.sta
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.odb
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.pes
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.pmg
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.prt
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.par
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.msg
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.dat
rm /home/ms20s284/FABTIB2/02_Results/Abaqus/{Sample[:10]}.sim

"""

    with open('RunAbaqus.bash','a') as File:
        File.write(Text)

    return

#%% Main

def Main(Arguments):

    MorphoPath = Path(__file__).parents[1] / '02_Results/Morphometry'
    AbaqusPath = Path(__file__).parents[1] / '02_Results/Abaqus'

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        Samples = [F.name[:-4] for F in Path.iterdir(MorphoPath) if F.name.endswith('.csv')]

    Stiffness = np.zeros((len(Samples),6,6))
    for s, Sample in enumerate([Samples[0]]):

        Morpho = pd.read_csv(MorphoPath / (Sample + '.csv'), delimiter=';')
        Abaqus = open(AbaqusPath / (Sample + '.out'), 'r').readlines()

        Stress = np.zeros((6,6))
        Strain = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stress[i,j] = float(Abaqus[i+4].split()[j+1])
                Strain[i,j] = float(Abaqus[i+14].split()[j+1])

        Stress = np.load('Stress.npy')
        Strain = np.load('Strain.npy')
        Stiffness[s] = Stress / Strain
                



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

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
import pandas as pd
from pathlib import Path

#%% Functions

def WriteBash(Sample):

    Shift = 4
    Job = Sample[:Shift]
    Main = Sample[:Shift] + '_Main.inp'
    ODBFile = Sample[:Shift] + '.odb'
    OutFile = Sample[:Shift] + '.out'

    Text = f"""abaqus interactive job={Job} inp="/home/ms20s284/FABTIB2/02_Results/Abaqus/{Main}" cpus=24
abaqus python "/home/ms20s284/FABTIB2/01_Scripts/abqSeReader.py" in="/home/ms20s284/FABTIB2/02_Results/Abaqus/{ODBFile}"  out="/home/ms20s284/FABTIB2/02_Results/Abaqus/{OutFile}"  size="0.6;0.6;0.6"
rm *.com *.sta *.pes *.pmg *.prt *.par *.msg *.dat *.env *.fil *.odb
"""
    
# abaqus python "/home/ms20s284/FABTIB2/01_Scripts/ReadAbaqus.py" in="/home/ms20s284/FABTIB2/02_Results/Abaqus/{ODBFile}"  out="/home/ms20s284/FABTIB2/02_Results/Abaqus/{OutFile}"  BVTV={BVTV}


    with open('RunAbaqus.bash','a') as File:
        File.write(Text)

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/Morphometry'
        CSVs = [F for F in Path.iterdir(DataPath) if F.name.endswith('.csv')]

    for CSV in CSVs:
        Data = pd.read_csv(CSV, delimiter=';')
        WriteBash(CSV.name[:-4])



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

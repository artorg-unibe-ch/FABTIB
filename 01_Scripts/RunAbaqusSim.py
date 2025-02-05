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

    Shift = 10
    Main = Sample
    Job = Sample[:Shift]
    ODBFile = Sample[:Shift] + '.odb'
    OutFile = Sample[:Shift] + '.out'

    Text = f"""abaqus interactive job={Job} inp="/home/ms20s284/FABTIB2/02_Results/Abaqus/{Main}" cpus=24
abaqus python "/home/ms20s284/FABTIB2/01_Scripts/abqSeReader.py" in="/home/ms20s284/FABTIB2/02_Results/Abaqus/{ODBFile}"  out="/home/ms20s284/FABTIB2/02_Results/Abaqus/{OutFile}"  size="5.28;5.28;5.28" spec="Stress" 
rm *.com *.sta *.pes *.pmg *.prt *.par *.msg *.dat *.env *.fil *.lck *.odb 
"""

    with open('RunAbaqus.bash','a') as File:
        File.write(Text)

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.AbaqusInp:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/Abaqus/'
        AbaqusInps = [F for F in Path.iterdir(DataPath) if F.name.endswith('Main.inp')]

    for Input in AbaqusInps:
        Out = Input.parent / (Input.name[:-9] + '.out')
        if not Out.exists():
            WriteBash(Input.name)



if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--AbaqusInp', help='Abaqus input file', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

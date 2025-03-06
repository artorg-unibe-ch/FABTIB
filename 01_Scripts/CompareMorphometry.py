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
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


#%% Main

def Main():

    # Read files
    Data = pd.read_csv(Path(__file__).parents[1] / '02_Results/Morphometry.csv', index_col=[0,1])
    MetaData = pd.read_csv(Path(__file__).parents[1] / '00_Data/SampleList.csv')

    # Plot BV/TV and CV
    for C in ['BV/TV', 'Tb.Th.', 'Tb.N.', 'Tb.Sp.', 'DA']:
        Min = min([MetaData[C].min(), Data[C].min()])
        Max = max([MetaData[C].max(), Data[C].max()])
        Figure, Axis = plt.subplots(1,1,dpi=200)
        Axis.plot(np.repeat(MetaData[C],3), Data[C],
                linestyle='none', color=(1,0,0), marker='o', label='Ctrl')
        Axis.plot([Min, Max], [Min, Max], linestyle='--', color=(0,0,0))
        Axis.set_xlabel('Full Samples')
        Axis.set_ylabel('ROIs')
        plt.show(Figure)

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main()

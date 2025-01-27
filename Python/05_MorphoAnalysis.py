#%% !/usr/bin/env python3

Description = """
Read .npy ROI and compute fabric using mean intercept length
"""

__author__ = ['Mathieu Simon']
__date_created__ = '28-10-2024'
__date__ = '06-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#%% Function

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputFile:
        InputFile = Arguments.InputFile
    else:
        InputFile = '02_Results/Morphometry.csv'

    Data = pd.read_csv(Path.cwd().parent / InputFile, index_col=[0,1])

    rect = [0.3,0.3,0.65,0.65]

    Figure, Axis = plt.subplots(1,1)
    Axis.plot(Data['BV/TV'], Data['CV'], linestyle='none', color=(1,0,0), marker='o')
    Axis.plot([min(Data['BV/TV']), max(Data['BV/TV'])], [0.263,0.263], linestyle='--', color=(0,0,0))
    Axis1 = add_subplot_axes(Axis,rect)
    Axis1.plot(Data['BV/TV'], Data['CV'], linestyle='none', color=(1,0,0), marker='o')
    Axis1.plot([min(Data['BV/TV']), max(Data['BV/TV'])], [0.263,0.263], linestyle='--', color=(0,0,0))
    Axis.set_xlim([0, 0.6])
    Axis.set_ylim([0, 1.6])
    Axis.set_xlabel('BV/TV')
    Axis.set_ylabel('CV')
    plt.show(Figure)

    
        
    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputFile', help='Morphometry results file', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the plot', type=str, default='02_Results/')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

#%%

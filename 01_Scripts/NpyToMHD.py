#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '28-10-2024'
__date__ = '29-10-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from Utils import Time

#%% Functions


#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])
        
    for ROI in [InputROIs[0]]:

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = np.load(ROI)
        Array = (Array > 3000)*1
        Image = sitk.GetImageFromArray(Array)
        Image.SetSpacing((0.0148, 0.0148, 0.0148))
        Image = sitk.Cast(Image, sitk.sitkUInt8)
        sitk.WriteImage(Image, str(ROI.parent / (ROI.name[:-4] + '.mhd')))

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputNPY', help='File name of the npy scan', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

#%%

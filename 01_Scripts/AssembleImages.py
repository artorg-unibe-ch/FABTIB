#%% !/usr/bin/env python3

Description = """
Combine scan and ROI plots into a single image
"""

__author__ = ['Mathieu Simon']
__date_created__ = '04-03-2025'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path


#%% Main

def Main():

    # Define paths
    ResultsPath = Path(__file__).parents[1] / '02_Results'
    ROIPath = ResultsPath / 'ROIs'
    ScanPath = ResultsPath / 'Scans'
    
    # Read ROI locations
    Data = pd.read_csv(ResultsPath / 'Parameters.csv', sep=';')
    del Data['Unnamed: 7']

    Samples = Data['$Sample'].unique()

    for Idx, Sample in enumerate(Samples):

        Full = np.zeros((768, 1024+342, 3),int)

        # Read full scan plot
        Scan = io.imread(ScanPath / (Sample + '.png'))
        Full[:,:1024] = Scan
    
        for i in range(3):
            ROI = io.imread(ROIPath / (Sample + f'_{i+1}.png'))
            Full[i*256:(i+1)*256,1024:] = ROI[::3,::3]

        XLine = np.linspace(550,1050,501).astype(int)
        YLine = np.linspace(300,175,501).astype(int)
        Full[YLine,XLine] = [255, 0, 0]

        YLine = np.linspace(400,400,501).astype(int)
        Full[YLine,XLine] = [255, 0, 0]

        YLine = np.linspace(500,625,501).astype(int)
        Full[YLine,XLine] = [255, 0, 0]

        io.imsave(ScanPath / (Sample + '_Full.png'), Full.astype('uint8'))


    # Assemble images per group
    Data = pd.read_csv(Path(__file__).parents[1] / '00_Data/SampleList.csv')
    Ctrl = Data['Group'] == 'Ctrl'
    T2D = Data['Group'] == 'T2D'

    i, j = 0, 0
    FullCtrl = np.zeros((3072,9562,3),'uint8')
    Sample = ''
    for _, Row in Data[Ctrl].iterrows():
        if Row['Sample'] != Sample:
            Sample = Row['Sample']
            Image = io.imread(ScanPath / (Sample + '_Full.png'))
            RStart, RStop = i*768, (i+1)*768
            CStart, CStop = j*1366, (j+1)*1366
            FullCtrl[RStart:RStop,CStart:CStop] = Image
            if j < 7:
                j += 1
            if j == 7:
                j = 0
                i += 1

    io.imsave(ScanPath / ('Controls.png'), FullCtrl.astype('uint8'))


    i, j = 0, 0
    FullCtrl = np.zeros((3072,9562,3),'uint8')
    Sample = ''
    for _, Row in Data[T2D].iterrows():
        if Row['Sample'] != Sample:
            Sample = Row['Sample']
            Image = io.imread(ScanPath / (Sample + '_Full.png'))
            RStart, RStop = i*768, (i+1)*768
            CStart, CStop = j*1366, (j+1)*1366
            FullCtrl[RStart:RStop,CStart:CStop] = Image
            if j < 7:
                j += 1
            if j == 7:
                j = 0
                i += 1

    io.imsave(ScanPath / ('Diabetics.png'), FullCtrl.astype('uint8'))


    return



if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main()

#%%

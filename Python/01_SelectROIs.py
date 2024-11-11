#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '28-10-2024'
__date__ = '28-10-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pyvista as pv
from pathlib import Path
from Utils import ReadISQ, Time
from skimage.measure import label
from skimage.filters import threshold_otsu

#%% Functions

def CropROIs(VoxelModel, AddData, L, N):

    # ROI voxel side length
    Nx = L // AddData['ElementSpacing'][0]
    Ny = L // AddData['ElementSpacing'][1]
    Nz = L // AddData['ElementSpacing'][2]

    # Compute ROI Z positions
    Height = (AddData['DimSize'][2] - Nz) # Sample height accounting for ROI size
    Step = Height / (N-1)                 # Z distance between ROIs
    Zc = np.arange(N) * Step + Nz /2

    # Extract corresponding Z-stacks
    CropedModels = []
    CropedSegments = []
    for i in range(N):
        Start = int(Zc[i] - Nz/2)
        Stop = int(Zc[i] + Nz/2)
        CropedModel = VoxelModel[Start:Stop,:,:]
        CropedModels.append(CropedModel)

    # Compute center of mass of extracted stacks
    X = np.arange(AddData['DimSize'][0])
    Y = np.arange(AddData['DimSize'][1])
    C = []
    for CropedModel in CropedModels:
        Xc = np.sum(CropedModel, axis=(0,1)) * X
        Xc = sum(Xc) / np.sum(CropedModel)
        Yc = np.sum(CropedModel, axis=(0,2)) * Y
        Yc = sum(Yc) / np.sum(CropedModel)
        C.append([Xc,Yc])

    # Crop ROI
    ROIs = []
    for i, (Xc, Yc) in enumerate(C):
        XStart = int(round(Xc - Nx/2))
        XStop = XStart + int(Nx)
        YStart = int(round(Yc - Ny/2))
        YStop = YStart + int(Ny)
        ROI = CropedModels[i][:,YStart:YStop,XStart:XStop]
        ROIs.append(ROI)

    return ROIs

def CleanROI(ROI:np.array):

    # Measure connected components
    Labels, N = label(ROI, return_num=True)

    # Find biggest volume element (skip 0 corresponding to background)
    S = 0
    Label = 0
    for L in range(1,N+1):
        Sum = np.sum(Labels == L)
        if Sum > S:
            S = Sum
            Label = L

    # Keep only biggest volume element
    Cleaned = Labels == Label

    return Cleaned * 1

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputISQ:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '00_Data'
        InputISQs = [F for F in Path.iterdir(DataPath) if F.name.endswith('.ISQ')]

    # Create output directory if necessary
    Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)

    for iISQ, ISQ in enumerate(InputISQs):

        # Read scan
        Time.Process(1, 'Read ' + ISQ.name[:-4])
        VoxelModel, AdditionalData = ReadISQ(ISQ, ASCII=False)
        VoxelModel = VoxelModel.astype(float)

        # Compute Otsu threshold to segment ROIS
        Time.Update(1/5,'Compute Otsu')
        Otsu = threshold_otsu(VoxelModel)

        # Select ROI at center
        Time.Update(2/5,f'Select {Arguments.NROIs} ROIs')
        ROISize = 5.3   # ROI side length in mm
        ROIs = CropROIs(VoxelModel, AdditionalData, ROISize, N=3)

        for iROI, ROI in enumerate(ROIs):

            # Segment ROI using Otsu threshold
            Seg = ROI > Otsu

            # Clean ROI (remove unconnected parts)
            Progress = 3/5 + (iROI/len(ROIs)) * 1/5
            Time.Update(Progress, f'Clean ROI {iROI}')
            Clean = CleanROI(Seg)

            # File name for output
            FName = Path(Arguments.OutputPath) / (Path(ISQ).name[:-4] + '_' + str(iROI))

            # Plot using pyvista
            Progress = 3/5 + (iROI/len(ROIs)) * 2/5
            Time.Update(Progress, f'Plot ROI {iROI}')
            pl = pv.Plotter(off_screen=True)
            actors = pl.add_volume(Clean[::2,::2,::2].T,
                        cmap='bone',
                        show_scalar_bar=False,
                        )
            actors.prop.interpolation_type = 'linear'
            pl.camera_position = 'xz'
            pl.camera.roll = 0
            pl.camera.elevation = 30
            pl.camera.azimuth = 30
            pl.camera.zoom(1.0)
            pl.screenshot(FName.parent / (FName.name + '.png'))
            # pl.show()

            # Save ROI for later analysis
            np.save(FName.parent / (FName.name + '.npy'), Clean)
    
        # Update time
        Time.Process(0, f'Done ISQ {iISQ+1} / {len(InputISQs)}')


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputISQ', help='File name of the ISQ scan', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI and png image of the plot', type=str, default='02_Results/ROIs')
    Parser.add_argument('--NROIs', help='Number of region of interests to extract', type=int, default=3)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

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
import pandas as pd
from numba import njit
from pathlib import Path
import SimpleITK as sitk
from Utils import ReadISQ, Time
from scipy.ndimage import distance_transform_edt

#%% Functions

@njit
def ComputeHildebrandThickness(
    Thicknesses: np.ndarray,
    Distances: np.ndarray,
    Indices: np.ndarray,
    Spacing: np.array) -> np.ndarray:
    """
    Use Hildebrand's sphere-fitting method to compute the local thickness field for a
    binary image, given an array to fill in and the sorted distance map of the binary image.
    Since the distances are sorted by distance values in ascending order, we can iterate through and assign each voxel's
    distance value to all voxels within that distance. Voxels corresponding to larger spheres will be processed
    later and overwrite values assigned by smaller spheres, and so each voxel will eventually be assigned the
    diameter of the largest sphere that it lies within.
    Finally, we do not check every voxel in the image for whether it lies within a sphere. We only check voxels that
    lie within the cube with side length equal to the sphere diameter around the center voxel. This saves a lot of
    computational effort.
    Parameters
    ----------
    Thicknesses : np.ndarray
        A numpy array that is initialized as zeros.
    Distances : np.ndarray
        A numpy array that is the sorted distance ridge of a mask, but the distances only. Each element is a float
        that corresponds to a distance value on the distance ridge, in ascending order.
    Indices : np.ndarray
        A numpy array that is the integer indices of the location of the distance. Each row in this array corresponds to
        the distance at the same position in the `dist_ridge` parameter, and then the three elements in each row are
        the i, j, k indices of the location of that voxel of the distance ridge in the binary image.
    Spacing : np.ndarray
        A numpy array with shape (3,) that gives the width of voxels in each dimension.
    Returns
    -------
    np.ndarray
        The local thickness field.
    """

    for rd, (ri, rj, rk) in zip(Distances, Indices):
        rd_vox = rd / Spacing
        di_min = np.maximum(np.floor(ri - rd_vox[0]) - 1, 0)
        di_max = np.minimum(np.ceil(ri + rd_vox[0]) + 2, Thicknesses.shape[0])
        dj_min = np.maximum(np.floor(rj - rd_vox[1]) - 1, 0)
        dj_max = np.minimum(np.ceil(rj + rd_vox[1]) + 2, Thicknesses.shape[1])
        dk_min = np.maximum(np.floor(rk - rd_vox[2]) - 1, 0)
        dk_max = np.minimum(np.ceil(rk + rd_vox[2]) + 2, Thicknesses.shape[2])
        for di in range(di_min, di_max):
            for dj in range(dj_min, dj_max):
                for dk in range(dk_min, dk_max):
                    if ((Spacing[0] * (di - ri)) ** 2 
                        + (Spacing[1] * (dj - rj)) ** 2
                        + (Spacing[2] * (dk - rk)) ** 2) < rd ** 2:
                        Thicknesses[di, dj, dk] = 2 * rd
    return Thicknesses

def TrabecularThicknesses(BinaryArray:np.array, VoxelSpacing:np.array):

    # Step 1: skeletonize the image
    Image = sitk.GetImageFromArray(BinaryArray.astype(int))
    Skeleton = sitk.BinaryThinning(Image)
    Skeleton = sitk.GetArrayFromImage(Skeleton).astype(bool)

    # Step 2: Compute distances
    Distances = distance_transform_edt(BinaryArray, VoxelSpacing)

    # Step 3: Keep skeleton distances
    SkeletonDistances = (Skeleton > 0) * Distances

    # Step 4: Sort distances
    Indices = np.nonzero(SkeletonDistances)
    Sorted = np.argsort(SkeletonDistances[Indices])

    Distances = SkeletonDistances[Indices][Sorted]
    Indices = np.array(Indices).T[Sorted]

    # Step 5: Compute thickness
    Thicknesses = np.zeros(Skeleton.shape)
    Thicknesses = ComputeHildebrandThickness(Thicknesses, Distances, Indices, VoxelSpacing)

    return Thicknesses

def ComputeCV(BinaryArray:np.array):

    """
    Compute coefficient of variation of ROI
    """

    # Step 1: Trim the array to ensure all dimensions are even
    TrimmedArray = BinaryArray[
        :BinaryArray.shape[0] - (BinaryArray.shape[0] % 2),
        :BinaryArray.shape[1] - (BinaryArray.shape[1] % 2),
        :BinaryArray.shape[2] - (BinaryArray.shape[2] % 2)
        ]
        
    # Step 2: Reshape and split array into subcubes
    SubCubes = TrimmedArray.reshape(
        TrimmedArray.shape[0]//2, 2,
        TrimmedArray.shape[1]//2, 2,
        TrimmedArray.shape[2]//2, 2
        ).swapaxes(1, 2).reshape(8, -1)
    
    # Step 3: Compute bone volume within each subcube
    BV = SubCubes.sum(axis=1)

    # Step 4: Compute mean and standard deviation of masses
    Mean = np.mean(BV)
    Std = np.std(BV)

    # Step 5: Compute and return the CV
    CV = Std / Mean
    
    return CV


#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])

    # Read or create data frame to store results
    try:
        Data = pd.read_csv(Path.cwd().parent / '02_Results/Morphometry.csv', index_col=[0,1])
    except:
        Cols = ['ISQ','ROI','Tb.Th.','Tb.Sp.','Tb.N.','CV','BV/TV']
        Data = pd.DataFrame(columns=Cols)
        Data = Data.set_index(['ISQ','ROI'])
        
    pROI = None
    for ROI in InputROIs:

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = np.load(ROI)

        # If ROI come from same scane as previous ROI, use same threshold
        if pROI == None or pROI.name[:-6] != ROI.name[:-6]:
            # ISQ = Path(__file__).parents[1] / '00_Data' / (ROI.name[:-6] + '.ISQ')
            # AdditionalData = ReadISQ(ISQ, ASCII=False)[1]
            # Spacing = np.array(AdditionalData['ElementSpacing'])
            Spacing = np.array([0.0147998046875, 0.0147998046875, 0.014799019607843137])
        

        # Compute bone volume fraction
        BVTV = np.sum(Array) / Array.size

        # If bone is present, compute trabecular values
        if BVTV > 0.01:

            # Trabecular Thickness
            Time.Update(1/4,'Tb.Th.')
            Thicknesses = TrabecularThicknesses(Array, Spacing)

            # Trabecular spacing
            Time.Update(2/4,'Tb.Sp.')
            Spacings = TrabecularThicknesses(1-Array, Spacing)

            # Trabecular number
            Time.Update(3/4,'Tb.N.')
            Number = np.divide(1,Thicknesses+Spacings, where=Thicknesses+Spacings > 0)

            # Coefficient of variation
            Time.Update(4/4,'CV')
            CV = ComputeCV(Array)

            # Store data
            Data.loc[(ROI.name[:-6], ROI.name[-5]),'Tb.Th.'] = Thicknesses[Thicknesses > 0].mean()
            Data.loc[(ROI.name[:-6], ROI.name[-5]),'Tb.Sp.'] = Spacings[Spacings > 0].mean()
            Data.loc[(ROI.name[:-6], ROI.name[-5]),'Tb.N.'] = Number.mean()
            Data.loc[(ROI.name[:-6], ROI.name[-5]),'CV'] = CV
            Data.loc[(ROI.name[:-6], ROI.name[-5]),'BV/TV'] = np.sum(Array) / Array.size

        else:
            print('BV/TV < 1%')
            Data.loc[ROI,'Tb.Th.'] = np.nan
            Data.loc[ROI,'Tb.Sp.'] = np.nan
            Data.loc[ROI,'Tb.N.'] = np.nan
            Data.loc[ROI,'CV'] = np.nan

        # Print time and update
        Time.Process(0,ROI.name[:-4])
        pROI = ROI

    # Save data
    Data.to_csv(Path.cwd().parent / '02_Results/Morphometry.csv')
        

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputROI', help='File name of the ROI npy file', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

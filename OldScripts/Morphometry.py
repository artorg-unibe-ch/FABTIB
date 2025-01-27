#%% !/usr/bin/env python3

"""
Script description
From https://github.com/SpectraCollab/ORMIR_XCT/blob/main/ormir_xct/util/hildebrand_thickness.py#L30
"""
__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import numpy as np
import pandas as pd
from Time import Time
from numba import njit
import SimpleITK as sitk
from pathlib import Path
from scipy.stats.distributions import t

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#%% Functions

@njit
def ComputeHildebrandThickness(
    Thicknesses: np.ndarray,
    Distances: np.ndarray,
    Indices: np.ndarray) -> np.ndarray:
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
    local_thickness : np.ndarray
        A numpy array that is initialized as zeros.
    sorted_dists : np.ndarray
        A numpy array that is the sorted distance ridge of a mask, but the distances only. Each element is a float
        that corresponds to a distance value on the distance ridge, in ascending order.
    sorted_dists_indices : np.ndarray
        A numpy array that is the integer indices of the location of the distance. Each row in this array corresponds to
        the distance at the same position in the `dist_ridge` parameter, and then the three elements in each row are
        the i, j, k indices of the location of that voxel of the distance ridge in the binary image.
    voxel_width : np.ndarray
        A numpy array with shape (3,) that gives the width of voxels in each dimension.
    rd_extra : float
        An extra bit of distance added to `rd` for the purposes of determining whether a voxel falls within the sphere
        centered at another voxel. This is needed because if you use the oversampling distance transform, your sphere
        diameters will be smaller (since they are measuring the distance from that voxel centroid to the boundary of
        the shape rather than to the nearest centroid outside the shape). This causes voxels along the boundary of a
        shape to not be included in the spheres centered on voxels further in. Set this to `0.5` if you are using
        oversampling and `0` if you are not.
    Returns
    -------
    np.ndarray
        The local thickness field.
    """

    for rd, (ri, rj, rk) in zip(Distances, Indices):
        di_min = np.maximum(np.floor(ri - rd) - 1, 0)
        di_max = np.minimum(np.ceil(ri + rd) + 2, Thicknesses.shape[0])
        dj_min = np.maximum(np.floor(rj - rd) - 1, 0)
        dj_max = np.minimum(np.ceil(rj + rd) + 2, Thicknesses.shape[1])
        dk_min = np.maximum(np.floor(rk - rd) - 1, 0)
        dk_max = np.minimum(np.ceil(rk + rd) + 2, Thicknesses.shape[2])
        for di in range(di_min, di_max):
            for dj in range(dj_min, dj_max):
                for dk in range(dk_min, dk_max):
                    if ((di - ri) ** 2 + (dj - rj) ** 2 + (dk - rk) ** 2) < rd ** 2:
                        Thicknesses[di, dj, dk] = 2 * rd
    return Thicknesses

def TrabecularThickness(Image):

    # Step 1: Skeletonize the image
    Skeleton = sitk.BinaryThinning(Image)
    Skeleton = sitk.GetArrayFromImage(Skeleton)

    # Step 2: Compute distances
    Distances = sitk.DanielssonDistanceMap(1-Image)
    Distances = sitk.GetArrayFromImage(Distances)

    # Step 3: Keep skeleton distances
    SkeletonDistances = (Skeleton > 0) * Distances

    # Step 4: Sort distances
    Indices = np.nonzero(SkeletonDistances)
    Sorted = np.argsort(SkeletonDistances[Indices])

    Distances = SkeletonDistances[Indices][Sorted]
    Indices = np.array(Indices).T[Sorted]

    # Step 5: Compute thickness
    Thicknesses = np.zeros(Skeleton.shape)
    Thicknesses = ComputeHildebrandThickness(Thicknesses, Distances, Indices)
    
    # Step 6: Multiply by voxel spacing
    Thicknesses = Thicknesses * Image.GetSpacing()[0]

    return Thicknesses

def ReadDA(File):

    eValues = np.load(File)

    return max(eValues) / min(eValues)

def OLS(X:np.array, Y:np.array, Labels=None, Alpha=0.95, FName=None) -> None:
    
    """
    Plot linear regression between to variables X and Y


    Parameters
    ----------
    X: Independent variable
    Y: Dependent variable
    Labels: Labels for the different axes/variables (X and Y)
    Alpha: Conficence level
    FName: Figure name (to save it)

    Returns
    -------
    None
    """

    if Labels == None:
        Labels = ['X', 'Y']
    
    # Perform linear regression
    Xm = np.matrix([np.ones(len(X)), X]).T
    Ym = np.matrix(Y).T
    Intercept, Slope = np.linalg.inv(Xm.T * Xm) * Xm.T * Ym
    Intercept = np.array(Intercept)[0,0]
    Slope = np.array(Slope)[0,0]

    # Build arrays and matrices
    Y_Obs = Y
    Y_Fit = X * Slope + Intercept
    N = len(Y)
    X = np.matrix(X)

    # Sort X values and Y accordingly
    Sort = np.argsort(np.array(Xm[:,1]).reshape(len(Xm)))
    X_Obs = np.sort(np.array(Xm[:,1]).reshape(len(Xm)))
    Y_Fit = Y_Fit[Sort]
    Y_Obs = Y_Obs[Sort]

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / (N - 2))
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS
    R2adj = 1 - RSS/TSS * (N-1)/(N-Xm.shape[1]+1-1)

    ## Compute variance-covariance matrix
    C = np.linalg.inv(Xm.T * Xm)

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(Xm * C * Xm.T)))
    t_Alpha = t.interval(Alpha, N - Xm.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0[Sort]
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0[Sort]

    # Plots
    DPI = 96
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI)
    Axes.plot(X_Obs, Y_Obs, linestyle='none', marker='o', color=(0,0,1), fillstyle='none')
    Axes.plot(X_Obs, Y_Fit, color=(1,0,0))
    Axes.fill_between(X_Obs, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)

    # Add annotations
    if Slope > 0:

        # Number of observations
        YPos = 0.925
        Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos -= 0.075
        Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos -= 0.075
        Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
        
        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.025
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] *  np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin], Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
        YPos += 0.075

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin], Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')

    elif Slope < 0:

        # Number of observations
        YPos = 0.025
        Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos += 0.075
        Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos += 0.075
        Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.925
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin],Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
        YPos -= 0.075

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin],Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
    
    Axes.set_xlabel(Labels[0])
    Axes.set_ylabel(Labels[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if FName:
        plt.savefig(FName, dpi=196)
    plt.show(Figure)

    return

#%% Main

FilePath = Path.cwd().parent / '01_Data/'
Files = sorted([F for F in FilePath.iterdir() if F.name.endswith('.mhd')])

try:
    Data = pd.read_csv(Path.cwd().parent / '03_Results/Morphometry.csv', index_col=0)
except:
    Cols = ['Tb.Th.','Tb.Sp.','Tb.N.','CV','DA','BVTV']
    Data = pd.DataFrame(columns=Cols, index=[F.name[:-4] for F in Files])


for File in Files:
    print('\n' + File.name + ' - Morphometric Analysis')
    Time.Process(1, '')

    # Read image
    Binary = sitk.ReadImage(File) - 1

    # Trabecular thickness
    Time.Update(1/4,'Tb.Th.')
    Thicknesses = TrabecularThickness(Binary)
    MinThickness = 0.0
    Thicknesses_Structure = np.maximum(Thicknesses[Thicknesses > 0], MinThickness)
    Thicknesses_Structure.std()
    Thicknesses_Structure.min()
    Thicknesses_Structure.max()

    # Trabecular spacing
    Time.Update(2/4,'Tb.Sp.')
    Spacings = TrabecularThickness(1-Binary)
    Spacings.mean()
    Spacings.std()
    Spacings.min()
    Spacings.max()

    # Trabecular number
    Time.Update(3/4,'Tb.N.')
    Number = np.divide(1,Thicknesses+Spacings, where=Thicknesses+Spacings > 0)
    Number[Number > 0].mean()
    Number[Number > 0].std()
    Number[Number > 0].min()
    Number[Number > 0].max()

    # Coefficient of variation
    Time.Update(4/4,'CV')
    Size = Binary.GetSize()
    Sub1 = sum(Binary[:Size[0]//2,:Size[1]//2,:Size[2]//2]) / (Size[0]*Size[1]*Size[2]/8)
    Sub2 = sum(Binary[Size[0]//2:,:Size[1]//2,:Size[2]//2]) / (Size[0]*Size[1]*Size[2]/8)
    Sub3 = sum(Binary[:Size[0]//2,Size[1]//2:,:Size[2]//2]) / (Size[0]*Size[1]*Size[2]/8)
    Sub4 = sum(Binary[:Size[0]//2,:Size[1]//2,Size[2]//2:]) / (Size[0]*Size[1]*Size[2]/8)
    Sub5 = sum(Binary[Size[0]//2:,Size[1]//2:,:Size[2]//2]) / (Size[0]*Size[1]*Size[2]/8)
    Sub6 = sum(Binary[Size[0]//2:,:Size[1]//2,Size[2]//2:]) / (Size[0]*Size[1]*Size[2]/8)
    Sub7 = sum(Binary[:Size[0]//2,Size[1]//2:,Size[2]//2:]) / (Size[0]*Size[1]*Size[2]/8)
    Sub8 = sum(Binary[Size[0]//2:,Size[1]//2:,Size[2]//2:]) / (Size[0]*Size[1]*Size[2]/8)
    Std = np.std([Sub1, Sub2, Sub3, Sub4, Sub5, Sub6, Sub7, Sub8])
    Mean = np.mean([Sub1, Sub2, Sub3, Sub4, Sub5, Sub6, Sub7, Sub8])
    CV = Std / Mean

    # Store data
    Data.loc[File.name[:-4],'Tb.Th.'] = Thicknesses_Structure.mean()
    Data.loc[File.name[:-4],'Tb.Sp.'] = Spacings.mean()
    Data.loc[File.name[:-4],'Tb.N.'] = Number.mean()
    Data.loc[File.name[:-4],'CV'] = CV
    Time.Process(0, '')

Data.to_csv(Path.cwd().parent / '03_Results/Morphometry.csv')

#%% Compare to Medtool

import matplotlib.pyplot as plt

MedData = Data.copy()
MedPath = Path.cwd().parent / 'OldResults/'
FabPath = Path.cwd().parent / '03_Results/02_Fabric'

for File in Files:
    ROIData = pd.read_csv(MedPath / (File.name[:-14] + '.csv'), sep=';')
    MedData.loc[File.name[:-4],'Tb.Th.'] = ROIData['$Tb_Th_mean'].values[0]
    MedData.loc[File.name[:-4],'Tb.Sp.'] = ROIData['$Tb_Sp_mean'].values[0]
    MedData.loc[File.name[:-4],'Tb.N.'] = ROIData['$Tb_N_mean'].values[0]
    MedData.loc[File.name[:-4],'DA'] = ReadDA(FabPath / (File.name[:-4] + '_eValues_MT.npy'))

Variable = 'Tb.Sp.'
OLS(MedData[Variable].values, Data[Variable].values, Labels=['Medtool','Python'], FName=Variable + '.png')

Figure, Axis = plt.subplots(1,1)
Min = min([MedData[Variable].min(), Data[Variable].min()])
Max = max([MedData[Variable].max(), Data[Variable].max()])
Axis.plot([Min, Max], [Min, Max], linestyle='--', color=(0,0,0))
Axis.scatter(MedData[Variable], Data[Variable], color=(1,0,0), facecolor=(0,0,0,0))
Axis.set_xlabel('Medtool')
Axis.set_ylabel('Python')
Axis.set_xlim([1,7])
Axis.set_ylim([1,7])
plt.show(Figure)


#%%

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
import SimpleITK as sitk
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt

#%% Functions

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

def ComputeHistogram(Variable:np.array):

    # Get data attributes
    SortedValues = np.sort(Variable).astype(float)
    N = len(Variable)
    X_Bar = np.mean(Variable)
    S_X = np.std(Variable, ddof=1)
    Q025 = np.quantile(Variable, 0.25)
    Q075 = np.quantile(Variable, 0.75)

    Histogram, Edges = np.histogram(Variable, bins=20)
    Width = (Edges[1] - Edges[0])
    Center = (Edges[:-1] + Edges[1:]) / 2

    # Kernel density estimation (Gaussian kernel)
    KernelEstimator = np.zeros(N)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25,0.75]), 0, 1)))
    DataIQR = np.abs(Q075) - np.abs(Q025)
    KernelHalfWidth = 0.9*N**(-1/5) * min(S_X,DataIQR/NormalIQR)
    for Value in SortedValues:
        Norm = norm.pdf(SortedValues-Value,loc=0,scale=KernelHalfWidth*2)
        KernelEstimator += Norm / max(Norm)
    KernelEstimator = KernelEstimator/N*max(Histogram)

    # Histogram and density distribution
    TheoreticalDistribution = norm.pdf(SortedValues,X_Bar,S_X)

    return SortedValues, Center, Histogram, Width, KernelEstimator, TheoreticalDistribution

def PlotHistogram(Variable, Groups, Name, Path):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=200)

    Ctrl = Variable[Groups[0]]
    SortedValues, Center, Histogram, Width, KernelEstimator, TheoreticalDistribution = ComputeHistogram(Ctrl)
    Axes.fill_between(SortedValues,np.zeros(len(SortedValues)), KernelEstimator,color=(1,0,0,0.5))
    Axes.plot(SortedValues, KernelEstimator,color=(1,0,0),label='Ctrl')
    Axes.bar(Center, Histogram, align='center', width=Width,edgecolor=(1,0,0),color=(1,1,1,0))

    T2D = Variable[Groups[1]]
    SortedValues, Center, Histogram, Width, KernelEstimator, TheoreticalDistribution = ComputeHistogram(T2D)
    Axes.fill_between(SortedValues,np.zeros(len(SortedValues)), KernelEstimator,color=(0,0,1,0.5))
    Axes.plot(SortedValues, KernelEstimator,color=(0,0,1),label='T2D')
    Axes.bar(Center, Histogram, align='center', width=Width,edgecolor=(0,0,1),color=(1,1,1,0))

    plt.xlabel(Name)
    plt.ylabel('Frequency (-)')
    plt.yticks([])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.savefig(Path / (Name.replace('/','') + '.png'))
    # plt.show()
    plt.close(Figure)

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if not Arguments.DataPath:
        DataPath = Path(__file__).parents[1] / '02_Results/Morphometry/'
    DataFiles = [F for F in Path.iterdir(DataPath) if F.name.endswith('.csv')]

    Cols = ['BV/TV', 'Tb.N.', 'Tb.Th.', 'Tb.Sp.', 'Tb.Sp.SD', 'DA', 'CV']
    Names = set([F.name[:-6] for F in DataFiles])
    Idx = pd.MultiIndex.from_product([Names, range(1,4)])
    Data = pd.DataFrame(index=Idx, columns=Cols)
    Data = Data.sort_index()
    MedCols = ['$BVTV_voxel', '$Tb_N_mean', '$Tb_Th_mean', '$Tb_Sp_mean', '$Tb_Th_stddev', '$DA_value']
    for File in DataFiles:

        # Collect morphometry data
        SampleData = pd.read_csv(File, sep=';')
        for Col1, Col2 in zip(Cols[:-1], MedCols):
            Data.loc[(File.name[:-6],int(File.name[-5])),Col1] = SampleData[Col2].values[0]

        # Compute CV
        ROI = sitk.ReadImage(str(DataPath.parent / 'ROIs' / (File.name[:-4] + '.mhd')))
        BinArray = sitk.GetArrayFromImage(ROI-1).astype(bool)
        CV = ComputeCV(BinArray)
        Data.loc[(File.name[:-6],int(File.name[-5])), 'CV'] = CV
    Data.to_csv(Path(__file__).parents[1] / '02_Results/Morphometry.csv')

    # Read metadata file
    MetaData = pd.read_excel(Path(__file__).parents[1] / '00_Data/SampleList.xlsx')
    Ctrl = MetaData['Group (T2D or Ctrl)'] == 'Ctrl'
    T2D = MetaData['Group (T2D or Ctrl)'] == 'T2D'
    FH = MetaData['Anatomical Location'] == 'Femoral Head'
    DF = MetaData['Anatomical Location'] == 'Distal Femur'
    CtrlDF = MetaData['Filename'][Ctrl&DF].values
    CtrlSamples = MetaData['Filename'][Ctrl&FH].values
    T2DSamples = MetaData['Filename'][T2D&FH].values
    Samples = [i[0] for i in Data.index]
    CtrlDF = [S in CtrlDF for S in Samples]
    Ctrl = [S in CtrlSamples for S in Samples]
    T2D = [S in T2DSamples for S in Samples]


    # Plot BV/TV and CV
    Figure, Axis = plt.subplots(1,1,dpi=200)
    Axis.plot(Data['BV/TV'][Ctrl], Data['CV'][Ctrl], linestyle='none', color=(0,0,1), marker='o', label='Ctrl femoral head')
    Axis.plot(Data['BV/TV'][T2D], Data['CV'][T2D], linestyle='none', color=(1,0,0), marker='o', label='T2D femoral head')
    Axis.plot(Data['BV/TV'][CtrlDF], Data['CV'][CtrlDF], linestyle='none', color=(0,1,0), marker='o', label='Ctrl distal femur')
    Axis.plot([min(Data['BV/TV']), max(Data['BV/TV'])], [0.263,0.263], linestyle='--', color=(0,0,0))
    Axis.set_xlabel('BV/TV')
    Axis.set_ylabel('CV')
    plt.legend(loc='upper right')
    plt.savefig(Path(__file__).parents[1] / '02_Results/CV_BVTV.png')
    plt.show(Figure)

    # Plot Histograms
    for Col in Data.columns:
        PlotHistogram(Data[Col], [Ctrl,T2D], Col, DataPath)


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

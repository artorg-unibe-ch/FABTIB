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
from scipy.stats.distributions import norm
from scipy.stats import shapiro, levene, bartlett
from scipy.stats import ttest_ind, mannwhitneyu, permutation_test


#%% Functions

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def Engineering2MandelNotation(A):

    B = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            if i < 3 and j >= 3:
                B[i,j] = A[i,j] * np.sqrt(2)
            elif i >= 3 and j < 3:
                B[i,j] = A[i,j] * np.sqrt(2)
            elif i >= 3 and j >= 3:
                B[i,j] = A[i,j] * 2
            else:
                B[i, j] = A[i, j]

    return B

def IsoMorphism66_3333(A):

    # Check symmetry
    Symmetry = True
    for i in range(6):
        for j in range(6):
            if not A[i,j] == A[j,i]:
                Symmetry = False
                break
    if Symmetry == False:
        print('Matrix is not symmetric!')
        return

    B = np.zeros((3,3,3,3))

    # Build 4th tensor
    B[0, 0, 0, 0] = A[0, 0]
    B[1, 1, 0, 0] = A[1, 0]
    B[2, 2, 0, 0] = A[2, 0]
    B[1, 2, 0, 0] = A[3, 0] / np.sqrt(2)
    B[2, 0, 0, 0] = A[4, 0] / np.sqrt(2)
    B[0, 1, 0, 0] = A[5, 0] / np.sqrt(2)

    B[0, 0, 1, 1] = A[0, 1]
    B[1, 1, 1, 1] = A[1, 1]
    B[2, 2, 1, 1] = A[2, 1]
    B[1, 2, 1, 1] = A[3, 1] / np.sqrt(2)
    B[2, 0, 2, 1] = A[4, 1] / np.sqrt(2)
    B[0, 1, 2, 1] = A[5, 1] / np.sqrt(2)

    B[0, 0, 2, 2] = A[0, 2]
    B[1, 1, 2, 2] = A[1, 2]
    B[2, 2, 2, 2] = A[2, 2]
    B[1, 2, 2, 2] = A[3, 2] / np.sqrt(2)
    B[2, 0, 2, 2] = A[4, 2] / np.sqrt(2)
    B[0, 1, 2, 2] = A[5, 2] / np.sqrt(2)

    B[0, 0, 1, 2] = A[0, 3] / np.sqrt(2)
    B[1, 1, 1, 2] = A[1, 3] / np.sqrt(2)
    B[2, 2, 1, 2] = A[2, 3] / np.sqrt(2)
    B[1, 2, 1, 2] = A[3, 3] / 2
    B[2, 0, 1, 2] = A[4, 3] / 2
    B[0, 1, 1, 2] = A[5, 3] / 2

    B[0, 0, 2, 0] = A[0, 4] / np.sqrt(2)
    B[1, 1, 2, 0] = A[1, 4] / np.sqrt(2)
    B[2, 2, 2, 0] = A[2, 4] / np.sqrt(2)
    B[1, 2, 2, 0] = A[3, 4] / 2
    B[2, 0, 2, 0] = A[4, 4] / 2
    B[0, 1, 2, 0] = A[5, 4] / 2

    B[0, 0, 0, 1] = A[0, 5] / np.sqrt(2)
    B[1, 1, 0, 1] = A[1, 5] / np.sqrt(2)
    B[2, 2, 0, 1] = A[2, 5] / np.sqrt(2)
    B[1, 2, 0, 1] = A[3, 5] / 2
    B[2, 0, 0, 1] = A[4, 5] / 2
    B[0, 1, 0, 1] = A[5, 5] / 2



    # Add minor symmetries ijkl = ijlk and ijkl = jikl

    B[0, 0, 0, 0] = B[0, 0, 0, 0]
    B[0, 0, 0, 0] = B[0, 0, 0, 0]

    B[0, 0, 1, 0] = B[0, 0, 0, 1]
    B[0, 0, 0, 1] = B[0, 0, 0, 1]

    B[0, 0, 1, 1] = B[0, 0, 1, 1]
    B[0, 0, 1, 1] = B[0, 0, 1, 1]

    B[0, 0, 2, 1] = B[0, 0, 1, 2]
    B[0, 0, 1, 2] = B[0, 0, 1, 2]

    B[0, 0, 2, 2] = B[0, 0, 2, 2]
    B[0, 0, 2, 2] = B[0, 0, 2, 2]

    B[0, 0, 0, 2] = B[0, 0, 2, 0]
    B[0, 0, 2, 0] = B[0, 0, 2, 0]



    B[0, 1, 0, 0] = B[0, 1, 0, 0]
    B[1, 0, 0, 0] = B[0, 1, 0, 0]

    B[0, 1, 1, 0] = B[0, 1, 0, 1]
    B[1, 0, 0, 1] = B[0, 1, 0, 1]

    B[0, 1, 1, 1] = B[0, 1, 1, 1]
    B[1, 0, 1, 1] = B[0, 1, 1, 1]

    B[0, 1, 2, 1] = B[0, 1, 1, 2]
    B[1, 0, 1, 2] = B[0, 1, 1, 2]

    B[0, 1, 2, 2] = B[0, 1, 2, 2]
    B[1, 0, 2, 2] = B[0, 1, 2, 2]

    B[0, 1, 0, 2] = B[0, 1, 2, 0]
    B[1, 0, 2, 0] = B[0, 1, 2, 0]



    B[1, 1, 0, 0] = B[1, 1, 0, 0]
    B[1, 1, 0, 0] = B[1, 1, 0, 0]

    B[1, 1, 1, 0] = B[1, 1, 0, 1]
    B[1, 1, 0, 1] = B[1, 1, 0, 1]

    B[1, 1, 1, 1] = B[1, 1, 1, 1]
    B[1, 1, 1, 1] = B[1, 1, 1, 1]

    B[1, 1, 2, 1] = B[1, 1, 1, 2]
    B[1, 1, 1, 2] = B[1, 1, 1, 2]

    B[1, 1, 2, 2] = B[1, 1, 2, 2]
    B[1, 1, 2, 2] = B[1, 1, 2, 2]

    B[1, 1, 0, 2] = B[1, 1, 2, 0]
    B[1, 1, 2, 0] = B[1, 1, 2, 0]



    B[1, 2, 0, 0] = B[1, 2, 0, 0]
    B[2, 1, 0, 0] = B[1, 2, 0, 0]

    B[1, 2, 1, 0] = B[1, 2, 0, 1]
    B[2, 1, 0, 1] = B[1, 2, 0, 1]

    B[1, 2, 1, 1] = B[1, 2, 1, 1]
    B[2, 1, 1, 1] = B[1, 2, 1, 1]

    B[1, 2, 2, 1] = B[1, 2, 1, 2]
    B[2, 1, 1, 2] = B[1, 2, 1, 2]

    B[1, 2, 2, 2] = B[1, 2, 2, 2]
    B[2, 1, 2, 2] = B[1, 2, 2, 2]

    B[1, 2, 0, 2] = B[1, 2, 2, 0]
    B[2, 1, 2, 0] = B[1, 2, 2, 0]



    B[2, 2, 0, 0] = B[2, 2, 0, 0]
    B[2, 2, 0, 0] = B[2, 2, 0, 0]

    B[2, 2, 1, 0] = B[2, 2, 0, 1]
    B[2, 2, 0, 1] = B[2, 2, 0, 1]

    B[2, 2, 1, 1] = B[2, 2, 1, 1]
    B[2, 2, 1, 1] = B[2, 2, 1, 1]

    B[2, 2, 2, 1] = B[2, 2, 1, 2]
    B[2, 2, 1, 2] = B[2, 2, 1, 2]

    B[2, 2, 2, 2] = B[2, 2, 2, 2]
    B[2, 2, 2, 2] = B[2, 2, 2, 2]

    B[2, 2, 0, 2] = B[2, 2, 2, 0]
    B[2, 2, 2, 0] = B[2, 2, 2, 0]



    B[2, 0, 0, 0] = B[2, 0, 0, 0]
    B[0, 2, 0, 0] = B[2, 0, 0, 0]

    B[2, 0, 1, 0] = B[2, 0, 0, 1]
    B[0, 2, 0, 1] = B[2, 0, 0, 1]

    B[2, 0, 1, 1] = B[2, 0, 1, 1]
    B[0, 2, 1, 1] = B[2, 0, 1, 1]

    B[2, 0, 2, 1] = B[2, 0, 1, 2]
    B[0, 2, 1, 2] = B[2, 0, 1, 2]

    B[2, 0, 2, 2] = B[2, 0, 2, 2]
    B[0, 2, 2, 2] = B[2, 0, 2, 2]

    B[2, 0, 0, 2] = B[2, 0, 2, 0]
    B[0, 2, 2, 0] = B[2, 0, 2, 0]


    # Complete minor symmetries
    B[0, 2, 1, 0] = B[0, 2, 0, 1]
    B[0, 2, 0, 2] = B[0, 2, 2, 0]
    B[0, 2, 2, 1] = B[0, 2, 1, 2]

    B[1, 0, 1, 0] = B[1, 0, 0, 1]
    B[1, 0, 0, 2] = B[1, 0, 2, 0]
    B[1, 0, 2, 1] = B[1, 0, 1, 2]

    B[2, 1, 1, 0] = B[2, 1, 0, 1]
    B[2, 1, 0, 2] = B[2, 1, 2, 0]
    B[2, 1, 2, 1] = B[2, 1, 1, 2]


    # Add major symmetries ijkl = klij
    B[0, 1, 1, 1] = B[1, 1, 0, 1]
    B[1, 0, 1, 1] = B[1, 1, 1, 0]

    B[0, 2, 1, 1] = B[1, 1, 0, 2]
    B[2, 0, 1, 1] = B[1, 1, 2, 0]


    return B

def CheckMinorSymmetry(A):
    MinorSymmetry = True
    for i in range(3):
        for j in range(3):
            PartialTensor = A[:,:, i, j]
            if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                MinorSymmetry = True
            else:
                MinorSymmetry = False
                break

    if MinorSymmetry == True:
        for i in range(3):
            for j in range(3):
                PartialTensor = np.squeeze(A[i, j,:,:])
                if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                    MinorSymmetry = True
                else:
                    MinorSymmetry = False
                    break

    return MinorSymmetry

def IsoMorphism3333_66(A):

    if CheckMinorSymmetry == False:
        print('Tensor does not present minor symmetry')
    else:

        B = np.zeros((6,6))

        B[0, 0] = A[0, 0, 0, 0]
        B[0, 1] = A[0, 0, 1, 1]
        B[0, 2] = A[0, 0, 2, 2]
        B[0, 3] = np.sqrt(2) * A[0, 0, 1, 2]
        B[0, 4] = np.sqrt(2) * A[0, 0, 2, 0]
        B[0, 5] = np.sqrt(2) * A[0, 0, 0, 1]

        B[1, 0] = A[1, 1, 0, 0]
        B[1, 1] = A[1, 1, 1, 1]
        B[1, 2] = A[1, 1, 2, 2]
        B[1, 3] = np.sqrt(2) * A[1, 1, 1, 2]
        B[1, 4] = np.sqrt(2) * A[1, 1, 2, 0]
        B[1, 5] = np.sqrt(2) * A[1, 1, 0, 1]

        B[2, 0] = A[2, 2, 0, 0]
        B[2, 1] = A[2, 2, 1, 1]
        B[2, 2] = A[2, 2, 2, 2]
        B[2, 3] = np.sqrt(2) * A[2, 2, 1, 2]
        B[2, 4] = np.sqrt(2) * A[2, 2, 2, 0]
        B[2, 5] = np.sqrt(2) * A[2, 2, 0, 1]

        B[3, 0] = np.sqrt(2) * A[1, 2, 0, 0]
        B[3, 1] = np.sqrt(2) * A[1, 2, 1, 1]
        B[3, 2] = np.sqrt(2) * A[1, 2, 2, 2]
        B[3, 3] = 2 * A[1, 2, 1, 2]
        B[3, 4] = 2 * A[1, 2, 2, 0]
        B[3, 5] = 2 * A[1, 2, 0, 1]

        B[4, 0] = np.sqrt(2) * A[2, 0, 0, 0]
        B[4, 1] = np.sqrt(2) * A[2, 0, 1, 1]
        B[4, 2] = np.sqrt(2) * A[2, 0, 2, 2]
        B[4, 3] = 2 * A[2, 0, 1, 2]
        B[4, 4] = 2 * A[2, 0, 2, 0]
        B[4, 5] = 2 * A[2, 0, 0, 1]

        B[5, 0] = np.sqrt(2) * A[0, 1, 0, 0]
        B[5, 1] = np.sqrt(2) * A[0, 1, 1, 1]
        B[5, 2] = np.sqrt(2) * A[0, 1, 2, 2]
        B[5, 3] = 2 * A[0, 1, 1, 2]
        B[5, 4] = 2 * A[0, 1, 2, 0]
        B[5, 5] = 2 * A[0, 1, 0, 1]

        return B
    
def TransformTensor(A,OriginalBasis,NewBasis):

    # Build change of coordinate matrix
    O = OriginalBasis
    N = NewBasis

    Q = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Q[i,j] = np.dot(O[i,:],N[j,:])

    if A.size == 36:
        A4 = IsoMorphism66_3333(A)

    elif A.size == 81 and A.shape == (3,3,3,3):
        A4 = A

    TransformedA = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    TransformedA[i, j, k, l] += Q[m,i]*Q[n,j]*Q[o,k]*Q[p,l] * A4[m, n, o, p]
    if A.size == 36:
        TransformedA = IsoMorphism3333_66(TransformedA)

    return TransformedA

def Mandel2EngineeringNotation(A):

    B = np.zeros((6,6))

    for i in range(6):
        for j in range(6):

            if i < 3 and j >= 3:
                B[i,j] = A[i,j] / np.sqrt(2)

            elif i >= 3 and j < 3:
                B[i,j] = A[i,j] / np.sqrt(2)

            elif i >= 3 and j >= 3:
                B[i,j] = A[i,j] / 2

            else:
                B[i, j] = A[i, j]

    return B

def BoxPlot(ConstantsCtrl, ConstantsT2D, Idx, Labels, FigName):
    
    Figure, Axis = plt.subplots(1,1, dpi=100, figsize=(9,4.5))

    Max = 0
    for i in Idx:

        Ctrl = ConstantsCtrl[:,i]
        T2D = ConstantsT2D[:,i]

        if Max < Ctrl.max() or Max < T2D.max():
            Max = max([Ctrl.max(), T2D.max()])

        # Create random positions
        Array = np.sort(Ctrl)
        Norm = norm.pdf(np.linspace(-3,3,len(Array)), scale=1.5)
        Norm = Norm / max(Norm)
        CtrlPos = np.random.normal(0,0.03,len(Array)) * Norm

        Array = np.sort(T2D)
        Norm = norm.pdf(np.linspace(-3,3,len(Array)), scale=1.5)
        Norm = Norm / max(Norm)
        T2DPos = np.random.normal(0,0.03,len(Array)) * Norm

        # Plot data
        Axis.plot(CtrlPos - 0.2 + i, Ctrl, linestyle='none',
                marker='o',fillstyle='none', color=(0,0,1), ms=5)
        Axis.plot(T2DPos + 0.2 + i, T2D, linestyle='none',
                  marker='o',fillstyle='none', color=(1,0,0), ms=5)

        Axis.boxplot(Ctrl, vert=True, widths=0.3,
                    showmeans=False,meanline=False,
                    showfliers=False, positions=[i - 0.2],
                    capprops=dict(color=(0,0,1)),
                    boxprops=dict(color=(0,0,1)),
                    whiskerprops=dict(color=(0,0,1),linestyle='--'),
                    medianprops=dict(color=(0,0,1)))
        Axis.boxplot(T2D, vert=True, widths=0.3,
                    showmeans=False,meanline=False,
                    showfliers=False, positions=[i + 0.2],
                    capprops=dict(color=(1,0,0)),
                    boxprops=dict(color=(1,0,0)),
                    whiskerprops=dict(color=(1,0,0),linestyle='--'),
                    medianprops=dict(color=(1,0,0)))
        
        # Perform Mann-Whitney test for difference
        TestRes = mannwhitneyu(Ctrl, T2D)

        # Plot stars for significance
        YLine = 1.05 * max(Ctrl.max(), T2D.max())
        Plot = Axis.plot([i-0.2, i+0.2], [YLine, YLine], color=(0,0,0), marker='|',linewidth=0.5)
        MarkerSize = Plot[0].get_markersize()
                
        # Mark significance level
        if TestRes.pvalue < 0.001:
            Text = '***'
        elif TestRes.pvalue < 0.01:
            Text = '**' 
        elif TestRes.pvalue < 0.05:
            Text = '*'
        else:
            Text = 'n.s.'
        Axis.annotate(Text, xy=[i, YLine], ha='center',
                      xytext=(0, 1.2*MarkerSize), textcoords='offset points',)

    Axis.set_xticks(Idx)
    Axis.set_xticklabels(Labels, rotation=0)
    Axis.set_ylabel('Stiffness (MPa)')
    Axis.set_xlabel('Component (-)')

    # Adjust y limits
    Axis.set_ylim(0, Max*1.2)

    # Add legend
    Axis.plot([], linestyle='none', marker='o',fillstyle='none', color=(0,0,1), label='Ctrl')
    Axis.plot([], linestyle='none', marker='o',fillstyle='none', color=(1,0,0), label='T2D')
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.125))
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=196)
    plt.show(Figure)

    return

#%% Main

def Main():

    # Define files and simulation paths
    Data = pd.read_csv(Path(__file__).parents[1] / '02_Results/Morphometry.csv', index_col=[0,1])
    Abaqus = Path(__file__).parents[1] / '02_Results/Abaqus'
    Morpho = Path(__file__).parents[1] / '02_Results/Morphometry'

    # Read metadata file
    MetaData = pd.read_csv(Path(__file__).parents[1] / '00_Data/SampleList.csv')
    Ctrl = np.repeat(MetaData['Group'].values == 'Ctrl',3)
    T2D = np.repeat(MetaData['Group'].values == 'T2D',3)

    # Filter out ROIs of cortical bone
    F = Data['BV/TV'] < 0.5

    # Compare morphometric values
    Stats = pd.DataFrame()
    for Col in Data.columns:

        # Get values
        Controls = Data[Ctrl&F][Col].values
        Diabetics = Data[T2D&F][Col].values

        # Shapiro-Wilk test for normality
        W, p_Control = shapiro(Controls)
        W, p_Diabetics = shapiro(Diabetics)


        # Check equal variances
        if p_Control > 5/100 and p_Diabetics > 5/100:

            # If both groups shows normal distribution
            Dist = 'Normal'
            Test, pValue = bartlett(Controls, Diabetics)

        else:

            # If not, Brown – Forsythe test
            Dist = 'Not-normal'
            Test, pValue = levene(Controls, Diabetics)

        if pValue > 5/100:

            Variance = 'Equal'
            if Dist == 'Normal':
                Test = 't-test'
                TestRes = ttest_ind(Controls, Diabetics)

            else:
                Test = 'MannWhitney'
                TestRes = mannwhitneyu(Controls, Diabetics)

        else:

            Variance = 'Not-equal'
            Test = 'Permutation'
            TestRes = permutation_test([Controls, Diabetics], statistic)

        # Force mann-whitney test    
        Test = 'MannWhitney'
        TestRes = mannwhitneyu(Controls, Diabetics)


        # Store results
        Stats.loc[Col,'Distributions'] = Dist 
        Stats.loc[Col,'Variances'] = Variance
        Stats.loc[Col,'Test'] = Test
        Stats.loc[Col,'p-value'] = TestRes.pvalue
        Stats.loc[Col,'Ctrl Median'] = np.median(Controls)
        Stats.loc[Col,'Ctrl 0.25 IQR'] = np.quantile(Controls,0.25)
        Stats.loc[Col,'Ctrl 0.75 IQR'] = np.quantile(Controls,0.75)
        Stats.loc[Col,'T2D Median'] = np.median(Diabetics)
        Stats.loc[Col,'T2D 0.25 IQR'] = np.quantile(Diabetics,0.25)
        Stats.loc[Col,'T2D 0.75 IQR'] = np.quantile(Diabetics,0.75)

    print(Stats.round(2))

    # Get stiffness values
    Strain = np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002])
    ConstantsCtrl, ConstantsT2D = [], []
    for Idx, Row in Data.iterrows():

        if Row['CV'] < 0.263 and Row['BV/TV'] < 0.5:

            # Determine group and location
            Group = MetaData[MetaData['Sample'] == Idx[0]]['Group']

            # Step 1: Get fabric info
            MorphoFile = pd.read_csv(Morpho / (Idx[0] + '_' + str(Idx[1]) + '.csv'), delimiter=';')
            BVTV = MorphoFile['$BVTV_voxel'].values[0]

            # Eigenvalues
            m1 = MorphoFile['$DA_lam_1'].values[0]
            m2 = MorphoFile['$DA_lam_2'].values[0]
            m3 = MorphoFile['$DA_lam_3'].values[0]
            eValues = np.array([m1,m2,m3])

            # Eigenvectors
            m11 = MorphoFile['$DA_vec_1x'].values[0]
            m12 = MorphoFile['$DA_vec_1y'].values[0]
            m13 = MorphoFile['$DA_vec_1z'].values[0]

            m21 = MorphoFile['$DA_vec_2x'].values[0]
            m22 = MorphoFile['$DA_vec_2y'].values[0]
            m23 = MorphoFile['$DA_vec_2z'].values[0]

            m31 = MorphoFile['$DA_vec_3x'].values[0]
            m32 = MorphoFile['$DA_vec_3y'].values[0]
            m33 = MorphoFile['$DA_vec_3z'].values[0]
            eVectors = np.array([[m11,m12,m13], [m21,m22,m23], [m31,m32,m33]])

            # Sort fabric
            Arg = np.argsort(eValues)
            eValues = eValues[Arg]
            eVectors = eVectors[Arg]
            m1, m2, m3 = eValues

            # Get stiffness
            File = open(Abaqus / (Idx[0] + '_' + str(Idx[1]) + '.out')).readlines()

            Stress = np.zeros((6,6))
            for i in range(6):
                for j in range(6):
                    Stress[i,j] = float(File[i+4].split()[j+1])

            Stiffness = np.zeros((6,6))
            for i in range(6):
                for j in range(6):
                    Stiffness[i,j] = Stress[i,j] / Strain[i]

            # Symetrize matrix
            Stiffness = 1/2 * (Stiffness + Stiffness.T)

            # Write tensor into mandel notation
            Mandel = Engineering2MandelNotation(Stiffness)

            # Step 3: Transform tensor into fabric coordinate system
            I = np.eye(3)
            Q = np.array(eVectors)
            Transformed = TransformTensor(Mandel, I, Q)

            # Project onto orthotropy
            Orthotropic = np.zeros(Transformed.shape)
            for i in range(Orthotropic.shape[0]):
                for j in range(Orthotropic.shape[1]):
                    if i < 3 and j < 3:
                        Orthotropic[i, j] = Transformed[i, j]
                    elif i == j:
                        Orthotropic[i, j] = Transformed[i, j]

            # Get tensor back to engineering notation
            Stiffness = Mandel2EngineeringNotation(Orthotropic)

            # Store resulting constants
            Constants = [Stiffness[0,0],
                        Stiffness[1,1],
                        Stiffness[2,2],
                        Stiffness[0,1],
                        Stiffness[0,2],
                        Stiffness[1,2],
                        Stiffness[3,3],
                        Stiffness[4,4],
                        Stiffness[5,5]]
            
            if Group.values[0] == 'Ctrl':
                ConstantsCtrl.append(Constants)
            elif Group.values[0] == 'T2D':
                ConstantsT2D.append(Constants)

    ConstantsCtrl = np.array(ConstantsCtrl)
    ConstantsT2D = np.array(ConstantsT2D)

    # Boxplot stiffnesses
    Labels = [r'$\lambda_{11}$',r'$\lambda_{22}$',r'$\lambda_{33}$']
    FigName = Path(__file__).parents[1] / '02_Results/Lii.png'
    BoxPlot(ConstantsCtrl, ConstantsT2D, np.arange(0,3), Labels, FigName)
    Labels = [r'$\lambda_{12}$',r'$\lambda_{13}$',r'$\lambda_{23}$']
    FigName = Path(__file__).parents[1] / '02_Results/Lij.png'
    BoxPlot(ConstantsCtrl, ConstantsT2D, np.arange(3,6), Labels, FigName)
    Labels = [r'$\mu_{23}$',r'$\mu_{31}$',r'$\mu_{12}$']
    FigName = Path(__file__).parents[1] / '02_Results/Mii.png'
    BoxPlot(ConstantsCtrl, ConstantsT2D, np.arange(6,9), Labels, FigName)


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

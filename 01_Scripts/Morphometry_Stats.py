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
from scipy.stats import shapiro, levene, bartlett
from scipy.stats import ttest_ind, mannwhitneyu, permutation_test


#%% Functions

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

#%% Main

def Main():

    # Morphometric data
    Data = pd.read_csv(Path(__file__).parents[1] / '02_Results/Morphometry.csv', index_col=[0,1])

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

    Stats = pd.DataFrame()
    for Col in Data.columns:

        # Get values
        Controls = Data[Ctrl][Col].values
        Diabetics = Data[T2D][Col].values

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
            
        # Store results
        Stats.loc[Col,'Distributions'] = Dist 
        Stats.loc[Col,'Variances'] = Variance
        Stats.loc[Col,'Test'] = Test
        Stats.loc[Col,'p-value'] = TestRes.pvalue
        Stats.loc[Col,'Ctrl Mean'] = Controls.mean()
        Stats.loc[Col,'Ctrl Std'] = Controls.std()
        Stats.loc[Col,'T2D Mean'] = Diabetics.mean()
        Stats.loc[Col,'T2D Std'] = Diabetics.std()

    print(Stats)

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

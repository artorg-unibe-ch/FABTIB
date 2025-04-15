#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '12-11-2024'
__date__ = '15-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

np.set_printoptions(linewidth=300,
                    suppress=True,
                    formatter={'float_kind':'{:3}'.format})
#%% Functions

def DyadicProduct(A,B):

    if A.size == 3:
        C = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                C[i,j] = A[i]*B[j]

    elif A.size == 9:
        C = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                            C[i,j,k,l] = A[i,j] * B[k,l]

    else:
        print('Matrices sizes mismatch')

    return C

def SymmetricProduct(A,B):

    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = (1/2)*(A[i,k]*B[j,l]+A[i,l]*B[j,k])

    return C

def ComplianceTensor(E1, E2, E3, Mu23, Mu31, Mu12, Nu12, Nu13, Nu23, EigenVectors=np.eye(3)):

    # Define constants
    Mu32, Mu13, Mu21 = Mu23, Mu31, Mu12
    Nu21 = Nu12 * E2 / E1
    Nu31 = Nu13 * E3 / E1
    Nu32 = Nu23 * E3 / E2

    # Group into list for loop computation
    E = [E1, E2, E3]
    Nu = np.array([[Nu13, Nu12], [Nu21, Nu23], [Nu32, Nu31]])
    Mu = np.array([[Mu13, Mu12], [Mu21, Mu23], [Mu32, Mu31]])

    # Build compliance tensor
    ComplianceTensor = np.zeros((3, 3, 3, 3))
    for i in range(3):
        Mi = DyadicProduct(EigenVectors[i], EigenVectors[i])
        Part1 = 1 / E[i] * DyadicProduct(Mi, Mi)
        ComplianceTensor += Part1

        for ii in range(3 - 1):
            j = i - ii - 1
            Mj = DyadicProduct(EigenVectors[j], EigenVectors[j])
            Part2 = -Nu[i, ii] / E[i] * DyadicProduct(Mi, Mj)
            Part3 = 1 / (2 * Mu[i, ii]) * SymmetricProduct(Mi, Mj)
            ComplianceTensor += Part2 + Part3

    ComplianceTensor = IsoMorphism3333_66(ComplianceTensor)

    return ComplianceTensor

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

def OLS(X, Y, FName, Alpha=0.95, Colors=[(0,0,1),(0,1,0),(1,0,0)]):

    # Solve linear system
    XTXi = np.linalg.inv(X.T * X)
    B = XTXi * X.T * Y

    # Compute residuals, variance, and covariance matrix
    Y_Obs = np.exp(Y)
    Y_Fit = np.exp(X * B)
    Residuals = Y - X*B
    DOFs = len(Y) - X.shape[1]
    Sigma = Residuals.T * Residuals / DOFs
    Cov = Sigma[0,0] * XTXi

    # Compute B confidence interval
    t_Alpha = t.interval(Alpha, DOFs)
    B_CI_Low = B.T + t_Alpha[0] * np.sqrt(np.diag(Cov))
    B_CI_Top = B.T + t_Alpha[1] * np.sqrt(np.diag(Cov))

    # Store parameters in data frame
    Parameters = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0','k','l'])
    Parameters.loc['Value'] = [np.exp(B[0,0]) - 2*np.exp(B[2,0]), np.exp(B[1,0]), np.exp(B[2,0]), B[3,0], B[4,0]]
    Parameters.loc['95% CI Low'] = [np.exp(B_CI_Low[0,0]) - 2*np.exp(B_CI_Top[0,2]), np.exp(B_CI_Low[0,1]), np.exp(B_CI_Low[0,2]), B_CI_Low[0,3], B_CI_Low[0,4]]
    Parameters.loc['95% CI Top'] = [np.exp(B_CI_Top[0,0]) - 2*np.exp(B_CI_Low[0,2]), np.exp(B_CI_Top[0,1]), np.exp(B_CI_Top[0,2]), B_CI_Top[0,3], B_CI_Top[0,4]]

    # Compute R2 and standard error of the estimate
    RSS = np.sum([R**2 for R in Residuals])
    SE = np.sqrt(RSS / DOFs)
    TSS = np.sum([R**2 for R in (Y - Y.mean())])
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    # Compute R2adj and NE
    R2adj = 1 - RSS/TSS * (len(Y)-1)/(len(Y)-X.shape[1]-1)

    NE = []
    for i in range(0,len(Y),12):
        T_Obs = Y_Obs[i:i+12]
        T_Fit = Y_Fit[i:i+12]
        Numerator = np.sum([T**2 for T in (T_Obs-T_Fit)])
        Denominator = np.sum([T**2 for T in T_Obs])
        NE.append(np.sqrt(Numerator/Denominator))
    NE = np.array(NE)


    # Prepare data for plot
    Line = np.linspace(min(Y.min(), (X*B).min()),
                       max(Y.max(), (X*B).max()), len(Y))
    # B_0 = np.sort(np.sqrt(np.diag(X * Cov * X.T)))
    # CI_Line_u = np.exp(Line + t_Alpha[0] * B_0)
    # CI_Line_o = np.exp(Line + t_Alpha[1] * B_0)

    # Plots
    DPI = 500
    SMax = max([Y_Obs.max(), Y_Fit.max()]) * 5
    SMin = min([Y_Obs.min(), Y_Fit.min()]) / 5

    # Set boundaries of fabtib
    SMax = 1e4
    SMin = 5e1

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI)
    # Axes.fill_between(np.exp(Line), CI_Line_u, CI_Line_o, color=(0.8,0.8,0.8))
    Axes.plot(Y_Obs[X[:, 0] == 1], Y_Fit[X[:, 0] == 1],
              color=Colors[0], linestyle='none', marker='s')
    Axes.plot(Y_Obs[X[:, 1] == 1], Y_Fit[X[:, 1] == 1],
              color=Colors[1], linestyle='none', marker='o')
    Axes.plot(Y_Obs[X[:, 2] == 1], Y_Fit[X[:, 2] == 1],
              color=Colors[2], linestyle='none', marker='^')
    Axes.plot([], color=Colors[0], linestyle='none', marker='s', label=r'$\lambda_{ii}$')
    Axes.plot([], color=Colors[1], linestyle='none', marker='o', label=r'$\lambda_{ij}$')
    Axes.plot([], color=Colors[2], linestyle='none', marker='^', label=r'$\mu_{ij}$')
    Axes.plot(np.exp(Line), np.exp(Line), color=(0, 0, 0), linestyle='--')
    Axes.annotate(r'N ROIs   : ' + str(len(Y)//12), xy=(0.3, 0.1), xycoords='axes fraction')
    Axes.annotate(r'N Points : ' + str(len(Y)), xy=(0.3, 0.025), xycoords='axes fraction')
    Axes.annotate(r'$R^2_{ajd}$: ' + format(round(R2adj, 3),'.3f'), xy=(0.65, 0.1), xycoords='axes fraction')
    Axes.annotate(r'NE : ' + format(round(NE.mean(), 2), '.2f') + r'$\pm$' + format(round(NE.std(), 2), '.2f'), xy=(0.65, 0.025), xycoords='axes fraction')
    Axes.set_xlabel(r'Observed $\mathrm{\mathbb{S}}$ (MPa)')
    Axes.set_ylabel(r'Fitted $\mathrm{\mathbb{S}}$ (MPa)')
    Axes.set_xlim([SMin, SMax])
    Axes.set_ylim([SMin, SMax])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(FName)
    plt.show()

    return Parameters, R2adj, NE

def SolveSeparate(Parameters,Xh,Yh,Xd,Yd):

    # Solve linear systems
    X = np.matrix(np.vstack(Xh))
    Y = np.matrix(np.vstack(Yh))
    FName = Path(__file__).parents[1] / '02_Results/FabricElasticity_Ctrl.png'
    Parametersh, R2adj, NE = OLS(X, Y, FName, Colors=[(0,0,1),(0,0.6,1),(0,1,1)])

    X = np.matrix(np.vstack(Xd))
    Y = np.matrix(np.vstack(Yd))
    FName = Path(__file__).parents[1] / '02_Results/FabricElasticity_T2D.png'
    Parametersd, R2adj, NE = OLS(X, Y, FName, Colors=[(1,0,0),(1,0,1),(0.6,0,1)])

    # Plot 95% CI
    Colors = [(0,0,0),(0,0,1),(1,0,0)]
    Variables = ['Lambda0','Lambda0p','Mu0','k','l']
    Figure, Axis = plt.subplots(1,len(Variables), figsize=(2*len(Variables)+2,4), sharey=False)
    for v, Variable in enumerate(Variables):
        for i, P in enumerate([Parameters, Parametersh, Parametersd]):
            V = P.loc['Value',Variable]
            V_Low = abs(P.loc['Value',Variable] - P.loc['95% CI Low',Variable])
            V_Top = abs(P.loc['95% CI Top',Variable] - P.loc['Value',Variable])
            Axis[v].errorbar([i], V, yerr=[[V_Low], [V_Top]], marker='o', color=Colors[i])
            Axis[v].set_xlabel(Variables[v])
            Axis[v].set_xticks(range(3),['Grouped','Ctrl','T2D'])
    Axis[0].set_ylabel('Values (-)')
    plt.tight_layout()
    plt.show(Figure)
    return

def OLS_Exponents(X, Y, k0, l0, Alpha=0.95):

    # Solve linear system
    XTXi = np.linalg.inv(X.T * X)
    B = XTXi * X.T * Y

    # Compute residuals, variance, and covariance matrix
    Y_Obs = np.exp(Y)
    Y_Fit = np.exp(X * B)
    Residuals = Y - X*B
    DOFs = len(Y) - X.shape[1]
    Sigma = Residuals.T * Residuals / DOFs
    Cov = Sigma[0,0] * XTXi

    # Compute B confidence interval
    t_Alpha = t.interval(Alpha, DOFs)
    B_CI_Low = B.T + t_Alpha[0] * np.sqrt(np.diag(Cov))
    B_CI_Top = B.T + t_Alpha[1] * np.sqrt(np.diag(Cov))

    # Store parameters in data frame
    Parameters = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0','k','l'])
    Parameters.loc['Value'] = [np.exp(B[0,0])-2*np.exp(B[2,0]), np.exp(B[1,0]), np.exp(B[2,0]), k0, l0]
    Parameters.loc['95% CI Low'] = [np.exp(B_CI_Top[0,0])-2*np.exp(B_CI_Top[0,2]), np.exp(B_CI_Top[0,1]), np.exp(B_CI_Top[0,2]), k0, l0]
    Parameters.loc['95% CI Top'] = [np.exp(B_CI_Low[0,0])-2*np.exp(B_CI_Low[0,2]), np.exp(B_CI_Low[0,1]), np.exp(B_CI_Low[0,2]), k0, l0]

    # Compute R2 and standard error of the estimate
    RSS = np.sum([R**2 for R in Residuals])
    SE = np.sqrt(RSS / DOFs)
    TSS = np.sum([R**2 for R in (Y - Y.mean())])
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    # Compute R2adj and NE
    R2adj = 1 - RSS/TSS * (len(Y)-1)/(len(Y)-X.shape[1]-1)

    NE = []
    for i in range(0,len(Y),12):
        T_Obs = Y_Obs[i:i+12]
        T_Fit = Y_Fit[i:i+12]
        Numerator = np.sum([T**2 for T in (T_Obs-T_Fit)])
        Denominator = np.sum([T**2 for T in T_Obs])
        NE.append(np.sqrt(Numerator/Denominator))
    NE = np.array(NE)

    return Parameters, R2adj, NE

def Solve_Exponents(Parameters, Xh, Yh, Xd, Yd):

    k0 = Parameters.loc['Value','k']
    l0 = Parameters.loc['Value','l']

    X = np.matrix(np.vstack(Xh))
    Y = np.matrix(np.vstack(Yh))
    Yr = Y - X[:,3]*k0 - X[:,4]*l0
    Parametersh, R2adjh, NEh = OLS_Exponents(X[:,:3], Yr , k0, l0)

    X = np.matrix(np.vstack(Xd))
    Y = np.matrix(np.vstack(Yd))
    Yr = Y - X[:,3]*k0 - X[:,4]*l0
    Parametersd, R2adjh, NEh = OLS_Exponents(X[:,:3], Yr , k0, l0)

    # Plot 95% CI
    Colors = [(0,0,0),(0,0,1),(1,0,0)]
    Variables = ['Lambda0','Lambda0p','Mu0']
    Figure, Axis = plt.subplots(1,len(Variables), figsize=(2*len(Variables)+2,4), sharey=False)
    for v, Variable in enumerate(Variables):
        for i, P in enumerate([Parameters, Parametersh, Parametersd]):
            V = P.loc['Value',Variable]
            V_Low = abs(P.loc['Value',Variable] - P.loc['95% CI Low',Variable])
            V_Top = abs(P.loc['95% CI Top',Variable] - P.loc['Value',Variable])
            Axis[v].errorbar([i], V, yerr=[[V_Low], [V_Top]], marker='o', color=Colors[i])
            Axis[v].set_xlabel(Variables[v])
            Axis[v].set_xticks(range(3),['Grouped','Ctrl','T2D'])
    Axis[0].set_ylabel('Values (-)')
    plt.tight_layout()
    plt.savefig(Path(__file__).parents[1] / '02_Results/Stiffness.png')
    plt.show(Figure)
    return

def OLS_Stiffness(X, Y, L0, L0p, M0, Alpha=0.95):

    # Solve linear system
    XTXi = np.linalg.inv(X.T * X)
    B = XTXi * X.T * Y

    # Compute residuals, variance, and covariance matrix
    Y_Obs = np.exp(Y)
    Y_Fit = np.exp(X * B)
    Residuals = Y - X*B
    DOFs = len(Y) - X.shape[1]
    Sigma = Residuals.T * Residuals / DOFs
    Cov = Sigma[0,0] * XTXi

    # Compute B confidence interval
    t_Alpha = t.interval(Alpha, DOFs)
    B_CI_Low = B.T + t_Alpha[0] * np.sqrt(np.diag(Cov))
    B_CI_Top = B.T + t_Alpha[1] * np.sqrt(np.diag(Cov))

    # Store parameters in data frame
    Parameters = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0','k','l'])
    Parameters.loc['Value'] = [L0, L0p, M0, B[0,0], B[1,0]]
    Parameters.loc['95% CI Low'] = [L0, L0p, M0, B_CI_Top[0,0], B_CI_Top[0,1]]
    Parameters.loc['95% CI Top'] = [L0, L0p, M0, B_CI_Low[0,0], B_CI_Low[0,1]]

    # Compute R2 and standard error of the estimate
    RSS = np.sum([R**2 for R in Residuals])
    SE = np.sqrt(RSS / DOFs)
    TSS = np.sum([R**2 for R in (Y - Y.mean())])
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    # Compute R2adj and NE
    R2adj = 1 - RSS/TSS * (len(Y)-1)/(len(Y)-X.shape[1]-1)

    NE = []
    for i in range(0,len(Y),12):
        T_Obs = Y_Obs[i:i+12]
        T_Fit = Y_Fit[i:i+12]
        Numerator = np.sum([T**2 for T in (T_Obs-T_Fit)])
        Denominator = np.sum([T**2 for T in T_Obs])
        NE.append(np.sqrt(Numerator/Denominator))
    NE = np.array(NE)

    return Parameters, R2adj, NE

def Solve_Stiffness(Parameters, Xh, Yh, Xd, Yd):

    L0 = Parameters.loc['Value','Lambda0']
    L0p = Parameters.loc['Value','Lambda0p']
    M0 = Parameters.loc['Value','Mu0']

    X = np.matrix(np.vstack(Xh))
    Y = np.matrix(np.vstack(Yh))
    Yr = Y - X[:,0]*np.log(L0+2*M0) - X[:,1]*np.log(L0p) - X[:,2]*np.log(M0)
    Parametersh, R2adjh, NEh = OLS_Stiffness(X[:,3:], Yr , L0, L0p, M0)
    
    X = np.matrix(np.vstack(Xd))
    Y = np.matrix(np.vstack(Yd))
    Yr = Y - X[:,0]*np.log(L0+2*M0) - X[:,1]*np.log(L0p) - X[:,2]*np.log(M0)
    Parametersd, R2adjh, NEh = OLS_Stiffness(X[:,3:], Yr , L0, L0p, M0)

    # Plot 95% CI
    Colors = [(0,0,0),(0,0,1),(1,0,0)]
    Variables = ['k','l']
    Figure, Axis = plt.subplots(1,len(Variables), figsize=(2*len(Variables)+2,4), sharey=False)
    for v, Variable in enumerate(Variables):
        for i, P in enumerate([Parameters, Parametersh, Parametersd]):
            V = P.loc['Value',Variable]
            V_Low = abs(P.loc['Value',Variable] - P.loc['95% CI Low',Variable])
            V_Top = abs(P.loc['95% CI Top',Variable] - P.loc['Value',Variable])
            Axis[v].errorbar([i], V, yerr=[[V_Low], [V_Top]], marker='o', color=Colors[i])
            Axis[v].set_xlabel(Variables[v])
            Axis[v].set_xticks(range(3),['Grouped','Ctrl','T2D'])
    Axis[0].set_ylabel('Values (-)')
    plt.tight_layout()
    plt.savefig(Path(__file__).parents[1] / '02_Results/Exponents.png')
    plt.show(Figure)
    return Parametersh, Parametersd

#%% Main

def Main():

    # Define paths
    MorphoPath = Path(__file__).parents[1] / '02_Results/Morphometry'
    AbaqusPath = Path(__file__).parents[1] / '02_Results/Abaqus'

    # Read Arguments
    Samples = sorted([F.name[:-4] for F in Path.iterdir(AbaqusPath) if F.name.endswith('.out')])

    # Read metadata file
    Data = pd.read_csv(Path(__file__).parents[1] / '00_Data/SampleList.csv')

    # Define the two groups
    Ctrl = Data['Group'] == 'Ctrl'
    T2D = Data['Group'] == 'T2D'
    Ctrl = Data[Ctrl]['Sample'].values
    T2D = Data[T2D]['Sample'].values

    # Collect BVTV and DA
    Morpho = pd.read_csv(Path(__file__).parents[1] / '02_Results/Morphometry.csv', index_col=[0,1])
    Ctrl_Morpho, T2D_Morpho = [], []
    CtrlList, T2DList = [], []
    for s, Sample in Morpho.iterrows():
        BVTV = Sample['BV/TV']
        DA = Sample['DA']
        CV = Sample['CV']
        if s[0] in Ctrl:
            Ctrl_Morpho.append([BVTV, DA, CV])
            CtrlList.append([s[0], s[1]])
        elif s[0] in T2D:
            T2D_Morpho.append([BVTV, DA, CV])
            T2DList.append([s[0], s[1]])
    Ctrl_Morpho = np.array(Ctrl_Morpho)
    T2D_Morpho = np.array(T2D_Morpho)

    # Filter samples
    Ctrl_Morpho = Ctrl_Morpho[Ctrl_Morpho[:, 0] < 0.5]
    T2D_Morpho = T2D_Morpho[T2D_Morpho[:, 0] < 0.5]
    Ctrl_Morpho = Ctrl_Morpho[Ctrl_Morpho[:, 2] < 0.263]
    T2D_Morpho = T2D_Morpho[T2D_Morpho[:, 2] < 0.263]
    Ctrl_Morpho = Ctrl_Morpho[:,:2]
    T2D_Morpho = T2D_Morpho[:,:2]
    
    # Compute samples differences
    Differences = Ctrl_Morpho[:, np.newaxis, :] - T2D_Morpho[np.newaxis, :, :]
    Distances = np.sqrt(np.sum(Differences**2, axis=2))

    # Solve the assignment problem using the Hungarian algorithm
    Ctrl_Idx, T2D_Idx = linear_sum_assignment(Distances)
    Ctrl_Samples = [CtrlList[I][0] + '_' + str(CtrlList[I][1]) for I in Ctrl_Idx]
    T2D_Samples = [T2DList[I][0] + '_' + str(T2DList[I][1]) for I in T2D_Idx]

    # Collect homogenization data
    Strain = np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002])
    Xh, Yh, Xd, Yd = [], [], [], []
    for s, Sample in enumerate(Samples):

        # Step 1: Get fabric info
        Morpho = pd.read_csv(MorphoPath / (Sample + '.csv'), delimiter=';')
        BVTV = Morpho['$BVTV_voxel'].values[0]

        # Eigenvalues
        m1 = Morpho['$DA_lam_1'].values[0]
        m2 = Morpho['$DA_lam_2'].values[0]
        m3 = Morpho['$DA_lam_3'].values[0]
        eValues = np.array([m1,m2,m3])

        # Eigenvectors
        m11 = Morpho['$DA_vec_1x'].values[0]
        m12 = Morpho['$DA_vec_1y'].values[0]
        m13 = Morpho['$DA_vec_1z'].values[0]

        m21 = Morpho['$DA_vec_2x'].values[0]
        m22 = Morpho['$DA_vec_2y'].values[0]
        m23 = Morpho['$DA_vec_2z'].values[0]

        m31 = Morpho['$DA_vec_3x'].values[0]
        m32 = Morpho['$DA_vec_3y'].values[0]
        m33 = Morpho['$DA_vec_3z'].values[0]
        eVectors = np.array([[m11,m12,m13], [m21,m22,m23], [m31,m32,m33]])

        # Sort fabric
        Arg = np.argsort(eValues)
        eValues = eValues[Arg]
        eVectors = eVectors[Arg]
        m1, m2, m3 = eValues

        # Step 2: Get stress results
        Abaqus = open(AbaqusPath / (Sample + '.out'), 'r').readlines()

        Stress = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stress[i,j] = float(Abaqus[i+4].split()[j+1])

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

        # Build linear system
        Y = np.log([[Stiffness[0,0]],
                    [Stiffness[0,1]],
                    [Stiffness[0,2]],
                    [Stiffness[1,0]],
                    [Stiffness[1,1]],
                    [Stiffness[1,2]],
                    [Stiffness[2,0]],
                    [Stiffness[2,1]],
                    [Stiffness[2,2]],
                    [Stiffness[3,3]],
                    [Stiffness[4,4]],
                    [Stiffness[5,5]]])
        
        X = np.array([[1, 0, 0, np.log(BVTV), np.log(m1 ** 2)],
                      [0, 1, 0, np.log(BVTV), np.log(m1 * m2)],
                      [0, 1, 0, np.log(BVTV), np.log(m1 * m3)],
                      [0, 1, 0, np.log(BVTV), np.log(m2 * m1)],
                      [1, 0, 0, np.log(BVTV), np.log(m2 ** 2)],
                      [0, 1, 0, np.log(BVTV), np.log(m2 * m3)],
                      [0, 1, 0, np.log(BVTV), np.log(m3 * m1)],
                      [0, 1, 0, np.log(BVTV), np.log(m3 * m2)],
                      [1, 0, 0, np.log(BVTV), np.log(m3 ** 2)],
                      [0, 0, 1, np.log(BVTV), np.log(m2 * m3)],
                      [0, 0, 1, np.log(BVTV), np.log(m3 * m1)],
                      [0, 0, 1, np.log(BVTV), np.log(m1 * m2)]])

        # Store data in corresponding group
        if Sample in Ctrl_Samples:
            Xh.append(X)
            Yh.append(Y)
        elif Sample in T2D_Samples:
            Xd.append(X)
            Yd.append(Y)



    # Determine k and l
    X = np.matrix(np.vstack([np.vstack(Xh),np.vstack(Xd)]))
    Y = np.matrix(np.vstack([np.vstack(Yh),np.vstack(Yd)]))
    FName = Path(__file__).parents[1] / '02_Results/FabricElasticity_Grouped.png'
    Parameters, R2adjg, NEg = OLS(X, Y, FName)

    # Solve separated linear systems
    SolveSeparate(Parameters, Xh, Yh, Xd, Yd)

    # Impose exponent values as in previous work
    Solve_Exponents(Parameters, Xh, Yh, Xd, Yd)

    # Impose stiffness values to compare exponents
    Ph, Pd = Solve_Stiffness(Parameters, Xh, Yh, Xd, Yd)

    # Build stiffness and compliance tensors for the full material
    S = np.zeros((6,6))
    for i in range(6):
        if i < 3:
            S[i,i] = Parameters.loc['Value','Lambda0'] + 2*Parameters.loc['Value','Mu0']
            for j in range(3):
                if i != j:
                    S[i,j] = Parameters.loc['Value','Lambda0p']
        else:
            S[i,i] = 2*Parameters.loc['Value','Mu0']
    E = np.linalg.inv(S)

    # Plot theorical stiffness as function of rho with obtained parameters
    Rho = np.linspace(0, 0.5)
    E1_h = 1/E[0,0] * Rho**Ph.loc['Value','k']
    E1_d = 1/E[0,0] * Rho**Pd.loc['Value','k']
    Figure, Axis = plt.subplots(1,1, dpi=200)
    Axis.plot(Rho, E1_h, color=(0,0,1), label='Ctrl')
    Axis.plot(Rho, E1_d, color=(1,0,0), label='T2D')
    Axis.set_xlabel(r'$\rho$ (-)')
    Axis.set_ylabel('Modulus (MPa)')
    plt.legend()
    plt.savefig(Path(__file__).parents[1] / '02_Results/Modulus_vs_Rho.png')
    plt.show(Figure)


    # Plot study with OI
    Colors = [(0,0,0),(0,0,1),(1,0,0)]
    Variables = ['Lambda0','Lambda0p','Mu0']
    Grouped = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0'])
    Grouped.loc['Value'] = [4626, 2695, 3541]
    Grouped.loc['95% CI Low'] = [3892, 2472, 3246]
    Grouped.loc['95% CI Top'] = [5494, 2937, 3862]

    Healthy = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0'])
    Healthy.loc['Value'] = [4318, 2685, 3512]
    Healthy.loc['95% CI Low'] = [3844, 2533, 3306]
    Healthy.loc['95% CI Top'] = [4851, 2845, 3731]

    OI = pd.DataFrame(columns=['Lambda0','Lambda0p','Mu0'])
    OI.loc['Value'] = [4983, 2727, 3600]
    OI.loc['95% CI Low'] = [4345, 2547, 3355]
    OI.loc['95% CI Top'] = [5716, 2921, 3863]


    Figure, Axis = plt.subplots(1,len(Variables), figsize=(2*len(Variables)+2,4), sharey=False)
    for v, Variable in enumerate(Variables):
        for i, P in enumerate([Grouped, Healthy, OI]):
            V = P.loc['Value',Variable]
            V_Low = abs(P.loc['Value',Variable] - P.loc['95% CI Low',Variable])
            V_Top = abs(P.loc['95% CI Top',Variable] - P.loc['Value',Variable])
            Axis[v].errorbar([i], V, yerr=[[V_Low], [V_Top]], marker='o', color=Colors[i])
            Axis[v].set_xlabel(Variables[v])
            Axis[v].set_xticks(range(3),['Grouped','Ctrl','OI'])
    Axis[0].set_ylabel('Values (-)')
    plt.tight_layout()
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

#%%

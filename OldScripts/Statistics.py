#%% #!/usr/bin/env python3

"""
Script description
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
import TensorAlgebra as TA
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#%% Functions

def ReadDat(File):

    with open(File) as F:
        Text = F.read()

    Lines = Text.split('\n')

    S = np.zeros((6,6))
    for i in range(6):
        S[i] = np.array(Lines[48+i].split(),float)

    return S

def ReadFab(File):

    with open(File) as F:
        Text = F.read()

    Lines = Text.split('\n')

    eValues = np.array(Lines[18].split()[-3:],float)
    eVectors = np.zeros((3,3))
    for i in range(3):
        eVectors[i] = np.array(Lines[19+i].split()[-3:],float)

    BVTV = float(Lines[12].split()[-1])

    return eValues, eVectors, BVTV

#%% Main

FilePath = Path.cwd().parent / '01_Data/'
ImagePath = Path.cwd().parent / '03_Results/01_ROIs'
FabricPath = Path.cwd().parent / '03_Results/02_Fabric'
HomPath = Path.cwd().parent / '03_Results/03_Homogenization'
OldPath = Path.cwd().parent / 'OldResults/'
Files = sorted([F for F in FilePath.iterdir() if F.name.endswith('.mhd')])

Data = pd.read_csv(Path.cwd().parent / '03_Results/Morphometry.csv', index_col=0)

#%%

S = np.zeros((len(Data),6,6))
eVectors = np.zeros((len(Data),3,3))
m1, m2, m3 = np.zeros(len(Data)), np.zeros(len(Data)), np.zeros(len(Data))
BVTV = np.zeros(len(Data))

OldS = np.zeros((len(Data),6,6))
OldBVTV = np.zeros(len(Data))
Oldm1, Oldm2, Oldm3 = np.zeros(len(Data)), np.zeros(len(Data)), np.zeros(len(Data))
OldeVectors = np.zeros((len(Data),3,3))

for n, ROI in enumerate(Data.index):

    # Read fabric
    eValues = np.load(FabricPath / (ROI + '_eValues.npy'))
    # eVectors[n] = np.load(FabricPath / (ROI + '_eVectors.npy'))

    # BVTV[n] = Data.loc[ROI,'BVTV']
    m1[n] = eValues[0]
    m2[n] = eValues[1]
    m3[n] = eValues[2]

    OldS[n] = ReadDat(OldPath / (ROI[:-10] + '.dat'))
    # eValues, eVectors[n], OldBVTV[n] = ReadFab(OldPath / (ROI[:-10] + '.fab'))
    eValues = np.load(FabricPath / (ROI + '_eValues_MT.npy'))
    Oldm1[n] = eValues[0]
    Oldm2[n] = eValues[1]
    Oldm3[n] = eValues[2]

    # Read stiffness and symmetrize it
    S[n] = np.load(str(HomPath / (ROI + '.npy')))
    Stiffness = 0.5 * (Stiffness + Stiffness.T)

    # Transform stiffness tensor into fabric coordinate system
    MS = TA.Engineering2MandelNotation(Stiffness)
    MS4 = TA.IsoMorphism66_3333(MS)

    I = np.eye(3)
    Q = np.array(eVectors[n].T)

    TS4 = TA.TransformTensor(MS4, I, Q)
    TS2 = TA.IsoMorphism3333_66(TS4)

    # Project onto orthotropy
    OrthoS = np.zeros(TS2.shape)
    for i in range(OrthoS.shape[0]):
        for j in range(OrthoS.shape[1]):
            if i < 3 and j < 3:
                OrthoS[i, j] = TS2[i, j]
            elif i == j:
                OrthoS[i, j] = TS2[i, j]

    MOrthoS = TA.Mandel2EngineeringNotation(OrthoS)

    # Store into array
    S[n] = MOrthoS

# Figure, Axis = plt.subplots(1,1)
# Axis.scatter(OldBVTV, BVTV, color=(1,0,0), facecolor=(0,0,0,0))
# Axis.plot([min(OldBVTV.min(), BVTV.min()), max(OldBVTV.max(), BVTV.max())],
#           [min(OldBVTV.min(), BVTV.min()), max(OldBVTV.max(), BVTV.max())],
#           color=(0,0,0), linestyle = '--')
# Axis.set_xlabel('Publication')
# Axis.set_ylabel('Actual')
# plt.show(Figure)

Figure, Axis = plt.subplots(1,1)
Axis.scatter(Oldm1, m1, color=(1,0,0), facecolor=(0,0,0,0))
Axis.scatter(Oldm2, m2, color=(0,0,1), facecolor=(0,0,0,0))
Axis.scatter(Oldm3, m3, color=(0,1,0), facecolor=(0,0,0,0))
Min = min(Oldm1.min(), Oldm2.min(), Oldm3.min(), m1.min(), m2.min(), m3.min())
Max = max(Oldm1.max(), Oldm2.max(), Oldm3.max(), m1.max(), m2.max(), m3.max())
Axis.plot([Min, Max], [Min, Max], color=(0,0,0), linestyle = '--')
Axis.set_xlabel('Medtool')
Axis.set_ylabel('Custom')
plt.show(Figure)

DA = m3 / m1
OldDA = Oldm3 / Oldm1
Figure, Axis = plt.subplots(1,1)
Axis.scatter(OldDA, DA, color=(1,0,0), facecolor=(0,0,0,0))
Axis.plot([min(OldDA.min(), DA.min()), max(OldDA.max(), DA.max())], [min(OldDA.min(), DA.min()), max(OldDA.max(), DA.max())], color=(0,0,0), linestyle = '--')
Axis.set_xlabel('Medtool')
Axis.set_ylabel('Custom')
plt.show(Figure)

Figure, Axis = plt.subplots(1,1)
for i in range(3):
    Axis.scatter(OldS[:,i,i], S[:,i,i], color=(1,0,0), facecolor=(0,0,0,0))
    for j in range(3):
        if j != i:
            Axis.scatter(OldS[:,i,j], S[:,i,j], color=(0,1,0), facecolor=(0,0,0,0))
            Axis.scatter(OldS[:,j,i], S[:,j,i], color=(0,1,0), facecolor=(0,0,0,0))
    Axis.scatter(OldS[:,i+3,i+3], S[:,i+3,i+3], color=(0,0,1), facecolor=(0,0,0,0))
Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(1,0,0), label=r'$\lambda_{ii}$')
Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(0,1,0), label=r'$\lambda_{ij}$')
Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(0,0,1), label=r'$\mu_{ij}$')
Min = min(OldS.min(), S.min())
Max = max(OldS.max(), S.max())
Axis.plot([Min, Max], [Min, Max], color=(0,0,0), linestyle = '--')

N = len(Data)
E = OldS.ravel() - S.ravel()
RSS = np.sum(E ** 2)
SE = np.sqrt(RSS / (len(E) - 1))
TSS = np.sum((OldS - OldS.mean()) ** 2)
RegSS = TSS - RSS
R2 = RegSS / TSS
Axis.annotate(r'$R^2_{ajd}$: ' + format(round(R2, 3), '.3f'), xy=(0.75, 0.1), xycoords='axes fraction')

NE = np.array([])
for i in range(0,N,12):
    ObservedTensor = S[i:i+12]
    PredictedTensor = OldS[i:i+12]
    Numerator = np.sum((ObservedTensor-PredictedTensor)**2)
    Denominator = np.sum(ObservedTensor**2)
    NE = np.append(NE,np.sqrt(Numerator/Denominator))

Axis.annotate(r'$NE$ : ' + format(round(NE.mean(), 2), '.2f') + r'$\pm$' + format(round(NE.std(), 2), '.2f'),
                xy=(0.75, 0.025), xycoords='axes fraction')

Y = np.matrix(S.ravel()).T
X = np.ones((len(Y),2))
X[:,1] = OldS.ravel()
X = np.matrix(X)
Slope, Intercept = np.linalg.inv(X.T * X) * X.T * Y
Axis.annotate(r'$Slope$ : ' + format(round(Slope[0,0], 2), '.2f'),
                xy=(0.5, 0.1), xycoords='axes fraction')
Axis.annotate(r'$Intercept$ : ' + format(round(Intercept[0,0], 2), '.2f'),
                xy=(0.5, 0.025), xycoords='axes fraction')

Axis.set_xlabel('Medtool Abaqus (MPa)')
Axis.set_ylabel('Python FEniCS (MPa)')
Axis.set_xscale('log')
Axis.set_yscale('log')
plt.legend(loc='upper left')
# plt.savefig('Test.png')
plt.show(Figure)


# Build system for linear regression
X = np.matrix(np.zeros((12*len(Data), 5)))
Y = np.matrix(np.zeros((12*len(Data), 1)))
Ones = np.ones(len(Data), int)
Zeros = np.zeros(len(Data), int)

X[0::12]  = np.array([Ones, Zeros, Zeros, np.log(BVTV), np.log(m1 ** 2)]).T
X[1::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m1 * m2)]).T
X[2::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m1 * m3)]).T
X[3::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m2 * m1)]).T
X[4::12]  = np.array([Ones, Zeros, Zeros, np.log(BVTV), np.log(m2 ** 2)]).T
X[5::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m2 * m3)]).T
X[6::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m3 * m1)]).T
X[7::12]  = np.array([Zeros, Ones, Zeros, np.log(BVTV), np.log(m3 * m2)]).T
X[8::12]  = np.array([Ones, Zeros, Zeros, np.log(BVTV), np.log(m3 ** 2)]).T
X[9::12]  = np.array([Zeros, Zeros, Ones, np.log(BVTV), np.log(m2 * m3)]).T
X[10::12] = np.array([Zeros, Zeros, Ones, np.log(BVTV), np.log(m3 * m1)]).T
X[11::12] = np.array([Zeros, Zeros, Ones, np.log(BVTV), np.log(m1 * m2)]).T

Y[0::12]  = np.log(S[:,0,0]).reshape((len(Data),1))
Y[1::12]  = np.log(S[:,0,1]).reshape((len(Data),1))
Y[2::12]  = np.log(S[:,0,2]).reshape((len(Data),1))
Y[3::12]  = np.log(S[:,1,0]).reshape((len(Data),1))
Y[4::12]  = np.log(S[:,1,1]).reshape((len(Data),1))
Y[5::12]  = np.log(S[:,1,2]).reshape((len(Data),1))
Y[6::12]  = np.log(S[:,2,0]).reshape((len(Data),1))
Y[7::12]  = np.log(S[:,2,1]).reshape((len(Data),1))
Y[8::12]  = np.log(S[:,2,2]).reshape((len(Data),1))
Y[9::12]  = np.log(S[:,1,2]).reshape((len(Data),1))
Y[10::12] = np.log(S[:,2,0]).reshape((len(Data),1))
Y[11::12] = np.log(S[:,0,1]).reshape((len(Data),1))

Parameters = np.linalg.inv(X.T * X) * X.T * Y
l = Parameters[4][0,0]
k = Parameters[3][0,0]
Mu0 = np.exp(Parameters[2])[0,0]
Lambda0p = np.exp(Parameters[1])[0,0]
Lambda0 = np.exp(Parameters[0])[0,0] - 2*Mu0

# Plot fit results
SFit = np.array(X * Parameters).ravel()
SObs = np.array(Y)[:,0]
Line = [min(SFit.min(), SObs.min()), max(SFit.max(), SObs.max())]

N = len(Data)
E = SObs - SFit
RSS = np.sum(E ** 2)
SE = np.sqrt(RSS / (len(E) - len(Parameters)))
TSS = np.sum((SObs - SObs.mean()) ** 2)
RegSS = TSS - RSS
R2 = RegSS / TSS

## Compute R2 adj and NE
R2adj = 1 - RSS/TSS * (12*N-1)/(12*N-len(Parameters)-1)

NE = np.array([])
for i in range(0,N,12):
    ObservedTensor = SObs[i:i+12]
    PredictedTensor = SFit[i:i+12]

    Numerator = np.sum((ObservedTensor-PredictedTensor)**2)
    Denominator = np.sum(ObservedTensor**2)

    NE = np.append(NE,np.sqrt(Numerator/Denominator))

Figure, Axis = plt.subplots(1,1)
Axis.plot(np.exp(SObs[::4]), np.exp(SFit[::4]), linestyle='none', marker='s', fillstyle='none', color=(0,0,1))
for i in [1,2,3,5,6,7]:
    Axis.plot(np.exp(SObs[i::12]), np.exp(SFit[i::12]), linestyle='none', marker='o', fillstyle='none', color=(0,1,0))
for i in [9,10,11]:
    Axis.plot(np.exp(SObs[i::12]), np.exp(SFit[i::12]), linestyle='none', marker='^', fillstyle='none', color=(1,0,0))
Axis.plot(np.exp([Line[0],Line[1]]), np.exp([Line[0],Line[1]]), color=(0,0,0),linestyle='--', linewidth=1)
Axis.annotate(r'N ROIs   : ' + str(N), xy=(0.5, 0.1), xycoords='axes fraction')
Axis.annotate(r'N Points : ' + str(len(SObs)), xy=(0.5, 0.025), xycoords='axes fraction')
Axis.annotate(r'$R^2_{ajd}$: ' + format(round(R2adj, 3), '.3f'), xy=(0.75, 0.1), xycoords='axes fraction')
Axis.annotate(r'$NE$ : ' + format(round(NE.mean(), 2), '.2f') + r'$\pm$' + format(round(NE.std(), 2), '.2f'),
                xy=(0.75, 0.025), xycoords='axes fraction')
Axis.set_xlabel(r'Observed $\mathrm{\mathbb{S}}$ (MPa)')
Axis.set_ylabel(r'Fitted $\mathrm{\mathbb{S}}$ (MPa)')
Axis.set_xscale('log')
Axis.set_yscale('log')
Axis.set_xlim([10**0, 10**4])
Axis.set_ylim([10**0, 10**4])
Axis.plot([], linestyle='none', marker='s', fillstyle='none', color=(0,0,1), label=r'$\lambda_{ii}$')
Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(0,1,0), label=r'$\lambda_{ij}$')
Axis.plot([], linestyle='none', marker='^', fillstyle='none', color=(1,0,0), label=r'$\mu_{ij}$')
plt.legend(loc='upper left')
plt.savefig('Test.png')
plt.show(Figure)




#%%

#%% #!/usr/bin/env python

"""
Compute fabric tensor by MIL.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import numpy as np

#%% Functions

def MIL(Array, Directions, NRays=50):

    # Define rays to count interfaces in unit cube
    NCoords = Array.shape[0]
    Rays = np.zeros((NRays**2, NCoords, 3))
    Zc = np.linspace(-0.5,0.5,NCoords) * np.sqrt(3)
    Yc = np.linspace(-0.5,0.5,NRays) * np.sqrt(3)
    Xc = np.linspace(-0.5,0.5,NRays) * np.sqrt(3)
    Yc += (Yc[1] - Yc[0]) / 2
    Xc += (Xc[1] - Xc[0]) / 2
    for xRay in range(NRays):
        for yRay in range(NRays):
            Ray = yRay + xRay * NRays
            Rays[Ray,:,2] = Zc
            Rays[Ray,:,1] = Yc[yRay]
            Rays[Ray,:,0] = Xc[xRay]

    # Rotate rays in given directions
    Coords = np.zeros((len(Directions),) + Rays.shape)
    Ref = np.array([0, 0, 1])
    for iD, Dir in enumerate(Directions):
        R = RotateFromVectors(Ref,Dir)
        Coords[iD] = np.einsum('ij, lkj -> lki', R, Rays)

    # Center and scale coordinates
    Coords = (Coords + 0.5) * np.array(Array.shape)

    # Mask rays within cube
    MinMask = Coords >= 0
    MaxMask = Coords < Array.shape[0]
    Mask = np.sum((MinMask & MaxMask)*1, axis=-1) == 3
    Lengths = np.sum(Mask, axis=-1)

    # Count intercepts
    pArray = np.pad(Array,1)
    rCoords = np.round(Coords).astype(int)
    NDirs, NRays = Coords.shape[:2]
    I = np.zeros(NDirs)
    for Dir in range(NDirs):
        nI = []
        for nRay in range(NRays):
            if Lengths[Dir,nRay] > 0:
                Ray = rCoords[Dir,nRay][Mask[Dir,nRay]]
                Voxels = pArray[Ray[:,0], Ray[:,1], Ray[:,2]]
                Voxels[1:] = np.maximum(Voxels[1:], Voxels[:-1])
                Voxels[:-1] = np.maximum(Voxels[1:], Voxels[:-1])
                Interfaces = Voxels[1:] - Voxels[:-1]
                nI.append(np.sum(Interfaces > 0))
        I[Dir] = np.sum(nI)

    # Fit MIL
    MIL = 1 / I

    return MIL

def MSL(Vectors, Normals, Areas, Volume):

    Sum = 0
    for i in range(len(Areas)):

        Dot1 = np.einsum('i,ji->j', Normals[i],Vectors)
        Outer = np.einsum('i,j', Normals[i], Dot1)
        Dot2 = np.einsum('ji,ij->j', Vectors, Outer)
        Sum += Areas[i] * np.sqrt(Dot2)

    return 2*Volume / Sum

def RotationMatrix(Phi=0, Theta=0, Psi=0):

    Rx = np.array([[1,            0,            0],
                   [0,  np.cos(Phi), -np.sin(Phi)],
                   [0,  np.sin(Phi),  np.cos(Phi)]]).T

    Ry = np.array([[ np.cos(Theta),   0, np.sin(Theta)],
                   [             0,   1,             0],
                   [-np.sin(Theta),   0, np.cos(Theta)]]).T

    Rz = np.array([[np.cos(Psi), -np.sin(Psi), 0],
                   [np.sin(Psi),  np.cos(Psi), 0],
                   [          0,            0, 1]]).T

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def FitFabric(MILValues, Directions):

    """ 
    Compute the fabric tensors using an ellipsoidal fit        
    """

    nDir = len(Directions)
    nHat = np.array(np.zeros((nDir, 6), float))
    An = np.array(np.zeros(nDir, float))
    H = np.array(np.zeros((3, 3), float))
    d = 0
    for i, n in enumerate(Directions):
        nHat[d, 0] = n[0] * n[0]
        nHat[d, 1] = n[1] * n[1]
        nHat[d, 2] = n[2] * n[2]
        nHat[d, 3] = np.sqrt(2.0) * n[1] * n[2]
        nHat[d, 4] = np.sqrt(2.0) * n[2] * n[0]
        nHat[d, 5] = np.sqrt(2.0) * n[0] * n[1]
        MILn = np.array(MILValues[i], dtype=float)
        An[d] = 1.0 / MILn * (1.0 / MILn)
        d += 1

    N1 = np.dot(np.transpose(nHat), nHat)
    N2 = np.dot(np.transpose(nHat), An)
    VM = np.dot(np.linalg.inv(N1), N2)
    H[(0, 0)] = VM[0]
    H[(1, 1)] = VM[1]
    H[(2, 2)] = VM[2]
    H[(1, 2)] = VM[3] / np.sqrt(2.0)
    H[(2, 0)] = VM[4] / np.sqrt(2.0)
    H[(0, 1)] = VM[5] / np.sqrt(2.0)
    H[(2, 1)] = VM[3] / np.sqrt(2.0)
    H[(0, 2)] = VM[4] / np.sqrt(2.0)
    H[(1, 0)] = VM[5] / np.sqrt(2.0)

    return H

def Fabric(m1,m2,m3,Alpha,Beta,Gamma):

    """
    Express the fabric tensor based on eigen values m1, m2, and m3
    In the coordinate system rotated by Alpha, Beta, and Gamma
    """

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    R = RotationMatrix(Alpha,Beta,Gamma)

    Dot = np.dot(np.outer(e1,e1), R.T)
    M1 = m1 * np.dot(R, Dot)

    Dot = np.dot(np.outer(e2,e2), R.T)
    M2 = m2 * np.dot(R, Dot)

    Dot = np.dot(np.outer(e3,e3), R.T)
    M3 = m3 * np.dot(R, Dot)

    return M1 + M2 + M3

def FibonacciSphere(N):

    """
    Generate a sphere with (almost) equidistant points
    https://arxiv.org/pdf/0912.4540.pdf
    """

    i = np.arange(N)
    Phi = np.pi * (np.sqrt(5) - 1)  # golden angle in radians

    Z = 1 - (i / float(N - 1)) * 2  # z goes from 1 to -1
    Radius = np.sqrt(1 - Z*Z)     # radius at z

    Theta = Phi * i                 # golden angle increment

    X = np.cos(Theta) * Radius
    Y = np.sin(Theta) * Radius

    return np.vstack([X,Y,Z]).T

def AreasAndNormals(Array):

    # Primary normals (faces)
    NW = np.array([-1,  0,  0])
    NE = np.array([ 1,  0,  0])
    NS = np.array([ 0, -1,  0])
    NN = np.array([ 0,  1,  0])
    NB = np.array([ 0,  0, -1])
    NT = np.array([ 0,  0,  1])

    # Secondary normals (edges)
    NWS = np.array([-1, -1,  0])
    NWN = np.array([-1,  1,  0])
    NWB = np.array([-1,  0, -1])
    NWT = np.array([-1,  0,  1])

    NEN = np.array([ 1,  1,  0])
    NES = np.array([ 1, -1,  0])
    NET = np.array([ 1,  0,  1])
    NEB = np.array([ 1,  0, -1])

    # NSW = np.array([ -1, -1,  0])      # Same as NWS
    # NSE = np.array([  1, -1,  0])      # Same as NES
    NSB = np.array([ 0, -1, -1])
    NST = np.array([ 0, -1,  1])

    # NNE = np.array([ 1,  1,  0])       # Same as NEN
    # NNW = np.array([-1,  1,  0])       # Same as NWN
    NNT = np.array([ 0,  1,  1])
    NNB = np.array([ 0,  1, -1])

    # Tertiary normals (corners)
    NWSB = np.array([-1, -1, -1])
    NWST = np.array([-1, -1,  1])
    NWNB = np.array([-1,  1, -1])
    NWNT = np.array([-1,  1,  1])
    NENT = np.array([ 1,  1,  1])
    NENB = np.array([ 1,  1, -1])
    NEST = np.array([ 1, -1,  1])
    NESB = np.array([ 1, -1, -1])


    # Compute free surfaces
    PadArray = np.pad(Array,1)

    # Primary normals (faces)
    AW = Array - PadArray[  2:, 1:-1, 1:-1]
    AE = Array - PadArray[ :-2, 1:-1, 1:-1]
    AS = Array - PadArray[1:-1,   2:, 1:-1]
    AN = Array - PadArray[1:-1,  :-2, 1:-1]
    AB = Array - PadArray[1:-1, 1:-1:,  2:]
    AT = Array - PadArray[1:-1, 1:-1:, :-2]

    # Secondary normals (edges)
    AWS = Array - PadArray[  2:,   2:, 1:-1]
    AWN = Array - PadArray[  2:,  :-2, 1:-1]
    AWB = Array - PadArray[  2:, 1:-1,   2:]
    AWT = Array - PadArray[  2:, 1:-1,  :-2]

    AES = Array - PadArray[ :-2,   2:, 1:-1]
    AEN = Array - PadArray[ :-2,  :-2, 1:-1]
    AEB = Array - PadArray[ :-2, 1:-1,   2:]
    AET = Array - PadArray[ :-2, 1:-1,  :-2]

    # ASW = Array - PadArray[  2:,   2:, -1:1]      # Same as AWS
    # ASE = Array - PadArray[ :-2,   2:, -1:1]      # Same as AES
    ASB = Array - PadArray[1:-1,   2:,   2:]
    AST = Array - PadArray[1:-1,   2:,  :-2]

    # ANW = Array - PadArray[  2:,  :-2, -1:1]      # Same as AWN
    # ANE = Array - PadArray[ :-2,  :-2, -1:1]      # Same as AEN
    ANB = Array - PadArray[1:-1,   :-2,   2:]
    ANT = Array - PadArray[1:-1,   :-2,  :-2]

    # Tertiary normals (corners)
    AWSB = Array - PadArray[  2:,   2:,   2:]
    AWST = Array - PadArray[  2:,   2:,  :-2]
    AWNB = Array - PadArray[  2:,  :-2,   2:]
    AWNT = Array - PadArray[  2:,  :-2,  :-2]
    AESB = Array - PadArray[ :-2,   2:,   2:]
    AEST = Array - PadArray[ :-2,   2:,  :-2]
    AENB = Array - PadArray[ :-2,  :-2,   2:]
    AENT = Array - PadArray[ :-2,  :-2,  :-2]

    # Group areas and normals
    AreaList = [AW, AE, AS, AN, AB, AT,
                AWS, AWN, AWB, AWT, AES, AEN, AEB, AET, ASB, AST, ANB, ANT,
                AWSB, AWST, AWNB, AWNT, AESB, AEST, AENB, AENT]
    
    Areas = [np.sum(A==1) for A in AreaList]
    
    Normals = [NW, NE, NS, NN, NB, NT,
               NWS, NWN, NWB, NWT, NES, NEN, NEB, NET, NSB, NST, NNB, NNT,
               NWSB, NWST, NWNB, NWNT, NESB, NEST, NENB, NENT]

    return Areas, Normals

def RotateFromVectors(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    https://gist.github.com/aormorningstar/3e5dda91f155d7919ef6256cb057ceee
    """

    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)

    # dimension of the space and identity
    dim = u.size
    I = np.identity(dim)

    # the cos angle between the vectors

    c = np.dot(u, Ru)

    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)

def MILEigenValuesAndVectors(Array,NDir=128,NRays=50):

    Directions = FibonacciSphere(NDir)
    MILValues = MIL(Array, Directions, NRays)
    
    # Fit fabric tensor
    H = FitFabric(MILValues, Directions)
    eValues, eVectors = np.linalg.eig(H)
    eValues = 1 / np.sqrt(eValues)

    # Normalize values and sort
    eValues = 3 * eValues / sum(eValues)
    eVectors = eVectors.T[np.argsort(eValues)]
    eValues = np.sort(eValues)

    return eValues, eVectors

def MSLEigenValuesAndVectors(Array,N=128):

    # Compute areas and normals
    Areas, Normals = AreasAndNormals(Array)

    # Compute MIL
    Directions = FibonacciSphere(N)
    Volume = Array.sum()
    MSLValues = MSL(Directions,Normals,Areas,Volume)
    
    # Fit fabric tensor
    H = FitFabric(MSLValues, Directions)
    eValues, eVectors = np.linalg.eig(H)
    # eValues = 1 / np.sqrt(eValues)

    # Normalize values and sort
    eValues = 3 * eValues / sum(eValues)
    eVectors = eVectors[np.argsort(eValues)]
    eValues = np.sort(eValues)

    return eValues, eVectors

#%%

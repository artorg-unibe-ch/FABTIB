#%% !/usr/bin/env python3

"""
Script intended to reproduce MIL from mathematica script
of Philippe Zysset
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '26-04-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import numpy as np
from skimage import measure
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import art3d
from skimage.morphology import binary_dilation

#%% Tensor algebra
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Functions

def MSL(Vectors, Normals, Areas, Volume):

    Sum = 0
    for i in range(len(Areas)):

        Dot1 = np.einsum('i,ji->j', Normals[i],Vectors)
        Outer = np.einsum('i,j', Normals[i], Dot1)
        Dot2 = np.einsum('ji,ij->j', Vectors, Outer)
        Sum += Areas[i] * np.sqrt(Dot2)

    return 2*Volume / Sum

def RotateFromVector(Vector, Angle):

    U = np.outer(Vector, Vector)
    Ux = np.array([[         0, -Vector[2],  Vector[1]],
                   [ Vector[2],          0, -Vector[0]],
                   [-Vector[1],  Vector[0],         0]])
    I = np.eye(3)

    R = np.cos(Angle) * I + np.sin(Angle) * Ux + (1-np.cos(Angle)) * U

    return R

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

def MIL(Array, NDir=128, NRays=50):

    # Define rays to count interfaces in unit cube
    NCoords = 100
    NRays = 50
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
    Directions = FibonacciSphere(NDir)
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

def RotationMatrices(Phi=np.zeros(1), Theta=np.zeros(1), Psi=np.zeros(1)):
    
    
    Ones = np.ones(len(Phi))
    Zeros = np.zeros(len(Phi))

    Rx = np.array([[ Ones,        Zeros,        Zeros],
                   [Zeros,  np.cos(Phi), -np.sin(Phi)],
                   [Zeros,  np.sin(Phi),  np.cos(Phi)]]).T

    Ry = np.array([[ np.cos(Theta), Zeros, np.sin(Theta)],
                   [        Zeros,   Ones,         Zeros],
                   [-np.sin(Theta), Zeros, np.cos(Theta)]]).T

    Rz = np.array([[np.cos(Psi), -np.sin(Psi), Zeros],
                   [np.sin(Psi),  np.cos(Psi), Zeros],
                   [      Zeros,        Zeros,  Ones]]).T

    RR = np.einsum('ijk,ikl->ijl', Ry, Rx)
    R = np.einsum('ijk,ikl->ijl', Rz, RR)

    return R

def FitFabric(MILValues, Directions):

    """ 
    Compute the fabric tensors using an ellipsoidal fit
    
        :param MILValues: Values of the MIL distribution
               Directions: Corresponding directions          
                    
        :return: M: fabric tensor from ellipsoidal fit 
                - TYPE: float numpy.array[3,3]            
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

    R = RotationMatrices([Alpha],[Beta],[Gamma])[0]

    Dot = np.dot(np.outer(e1,e1), R.T)
    M1 = m1 * np.dot(R, Dot)

    Dot = np.dot(np.outer(e2,e2), R.T)
    M2 = m2 * np.dot(R, Dot)

    Dot = np.dot(np.outer(e3,e3), R.T)
    M3 = m3 * np.dot(R, Dot)

    return M1 + M2 + M3

def FibonacciSphere(N=128):

    """
    https://arxiv.org/pdf/0912.4540.pdf
    """

    i = np.arange(N)
    phi = np.pi * (np.sqrt(5) - 1)  # golden angle in radians

    z = 1 - (i / float(N - 1)) * 2  # z goes from 1 to -1
    radius = np.sqrt(1 - z * z)     # radius at z

    theta = phi * i  # golden angle increment

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    return np.vstack([x,y,z]).T

def MSLEigenValuesAndVectors(Array,N=128):

    # Compute areas and normals
    Vertices, Faces, Normals, Values = measure.marching_cubes(Array)
    Triangles = Vertices[Faces]
    V1 = Triangles[:,1] - Triangles[:,0]
    V2 = Triangles[:,2] - Triangles[:,0]
    Normals = np.cross(V1, V2)
    Areas = 0.5 * np.linalg.norm(Normals, axis=-1)
    Normals = Normals / np.expand_dims(np.linalg.norm(Normals, axis=-1),-1)

    # Compute MSL
    Directions = FibonacciSphere(N)
    Volume = Array.sum()
    MSLValues = MSL(Directions,Normals,Areas,Volume)
    
    # Fit fabric tensor
    H = FitFabric(MSLValues, Directions)
    eValues, eVectors = np.linalg.eig(H)
    eValues = 1 / np.sqrt(eValues)

    # Normalize values and sort
    eValues = 3 * eValues / sum(eValues)
    eVectors = eVectors[np.argsort(eValues)]
    eValues = np.sort(eValues)

    return eValues, eVectors

def AreasAndNormals(Array):

    N1 = np.array([ 1, 0, 0])
    N2 = np.array([-1, 0, 0])
    N3 = np.array([ 0, 1, 0])
    N4 = np.array([ 0,-1, 0])
    N5 = np.array([ 0, 0, 1])
    N6 = np.array([ 0, 0,-1])

    PadArray = np.pad(Array,1)
    A1 = PadArray[ 2:,:,:] - PadArray[1:-1,:,:]
    A2 = PadArray[:-2,:,:] - PadArray[1:-1,:,:]
    A3 = PadArray[:, 2:,:] - PadArray[:,1:-1,:]
    A4 = PadArray[:,:-2,:] - PadArray[:,1:-1,:]
    A5 = PadArray[:,:, 2:] - PadArray[:,:,1:-1]
    A6 = PadArray[:,:,:-2] - PadArray[:,:,1:-1]

    Areas = [np.sum(A==1) for A in [A1,A2,A3,A4,A5,A6]]
    Normals = [N1,N2,N3,N4,N5,N6]

    return Areas, Normals

#%% Example of a hexahedral bar

def Main():
    nFaces = 6
    Normals = [e3, -e3, e1, -e1, e2, -e2]
    Areas = [1, 1, 5, 5, 5, 5]
    Volume = 5
    Area = sum([Areas[i] for i in range(nFaces)])

    Vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,5], [1,0,5], [1,1,5], [0,1,5]])
    Faces = [[0,1,2,3],[0,1,5,4],[1,2,6,5],[2,3,7,6],[0,3,7,4],[4,5,6,7]]

    Figure, Axis = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    PC = art3d.Poly3DCollection(Vertices[Faces], facecolors=(1,0,0,0.5), edgecolor=(0,0,0))
    Axis.add_collection(PC)
    Axis.auto_scale_xyz([-2, 3], [-2, 3], [0, 5])
    plt.show(Figure)

    Points = FibonacciSphere(1024)
    Figure, Axis = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    Axis.scatter(Points[:,0],Points[:,1],Points[:,2], facecolor=(1,0,0))
    Axis.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    plt.show(Figure)

    Mil= MIL(Points, Normals, Areas, nFaces)
    MilPoints = Points * np.expand_dims(Mil,-1)

    Figure, Axis = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    Axis.scatter(MilPoints[:,0],MilPoints[:,1],MilPoints[:,2], facecolor=(1,0,0))
    # Axis.auto_scale_xyz([-5, 5], [-5, 5], [-5, 5])
    plt.show(Figure)


    def Fit(ms):

        m1, m2, m3 = ms

        MILPoints = MIL(Points, Normals, Areas, nFaces)
        Dot = np.einsum('ij,kj->kj',Fabric(1/m1**2, 1/m2**2, 1/m3**2, 0, 0, 0), Points)
        Sum2 = np.einsum('ij,ij->i',Points, Dot)
        
        return np.sum((MILPoints - 1/np.sqrt(Sum2))**2)

    Bounds = [(1E-3, 10), (1E-3, 10), (1E-3, 10)]
    Results = minimize(Fit, np.array([1,1,1]), bounds=Bounds)['x']

    MILValues = MIL(Points, Normals, Areas, nFaces)
    H = FitFabric(MILValues, Points)
    eValues, eVectors = np.linalg.eig(H)
    1 / eValues
    eVectors[np.argsort(eValues)]

    Array = np.zeros((10,10,10))
    Array[5,5,2:8] = 1

    Areas, Normals = AreasAndNormals(Array)
    Volume = Array.sum()
    Points = FibonacciSphere(N=128)
    MILValues = MIL(Points,Normals,Areas,Volume)
    H = FitFabric(MILValues, Points)
    eValues, eVectors = np.linalg.eig(H)
    1 / eValues
    max(eValues) / min(eValues)

    import FabricAnalysis as FA
    FA.MeanInterceptLength(Array)


    # MIL according to Launeau and Robin (https://doi.org/10.1016/S0040-1951(96)00091-1)
    Array = np.zeros((500,500))
    R1 = 250
    R2 = 125
    Theta = 2*np.pi/3
    R = np.matrix([[np.cos(Theta), -np.sin(Theta)],[np.sin(Theta), np.cos(Theta)]])
    X = np.arange(-Array.shape[0]/2, Array.shape[0]/2, 0.5)
    Y = np.arange(-Array.shape[1]/2, Array.shape[1]/2, 0.5)
    for x in X:
        if abs(x) <= R1:
            for y in Y:
                if np.abs(y) <= np.abs(R2/R1 * np.sqrt(R1**2 - x**2)):
                    xy = np.round(R * np.array([[x],[y]]))
                    xc = int(xy[0,0]) + Array.shape[0] // 2
                    yc = int(xy[1,0]) + Array.shape[1] // 2
                    Array[xc,yc] = 1

    # Define rays to count interfaces in unit cube
    NCoords = 500
    NRays = 500
    Rays = np.zeros((NRays, NCoords, 2))
    Xc = np.linspace(-0.5,0.5,NCoords) * np.sqrt(2)
    Yc = np.linspace(-0.5,0.5,NRays+1) * np.sqrt(2)
    Yc += (Yc[1] - Yc[0]) / 2
    for Ray in range(NRays):
        Rays[Ray,:,0] = Xc
        Rays[Ray,:,1] = Yc[Ray]

    # Rotate rays in given directions
    NRot = 180
    Angles = np.arange(0,np.pi,np.pi/NRot)
    Coords = np.zeros((NRot, NRays, NCoords, 2))
    Dir = np.zeros((NRot,2))
    Dir[:,1] = 1
    for NAngle, Angle in enumerate(Angles):
        R = np.matrix([[np.cos(Angle), -np.sin(Angle)],[np.sin(Angle), np.cos(Angle)]])
        Coords[NAngle] = np.einsum('ij, lkj -> lki', R, Rays)
        Dir[NAngle] = Dir[NAngle] * R

    # Center and scale coordinates
    Coords = (Coords + 0.5) * np.array(Array.shape)
    Xc = Coords[...,0]
    Yc = Coords[...,1]

    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Array, cmap='binary_r')
    # for Rot in range(NRot):
    #     for Ray in range(NRays//10):
    #         Axis.plot(Yc[Rot,Ray*10], Xc[Rot,Ray*10], color=(1,0,0))
    #         Axis.axis('off')
    # plt.show()

    # Mask rays within cube
    MinMask = Coords >= 0
    MaxMask = Coords < Array.shape[0]
    Mask = np.sum((MinMask & MaxMask)*1, axis=-1) == 2
    Lengths = np.sum(Mask, axis=-1)

    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Array, cmap='binary_r')
    # # for Rot in range(NRot):
    # for Ray in range(NRays//10):
    #     Axis.plot(Yc[Rot,Ray*10][Mask[Rot,Ray*10]], Xc[Rot,Ray*10][Mask[Rot,Ray*10]], color=(1,0,0))
    #     Axis.axis('off')
    # plt.show()

    # Get MIL
    I = np.zeros(NRot)
    for Rot in range(NRot):
        nI = []
        for nRay in range(NRays):
            if Lengths[Rot,nRay] > 0:
                Ray = np.round(Coords[Rot,nRay][Mask[Rot,nRay]]).astype(int)
                Voxels = np.pad(Array,1)[Ray[:,0], Ray[:,1]]
                bVoxels =  binary_dilation(Voxels) * 1
                Interfaces = bVoxels[1:] - bVoxels[:-1]
                nI.append(np.sum(Interfaces > 0))
        I[Rot] = np.sum(nI)

    # Plot rose of intercepts
    I_Plot = I / np.max(I) * max(Array.shape) / 2
    I_Plot = Dir*np.expand_dims(I_Plot,-1)

    Figure, Axis = plt.subplots(1,1)
    plt.imshow(Array, cmap='binary_r')
    Axis.plot(I_Plot[:,0] + Array.shape[0]/2, I_Plot[:,1] + Array.shape[1]/2, color=(1,0,0))
    Axis.plot(-I_Plot[:,0] + Array.shape[0]/2, -I_Plot[:,1] + Array.shape[1]/2, color=(1,0,0))
    plt.show()

    # Plot Mean Intercept Length rose
    MIL = 1 / I
    MIL_Plot = MIL / np.max(MIL) * max(Array.shape) / 2
    MIL_Plot = Dir*np.expand_dims(MIL_Plot,-1)

    Figure, Axis = plt.subplots(1,1)
    plt.imshow(Array, cmap='binary_r')
    Axis.plot(MIL_Plot[:,0] + Array.shape[0]/2, MIL_Plot[:,1] + Array.shape[1]/2, color=(1,0,0))
    Axis.plot(-MIL_Plot[:,0] + Array.shape[0]/2, -MIL_Plot[:,1] + Array.shape[1]/2, color=(1,0,0))
    plt.show()

    # 3D MIL
    Array = np.zeros((100,100,100))
    Rx = 10
    Ry = 20
    Rz = 40
    Theta = 2*np.pi/3
    R = RotateFromVector(np.array([0,1,0]), Theta)
    X = np.arange(-Array.shape[0]/2, Array.shape[0]/2, 0.5)
    Y = np.arange(-Array.shape[1]/2, Array.shape[1]/2, 0.5)
    Z = np.arange(-Array.shape[2]/2, Array.shape[2]/2, 0.5)
    for z in Z[np.abs(Z) <= Rz]:
        for y in Y[np.abs(Y) <= Ry]:
            for x in X[np.abs(X) <= Rx]:
                if abs(x) <= np.sqrt(Rx**2 * abs(1 - y**2/Ry**2 - z**2/Rz**2)):
                    if np.abs(y) <= Ry/Rx * np.sqrt(Rx**2 - x**2):
                        if np.abs(z) <= Rz/Ry * np.sqrt(Ry**2 - y**2):
                            xyz = np.round(R * np.matrix([[x],[y],[z]]))
                            # xyz = np.array([[x],[y],[z]])
                            xc = int(xyz[0,0]) + Array.shape[2] // 2
                            yc = int(xyz[1,0]) + Array.shape[1] // 2
                            zc = int(xyz[2,0]) + Array.shape[0] // 2
                            Array[xc,yc,zc] = 1

    Figure, Axis = plt.subplots(2,2,figsize=(5.5,5.5))
    Axis[0,0].imshow(Array[:,:,Array.shape[2]//2], cmap='binary')
    Axis[0,1].imshow(Array[:,Array.shape[1]//2,:], cmap='binary')
    Axis[1,0].imshow(Array[Array.shape[0]//2,:,:].T, cmap='binary')
    for i in range(2):
        for j in range(2):
            Axis[i,j].axis('off')
    Axis = Figure.add_subplot(2, 2, 4, projection='3d')
    Axis.voxels(Array[:,:,::-1].T, color=(0,0,0,0.1))
    Axis.set_xticks([])
    Axis.set_yticks([])
    Axis.set_zticks([])
    plt.show(Figure)

    # Define rays to count interfaces in unit cube
    NCoords = 100
    NRays = 50
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
    NDir = 144
    # Alphas = np.arange(0, np.pi, np.pi/NDir)
    # Betas = np.arange(0, np.pi, np.pi/18)
    # Angles = np.array(list(product(Alphas, Betas)))
    # Rs = RotationMatrices(Angles[:,0], Angles[:,1], np.zeros(len(Angles)))
    # Coords = np.einsum('ijk, mlk -> imlj', Rs, Rays)
    # Directions = np.einsum('ijk, k -> ij', Rs, np.array([0, 0, 1]))
    Directions = FibonacciSphere(NDir)
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

    # Plot
    Figure, Axis = plt.subplots(2,2,figsize=(5.5,5.5))
    Axis[0,0].imshow(Array[:,:,Array.shape[2]//2], cmap='binary')
    Axis[0,1].imshow(Array[:,Array.shape[1]//2,:], cmap='binary')
    Axis[1,0].imshow(Array[Array.shape[0]//2,:,:].T, cmap='binary')
    for i in range(2):
        for j in range(2):
            Axis[i,j].axis('off')
    # Axis = Figure.add_subplot(2, 2, 4, projection='3d')
    # Axis.voxels(Array[:,:,::-1].T, color=(0,0,0,0.1))
    # Axis.set_xticks([])
    # Axis.set_yticks([])
    # Axis.set_zticks([])
    for Dir in range(0,144,12):
        for i, Ray in enumerate(Coords[Dir]):
            Axis[0,0].plot(Ray[Mask[Dir,i]][:,0],Ray[Mask[Dir,i]][:,1],color=(1,0,0))
            Axis[0,1].plot(Ray[Mask[Dir,i]][:,0],Ray[Mask[Dir,i]][:,2],color=(1,0,0))
            Axis[1,0].plot(Ray[Mask[Dir,i]][:,1],Ray[Mask[Dir,i]][:,2],color=(1,0,0))
    plt.show(Figure)

    # Count intercepts
    def CountIntercepts(Array, Coords, Mask):
        NDirs, NRays = Coords.shape[:2]
        I = np.zeros(NDirs)
        for Dir in range(NDirs):
            nI = []
            for nRay in range(NRays):
                if Lengths[Dir,nRay] > 0:
                    Ray = Coords[Dir,nRay][Mask[Dir,nRay]]
                    Voxels = Array[Ray[:,0], Ray[:,1], Ray[:,2]]
                    Voxels[1:] = np.maximum(Voxels[1:], Voxels[:-1])
                    Voxels[:-1] = np.maximum(Voxels[1:], Voxels[:-1])
                    Interfaces = Voxels[1:] - Voxels[:-1]
                    nI.append(np.sum(Interfaces > 0))
            I[Dir] = np.sum(nI)
        return I

    I = CountIntercepts(np.pad(Array,1), np.round(Coords).astype(int), Mask)

    # Plot rose of intercepts
    I_Plot = I / np.max(I) * max(Array.shape) / 2
    I_Plot = Directions*np.expand_dims(I_Plot,-1)

    Figure, Axis = plt.subplots(2,2, figsize=(5.5,5.5))
    Axis[0,0].imshow(Array[:,:,Array.shape[2]//2].T, cmap='binary_r')
    Axis[0,0].scatter(I_Plot[:,0] + Array.shape[0]/2, I_Plot[:,1] + Array.shape[1]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,0].scatter(-I_Plot[:,0] + Array.shape[0]/2, -I_Plot[:,1] + Array.shape[1]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,1].imshow(Array[:,Array.shape[1]//2,:].T, cmap='binary_r')
    Axis[0,1].scatter(I_Plot[:,0] + Array.shape[0]/2, I_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,1].scatter(-I_Plot[:,0] + Array.shape[0]/2, -I_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[1,0].imshow(Array[Array.shape[0]//2,:,:].T, cmap='binary_r')
    Axis[1,0].scatter(I_Plot[:,1] + Array.shape[1]/2, I_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[1,0].scatter(-I_Plot[:,1] + Array.shape[1]/2, -I_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    for i in range(2):
        for j in range(2):
            Axis[i,j].axis('off')
    Axis = Figure.add_subplot(2, 2, 4, projection='3d')
    Axis.voxels(Array, color=(0,0,0,0.1))
    Axis.scatter3D(I_Plot[:,0]+Array.shape[0]/2, I_Plot[:,1]+Array.shape[1]/2, I_Plot[:,2]+Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis.scatter3D(-I_Plot[:,0]+Array.shape[0]/2, -I_Plot[:,1]+Array.shape[1]/2, -I_Plot[:,2]+Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis.set_xticks([])
    Axis.set_yticks([])
    Axis.set_zticks([])
    plt.show(Figure)

    # Plot Mean Intercept Length rose
    MIL = 1 / I
    MIL_Plot = MIL / np.max(MIL) * max(Array.shape) / 2
    MIL_Plot = Directions*np.expand_dims(MIL_Plot,-1)

    Figure, Axis = plt.subplots(2,2, figsize=(5.5,5.5))
    Axis[0,0].imshow(Array[:,:,Array.shape[2]//2].T, cmap='binary_r')
    Axis[0,0].scatter(MIL_Plot[:,0] + Array.shape[0]/2, MIL_Plot[:,1] + Array.shape[1]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,0].scatter(-MIL_Plot[:,0] + Array.shape[0]/2, -MIL_Plot[:,1] + Array.shape[1]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,1].imshow(Array[:,Array.shape[1]//2,:].T, cmap='binary_r')
    Axis[0,1].scatter(MIL_Plot[:,0] + Array.shape[0]/2, MIL_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[0,1].scatter(-MIL_Plot[:,0] + Array.shape[0]/2, -MIL_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[1,0].imshow(Array[Array.shape[0]//2,:,:].T, cmap='binary_r')
    Axis[1,0].scatter(MIL_Plot[:,1] + Array.shape[1]/2, MIL_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis[1,0].scatter(-MIL_Plot[:,1] + Array.shape[1]/2, -MIL_Plot[:,2] + Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    for i in range(2):
        for j in range(2):
            Axis[i,j].axis('off')
    Axis = Figure.add_subplot(2, 2, 4, projection='3d')
    Axis.voxels(Array, color=(0,0,0,0.1))
    Axis.scatter3D(MIL_Plot[:,0]+Array.shape[0]/2, MIL_Plot[:,1]+Array.shape[1]/2, MIL_Plot[:,2]+Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis.scatter3D(-MIL_Plot[:,0]+Array.shape[0]/2,-MIL_Plot[:,1]+Array.shape[1]/2, -MIL_Plot[:,2]+Array.shape[2]/2, color=(1,0,0), facecolor=(0,0,0,0))
    Axis.set_xticks([])
    Axis.set_yticks([])
    Axis.set_zticks([])
    plt.show(Figure)

    # Next step: Fit MIL
    H = FitFabric(MIL, Directions)
    eValues, eVectors = np.linalg.eig(H)
    eValues = 1 / np.sqrt(eValues)

    # Normalize values and sort
    eValues = 3 * eValues / sum(eValues)
    eVectors = eVectors.T[np.argsort(eValues)]
    eValues = np.sort(eValues)

    DA = eValues[2] / eValues[0]
    eValues[2] / eValues[1]

    MIL_Plot = MIL / np.max(MIL)
    MIL_Plot = Directions*np.expand_dims(MIL_Plot,-1)

    Figure, Axis = plt.subplots(1,1, subplot_kw={'projection':'3d'})
    Axis.quiver3D(0,0,0,eVectors[0,0],eVectors[1,0],eVectors[2,0], color=(1,0,0))
    Axis.quiver3D(0,0,0,eVectors[0,1],eVectors[1,1],eVectors[2,1], color=(0,1,0))
    Axis.quiver3D(0,0,0,eVectors[0,2],eVectors[1,2],eVectors[2,2], color=(0,0,1))
    Axis.scatter3D(MIL_Plot[:,0],MIL_Plot[:,1],MIL_Plot[:,2],color=(0,0,0,0.5))
    Axis.scatter3D(-MIL_Plot[:,0],-MIL_Plot[:,1],-MIL_Plot[:,2],color=(0,0,0,0.5))
    plt.show(Figure)
    


#%%

#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Script aimed to analyse three scans for the BalmerLab project.
        1. Full tibia scan: Length
        2. Midshaft section: Cortical morphometry
        3. Proximal section: trabecular morphometry

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: January 2023
    """

#%% Imports
# Modules import

import os
import time
import argparse
import numpy as np
import pandas as pd
from numba import njit
import SimpleITK as sitk
from pathlib import Path
from numba.core import types
import matplotlib.pyplot as plt
from numba.typed import Dict, List
from skimage import measure, morphology
# from pypore3d.p3dSITKPy import py_p3dReadRaw8 as ReadRaw8
# from pypore3d.p3dBlobPy import py_p3dMorphometricAnalysis as MA


# Time info utility functions
class Time:

    def __init__(self):
        self.Width = 15
        self.Length = 16
        self.Text = 'Process'
        self.Tic = time.time()
        pass

    def Print(self, Toc, Tic=None):

        """
        Print elapsed time in seconds to time in HH:MM:SS format
        :param Tic: Actual time at the beginning of the process
        :param Toc: Actual time at the end of the process
        """

        if Tic == None:
            Tic = self.Tic

        Delta = Toc - Tic

        Hours = np.floor(Delta / 60 / 60)
        Minutes = np.floor(Delta / 60) - 60 * Hours
        Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

        print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

        return

    def Update(self, Progress, Text=''):

        Percent = int(round(Progress * 100))
        Np = self.Width * Percent // 100
        Nb = self.Width - Np

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        Ns = self.Length - len(Text)
        if Ns >= 0:
            Text += Ns*' '
        else:
            Text = Text[:self.Length]
        
        Line = '\r' + Text + ' [' + Np*'=' + Nb*' ' + ']' + f' {Percent:.0f}%'
        print(Line, sep='', end='', flush=True)

    def Process(self, StartStop:bool, Text=''):

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        if StartStop*1 == 1:
            print('')
            self.Tic = time.time()
            self.Update(0, Text)

        elif StartStop*1 == 0:
            Toc = time.time()
            self.Update(1, Text)
            self.Print(Toc)

Time = Time()

# Morphometry functions
class Morphometry():

    def __init__(self):
        self.Echo = True
        pass

    def SplitTriangle(self, Tri):

        """ 
        Used in SphereTriangles for MIL computation
        Splits one triange into four triangles. 
        """

        P1 = Tri[0]
        P2 = Tri[1]
        P3 = Tri[2]
        P1x = P1[0]
        P1y = P1[1]
        P1z = P1[2]
        P2x = P2[0]
        P2y = P2[1]
        P2z = P2[2]
        P3x = P3[0]
        P3y = P3[1]
        P3z = P3[2]
        P4 = ((P1x + P2x) / 2, (P1y + P2y) / 2, (P1z + P2z) / 2)
        P5 = ((P3x + P2x) / 2, (P3y + P2y) / 2, (P3z + P2z) / 2)
        P6 = ((P1x + P3x) / 2, (P1y + P3y) / 2, (P1z + P3z) / 2)
        nTri1 = (P1, P4, P6)
        nTri2 = (P4, P2, P5)
        nTri3 = (P4, P5, P6)
        nTri4 = (P5, P3, P6)
        return nTri1, nTri2, nTri3, nTri4
    
    def CorrectValues(self, X, Y, Z, Precision=1e-06):

        """
        Used in Project2UnitSphere for MIL computation
        Ensure that current direction do not go through corner or edge  
        i.e. has an angle of 45 deg.
        """

        C1 = abs(int(X / Precision)) == abs(int(Y / Precision))
        C2 = abs(int(X / Precision)) == abs(int(Z / Precision))

        if C1 and C2:
            X += Precision
            Z += 2.0 * Precision
        elif abs(int(X / Precision)) == abs(int(Y / Precision)):
            X += Precision
        elif abs(int(X / Precision)) == abs(int(Z / Precision)):
            X += Precision
        elif abs(int(Z / Precision)) == abs(int(Y / Precision)):
            Z += Precision

        return (X, Y, Z)

    def Project2UnitSphere(self, PointRS):

        """
        Used in SphereTriangles for MIL computation
        Projects an equally sided triangle patch to a unit sphere
        """

        S45 = np.sin(np.pi / 4.0)
        XYZ = [(0.0, 0.0, 0.0),
               (1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0),
               (0.0, 0.0, 1.0),
               (0.5, 0.0, 0.0),
               (S45, S45, 0.0),
               (0.0, 0.5, 0.0),
               (S45, 0.0, S45),
               (0.0, S45, S45),
               (0.0, 0.0, 0.5)]

        R = PointRS[0]
        S = PointRS[1]
        T = PointRS[2]

        N5 = 4.0 * R * (1.0 - R - S - T)
        N6 = 4.0 * R * S
        N7 = 4.0 * S * (1.0 - R - S - T)
        N8 = 4.0 * R * T
        N9 = 4.0 * S * T
        N10 = 4.0 * T * (1.0 - R - S - T)

        N1 = 1.0 - R - S - T - 0.5 * N5 - 0.5 * N7 - 0.5 * N10
        N2 = R - 0.5 * N5 - 0.5 * N6 - 0.5 * N8
        N3 = S - 0.5 * N6 - 0.5 * N7 - 0.5 * N9
        N4 = T - 0.5 * N8 - 0.5 * N9 - 0.5 * N10

        aN = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]
        X = 0.0
        Y = 0.0
        Z = 0.0
        for Node in range(10):
            X += XYZ[Node][0] * aN[Node]
            Y += XYZ[Node][1] * aN[Node]
            Z += XYZ[Node][2] * aN[Node]

        X, Y, Z = self.CorrectValues(X, Y, Z)

        Factor = 1.0 / np.sqrt(X * X + Y * Y + Z * Z)

        return (Factor * X, Factor * Y, Factor * Z)

    def SphereTriangles(self, nDirs):

        """ 
         Used in MIL computation to setup a mesh for a unit sphere. 
         
         :param nDirs: Parameter for number of triangles on unit sphere 
                       (No of triangles = 8*4^power). 
                       - TYPE: int          
                      
         :return: Triangles: List of triangles 
                  - TYPE: list[ (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) ]
                  - float xi, yi, zi ... x,y,z coordinates of triangle corner point i                          
        """

        Triangles = []
        Triangles.append(((1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0),
                          (0.0, 0.0, 1.0)))
        for cDir in range(int(nDirs)):
            NewTriangles = []
            for Triangle in Triangles:
                nTri1, nTri2, nTri3, nTri4 = self.SplitTriangle(Triangle)
                NewTriangles.append(nTri1)
                NewTriangles.append(nTri2)
                NewTriangles.append(nTri3)
                NewTriangles.append(nTri4)

            Triangles = NewTriangles

        NewTriangles = []
        for Triangle in Triangles:
            nP1 = self.Project2UnitSphere(Triangle[0])
            nP2 = self.Project2UnitSphere(Triangle[1])
            nP3 = self.Project2UnitSphere(Triangle[2])
            nTr = (nP1, nP2, nP3)
            NewTriangles.append(nTr)

        Triangles = NewTriangles
        NewTriangles2 = []
        for Triangle in Triangles:
            NewTriangles2.append(Triangle)
            T1 = (Triangle[0][0], -Triangle[0][1], Triangle[0][2])
            T2 = (Triangle[1][0], -Triangle[1][1], Triangle[1][2])
            T3 = (Triangle[2][0], -Triangle[2][1], Triangle[2][2])
            NewTriangles2.append((T1, T2, T3))

        Triangles = NewTriangles2
        NewTriangles3 = []
        for Triangle in Triangles:
            NewTriangles3.append(Triangle)
            T1 = (-Triangle[0][0], Triangle[0][1], Triangle[0][2])
            T2 = (-Triangle[1][0], Triangle[1][1], Triangle[1][2])
            T3 = (-Triangle[2][0], Triangle[2][1], Triangle[2][2])
            NewTriangles3.append((T1, T2, T3))

        Triangles = NewTriangles3
        NewTriangles4 = []
        for Triangle in Triangles:
            NewTriangles4.append(Triangle)
            T1 = (Triangle[0][0], Triangle[0][1], -Triangle[0][2])
            T2 = (Triangle[1][0], Triangle[1][1], -Triangle[1][2])
            T3 = (Triangle[2][0], Triangle[2][1], -Triangle[2][2])
            NewTriangles4.append((T1, T2, T3))

        Triangles = NewTriangles4
        return Triangles

    def AreaAndCOG(self, Triangle):

        """ 
        Used in NormalAndArea for MIL computation
        Computes area and center of gravity of a triangle
        The length of the normal is "1". 
        """
        P1 = np.array(Triangle[0])
        P2 = np.array(Triangle[1])
        P3 = np.array(Triangle[2])

        P21 = P2 - P1
        P31 = P3 - P1

        A = 0.5 * np.linalg.norm(np.cross(P21, P31))

        X = (P1[0] + P2[0] + P3[0]) / 3.0
        Y = (P1[1] + P2[1] + P3[1]) / 3.0
        Z = (P1[2] + P2[2] + P3[2]) / 3.0

        Factor = 1.0 / np.sqrt(X * X + Y * Y + Z * Z)

        return (A, (Factor * X, Factor * Y, Factor * Z))

    def NormalAndArea(self, Power):

        """
        Used in OriginalDistribution for MIL computation
        Computes the normals at COG and area (weight) of 
        a triangulated unit sphere.        
        
        :param Power: Parameter for number of triangles on unit sphere 
                      (No of triangles = 8*4^power). 
                      - TYPE: int
                     
        :return: Normals: normals from COG with unit length 
                 - TYPE: list[ (nix, niy, niz) ]  
                 - float nix, niy, niz ... components of normal vectors 
                                  
                 Area_n: area of triangles which build the surface of sphere    
                 - TYPE: dict[ (nix, niy, niz) ] = value    
                 - float nix, niy, niz ... components of normal vectors 
                 - float value         ... Area for that direction                         
        """
        Triangles = self.SphereTriangles(Power)
        Normals = []
        Area_n = {}
        ASum = 0.0
        for Triangle in Triangles:
            A, COG = self.AreaAndCOG(Triangle)
            Normals.append(COG)
            Area_n[COG] = A
            ASum += A

        k = 4.0 * np.pi / ASum
        for n in Area_n:
            Area_n[n] = Area_n[n] * k

        return Normals, Area_n

    def OriginalDistribution(self, Array, Step, Power):

        """
        Used in step 2 of MIL computation
        Function computes MIL/SLD/SVD distributions for direction vectors "n" 
        using a voxel ray field going trought the RVE. Normals n = tuple(nix,niy,niz) 
        are the directions from the midpoint of a unit sphere to the COG of triangles 
        which build the surface of the sphere. Very similar to 
        self.computeOrigDistribution_STAR(). 
        A segement voxel model with isotropic resolution is needed.   
        
        :param Array: Segmented voxel model
                      - TYPE: numpy.array[kZ, jY, iX] = grayValue 
                      - int iX, jY, kZ ... voxels number ID in x,y,z start a 0, x fastest.       
                      - int grayValue  ... gray value of voxel, 0..255                  
        :param Step: Step in the considered voxel 
                      - TYPE: int > 0 
        :param Power: Power for the number of star directions = 8*4^power
                      - TYPE: int > 1
        :param Valid: Smallest valid intersection length.
                      - TYPE: float
        :param Echo: Flag for printing the  " ... Threshold Data" on stdout
                      - TYPE: bool
                                     
        @return: MIL: Mean Intercept Length 
                      - TYPE: dict[ (nix, niy, niz) ] = value 
                      - float nix, niy, niz ... components of normal vectors 
                      - float value         ... MIL for that direction          
                 SLD: Star Length Distribution 
                      - TYPE: same as for MIL 
                 SVD: Star Volume Distribution 
                      - TYPE: same as for MIL       
                 Area: Weight (Area of triange on unit sphere) for each direction 
                      - TYPE: same as for MIL          
        """

        if self.Echo:
            Text = 'Compute MIL'
            Time.Process(1, Text)

        MIL = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)
        SVD = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)
        SLD = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)

        SumL = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)
        SumL2 = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)
        SumL4 = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64,)

        Corners = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:],)
        Corners['swb'] = np.asarray([0.0, 0.0, 0.0], dtype='f8')
        Corners['seb'] = np.asarray([float(self.nX), 0.0, 0.0], dtype='f8')
        Corners['neb'] = np.asarray([float(self.nX), float(self.nY), 0.0], dtype='f8')
        Corners['nwb'] = np.asarray([0.0, float(self.nY), 0.0], dtype='f8')
        Corners['swt'] = np.asarray([0.0, 0.0, float(self.nZ)], dtype='f8')
        Corners['set'] = np.asarray([float(self.nX), 0.0, float(self.nZ)], dtype='f8')
        Corners['net'] = np.asarray([float(self.nX), float(self.nY), float(self.nZ)], dtype='f8')
        Corners['nwt'] = np.asarray([0.0, float(self.nY), float(self.nZ)], dtype='f8')


        sDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        sDict['r-dir'] = (1.0, 0.0, 0.0)
        sDict['s-dir'] = (0.0, 0.0, 1.0)
        eDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        eDict['r-dir'] = (0.0, 1.0, 0.0)
        eDict['s-dir'] = (0.0, 0.0, 1.0)
        nDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        nDict['r-dir'] = (1.0, 0.0, 0.0)
        nDict['s-dir'] = (0.0, 0.0, 1.0)
        wDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        wDict['r-dir'] = (0.0, 1.0, 0.0)
        wDict['s-dir'] = (0.0, 0.0, 1.0)
        bDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        bDict['r-dir'] = (1.0, 0.0, 0.0)
        bDict['s-dir'] = (0.0, 1.0, 0.0)
        tDict = Dict.empty(key_type=types.unicode_type, value_type=types.UniTuple(types.float64, 3),)
        tDict['r-dir'] = (1.0, 0.0, 0.0)
        tDict['s-dir'] = (0.0, 1.0, 0.0)


        InnerDict = types.DictType(types.unicode_type, types.UniTuple(types.float64, 3))
        ModelPlanes = Dict.empty(key_type=types.unicode_type, value_type=InnerDict,)
        ModelPlanes['s'] = sDict
        ModelPlanes['e'] = eDict
        ModelPlanes['n'] = nDict
        ModelPlanes['w'] = wDict
        ModelPlanes['b'] = bDict
        ModelPlanes['t'] = tDict
        
        BaseModel = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type,)
        BaseModel['s'] = 'swb'
        BaseModel['e'] = 'seb'
        BaseModel['n'] = 'nwb'
        BaseModel['w'] = 'swb'
        BaseModel['b'] = 'swb'
        BaseModel['t'] = 'swt'
        
        ViewerAt = {}
        ViewerAt['swb'] = (1.0, 1.0, 1.0)
        ViewerAt['seb'] = (-1.0, 1.0, 1.0)
        ViewerAt['neb'] = (-1.0, -1.0, 1.0)
        ViewerAt['nwb'] = (1.0, -1.0, 1.0)

        ViewerTo = {}
        ViewerTo['swb'] = 'net'
        ViewerTo['seb'] = 'nwt'
        ViewerTo['neb'] = 'swt'
        ViewerTo['nwb'] = 'set'

        Normals, Area = self.NormalAndArea(Power)
        Dict1 = {}
        Direction = ''
        for n in Normals:
            nX = n[0]
            nY = n[1]
            nZ = n[2]
            if nX >= 0.0 and nY >= 0.0 and nZ >= 0.0:
                VoxX = 1
                VoxY = 1
                VoxZ = 1
                StepX = 1
                StepY = 1
                StepZ = 1
                VoxelRay = []
                VoxelRay.append((VoxX, VoxY, VoxZ))
                if abs(nX) > abs(nY):
                    if abs(nZ) > abs(nX):
                        Direction = 'Z'
                    else:
                        Direction = 'X'
                elif abs(nZ) > abs(nY):
                    Direction = 'Z'
                else:
                    Direction = 'Y'
                PreVoxX = 1
                PreVoxY = 1
                PreVoxZ = 1
                C1 = abs(VoxX) <= self.nX
                C2 = abs(VoxY) <= self.nY
                C3 = abs(VoxZ) <= self.nZ
                while C1 and C2 and C3:
                    TMaxX = VoxX / nX
                    TMaxY = VoxY / nY
                    TMaxZ = VoxZ / nZ
                    if abs(TMaxX) < abs(TMaxY):
                        if abs(TMaxX) < abs(TMaxZ):
                            VoxX = VoxX + StepX
                        else:
                            VoxZ = VoxZ + StepZ
                    elif abs(TMaxY) < abs(TMaxZ):
                        VoxY = VoxY + StepY
                    else:
                        VoxZ = VoxZ + StepZ

                    Cc1 = abs(VoxX) <= self.nX
                    Cc2 = abs(VoxY) <= self.nY
                    Cc3 = abs(VoxZ) <= self.nZ
                    if Cc1 and Cc2 and Cc3:
                        if Direction == 'X':
                            if VoxX > PreVoxX:
                                VoxelRay.append((VoxX, VoxY, VoxZ))
                        if Direction == 'Y':
                            if VoxY > PreVoxY:
                                VoxelRay.append((VoxX, VoxY, VoxZ))
                        if Direction == 'Z':
                            if VoxZ > PreVoxZ:
                                VoxelRay.append((VoxX, VoxY, VoxZ))
                    PreVoxX = VoxX
                    PreVoxY = VoxY
                    PreVoxZ = VoxZ

                    C1 = abs(VoxX) <= self.nX
                    C2 = abs(VoxY) <= self.nY
                    C3 = abs(VoxZ) <= self.nZ

                Dict1[n] = VoxelRay

        i = 0
        Sum = len(ViewerAt) * 4.0 ** float(Power)
        DictList = []
        ListNLs = []
        ListSumL = []
        for v in ViewerAt:
            Dict2 = Dict.empty(key_type=types.UniTuple(types.float64, 3), value_type=types.float64[:,:],)
            CornVoxX = int(Corners[v][0])
            CornVoxY = int(Corners[v][1])
            CornVoxZ = int(Corners[v][2])
            if CornVoxX == 0:
                CornVoxX = 1
            if CornVoxY == 0:
                CornVoxY = 1
            if CornVoxZ == 0:
                CornVoxZ = 1
            StepX = int(ViewerAt[v][0])
            StepY = int(ViewerAt[v][1])
            StepZ = int(ViewerAt[v][2])
            for n in Dict1:
                VoxelRay = Dict1[n]
                NewVoxelRay = []
                for Voxel in VoxelRay:
                    VoxelX = CornVoxX + StepX * Voxel[0] - StepX
                    VoxelY = CornVoxY + StepY * Voxel[1] - StepY
                    VoxelZ = CornVoxZ + StepZ * Voxel[2] - StepZ
                    NewVoxelRay.append((VoxelX, VoxelY, VoxelZ))

                D = n[0] * ViewerAt[v][0], n[1] * ViewerAt[v][1], n[2] * ViewerAt[v][2]
                Dict2[tuple(D)] = np.asarray(NewVoxelRay, dtype='f8')

            EntryPlanes = List([v[0], v[1], v[2]])

            if self.Echo:
                Time.Update(i/Sum, 'Setup Data')

            CVx = List([CornVoxX, CornVoxY, CornVoxZ])
            Ns = List([self.nX, self.nY, self.nZ])
            Vars = List([MIL, SVD, SLD])
            Sums = List([SumL, SumL2, SumL4])
            i, MIL, SVD, SLD, NLs, SumL, Voxels = NumbaSetupMILData(i, v, Step, Array,
                                                 CVx, Ns, Vars, Sums,
                                                 Dict2, Corners, EntryPlanes,
                                                 ModelPlanes, BaseModel)
            
            DictList.append(Dict2)
            ListNLs.append(NLs)
            ListSumL.append(SumL)

        if self.Echo:
            Time.Process(0, Text)
        
        return MIL, SVD, SLD, Area, Dict1, DictList, ListNLs, ListSumL, Voxels

    def FabricTensor(self, OrgMIL):

        """ 
        Used in ApproximalDistribution for MIL computation
        Compute the fabric tensors using an ellipsoidal fit
        
         :param OrgMIL: Original distribution 
                - TYPE: dict[ (nix, niy, niz) ] = value 
                - float nix, niy, niz ... components of normal vector 
                - float value         ... value for that direction             
                      
         :return: M: fabric tensor from ellipsoidal fit 
                  - TYPE: float numpy.array[3,3]            
        """

        nDir = len(OrgMIL)
        nHat = np.array(np.zeros((nDir, 6), float))
        An = np.array(np.zeros(nDir, float))
        H = np.array(np.zeros((3, 3), float))
        d = 0
        for n in OrgMIL:
            nHat[d, 0] = n[0] * n[0]
            nHat[d, 1] = n[1] * n[1]
            nHat[d, 2] = n[2] * n[2]
            nHat[d, 3] = np.sqrt(2.0) * n[1] * n[2]
            nHat[d, 4] = np.sqrt(2.0) * n[2] * n[0]
            nHat[d, 5] = np.sqrt(2.0) * n[0] * n[1]
            MILn = np.array(OrgMIL[n], dtype=float)
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

    def EigenValuesAndVectors(self, orgMIL):
        
        """
        Used in step 4 of MIL computation
        computes the eigenvalueS and eigenvectors by fitting an ellipsoid 
        
        :param  orgMIL: Original distribution 
                - TYPE: dict[ (nix, niy, niz) ] = value 
                - float nix, niy, niz ... components of normal vector 
                - float value         ... value for that direction             
            
        :return: evalue: Eigenvalues of fabric tensor 
                 - TYPE: numpy.array[evalID] = eval
                 - int evalID ... eigenvalues ID. 0,1, or 2
                 - float eval ... current eigenvalue 
                 evector : Eigenvectors of fabric tensor
                 - TYPE: numpy.array[evectID, evalID] = evect
                 - int evectID ... eigenvector ID. 0,1, or 2                         
                 - int evalID  ... eigenvalue ID. 0,1, or 2
                 - float evect ... component of eigenvectors, e.g. evector[0,2] = ev1_z
        """

        M = self.FabricTensor(orgMIL)
        eValue, eVector = np.linalg.eig(M)
        eValue[0] = 1.0 / np.sqrt(eValue[0])
        eValue[1] = 1.0 / np.sqrt(eValue[1])
        eValue[2] = 1.0 / np.sqrt(eValue[2])


        Norm = (eValue[0] + eValue[1] + eValue[2]) / 3.0

        eValue[0] = eValue[0] / Norm
        eValue[1] = eValue[1] / Norm
        eValue[2] = eValue[2] / Norm

        return eValue, eVector

    def MIL(self, Image, Power2=4, Step=5, Power=2):

        """
        Compute Mean Intercept Length of image
        Based on mia.py from medtool from D. H. Pahr
        """

        if hasattr(Image, 'GetSize'):
            self.nX, self.nY, self.nZ = Image.GetSize()
            Array = sitk.GetArrayFromImage(Image)
        elif hasattr(Image, 'shape'):
            self.nZ, self.nY, self.nX = Image.shape
            Array = Image
        else:
            print('Image must be either numpy array or sitk image')

        # Step 1: Compute original distribution
        OrgMIL, OrgSVD, OrgSLD, Area = self.OriginalDistribution(Array, Step, Power)

        # Step 2: Compute eigen values and eigen vectors
        eValMIL, eVectMIL = self.EigenValuesAndVectors(OrgMIL)

        return eValMIL, eVectMIL

    def Trabecular(self, Image, Mask=None, DA=True):

        """
        Perform trabecular morphological analysis
        :param Image: Segmented image of trabecular bone
                      - Type: binary 0-255 sitkImage
        :param Mask : Mask for region of interest selection
                      - Type: binary 0-255 sitkImage
        :param DA   : Compute or not the degree of anisotropy
                      using mean intercept length
        :return Data: Pandas data frame with computed parameters
        """

        Data = pd.DataFrame()

        # Write temporary images to disk
        dX, dY, dZ = Image.GetSize()
        Spacing = Image.GetSpacing()[0]
        sitk.WriteImage(Image, 'TempROI.mhd')
        sitk.WriteImage(Mask, 'TempMask.mhd')

        # Perform morphometric analysis
        ROI = ReadRaw8('TempROI.raw', dX, dY, dimz=dZ)
        ROIMask = ReadRaw8('TempMask.raw', dX, dY, dimz=dZ)
        MS = MA(ROI, ROIMask, dX, dY, dimz=dZ, resolution=Spacing)

        # Remove temporary files and write data to disk
        for F in ['TempROI','TempMask']:
            os.remove(F + '.mhd')
            os.remove(F + '.raw')

        # Store data
        Props = ['BV/TV (-)', 'Tb.Th. (mm)', 'Tb.N. (-)', 'Tb.Sp. (mm)']
        Measures = [MS.BvTv, MS.TbTh, MS.TbN, MS.TbSp]
        for Prop, Stat in zip(Props, Measures):
            Data.loc[0,Prop] = Stat

        # Compute MIL for degree of anisotropy assessment
        if DA:
            Masked = sitk.Mask(Image, Mask)
            eVal, eVect = self.MIL(Masked)
            Data.loc[0,'DA'] = max(eVal) / min(eVal)

        return Data

    def Cortical(self, Image):

        """
        Compute morphology standard parameters for cortical bone
        :param Image: Segmented image of trabecular bone
                      - Type: binary sitkImage
        :return Data: Pandas data frame with computed parameters
        """

        Data = pd.DataFrame()
        Spacing = Image.GetSpacing()[0]
        Size = Image.GetSize()

        T = []
        I = []
        D = []
        Circle = measure.CircleModel()
        for S in range(Size[2]):
            Slice = sitk.Slice(Image, (0, 0, S), (Size[0], Size[1], S+1))
            Pad = sitk.ConstantPad(Slice, (1, 1, 0), (1, 1, 0))
            Array = sitk.GetArrayFromImage(Pad)

            if Array.sum() > 0:
                # Cortical thickness
                Skeleton, Distance = morphology.medial_axis(Array[0], return_distance=True)
                T.append(2 * np.mean(Distance[Skeleton]) * Spacing)

                # Inertia
                Properties = measure.regionprops(Array)[0]
                I.append(Properties.inertia_tensor[0,0] * Properties.area * Spacing**2)

                # Fitted diameter
                Circle.estimate(Properties.coords[:,1:])
                D.append(Circle.params[2] * Spacing * 2)

        
        Props = ['C.Th (mm)', 'I (mm4)', 'D (mm)']
        Measures = [np.median(T), np.median(I), np.median(D)]
        for Prop, Stat in zip(Props, Measures):
            Data.loc[0,Prop] = Stat
        
        return Data

    def SegmentBone(self, Image, Sigma=0.02, Threshold=None, nThresholds=2, Mask=True, CloseSize=None):

        """
        Perform segmentation of bone form gray value image
        Step 1: gaussian filter to smooth values
        Step 2: multiple Otsu's algorithm for segmentation
        Step 3: Crop image at bone limits to reduce memory
        Step 4: Pad to avoid border contacts
        Optional
        Step 5: Close contour for further hole filling
        Step 6: Mask generation by filling holes in consecutive z slices

        :param Image: Gray value image
                      - Type: sitkImage
        :param Sigma: Filter width
                      - Type: float
        :param nThresholds: Number of Otsu's threshold
                      - Type: int
        :param Mask: Generate mask or not
                      - Type: bool
        :param CloseSize: Radius used to close the contour
                      - Type: int

        :return GrayCrop: Cropped gray value image
                BinCrop: Cropped binary segmented image
                Mask: Generated mask
        """

        # Filter image to reduce noise
        Gauss = sitk.SmoothingRecursiveGaussianImageFilter()
        Gauss.SetSigma(Sigma)
        Smooth  = Gauss.Execute(Image)

        if Threshold:
            # Segment image using single threshold
            Binarize = sitk.BinaryThresholdImageFilter()
            Binarize.SetUpperThreshold(Threshold)
            Binarize.SetOutsideValue(255)
            Binarize.SetInsideValue(0)
            Bin = Binarize.Execute(Smooth)

        else:
            # Segment image by thresholding using Otsu's method
            Otsu = sitk.OtsuMultipleThresholdsImageFilter()
            Otsu.SetNumberOfThresholds(nThresholds)
            Seg = Otsu.Execute(Smooth)

            # Binarize image to keep bone only
            Binarize = sitk.BinaryThresholdImageFilter()
            Binarize.SetUpperThreshold(nThresholds-1)
            Binarize.SetOutsideValue(255)
            Binarize.SetInsideValue(0)
            Bin = Binarize.Execute(Seg)

        # Crop image to bone
        Array = sitk.GetArrayFromImage(Bin)
        Z, Y, X = np.where(Array > 0)
        X1, X2 = int(X.min()), int(X.max())
        Y1, Y2 = int(Y.min()), int(Y.max())
        Z1, Z2 = int(Z.min()), int(Z.max())
        BinCrop = sitk.Slice(Bin, (X1, Y1, Z1), (X2, Y2, Z2))
        GrayCrop = sitk.Slice(Image, (X1, Y1, Z1), (X2, Y2, Z2))

        # Pad images to avoid contact with border
        BinCrop = sitk.ConstantPad(BinCrop, (1,1,1), (1,1,1))
        GrayCrop = sitk.ConstantPad(GrayCrop, (1,1,1), (1,1,1))

        if Mask:
            # Close contour
            Close = sitk.BinaryMorphologicalClosingImageFilter()
            Close.SetForegroundValue(255)
            Close.SetKernelRadius(CloseSize)
            Closed = Close.Execute(BinCrop)

            # Generate mask slice by slice
            Size = BinCrop.GetSize()
            Mask = BinCrop
            for Start in range(Size[2]):
                Slice = sitk.Slice(Closed, (0, 0, Start), (Size[0], Size[1], Start+1))

                # Pad slice in z direction to "close" holes
                Pad = sitk.ConstantPad(Slice, (0,0,1), (0,0,1), 255)

                # Fill holes
                Fill = sitk.BinaryFillholeImageFilter()
                Fill.SetForegroundValue(255)
                Filled = Fill.Execute(Pad)

                # Get center slice
                Slice = sitk.Slice(Filled, (0, 0, 1), (Size[0], Size[1], 2))

                # Paste slice into original image
                Mask = sitk.Paste(Mask, Slice, Slice.GetSize(), destinationIndex=(0,0,Start+1))

            return GrayCrop, BinCrop, Mask
        
        else:
            return GrayCrop, BinCrop

Morphometry = Morphometry()
Morphometry.Echo = False

@njit
def NumbaSetupMILData(i, v, Step, Array,
                      CVx, Ns, Vars, Sums,
                      Dict2, Corners, EntryPlanes,
                      ModelPlanes, BaseModel):
    
    # Unpack variables
    CornVoxX, CornVoxY, CornVoxZ = CVx
    nX, nY, nZ = Ns
    MIL, SVD, SLD = Vars
    SumL, SumL2, SumL4 = Sums
    NLs = {}
    Voxels = {}
    
    for n in Dict2:
        i += 1
        nL = 0
        nNotValid0 = 0
        nValid0 = 0
        nNotValid1 = 0
        nValid1 = 0
        nNotValid3 = 0
        nValid3 = 0
        SumL[n] = 0.0
        SumL2[n] = 0.0
        SumL4[n] = 0.0
        NewVoxelRay = Dict2[n]
        nn = np.array([n[0], n[1], n[2]])
        nb = np.array((0.0, 0.0, 1.0))
        ng = np.cross(nn, nb)
        ns = np.cross(ng, nn)
        nr = np.cross(ns, nn)
        ng = ng / np.linalg.norm(ng)
        ns = ns / np.linalg.norm(ns)
        nr = nr / np.linalg.norm(nr)
        rmax = 0.0
        rmin = 0.0
        smax = 0.0
        smin = 0.0
        r1c = Corners[v]
        for c in Corners:
            r0c = Corners[c]
            b = r0c - r1c
            a11 = nr[0]
            a12 = ns[0]
            a13 = -nn[0]
            a21 = nr[1]
            a22 = ns[1]
            a23 = -nn[1]
            a31 = nr[2]
            a32 = ns[2]
            a33 = -nn[2]
            DET = a11 * (a33 * a22 - a32 * a23) - a21 * (a33 * a12 - a32 * a13) + a31 * (a23 * a12 - a22 * a13)
            x = [0.0, 0.0, 0.0]
            x[0] = 1.0 / DET * ((a33 * a22 - a32 * a23) * b[0] - (a33 * a12 - a32 * a13) * b[1] + (a23 * a12 - a22 * a13) * b[2])
            x[1] = 1.0 / DET * (-(a33 * a21 - a31 * a23) * b[0] + (a33 * a11 - a31 * a13) * b[1] - (a23 * a11 - a21 * a13) * b[2])
            if (x[0] > rmax):
                rmax = x[0]
            if x[0] < rmin:
                rmin = x[0]
            if x[1] > smax:
                smax = x[1]
            if x[1] < smin:
                smin = x[1]

        DirVoxels = {}
        for curR in range(int(rmin), int(rmax + 1), Step):
            for curS in range(int(smin), int(smax + 1), Step):
                for Plane in EntryPlanes:
                    CutPlane = ModelPlanes[Plane]
                    r1 = Corners[BaseModel[Plane]]
                    r0 = curR * nr + curS * ns + r1c
                    at = nn
                    br = np.array(CutPlane['r-dir'])
                    cs = np.array(CutPlane['s-dir'])
                    b = r0 - r1
                    a11 = br[0]
                    a12 = cs[0]
                    a13 = -at[0]
                    a21 = br[1]
                    a22 = cs[1]
                    a23 = -at[1]
                    a31 = br[2]
                    a32 = cs[2]
                    a33 = -at[2]
                    DET = a11 * (a33 * a22 - a32 * a23) - a21 * (a33 * a12 - a32 * a13) + a31 * (a23 * a12 - a22 * a13)
                    x = [0.0, 0.0, 0.0]
                    x[0] = 1.0 / DET * ((a33 * a22 - a32 * a23) * b[0] - (a33 * a12 - a32 * a13) * b[1] + (a23 * a12 - a22 * a13) * b[2])
                    x[1] = 1.0 / DET * (-(a33 * a21 - a31 * a23) * b[0] + (a33 * a11 - a31 * a13) * b[1] - (a23 * a11 - a21 * a13) * b[2])
                    ipt = x[0] * br + x[1] * cs + r1
                    C1 = ipt[0] >= 0.0
                    C2 = ipt[1] >= 0.0
                    C3 = ipt[2] >= 0.0
                    C4 = ipt[0] <= nX
                    C5 = ipt[1] <= nY
                    C6 = ipt[2] <= nZ
                    if C1 and C2 and C3 and C4 and C5 and C6:
                        EntryVoxX = int(ipt[0] + 0.5)
                        EntryVoxY = int(ipt[1] + 0.5)
                        EntryVoxZ = int(ipt[2] + 0.5)
                        if EntryVoxX == 0:
                            EntryVoxX = 1
                        if EntryVoxY == 0:
                            EntryVoxY = 1
                        if EntryVoxZ == 0:
                            EntryVoxZ = 1
                        StartBone = (1, 1, 1)
                        EndBone = (1, 1, 1)
                        PrevVox = (1, 1, 1)
                        StartFlag = False
                        for StartRayVox in NewVoxelRay:
                            VoxX = StartRayVox[0] - (CornVoxX - EntryVoxX)
                            VoxY = StartRayVox[1] - (CornVoxY - EntryVoxY)
                            VoxZ = StartRayVox[2] - (CornVoxZ - EntryVoxZ)
                            Xv = int(VoxX - 1)
                            Yv = int(VoxY - 1)
                            Zv = int(VoxZ - 1)
                            Cc1 = VoxX < 1
                            Cc2 = VoxY < 1
                            Cc3 = VoxZ < 1
                            Cc4 = VoxX > nX
                            Cc5 = VoxY > nY
                            Cc6 = VoxZ > nZ
                            if Cc1 or Cc2 or Cc3 or Cc4 or Cc5 or Cc6:
                                if StartFlag == True:
                                    if VoxX > nX or VoxY > nY or VoxZ > nZ:
                                        StartFlag = False
                                        EndBone = PrevVox[0],PrevVox[1], PrevVox[2]
                                        lx = StartBone[0] - EndBone[0]
                                        ly = StartBone[1] - EndBone[1]
                                        lz = StartBone[2] - EndBone[2]
                                        L2 = lx * lx + ly * ly + lz * lz
                                        if L2 > 0.0:
                                            nL += 1
                                            SumL[n] += L2 ** 0.5
                                            SumL2[n] += L2
                                            SumL4[n] += L2 * L2
                            elif Array[Zv, Yv, Xv] == 0:
                                if StartFlag == True:
                                    StartFlag = False
                                    EndBone = PrevVox[0], PrevVox[1], PrevVox[2]
                                    lx = StartBone[0] - EndBone[0]
                                    ly = StartBone[1] - EndBone[1]
                                    lz = StartBone[2] - EndBone[2]
                                    L2 = lx * lx + ly * ly + lz * lz
                                    Text2 = L2
                                    if L2 > 0.0:
                                        nL += 1
                                        SumL[n] += L2 ** 0.5
                                        SumL2[n] += L2
                                        SumL4[n] += L2 * L2
                            elif StartFlag == False:
                                StartBone = (VoxX, VoxY, VoxZ)
                                StartFlag = True
                            PrevVox = (VoxX, VoxY, VoxZ)
                            DirVoxels[(r0[0], r0[1], r0[2])] = ipt
                        break

        Voxels[n] = DirVoxels
        n2 = (-n[0], -n[1], -n[2])
        MIL[n] = SumL[n] / float(nL)
        MIL[n2] = SumL[n] / float(nL)
        NLs[n] = nL
        SLD[n] = SumL2[n] / SumL[n]
        SLD[n2] = SumL2[n] / SumL[n]
        SVD[n] = np.pi / 3.0 * SumL4[n] / SumL[n]
        SVD[n2] = np.pi / 3.0 * SumL4[n] / SumL[n]
        
    return i, MIL, SVD, SLD, NLs, SumL, Voxels

#%% Main
# Main code

def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])
        
    for ROI in [InputROIs[51]]:

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = np.load(ROI)

        # Perform trabecular morphometry analysis
        Time.Update(9/10,'Trab. Props')
        eValues, eVectors = Morphometry.MIL(Array)

    return

#%% Execution part
# Execution as main
if __name__ == '__main__':

    # Initiate the parser with a description
    FC = argparse.RawDescriptionHelpFormatter
    Parser = argparse.ArgumentParser(description=Description, formatter_class=FC)

    # Add long and short argument
    SV = Parser.prog + ' version ' + Version
    Parser.add_argument('-V', '--Version', help='Show script version', action='version', version=SV)
    Parser.add_argument('--File', help='File containing sample list', type=str, default='List.csv')

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments.File)
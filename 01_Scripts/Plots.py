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

import time
import struct
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import SimpleITK as sitk

# %% Time class
class Time():

    def __init__(self):
        self.Width = 15
        self.Length = 16
        self.Text = 'Process'
        self.Tic = time.time()
        pass
    
    def Set(self, Tic=None):
        
        if Tic == None:
            self.Tic = time.time()
        else:
            self.Tic = Tic

    def Print(self, Tic=None,  Toc=None):

        """
        Print elapsed time in seconds to time in HH:MM:SS format
        :param Tic: Actual time at the beginning of the process
        :param Toc: Actual time at the end of the process
        """

        if Tic == None:
            Tic = self.Tic
            
        if Toc == None:
            Toc = time.time()


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
            self.Tic = time.time()
            self.Update(0, Text)

        elif StartStop*1 == 0:
            self.Update(1, Text)
            self.Print()

Time = Time()

#%% Functions

def ReadISQ(File, InfoFile=False, Echo=False, ASCII=False):

    """
    This function read an ISQ file from Scanco and return an ITK image and additional data.
    
    Adapted from https://github.com/mdoube/BoneJ/blob/master/src/org/bonej/io/ISQReader.java
    
    Little endian byte order (the least significant bit occupies the lowest memory position.
    00   char    check[16];              // CTDATA-HEADER_V1
    16   int     data_type;
    20   int     nr_of_bytes;
    24   int     nr_of_blocks;
    28   int     patient_index;          //p.skip(28);
    32   int     scanner_id;				//p.skip(32);
    36   int     creation_date[2];		//P.skip(36);
    44   int     dimx_p;					//p.skip(44);
    48   int     dimy_p;
    52   int     dimz_p;
    56   int     dimx_um;				//p.skip(56);
    60   int     dimy_um;
    64   int     dimz_um;
    68   int     slice_thickness_um;		//p.skip(68);
    72   int     slice_increment_um;		//p.skip(72);
    76   int     slice_1_pos_um;
    80   int     min_data_value;
    84   int     max_data_value;
    88   int     mu_scaling;             //p.skip(88);  /* p(x,y,z)/mu_scaling = value [1/cm]
    92	 int     nr_of_samples;
    96	 int     nr_of_projections;
    100  int     scandist_um;
    104  int     scanner_type;
    108  int     sampletime_us;
    112  int     index_measurement;
    116  int     site;                   //coded value
    120  int     reference_line_um;
    124  int     recon_alg;              //coded value
    128  char    name[40]; 		 		//p.skip(128);
    168  int     energy;        /* V     //p.skip(168);
    172  int     intensity;     /* uA    //p.skip(172);
    ...
    508 int     data_offset;     /* in 512-byte-blocks  //p.skip(508);
    * So the first 16 bytes are a string 'CTDATA-HEADER_V1', used to identify
    * the type of data. The 'int' are all 4-byte integers.
    *
    * dimx_p is the dimension in pixels, dimx_um the dimensions in micrometer
    *
    * So dimx_p is at byte-offset 40, then dimy_p at 44, dimz_p (=number of
    * slices) at 48.
    *
    * The microCT calculates so called 'x-ray linear attenuation' values. These
    * (float) values are scaled with 'mu_scaling' (see header, e.g. 4096) to
    * get to the signed 2-byte integers values that we save in the .isq file.
    *
    * e.g. Pixel value 8192 corresponds to lin. att. coeff. of 2.0 [1/cm]
    * (8192/4096)
    *
    * Following to the headers is the data part. It is in 2-byte short integers
    * (signed) and starts from the top-left pixel of slice 1 to the left, then
    * the next line follows, until the last pixel of the last sclice in the
    * lower right.
    """

    if Echo:
        Text = 'Read ISQ'
        Time.Process(1, Text)

    try:
        f = open(File, 'rb')
    except IOError:
        print("\n **ERROR**: ISQReader: intput file ' % s' not found!\n\n" % File)
        print('\n E N D E D  with ERRORS \n\n')

    for Index in np.arange(0, 200, 4):
        f.seek(Index)
        f.seek(Index)

    f.seek(32)
    CT_ID = struct.unpack('i', f.read(4))[0]

    f.seek(28)
    sample_nb = struct.unpack('i', f.read(4))[0]

    f.seek(108)
    Scanning_time = struct.unpack('i', f.read(4))[0] / 1000

    f.seek(168)
    Energy = struct.unpack('i', f.read(4))[0] / 1000.

    f.seek(172)
    Current = struct.unpack('i', f.read(4))[0]

    f.seek(44)
    X_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(48)
    Y_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(52)
    Z_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(56)
    Res_General_X = struct.unpack('i', f.read(4))[0]

    f.seek(60)
    Res_General_Y = struct.unpack('i', f.read(4))[0]

    f.seek(64)
    Res_General_Z = struct.unpack('i', f.read(4))[0]

    Res_X = Res_General_X / float(X_pixel)
    Res_Y = Res_General_Y / float(Y_pixel)
    Res_Z = Res_General_Z / float(Z_pixel)

    Header_Txt = ['scanner ID:                 %s' % CT_ID,
                'scaning time in ms:         %s' % Scanning_time,
                'scaning time in ms:         %s' % Scanning_time,
                'Energy in keV:              %s' % Energy,
                'Current in muA:             %s' % Current,
                'nb X pixel:                 %s' % X_pixel,
                'nb Y pixel:                 %s' % Y_pixel,
                'nb Z pixel:                 %s' % Z_pixel,
                'resolution general X in mu: %s' % Res_General_X,
                'resolution general Y in mu: %s' % Res_General_Y,
                'resolution general Z in mu: %s' % Res_General_Z,
                'pixel resolution X in mu:   %.2f' % Res_X,
                'pixel resolution Y in mu:   %.2f' % Res_Y,
                'pixel resolution Z in mu:   %.2f' % Res_Z]

    if InfoFile:
        Write_File = open(File.split('.')[0] + '_info.txt', 'w')
        for Item in Header_Txt:
            Write_File.write("%s\n" % Item)
        Write_File.close()

    f.seek(44)
    Header = np.zeros(6)
    for i in range(0, 6):
        Header[i] = struct.unpack('i', f.read(4))[0]

    ElementSpacing = [Header[3] / Header[0] / 1000, Header[4] / Header[1] / 1000, Header[5] / Header[2] / 1000]
    f.seek(508)

    HeaderSize = 512 * (1 + struct.unpack('i', f.read(4))[0])
    f.seek(HeaderSize)

    NDim = [int(Header[0]), int(Header[1]), int(Header[2])]
    LDim = [float(ElementSpacing[0]), float(ElementSpacing[1]), float(ElementSpacing[2])]

    AdditionalData = {'-LDim': LDim,
                    '-NDim': NDim,
                    'ElementSpacing': LDim,
                    'DimSize': NDim,
                    'HeaderSize': HeaderSize,
                    'TransformMatrix': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    'CenterOfRotation': [0.0, 0.0, 0.0],
                    'Offset': [0.0, 0.0, 0.0],
                    'AnatomicalOrientation': 'LPS',
                    'ElementType': 'int16',
                    'ElementDataFile': File}

    if ASCII == False:
        VoxelModel = np.fromfile(f, dtype='i2')
        try:
            VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
            f.close()
            del f

        except:
            # if the length does not fit the dimensions (len(VoxelModel) != NDim[2] * NDim[1] * NDim[0]),
            # add an offset with seek to reshape the image -> actualise length, delta *2 = seek

            Offset = (len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))
            f.seek(0)
            VoxelModel = np.fromfile(f, dtype='i2')

            if Echo:
                print('len(VoxelModel) = ', len(VoxelModel))
                print('Should be ', (NDim[2] * NDim[1] * NDim[0]))
                print('Delta:', len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))

            f.seek((len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0])) * 2)
            VoxelModel = np.fromfile(f, dtype='i2')
            f.close()
            del f

            VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
            # the image is flipped by the Offset --> change the order to obtain the continuous image:
            VoxelModel = np.c_[VoxelModel[:, :, -Offset:], VoxelModel[:, :, :(VoxelModel.shape[2] - Offset)]]

    # If ISQ file was transfered with the Scanco microCT SFT in ASCII mode
    # a character is added every 256 bites, so build array by croping it
    else:
        LData = NDim[0] * NDim[1] * NDim[2]     # Data length
        NBytes = int(LData / 256)               # Number of bytes to store data
        Data = np.fromfile(f,dtype='i2')        # Read data
        Data = Data[::-1]                       # Reverse because data is at the end of the file
        cData = Data[:NBytes*257]               # Crop to data length
        sData = np.reshape(cData,(NBytes,257))  # Reshape for each 256 bytes
        sData = sData[:,1:]                     # Crop first byte artificially added by ascii ftp transfer
        
        # Reshape to scan dimensions
        VoxelModel = np.reshape(sData,(NDim[2],NDim[1],NDim[0]))

    if Echo:
        Time.Process(0,Text)
        print('\nScanner ID:                 ', CT_ID)
        print('Scanning time in ms:         ', Scanning_time)
        print('Energy in keV:              ', Energy)
        print('Current in muA:             ', Current)
        print('Nb X pixel:                 ', X_pixel)
        print('Nb Y pixel:                 ', Y_pixel)
        print('Nb Z pixel:                 ', Z_pixel)
        print('Pixel resolution X in mu:    %.2f' % Res_X)
        print('Pixel resolution Y in mu:    %.2f' % Res_Y)
        print('Pixel resolution Z in mu:    %.2f' % Res_Z)

    return VoxelModel, AdditionalData

def GetFabric(FileName):

    Morpho = pd.read_csv(FileName, delimiter=';')

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

    return eValues, eVectors

def PlotFabricROI(ROI:np.array, eValues:np.array, eVectors:np.array, FileName:Path) -> None:

    """
    Plots a 3D ellipsoid representing a region of interest (ROI) with scaling based on the
    eigenvalues and eigenvectors provided. The ellipsoid is overlaid on a binary structure mesh,
    and the plot is generated with the ability to visualize the MIL (Mean Intercept Length) values.

    Parameters:
    -----------
    ROI (3D array): A 3D binary array representing the region of interest (ROI).
        
    eValues (1D array): A 1D array containing the eigenvalues of the fabric.
        
    eVectors (3D array) : A 2D array (shape: 3x3) containing the eigenvectors of the fabric.
        
    Returns:
    --------
    None
    """

    # Create a unit sphere and transform it to an ellipsoid
    Sphere = pv.Sphere(radius=ROI.shape[0]/2, theta_resolution=50, phi_resolution=50)

    # Scale the sphere by the square roots of the eigenvalues
    ScaleMatrix = np.diag(np.sqrt(eValues))
    TransformMatrix = np.matmul(eVectors, ScaleMatrix)

    # Transform the sphere points to ellipsoid points
    Points = np.matmul(Sphere.points, TransformMatrix.T)

    # Center the ellipsoid at the structure's midpoint
    Offset = np.array(ROI.shape) / 2
    EllispoidPoints = Points + Offset
    Ellispoid = pv.PolyData(EllispoidPoints, Sphere.faces)

    # Calculate the radius for each ellipsoid point to color by radius
    Radii = np.linalg.norm(Ellispoid.points - Offset, axis=1)
    Radii = (Radii - min(Radii)) / (max(Radii) - min(Radii))
    Radii = Radii * (max(eValues) - min(eValues)) + min(eValues)
    Ellispoid['MIL'] = Radii

    # Plotting
    sargs = dict(font_family='times', 
                    width=0.05,
                    height=0.75,
                    vertical=True,
                    position_x=0.9,
                    position_y=0.125,
                    title_font_size=30,
                    label_font_size=20
                    )
    
    pl = pv.Plotter(off_screen=True)
    actors = pl.add_volume(ROI,
                           cmap='bone',
                           show_scalar_bar=False, opacity=[0,1,1])
    actors.prop.interpolation_type = 'linear'
    pl.add_mesh(Ellispoid, scalars='MIL', cmap='jet', style='wireframe', line_width=3, scalar_bar_args=sargs)
    pl.camera_position = 'xz'
    pl.camera.roll = 0
    pl.camera.elevation = 30
    pl.camera.azimuth = 30
    pl.camera.zoom(1.0)
    # pl.add_axes(viewport=(0,0,0.25,0.25))
    pl.screenshot(FileName, scale=2)
    # pl.show()

    return

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

def FrobeniusProduct(A,B):

    s = 0

    if A.size == 9 and B.size == 9:
        for i in range(3):
            for j in range(3):
                s += A[i, j] * B[i, j]

    elif A.size == 36 and B.size == 36:
        for i in range(6):
            for j in range(6):
                s = s + A[i, j] * B[i, j]

    elif A.shape == (9,9) and B.shape == (9,9):
        for i in range(9):
            for j in range(9):
                s = s + A[i, j] * B[i, j]

    elif A.shape == (3,3,3,3) and B.shape == (3,3,3,3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        s = s + A[i, j, k, l] * B[i, j, k, l]

    else:
        print('Matrices sizes mismatch')

    return s

def Transform(A,B):

    if A.size == 9 and B.size == 3:

        c = np.zeros(3)
        for i in range(3):
            for j in range(3):
                c[i] += A[i,j] * B[j]

        return c

    elif A.size == 27 and B.size == 9:

        c = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c[i] += A[i,j,k] * B[j,k]

        return c

    elif A.size == 81 and B.size == 9:

        C = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i,j] += A[i,j,k,l] * B[k,l]

        return C

    else:
        print('Matrices sizes mismatch')

def DyadicProduct(A,B):

    if A.size == 3 and B.size == 3:
        C = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                C[i,j] = A[i]*B[j]

    elif A.size == 9 and B.size == 9:
        C = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i,j,k,l] = A[i,j] * B[k,l]

    else:
        print('Matrices sizes mismatch')

    return C

def PlotStiffnessROI(ROI:np.array, StiffnessTensor:np.array, FileName:Path) -> None:

    """
    Plots a 3D ellipsoid representing a region of interest (ROI) with scaling based on the
    eigenvalues and eigenvectors provided. The ellipsoid is overlaid on a binary structure mesh,
    and the plot is generated with the ability to visualize the MIL (Mean Intercept Length) values.

    Parameters:
    -----------
    ROI (3D array): A 3D binary array representing the region of interest (ROI).
        
    eValues (1D array): A 1D array containing the eigenvalues of the fabric.
        
    eVectors (3D array) : A 2D array (shape: 3x3) containing the eigenvectors of the fabric.
        
    Returns:
    --------
    None
    """

    # Create a unit sphere and transform it to an ellipsoid
    Sphere = pv.Sphere(radius=1, theta_resolution=50, phi_resolution=50)

    # Compute elongation and bulk modulus
    I = np.eye(3)
    ElongationModulus = np.zeros(Sphere.points.shape)
    BulkModulus = np.zeros(len(Sphere.points))
    for p, Point in enumerate(Sphere.points):
        N = DyadicProduct(Point, Point)
        SN = Transform(StiffnessTensor, N)
        ElongationModulus[p] = FrobeniusProduct(N, SN) * Point
        BulkModulus[p] = FrobeniusProduct(I, SN)

    # # Scale the sphere by the square roots of the eigenvalues
    # Scale = ROI.shape[0]/2 / max(np.linalg.norm(ElongationModulus, axis=1))
    # Points = ElongationModulus * Scale

    # # Center the ellipsoid at the structure's midpoint
    # Offset = np.array(ROI.shape) / 2
    # EllispoidPoints = Points + Offset
    Ellispoid = pv.PolyData(ElongationModulus, Sphere.faces)
    Ellispoid['Bulk Modulus'] = BulkModulus

    # Plotting
    SArgs = dict(font_family='times', 
                 width=0.05,
                 height=0.75,
                 vertical=True,
                 position_x=0.85,
                 position_y=0.125,
                 title_font_size=30,
                 label_font_size=20
                 )
    
    BArgs = dict(font_family='times', 
                 font_size=30,
                 location='back',
                 n_xlabels=3,
                 n_ylabels=3,
                 n_zlabels=3,
                 all_edges=True,
                 fmt='%i',
                 xtitle='',
                 ytitle='',
                 ztitle='',
                 use_3d_text=False
                 )
    
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(Ellispoid, scalars='Bulk Modulus', cmap='jet', scalar_bar_args=SArgs)
    pl.camera_position = 'xz'
    pl.camera.roll = 0
    pl.camera.elevation = 30
    pl.camera.azimuth = 30
    pl.camera.zoom(0.8)
    pl.add_axes(viewport=(0,0,0.25,0.25),
                label_size=(0.065, 0.065),
                xlabel='e1',
                ylabel='e2',
                zlabel='e3')
    pl.show_bounds(**BArgs)
    pl.screenshot(FileName, return_img=False, scale=2)
    # pl.show()

    return


#%% Main

def Main(Arguments):

    ScanPath = Path(__file__).parents[1] / '00_Data'
    ResultsPath = Path(__file__).parents[1] / '02_Results'
    Data = pd.read_csv(ResultsPath / 'Parameters.csv', sep=';')
    del Data['Unnamed: 7']
    Otsu = int(round(Data['$Threshold'].mean()))

    GroupedData = Data.groupby('$Sample')

    List = [F.stem for F in Path(ResultsPath / 'Scans').iterdir()]

    for Sample, SampleData in GroupedData:

        if Sample in List:
            pass
        else:

            # Read scan
            Time.Process(1, 'Read ' + Sample)
            VoxelModel, AdditionalData = ReadISQ(ScanPath / (Sample + '.ISQ'), ASCII=False)
            VoxelModel = VoxelModel.astype(float)

            # Scale scan values for plotting
            Scaled = VoxelModel / Otsu

            # Plot using pyvista
            Time.Update(2/5,'Plot scan')
            pl = pv.Plotter(off_screen=True)
            actors = pl.add_volume(Scaled[::5,::5,::5].T,
                        cmap='bone',
                        show_scalar_bar=False,
                        opacity='sigmoid_5')
            actors.prop.interpolation_type = 'linear'
            pl.camera_position = 'xz'
            pl.camera.azimuth = 0
            pl.camera.elevation = 30
            pl.camera.roll = 0
            pl.camera.zoom(1.2)
            pl.screenshot(ResultsPath / 'Scans' / (Sample + '.png'), return_img=False)

            Time.Update(4/5,'Plot ROIs')
            for _, Row in SampleData.iterrows():
                
                X = Row['$XPos']
                Y = Row['$YPos']
                Z = Row['$ZPos']
                Dim = Row['$Dim']

                ROI = Scaled[Z:Z+Dim,Y:Y+Dim,X:X+Dim].T
                Name = ResultsPath / 'Scans' / (Sample + '_' + str(Row['$ROI']) + '.png')

                pl = pv.Plotter(off_screen=True)
                actors = pl.add_volume(ROI,
                            cmap='bone',
                            show_scalar_bar=False,
                            opacity='sigmoid_5')
                actors.prop.interpolation_type = 'linear'
                pl.camera_position = 'xz'
                pl.camera.roll = 0
                pl.camera.elevation = 30
                pl.camera.azimuth = 30
                pl.camera.zoom(1.0)
                pl.screenshot(Name, return_img=False)

            #     # # Read ROI segmented and cleaned ROI
            #     Name = ResultsPath / 'ROIs' / (Sample + '_' + str(Row['$ROI']) + '.mhd')
            #     ROI = sitk.ReadImage(str(Name))
            #     ROI = sitk.GetArrayFromImage(ROI).T

            #     # pl = pv.Plotter(off_screen=True)
            #     # actors = pl.add_volume(ROI,
            #     #             cmap='bone',
            #     #             show_scalar_bar=False, opacity=[0,1,1])
            #     # actors.prop.interpolation_type = 'linear'
            #     # pl.camera_position = 'xz'
            #     # pl.camera.roll = 0
            #     # pl.camera.elevation = 30
            #     # pl.camera.azimuth = 30
            #     # pl.camera.zoom(1.0)
            #     # pl.screenshot(str(Name)[:-4] + '.png', return_img=False)
            #     # # pl.show()

            #     # Plot fabric
            #     Name = ResultsPath / 'Morphometry' / (Sample + '_' + str(Row['$ROI']) + '.csv')
            #     eValues, eVectors = GetFabric(Name)
            #     FileName = str(Name)[:-4] + '.png'
            #     PlotFabricROI(ROI, eValues, eVectors, FileName)

            #     # Plot stiffness
            #     Abaqus = open(ResultsPath / 'Abaqus' / (Sample + '_' + str(Row['$ROI']) + '.out'), 'r').readlines()
            #     Strain = np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002])

            #     Stress = np.zeros((6,6))
            #     for i in range(6):
            #         for j in range(6):
            #             Stress[i,j] = float(Abaqus[i+4].split()[j+1])

            #     Stiffness = np.zeros((6,6))
            #     for i in range(6):
            #         for j in range(6):
            #             Stiffness[i,j] = Stress[i,j] / Strain[i]

            #     # Symetrize matrix
            #     Stiffness = 1/2 * (Stiffness + Stiffness.T)

            #     # Write tensor into mandel notation
            #     Mandel = Engineering2MandelNotation(Stiffness)

            #     # Step 3: Transform tensor into fabric coordinate system
            #     I = np.eye(3)
            #     Q = np.array(eVectors)
            #     S4 = IsoMorphism66_3333(Mandel)
            #     TS4 = TransformTensor(S4, I, Q)
            #     PlotStiffnessROI(ROI, TS4, 'Test.png')

            Time.Process(0,f'Done {Sample}')

    return



if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--OutputPath', help='Output path for the plots', default=Path(__file__).parents[1] / '02_Results')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

#%%

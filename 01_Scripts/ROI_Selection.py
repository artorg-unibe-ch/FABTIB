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
from skimage.filters import threshold_otsu

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

def ROICoords(VoxelModel, AddData, Dim, N):

    # Compute ROI Z positions
    Height = (AddData['DimSize'][2] - Dim) # Sample height accounting for ROI size
    Step = Height / (N-1)                 # Z distance between ROIs
    Zc = np.arange(N) * Step + Dim /2

    # Compute center of mass in X and Y for each Z
    Coords = []
    X = np.arange(VoxelModel.shape[2])
    Y = np.arange(VoxelModel.shape[1])
    for i in range(N):
        ZStart = int(Zc[i] - Dim/2)
        Stop = int(Zc[i] + Dim/2)
        CropedModel = VoxelModel[ZStart:Stop]

        Xc = np.sum(CropedModel, axis=(0,1)) * X
        Xc = sum(Xc) / np.sum(CropedModel)
        Yc = np.sum(CropedModel, axis=(0,2)) * Y
        Yc = sum(Yc) / np.sum(CropedModel)

        XStart = int(round(Xc - Dim/2))
        YStart = int(round(Yc - Dim/2))

        Coords.append((XStart, YStart, ZStart))

    return Coords


#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputISQ:
        InputISQs = [Arguments.InputISQ]
    else:
        DataPath = Path(__file__).parents[1] / '00_Data'
        InputISQs = [F for F in Path.iterdir(DataPath) if F.name.endswith('.ISQ')]

    # Create output directory if necessary
    Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)

    # Create csv file for coordinates
    Variables = ['$Sample','$ROI','$XPos','$YPos','$ZPos','$Dim','$Threshold']
    Data = pd.DataFrame(columns=Variables)

    for iISQ, ISQ in enumerate(InputISQs):

        # Read scan
        Time.Process(1, 'Read ' + ISQ.name[:-4])
        VoxelModel, AdditionalData = ReadISQ(ISQ, ASCII=False)
        VoxelModel = VoxelModel.astype(float)

        # Compute Otsu threshold to segment ROIS
        Time.Update(1/5,'Compute Otsu')
        Otsu = threshold_otsu(VoxelModel)

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
        pl.screenshot(Path(Arguments.OutputPath) / (Path(ISQ).name[:-4] + '.png'))

        # Select ROI at center of gravity
        Time.Update(3/5,f'Select {Arguments.NROIs} ROIs')
        ROISize = 5.3   # ROI side length in mm
        Dim = int(round(ROISize // AdditionalData['ElementSpacing'][0]))
        Coords = ROICoords(VoxelModel, AdditionalData, Dim, N=Arguments.NROIs)
        
        Time.Update(4/5,f'Plot {Arguments.NROIs} ROIs')
        for i, C in enumerate(Coords):
            Index = iISQ * Arguments.NROIs + i
            Data.loc[Index,'$Sample'] = ISQ.name[:-4]
            Data.loc[Index,'$ROI'] = i+1
            Data.loc[Index,'$XPos'] = C[0]
            Data.loc[Index,'$YPos'] = C[1]
            Data.loc[Index,'$ZPos'] = C[2]
            Data.loc[Index,'$Dim'] = Dim
            Data.loc[Index,'$Threshold'] = round(Otsu)

            ROI = Scaled[C[2]:C[2]+Dim,C[1]:C[1]+Dim,C[0]:C[0]+Dim].T
            Name = Arguments.OutputPath / (ISQ.name[:-4] + '_' + str(i+1) + '.png')

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
            pl.screenshot(Name)
    
        # Update time
        Time.Process(0, f'Done ISQ {iISQ+1} / {len(InputISQs)}')

    Data.to_csv(Arguments.OutputPath.parent / 'Parameters.csv',
                index=False, sep=';', line_terminator=';\n')


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputISQ', help='File name of the ISQ scan', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI and png image of the plot', default=Path(__file__).parents[1] / '02_Results/Scans')
    Parser.add_argument('--NROIs', help='Number of region of interests to extract', type=int, default=3)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

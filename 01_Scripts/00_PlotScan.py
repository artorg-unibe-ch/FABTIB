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

import argparse
import pyvista as pv
from pathlib import Path
from Utils import ReadISQ, Time

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

    for ISQ in InputISQs:
        
        # Read scan
        VoxelModel, AdditionalData = ReadISQ(ISQ, ASCII=False)
        VoxelModel = VoxelModel.astype(float)

        # Scale values
        Max = VoxelModel.max()
        Min = VoxelModel.min()
        Scaled = (VoxelModel - Min) / (Max - Min)

        # Plot using pyvista
        Time.Process(1, 'Plot ' + ISQ.name[:-4])
        pl = pv.Plotter(off_screen=True)
        actors = pl.add_volume(Scaled[::2,::2,::2].T,
                    cmap='bone',
                    show_scalar_bar=False,
                    opacity=[0, 0.02, 0.2, 0.4, 1]
                    )
        actors.prop.interpolation_type = 'linear'
        pl.camera_position = 'xz'
        pl.camera.azimuth = 0
        pl.camera.elevation = 30
        pl.camera.roll = 0
        pl.camera.zoom(1.2)
        pl.screenshot(Path(Arguments.OutputPath) / (Path(ISQ).name[:-4] + '.png'))
        Time.Process(0)
        pl.show()

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputISQ', help='File name of the ISQ scan', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the png image of the plot', type=str, default='02_Results/ScanPlots')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

    #%%

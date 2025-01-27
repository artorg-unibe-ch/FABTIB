#%% !/usr/bin/env python3

Description = """
Read .npy ROI and compute fabric using mean intercept length
"""

__author__ = ['Mathieu Simon']
__date_created__ = '28-10-2024'
__date__ = '06-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pyvista as pv
from numba import njit
from Utils import Time
from pathlib import Path

#%% Functions

def Octahedron() -> np.array:
    
    """
    Initialize the vertices and faces of an octahedron.
    """
    
    Vertices = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    Faces = [
        (Vertices[0], Vertices[2], Vertices[4]),
        (Vertices[0], Vertices[2], Vertices[5]),
        (Vertices[0], Vertices[3], Vertices[4]),
        (Vertices[0], Vertices[3], Vertices[5]),
        (Vertices[1], Vertices[2], Vertices[4]),
        (Vertices[1], Vertices[2], Vertices[5]),
        (Vertices[1], Vertices[3], Vertices[4]),
        (Vertices[1], Vertices[3], Vertices[5])
    ]
    return Faces

def SplitTriangle(Triangle:np.array) -> list:
    
    """
    Split a triangle into four smaller triangles.
    """
    
    P1, P2, P3 = Triangle
    
    # Midpoints of each edge
    P4 = tuple((np.array(P1) + np.array(P2)) / 2)
    P5 = tuple((np.array(P2) + np.array(P3)) / 2)
    P6 = tuple((np.array(P1) + np.array(P3)) / 2)
    
    return [(P1, P4, P6), (P4, P2, P5), (P4, P5, P6), (P5, P3, P6)]

def SubdivideFaces(Faces:np.array, Subdivisions:int) -> np.array:
    
    """
    Recursively subdivide each triangle face into smaller triangles.
    """
    
    for _ in range(Subdivisions):
        NewFaces = []
        for Triangle in Faces:
            NewFaces.extend(SplitTriangle(Triangle))
        Faces = NewFaces
    return Faces

def Project2UnitSphere(Face:np.array) -> np.array:
    
    """
    Project each vertex of a face onto the unit sphere.
    """

    UnitFace = []
    for Point in Face:
        X, Y, Z = Point
        Norm = (X**2 + Y**2 + Z**2)**0.5
        UnitFace.append((X/Norm, Y/Norm, Z/Norm))
        
    return UnitFace

def AreaAndCentroid(Face:np.array) -> np.array:
    
    """
    Calculate the area and centroid of an face.
    """
    
    P1, P2, P3 = map(np.array, Face)
    
    # Calculate the vectors for two edges
    Edge1 = P2 - P1
    Edge2 = P3 - P1
    
    # Area calculation (cross product norm gives twice the area of the triangle)
    Normal = np.cross(Edge1, Edge2)
    Area = 0.5 * np.sum(Normal**2)**0.5
    
    # Centroid of the triangle
    Centroid = (P1 + P2 + P3) / 3
    Centroid = Project2UnitSphere([Centroid])[0]  # Project the centroid to the unit sphere
    
    return Area, Centroid

def GenerateDirections(Subdivisions=2) -> np.array:
    
    """
    Generate directions by subdividing an octahedron and projecting to a sphere.
    """
    
    # Initialize the octahedron and subdivide
    OctaFaces = Octahedron()
    Faces = SubdivideFaces(OctaFaces, Subdivisions)
    
    # Normalize each face to project onto the unit sphere
    UnitFaces = []
    for Face in Faces:
        UnitFaces.append(Project2UnitSphere(Face))
    
    # Calculate area and centroid of each face
    Directions = []
    Areas = []
    for Face in UnitFaces:
        Area, Centroid = AreaAndCentroid(Face)
        Directions.append(Centroid)
        Areas.append(Area)
    
    # Normalize areas so they sum up to the sphere's surface area (4π for unit sphere)
    Factor = 4 * np.pi / sum(Areas)
    Areas = np.array([A * Factor for A in Areas])
    
    return np.array(Directions), Areas

# Numba functions
@njit
def GenerateVoxelRay(Direction:np.array, X:int, Y:int, Z:int) -> np.array:
    
    """
    Generate the path of voxel coordinates traversed by a ray within a 3D grid.
    """
    
    # Unpack direction components
    XDir, YDir, ZDir = Direction
    XSign = 1 if XDir >= 0 else -1
    YSign = 1 if YDir >= 0 else -1
    ZSign = 1 if ZDir >= 0 else -1
    
    # Initialize voxel position and allocate array for VoxelRay
    VoxX, VoxY, VoxZ = 0, 0, 0
    VoxelRay = [(VoxX, VoxY, VoxZ)]

    # Compute tMax and deltaT for each axis
    MaxStepsX = (1 / abs(XDir)) if XDir != 0 else np.inf
    MaxStepsY = (1 / abs(YDir)) if YDir != 0 else np.inf
    MaxStepsZ = (1 / abs(ZDir)) if ZDir != 0 else np.inf
    
    DeltaX = abs(1 / XDir) if XDir != 0 else np.inf
    DeltaY = abs(1 / YDir) if YDir != 0 else np.inf
    DeltaZ = abs(1 / ZDir) if ZDir != 0 else np.inf
    
    # Determine the primary direction
    if abs(XDir) >= abs(YDir) and abs(XDir) >= abs(ZDir):
        PrimDir = 0
    elif abs(YDir) >= abs(XDir) and abs(YDir) >= abs(ZDir):
        PrimDir = 1
    else:
        PrimDir = 2
        
    # Iteratively build the array
    for _ in range(max(X, Y, Z)):
        if MaxStepsX < MaxStepsY:
            if MaxStepsX < MaxStepsZ:
                VoxX += int(XSign)
                MaxStepsX += DeltaX
            else:
                VoxZ += int(ZSign)
                MaxStepsZ += DeltaZ
        elif MaxStepsY < MaxStepsZ:
            VoxY += int(YSign)
            MaxStepsY += DeltaY
        else:
            VoxZ += int(ZSign)
            MaxStepsZ += DeltaZ

        # If voxel is in array boundaries
        if 0 <= VoxX < X and 0 <= VoxY < Y and 0 <= VoxZ < Z:
            if (PrimDir == 0 and VoxX != VoxelRay[-1][0]) or \
               (PrimDir == 1 and VoxY != VoxelRay[-1][1]) or \
               (PrimDir == 2 and VoxZ != VoxelRay[-1][2]):
                VoxelRay.append((VoxX, VoxY, VoxZ))
    
    return np.array(VoxelRay)

@njit
def NumpyDot(Points:np.array, Vector:np.array) -> np.array:
    
    """
    Computes projections of points onto direction vector.
    """
    
    n = Points.shape[0]
    m = Vector.shape[0]
    Proj = np.empty((n, m), dtype=np.float64)
    
    # Loop through each point and each vector to compute the dot products
    for i in range(n):
        for j in range(m):
            Proj[i, j] = Points[i, 0] * Vector[j, 0] + \
                        Points[i, 1] * Vector[j, 1] + \
                        Points[i, 2] * Vector[j, 2]

    return Proj

@njit
def NumbaGrid(Array1:np.array, Array2:np.array, Vector1:np.array, Vector2:np.array) -> np.array:
    
    """
    Creates a 2D grid of points by linearly combining elements of
    Array1 and Array2 with Vector1 and Vector2.
    
    Parameters:
        Array1 (1D array): Array of values along the Vector1 direction.
        Array2 (1D array): Array of values along the Vector2 direction.
        Vector1 (1D array): 3-element array representing the first direction vector.
        Vector2 (1D array): 3-element array representing the second direction vector.
    
    Returns:
        2D array: Grid of combined points with shape (len(Array1) * len(Array2), 3).
    """
    
    # Get arrays lengths
    Len1 = len(Array1)
    Len2 = len(Array2)
    
    # Initialize the result array to hold all points in the grid
    Grid = np.empty((Len1 * Len2, 3), dtype=np.float64)
    
    Index = 0
    for i in range(Len1):
        for j in range(Len2):
            
            # Compute the linear combination for each grid point
            Grid[Index, 0] = Array1[i] * Vector1[0] + Array2[j] * Vector2[0]
            Grid[Index, 1] = Array1[i] * Vector1[1] + Array2[j] * Vector2[1]
            Grid[Index, 2] = Array1[i] * Vector1[2] + Array2[j] * Vector2[2]
            Index += 1
    
    return Grid

@njit
def NumbaComputeIntersections(Grid:np.array, Lengths:np.array, Vector:np.array) -> np.array:
    
    """
    Computes intersections of Grid points along the Vector, scaled by Lengths.
    
    Parameters:
        Grid (2D array): Array of points on the grid, shape (n, 3).
        Lengths (2D array): Scaling factors for each point along each axis, shape (n, 3).
        Vector (1D array): Direction vector, shape (3,).
    
    Returns:
        3D array: Intersections, shape (n, 3, 3).
    """
    
    n = Grid.shape[0]
    Intersections = np.empty((3*n, 3), dtype=np.float64)
    
    # Loop through each grid point and each plane direction to compute intersections
    for i in range(n):
        for j in range(3):  # Assuming Lengths has 3 entries per point
            Intersections[i + j*n, 0] = Grid[i, 0] + Lengths[i, j] * Vector[0]
            Intersections[i + j*n, 1] = Grid[i, 1] + Lengths[i, j] * Vector[1]
            Intersections[i + j*n, 2] = Grid[i, 2] + Lengths[i, j] * Vector[2]
    
    return Intersections

@njit
def NumbaMIL(Array:np.array, Directions:np.array, StepSize:int) -> np.array:

    """
    Function to calculate the Mean Intercept Length (MIL) for 
    given directions in a 3D binary array. The calculation considers
    different symmetric ray orientations (for positive and negative axes).

    Parameters:
        Array (3D array): Binary array of a structure to analyze.
        Directions (2D array): Direction vectors to compute the MIL.
        StepSize (int): Distance in voxels between rays.
    
    Returns:
        MIL (1D array): Mean intercept length values.
        MILDir (2D array): Directions corresponding to MIL values.
    """

    # Get the shape of the input array to define the grid size
    Z, Y, X = np.array(Array.shape)
    
    # Define the 8 corners of the 3D grid (bounding box)
    Corners = np.array([[0, 0, 0], [X, 0, 0], [0, Y, 0], [0, 0, Z],
                        [X, Y, 0], [X, 0, Z], [0, Y, Z], [X, Y, Z]])
    
    # Define the 3 planes for projecting the grid onto (for use with cross-products)
    Planes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Initialize arrays to store the MIL values and the direction vectors for each ray direction
    MIL = np.empty(len(Directions)*2*4, dtype=np.float64)  # Mean Intercept Length values
    MILDir = np.empty((len(Directions)*2*4, 3), dtype=np.float64)  # Direction vectors
    MILIndex = 0  # Index for tracking where to store results

    # Loop through each direction in the provided Directions list
    for D in Directions:
        
        # Generate the voxel ray for the current direction
        VoxelRay = GenerateVoxelRay(D, X, Y, Z)

        # Loop through the 4 symmetries (for each axis direction and its inverse)
        for Sym in range(4):
            
            # Copy the original voxel ray for the symmetry adjustment
            SymRay = VoxelRay.copy()

            # Define the direction for each symmetry
            if Sym == 0:
                SymD = np.array([D[0], D[1], D[2]])  # Original direction
            elif Sym == 1:
                SymD = np.array([-D[0], D[1], D[2]])  # Reflect across the X-axis
                SymRay[:,0] = X - VoxelRay[:,0] - 1   # Adjust the ray positions accordingly
            elif Sym == 2:
                SymD = np.array([D[0], -D[1], D[2]])  # Reflect across the Y-axis
                SymRay[:,1] = Y - VoxelRay[:,1] - 1   # Adjust the ray positions accordingly
            elif Sym == 3:
                SymD = np.array([D[0], D[1], -D[2]])  # Reflect across the Z-axis
                SymRay[:,2] = Z - VoxelRay[:,2] - 1   # Adjust the ray positions accordingly

            # Cross-product to find orthogonal vectors for ray projection
            eZ = np.cross(SymD, np.array([0.0, 0.0, 1.0]))
            e1 = np.cross(SymD, eZ)
            e1 = e1 / np.sum(e1**2)**0.5  # Normalize the e1 vector
            e2 = np.cross(SymD, e1)
            e2 = e2 / np.sum(e2**2)**0.5  # Normalize the e2 vector

            # Project the grid corners onto the e1 and e2 directions
            e1Proj = NumpyDot(Corners, np.reshape(e1, (1, 3)))
            e2Proj = NumpyDot(Corners, np.reshape(e2, (1, 3)))

            # Find the min and max values along e1 and e2 projections
            e1Min, e1Max = e1Proj.min(), e1Proj.max()
            e2Min, e2Max = e2Proj.min(), e2Proj.max()

            # Define the range of values for e1 and e2 directions, with the specified step size
            e1Val = np.arange(e1Min, e1Max + 1, StepSize)
            e2Val = np.arange(e2Min, e2Max + 1, StepSize)

            # Create a grid based on the ranges of e1 and e2
            Grid = NumbaGrid(e1Val, e2Val, e1, e2)

            # Calculate the distances to the planes for each grid point
            Num = -NumpyDot(Grid, Planes.T)
            Denom = NumpyDot(np.reshape(SymD, (1, 3)), Planes.T)
            DLengths = Num / Denom  # Ray lengths to reach each plane

            # Compute the intersections of the ray with the grid
            Entries = NumbaComputeIntersections(Grid, DLengths, SymD)

            # Initialize variables to track the intercepts and total lengths
            Intercepts = 0.0
            Lengths = 0.0

            # Loop through the entries (intersections) to calculate intercepts and lengths
            for Entry in Entries:
                
                # Round the entry points and convert to integer indices
                Entry = np.array([round(E) for E in Entry], np.int32)
                
                # Compute the ray path from the entry point, adjusted for symmetry
                Ray = Entry + SymRay
                
                # Filter out rays that are outside the grid boundaries
                Ray = Ray[(Ray[:, 0] >= 0) & (Ray[:, 1] >= 0) & (Ray[:, 2] >= 0) & 
                          (Ray[:, 0] < X) & (Ray[:, 1] < Y) & (Ray[:, 2] < Z)]

                # Skip if no valid rays
                if Ray.size == 0:
                    continue

                # Initialize the values array for ray intensities or grid values
                Values = np.empty(len(Ray) + 2, np.int32)
                Values[0] = 0  # Set to 0 (no bone outside ray)
                Values[-1] = 0  # Set to 0 (no bone outside ray)

                # Fill in the intensity values for each ray point
                for i in range(1, len(Values) - 1):
                    Values[i] = Array[Ray[i-1, 2], Ray[i-1, 1], Ray[i-1, 0]]

                # Calculate the difference (intercept) between consecutive values
                RayIntercept = Values[1:] - Values[:-1]

                # Find the start and end points of the ray segments
                Starts = Ray[RayIntercept[:-1] == 1]
                Ends = Ray[RayIntercept[1:] == -1]

                # Calculate the segment lengths (Euclidean distance between starts and ends)
                SegmentLengths = np.sum((Ends - Starts)**2, axis=1)**0.5

                # Count the number of intercepts and sum the lengths of valid segments
                Intercepts += sum(SegmentLengths > 1)
                Lengths += sum(SegmentLengths)

            # Store the MIL value for the current ray direction
            MIL[MILIndex] = Lengths / Intercepts
            MILDir[MILIndex] = SymD
            MILIndex += 1

            # Store the negative of the direction (for symmetry)
            MIL[MILIndex] = Lengths / Intercepts
            MILDir[MILIndex] = -SymD
            MILIndex += 1

    return MIL, MILDir

def ComputeMIL(Array:np.array, Subdivisions=2, StepSize=5) -> np.array:

    """
    Function used to compute the mean intercept length of a binary structure

    Parameters:
        Array (3D array): Binary array of a structure to analyze.
        Subdivisions (int): Number of subdivision of the initial octahedron.
        StepSize (int): Distance in voxels between rays.
    
    Returns:
        MIL (1D array): Mean intercept length values.
        MILDir (2D array): Directions corresponding to MIL values.
    """

    # Generate directions to compute MIL
    Directions, Areas = GenerateDirections(Subdivisions)

    # Filter directions to keep only 1 quadrant
    Areas = Areas[(Directions > 0).all(axis=1)]
    Directions = Directions[(Directions > 0).all(axis=1)]

    # Compute MIL using numba
    MIL, MILDir = NumbaMIL(Array, Directions, StepSize)

    return MIL, MILDir

def FitFabricTensor(MIL:np.array, Directions:np.array) -> np.array:
    
    """
    Ellipsoidal fit of the fabric tensor M directly from MIL values and directions using least squares.
    See T. P. HARRIGAN, R. W. MANN
    Characterization of microstructural anisotropy in orthotropic materials using a second rank tensor
    Journal of materials science 19 (1984) 761-767
    
    Parameters:
    - MIL (1D array): Array of MIL values for each direction (shape N).
    - Directions (2D array): Array of unit vectors for each direction (shape N x 3).
    
    Returns:
    - Fitted fabric tensor M (3x3 symmetric matrix).
    """
    
    
    # Create the design matrix A (N x 6)
    N = len(MIL)
    A = np.zeros((N, 6))
    for i in range(N):
        v1, v2, v3 = Directions[i]
        A[i] = [v1**2, v2**2, v3**2, np.sqrt(2)*v1*v2, np.sqrt(2)*v1*v3, np.sqrt(2)*v2*v3]
    
    # The target vector b (MIL values squared inversely)
    b = 1/MIL**2

    # Solve the normal equation: (A.T @ A) v = A.T @ b
    AT_A = np.dot(np.transpose(A), A)
    AT_b = np.dot(np.transpose(A), b)
    v = np.dot(np.linalg.inv(AT_A), AT_b)

    # Construct the fabric tensor from the solution vector v
    M = np.array([[v[0], v[5] / np.sqrt(2), v[4] / np.sqrt(2)],
                  [v[5] / np.sqrt(2), v[1], v[3] / np.sqrt(2)],
                  [v[4] / np.sqrt(2), v[3] / np.sqrt(2), v[2]]])
    
    # Compute eigen values and eigen vectors
    eValues, eVectors = np.linalg.eig(M)

    # Compute MIL eigen values (as they were squared inversely)
    eValues = 1.0 / np.sqrt(abs(eValues))

    # Scale MIL eigen values to sum up to 3
    eValues = eValues / sum(eValues) * 3.0

    # Build again fabric tensor
    M = np.zeros((3,3))
    for i in range(3):
        M += eValues[i] * np.outer(eVectors[i],eVectors[i])

    return M

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

    # Create the structure mesh from the binary array
    StructureMesh = pv.wrap(ROI)
    Structure = StructureMesh.contour([0.5])  # Isosurface at level 0.5

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
    pl.add_mesh(Ellispoid, scalars='MIL', cmap='jet', opacity=1, scalar_bar_args=sargs)
    pl.add_mesh(Structure, color=(0.87,0.91,0.91), opacity=1, show_edges=False)
    pl.camera_position = 'xz'
    pl.camera.roll = 0
    pl.camera.elevation = 30
    pl.camera.azimuth = 30
    pl.camera.zoom(1.0)
    pl.add_axes(viewport=(0,0,0.25,0.25))
    pl.screenshot(FileName)
    # pl.show()

    return

#%% Main

def Main(Arguments):

    # Read Arguments
    if Arguments.InputROI:
        InputROIs = [Arguments.InputROI]
    else:
        DataPath = Path(__file__).parents[1] / '02_Results/ROIs'
        InputROIs = sorted([F for F in Path.iterdir(DataPath) if F.name.endswith('.npy')])
    
    if Arguments.OutputPath:
        Path.mkdir(Path(Arguments.OutputPath), exist_ok=True)
    else:
        Path.mkdir(Path('FabricResults'), exist_ok=True)
        
    for i, ROI in enumerate(InputROIs):

        # Print time
        Time.Process(1,ROI.name[:-4])

        # Read scan
        Array = np.load(ROI)

        # Compute MIL
        Time.Update(1/2, 'Compute MIL')
        MIL, Directions = ComputeMIL(Array, Subdivisions=2, StepSize=5)

        # Fit fabric tensor
        M = FitFabricTensor(MIL, Directions)

        # Save result
        FName = Path(Arguments.OutputPath) / (ROI.name[:-4] + '.npy')
        np.save(FName, M)

        # Plot fabric with ROI
        Time.Update(1.0, 'Plot MIL')
        eValues, eVectors = np.linalg.eig(M)
        FName = Path(Arguments.OutputPath) / (ROI.name[:-4] + '.png')
        PlotFabricROI(Array, eValues, eVectors, FName)

        # Print time
        Time.Process(0,f'ROI {i+1}/{len(InputROIs)} done')
        
    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputROI', help='File name of the binary ROI', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI fabric and png image of the plot', type=str, default='02_Results/Fabric')

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

#%%

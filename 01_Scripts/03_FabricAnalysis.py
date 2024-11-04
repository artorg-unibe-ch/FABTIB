#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '28-10-2024'
__date__ = '29-10-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import argparse
import numpy as np
import pyvista as pv
import tensorflow as tf
from pathlib import Path
from Utils import ReadISQ, Time

#%% Functions

def initialize_octahedron():
    """Initialize the vertices and faces of an octahedron."""
    vertices = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    faces = [
        (vertices[0], vertices[2], vertices[4]),
        (vertices[0], vertices[2], vertices[5]),
        (vertices[0], vertices[3], vertices[4]),
        (vertices[0], vertices[3], vertices[5]),
        (vertices[1], vertices[2], vertices[4]),
        (vertices[1], vertices[2], vertices[5]),
        (vertices[1], vertices[3], vertices[4]),
        (vertices[1], vertices[3], vertices[5])
    ]
    return faces

def split_triangle(triangle):
    """Split a triangle into four smaller triangles."""
    P1, P2, P3 = triangle
    # Midpoints of each edge
    P4 = tuple((np.array(P1) + np.array(P2)) / 2)
    P5 = tuple((np.array(P2) + np.array(P3)) / 2)
    P6 = tuple((np.array(P1) + np.array(P3)) / 2)
    
    return [
        (P1, P4, P6),
        (P4, P2, P5),
        (P4, P5, P6),
        (P5, P3, P6)
    ]

def subdivide_faces(faces, subdivisions):
    """Recursively subdivide each triangle face into smaller triangles."""
    for _ in range(subdivisions):
        new_faces = []
        for triangle in faces:
            new_faces.extend(split_triangle(triangle))
        faces = new_faces
    return faces

def project_to_unit_sphere(point):
    """Normalize a point to lie on the unit sphere."""
    x, y, z = point
    norm = np.sqrt(x**2 + y**2 + z**2)
    return (x / norm, y / norm, z / norm)

def normalize_faces(faces):
    """Project each vertex of every face onto the unit sphere."""
    return [
        (project_to_unit_sphere(face[0]),
         project_to_unit_sphere(face[1]),
         project_to_unit_sphere(face[2]))
        for face in faces
    ]

def calculate_area_and_centroid(triangle):
    """Calculate the area and centroid of a triangle on the unit sphere."""
    P1, P2, P3 = map(np.array, triangle)
    
    # Calculate the vectors for two edges
    edge1 = P2 - P1
    edge2 = P3 - P1
    
    # Area calculation (cross product norm gives twice the area of the triangle)
    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    
    # Centroid of the triangle
    centroid = (P1 + P2 + P3) / 3
    centroid = project_to_unit_sphere(centroid)  # Project the centroid to the unit sphere
    
    return area, centroid

def generate_sphere_directions(subdivisions=2):
    """Generate directions by subdividing an octahedron and projecting to a sphere."""
    # Initialize the octahedron and subdivide
    faces = initialize_octahedron()
    faces = subdivide_faces(faces, subdivisions)
    
    # Normalize each face to project onto the unit sphere
    faces = normalize_faces(faces)
    
    # Calculate area and centroid of each face
    directions = []
    total_area = 0
    for face in faces:
        area, centroid = calculate_area_and_centroid(face)
        directions.append((centroid, area))
        total_area += area
    
    # Normalize areas so they sum up to the sphere's surface area (4π for unit sphere)
    scale_factor = 4 * np.pi / total_area
    directions = [(centroid, area * scale_factor) for centroid, area in directions]
    
    return directions

def rotate_to_align_with_direction(Points, Direction, Center):
    
    """
    Rotate corners of the binary array to align the z-axis with the given direction vector.
    
    Parameters:
    corners (np.ndarray): Array of corner points of the binary array.
    direction (np.ndarray): Direction vector to align with.
    
    Returns:
    np.ndarray: Rotated corner coordinates.
    """
    
    # Compute axis of rotation (cross product of z-axis and direction vector)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, Direction)
    rotation_angle = np.arccos(np.dot(z_axis, Direction))
    
    if np.allclose(rotation_axis, 0):
        return Points  # Already aligned with z-axis

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    ux, uy, uz = rotation_axis

    # Rodrigues' rotation formula for rotation matrix
    R = np.array([
        [cos_theta + ux*ux*(1 - cos_theta), uy*ux*(1 - cos_theta) - uz*sin_theta, uz*ux*(1 - cos_theta) + uy*sin_theta],
        [ux*uy*(1 - cos_theta) + uz*sin_theta, cos_theta + uy*uy*(1 - cos_theta), uz*uy*(1 - cos_theta) - ux*sin_theta],
        [ux*uz*(1 - cos_theta) - uy*sin_theta, uy*uz*(1 - cos_theta) + ux*sin_theta, cos_theta + uz*uz*(1 - cos_theta)]
    ])

    # Rotate corners around their center
    RPoints = np.einsum('ij,jk->ik', Points - Center, R)

    return RPoints + Center

def rotate_back_to_align_with_direction(Points, Direction, Center):
    
    """
    Rotate corners of the binary array to align the z-axis with the given direction vector.
    
    Parameters:
    corners (np.ndarray): Array of corner points of the binary array.
    direction (np.ndarray): Direction vector to align with.
    
    Returns:
    np.ndarray: Rotated corner coordinates.
    """
    
    # Compute axis of rotation (cross product of z-axis and direction vector)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, Direction)
    rotation_angle = np.arccos(np.dot(z_axis, Direction))
    
    if np.allclose(rotation_axis, 0):
        return Points  # Already aligned with z-axis

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_theta = np.cos(-rotation_angle)
    sin_theta = np.sin(-rotation_angle)
    ux, uy, uz = rotation_axis

    # Rodrigues' rotation formula for rotation matrix
    R = np.array([
        [cos_theta + ux*ux*(1 - cos_theta), uy*ux*(1 - cos_theta) - uz*sin_theta, uz*ux*(1 - cos_theta) + uy*sin_theta],
        [ux*uy*(1 - cos_theta) + uz*sin_theta, cos_theta + uy*uy*(1 - cos_theta), uz*uy*(1 - cos_theta) - ux*sin_theta],
        [ux*uz*(1 - cos_theta) - uy*sin_theta, uy*uz*(1 - cos_theta) + ux*sin_theta, cos_theta + uz*uz*(1 - cos_theta)]
    ])

    # Rotate corners around their center
    RPoints = np.einsum('ij,jk->ik', Points - Center, R)

    return RPoints + Center

import numpy as np

def dda_ray_trace(grid_shape, origin, direction):
    """
    Trace a ray through a 3D grid using the DDA algorithm.
    
    Parameters:
        grid_shape (tuple): Shape of the 3D grid (num_x, num_y, num_z).
        origin (array): Starting point of the ray (x0, y0, z0).
        direction (array): Ray direction vector (dx, dy, dz).
        
    Returns:
        List of traversed voxel indices and entry points.
    """
    # Grid dimensions
    nx, ny, nz = grid_shape
    
    # Bounding box limits
    bounds_min = np.array([0, 0, 0])
    bounds_max = np.array([nx - 1, ny - 1, nz - 1])
    
    # Check if the ray starts outside the grid and compute entry point if necessary
    t_min = 0.0
    t_max = np.inf
    
    # Iterate over each axis to compute intersection times with the bounding planes
    for i in range(3):
        if direction[i] != 0:
            # Time to hit the min and max boundary planes along the axis
            t1 = (bounds_min[i] - origin[i]) / direction[i]
            t2 = (bounds_max[i] - origin[i]) / direction[i]
            
            # Sort so t1 is the entry and t2 is the exit
            t_entry = min(t1, t2)
            t_exit = max(t1, t2)
            
            # Update global t_min and t_max to confine the ray within bounds
            t_min = max(t_min, t_entry)
            t_max = min(t_max, t_exit)
            
            # If t_min exceeds t_max, the ray does not intersect the grid
            if t_min > t_max:
                return []  # Ray misses the grid entirely
    
    # Calculate the entry point inside the grid
    entry_point = origin + t_min * direction

    # Convert entry point to voxel indices
    x, y, z = int(entry_point[0]), int(entry_point[1]), int(entry_point[2])

    # Set the sign of the direction for each axis
    sign_x = 1 if direction[0] > 0 else -1
    sign_y = 1 if direction[1] > 0 else -1
    sign_z = 1 if direction[2] > 0 else -1

    # Calculate distances to next voxel boundary
    if direction[0] != 0:
        tx_delta = abs(1 / direction[0])  # Distance to cross an x voxel
        tx = ((x + (sign_x > 0)) - entry_point[0]) / direction[0]
    else:
        tx = np.inf
        tx_delta = np.inf

    if direction[1] != 0:
        ty_delta = abs(1 / direction[1])  # Distance to cross a y voxel
        ty = ((y + (sign_y > 0)) - entry_point[1]) / direction[1]
    else:
        ty = np.inf
        ty_delta = np.inf

    if direction[2] != 0:
        tz_delta = abs(1 / direction[2])  # Distance to cross a z voxel
        tz = ((z + (sign_z > 0)) - entry_point[2]) / direction[2]
    else:
        tz = np.inf
        tz_delta = np.inf

    # List of traversed voxels
    path = [(x, y, z)]

    # Traverse the grid within bounds
    while 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
        # Determine the next boundary to cross and move accordingly
        if tx < ty and tx < tz:
            x += sign_x
            tx += tx_delta
        elif ty < tz:
            y += sign_y
            ty += ty_delta
        else:
            z += sign_z
            tz += tz_delta

        # Append the new voxel to the path
        path.append((x, y, z))

    return np.array(path)


def ComputeMIL(Array, Subdivision=2, StepSize=5):

    """
    Compute MIL
    """

    # Pad array to start and end ray in 0
    Array = np.pad(Array,1,constant_values=0)
    Z, Y, X = np.array(Array.shape) - 1

    # Define array corners
    Corners = np.array([[0, 0, 0], [X, 0, 0], [0, Y, 0], [0, 0, Z],
                        [X, Y, 0], [X, 0, Z], [0, Y, Z], [X, Y, Z]])
    
    # Define array planes (normal vectors and distances)
    Planes = np.array([[0, 0, 1],  # z = 0 (XY plane)
                       [0, 1, 0],  # y = 0 (XZ plane)
                       [1, 0, 0],  # x = 0 (YZ plane)
                       [0, 0, -1],  # z = nZ (far XY plane)
                       [0, -1, 0],  # y = nY (far XZ plane)
                       [-1, 0, 0]]) # x = nX (far YZ plane)
    
    PDist = np.array([0, 0, 0, Z, Y, X])  # Distances for near and far planes

    # Generate directions
    Directions = generate_sphere_directions(Subdivision)
    Directions = np.array([D[0] for D in Directions])
    Areas = np.array([D[1] for D in Directions])

    # Modifications to compare with Medtool
    # Directions = MedDir
    # Areas = np.array([Areas[tuple(n)] for n in Directions])

    # Keep directions in top quadrants (because of octahedron symetries)
    Areas = Areas[(Directions > 0).all(axis=1)]
    Directions = Directions[(Directions > 0).all(axis=1)]
    # Areas = Areas[Directions[:,2] > 0]
    # Directions = Directions[Directions[:,2] > 0]

    MIL = []
    MILDir = []
    for D in Directions:

        for Sym in range(4):
            Origin = Corners[Sym]
            if Sym == 0:
                SymD = np.array([D[0], D[1], D[2]])
            if Sym == 1:
                SymD = np.array([-D[0], D[1], D[2]])
            if Sym == 2:
                SymD = np.array([D[0], -D[1], D[2]])
            if Sym == 3:
                SymD = np.array([D[0], D[1], -D[2]])

            # Create an orthonormal basis with D as one of the axes
            e1 = np.cross(SymD, np.array([0.0, 0.0, 1.0]))
            e1 = e1 / np.sum(e1**2)**0.5
            e2 = np.cross(SymD, e1)
            e2 = e2 / np.sum(e2**2)**0.5

            # Project array corners onto the perpendicular plane directions
            e1Proj = np.matmul(Corners, e1)
            e2Proj = np.matmul(Corners, e2)

            # Determine min and max bounds for the perpendicular plane
            e1Min, e1Max = e1Proj.min(), e1Proj.max()
            e2Min, e2Max = e2Proj.min(), e2Proj.max()

            # Create grid of starting points across the perpendicular plane
            e1Pos = np.arange(0, e1Max, StepSize)
            e1Neg = np.arange(-StepSize, e1Min, -StepSize)
            e1Val = np.hstack([e1Neg[::-1], e1Pos])

            e2Pos = np.arange(0, e2Max, StepSize)
            e2Neg = np.arange(-StepSize, e2Min, -StepSize)
            e2Val = np.hstack([e2Neg[::-1], e2Pos])
            
            e1Grid, e2Grid = np.meshgrid(e1Val, e2Val, indexing='ij')
            Grid = e1Grid[..., np.newaxis] * e1 + e2Grid[..., np.newaxis] * e2
            Grid = Grid.reshape(-1, 3) + Origin

            # Find intersections with the box planes
            Num = -(np.dot(Grid, Planes.T) + PDist)
            Denom = np.dot(SymD, Planes.T)
            DLengths = Num / Denom  # Shape: (num_rays, num_planes)

            # Calculate intersection points
            Intersections = np.expand_dims(Grid, axis=1) + DLengths[..., np.newaxis] * SymD  # Shape: (num_rays, num_planes, 3)

            # Separate entry and exit points based on t values
            if Sym == 0:
                 # Intersections with near planes
                Entries = np.vstack([Intersections[:, 0],
                                     Intersections[:, 1],
                                     Intersections[:, 2]])
                # Intersections with far planes
                Exits = np.vstack([Intersections[:, 3],
                                   Intersections[:, 4],
                                   Intersections[:, 5]])
            if Sym == 1:
                 # Intersections with near planes
                Entries = np.vstack([Intersections[:, 0],
                                     Intersections[:, 4],
                                     Intersections[:, 5]])
                # Intersections with far planes
                Exits = np.vstack([Intersections[:, 3],
                                   Intersections[:, 1],
                                   Intersections[:, 2]])
            if Sym == 2:
                 # Intersections with near planes
                Entries = np.vstack([Intersections[:, 3],
                                     Intersections[:, 1],
                                     Intersections[:, 5]])
                # Intersections with far planes
                Exits = np.vstack([Intersections[:, 0],
                                   Intersections[:, 4],
                                   Intersections[:, 2]])
            if Sym == 3:
                 # Intersections with near planes
                Entries = np.vstack([Intersections[:, 3],
                                     Intersections[:, 4],
                                     Intersections[:, 2]])
                # Intersections with far planes
                Exits = np.vstack([Intersections[:, 0],
                                   Intersections[:, 1],
                                   Intersections[:, 5]])
                
            # Generate rays
            Rays = Exits - Entries

            # Keep ray having the shortest path between 2 planes
            RayLength = np.sum(Rays**2, axis=1)**0.5
            Shortest = min(RayLength)
            Shortest = np.isclose(Shortest, RayLength, rtol=1e-6)

            # Select the corresponding entry and exit points for the shortest rays
            Entries = Entries[Shortest]
            RayLength = min(RayLength[Shortest])

            RayStep = min(1 / np.abs(SymD))
            NSteps = np.round(RayLength / RayStep).astype(int) + 1
            Steps = np.linspace(0, RayLength, NSteps+1).reshape(-1, 1) * SymD

            # Iterate through each entry points
            Intercepts = np.zeros(len(Entries))
            Lengths = np.zeros(len(Entries))
            for i, Entry in enumerate(Entries):

                # Generate ray
                Ray = np.round(Steps + Entry).astype(int)
                # Ray = dda_ray_trace(Array.shape, Entry, D)
                # if len(Ray) == 0:
                #     continue

                # Filter points within bounds of binary array
                Points = Ray[(Ray[:, 0] >= 0) & (Ray[:, 1] >= 0) & (Ray[:, 2] >= 0) &
                            (Ray[:, 0] <= X) & (Ray[:, 1] <= Y) & (Ray[:, 2] <= Z)]

                # Skip if no valid points
                if Points.size == 0:
                    continue

                # Get array values
                Values = Array[Points[:,0], Points[:,1], Points[:,2]]

                # Count intercepts
                RayIntercept = Values[1:] - Values[:-1]
                    
                # Identify the start and end of each segment of 1s
                Starts = Points[:-1][RayIntercept == 1]
                Ends = Points[:-1][RayIntercept == -1]

                # Calculate lengths of segments
                SegmentLengths = np.sum((Ends - Starts)**2, axis=1)**0.5

                # Store number of intercepts and length
                Intercepts[i] += len(SegmentLengths)
                Lengths[i] += np.sum(SegmentLengths)
        
            MILDir.append(SymD)
            MIL.append(sum(Lengths) / sum(Intercepts))

    # Complete for full values
    MIL = np.concatenate([MIL,MIL])
    MILDir = np.vstack([MILDir, -np.array(MILDir)])

    # Args = []
    # for Dir in MedDir:
    #     Args.append(np.where((MILDir == Dir).all(axis=1))[0][0])

    # import matplotlib.pyplot as plt

    # Figure, Axis = plt.subplots(1,1)
    # Axis.plot(MedDir[:len(Args)][:,2], MILDir[Args][:,2], linestyle='none', marker='o',color=(1,0,0))
    # plt.show(Figure)


    # Figure, Axis = plt.subplots(1,1)
    # Axis.plot([MedMIL[(d[0],d[1],d[2])] for d in MedDir], MIL[Args], linestyle='none', marker='o',color=(1,0,0))
    # plt.show(Figure)

    return MIL, MILDir

def fit_fabric_tensor(MIL_values, directions):
    """
    Ellipsoidal fit of the fabric tensor M directly from MIL values and directions using least squares.
    See T. P. HARRIGAN, R. W. MANN
    Characterization of microstructural anisotropy in orthotropic materials using a second rank tensor
    Journal of materials science 19 (1984) 761-767
    
    Parameters:
    - MIL_values: Array of MIL values for each direction (shape N).
    - directions: Array of unit vectors for each direction (shape N x 3).
    
    Returns:
    - Fitted fabric tensor M (3x3 symmetric matrix).
    """
    N = len(MIL_values)
    
    # Create the design matrix A (N x 6)
    A = np.zeros((N, 6))
    for i in range(N):
        v1, v2, v3 = directions[i]
        A[i] = [v1**2, v2**2, v3**2, np.sqrt(2)*v1*v2, np.sqrt(2)*v1*v3, np.sqrt(2)*v2*v3]
    
    # The target vector b (MIL values squared inversely)
    b = 1/MIL_values**2

    # Solve the normal equation: (A.T @ A) v = A.T @ b
    AT_A = np.dot(np.transpose(A), A)
    AT_b = np.dot(np.transpose(A), b)
    v = np.dot(np.linalg.inv(AT_A), AT_b)

    # Construct the fabric tensor from the solution vector v
    M = np.array([
        [v[0], v[5] / np.sqrt(2), v[4] / np.sqrt(2)],
        [v[5] / np.sqrt(2), v[1], v[3] / np.sqrt(2)],
        [v[4] / np.sqrt(2), v[3] / np.sqrt(2), v[2]]
    ])
    
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

#%% Main

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

        # Compute MIL
        MIL, Directions = ComputeMIL(Array, Subdivision=2, StepSize=5)
        M = fit_fabric_tensor(MIL, Directions)
        eVal, eVec = np.linalg.eig(M)

        # modify for symetries
        eVec[:,0] = -eVec[:,0]
        eVec[:,1] = -eVec[:,1]
        eVal[:2] = eVal[:2][::-1]
        print(eVal)
        print(MeVal)
        print(eVec)
        print(MeVec)

        # File name for output
        # FName = Path(Arguments.OutputPath) / ROI.name[:-4]
        
        
        
        Morphometry.nX = Array.shape[2]  
        Morphometry.nY = Array.shape[1]  
        Morphometry.nZ = Array.shape[0]  
        MedMIL, _, _, Areas, Dict1, DictList, ListNLs, ListSumL, Voxels = Morphometry.OriginalDistribution(Array,Step=5,Power=2)
        Keys = [K for K in Dict1.keys()]
        R0 = [K for K in Voxels[Keys[0]].keys()]
        VoxRay = [Voxels[Keys[0]][C] for C in R0]

        MedDir = np.array([k for k in MedMIL.keys()])
        MedVal = np.array([MedMIL[k] for k in MedMIL.keys()])
        MedRay = np.array([Dict1[k] for k in Keys]) - 1
        MedNum = np.array([ListNLs[3][k] for k in Keys])
        MedSum = np.array([ListSumL[3][k] for k in Keys])


        MedM = fit_fabric_tensor(MedVal, MedDir)
        MeVal, MeVec = np.linalg.eig(MedM)
        MeVec[np.argsort(MeVal)]

        Points = np.reshape(MIL,(-1,1)) * Directions
        PC = pv.PolyData(Points / max(MIL))
        MedPoints = np.reshape(MILVal,(-1,1)) * MedDir
        MedPC = pv.PolyData(MedPoints / max(MILVal))

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(PC, color="red", point_size=10, render_points_as_spheres=True)
        pl.add_mesh(MedPC, color="blue", point_size=10, render_points_as_spheres=True)
        pl.camera_position = 'xz'
        pl.camera.roll = 0
        pl.camera.elevation = 30
        pl.camera.azimuth = 30
        pl.camera.zoom(1.2)
        pl.add_axes(viewport=(0,0,0.3,0.3))
        # pl.screenshot(FName.parent / (FName.name + '.png'))
        pl.show()

    





        chart = pv.Chart2D()
        plot = chart.scatter(MedVals, MIL)
        chart.show()
        
        # Weight MIL according to area
        MIL = MIL * Areas / np.sum(Areas)

        M = fit_fabric_tensor(MIL, Directions)
        eValues, eVectors = np.linalg.eig(M)

        # Sort eigen values and egein vectors
        eVectors = eVectors[np.argsort(eValues)]
        eValues = np.abs(np.sort(eValues))

        # Create a unit sphere and transform it to an ellipsoid
        sphere = pv.Sphere(radius=Array.shape[0]/2, theta_resolution=50, phi_resolution=50)

        # Scale the sphere by the square roots of the eigenvalues
        scaling_matrix = np.diag(np.sqrt(eValues))
        transformation_matrix = np.matmul(eVectors, scaling_matrix)

        # Transform the sphere points to ellipsoid points
        ellipsoid_points = np.matmul(sphere.points, transformation_matrix.T)

        # Center the ellipsoid at the structure's midpoint
        center_offset = np.array(Array.shape) / 2
        ellipsoid_points += center_offset
        ellipsoid = pv.PolyData(ellipsoid_points, sphere.faces)

        # Calculate the radius for each ellipsoid point to color by radius
        radii = np.linalg.norm(ellipsoid.points - center_offset, axis=1)
        radii = (radii - min(radii)) / (max(radii) - min(radii))
        radii = radii * (max(eValues) - min(eValues)) + min(eValues)
        ellipsoid['MIL'] = radii

        # Create the structure mesh from the binary array
        structure_mesh = pv.wrap(Array)
        structure_iso = structure_mesh.contour([0.5])  # Isosurface at level 0.5

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
        pl.add_mesh(ellipsoid, scalars='MIL', cmap='jet', opacity=1, scalar_bar_args=sargs)
        pl.add_mesh(structure_iso, color=(0.87,0.91,0.91), opacity=1, show_edges=False)
        pl.camera_position = 'xz'
        pl.camera.roll = 0
        pl.camera.elevation = 30
        pl.camera.azimuth = 30
        pl.camera.zoom(1.2)
        pl.add_axes(viewport=(0,0,0.3,0.3))
        # pl.screenshot(FName.parent / (FName.name + '.png'))
        pl.show()


        # Save ROI for later analysis
        # np.save(FName.parent / (FName.name + '.npy'), ROI)

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--InputISQ', help='File name of the ISQ scan', type=str)
    Parser.add_argument('--OutputPath', help='Output path for the ROI and png image of the plot', type=str, default='02_Results/ROIs')
    Parser.add_argument('--NROIs', help='Number of region of interests to extract', type=int, default=3)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main(Arguments)

#%%

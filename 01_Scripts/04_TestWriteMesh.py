import gmsh
import numpy as np

def create_hexahedral_mesh_with_properties(voxel_array, element_size=1.0):
    """
    Function to create a hexahedral mesh from a 3D NumPy array with non-zero values
    and store the voxel value for FEniCSx usage.

    Args:
        binary_array (numpy.ndarray): 3D NumPy array where 0 indicates no meshing and
                                      other values (0 < value <= 1) represent material properties.
        element_size (float): Size of each hexahedral element (cube side length).
    """

    import gmsh

    if gmsh.is_initialized():
        gmsh.clear()
    else:
        gmsh.initialize()
    
    gmsh.model.add('Cube')

    P1 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=0.1, tag=1)
    P2 = gmsh.model.geo.addPoint(1, 0, 0, meshSize=0.1, tag=2)
    P3 = gmsh.model.geo.addPoint(0, 1, 0, meshSize=0.1, tag=3)
    P4 = gmsh.model.geo.addPoint(1, 1, 0, meshSize=0.1, tag=4)
    P5 = gmsh.model.geo.addPoint(0, 0, 1, meshSize=0.1, tag=5)
    P6 = gmsh.model.geo.addPoint(1, 0, 1, meshSize=0.1, tag=6)
    P7 = gmsh.model.geo.addPoint(0, 1, 1, meshSize=0.1, tag=7)
    P8 = gmsh.model.geo.addPoint(1, 1, 1, meshSize=0.1, tag=8)

    L01 = gmsh.model.geo.addLine(1, 2, tag=1)
    L02 = gmsh.model.geo.addLine(2, 4, tag=2)
    L03 = gmsh.model.geo.addLine(4, 3, tag=3)
    L04 = gmsh.model.geo.addLine(3, 1, tag=4)
    L05 = gmsh.model.geo.addLine(5, 6, tag=5)
    L06 = gmsh.model.geo.addLine(6, 8, tag=6)
    L07 = gmsh.model.geo.addLine(8, 7, tag=7)
    L08 = gmsh.model.geo.addLine(7, 5, tag=8)
    L09 = gmsh.model.geo.addLine(1, 5, tag=9)
    L10 = gmsh.model.geo.addLine(2, 6, tag=10)
    L11 = gmsh.model.geo.addLine(3, 7, tag=11)
    L12 = gmsh.model.geo.addLine(4, 8, tag=12)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)
    gmsh.model.geo.addCurveLoop([1, 10, -5, -9], tag=3)
    gmsh.model.geo.addCurveLoop([2, 12, -6, -10], tag=4)
    gmsh.model.geo.addCurveLoop([3, 11, -7, -12], tag=5)
    gmsh.model.geo.addCurveLoop([4, 9, -8, -11], tag=6)

    S1 = gmsh.model.geo.addPlaneSurface([1], tag=1)
    S2 = gmsh.model.geo.addPlaneSurface([2], tag=2)
    S3 = gmsh.model.geo.addPlaneSurface([3], tag=3)
    S4 = gmsh.model.geo.addPlaneSurface([4], tag=4)
    S5 = gmsh.model.geo.addPlaneSurface([5], tag=5)
    S6 = gmsh.model.geo.addPlaneSurface([6], tag=6)


    gmsh.model.geo.addSurfaceLoop([1,2,3,4,5,6], tag=1)
    V1 = gmsh.model.geo.addVolume([1])

    gmsh.model.geo.mesh.setTransfiniteCurve(1,4, meshType='Bump')
    gmsh.model.geo.mesh.setTransfiniteCurve(2,4, meshType='Bump')
    gmsh.model.geo.mesh.setTransfiniteCurve(2,4, meshType='Bump')
    gmsh.model.geo.mesh.setTransfiniteCurve(3,4, meshType='Bump')
    gmsh.model.geo.mesh.setTransfiniteCurve(4,4, meshType='Bump')
    gmsh.model.geo.mesh.setTransfiniteCurve(5,4, meshType='Bump')


    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setTransfiniteSurface(2)
    gmsh.model.geo.mesh.setTransfiniteSurface(3)
    gmsh.model.geo.mesh.setTransfiniteSurface(4)
    gmsh.model.geo.mesh.setTransfiniteSurface(5)
    gmsh.model.geo.mesh.setTransfiniteSurface(6)

    gmsh.model.geo.mesh.setTransfiniteVolume(1)

    Cube = gmsh.model.addPhysicalGroup(3, [1])
    gmsh.model.setPhysicalName(3, Cube, 'Unit Cube')

    # gmsh.model.geo.mesh.setRecombine(dim2, 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)  # <-- that's it!
    # gmsh.model.mesh.setOrder(2)   # generates Laplacian elements, but I want Serendipity

    gmsh.write("mesh_order2.msh")
    gmsh.finalize()

def generate_hexahedral_mesh_from_voxel_array(voxel_array, element_size, output_file="hexahedral_mesh.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.add("Hexahedral Mesh")

    nx, ny, nz = voxel_array.shape
    nodes = []
    elements = []

    # Create nodes for the mesh (use a 1D index for each node)
    node_tag = 1
    node_dict = {}
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if voxel_array[i, j, k] > 0:  # Only create nodes where the voxel value is greater than 0
                    x, y, z = i * element_size, j * element_size, k * element_size
                    nodes.append([x, y, z])
                    node_dict[(i, j, k)] = node_tag
                    node_tag += 1

    gmsh.model.addDiscreteEntity(3, 1)

    # Add nodes to Gmsh
    node_tags = []
    coords = []
    
    for node in nodes:
        node_tags.append(node_tag)
        coords.extend(node)

    gmsh.model.mesh.addNodes(3, 1, node_tags, coords)

    # Create hexahedral elements based on the nodes (6 nodes per hexahedron)
    element_tag = 1
    for k in range(nz-1):  # we can't create hexahedrons at the boundary
        for j in range(ny-1):
            for i in range(nx-1):
                if voxel_array[i, j, k] > 0 and voxel_array[i+1, j, k] > 0 and voxel_array[i, j+1, k] > 0 and voxel_array[i+1, j+1, k] > 0 and voxel_array[i, j, k+1] > 0 and voxel_array[i+1, j, k+1] > 0 and voxel_array[i, j+1, k+1] > 0 and voxel_array[i+1, j+1, k+1] > 0:
                    # Get the node tags of the current voxel
                    n1 = node_dict.get((i, j, k))
                    n2 = node_dict.get((i+1, j, k))
                    n3 = node_dict.get((i+1, j+1, k))
                    n4 = node_dict.get((i, j+1, k))
                    n5 = node_dict.get((i, j, k+1))
                    n6 = node_dict.get((i+1, j, k+1))
                    n7 = node_dict.get((i+1, j+1, k+1))
                    n8 = node_dict.get((i, j+1, k+1))
                    
                    # Define the 8 nodes for the current hexahedron
                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])
                    element_tag += 1

    # Add elements to Gmsh
    gmsh.model.mesh.addElements(3, 1, [1] * len(elements), elements, node_tags)

    # Save the mesh
    gmsh.write(output_file)
    gmsh.finalize()

def generate_hexahedral_mesh_from_binary_array(voxel_array, element_size, output_file="mesh.msh"):
    """
    Generates a hexahedral mesh from a 3D binary numpy array using Gmsh.

    Parameters:
    - voxel_array (np.ndarray): 3D binary array where 1 represents material and 0 represents empty space.
    - element_size (float): The size of each hexahedral element in the mesh.
    - output_file (str): Name of the output .msh file.
    """
    gmsh.initialize()
    
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.add("3D Hexahedral Mesh")
    N = 4
    Rec2d = True  # tris or quads
    Rec3d = True  # tets, prisms or hexas
    
    nx, ny, nz = voxel_array.shape

    # Loop through each voxel in the binary array
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxel_array[i, j, k] == 1:  # Only create a hexahedron if voxel is 1
                    # Calculate coordinates of the bottom corner of the voxel
                    x = i * element_size
                    y = j * element_size
                    z = k * element_size

                    # Create the bottom-left corner point of the voxel
                    point = gmsh.model.geo.addPoint(x, y, z)

                    # Extrude in x-direction to create the first line segment
                    line = gmsh.model.geo.extrude([(0, point)], element_size, 0, 0, numElements=[1], heights=[1])

                    # Extrude in y-direction to create a surface from the line segment
                    surface = gmsh.model.geo.extrude([line[1]], 0, element_size, 0, numElements=[1], heights=[1], recombine=True)

                    # Extrude in z-direction to create the volume from the surface
                    volume = gmsh.model.geo.extrude([surface[1]], 0, 0, element_size, numElements=[1], heights=[1], recombine=True)

    gmsh.model.geo.synchronize()

    # Add the volume as a physical group for easy access in FEniCSx
    gmsh.model.addPhysicalGroup(3, [volume[1][1]])

    gmsh.model.mesh.generate(3)

    
    # Save the mesh in Gmsh format
    gmsh.write(output_file)
    gmsh.finalize()

    return

voxel_array = np.array([
    [[0, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 1, 1, 1],
     [0, 0, 1, 0]],

    [[1, 0, 1, 1],
     [1, 1, 1, 0],
     [0, 1, 1, 1],
     [1, 1, 1, 0]],

    [[0, 0, 1, 0],
     [1, 0, 1, 1],
     [1, 1, 0, 0],
     [1, 1, 1, 1]],

    [[1, 0, 0, 1],
     [0, 1, 0, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 1]]
])

element_size = 1.0
output_file = "hexahedral_mesh.msh"

# Generate the hexahedral mesh
generate_hexahedral_mesh_from_binary_array(voxel_array, element_size, output_file)

    gmsh.initialize()
    gmsh.model.add("hexahedral_mesh")
    
    nx, ny, nz = voxel_array.shape
    point_tags = {}

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x, y, z = i * element_size, j * element_size, k * element_size
                tag = gmsh.model.geo.addPoint(x, y, z, element_size)
                point_tags[(i, j, k)] = tag

    # Dictionary to store volumes grouped by property value
    property_volumes = {}

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxel_array[i, j, k] > 0:  # Only include non-empty voxels
                    p1 = point_tags[(i, j, k)]
                    p2 = point_tags[(i + 1, j, k)]
                    p3 = point_tags[(i + 1, j + 1, k)]
                    p4 = point_tags[(i, j + 1, k)]
                    p5 = point_tags[(i, j, k + 1)]
                    p6 = point_tags[(i + 1, j, k + 1)]
                    p7 = point_tags[(i + 1, j + 1, k + 1)]
                    p8 = point_tags[(i, j + 1, k + 1)]

                    l1 = gmsh.model.geo.addLine(p1, p2)
                    l2 = gmsh.model.geo.addLine(p2, p3)
                    l3 = gmsh.model.geo.addLine(p3, p4)
                    l4 = gmsh.model.geo.addLine(p4, p1)
                    l5 = gmsh.model.geo.addLine(p5, p6)
                    l6 = gmsh.model.geo.addLine(p6, p7)
                    l7 = gmsh.model.geo.addLine(p7, p8)
                    l8 = gmsh.model.geo.addLine(p8, p5)
                    l9 = gmsh.model.geo.addLine(p1, p5)
                    l10 = gmsh.model.geo.addLine(p2, p6)
                    l11 = gmsh.model.geo.addLine(p3, p7)
                    l12 = gmsh.model.geo.addLine(p4, p8)

                    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
                    s1 = gmsh.model.geo.addPlaneSurface([cl1])

                    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
                    s2 = gmsh.model.geo.addPlaneSurface([cl2])

                    cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
                    s3 = gmsh.model.geo.addPlaneSurface([cl3])

                    cl4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
                    s4 = gmsh.model.geo.addPlaneSurface([cl4])

                    cl5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
                    s5 = gmsh.model.geo.addPlaneSurface([cl5])

                    cl6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
                    s6 = gmsh.model.geo.addPlaneSurface([cl6])

                    volume = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
                    vol = gmsh.model.geo.addVolume([volume])

                    # Retrieve the property value for this voxel
                    property_value = voxel_array[i, j, k]
                    
                    # Group volumes by property value
                    if property_value not in property_volumes:
                        property_volumes[property_value] = []
                    property_volumes[property_value].append(vol)

    # Assign a unique physical group tag for each property value
    for tag, (prop_val, volumes) in enumerate(property_volumes.items(), start=1):
        gmsh.model.addPhysicalGroup(3, volumes, tag=tag)
        gmsh.model.setPhysicalName(3, tag, f"material_{prop_val}")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write(f'Test.msh')
    
# Example usage:
# Create a small binary numpy array with values between 0 and 1
voxel_array = np.zeros((5, 5, 5), dtype=int)
voxel_array[1:4, :2, 1:4] = 1

create_hexahedral_mesh_with_properties(binary_array)

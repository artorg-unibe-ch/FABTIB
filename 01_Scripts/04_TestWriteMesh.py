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

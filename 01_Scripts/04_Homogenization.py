#%% #!/usr/bin/env python3

"""
Script used to perform cube homogenization using FEniCSx
doi: 10.1007/s10237-007-0109-7
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import ufl
import gmsh
import numpy as np
from Time import Time
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
from dolfinx import io, fem, mesh
from dolfinx.fem.petsc import LinearProblem

#%% Define geometric spaces - Faces

def LocateFaces(Mesh):
    Geometry = Mesh.geometry.x
    F_Bottom = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[2], Geometry[:,2].min()))
    F_Top = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[2], Geometry[:,2].max()))
    F_North = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].min()))
    F_South = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].max()))
    F_East = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].min()))
    F_West = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].min()))
    return [F_Bottom, F_Top, F_North, F_South, F_East, F_West]

#%% Kinematic Uniform Boundary Conditions

def KUBCs(E_Hom, Faces, Geometry, Mesh, V):

    # Reference nodes and face vertices
    V_Bottom, V_Top, V_North, V_South, V_East, V_West = Faces

    BCs = []
    Constrained = []
    for Vertice in V_West:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_South:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for i, Vertice in enumerate(V_Bottom):

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for i, Vertice in enumerate(V_East):

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for i, Vertice in enumerate(V_North):

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for i, Vertice in enumerate(V_Top):

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(3), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    return BCs

#%% Periodicity-Mixed Uniform Boundary Conditions

def Tensile1BC(Value, L1, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L1/2))
    u2 = fem.Constant(Mesh,(-Value*L1/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_East = fem.dirichletbc(u1, D_East, V.sub(0))
    BC_West = fem.dirichletbc(u2, D_West, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Tensile2BC(Value, L2, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L2/2))
    u2 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u1, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u2, D_South, V.sub(1))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Tensile3BC(Value, L3, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L3/2))
    u2 = fem.Constant(Mesh,(-Value*L3/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u1, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u2, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear12BC(Value, L1, L2, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u23 = fem.Constant(Mesh,(Value*L1/2))
    u32 = fem.Constant(Mesh,(-Value*L1/2))
    u13 = fem.Constant(Mesh,(Value*L2/2))
    u31 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(0), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(0), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(1), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(1), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u13, D_North, V.sub(0))
    BC_South = fem.dirichletbc(u31, D_South, V.sub(0))
    BC_East = fem.dirichletbc(u23, D_East, V.sub(1))
    BC_West = fem.dirichletbc(u32, D_West, V.sub(1))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear13BC(Value, L1, L3, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u23 = fem.Constant(Mesh,(Value*L1/2))
    u32 = fem.Constant(Mesh,(-Value*L1/2))
    u12 = fem.Constant(Mesh,(Value*L3/2))
    u21 = fem.Constant(Mesh,(-Value*L3/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(0), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(0), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(2), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(2), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u12, D_Top, V.sub(0))
    BC_Bottom = fem.dirichletbc(u21, D_Bottom, V.sub(0))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_East = fem.dirichletbc(u23, D_East, V.sub(2))
    BC_West = fem.dirichletbc(u32, D_West, V.sub(2))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear23BC(Value, L2, L3, Faces, Mesh, V):

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u12 = fem.Constant(Mesh,(Value*L3/2))
    u21 = fem.Constant(Mesh,(-Value*L3/2))
    u13 = fem.Constant(Mesh,(Value*L2/2))
    u31 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(1), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(1), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(2), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(2), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u12, D_Top, V.sub(1))
    BC_Bottom = fem.dirichletbc(u21, D_Bottom, V.sub(1))
    BC_North = fem.dirichletbc(u13, D_North, V.sub(2))
    BC_South = fem.dirichletbc(u31, D_South, V.sub(2))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def PMUBC(E_Hom, Faces, L1, L2, L3, Mesh, V):

    if E_Hom[0,0] != 0:
        BCs = Tensile1BC(E_Hom[0,0], L1, Faces, Mesh, V)
    elif E_Hom[1,1] != 0:
        BCs = Tensile2BC(E_Hom[1,1], L2, Faces, Mesh, V)
    elif E_Hom[2,2] != 0:
        BCs = Tensile3BC(E_Hom[2,2], L3, Faces, Mesh, V)
    elif E_Hom[0,1] != 0:
        BCs = Shear12BC(E_Hom[0,1], L1, L2, Faces, Mesh, V)
    elif E_Hom[0,2] != 0:
        BCs = Shear13BC(E_Hom[0,2], L1, L3, Faces, Mesh, V)
    elif E_Hom[1,2] != 0:
        BCs = Shear23BC(E_Hom[1,2], L2, L3, Faces, Mesh, V)

    return BCs

#%% Main

def Homogenize(MeshFile, Parameters, BCsType):

    # Read Mesh and create model
    if gmsh.is_initialized():
        gmsh.clear()
    else:
        gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.merge(str(MeshFile))
    gmsh.model.mesh.generate()
    Mesh, Tags, Classes = io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)

    # Define Material Constants
    E      = PETSc.ScalarType(Parameters[0])                  # Young's modulus (Pa)
    Nu     = PETSc.ScalarType(Parameters[1])                  # Poisson's ratio (-)
    Mu     = fem.Constant(Mesh, E/(2*(1 + Nu)))               # Shear modulus (kPa)
    Lambda = fem.Constant(Mesh, E*Nu/((1 + Nu)*(1 - 2*Nu)))   # First Lamé parameter (kPa)
    
    # Stiffness matrix initialization
    S = np.zeros((6,6))

    # External surfaces
    Geometry = Mesh.geometry.x
    L1 = max(Geometry[:,0]) - min(Geometry[:,0])
    L2 = max(Geometry[:,1]) - min(Geometry[:,1])
    L3 = max(Geometry[:,2]) - min(Geometry[:,2])
    S1 = L2 * L3
    S2 = L1 * L3
    S3 = L1 * L2
    Volume = L1 * L2 * L3

    # Functions space over the mesh domain
    ElementType = 'Lagrange'
    PolDegree = 1
    Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree)
    V = fem.FunctionSpace(Mesh, Ve)
    u = ufl.TrialFunction(V)     # Incremental displacement
    v = ufl.TestFunction(V)      # Test function

    # Kinematics
    d = len(u)                         # Spatial dimension
    I = ufl.variable(ufl.Identity(d))  # Identity tensor

    # Variational formulation (Linear elasticity)
    def Epsilon(u):
        return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    def Sigma(u):
        return Lambda * ufl.nabla_div(u) * I + 2 * Mu * Epsilon(u)
    Psi = ufl.inner(Sigma(u), Epsilon(v)) * ufl.dx

    # Load cases
    LCs = ['Tensile1', 'Tensile2', 'Tensile3', 'Shear12', 'Shear13', 'Shear23']

    # Corresponding homogenized strain
    Value = 0.0001
    E_Homs = np.zeros((6,3,3))
    E_Homs[0,0,0] = Value
    E_Homs[1,1,1] = Value
    E_Homs[2,2,2] = Value
    E_Homs[3,0,1] = Value
    E_Homs[3,1,0] = Value
    E_Homs[4,0,2] = Value
    E_Homs[4,2,0] = Value
    E_Homs[5,1,2] = Value
    E_Homs[5,2,1] = Value

    # Locate faces vertices
    Faces = LocateFaces(Mesh)

    # Boundary conditions (external loads)
    f = fem.Constant(Mesh,(0.0, 0.0, 0.0))
    Load = ufl.dot(f, u) * ufl.ds

    print('\nStart Homogenization')
    Time.Process(1, '')
    for LoadCase in range(6):

        Time.Update(LoadCase/6, LCs[LoadCase])

        E_Hom = E_Homs[LoadCase]
        if BCsType == 'KUBC':
            BCs = KUBCs(E_Hom, Faces, Geometry, Mesh, V)
        elif BCsType == 'PMUBC':
            BCs = PMUBC(E_Hom, Faces, L1, L2, L3, Mesh, V)

        # Solve problem
        Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
        uh = Problem.solve()

        # Compute homogenized stress
        S_Matrix = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                S_Matrix[i,j] = fem.assemble_scalar(fem.form(Sigma(uh)[i,j]*ufl.dx))
        S_Hom = S_Matrix / Volume

        # Build stiffness matrix
        epsilon = [E_Hom[0,0], E_Hom[1,1], E_Hom[2,2], 2*E_Hom[0,1], 2*E_Hom[2,0], 2*E_Hom[1,2]]
        sigma = [S_Hom[0,0], S_Hom[1,1], S_Hom[2,2], S_Hom[0,1], S_Hom[2,0], S_Hom[1,2]]

        for i in range(6):
            S[i,LoadCase] = sigma[i] / epsilon[LoadCase]

    Time.Process(0,'Done')

    return S

        
#%% #!/usr/bin/env python3

"""
Script description
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Import

import numpy as np

#%% Functions

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

    # Build 4th order tensor
    B = np.zeros((3,3,3,3))

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
def DyadicProduct(A,B):

    if A.size == 3:

        C = np.zeros((3,3))

        for i in range(3):
            for j in range(3):
                C[i,j] = A[i]*B[j]

    elif A.size == 9:

        C = np.zeros((3,3,3,3))

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i,j,k,l] = A[i,j] * B[k,l]

    else:
        print('Matrices sizes mismatch')

    return C
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
def Mandel2EngineeringNotation(A):

    B = np.zeros((6,6))

    for i in range(6):
        for j in range(6):

            if i < 3 and j >= 3:
                B[i,j] = A[i,j] / np.sqrt(2)

            elif i >= 3 and j < 3:
                B[i,j] = A[i,j] / np.sqrt(2)

            elif i >= 3 and j >= 3:
                B[i,j] = A[i,j] / 2

            else:
                B[i, j] = A[i, j]

    return B
def PlotStiffnessTensor(S4, NPoints):

    I = np.eye(3)

    ## Build data for plotting tensor
    u = np.arange(0, 2 * np.pi + 2 * np.pi / NPoints, 2 * np.pi / NPoints)
    v = np.arange(0, np.pi + np.pi / NPoints, np.pi / NPoints)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    Color = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            n = np.array([X[i, j], Y[i, j], Z[i, j]])
            N = DyadicProduct(n, n)

            Elongation = FrobeniusProduct(N, Transform(S4, N))
            X[i, j], Y[i, j], Z[i, j] = np.array([X[i, j], Y[i, j], Z[i, j]]) * Elongation

            BulkModulus = FrobeniusProduct(I, Transform(S4, N))
            Color[i, j] = BulkModulus

    MinX, MaxX = int(X.min()), int(X.max())
    MinY, MaxY = int(Y.min()), int(Y.max())
    MinZ, MaxZ = int(Z.min()), int(Z.max())

    if Color.max() - Color.min() > 1:
        NormedColor = Color - Color.min()
        NormedColor = NormedColor / NormedColor.max()
    else:
        NormedColor = np.round(Color / Color.max()) / 2

    ## Plot tensor in image coordinate system
    Figure = plt.figure(figsize=(5.5, 4))
    Axe = Figure.add_subplot(111, projection='3d')
    Axe.plot_surface(X, Y, Z, facecolors=plt.cm.jet(NormedColor), rstride=1, cstride=1, alpha=0.2, shade=False)
    Axe.plot_wireframe(X, Y, Z, color='k', rstride=1, cstride=1, linewidth=0.2)
    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axe.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    # make the panes transparent
    Axe.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axe.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axe.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    Axe.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axe.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axe.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # modify ticks
    Axe.set_xticks([MinX, 0, MaxX])
    Axe.set_yticks([MinY, 0, MaxY])
    Axe.set_zticks([MinZ, 0, MaxZ])
    Axe.xaxis.set_ticklabels([MinX, 0, MaxX])
    Axe.yaxis.set_ticklabels([MinY, 0, MaxY])
    Axe.zaxis.set_ticklabels([MinZ, 0, MaxZ])

    Axe.xaxis.set_rotate_label(False)
    Axe.set_xlabel('X (MPa)', rotation=0)
    Axe.yaxis.set_rotate_label(False)
    Axe.set_ylabel('Y (MPa)', rotation=0)
    Axe.zaxis.set_rotate_label(False)
    Axe.set_zlabel('Z (MPa)', rotation=90)

    Axe.set_title('Elasticity tensor')

    ColorMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    ColorMap.set_array(Color)
    if not NormedColor.max() == 1:
        ColorBar = plt.colorbar(ColorMap, ticks=[int(Color.mean() - 1), int(Color.mean()), int(Color.mean() + 1)])
        plt.cm.ScalarMappable.set_clim(self=ColorMap, vmin=int(Color.mean() - 1), vmax=int(Color.mean() + 1))
    else:
        ColorBar = plt.colorbar(ColorMap)
    ColorBar.set_label('Bulk modulus (MPa)')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def WriteStiffnessVTK(S4, NPoints, Directory):

    I = np.eye(3)

    ## Build data for plotting tensor
    u = np.arange(0, 2 * np.pi + 2 * np.pi / NPoints, 2 * np.pi / NPoints)
    v = np.arange(0, np.pi + np.pi / NPoints, np.pi / NPoints)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    Color = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            n = np.array([X[i, j], Y[i, j], Z[i, j]])
            N = DyadicProduct(n, n)

            Elongation = FrobeniusProduct(N, Transform(S4, N))
            X[i, j], Y[i, j], Z[i, j] = np.array([X[i, j], Y[i, j], Z[i, j]]) * Elongation

            BulkModulus = FrobeniusProduct(I, Transform(S4, N))
            Color[i, j] = BulkModulus

    MinX, MaxX = int(X.min()), int(X.max())
    MinY, MaxY = int(Y.min()), int(Y.max())
    MinZ, MaxZ = int(Z.min()), int(Z.max())

    # Write VTK file
    VTKFile = open(Directory + 'Stiffness.vtk', 'w')

    # Write header
    VTKFile.write('# vtk DataFile Version 4.2\n')
    VTKFile.write('VTK from Python\n')
    VTKFile.write('ASCII\n')
    VTKFile.write('DATASET UNSTRUCTURED_GRID\n')

    # Write points coordinates
    Points = int(X.shape[0] * X.shape[1])
    VTKFile.write('\nPOINTS ' + str(Points) + ' floats\n')
    for i in range(Points):
        VTKFile.write(str(X.reshape(Points)[i].round(3)))
        VTKFile.write(' ')
        VTKFile.write(str(Y.reshape(Points)[i].round(3)))
        VTKFile.write(' ')
        VTKFile.write(str(Z.reshape(Points)[i].round(3)))
        VTKFile.write('\n')

    # Write cells connectivity
    Cells = int(NPoints**2)
    ListSize = int(Cells*5)
    VTKFile.write('\nCELLS ' + str(Cells) + ' ' + str(ListSize) + '\n')

    ## Add connectivity of each cell
    Connectivity = np.array([0, 1])
    Connectivity = np.append(Connectivity,[NPoints+2,NPoints+1])

    for i in range(Cells):
        VTKFile.write('4')

        if i > 0 and np.mod(i,NPoints) == 0:
            Connectivity = Connectivity + 1

        for j in Connectivity:
            VTKFile.write(' ' + str(j))
        VTKFile.write('\n')

        ## Update connectivity
        Connectivity = Connectivity+1

    # Write cell types
    VTKFile.write('\nCELL_TYPES ' + str(Cells) + '\n')
    for i in range(Cells):
        VTKFile.write('9\n')

    # Write MIL values
    VTKFile.write('\nPOINT_DATA ' + str(Points) + '\n')
    VTKFile.write('SCALARS Bulk_modulus float\n')
    VTKFile.write('LOOKUP_TABLE default\n')

    for i in range(NPoints+1):
        for j in range(NPoints+1):
            VTKFile.write(str(Color.reshape(Points)[j + i * (NPoints+1)].round(3)))
            VTKFile.write(' ')
        VTKFile.write('\n')
    VTKFile.close()

    return

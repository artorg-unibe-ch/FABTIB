import numpy as np
from odbAccess import *

Odb = openOdb('Test.odb')
coordSys = Odb.rootAssembly.DatumCsysByThreePoints(name='CSYS-1',
                                                   coordSysType=CARTESIAN,
                                                   origin=(0.0, 0.0, 0.0),
                                                   point1=(1.0, 0.0, 0.0),
                                                   point2=(0.0, 1.0, 0.0))

Steps = list(Odb.steps.values())

Stress = np.zeros((6,6))
Strain = np.zeros((6,6))

for i, Step in enumerate(Steps):

    print('   ... Processing Step : %s\n' % Step.name)

    for Frame in Step.frames:
        if Frame.incrementNumber == 1:
            Fields = Frame.fieldOutputs
            Set = Fields['S'].getTransformedField(datumCsys=coordSys)
            for Value in Set.values:
                if Value.position == INTEGRATION_POINT :
                    Stress[i] += Value.data

            Set = Fields['E'].getTransformedField(datumCsys=coordSys)
            for Value in Set.values:
                if Value.position == INTEGRATION_POINT :
                    Strain[i] += Value.data

np.save('Stress.npy', Stress)
np.save('Strain.npy', Strain)
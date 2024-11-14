"""
Abaqus - Se-Reader
------------------

This program opens an Abaqus output file and computes the stress/strain
averages of an RVE. The output file is a simple text file and can be used 
in the SeAnalyzer to compute the homogenized stiffness matrix. In case of
porous samples only the stress can be averaged i.e. -spec = STRESS has to 
be selected. In this case the volume average over the RVE is returned!

Note: This script requires an Abaqus Python installation! 


Usage
~~~~~

Command line: ::

  abaqus python abqSeReader  ...

     -in      inputFilename
     -out     outputFilename
     -size    size1;size2;size3
     [-exe]   filename;option    (optional)
     [-spec]  type               (optional)


Parameters
~~~~~~~~~~

-in   :  filename

         Finite element results file from Abaqus (ODB file) with 6 independent 
         mechanical load cases inside. Additionally one thermal load case 
         could be given. 
         Step-1 -- 6 are used for stiffness calculation and an optional 
         Step-7 load case may be a pure thermal loadcase with dT=+1K!

         Note: Engineering strains are needed in the output file 
               (gam12, gam13, gam23 instead of eps12, eps13, eps23)


-out  :  filename

         Name of the output file where stress/strain averages and 
         porosity will be written. The Format is (last line optionial) ::

           ************************************************************
           *STRESS 
           11 22 33 23 13 12      ** stress/strain component order
           ** LC name    S11     S22     S33    S23    S13    S12
           Step-1        13.5    34.5    0.34   0.01   0.4    -0.2
           Step-2        ...     ...     ...    ...    ...    ...
           Step-3        ...     ...     ...    ...    ...    ...
           Step-4        ...     ...     ...    ...    ...    ...
           Step-5        ...     ...     ...    ...    ...    ...
           Step-6        ...     ...     ...    ...    ...    ...
          [Step-7        ...     ...     ...    ...    ...    ...]

         Quantaties are volume averages. 


-exe   :  filename;option

          This option tells the GUI that a different python interpreter 
          should be used to call this script. Abaqus post-processing
          scripts require 'abaqus python' to be called.  
          'filename' is the Abaqus exe file which should be used and 
          'option' is a call option. For this script it is always 'python'.


-size  : size1;size2;size3

         This is the size of the unit cell in x,y,z. This is necessary 
         to compute the total volume for the volume average. 


-spec  : type

         Variable specifier which should be read it can be 'STRESS'
         'STRAIN'. If this parameter is not given both will be read.


-help  : Print usage


Info
~~~~

- File:   abqSeReader.py
- Author: D. H. Pahr
  
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
abqImportError = False
try:
  from odbAccess import *
except ImportError:
  abqImportError = True
from sys import argv, exit, stdout
import dpUtils

class abqSeReader :  
  __version__='V_17.04.2024'
  __author__='D. H. Pahr'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getSeAverageAbaqus(odbName, BVTV, spec=None):
  """ Print max mises location and value given odbName
      and elset(optional)
  """
  stdout.write("   ... reading from file %s\n" % (odbName) )

  readAllSij = False
  readAllEij = False
  if spec != None : 
    if spec.find('STRESS') > -1 : readAllSij = True
    if spec.find('STRAIN') > -1 : readAllEij = True
  else : 
    readAllSij = True
    readAllEij = True


  region = "over the entire model"
  """ Open the output database """
  odb = openOdb(odbName)
  coordSys = odb.rootAssembly.DatumCsysByThreePoints(name='CSYS-1', 
                                                      coordSysType=CARTESIAN,
                                                      origin=(0.0, 0.0, 0.0),
                                                      point1=(1.0, 0.0, 0.0),
                                                      point2=(0.0, 1.0, 0.0))

  S_step = {}
  E_step = {}

  for step in list(odb.steps.values()):
      S_sum={}
      S_sum['S11']=0.
      S_sum['S22']=0.
      S_sum['S33']=0.
      S_sum['S12']=0.
      S_sum['S13']=0.
      S_sum['S23']=0.
      
      E_sum={}
      E_sum['E11']=0.
      E_sum['E22']=0.
      E_sum['E33']=0.
      E_sum['E12']=0.
      E_sum['E13']=0.
      E_sum['E23']=0.

      nS = 0
      nE = 0      
      stdout.write('   ... Processing Step : %s\n' % step.name)
      stdout.flush()
      for frame in step.frames:
        if frame.incrementNumber == 1:
          allFields = frame.fieldOutputs
          if readAllSij == True:
            if ('S' in allFields):
              stressSet = allFields['S'].getTransformedField(datumCsys=coordSys)
              for stressValue in stressSet.values:
                if stressValue.position == INTEGRATION_POINT : 
                  S_sum['S11'] += stressValue.data[0]
                  S_sum['S22'] += stressValue.data[1]
                  S_sum['S33'] += stressValue.data[2]
                  S_sum['S12'] += stressValue.data[3]
                  S_sum['S13'] += stressValue.data[4]
                  S_sum['S23'] += stressValue.data[5]
                  nS+=1
                  #print elId, ipId, S_sum['S23']/V_sum
          if readAllEij == True:
            if ('E' in allFields):
              strainSet = allFields['E'].getTransformedField(datumCsys=coordSys)
              for strainValue in strainSet.values:
                if strainValue.position == INTEGRATION_POINT : 
                  E_sum['E11'] += strainValue.data[0]
                  E_sum['E22'] += strainValue.data[1]
                  E_sum['E33'] += strainValue.data[2]
                  E_sum['E12'] += strainValue.data[3]
                  E_sum['E13'] += strainValue.data[4]
                  E_sum['E23'] += strainValue.data[5]
                  nE+=1

      for Sij in S_sum : 
        S_sum[Sij] = S_sum[Sij]
      S_step[step.name] = S_sum

      for Eij in E_sum : 
        E_sum[Eij] = E_sum[Eij]
      E_step[step.name] = E_sum          

      #if nV != nS or nV!=nE or nE != nS : 
      #  print ' **ERROR** Number of IP and S/E value not the same!'
      #  exit(1)

  """ Close the output database before exiting the program """
  odb.close()

  if readAllSij == False : S_step = None
  if readAllEij == False : E_step = None


  # if readAllEij == True and abs(V_sum/volume-1.0) > 0.000001 : 
  #     stdout.write("\n **ERROR**: Strain averaging select but denstity < 1.0! \n" ); stdout.flush()
  #     stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
  #     exit(1)
  # if  V_sum/volume > 1.0 : 
  #     stdout.write("\n **ERROR**: Check the RVE dimensions: denstity > 1.0! \n" ); stdout.flush()
  #     stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
  #     exit(1)


  return S_step, E_step

#################################################################

def write(OS, S_step=None, E_step=None) :

    order = ['11', '22', '33', '23', '13', '12']
    quant = []
    spec  = []
    value = []

    if S_step != None : 
      quant.append('STRESS')
      spec.append('S')
      value.append(S_step)
    if E_step != None :
      quant.append('STRAIN')
      spec.append('E')
      value.append(E_step)

    for i in range(len(quant)) :
      OS.write('\n*******************************************************************************')
      OS.write('\n*%s' % (quant[i]) )
      OS.write('\n%s %s %s %s %s %s      ** stress/strain component order ' % (order[0],order[1],order[2],order[3],order[4],order[5]) )

      for row in range(len(value[i])) :
        OS.write('\n')
        loadcase = 'Step-'+repr(row+1)
        OS.write('%s ' % (loadcase) )
        for col in range(6) :
            OS.write('%+13.7g ' % value[i]["Step-"+repr(row+1)][spec[i]+order[col]])
      OS.write('\n')


#==================================================================
# S T A R T
#    
if __name__ == '__main__':

    inName = None
    outName = None
    BVTV    = None
    spec    = None
    argList = argv
    argc = len(argList)

    # Type             Param Description               DefaultValue   Optional    Info"
    guiInit = \
   "*modulName              'SeReader'                                                                             \n"\
   +"*fileEntryIn     -in   'Abaqus ODB File         | filename'          testIn.odb     no          odb           \n"\
   +"*fileEntryOut    -out  'Output File Name        | filename'          testOut.out    no          out;txt;dat   \n"\
   +"*entry           -exe  'Run Command             | filename;option'   /programs/hks/Commands/abq;python  yes  1   \n"\
   +"*entry           -BVTV 'Bone volume fraction    | bvtv'              >1%            no          3             \n"\
   +"*combo           -spec 'Variable Specifier      | type'              STRESS         yes         STRESS;STRAIN \n"

    argList  = argv
    argc = len(argList)
    i=0
    while (i < argc):
        if (argList[i][:4] == "-in"):
            i += 1
            inName = argList[i]
        elif (argList[i][:4] == "-out"):
            i += 1
            outName = argList[i]
        elif (argList[i][:5] == "-BVTV"):
            i += 1
            BVTV = argList[i]
        elif (argList[i][:5] == "-spec"):
            i += 1
            spec = argList[i]        
        elif (argList[i][:4] == "-gui"):
            stdout.write( '%s' % guiInit); stdout.flush()
            exit(0)
        elif (argList[i][:2] == "-h"):
            print(__doc__)
            exit(0)
        elif (argList[i][:8] == "-version"):
            stdout.write( '%s\n' %  dpUtils.getVersion(abqSeReader.__version__)); stdout.flush()
            exit(0)
        elif (argList[i][:7] == "-author"):
            stdout.write( '%s\n' %  abqSeReader.__author__); stdout.flush()
            exit(0)
        i += 1

    stdout.write( '\n S T A R T  %s %s - %s \n\n' % ("abqSeReader",'V_04.01.2022','D. H. Pahr') )
    stdout.flush()

    if not (inName):
        stdout.write( __doc__ ); stdout.flush()
        stdout.write( "\n **ERROR** '-in' file name not given\n\n" ); stdout.flush()
        stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
        exit(1)      
    if not (outName):
        stdout.write( __doc__ ); stdout.flush()
        stdout.write( "\n **ERROR** '-out' file name not given\n\n" ); stdout.flush()
        stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
        exit(1)            
    if not (BVTV):
        stdout.write( __doc__ ); stdout.flush()
        stdout.write( "\n **ERROR** '-BVTV' bone volume fraction not given\n\n" ); stdout.flush()
        stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
        exit(1)

    # if apriori stress/strain is given avoid to long reading/analyses!
    typ = None
    if spec != None : 
      if spec.upper() == "STRESS" :
        typ = "STRESS"
      elif spec.upper() == "STRAIN" :
        typ = "STRAIN"
      else :
        stdout.write(" **ERROR** -spec: type %s not known!\n" % (typ) )
        stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
        exit(1)

    # read analyse
    if inName.upper().find("ODB") > -1 :
      if abqImportError: 
         stdout.write( "\n **ERROR** ABAQUS python modules could not be loaded!\n\n" ); stdout.flush()
         stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
         exit(1)
      S_avg, E_avg = getSeAverageAbaqus(inName,BVTV,spec=typ)
    else :
      stdout.write(" **ERROR** -in: Inpute file format in %s not known!" % (inName) )
      stdout.write( '\n E N D E D  with ERRORS \n\n' ); stdout.flush()
      exit(1)

    # write to file
    OS = open(outName,'w')
    stdout.write("   ... write to file %s\n" % (outName) )
    write(OS, S_avg, E_avg)
    #write(stdout, S_avg, E_avg)
    OS.close()

    stdout.write( '\n E N D E D  SUCCESSFULLY \n\n' ); stdout.flush()


"""
CHANGES:
07.12.2015 : Change to ";" delimiter, update documentation, userSplit
17.10.2021 : Ready for Python 3 - not testet!
04.01.2022 : Remove import split for Python 3
11.08.2023 : add -version and -author, change description (Output) in guiInit
17.04.2024 : fix for Abaqus 2024 by SBA
"""



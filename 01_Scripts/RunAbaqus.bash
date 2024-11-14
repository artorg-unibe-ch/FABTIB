abaqus interactive job=Test inp="/home/ms20s284/FABTIB2/02_Results/Abaqus/Test_Main.inp" cpus=24
abaqus python "/home/ms20s284/FABTIB2/01_Scripts/abqSeReader.py" in="/home/ms20s284/FABTIB2/02_Results/Abaqus/Test.odb"  out="/home/ms20s284/FABTIB2/02_Results/Abaqus/Test.out"  size="0.6;0.6;0.6"
rm *.com *.sta *.pes *.pmg *.prt *.par *.msg *.dat *.env *.fil *.odb

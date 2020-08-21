#!/usr/bin/env sh


curr_dir=$(pwd)
for dir in $(ls -d */); do
  cd $curr_dir/$dir
  rm -vf CHG* \
    CONTCAR \
    DOSCAR \
    EIGENVAL \
    IBZKPT \
    OSZICAR \
    PCDAT \
    POTCAR \
    PROCAR \
    REPORT \
    XDATCAR \
    out* \
    sub_vasp* \
    vasprun.xml \
    vdw* \
    INCAR \
    KPOINTS \
    *.vasp
done

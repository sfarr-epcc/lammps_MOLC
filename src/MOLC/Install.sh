# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# arg1 = file, arg2 = file it depends on

# enforce using portable C locale
LC_ALL=C
export LC_ALL

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# the KSPACE, MOLECULE and ASPHERE packages must be installed.

if (test $1 = 1) then
  if (test ! -e ../pppm.h) then
    echo "Must install KSPACE package with USER-MOLC"
    exit 1
  fi
  if (test ! -e ../fix_nve_asphere.cpp) then
    echo "Must install ASPHERE package with USER-MOLC"
    exit 1
  fi
  if (test ! -e ../molecule.h) then
    echo "Must install MOLECULE package with USER-MOLC"
    exit 1
  fi
fi


# list of files with dependcies
action bond_ellipsoid.h atom_vec_ellipsoid.h
action bond_ellipsoid.h molecule.h
action bond_ellipsoid.cpp atom_vec_ellipsoid.h
action bond_ellipsoid.cpp molecule.h
action bond_molc.h
action bond_molc.cpp
action pppm_molc.h pppm.h
action pppm_molc.cpp pppm.h
action pair_molc_cut.h
action pair_molc_cut.cpp
action pair_molc_long.h
action pair_molc_long.cpp
action compute_inter.h 
action compute_inter.cpp
action compute_inter_molc.h
action compute_inter_molc.cpp
action fix_temp_berendsen_asphere.h
action fix_temp_berendsen_asphere.cpp

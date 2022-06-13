/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
   Contributing author: Matteo Ricci <matteoeghirotta@gmail.com>
--------------------------------------------------------------------------- */

#ifndef LMP_BOND_MOLC_H
#define LMP_BOND_MOLC_H

#include <cstdio>
#include "pointers.h"
#include "bond.h"

namespace LAMMPS_NS {

class BondMolc : public Bond {
 public:

  BondMolc(class LAMMPS *lmp):Bond(lmp) {}

 protected:
  void ev_tally(int, int, int, int, double, double, double, double, double);
  void ev_tally_ang(int i, int j, int nlocal, int newton_bond,
                    double,
		    double, // r12 dependent
		    double, // r.u dependent
		    double, // r.u dependent
		    double, // r.u dependent
		    double, // u.r dependent
		    double, // u.r dependent
		    double, // u.r dependent
                    double, double, double, // r12 dependent
		    double, double, double, //rotation u.r
		    double, double, double,
		    double, double, double,
		    double, double, double, //rotation r.u
		    double, double, double,
		    double, double, double);
};

}

#endif


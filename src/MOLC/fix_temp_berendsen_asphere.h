/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matteo Ricci
   -------------------------------------------------------------------------*/

#ifdef FIX_CLASS

FixStyle(temp/berendsen/asphere,FixTempBerendsenAsphere)

#else

#ifndef   	FIX_TEMP_BERENDSEN_ASPHERE_H_
# define   	FIX_TEMP_BERENDSEN_ASPHERE_H_

#include "fix.h"

namespace LAMMPS_NS {

class FixTempBerendsenAsphere : public Fix {
 public:
  FixTempBerendsenAsphere(class LAMMPS *, int, char **);
  ~FixTempBerendsenAsphere();
  int setmask();
  void init();
  void end_of_step();
  int modify_param(int, char **);
  void reset_target(double);
  double compute_scalar();
  double compute_vector(int);
  virtual void *extract(const char *, int &);

 private:
  int which;
  double t_start,t_stop,t_target,t_period;
  double coupling_factor;
  double energy;
  int tstyle,tvar;
  char *tstr;

  double stored_data[6];

  char *id_temp_transl;
  char *id_temp_rot;
  class Compute *temperature_transl;
  class Compute *temperature_rot;
  int tflag;
  bool debug;
};

}

#endif
#endif

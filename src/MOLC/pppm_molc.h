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



#ifdef KSPACE_CLASS

KSpaceStyle(pppm/molc,PPPMMolc)

#else

#ifndef LMP_PPPM_MOLC_H
#define LMP_PPPM_MOLC_H

#include "pppm.h"



namespace LAMMPS_NS {

class PPPMMolc : public PPPM {
 public:
  PPPMMolc(class LAMMPS *);
  ~PPPMMolc();
  void init() override;
  void settings(int, char **) override;
  void qsum_qsq();
  void compute(int, int) override;
  int getNsitesOf(int);
  double* getCharges(int);
  void setCharges(int, int, double);

 protected:


  //support offcentre charges

  bigint ncharges; // total charges
  double max_charge_offset; // max offcentre charge offset
  class AtomVecEllipsoid *avec;
  int max_nsites;              // maximum number of sites per atom
  int* nsites;
  double ***molFrameSite;  // positions of sites
  double **molFrameCharge;  // positions of sites

  int ***part2grid;    // storage for particle -> grid mapping

  void set_grid_global() override;
  double compute_df_kspace();
  double newton_raphson_f() override;
  double final_accuracy();
  void particle_map() override;
  void make_rho() override;
  void fieldforce_ik() override;
  void fieldforce_ad() override;
  void fieldforce_peratom() override;
  void slabcorr() override;
  void make_rho_groups(int, int, int) override;
  void slabcorr_groups(int,int,int) override;

};

}    // namespace LAMMPS_NS

#endif
#endif



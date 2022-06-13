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

/* -------------------------------------------------------------------------
   Contributing authors: Matteo Ricci <matteoeghirotta@gmail.com>
                         Stephen Farr <s.farr@epcc.ed.ac.uk>
--------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(molc/cut,PairMolcCut)

#else

#ifndef LMP_PAIR_MOLC_CUT_H
#define LMP_PAIR_MOLC_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMolcCut : public Pair {
 public:
  PairMolcCut(class LAMMPS *);
  ~PairMolcCut() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void *extract(const char *, int &) override;

 protected:

  // parts for offcentre charges
  double cut_coul_global;
  double **cut_coul, **cut_coulsq;

  class AtomVecEllipsoid *avec;

  int* nsites;
  double ***molFrameSite;  // positions of sites
  double **molFrameCharge;  // positions of sites


  // parts for Gay-Berne
  enum{SPHERE_SPHERE,SPHERE_ELLIPSE,ELLIPSE_SPHERE,ELLIPSE_ELLIPSE};

  double cut_lj_global;
  double **cut_lj, **cut_ljsq;

  double gamma,upsilon,mu;   // Gay-Berne parameters
  double **shape1;           // per-type radii in x, y and z
  double **shape2;           // per-type radii in x, y and z SQUARED
  double *lshape;            // precalculation based on the shape
  double **well;             // well depth scaling along each axis ^ -1.0/mu
  double **epsilon,**sigma;  // epsilon and sigma values for atom-type pairs


  int **form;
  double **lj1,**lj2,**lj3,**lj4;
  double **offset;
  int *setwell;

  void allocate();


  double gayberne_analytic(const int i, const int j, double a1[3][3],
                           double a2[3][3], double b1[3][3], double b2[3][3],
                           double g1[3][3], double g2[3][3], double *r12,
                           const double rsq, double *fforce, double *ttor,
                           double *rtor);
  double gayberne_lj(const int i, const int j, double a1[3][3],
                     double b1[3][3],double g1[3][3],double *r12,
                     const double rsq, double *fforce, double *ttor);
  void compute_eta_torque(double m[3][3], double m2[3][3],
                          double *s, double ans[3][3]);
};

}

#endif
#endif

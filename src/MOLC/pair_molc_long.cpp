/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   Contributing author: Matteo Ricci
                        Stephen Farr (EPCC)
   -------------------------------------------------------------------------*/

#include "math.h"
#include "math_extra.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_molc_long.h"
#include "atom_vec_ellipsoid.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include <iostream>

using namespace LAMMPS_NS;

//#define DEBUG

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairMolcLong::PairMolcLong(LAMMPS *lmp) : Pair(lmp)
{

  #ifdef DEBUG
  printf("constructor\n");
  #endif

  ftable = NULL;
  ewaldflag = pppmflag = 1;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"Pair Molc requires atom style ellipsoid");

  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairMolcLong::~PairMolcLong()
{
  #ifdef DEBUG
  printf("destructor\n");
  #endif

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cut_coulsq);

    memory->destroy(cutsq);
    memory->destroy(cut_ljsq);
    memory->destroy(cut_coul);

    memory->destroy(form);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(shape1);
    memory->destroy(shape2);
    memory->destroy(well);
    memory->destroy(cut_lj);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    memory->destroy(scale);

    delete [] lshape;
    delete [] setwell;

    for (int n = 1; n <= atom->ntypes; ++n){
      for(int site=1; site<=nsites[n]; ++site){
      delete [] molFrameSite[n][site];
      }
      delete [] molFrameSite[n];
      delete [] molFrameCharge[n];
    }

    delete [] molFrameCharge;
    delete [] molFrameSite;
    delete[] nsites;
  }




  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */


// messy version
void PairMolcLong::compute(int eflag, int vflag)
{
  int i,j,ii,jj,itable,itype,jtype;
  double q1, q2;
  double ecoul,fpair;
  double fraction,table;
  double one_eng,evdwl,r,r2inv,forcecoul,factor_coul, r6inv,forcelj,factor_lj;;
  double grij,expm2,prefactor,t,erfc;
  double rsq;
  double *iquat,*jquat;
  double fforce_coul[3],fforce_lj[3],ttor_coul[3],ttor_lj[3],rtor_lj[3],r12[3];
  double a1[3][3],b1[3][3],g1[3][3],a2[3][3],b2[3][3],g2[3][3],temp[3][3];

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;

  int inum = list->inum;
  int *ilist = list->ilist;

  int *jlist, jnum;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    if (form[itype][itype] == ELLIPSE_ELLIPSE
        || form[itype][itype] == ELLIPSE_SPHERE) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat_trans(iquat,a1);
      MathExtra::diag_times3(well[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,b1);
      MathExtra::diag_times3(shape2[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,g1);
    }

    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    // rotate site i in lab frame
    double rotMat1[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat1);
    }


    // loop over neighbors of i
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      //bool sameAtom = i == j;
      bool sameAtom=false;
      factor_coul = special_coul[sbmask(j)];
      //factor_coul = sameAtom ? 0.0 : factor_coul;
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      //#ifdef DEBUG
      //printf("i = %d, j=%d\n", i,j, factor_coul, factor_lj);
      //#endif

      // start compute i,j for coul
      // rotate site j in lab frame
      double rotMat2[3][3];
      if (nsites[jtype] > 0) {
        jquat = bonus[ellipsoid[j]].quat;
        MathExtra::quat_to_mat(jquat, rotMat2);
      }

      // loop over charges on particle i
      for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
        q1 = molFrameCharge[itype][s1];

        double labFrameSite1[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s1][0] != 0.0 ||
          molFrameSite[itype][s1][1] != 0.0 ||
          molFrameSite[itype][s1][2] != 0.0){
          double ms1[3] = {
            molFrameSite[itype][s1][0],
            molFrameSite[itype][s1][1],
            molFrameSite[itype][s1][2]
          };

        MathExtra::matvec(rotMat1, ms1, labFrameSite1);
        } // end if

        double rsite1[3] = {
          labFrameSite1[0]+x[i][0],
          labFrameSite1[1]+x[i][1],
          labFrameSite1[2]+x[i][2]
        };
      
        // loop over charges of site j
        for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
          double labFrameSite2[3] = {0.0, 0.0, 0.0};
          if (molFrameSite[jtype][s2][0] != 0.0 ||
            molFrameSite[jtype][s2][1] != 0.0 ||
            molFrameSite[jtype][s2][2] != 0.0){
        
            double ms2[3] = {
              molFrameSite[jtype][s2][0],
              molFrameSite[jtype][s2][1],
              molFrameSite[jtype][s2][2]
            };

            MathExtra::matvec(rotMat2, ms2, labFrameSite2);
          } // end if

          double rsite2[3] = {
            labFrameSite2[0]+x[j][0],
            labFrameSite2[1]+x[j][1],
            labFrameSite2[2]+x[j][2]
          };

          // r12 = site center to site center vector
          r12[0] = rsite1[0]-rsite2[0];
          r12[1] = rsite1[1]-rsite2[1];
          r12[2] = rsite1[2]-rsite2[2];

          rsq = MathExtra::dot3(r12, r12);



          if ((rsq < cut_coulsq_global) && ((!sameAtom) || (s1 < s2))) {
          //if (rsq < cut_coulsq_global){


            // #ifdef DEBUG
            // printf("i = %d, j=%d, s1=%d, s2=%d, fc=%f, fl=%f, r2=%f, cut2=%f\n", i,j,s1,s2, factor_coul, factor_lj, rsq, cut_coulsq_global);
            // #endif

            r2inv = 1.0/rsq;
            q2 = molFrameCharge[jtype][s2];

            if (!ncoultablebits || rsq <= tabinnersq) {
              r = sqrt(rsq);
              grij = g_ewald * r;
              expm2 = exp(-grij*grij);
              t = 1.0 / (1.0 + EWALD_P*grij);
              erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
              prefactor = qqrd2e * scale[itype][jtype] * q1*q2/r;
              forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
              if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
            } else {
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              itable = rsq_lookup.i & ncoulmask;
              itable >>= ncoulshiftbits;
              fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
              table = ftable[itable] + fraction*dftable[itable];
              forcecoul = scale[itype][jtype] * q1*q2 * table;
              if (factor_coul < 1.0) {
                table = ctable[itable] + fraction*dctable[itable];
                prefactor = scale[itype][jtype] * q1*q2 * table;
                forcecoul -= (1.0-factor_coul)*prefactor;
              }
            }


            // printf("forcecoul=%f, scale=%f\n", forcecoul, scale[itype][jtype]);

            // if (!sameAtom) {
              fpair = forcecoul * r2inv;

              fforce_coul[0] = r12[0]*fpair; ///r12n;
              fforce_coul[1] = r12[1]*fpair; ///r12n;
              fforce_coul[2] = r12[2]*fpair; ///r12n;

              // F_parallel = F_tot . r_normalized
              f[i][0] += fforce_coul[0];
              f[i][1] += fforce_coul[1];
              f[i][2] += fforce_coul[2];

              // Torque = r x F_tot
              // torque on 1 = -pos1 x grad_pos1
              MathExtra::cross3(labFrameSite1, fforce_coul, ttor_coul);
              tor[i][0] += ttor_coul[0];
              tor[i][1] += ttor_coul[1];
              tor[i][2] += ttor_coul[2];

              if (newton_pair || j < nlocal) {
                // F_parallel = F_tot . r_normalized
                f[j][0] -= fforce_coul[0]; ///r12n;
                f[j][1] -= fforce_coul[1]; ///r12n;
                f[j][2] -= fforce_coul[2]; ///r12n;
                MathExtra::cross3(labFrameSite2, fforce_coul, ttor_coul);

                tor[j][0] -= ttor_coul[0];
                tor[j][1] -= ttor_coul[1];
                tor[j][2] -= ttor_coul[2];
              }
            // }

            if (eflag) {
              if (!ncoultablebits || rsq <= tabinnersq)
                ecoul = prefactor*erfc;
              else {
                table = etable[itable] + fraction*detable[itable];
                ecoul = scale[itype][jtype] * q1*q2 * table;
              }
              if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
            }

            if (evflag) {
              // charges inside same bead should not contribute to virial
              //if (sameAtom) fpair = 0.0;

              ev_tally(i,j,nlocal,newton_pair,
                      0.0,ecoul,fpair,r12[0],r12[1],r12[2]);
            }
          } // end if ((rsq < cut_coulsq_global) && ((!sameAtom) || (s1 < s2)))
        } // end for s2
      } // end for s1

      // end compute i,j for coul
      
      // start compute i,j for lj
      // r12 = center to center vector

      r12[0] = x[j][0]-x[i][0];
      r12[1] = x[j][1]-x[i][1];
      r12[2] = x[j][2]-x[i][2];
      rsq = MathExtra::dot3(r12,r12);
      jtype = type[j];
      
      // compute if less than cutoff
      if (rsq < cut_ljsq[itype][jtype]) {
        switch (form[itype][jtype]) {
        case SPHERE_SPHERE:
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          forcelj *= -r2inv;
          if (eflag) one_eng =
            r6inv*(r6inv*lj3[itype][jtype]-lj4[itype][jtype]) -
              offset[itype][jtype];
          fforce_lj[0] = r12[0]*forcelj;
          fforce_lj[1] = r12[1]*forcelj;
          fforce_lj[2] = r12[2]*forcelj;
          ttor_lj[0] = ttor_lj[1] = ttor_lj[2] = 0.0;
          rtor_lj[0] = rtor_lj[1] = rtor_lj[2] = 0.0;
          break;

        case SPHERE_ELLIPSE:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_lj(j,i,a2,b2,g2,r12,rsq,fforce_lj,rtor_lj);
          ttor_lj[0] = ttor_lj[1] = ttor_lj[2] = 0.0;
          break;

        case ELLIPSE_SPHERE:
          one_eng = gayberne_lj(i,j,a1,b1,g1,r12,rsq,fforce_lj,ttor_lj);
          rtor_lj[0] = rtor_lj[1] = rtor_lj[2] = 0.0;
          break;

        default:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_analytic(i,j,a1,a2,b1,b2,g1,g2,r12,rsq,
                                      fforce_lj,ttor_lj,rtor_lj);
          break;
        } // end switch

        fforce_lj[0] *= factor_lj;
        fforce_lj[1] *= factor_lj;
        fforce_lj[2] *= factor_lj;
        ttor_lj[0] *= factor_lj;
        ttor_lj[1] *= factor_lj;
        ttor_lj[2] *= factor_lj;

        f[i][0] += fforce_lj[0];
        f[i][1] += fforce_lj[1];
        f[i][2] += fforce_lj[2];
        tor[i][0] += ttor_lj[0];
        tor[i][1] += ttor_lj[1];
        tor[i][2] += ttor_lj[2];

        if (newton_pair || j < nlocal) {
          rtor_lj[0] *= factor_lj;
          rtor_lj[1] *= factor_lj;
          rtor_lj[2] *= factor_lj;
          f[j][0] -= fforce_lj[0];
          f[j][1] -= fforce_lj[1];
          f[j][2] -= fforce_lj[2];
          tor[j][0] += rtor_lj[0];
          tor[j][1] += rtor_lj[1];
          tor[j][2] += rtor_lj[2];
        }
        
        if (eflag) evdwl = factor_lj*one_eng;

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
            evdwl,0.0,fforce_lj[0],fforce_lj[1],fforce_lj[2],
            -r12[0],-r12[1],-r12[2]);

      } // end if rsq

      // end compute i,j for lj
    } // end loop over jj

    // long range self interactions of own charges
    j=i;
    jtype=itype;
    //bool sameAtom = true;
    factor_coul = 0.0;

    // loop over charges on particle i
    for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
      q1 = molFrameCharge[itype][s1];

      double labFrameSite1[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s1][0] != 0.0 ||
        molFrameSite[itype][s1][1] != 0.0 ||
        molFrameSite[itype][s1][2] != 0.0){
        double ms1[3] = {
          molFrameSite[itype][s1][0],
          molFrameSite[itype][s1][1],
          molFrameSite[itype][s1][2]
        };

      MathExtra::matvec(rotMat1, ms1, labFrameSite1);
      } // end if

      double rsite1[3] = {
        labFrameSite1[0]+x[i][0],
        labFrameSite1[1]+x[i][1],
        labFrameSite1[2]+x[i][2]
      };
    
      // loop over charges of site j
      for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
        double labFrameSite2[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[jtype][s2][0] != 0.0 ||
          molFrameSite[jtype][s2][1] != 0.0 ||
          molFrameSite[jtype][s2][2] != 0.0){
      
          double ms2[3] = {
            molFrameSite[jtype][s2][0],
            molFrameSite[jtype][s2][1],
            molFrameSite[jtype][s2][2]
          };

          MathExtra::matvec(rotMat1, ms2, labFrameSite2);
        } // end if

        double rsite2[3] = {
          labFrameSite2[0]+x[j][0],
          labFrameSite2[1]+x[j][1],
          labFrameSite2[2]+x[j][2]
        };

        // r12 = site center to site center vector
        r12[0] = rsite1[0]-rsite2[0];
        r12[1] = rsite1[1]-rsite2[1];
        r12[2] = rsite1[2]-rsite2[2];

        rsq = MathExtra::dot3(r12, r12);

        if ((rsq < cut_coulsq_global) && (s1 < s2)) {
          //double r2inv = 1.0/rsq;
          double q2 = molFrameCharge[jtype][s2];

          if (!ncoultablebits || rsq <= tabinnersq) {
            double r = sqrt(rsq);
            double grij = g_ewald * r;
            double expm2 = exp(-grij*grij);
            double t = 1.0 / (1.0 + EWALD_P*grij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            prefactor = qqrd2e * scale[itype][jtype] * q1*q2/r;
            forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
          } else {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table = ftable[itable] + fraction*dftable[itable];
            forcecoul = scale[itype][jtype] * q1*q2 * table;
            if (factor_coul < 1.0) {
              table = ctable[itable] + fraction*dctable[itable];
              prefactor = scale[itype][jtype] * q1*q2 * table;
              forcecoul -= (1.0-factor_coul)*prefactor;
            }
          }

          // if (!sameAtom) {
          //   fpair = forcecoul * r2inv;

          //   fforce_coul[0] = r12[0]*fpair; ///r12n;
          //   fforce_coul[1] = r12[1]*fpair; ///r12n;
          //   fforce_coul[2] = r12[2]*fpair; ///r12n;

          //   // F_parallel = F_tot . r_normalized
          //   f[i][0] += fforce_coul[0];
          //   f[i][1] += fforce_coul[1];
          //   f[i][2] += fforce_coul[2];

          //   // Torque = r x F_tot
          //   // torque on 1 = -pos1 x grad_pos1
          //   MathExtra::cross3(labFrameSite1, fforce_coul, ttor_coul);
          //   tor[i][0] += ttor_coul[0];
          //   tor[i][1] += ttor_coul[1];
          //   tor[i][2] += ttor_coul[2];

          //   if (newton_pair || j < nlocal) {
          //     // F_parallel = F_tot . r_normalized
          //     f[j][0] -= fforce_coul[0]; ///r12n;
          //     f[j][1] -= fforce_coul[1]; ///r12n;
          //     f[j][2] -= fforce_coul[2]; ///r12n;
          //     MathExtra::cross3(labFrameSite2, fforce_coul, ttor_coul);

          //     tor[j][0] -= ttor_coul[0];
          //     tor[j][1] -= ttor_coul[1];
          //     tor[j][2] -= ttor_coul[2];
          //   }
          // }

          if (eflag) {
            if (!ncoultablebits || rsq <= tabinnersq)
              ecoul = prefactor*erfc;
            else {
              table = etable[itable] + fraction*detable[itable];
              ecoul = scale[itype][jtype] * q1*q2 * table;
            }
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          }

          if (evflag) {
            // charges inside same bead should not contribute to virial
            //if (sameAtom) fpair = 0.0;
            fpair = 0.0;
            ev_tally(i,j,nlocal,newton_pair,
                    0.0,ecoul,fpair,r12[0],r12[1],r12[2]);
          }
        } // end if ((rsq < cut_coulsq_global) && ((!sameAtom) || (s1 < s2)))
      } // end for s2
    } // end for s1

  // end compute i,i for long range coul correction

  } //end loop over ii

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairMolcLong::allocate()
{
  #ifdef DEBUG
  printf("allocate\n");
  #endif

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");


memory->create(cut_ljsq, n+1,n+1,"pair:cut_ljsq");
memory->create(cut_coulsq, n+1,n+1,"pair:cut_coulsq");

memory->create(cut_coul, n+1,n+1,"pair:cut_coul");

memory->create(form,n+1,n+1,"pair:form");
memory->create(epsilon,n+1,n+1,"pair:epsilon");
memory->create(sigma,n+1,n+1,"pair:sigma");
memory->create(shape1,n+1,3,"pair:shape1");
memory->create(shape2,n+1,3,"pair:shape2");
memory->create(well,n+1,3,"pair:well");
memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
memory->create(lj1,n+1,n+1,"pair:lj1");
memory->create(lj2,n+1,n+1,"pair:lj2");
memory->create(lj3,n+1,n+1,"pair:lj3");
memory->create(lj4,n+1,n+1,"pair:lj4");
memory->create(offset,n+1,n+1,"pair:offset");

lshape = new double[n+1];
setwell = new int[n+1];
for (int i = 1; i <= n; i++) setwell[i] = 0;

nsites = new int[n+1];
for (int t = 1; t <= atom->ntypes; ++t)
  nsites[t] = 0;

molFrameSite = new double**[n+1];
molFrameCharge = new double*[n+1];


  memory->create(scale,n+1,n+1,"pair:scale");
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairMolcLong::settings(int narg, char **arg)
{
  // ensure mixing rules use arithmetic version
  mix_flag=ARITHMETIC;

  #ifdef DEBUG
  printf("settings\n");
  #endif
  
  
  if (narg != 5) error->all(FLERR,"Illegal pair_style command");

  gamma = utils::numeric(FLERR,arg[0],false,lmp);
  upsilon = utils::numeric(FLERR,arg[1],false,lmp)/2.0;
  mu = utils::numeric(FLERR,arg[2],false,lmp);
  cut_lj_global = utils::numeric(FLERR,arg[3],false,lmp);
  cut_coul_global = utils::numeric(FLERR,arg[4],false,lmp);

  #ifdef DEBUG
  printf("gamma: %f, upsilon: %f, mu: %f, cut_lj: %f, cut_coul: %f\n", gamma, upsilon, mu, cut_lj_global, cut_coul_global);
  #endif

  // coul sites are done in the pair coeffs now

  //int nCoulSites = force->inumeric(FLERR, arg[1]);
  //unsigned start_sitesspec_argcount = 2;

  //int atomType[nCoulSites+1];

  //nsites = new int[atom->ntypes+1];
  //for (int t = 1; t <= atom->ntypes; ++t)
  //  nsites[t] = 0;
  //
  //molFrameSite = new double**[atom->ntypes+1];
  //molFrameCharge = new double*[atom->ntypes+1];

  //int totsites = 0;
  //unsigned argcount = start_sitesspec_argcount;
  //for (int t = 1; t <= nCoulSites; ++t)
  //   {
  //     atomType[t] = force->inumeric(FLERR, arg[argcount++]);
  //     ++nsites[atomType[t]];

  //     argcount += 4;
  //   }

  // argcount = start_sitesspec_argcount;
  // for (int type = 1; type <= atom->ntypes; ++type)
  //   {
  //     molFrameSite[type] = new double*[nsites[type]+1];
  //     molFrameCharge[type] = new double[nsites[type]+1];

  //     for (int site = 1; site <= nsites[type]; ++site)
  //       {
  //         ++argcount;

  //         molFrameSite[type][site] = new double[3];
  //         molFrameSite[type][site][0] = force->numeric(FLERR,arg[argcount++]);
  //         molFrameSite[type][site][1] = force->numeric(FLERR,arg[argcount++]);
  //         molFrameSite[type][site][2] = force->numeric(FLERR,arg[argcount++]);

  //         molFrameCharge[type][site] = force->numeric(FLERR,arg[argcount++]);
  //       }

  //     totsites += nsites[type];
  //   }

  //allocate();

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]){
          cut_coul[i][j] = cut_coul_global;
          cut_lj[i][j] = cut_lj_global;
        }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   -------------------------------------------------------------------------*/

void PairMolcLong::coeff(int narg, char **arg)
{

  #ifdef DEBUG
  printf("coeff\n");
  #endif
  

  if (narg < 8) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  //int ilo,ihi,jlo,jhi;
  //force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  //force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  // only specific the i==j pairs, the rest are infered by mixing rules
  int i = utils::inumeric(FLERR,arg[0],false,lmp);
  int j = utils::inumeric(FLERR,arg[1],false,lmp);

  
  #ifdef DEBUG
  printf("i: %d, j %d\n", i,j);
  #endif

  if (i!=j) error->all(FLERR,"Incorrect args for pair coefficients");

  double cut_coul_one = cut_coul_global;
  double cut_lj_one = cut_lj_global;

  // first we have the Gay-Berne params

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double eia_one = utils::numeric(FLERR,arg[4],false,lmp);
  double eib_one = utils::numeric(FLERR,arg[5],false,lmp);
  double eic_one = utils::numeric(FLERR,arg[6],false,lmp);
  // double eja_one = force->numeric(FLERR,arg[7]);
  // double ejb_one = force->numeric(FLERR,arg[8]);
  // double ejc_one = force->numeric(FLERR,arg[9]);

  #ifdef DEBUG
  printf("i: %d, j: %d, e: %f, s: %f, e1 %f, e2: %f, e3: %f, cut_lj: %f, cut_coul: %f\n",i,j,epsilon_one,sigma_one,eia_one,eib_one,eic_one,cut_lj_one, cut_coul_one );
  #endif

  // set for i==j
  epsilon[i][j] = epsilon_one;
  sigma[i][j] = sigma_one;

  if (eia_one != 0.0 || eib_one != 0.0 || eic_one != 0.0) {
    well[i][0] = pow(eia_one,-1.0/mu);
    well[i][1] = pow(eib_one,-1.0/mu);
    well[i][2] = pow(eic_one,-1.0/mu);
    if (eia_one == eib_one && eib_one == eic_one) setwell[i] = 2;
    else setwell[i] = 1;
  }

  // now set the charges
  int nCoulSites = utils::inumeric(FLERR, arg[7],false,lmp);
  
  #ifdef DEBUG
  printf("nCoulSites: %d\n", nCoulSites);
  #endif

  int type = i;
  nsites[type] = nCoulSites;

  molFrameSite[type] = new double*[nsites[type]+1];
  molFrameCharge[type] = new double[nsites[type]+1];

  int argcount=8;
  for(int site=1; site<=nsites[type]; ++site){
    #ifdef DEBUG
    printf("site: %d\n", site);
    #endif
    molFrameSite[type][site] = new double[3];
    molFrameSite[type][site][0] = utils::numeric(FLERR,arg[argcount++],false,lmp);
    molFrameSite[type][site][1] = utils::numeric(FLERR,arg[argcount++],false,lmp);
    molFrameSite[type][site][2] = utils::numeric(FLERR,arg[argcount++],false,lmp);
    molFrameCharge[type][site]  = utils::numeric(FLERR,arg[argcount++],false,lmp);
 
    #ifdef DEBUG
    printf("type %d\n",type);
    printf("site %d: %f %f %f %f\n",site,molFrameSite[type][site][0],molFrameSite[type][site][1],molFrameSite[type][site][2],molFrameCharge[type][site] );
    #endif
  }


  // finally the cutoffs
  // if specificed

  if ((argcount+2)==narg){
    cut_lj_one   = utils::numeric(FLERR, arg[argcount+1],false,lmp);
    cut_coul_one = utils::numeric(FLERR, arg[argcount+2],false,lmp);
    }

  cut_coul[i][j] = cut_coul_one;
  cut_lj[i][j]   = cut_lj_one;

  scale[i][j] = 1.0;
  setflag[i][j]=1;

}


/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairMolcLong::init_style()
{

  #ifdef DEBUG
  printf("init_style\n");
  #endif

  // if (!atom->q_flag)
  //   error->all(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  neighbor->request(this);

  // from pair_gayberne
  // per-type shape precalculations
  // require that atom shapes are identical within each type
  // if shape = 0 for point particle, set shape = 1 as required by Gay-Berne
  for (int i = 1; i <= atom->ntypes; i++) {
    if (!atom->shape_consistency(i,shape1[i][0],shape1[i][1],shape1[i][2]))
      error->all(FLERR,
                 "Pair gayberne requires atoms with same type have same shape");
    if (shape1[i][0] == 0.0)
      shape1[i][0] = shape1[i][1] = shape1[i][2] = 1.0;
    shape2[i][0] = shape1[i][0]*shape1[i][0];
    shape2[i][1] = shape1[i][1]*shape1[i][1];
    shape2[i][2] = shape1[i][2]*shape1[i][2];
    lshape[i] = (shape1[i][0]*shape1[i][1]+shape1[i][2]*shape1[i][2]) *
      sqrt(shape1[i][0]*shape1[i][1]);
  }

  cut_coulsq_global = cut_coul_global * cut_coul_global;

  // set & error check interior rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0) {
    cut_respa = ((Respa *) update->integrate)->cutoff;
    if (cut_coul_global < cut_respa[3])
      error->all(FLERR,"Pair cutoff < Respa interior cutoff");
  } else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style is incompatible with KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairMolcLong::init_one(int i, int j)
{

  

  if (setwell[i] == 0 || setwell[j] == 0)
    error->all(FLERR,"Pair MOLC epsilon a,b,c coeffs are not all set");

  if (setflag[i][j] == 0){
    cut_coul[i][j] = mix_distance(cut_coul[i][i],cut_coul[j][j]);

    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);

    scale[i][j] = mix_distance(scale[i][i],scale[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  int ishape = 0;
  if (shape1[i][0] != shape1[i][1] ||
      shape1[i][0] != shape1[i][2] ||
      shape1[i][1] != shape1[i][2]) ishape = 1;
  if (setwell[i] == 1) ishape = 1;
  int jshape = 0;
  if (shape1[j][0] != shape1[j][1] ||
      shape1[j][0] != shape1[j][2] ||
      shape1[j][1] != shape1[j][2]) jshape = 1;
  if (setwell[j] == 1) jshape = 1;

  if (ishape == 0 && jshape == 0)
    form[i][i] = form[j][j] = form[i][j] = form[j][i] = SPHERE_SPHERE;
  else if (ishape == 0) {
    form[i][i] = SPHERE_SPHERE; form[j][j] = ELLIPSE_ELLIPSE;
    form[i][j] = SPHERE_ELLIPSE; form[j][i] = ELLIPSE_SPHERE;
  } else if (jshape == 0) {
    form[j][j] = SPHERE_SPHERE; form[i][i] = ELLIPSE_ELLIPSE;
    form[j][i] = SPHERE_ELLIPSE; form[i][j] = ELLIPSE_SPHERE;
  } else
  form[i][i] = form[j][j] = form[i][j] = form[j][i] = ELLIPSE_ELLIPSE;

  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];
  scale[j][i]  = scale[i][j];


  double cut = MAX(cut_lj[i][j],cut_coul[i][j]);
  
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
  cut_coulsq[i][j] = cut_coul[i][j] * cut_coul[i][j];
  
  cut_ljsq[j][i] = cut_ljsq[i][j];
  cut_coulsq[j][i] = cut_coulsq[i][j];
  
  #ifdef DEBUG
  printf("i: %d, j: %d, e: %f, s: %f, welli: %f %f %f, wellj: %f %f %f, cut_lj: %f, cut_coul: %f\n",i,j,epsilon[i][j],sigma[i][j],well[i][0],well[i][1],well[j][2],well[j][0],well[j][1],well[j][2],cut_lj[i][j], cut_coul[i][j] );
  #endif
  
  return cut;
}

/* ----------------------------------------------------------------------
   setup force tables used in compute routines
   ------------------------------------------------------------------------- */

void PairMolcLong::init_tables()
{
  int masklo,maskhi;
  double r,grij,expm2,derfc,rsw;
  double qqrd2e = force->qqrd2e;

  tabinnersq = tabinner*tabinner;
  init_bitmap(tabinner,cut_coul_global,ncoultablebits,
              masklo,maskhi,ncoulmask,ncoulshiftbits);

  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;

  // linear lookup tables of length N = 2^ncoultablebits
  // stored value = value at lower edge of bin
  // d values = delta from lower edge to upper edge of bin

  if (ftable) free_tables();

  memory->create(rtable,ntable,"pair:rtable");
  memory->create(ftable,ntable,"pair:ftable");
  memory->create(ctable,ntable,"pair:ctable");
  memory->create(etable,ntable,"pair:etable");
  memory->create(drtable,ntable,"pair:drtable");
  memory->create(dftable,ntable,"pair:dftable");
  memory->create(dctable,ntable,"pair:dctable");
  memory->create(detable,ntable,"pair:detable");

  if (cut_respa == NULL) {
    vtable = ptable = dvtable = dptable = NULL;
  } else {
    memory->create(vtable,ntable,"pair:vtable");
    memory->create(ptable,ntable,"pair:ptable");
    memory->create(dvtable,ntable,"pair:dvtable");
    memory->create(dptable,ntable,"pair:dptable");
  }

  union_int_float_t rsq_lookup;
  union_int_float_t minrsq_lookup;
  int itablemin;
  minrsq_lookup.i = 0 << ncoulshiftbits;
  minrsq_lookup.i |= maskhi;

  for (int i = 0; i < ntable; i++) {
    rsq_lookup.i = i << ncoulshiftbits;
    rsq_lookup.i |= masklo;
    if (rsq_lookup.f < tabinnersq) {
      rsq_lookup.i = i << ncoulshiftbits;
      rsq_lookup.i |= maskhi;
    }
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);
    if (cut_respa == NULL) {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      ctable[i] = qqrd2e/r;
      etable[i] = qqrd2e/r * derfc;
    } else {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      ctable[i] = 0.0;
      etable[i] = qqrd2e/r * derfc;
      ptable[i] = qqrd2e/r;
      vtable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]);
          ftable[i] += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          ctable[i] = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          ctable[i] = qqrd2e/r;
        }
      }
    }
    minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
  }

  tabinnersq = minrsq_lookup.f;

  int ntablem1 = ntable - 1;

  for (int i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0/(rtable[i+1] - rtable[i]);
    dftable[i] = ftable[i+1] - ftable[i];
    dctable[i] = ctable[i+1] - ctable[i];
    detable[i] = etable[i+1] - etable[i];
  }

  if (cut_respa) {
    for (int i = 0; i < ntablem1; i++) {
      dvtable[i] = vtable[i+1] - vtable[i];
      dptable[i] = ptable[i+1] - ptable[i];
    }
  }

  // get the delta values for the last table entries
  // tables are connected periodically between 0 and ntablem1

  drtable[ntablem1] = 1.0/(rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  if (cut_respa) {
    dvtable[ntablem1] = vtable[0] - vtable[ntablem1];
    dptable[ntablem1] = ptable[0] - ptable[ntablem1];
  }

  // get the correct delta values at itablemax
  // smallest r is in bin itablemin
  // largest r is in bin itablemax, which is itablemin-1,
  //   or ntablem1 if itablemin=0
  // deltas at itablemax only needed if corresponding rsq < cut*cut
  // if so, compute deltas between rsq and cut*cut

  double f_tmp,c_tmp,e_tmp,p_tmp,v_tmp;
  itablemin = minrsq_lookup.i & ncoulmask;
  itablemin >>= ncoulshiftbits;
  int itablemax = itablemin - 1;
  if (itablemin == 0) itablemax = ntablem1;
  rsq_lookup.i = itablemax << ncoulshiftbits;
  rsq_lookup.i |= maskhi;

  if (rsq_lookup.f < cut_coulsq_global) {
    rsq_lookup.f = cut_coulsq_global;
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);

    if (cut_respa == NULL) {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      c_tmp = qqrd2e/r;
      e_tmp = qqrd2e/r * derfc;
    } else {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      c_tmp = 0.0;
      e_tmp = qqrd2e/r * derfc;
      p_tmp = qqrd2e/r;
      v_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]);
          f_tmp += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          c_tmp = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          c_tmp = qqrd2e/r;
        }
      }
    }

    drtable[itablemax] = 1.0/(rsq_lookup.f - rtable[itablemax]);
    dftable[itablemax] = f_tmp - ftable[itablemax];
    dctable[itablemax] = c_tmp - ctable[itablemax];
    detable[itablemax] = e_tmp - etable[itablemax];
    if (cut_respa) {
      dvtable[itablemax] = v_tmp - vtable[itablemax];
      dptable[itablemax] = p_tmp - ptable[itablemax];
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairMolcLong::write_restart(FILE *fp)
{
  // WIP not fully implemented
  write_restart_settings(fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairMolcLong::read_restart(FILE *fp)
{
  // WIP not fully implemented
  read_restart_settings(fp);

  allocate();
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairMolcLong::write_restart_settings(FILE *fp)
{
  // WIP not fully implemented
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairMolcLong::read_restart_settings(FILE *fp)
{
  // WIP not fully implemented
  if (comm->me == 0) {
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   free memory for tables used in pair computations
   ------------------------------------------------------------------------- */

void PairMolcLong::free_tables()
{
  memory->destroy(rtable);
  memory->destroy(drtable);
  memory->destroy(ftable);
  memory->destroy(dftable);
  memory->destroy(ctable);
  memory->destroy(dctable);
  memory->destroy(etable);
  memory->destroy(detable);
  memory->destroy(vtable);
  memory->destroy(dvtable);
  memory->destroy(ptable);
  memory->destroy(dptable);
}

/* ---------------------------------------------------------------------- */

double PairMolcLong::single(int i, int j, int itype, int jtype,
                                     double rsq,
                                     double factor_coul, double factor_lj,
                                     double &fpair)
{
  double a1[3][3],b1[3][3],g1[3][3],a2[3][3],b2[3][3],g2[3][3],temp[3][3];
  double one_eng(0.0), r2inv(0.0), r6inv(0.0), forcelj(0.0);
  double fforce[3],ttor[3],rtor[3],r12[3];
  double forcecoul;
  double q1, q2;
  double *iquat,*jquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double qqrd2e = force->qqrd2e;


  double r, grij,expm2,t,erfc,prefactor;
  double fraction,table;
  int itable;

  double phicoul = 0.0;
  fpair = 0.0;

  bool sameAtom = i==j;

  // compute lj

  r12[0] = x[j][0]-x[i][0];
  r12[1] = x[j][1]-x[i][1];
  r12[2] = x[j][2]-x[i][2];

  // dont trust passed arg
  // needs to be recomputed incase this is being called by compute inter/molc
  rsq = r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2];

  // only compute if r < cutoff and not same atom
  if (rsq < cut_ljsq[itype][jtype] && (!sameAtom)){
    switch (form[itype][jtype]) {
      case SPHERE_SPHERE:
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        forcelj *= -r2inv;
        one_eng = r6inv*(r6inv*lj3[itype][jtype]-lj4[itype][jtype]) - offset[itype][jtype];
        fforce[0] = r12[0]*forcelj;
        fforce[1] = r12[1]*forcelj;
        fforce[2] = r12[2]*forcelj;
        ttor[0] = ttor[1] = ttor[2] = 0.0;
        rtor[0] = rtor[1] = rtor[2] = 0.0;
        break;

      case SPHERE_ELLIPSE:
        jquat = bonus[ellipsoid[j]].quat;
        MathExtra::quat_to_mat_trans(jquat,a2);
        MathExtra::diag_times3(well[jtype],a2,temp);
        MathExtra::transpose_times3(a2,temp,b2);
        MathExtra::diag_times3(shape2[jtype],a2,temp);
        MathExtra::transpose_times3(a2,temp,g2);
        one_eng = gayberne_lj(j,i,a2,b2,g2,r12,rsq,fforce,rtor);
        // printf("sphere-ell %f\n", one_eng);
        ttor[0] = ttor[1] = ttor[2] = 0.0;
        break;

      case ELLIPSE_SPHERE:
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat_trans(iquat,a1);
        MathExtra::diag_times3(well[itype],a1,temp);
        MathExtra::transpose_times3(a1,temp,b1);
        MathExtra::diag_times3(shape2[itype],a1,temp);
        MathExtra::transpose_times3(a1,temp,g1);
        one_eng = gayberne_lj(i,j,a1,b1,g1,r12,rsq,fforce,ttor);
        // printf("ell-sphere %f %f\n", one_eng, sqrt(rsq));
        rtor[0] = rtor[1] = rtor[2] = 0.0;
        break;

      default:
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat_trans(iquat,a1);
        MathExtra::diag_times3(well[itype],a1,temp);
        MathExtra::transpose_times3(a1,temp,b1);
        MathExtra::diag_times3(shape2[itype],a1,temp);
        MathExtra::transpose_times3(a1,temp,g1);

        jquat = bonus[ellipsoid[j]].quat;
        MathExtra::quat_to_mat_trans(jquat,a2);
        MathExtra::diag_times3(well[jtype],a2,temp);
        MathExtra::transpose_times3(a2,temp,b2);
        MathExtra::diag_times3(shape2[jtype],a2,temp);
        MathExtra::transpose_times3(a2,temp,g2);
        one_eng = gayberne_analytic(i,j,a1,a2,b1,b2,g1,g2,r12,rsq,
                                    fforce,ttor,rtor);
        break;
      }
  }else{
    one_eng = 0.0;
  }


  // compute coul energy

  // rotate site1 in lab frame
  double rotMat1[3][3];
  if (nsites[itype] > 0) {
    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat(iquat, rotMat1);
  }

  // rotate site2 in lab frame
  double rotMat2[3][3];
  if (nsites[jtype] > 0) {
    jquat = bonus[ellipsoid[j]].quat;
    MathExtra::quat_to_mat(jquat, rotMat2);
  }

  
  for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
    q1 = molFrameCharge[itype][s1];
    double labFrameSite1[3] = {0.0, 0.0, 0.0};
    if (molFrameSite[itype][s1][0] != 0.0 ||
        molFrameSite[itype][s1][1] != 0.0 ||
        molFrameSite[itype][s1][2] != 0.0)
    {
      double ms1[3] = {
        molFrameSite[itype][s1][0],
        molFrameSite[itype][s1][1],
        molFrameSite[itype][s1][2]
      };

      MathExtra::matvec(rotMat1, ms1, labFrameSite1);
    }

    double rsite1[3] = {
      labFrameSite1[0]+x[i][0],
      labFrameSite1[1]+x[i][1],
      labFrameSite1[2]+x[i][2]
    };

    for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
      double labFrameSite2[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[jtype][s2][0] != 0.0 ||
          molFrameSite[jtype][s2][1] != 0.0 ||
          molFrameSite[jtype][s2][2] != 0.0)
      {
        double ms2[3] = {
          molFrameSite[jtype][s2][0],
          molFrameSite[jtype][s2][1],
          molFrameSite[jtype][s2][2]
        };

        MathExtra::matvec(rotMat2, ms2, labFrameSite2);
      }

      double rsite2[3] = {
        labFrameSite2[0]+x[j][0],
        labFrameSite2[1]+x[j][1],
        labFrameSite2[2]+x[j][2]
      };

      // r12 = site center to site center vector
      r12[0] = rsite1[0]-rsite2[0];
      r12[1] = rsite1[1]-rsite2[1];
      r12[2] = rsite1[2]-rsite2[2];

      //domain->minimum_image(r12[0], r12[1], r12[2]);
      rsq = MathExtra::dot3(r12, r12);

      if ((rsq < cut_coulsq_global) && ((!sameAtom) || (s1 < s2))) {
        r2inv = 1.0/rsq;
        q2 = molFrameCharge[jtype][s2];

        if (!ncoultablebits || rsq <= tabinnersq) {
          r = sqrt(rsq);
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          prefactor = qqrd2e * scale[itype][jtype] * q1*q2/r;
          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          table = ftable[itable] + fraction*dftable[itable];
          forcecoul = scale[itype][jtype] * q1*q2 * table;
          if (factor_coul < 1.0) {
            table = ctable[itable] + fraction*dctable[itable];
            prefactor = scale[itype][jtype] * q1*q2 * table;
            forcecoul -= (1.0-factor_coul)*prefactor;
          }
        }

        if(!sameAtom){
        fpair += forcecoul * r2inv;
        }

        if (!ncoultablebits || rsq <= tabinnersq)
          phicoul += prefactor*erfc;
        else {
          table = etable[itable] + fraction*detable[itable];
          phicoul += scale[itype][jtype] * q1*q2 * table;
        }
        if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;

      }
    }
  }

  return phicoul + factor_lj*one_eng;
}

/* ---------------------------------------------------------------------- */

void *PairMolcLong::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_coul_global;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }

  if (strcmp(str,"nsites") == 0){
    dim = 1;
    return (void *) nsites;
  }

  if (strcmp(str, "molFrameSite") == 0){
    dim = 3;
    return (void *) molFrameSite;
  }

  if (strcmp(str, "molFrameCharge") == 0){
    dim = 2;
    return (void *) molFrameCharge;
  }

  return NULL;
}



/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor & rtor)
   based on rotation matrices a and precomputed matrices b and g
   if newton is off, rtor is not calculated for ghost atoms
   ------------------------------------------------------------------------- */

double PairMolcLong::gayberne_analytic(const int i,const int j,double a1[3][3],
                                       double a2[3][3], double b1[3][3],
                                       double b2[3][3], double g1[3][3],
                                       double g2[3][3], double *r12,
                                       const double rsq, double *fforce,
                                       double *ttor, double *rtor)
{
  double tempv[3], tempv2[3];
  double temp[3][3];
  double temp1,temp2,temp3;

  int *type = atom->type;
  int newton_pair = force->newton_pair;
  int nlocal = atom->nlocal;

  double r12hat[3];
  MathExtra::normalize3(r12,r12hat);
  double r = sqrt(rsq);

  // compute distance of closest approach

  double g12[3][3];
  MathExtra::plus3(g1,g2,g12);
  double kappa[3];
  int ierror = MathExtra::mldivide3(g12,r12,kappa);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = kappa[0]/r;
  tempv[1] = kappa[1]/r;
  tempv[2] = kappa[2]/r;
  double sigma12 = MathExtra::dot3(r12hat,tempv);
  sigma12 = pow(0.5*sigma12,-0.5);
  double h12 = r-sigma12;

  // printf("g1 %i %i %f %f %f\n", i, j, g1[0][0], g1[0][1], g1[0][2]);
  // printf("g1 %i %i %f %f %f\n", i, j, g1[1][0], g1[1][1], g1[1][2]);
  // printf("g1 %i %i %f %f %f\n", i, j, g1[3][0], g1[2][1], g1[2][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[0][0], g12[0][1], g12[0][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[1][0], g12[1][1], g12[1][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[3][0], g12[2][1], g12[2][2]);
  // printf("r %f %f %f\n", r12hat[0], r12hat[1], r12hat[2]);
  // printf("h %f %f %f\n", tempv[0], tempv[1], tempv[2]);
  // printf("s %f %f\n", sigma12, r);

  // energy
  // compute u_r

  double varrho = sigma[type[i]][type[j]]/(h12+gamma*sigma[type[i]][type[j]]);
  double varrho6 = pow(varrho,6.0);
  double varrho12 = varrho6*varrho6;
  double u_r = 4.0*epsilon[type[i]][type[j]]*(varrho12-varrho6);

  // compute eta_12

  double eta = 2.0*lshape[type[i]]*lshape[type[j]];
  double det_g12 = MathExtra::det3(g12);
  eta = pow(eta/det_g12,upsilon);

  // compute chi_12

  double b12[3][3];
  double iota[3];
  MathExtra::plus3(b1,b2,b12);
  ierror = MathExtra::mldivide3(b12,r12,iota);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = iota[0]/r;
  tempv[1] = iota[1]/r;
  tempv[2] = iota[2]/r;
  double chi = MathExtra::dot3(r12hat,tempv);
  chi = pow(chi*2.0,mu);

  // force
  // compute dUr/dr

  temp1 = (2.0*varrho12*varrho-varrho6*varrho)/sigma[type[i]][type[j]];
  temp1 = temp1*24.0*epsilon[type[i]][type[j]];
  double u_slj = temp1*pow(sigma12,3.0)/2.0;
  double dUr[3];
  temp2 = MathExtra::dot3(kappa,r12hat);
  double uslj_rsq = u_slj/rsq;
  dUr[0] = temp1*r12hat[0]+uslj_rsq*(kappa[0]-temp2*r12hat[0]);
  dUr[1] = temp1*r12hat[1]+uslj_rsq*(kappa[1]-temp2*r12hat[1]);
  dUr[2] = temp1*r12hat[2]+uslj_rsq*(kappa[2]-temp2*r12hat[2]);

  // compute dChi_12/dr

  double dchi[3];
  temp1 = MathExtra::dot3(iota,r12hat);
  temp2 = -4.0/rsq*mu*pow(chi,(mu-1.0)/mu);
  dchi[0] = temp2*(iota[0]-temp1*r12hat[0]);
  dchi[1] = temp2*(iota[1]-temp1*r12hat[1]);
  dchi[2] = temp2*(iota[2]-temp1*r12hat[2]);

  temp1 = -eta*u_r;
  temp3 = eta*chi;
  fforce[0] = temp1*dchi[0]-temp3*dUr[0];
  fforce[1] = temp1*dchi[1]-temp3*dUr[1];
  fforce[2] = temp1*dchi[2]-temp3*dUr[2];

  // torque for particle 1 and 2
  // compute dUr

  tempv[0] = -uslj_rsq*kappa[0];
  tempv[1] = -uslj_rsq*kappa[1];
  tempv[2] = -uslj_rsq*kappa[2];
  MathExtra::vecmat(kappa,g1,tempv2);
  MathExtra::cross3(tempv,tempv2,dUr);
  double dUr2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(kappa,g2,tempv2);
    MathExtra::cross3(tempv,tempv2,dUr2);
  }

  // compute d_chi

  MathExtra::vecmat(iota,b1,tempv);
  MathExtra::cross3(tempv,iota,dchi);
  dchi[0] *= temp2;
  dchi[1] *= temp2;
  dchi[2] *= temp2;
  double dchi2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(iota,b2,tempv);
    MathExtra::cross3(tempv,iota,dchi2);
    dchi2[0] *= temp2;
    dchi2[1] *= temp2;
    dchi2[2] *= temp2;
  }

  // compute d_eta

  double deta[3];
  deta[0] = deta[1] = deta[2] = 0.0;
  compute_eta_torque(g12,a1,shape2[type[i]],temp);
  temp1 = -eta*upsilon;
  for (int m = 0; m < 3; m++) {
    for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
    MathExtra::cross3(a1[m],tempv,tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // compute d_eta for particle 2

  double deta2[3];
  if (newton_pair || j < nlocal) {
    deta2[0] = deta2[1] = deta2[2] = 0.0;
    compute_eta_torque(g12,a2,shape2[type[j]],temp);
    for (int m = 0; m < 3; m++) {
      for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
      MathExtra::cross3(a2[m],tempv,tempv2);
      deta2[0] += tempv2[0];
      deta2[1] += tempv2[1];
      deta2[2] += tempv2[2];
    }
  }

  // torque

  temp1 = u_r*eta;
  temp2 = u_r*chi;
  temp3 = chi*eta;

  ttor[0] = (temp1*dchi[0]+temp2*deta[0]+temp3*dUr[0]) * -1.0;
  ttor[1] = (temp1*dchi[1]+temp2*deta[1]+temp3*dUr[1]) * -1.0;
  ttor[2] = (temp1*dchi[2]+temp2*deta[2]+temp3*dUr[2]) * -1.0;

  if (newton_pair || j < nlocal) {
    rtor[0] = (temp1*dchi2[0]+temp2*deta2[0]+temp3*dUr2[0]) * -1.0;
    rtor[1] = (temp1*dchi2[1]+temp2*deta2[1]+temp3*dUr2[1]) * -1.0;
    rtor[2] = (temp1*dchi2[2]+temp2*deta2[2]+temp3*dUr2[2]) * -1.0;
  }

  return temp1*chi;
}

/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor)
   between ellipsoid and lj particle
   ------------------------------------------------------------------------- */

double PairMolcLong::gayberne_lj(const int i,const int j,double a1[3][3],
                                 double b1[3][3],double g1[3][3],
                                 double *r12,const double rsq,double *fforce,
                                 double *ttor)
{
  double tempv[3], tempv2[3];
  double temp[3][3];
  double temp1,temp2,temp3;

  int *type = atom->type;

  double r12hat[3];
  MathExtra::normalize3(r12,r12hat);
  double r = sqrt(rsq);

  // compute distance of closest approach

  double g12[3][3];
  g12[0][0] = g1[0][0]+shape2[type[j]][0];
  g12[1][1] = g1[1][1]+shape2[type[j]][0];
  g12[2][2] = g1[2][2]+shape2[type[j]][0];
  g12[0][1] = g1[0][1]; g12[1][0] = g1[1][0];
  g12[0][2] = g1[0][2]; g12[2][0] = g1[2][0];
  g12[1][2] = g1[1][2]; g12[2][1] = g1[2][1];
  double kappa[3];
  int ierror = MathExtra::mldivide3(g12,r12,kappa);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = kappa[0]/r;
  tempv[1] = kappa[1]/r;
  tempv[2] = kappa[2]/r;
  double sigma12 = MathExtra::dot3(r12hat,tempv);
  sigma12 = pow(0.5*sigma12,-0.5);
  double h12 = r-sigma12;

  // printf("g1 %i %i %f %f %f\n", i, j, g1[0][0], g1[0][1], g1[0][2]);
  // printf("g1 %i %i %f %f %f\n", i, j, g1[1][0], g1[1][1], g1[1][2]);
  // printf("g1 %i %i %f %f %f\n", i, j, g1[3][0], g1[2][1], g1[2][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[0][0], g12[0][1], g12[0][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[1][0], g12[1][1], g12[1][2]);
  // printf("g12 %i %i %f %f %f\n", i, j, g12[3][0], g12[2][1], g12[2][2]);
  // printf("r %f %f %f\n", r12hat[0], r12hat[1], r12hat[2]);
  // printf("h %f %f %f\n", tempv[0], tempv[1], tempv[2]);
  // printf("f %f %f %f\n", shape2[type[j]][0], shape2[type[j]][1], shape2[type[j]][2]);
  // printf("s %f %f\n", sigma12, r);

  // energy
  // compute u_r

  double varrho = sigma[type[i]][type[j]]/(h12+gamma*sigma[type[i]][type[j]]);
  double varrho6 = pow(varrho,6.0);
  double varrho12 = varrho6*varrho6;
  double u_r = 4.0*epsilon[type[i]][type[j]]*(varrho12-varrho6);

  // compute eta_12

  double eta = 2.0*lshape[type[i]]*lshape[type[j]];
  double det_g12 = MathExtra::det3(g12);
  eta = pow(eta/det_g12,upsilon);

  // compute chi_12

  double b12[3][3];
  double iota[3];
  b12[0][0] = b1[0][0] + well[type[j]][0];
  b12[1][1] = b1[1][1] + well[type[j]][0];
  b12[2][2] = b1[2][2] + well[type[j]][0];
  b12[0][1] = b1[0][1]; b12[1][0] = b1[1][0];
  b12[0][2] = b1[0][2]; b12[2][0] = b1[2][0];
  b12[1][2] = b1[1][2]; b12[2][1] = b1[2][1];
  ierror = MathExtra::mldivide3(b12,r12,iota);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = iota[0]/r;
  tempv[1] = iota[1]/r;
  tempv[2] = iota[2]/r;
  double chi = MathExtra::dot3(r12hat,tempv);
  chi = pow(chi*2.0,mu);

  // force
  // compute dUr/dr

  temp1 = (2.0*varrho12*varrho-varrho6*varrho)/sigma[type[i]][type[j]];
  temp1 = temp1*24.0*epsilon[type[i]][type[j]];
  double u_slj = temp1*pow(sigma12,3.0)/2.0;
  double dUr[3];
  temp2 = MathExtra::dot3(kappa,r12hat);
  double uslj_rsq = u_slj/rsq;
  dUr[0] = temp1*r12hat[0]+uslj_rsq*(kappa[0]-temp2*r12hat[0]);
  dUr[1] = temp1*r12hat[1]+uslj_rsq*(kappa[1]-temp2*r12hat[1]);
  dUr[2] = temp1*r12hat[2]+uslj_rsq*(kappa[2]-temp2*r12hat[2]);

  // compute dChi_12/dr

  double dchi[3];
  temp1 = MathExtra::dot3(iota,r12hat);
  temp2 = -4.0/rsq*mu*pow(chi,(mu-1.0)/mu);
  dchi[0] = temp2*(iota[0]-temp1*r12hat[0]);
  dchi[1] = temp2*(iota[1]-temp1*r12hat[1]);
  dchi[2] = temp2*(iota[2]-temp1*r12hat[2]);

  temp1 = -eta*u_r;
  temp2 = eta*chi;
  fforce[0] = temp1*dchi[0]-temp2*dUr[0];
  fforce[1] = temp1*dchi[1]-temp2*dUr[1];
  fforce[2] = temp1*dchi[2]-temp2*dUr[2];

  // torque for particle 1 and 2
  // compute dUr

  tempv[0] = -uslj_rsq*kappa[0];
  tempv[1] = -uslj_rsq*kappa[1];
  tempv[2] = -uslj_rsq*kappa[2];
  MathExtra::vecmat(kappa,g1,tempv2);
  MathExtra::cross3(tempv,tempv2,dUr);

  // compute d_chi

  MathExtra::vecmat(iota,b1,tempv);
  MathExtra::cross3(tempv,iota,dchi);
  temp1 = -4.0/rsq;
  dchi[0] *= temp1;
  dchi[1] *= temp1;
  dchi[2] *= temp1;

  // compute d_eta

  double deta[3];
  deta[0] = deta[1] = deta[2] = 0.0;
  compute_eta_torque(g12,a1,shape2[type[i]],temp);
  temp1 = -eta*upsilon;
  for (int m = 0; m < 3; m++) {
    for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
    MathExtra::cross3(a1[m],tempv,tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // torque


  temp1 = u_r*eta;
  temp2 = u_r*chi;
  temp3 = chi*eta;

  ttor[0] = (temp1*dchi[0]+temp2*deta[0]+temp3*dUr[0]) * -1.0;
  ttor[1] = (temp1*dchi[1]+temp2*deta[1]+temp3*dUr[1]) * -1.0;
  ttor[2] = (temp1*dchi[2]+temp2*deta[2]+temp3*dUr[2]) * -1.0;

  return temp1*chi;
}

/* ----------------------------------------------------------------------
   torque contribution from eta
   computes trace in the last doc equation for the torque derivative
   code comes from symbolic solver dump
   m is g12, m2 is a_i, s is the shape for the particle
   ------------------------------------------------------------------------- */

void PairMolcLong::compute_eta_torque(double m[3][3], double m2[3][3],
                                      double *s, double ans[3][3])
{
  double den = m[1][0]*m[0][2]*m[2][1]-m[0][0]*m[1][2]*m[2][1]-
    m[0][2]*m[2][0]*m[1][1]+m[0][1]*m[2][0]*m[1][2]-
    m[1][0]*m[0][1]*m[2][2]+m[0][0]*m[1][1]*m[2][2];

  ans[0][0] = s[0]*(m[1][2]*m[0][1]*m2[0][2]+2.0*m[1][1]*m[2][2]*m2[0][0]-
                    m[1][1]*m2[0][2]*m[0][2]-2.0*m[1][2]*m2[0][0]*m[2][1]+
                    m2[0][1]*m[0][2]*m[2][1]-m2[0][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[0][1]+m[2][0]*m[1][2]*m2[0][1]+
                    m[1][0]*m2[0][2]*m[2][1]-m2[0][2]*m[2][0]*m[1][1])/den;

  ans[0][1] = s[0]*(m[0][2]*m2[0][0]*m[2][1]-m[2][2]*m2[0][0]*m[0][1]+
                    2.0*m[0][0]*m[2][2]*m2[0][1]-m[0][0]*m2[0][2]*m[1][2]-
                    2.0*m[2][0]*m[0][2]*m2[0][1]+m2[0][2]*m[1][0]*m[0][2]-
                    m[2][2]*m[1][0]*m2[0][0]+m[2][0]*m2[0][0]*m[1][2]+
                    m[2][0]*m2[0][2]*m[0][1]-m2[0][2]*m[0][0]*m[2][1])/den;

  ans[0][2] = s[0]*(m[0][1]*m[1][2]*m2[0][0]-m[0][2]*m2[0][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[0][1]+m[1][0]*m[0][2]*m2[0][1]-
                    m2[0][1]*m[0][0]*m[2][1]-m[2][0]*m[1][1]*m2[0][0]+
                    2.0*m[1][1]*m[0][0]*m2[0][2]-2.0*m[1][0]*m2[0][2]*m[0][1]+
                    m[1][0]*m[2][1]*m2[0][0]+m[2][0]*m2[0][1]*m[0][1])/den;

  ans[1][0] = s[1]*(-m[1][1]*m2[1][2]*m[0][2]+2.0*m[1][1]*m[2][2]*m2[1][0]+
                    m[1][2]*m[0][1]*m2[1][2]-2.0*m[1][2]*m2[1][0]*m[2][1]+
                    m2[1][1]*m[0][2]*m[2][1]-m2[1][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[1][1]+m[2][0]*m[1][2]*m2[1][1]-
                    m2[1][2]*m[2][0]*m[1][1]+m[1][0]*m2[1][2]*m[2][1])/den;

  ans[1][1] = s[1]*(m[0][2]*m2[1][0]*m[2][1]-m[0][1]*m[2][2]*m2[1][0]+
                    2.0*m[2][2]*m[0][0]*m2[1][1]-m2[1][2]*m[0][0]*m[1][2]-
                    2.0*m[2][0]*m2[1][1]*m[0][2]-m[1][0]*m[2][2]*m2[1][0]+
                    m[2][0]*m[1][2]*m2[1][0]+m[1][0]*m2[1][2]*m[0][2]-
                    m[0][0]*m2[1][2]*m[2][1]+m2[1][2]*m[0][1]*m[2][0])/den;

  ans[1][2] = s[1]*(m[0][1]*m[1][2]*m2[1][0]-m[0][2]*m2[1][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[1][1]+m[1][0]*m[0][2]*m2[1][1]+
                    2.0*m[1][1]*m[0][0]*m2[1][2]-m[0][0]*m2[1][1]*m[2][1]+
                    m[0][1]*m[2][0]*m2[1][1]-m2[1][0]*m[2][0]*m[1][1]-
                    2.0*m[1][0]*m[0][1]*m2[1][2]+m[1][0]*m2[1][0]*m[2][1])/den;

  ans[2][0] = s[2]*(-m[1][1]*m[0][2]*m2[2][2]+m[0][1]*m[1][2]*m2[2][2]+
                    2.0*m[1][1]*m2[2][0]*m[2][2]-m[0][1]*m2[2][1]*m[2][2]+
                    m[0][2]*m[2][1]*m2[2][1]-2.0*m2[2][0]*m[2][1]*m[1][2]-
                    m[1][0]*m2[2][1]*m[2][2]+m[1][2]*m[2][0]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][2]+m[2][1]*m[1][0]*m2[2][2])/den;

  ans[2][1] = s[2]*-(m[0][1]*m[2][2]*m2[2][0]-m[0][2]*m2[2][0]*m[2][1]-
                     2.0*m2[2][1]*m[0][0]*m[2][2]+m[1][2]*m2[2][2]*m[0][0]+
                     2.0*m2[2][1]*m[0][2]*m[2][0]+m[1][0]*m2[2][0]*m[2][2]-
                     m[1][0]*m[0][2]*m2[2][2]-m[1][2]*m[2][0]*m2[2][0]+
                     m[0][0]*m2[2][2]*m[2][1]-m2[2][2]*m[0][1]*m[2][0])/den;

  ans[2][2] = s[2]*(m[0][1]*m[1][2]*m2[2][0]-m[0][2]*m2[2][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[2][1]+m[1][0]*m[0][2]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][0]-m[2][1]*m2[2][1]*m[0][0]+
                    2.0*m[1][1]*m2[2][2]*m[0][0]+m[2][1]*m[1][0]*m2[2][0]+
                    m[2][0]*m[0][1]*m2[2][1]-2.0*m2[2][2]*m[1][0]*m[0][1])/den;
}


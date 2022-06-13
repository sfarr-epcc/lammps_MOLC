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

#include <cstring>
#include "bond_molc.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "suffix.h"
#include "atom_masks.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{NONE,LINEAR,SPLINE};

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
------------------------------------------------------------------------- */

void BondMolc::ev_tally(int i, int j, int nlocal, int newton_bond,
                        double ebond, double fbond,
                        double delx, double dely, double delz)
{
  double ebondhalf,v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) energy += ebond;
      else {
        ebondhalf = 0.5*ebond;
        if (i < nlocal) energy += ebondhalf;
        if (j < nlocal) energy += ebondhalf;
      }
    }
    if (eflag_atom) {
      ebondhalf = 0.5*ebond;
      if (newton_bond || i < nlocal) eatom[i] += ebondhalf;
      if (newton_bond || j < nlocal) eatom[j] += ebondhalf;
    }
  }

  if (vflag_either) {
    v[0] = delx*delx*fbond;
    v[1] = dely*dely*fbond;
    v[2] = delz*delz*fbond;
    v[3] = delx*dely*fbond;
    v[4] = delx*delz*fbond;
    v[5] = dely*delz*fbond;

    if (vflag_global) {
      if (newton_bond) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        vatom[i][0] += 0.5*v[0];
        vatom[i][1] += 0.5*v[1];
        vatom[i][2] += 0.5*v[2];
        vatom[i][3] += 0.5*v[3];
        vatom[i][4] += 0.5*v[4];
        vatom[i][5] += 0.5*v[5];
      }
      if (newton_bond || j < nlocal) {
        vatom[j][0] += 0.5*v[0];
        vatom[j][1] += 0.5*v[1];
        vatom[j][2] += 0.5*v[2];
        vatom[j][3] += 0.5*v[3];
        vatom[j][4] += 0.5*v[4];
        vatom[j][5] += 0.5*v[5];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
   also include virial contributions due to angular dependent bond terms
------------------------------------------------------------------------- */

void
BondMolc::ev_tally_ang(int i, int j, int nlocal, int newton_bond,
                       double ebond,
                       double fbond,  // r12 dependent
                       double fbond_angx1, // u.r dependent
                       double fbond_angy1, // u.r dependent
                       double fbond_angz1, // u.r dependent
                       double fbond_angx2, // r.u dependent
                       double fbond_angy2, // r.u dependent
                       double fbond_angz2, // r.u dependent
                       double delx, double dely, double delz,
                       double delx_angx1, double dely_angx1, double delz_angx1,
                       double delx_angy1, double dely_angy1, double delz_angy1,
                       double delx_angz1, double dely_angz1, double delz_angz1,
                       double delx_angx2, double dely_angx2, double delz_angx2,
                       double delx_angy2, double dely_angy2, double delz_angy2,
                       double delx_angz2, double dely_angz2, double delz_angz2)
{
  double ebondhalf,v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) energy += ebond;
      else {
        ebondhalf = 0.5*ebond;
        if (i < nlocal) energy += ebondhalf;
        if (j < nlocal) energy += ebondhalf;
      }
    }
    if (eflag_atom) {
      ebondhalf = 0.5*ebond;
      if (newton_bond || i < nlocal) eatom[i] += ebondhalf;
      if (newton_bond || j < nlocal) eatom[j] += ebondhalf;
    }
  }

  if (vflag_either) {
    // 0-6 ~ xx, yy, zz, xy, xz, yz.
    double xx = delx*delx;
    double yy = dely*dely;
    double zz = delz*delz;
    double xy = delx*dely;
    double xz = delx*delz;
    double yz = dely*delz;

    v[0] = xx*fbond;
    v[1] = yy*fbond;
    v[2] = zz*fbond;
    v[3] = xy*fbond;
    v[4] = xz*fbond;
    v[5] = yz*fbond;

    // vx1
    double gx = fbond_angx1*delx_angx1; // dU_dr_x
    double gy = fbond_angx1*dely_angx1; // dU_dr_y
    double gz = fbond_angx1*delz_angx1; // dU_dr_z
    double dx =  gx-gx*xx-gy*xy-gz*xz;  // (1-r x r) dU_dr
    double dy = -gx*xy+gy-gy*yy-gz*yz;
    double dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    // vy1
    gx = fbond_angy1*delx_angy1;
    gy = fbond_angy1*dely_angy1;
    gz = fbond_angy1*delz_angy1;
    dx =  gx-gx*xx-gy*xy-gz*xz;
    dy = -gx*xy+gy-gy*yy-gz*yz;
    dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    // vz1
    gx = fbond_angz1*delx_angz1;
    gy = fbond_angz1*dely_angz1;
    gz = fbond_angz1*delz_angz1;
    dx =  gx-gx*xx-gy*xy-gz*xz;
    dy = -gx*xy+gy-gy*yy-gz*yz;
    dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    // vx2
    gx = fbond_angx2*delx_angx2;
    gy = fbond_angx2*dely_angx2;
    gz = fbond_angx2*delz_angx2;
    dx =  gx-gx*xx-gy*xy-gz*xz;
    dy = -gx*xy+gy-gy*yy-gz*yz;
    dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    // vy2
    gx = fbond_angy2*delx_angy2;
    gy = fbond_angy2*dely_angy2;
    gz = fbond_angy2*delz_angy2;
    dx =  gx-gx*xx-gy*xy-gz*xz;
    dy = -gx*xy+gy-gy*yy-gz*yz;
    dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    // vz2
    gx = fbond_angz2*delx_angz2;
    gy = fbond_angz2*dely_angz2;
    gz = fbond_angz2*delz_angz2;
    dx =  gx-gx*xx-gy*xy-gz*xz;
    dy = -gx*xy+gy-gy*yy-gz*yz;
    dz = -gx*xz-gy*yz+gz-gz*zz;
    v[0] += delx*dx;
    v[1] += dely*dy;
    v[2] += delz*dz;
    v[3] += delx*dy;
    v[4] += delx*dz;
    v[5] += dely*dz;

    if (vflag_global) {
      if (newton_bond) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        vatom[i][0] += 0.5*v[0];
        vatom[i][1] += 0.5*v[1];
        vatom[i][2] += 0.5*v[2];
        vatom[i][3] += 0.5*v[3];
        vatom[i][4] += 0.5*v[4];
        vatom[i][5] += 0.5*v[5];
      }
      if (newton_bond || j < nlocal) {
        vatom[j][0] += 0.5*v[0];
        vatom[j][1] += 0.5*v[1];
        vatom[j][2] += 0.5*v[2];
        vatom[j][3] += 0.5*v[3];
        vatom[j][4] += 0.5*v[4];
        vatom[j][5] += 0.5*v[5];
      }
    }
  }
}


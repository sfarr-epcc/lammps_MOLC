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
   -------------------------------------------------------------------------*/

#include <math.h>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include "bond_ellipsoid.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"
#include "atom_vec_ellipsoid.h"
#include "math_extra.h"
#include "molecule.h"

using namespace LAMMPS_NS;

static const char cite_user_molc_package[] =
"USER-MOLC package:\n\n"
"Article{C9CP04120F,\n"
"  author = {Ricci, Matteo and Roscioni, Otello Maria and Querciagrossa, Lara and Zannoni, Claudio},\n"
"  title  = {MOLC. A reversible coarse grained approach using anisotropic beads for the modelling of organic functional materials},\n"
"  journal  = {Phys. Chem. Chem. Phys.},\n"
"  year  = 2019,\n"
"  volume = 21,\n"
"  issue  = 47,\n"
"  pages  = {26195-26211},\n"
"  publisher = {The Royal Society of Chemistry},\n"
"  doi  = {10.1039/C9CP04120F},\n"
"  url  = {http://dx.doi.org/10.1039/C9CP04120F},\n"
"  abstract  = {We describe the development and implementation of a coarse grained (CG) modelling approach where complex organic molecules, and particularly the \\pi-conjugated ones often employed in organic electronics, are modelled in terms of connected sets of attractive--repulsive biaxial Gay--Berne ellipsoidal beads. The CG model is aimed at reproducing realistically large scale morphologies (e.g. up to 100 nm thick films) for the materials involved, while being able to generate, with a back-mapping procedure, atomistic coordinates suitable, with limited effort, to be applied for charge transport calculations. Detailed methodology and an application to the common hole transporter material \\alpha-NPD are provided.}\n"
"}\n\n";

/* ---------------------------------------------------------------------- */

BondEllipsoid::BondEllipsoid(LAMMPS *lmp) : BondMolc(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_user_molc_package);

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"bond_ellipsoid requires atom style ellipsoid");

  verbose = false;
  numeric_gradients = false;
  use_numeric_gradients = false;
  delta_move = 0.001;
}

/* ---------------------------------------------------------------------- */

BondEllipsoid::~BondEllipsoid()
{
  if (allocated) {
    memory->destroy(setflag);
  }

  if (verbose) {
    //fclose(hdebug_coo);
    fclose(hdebug_val);
    fclose(hdebug_ene);
    fclose(hdebug_gra);
    fclose(hdebug_num);
  }
}

/* ---------------------------------------------------------------------- */

void BondEllipsoid::compute(int eflag, int vflag)
{
  typedef std::vector<double> bpv;
  int i1,i2,n,btype;
  double delx,dely,delz,ebond,fbond;
  double delx_angx1,dely_angx1,delz_angx1;
  double delx_angy1,dely_angy1,delz_angy1;
  double delx_angz1,dely_angz1,delz_angz1;
  double delx_angx2,dely_angx2,delz_angx2;
  double delx_angy2,dely_angy2,delz_angy2;
  double delx_angz2,dely_angz2,delz_angz2;
  double rsq,r;
  double *iquat,*jquat;
  double fbond_r = 0.0;
  double fbond_angx1 = 0.0;
  double fbond_angy1 = 0.0;
  double fbond_angz1 = 0.0;
  double fbond_angx2 = 0.0;
  double fbond_angy2 = 0.0;
  double fbond_angz2 = 0.0;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;

  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  //int *atom_type = atom->type;
  int *ellipsoid = atom->ellipsoid;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  int *tag = atom->tag;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    btype = bondlist[n][2];

    //assert(atom_type[i1] < atom_type[i2]);

    iquat = bonus[ellipsoid[i1]].quat;
    double rotMat1[3][3];
    MathExtra::quat_to_mat_trans(iquat, rotMat1);

    double x1[3] = { rotMat1[0][0],
      rotMat1[0][1],
      rotMat1[0][2] };

    double y1[3] = { rotMat1[1][0],
      rotMat1[1][1],
      rotMat1[1][2] };

    double z1[3] = { rotMat1[2][0],
      rotMat1[2][1],
      rotMat1[2][2] };

    jquat = bonus[ellipsoid[i2]].quat;
    double rotMat2[3][3];
    MathExtra::quat_to_mat_trans(jquat, rotMat2);

    double x2[3] = { rotMat2[0][0],
      rotMat2[0][1],
      rotMat2[0][2] };

    double y2[3] = { rotMat2[1][0],
      rotMat2[1][1],
      rotMat2[1][2] };

    double z2[3] = { rotMat2[2][0],
      rotMat2[2][1],
      rotMat2[2][2] };

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    double rv[] = {-delx/r, -dely/r, -delz/r};

    double var[pterms];
    double* v1[pterms];
    double* v2[pterms];

    var[0] = r; v1[0] = rv; v2[0] = rv;

#define SIGN(x) ((x) > 0.0 ? 1.0 : -1.0)

    double dxx = MathExtra::dot3(x1, x2);
    dxx = fabs(dxx) > 1.0 ? SIGN(dxx) : dxx;
    var[1] = dxx; v1[1] = x1; v2[1] = x2;

    double dxy = MathExtra::dot3(x1, y2);
    dxy = fabs(dxy) > 1.0 ? SIGN(dxy) : dxy;
    var[2] = dxy; v1[2] = x1; v2[2] = y2;

    double dxz = MathExtra::dot3(x1, z2);
    dxz = fabs(dxz) > 1.0 ? SIGN(dxz) : dxz;
    var[3] = dxz; v1[3] = x1; v2[3] = z2;

    double dyx = MathExtra::dot3(y1, x2);
    dyx = fabs(dyx) > 1.0 ? SIGN(dyx) : dyx;
    var[4] = dyx; v1[4] = y1; v2[4] = x2;

    double dyy = MathExtra::dot3(y1, y2);
    dyy = fabs(dyy) > 1.0 ? SIGN(dyy) : dyy;
    var[5] = dyy; v1[5] = y1; v2[5] = y2;

    double dyz = MathExtra::dot3(y1, z2);
    dyz = fabs(dyz) > 1.0 ? SIGN(dyz) : dyz;
    var[6] = dyz; v1[6] = y1; v2[6] = z2;

    double dzx = MathExtra::dot3(z1, x2);
    dzx = fabs(dzx) > 1.0 ? SIGN(dzx) : dzx;
    var[7] = dzx; v1[7] = z1; v2[7] = x2;

    double dzy = MathExtra::dot3(z1, y2);
    dzy = fabs(dzy) > 1.0 ? SIGN(dzy) : dzy;
    var[8] = dzy; v1[8] = z1; v2[8] = y2;

    double dzz = MathExtra::dot3(z1, z2);
    dzz = fabs(dzz) > 1.0 ? SIGN(dzz) : dzz;
    var[9] = dzz; v1[9] = z1; v2[9] = z2;

    double dxr = MathExtra::dot3(rv, x1);
    dxr = fabs(dxr) > 1.0 ? SIGN(dxr) : dxr;
    var[10] = dxr; v1[10] = x1; v2[10] = rv;

    double dyr = MathExtra::dot3(rv, y1);
    dyr = fabs(dyr) > 1.0 ? SIGN(dyr) : dyr;
    var[11] = dyr; v1[11] = y1; v2[11] = rv;

    double dzr = MathExtra::dot3(rv, z1);
    dzr = fabs(dzr) > 1.0 ? SIGN(dzr) : dzr;
    var[12] = dzr; v1[12] = z1; v2[12] = rv;

    double drx = MathExtra::dot3(rv, x2);
    drx = fabs(drx) > 1.0 ? SIGN(drx) : drx;
    var[13] = drx; v1[13] = rv; v2[13] = x2;

    double dry = MathExtra::dot3(rv, y2);
    dry = fabs(dry) > 1.0 ? SIGN(dry) : dry;
    var[14] = dry; v1[14] = rv; v2[14] = y2;

    double drz = MathExtra::dot3(rv, z2);
    drz = fabs(drz) > 1.0 ? SIGN(drz) : drz;
    var[15] = drz; v1[15] = rv; v2[15] = z2;

#undef SIGN

    std::vector<BondPotential*> bps = bondPotSet[btype];

    if (verbose) {
      fprintf(hdebug_val, "%li %i ", update->ntimestep, comm->me);
      fprintf(hdebug_ene, "%li %i ", update->ntimestep, comm->me);
      fprintf(hdebug_gra, "%li %i ", update->ntimestep, comm->me);
      fprintf(hdebug_num, "%li %i ", update->ntimestep, comm->me);

      fprintf(hdebug_val, "%i %i %i ", btype, tag[i1], tag[i2]);
      fprintf(hdebug_ene, "%i %i %i ", btype, tag[i1], tag[i2]);
      fprintf(hdebug_gra, "%i %i %i ", btype, tag[i1], tag[i2]);
    }

    double total_ene = 0.0;

    for (unsigned i = 0; i<bps.size(); ++i) {
      BondPotential* bp = bps[i];

      std::pair<double, double> p = bp->compute(var[i]);
      total_ene += p.first;
      fbond = -p.second;
      double fbrsq = fbond/rsq;

      if (bp->contribute_to_virial()) {
        if (i == 0) {
          fbond_r = fbond/r;
        } else if (i == 10) {
          double n[3];
          MathExtra::cross3(x1, rv, n);
          delx_angx1 = n[0];
          dely_angx1 = n[1];
          delz_angx1 = n[2];
          fbond_angx1 = fbrsq;
        } else if (i == 11) {
          double n[3];
          MathExtra::cross3(y1, rv, n);
          delx_angy1 = n[0];
          dely_angy1 = n[1];
          delz_angy1 = n[2];
          fbond_angy1 = fbrsq;
        } else if (i == 12) {
          double n[3];
          MathExtra::cross3(z1, rv, n);
          delx_angz1 = n[0];
          dely_angz1 = n[1];
          delz_angz1 = n[2];
          fbond_angz1 = fbrsq;
        } else if (i == 13) {
          double n[3];
          MathExtra::cross3(rv, x2, n);
          delx_angx2 = n[0];
          dely_angx2 = n[1];
          delz_angx2 = n[2];
          fbond_angx2 = fbrsq;
        } else if (i == 14) {
          double n[3];
          MathExtra::cross3(rv, y2, n);
          delx_angy2 = n[0];
          dely_angy2 = n[1];
          delz_angy2 = n[2];
          fbond_angy2 = fbrsq;
        } else if (i == 15) {
          double n[3];
          MathExtra::cross3(rv, z2, n);
          delx_angz2 = n[0];
          dely_angz2 = n[1];
          delz_angz2 = n[2];
          fbond_angz2 = fbrsq;
        }
      }

      if (std::isnan(fbond)) {
        printf("bond %s has NAN fbond var: %g index %i, bond atoms %i %i "
               "bond type %i\n",
               bp->get_name().c_str(), var[i], i, tag[i1], tag[i2], btype);
        fflush(stdout);
        exit(1);
      }

      if (verbose) {
        switch (i) {
        case(0):
          fprintf(hdebug_val, "%f ", r);
          break;
        case(1):
          fprintf(hdebug_val, "%f ", dxx);
          break;
        case(2):
          fprintf(hdebug_val, "%f ", dxy);
          break;
        case(3):
          fprintf(hdebug_val, "%f ", dxz);
          break;
        case(4):
          fprintf(hdebug_val, "%f ", dyx);
          break;
        case(5):
          fprintf(hdebug_val, "%f ", dyy);
          break;
        case(6):
          fprintf(hdebug_val, "%f ", dyz);
          break;
        case(7):
          fprintf(hdebug_val, "%f ", dzx);
          break;
        case(8):
          fprintf(hdebug_val, "%f ", dzy);
          break;
        case(9):
          fprintf(hdebug_val, "%f ", dzz);
          break;
        case(10):
          fprintf(hdebug_val, "%f ", dxr);
          break;
        case(11):
          fprintf(hdebug_val, "%f ", dyr);
          break;
        case(12):
          fprintf(hdebug_val, "%f ", dzr);
          break;
        case(13):
          fprintf(hdebug_val, "%f ", drx);
          break;
        case(14):
          fprintf(hdebug_val, "%f ", dry);
          break;
        case(15):
          fprintf(hdebug_val, "%f ", drz);
          break;
        }

        fprintf(hdebug_ene, "%f ", p.first);
        fprintf(hdebug_gra, "%f ", -p.second);
      }

      // apply force to each of 2 atoms
      bpv bpf1 = bp->potential()->force1(r, fbond, v1[i], v2[i]);
      bpv bpf2 = bp->potential()->force2(r, fbond, v1[i], v2[i]);
      bpv bpt1 = bp->potential()->torque1(fbond, v1[i], v2[i]);
      bpv bpt2 = bp->potential()->torque2(fbond, v1[i], v2[i]);

      if (!use_numeric_gradients) {
        if (newton_bond || (i1 < nlocal)) {
          f[i1][0] += bpf1[0];
          f[i1][1] += bpf1[1];
          f[i1][2] += bpf1[2];
          tor[i1][0] += bpt1[0];
          tor[i1][1] += bpt1[1];
          tor[i1][2] += bpt1[2];
        }

        if (newton_bond || (i2 < nlocal)) {
          f[i2][0] += bpf2[0];
          f[i2][1] += bpf2[1];
          f[i2][2] += bpf2[2];
          tor[i2][0] += bpt2[0];
          tor[i2][1] += bpt2[1];
          tor[i2][2] += bpt2[2];
        }
      }
    }

    if (numeric_gradients) {
      bpv bpf1 = force1_numeric(btype, i1, i2, delta_move);
      bpv bpf2 = force2_numeric(btype, i1, i2, delta_move);
      bpv bpt1 = torque1_numeric(btype, i1, i2, delta_move);
      bpv bpt2 = torque2_numeric(btype, i1, i2, delta_move);

      if (verbose) {
        fprintf(hdebug_num, " %g %g %g %g %g %g %g %g %g %g %g %g %g",
                total_ene,
                bpf1[0],
                bpf1[1],
                bpf1[2],
                bpf2[0],
                bpf2[1],
                bpf2[2],
                bpt1[0],
                bpt1[1],
                bpt1[2],
                bpt2[0],
                bpt2[1],
                bpt2[2]);
      }

      if (use_numeric_gradients) {
        if (newton_bond || (i1 < nlocal)) {
          f[i1][0] += bpf1[0];
          f[i1][1] += bpf1[1];
          f[i1][2] += bpf1[2];
          tor[i1][0] += bpt1[0];
          tor[i1][1] += bpt1[1];
          tor[i1][2] += bpt1[2];
        }

        if (newton_bond || (i2 < nlocal)) {
          f[i2][0] += bpf2[0];
          f[i2][1] += bpf2[1];
          f[i2][2] += bpf2[2];
          tor[i2][0] += bpt2[0];
          tor[i2][1] += bpt2[1];
          tor[i2][2] += bpt2[2];
        }
      }
    }

    if (verbose) {
      fprintf(hdebug_val, "\n");
      fprintf(hdebug_ene, "\n");
      fprintf(hdebug_gra, "\n");
      fprintf(hdebug_num, "\n");
    }
    if (eflag) ebond = total_ene;

    if (evflag) ev_tally_ang(i1,i2,nlocal,newton_bond,
                             ebond,
                             fbond_r,
                             fbond_angx1,
                             fbond_angy1,
                             fbond_angz1,
                             fbond_angx2,
                             fbond_angy2,
                             fbond_angz2,
                             delx,dely,delz,
                             delx_angx1,dely_angx1,delz_angx1,
                             delx_angy1,dely_angy1,delz_angy1,
                             delx_angz1,dely_angz1,delz_angz1,
                             delx_angx2,dely_angx2,delz_angx2,
                             delx_angy2,dely_angy2,delz_angy2,
                             delx_angz2,dely_angz2,delz_angz2);
  }
}

/* ---------------------------------------------------------------------- */

void BondEllipsoid::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  bondPotSet.resize(atom->nbondtypes + 1); // indices start from 1

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

char* BondEllipsoid::hfilename(char const* prefix)
{
  char *filename = (char*)malloc(strlen(prefix)+3+1);
  sprintf(filename, "%s_%02i", prefix, comm->me);
  return filename;
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void BondEllipsoid::settings(int narg, char **arg)
{
  if (narg > 4) error->all(FLERR,"Illegal bond_style command");

  int i = 0;
  while (i < narg) {
    if (strcmp(arg[i],"verbose") == 0) {
      verbose = true;
    } else if (strcmp(arg[i],"numeric_compute") == 0) {
      numeric_gradients = true;
      delta_move = atof(arg[i+1]);
      i++;
    } else if (strcmp(arg[i],"numeric_use") == 0) {
      numeric_gradients = true;
      use_numeric_gradients = true;
      i++;
    } else error->all(FLERR,"Unknown arg in bond style ellipsoid");

    i++;
  }

  if (verbose) {
    hdebug_val = fopen(hfilename("bond_ellipsoid_val"), "w");
    hdebug_ene = fopen(hfilename("bond_ellipsoid_ene"), "w");
    hdebug_gra = fopen(hfilename("bond_ellipsoid_gra"), "w");
    hdebug_num = fopen(hfilename("bond_ellipsoid_num"), "w");

    fprintf(hdebug_val, "# ts me type i1 i2  rr xx xy xz yx yy yz zx zy zz "
            "xr yr zr rx ry rz\n");

    fprintf(hdebug_ene, "# ts me type i1 i2  rr xx xy xz yx yy yz zx zy zz "
            "xr yr zr rx ry rz\n");

    fprintf(hdebug_gra, "# ts me type i1 i2  rr xx xy xz yx yy yz zx zy zz "
            "xr yr zr rx ry rz\n");

    fprintf(hdebug_num, "# ts me ene nfx1 nfy1 nfz1 nfx2 nfy2 nfz2 "
            "ntx1 nty1 ntz1 ntx2 nty2 ntz2 \n");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
   -------------------------------------------------------------------------*/

void BondEllipsoid::coeff(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  std::vector<BondPotential*> pset;

  int argcount = 1;
  for (int i = 0; i < pterms; ++i) {
    if (narg < argcount+1) {
      printf("unsufficient terms specified %i %i %i %i\n.Abort\n",
             narg, argcount+1, pterms+1, i);
      exit(1);
    }

    std::string ptype = std::string(arg[argcount++]);

    if (!factory.find(ptype)) {
      printf("cannot find potential type %s for term %i bond coeff %s\n."
             "Abort\n",
             ptype.c_str(), i, arg[0]);
      exit(1);
    }

    int rank = atoi(arg[argcount++]);

    BondPotential* bp = factory.create(ptype, rank, i);

    if (narg < argcount + bp->nparams()) {
      printf("unsufficient parameters specified for %s\n.Abort\n",
             ptype.c_str());
      exit(1);
    }

    for (int j = 0; j < bp->nparams(); ++j) {
      bp->get_params().push_back(utils::numeric(FLERR, arg[argcount++],false,lmp));
    }

    // params are hardcoded in some cases (cheb, herm), so must avoid sigsegv
    if (bp->get_params().size() != bp->maxparams()) {
      bp->get_params().resize(bp->maxparams());
    }

    pset.push_back(bp);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    bondPotSet[i] = pset;
    setflag[i] = 1;

    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check if pair defined and special_bond settings are valid
   -------------------------------------------------------------------------*/

void BondEllipsoid::init_style()
{
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
   -------------------------------------------------------------------------*/

double BondEllipsoid::equilibrium_distance(int i)
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
   -------------------------------------------------------------------------*/

void BondEllipsoid::write_restart(FILE *fp)
{
  // This function is not implemented, however it does not need to be 
  // implemented if we correctly set the bond_style and coeffs
  // when the simulation is restarted.


  //error->all(FLERR,"BondEllipsoid::write_restart not implemented");
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
   -------------------------------------------------------------------------*/

void BondEllipsoid::read_restart(FILE *fp)
{
  // This function is not implemented, however it does not need to be 
  // implemented if we correctly set the bond_style and coeffs
  // when the simulation is restarted.

  //error->all(FLERR,"BondEllipsoid::read_restart not implemented");
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
   -------------------------------------------------------------------------*/

void BondEllipsoid::write_data(FILE *fp)
{
  error->all(FLERR,"BondEllipsoid::write_data not implemented");
}

/* ---------------------------------------------------------------------- */

double BondEllipsoid::single(int btype,
                             double rsq_unused,
                             int i1, int i2,
                             double &fforce)
{
  double delx,dely,delz;
  //double ebond(0.0)
  double fbond(0.0);
  double r, rsq;
  double *iquat,*jquat;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;

  double **x = atom->x;
  //double **f = atom->f;
  //double **tor = atom->torque;
  int *ellipsoid = atom->ellipsoid;

  iquat = bonus[ellipsoid[i1]].quat;
  double rotMat1[3][3];
  MathExtra::quat_to_mat_trans(iquat, rotMat1);

  double x1[3] = { rotMat1[0][0],
    rotMat1[0][1],
    rotMat1[0][2] };

  double y1[3] = { rotMat1[1][0],
    rotMat1[1][1],
    rotMat1[1][2] };

  double z1[3] = { rotMat1[2][0],
    rotMat1[2][1],
    rotMat1[2][2] };

  jquat = bonus[ellipsoid[i2]].quat;
  double rotMat2[3][3];
  MathExtra::quat_to_mat_trans(jquat, rotMat2);

  double x2[3] = { rotMat2[0][0],
    rotMat2[0][1],
    rotMat2[0][2] };

  double y2[3] = { rotMat2[1][0],
    rotMat2[1][1],
    rotMat2[1][2] };

  double z2[3] = { rotMat2[2][0],
    rotMat2[2][1],
    rotMat2[2][2] };

  delx = x[i1][0] - x[i2][0];
  dely = x[i1][1] - x[i2][1];
  delz = x[i1][2] - x[i2][2];

  rsq = delx*delx + dely*dely + delz*delz;
  r = sqrt(rsq);

  // force & energy
  double rv[] = {-delx/r, -dely/r, -delz/r};

  double var[pterms];
  double* v1[pterms];
  double* v2[pterms];

  var[0] = r; v1[0] = rv; v2[0] = rv;

#define SIGN(x) ((x) > 0.0 ? 1.0 : -1.0)

  double dxx = MathExtra::dot3(x1, x2);
  dxx = fabs(dxx) > 1.0 ? SIGN(dxx) : dxx;
  var[1] = dxx; v1[1] = x1; v2[1] = x2;

  double dxy = MathExtra::dot3(x1, y2);
  dxy = fabs(dxy) > 1.0 ? SIGN(dxy) : dxy;
  var[2] = dxy; v1[2] = x1; v2[2] = y2;

  double dxz = MathExtra::dot3(x1, z2);
  dxz = fabs(dxz) > 1.0 ? SIGN(dxz) : dxz;
  var[3] = dxz; v1[3] = x1; v2[3] = z2;

  double dyx = MathExtra::dot3(y1, x2);
  dyx = fabs(dyx) > 1.0 ? SIGN(dyx) : dyx;
  var[4] = dyx; v1[4] = y1; v2[4] = x2;

  double dyy = MathExtra::dot3(y1, y2);
  dyy = fabs(dyy) > 1.0 ? SIGN(dyy) : dyy;
  var[5] = dyy; v1[5] = y1; v2[5] = y2;

  double dyz = MathExtra::dot3(y1, z2);
  dyz = fabs(dyz) > 1.0 ? SIGN(dyz) : dyz;
  var[6] = dyz; v1[6] = y1; v2[6] = z2;

  double dzx = MathExtra::dot3(z1, x2);
  dzx = fabs(dzx) > 1.0 ? SIGN(dzx) : dzx;
  var[7] = dzx; v1[7] = z1; v2[7] = x2;

  double dzy = MathExtra::dot3(z1, y2);
  dzy = fabs(dzy) > 1.0 ? SIGN(dzy) : dzy;
  var[8] = dzy; v1[8] = z1; v2[8] = y2;

  double dzz = MathExtra::dot3(z1, z2);
  dzz = fabs(dzz) > 1.0 ? SIGN(dzz) : dzz;
  var[9] = dzz; v1[9] = z1; v2[9] = z2;

  double dxr = MathExtra::dot3(rv, x1);
  dxr = fabs(dxr) > 1.0 ? SIGN(dxr) : dxr;
  var[10] = dxr; v1[10] = x1; v2[10] = rv;

  double dyr = MathExtra::dot3(rv, y1);
  dyr = fabs(dyr) > 1.0 ? SIGN(dyr) : dyr;
  var[11] = dyr; v1[11] = y1; v2[11] = rv;

  double dzr = MathExtra::dot3(rv, z1);
  dzr = fabs(dzr) > 1.0 ? SIGN(dzr) : dzr;
  var[12] = dzr; v1[12] = z1; v2[12] = rv;

  double drx = MathExtra::dot3(rv, x2);
  drx = fabs(drx) > 1.0 ? SIGN(drx) : drx;
  var[13] = drx; v1[13] = rv; v2[13] = x2;

  double dry = MathExtra::dot3(rv, y2);
  dry = fabs(dry) > 1.0 ? SIGN(dry) : dry;
  var[14] = dry; v1[14] = rv; v2[14] = y2;

  double drz = MathExtra::dot3(rv, z2);
  drz = fabs(drz) > 1.0 ? SIGN(drz) : drz;
  var[15] = drz; v1[15] = rv; v2[15] = z2;

#undef SIGN

  std::vector<BondPotential*> bps = bondPotSet[btype];

  double ene = 0.0;
  for (unsigned i = 0; i<bps.size(); ++i) {
    BondPotential* bp = bps[i];

    std::pair<double, double> p = bp->compute(var[i]);
    ene += p.first;
    fbond = -p.second;
    fforce += fbond;

    if (std::isnan(fbond)) {
      printf("bond %s has NAN fbond %g %i\n",
             bp->get_name().c_str(), var[i], i);

      printf("BondEllipsoid::compute axes1\n %f %f %f\n %f %f %f\n %f %f %f\n",
             x1[0], x1[1], x1[2],
             y1[0], y1[1], y1[2],
             z1[0], z1[1], z1[2]);
      printf("BondEllipsoid::compute axes2\n %f %f %f\n %f %f %f\n %f %f %f\n",
             x2[0], x2[1], x2[2],
             y2[0], y2[1], y2[2],
             z2[0], z2[1], z2[2]);

      fflush(stdout);
      exit(1);
    }
  }

  fforce = fbond;

  return ene;
}

std::vector<double>
BondEllipsoid::force1_numeric(int btype,
                              int atom1, int atom2,
                              double delta_move) {
  double x1save,y1save,z1save;
  double delx,dely,delz,rsq;
  double **x = atom->x;
  Bond *bond = force->bond;
  double eng, fbond;

  x1save = x[atom1][0];
  y1save = x[atom1][1];
  z1save = x[atom1][2];

  delx = x[atom2][0] - x[atom1][0];
  dely = x[atom2][1] - x[atom1][1];
  delz = x[atom2][2] - x[atom1][2];
  rsq = delx*delx + dely*dely + delz*delz;

  double ene_comp_trasl_1[3][2] = { { 0.0, 0.0 },
    { 0.0, 0.0 },
    { 0.0, 0.0 } };

  for (int comp = 0; comp < 3; comp++) {
    for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
      double delta = delta_sign*delta_move;

      x[atom1][0] = x1save + delta*(comp == 0 ? 1.0 : 0);
      x[atom1][1] = y1save + delta*(comp == 1 ? 1.0 : 0);
      x[atom1][2] = z1save + delta*(comp == 2 ? 1.0 : 0);

      if (btype > 0)
        eng = bond->single(btype,rsq,atom1,atom2,fbond);
      else eng = fbond = 0.0;

      if (delta_sign == -1)     ene_comp_trasl_1[comp][0] = eng;
      else if (delta_sign == 1)	ene_comp_trasl_1[comp][1] = eng;
    }

    x[atom1][0] = x1save;
    x[atom1][1] = y1save;
    x[atom1][2] = z1save;
  }

  double dspan = 2.0*delta_move;

  std::vector<double> f1(3);
  f1[0] = -(ene_comp_trasl_1[0][1]-ene_comp_trasl_1[0][0])/dspan;
  f1[1] = -(ene_comp_trasl_1[1][1]-ene_comp_trasl_1[1][0])/dspan;
  f1[2] = -(ene_comp_trasl_1[2][1]-ene_comp_trasl_1[2][0])/dspan;

  return f1;
}

std::vector<double>
BondEllipsoid::force2_numeric(int btype,
                              int atom1, int atom2,
                              double delta_move) {
  double x2save,y2save,z2save;
  double delx,dely,delz,rsq;
  double **x = atom->x;
  Bond *bond = force->bond;
  double eng, fbond;

  x2save = x[atom2][0];
  y2save = x[atom2][1];
  z2save = x[atom2][2];

  delx = x[atom2][0] - x[atom1][0];
  dely = x[atom2][1] - x[atom1][1];
  delz = x[atom2][2] - x[atom1][2];
  rsq = delx*delx + dely*dely + delz*delz;

  double ene_comp_trasl_2[3][2] = { { 0.0, 0.0 },
    { 0.0, 0.0 },
    { 0.0, 0.0 } };

  for (int comp = 0; comp < 3; comp++) {
    for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
      double delta = delta_sign*delta_move;

      x[atom2][0] = x2save + delta*(comp == 0 ? 1.0 : 0);
      x[atom2][1] = y2save + delta*(comp == 1 ? 1.0 : 0);
      x[atom2][2] = z2save + delta*(comp == 2 ? 1.0 : 0);

      if (btype > 0)
        eng = bond->single(btype,rsq,atom1,atom2,fbond);
      else eng = fbond = 0.0;

      if (delta_sign == -1)     ene_comp_trasl_2[comp][0] = eng;
      else if (delta_sign == 1)	ene_comp_trasl_2[comp][1] = eng;
    }

    x[atom2][0] = x2save;
    x[atom2][1] = y2save;
    x[atom2][2] = z2save;
  }

  double dspan = 2.0*delta_move;

  std::vector<double> f2(3);
  f2[0] = -(ene_comp_trasl_2[0][1]-ene_comp_trasl_2[0][0])/dspan;
  f2[1] = -(ene_comp_trasl_2[1][1]-ene_comp_trasl_2[1][0])/dspan;
  f2[2] = -(ene_comp_trasl_2[2][1]-ene_comp_trasl_2[2][0])/dspan;

  return f2;
}

std::vector<double>
BondEllipsoid::torque1_numeric(int btype,
                               int atom1, int atom2,
                               double delta_move) {
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  Bond *bond = force->bond;
  double eng, fbond;
  double runit[3];

  double *quat1 = bonus[ellipsoid[atom1]].quat;
  double q1save[4] = {quat1[0], quat1[1], quat1[2], quat1[3]};

  double ene_comp_rot_1[3][2] = { { 0.0, 0.0 },
    { 0.0, 0.0 },
    { 0.0, 0.0 } };

  for (int comp = 0; comp < 3; comp++) {
    runit[0] = comp == 0 ? 1.0 : 0.0;
    runit[1] = comp == 1 ? 1.0 : 0.0;
    runit[2] = comp == 2 ? 1.0 : 0.0;

    for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
      double delta_angle = delta_sign*delta_move;

      double q[4] = {0.0, 0.0, 0.0, 0.0};

      MathExtra::axisangle_to_quat(runit, delta_angle, q);
      MathExtra::quatquat(q, q1save, quat1);
      MathExtra::qnormalize(quat1);

      eng = bond->single(btype,0.0,atom1,atom2,fbond);

      if (delta_sign == -1)     ene_comp_rot_1[comp][0] = eng;
      else if (delta_sign == 1)	ene_comp_rot_1[comp][1] = eng;
    }

    quat1[0] = q1save[0];
    quat1[1] = q1save[1];
    quat1[2] = q1save[2];
    quat1[3] = q1save[3];
  }

  double dspan = 2.0*delta_move;

  std::vector<double> t1(3);
  t1[0] = -(ene_comp_rot_1[0][1]-ene_comp_rot_1[0][0])/dspan;
  t1[1] = -(ene_comp_rot_1[1][1]-ene_comp_rot_1[1][0])/dspan;
  t1[2] = -(ene_comp_rot_1[2][1]-ene_comp_rot_1[2][0])/dspan;

  return t1;
}

std::vector<double>
BondEllipsoid::torque2_numeric(int btype,
                               int atom1, int atom2,
                               double delta_move) {
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  Bond *bond = force->bond;
  double eng, fbond;
  double runit[3];

  double *quat2 = bonus[ellipsoid[atom2]].quat;
  double q2save[4] = {quat2[0], quat2[1], quat2[2], quat2[3]};

  double ene_comp_rot_2[3][2] = { { 0.0, 0.0 },
    { 0.0, 0.0 },
    { 0.0, 0.0 } };

  for (int comp = 0; comp < 3; comp++) {
    runit[0] = comp == 0 ? 1.0 : 0.0;
    runit[1] = comp == 1 ? 1.0 : 0.0;
    runit[2] = comp == 2 ? 1.0 : 0.0;

    for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
      double delta_angle = delta_sign*delta_move;

      double q[4] = {0.0, 0.0, 0.0, 0.0};

      MathExtra::axisangle_to_quat(runit, delta_angle, q);
      MathExtra::quatquat(q, q2save, quat2);
      MathExtra::qnormalize(quat2);

      eng = bond->single(btype,0.0,atom1,atom2,fbond);

      if (delta_sign == -1)	ene_comp_rot_2[comp][0] = eng;
      else if (delta_sign == 1)	ene_comp_rot_2[comp][1] = eng;
    }

    quat2[0] = q2save[0];
    quat2[1] = q2save[1];
    quat2[2] = q2save[2];
    quat2[3] = q2save[3];
  }

  double dspan = 2.0*delta_move;

  std::vector<double> t2(3);
  t2[0] = -(ene_comp_rot_2[0][1]-ene_comp_rot_2[0][0])/dspan;
  t2[1] = -(ene_comp_rot_2[1][1]-ene_comp_rot_2[1][0])/dspan;
  t2[2] = -(ene_comp_rot_2[2][1]-ene_comp_rot_2[2][0])/dspan;

  return t2;
}

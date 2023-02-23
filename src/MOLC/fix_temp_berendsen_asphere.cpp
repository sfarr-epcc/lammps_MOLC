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

#include "fix_temp_berendsen_asphere.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "math.h"
#include "modify.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum { NOBIAS, BIAS };
enum { CONSTANT, EQUAL };

/* ---------------------------------------------------------------------- */

FixTempBerendsenAsphere::FixTempBerendsenAsphere(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR, "Illegal fix temp/berendsen/asphere command");

  if (!atom->ellipsoid_flag || !atom->angmom_flag || !atom->torque_flag)
    error->all(FLERR,
               "Fix berendsen/asphere requires atom attributes "
               "quat, angmom, torque, shape");

  coupling_factor = 1.0;

  // Berendsen thermostat should be applied every step
  nevery = 1;
  scalar_flag = 1;
  global_freq = nevery;
  extscalar = 1;

  // Store also translational and rotational stored_data
  size_vector = 6;
  vector_flag = 1;
  extvector = 0;

  debug = false;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg + 4 > narg)
        error->all(FLERR,
                   "Illegal fix berendsen/asphere "
                   "command");

      tstr = NULL;
      if (strstr(arg[3], "v_") == arg[3]) {
        tstyle = EQUAL;
      } else {
        tstyle = CONSTANT;
      }

      t_start = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      t_target = t_start;
      t_stop = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      t_period = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR,
                   "Target temperature for fix berendsen/asphere "
                   "cannot be 0.0");
      iarg += 4;
    } else if (strcmp(arg[iarg], "debug") == 0) {
      debug = true;
      iarg++;
    } else if (strcmp(arg[iarg], "couple") == 0) {
      coupling_factor = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix berendsen/asphere command");
  }

  // error checks

  if (t_period <= 0.0) error->all(FLERR, "Fix temp/berendsen/asphere period must be > 0.0");

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  // rotational
  int n = strlen(id) + 17;
  id_temp_rot = new char[n];
  strcpy(id_temp_rot, id);
  strcat(id_temp_rot, "_temp_rotational");

  char **newarg = new char *[5];    //3
  newarg[0] = id_temp_rot;
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "temp/asphere";
  newarg[3] = (char *) "dof";
  newarg[4] = (char *) "rotate";

  modify->add_compute(5, newarg);

  delete[] newarg;

  // translational
  n = strlen(id) + 19;
  id_temp_transl = new char[n];
  strcpy(id_temp_transl, id);
  strcat(id_temp_transl, "_temp_traslational");

  newarg = new char *[3];
  newarg[0] = id_temp_transl;
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "temp";
  modify->add_compute(3, newarg);

  printf("[FixTempBerendsenAsphere] tstyle %d\n", tstyle);

  delete[] newarg;

  tflag = 1;

  // extra data
  energy = 0.0;
  stored_data[0] = 0.0;
  stored_data[1] = 0.0;
  stored_data[2] = 0.0;
  stored_data[3] = 0.0;
  stored_data[4] = 0.0;
  stored_data[5] = 0.0;

  if (debug) {
    printf("FixTempBerendsenAsphere "
           "t_target %f t_start %f t_stop %f t_damp %f t_coupling %f\n",
           t_target, t_start, t_stop, t_period, coupling_factor);
    fflush(stdout);
  }
}

/* ---------------------------------------------------------------------- */

FixTempBerendsenAsphere::~FixTempBerendsenAsphere()
{
  delete[] tstr;

  // delete temperature if fix created it
  if (tflag) {
    modify->delete_compute(id_temp_transl);
    modify->delete_compute(id_temp_rot);
  }

  delete[] id_temp_transl;
  delete[] id_temp_rot;
}

/* ---------------------------------------------------------------------- */

int FixTempBerendsenAsphere::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTempBerendsenAsphere::init()
{
  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0) error->all(FLERR, "Variable name for fix temp/berendsen/asphere does not exist");
    if (input->variable->equalstyle(tvar))
      tstyle = EQUAL;
    else
      error->all(FLERR, "Variable for fix temp/berendsen/asphere is invalid style");
  }

  int icompute_transl = modify->find_compute(id_temp_transl);
  if (icompute_transl < 0)
    error->all(FLERR, "Temp ID for fix temp/berendsen/asphere does not exist");
  temperature_transl = modify->compute[icompute_transl];

  int icompute_rot = modify->find_compute(id_temp_rot);
  if (icompute_rot < 0)
    error->all(FLERR,
               "Temp/rotational ID for fix temp/berendsen/asphere "
               "does not exist");

  temperature_rot = modify->compute[icompute_rot];

  if (temperature_transl->tempbias || temperature_rot->tempbias)
    which = BIAS;
  else
    which = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void FixTempBerendsenAsphere::end_of_step()
{
  if (debug) {
    printf("FixTempBerendsenAsphere::end_of_step\n");
    fflush(stdout);
  }

  double delta = update->ntimestep - update->beginstep;
  delta /= update->endstep - update->beginstep;
  t_target = t_start + delta * (t_stop - t_start);

  double t_current_transl = temperature_transl->compute_scalar();
  if (t_current_transl == 0.0) {
    t_current_transl = t_target * 0.000001;
    error->warning(FLERR,
                   "Computed translational temperature "
                   "for fix temp/berendsen/asphere was 0.0");
  }

  double t_current_rot = temperature_rot->compute_scalar();
  if (t_current_rot == 0.0) {
    t_current_rot = t_target * 0.000001;
    error->warning(FLERR,
                   "Computed rotational temperature "
                   "for fix temp/berendsen/asphere was 0.0");
  }

  // set current t_target
  // if variable temp, evaluate variable, wrap with clear/add
  if (tstyle == CONSTANT)
    t_target = t_start + delta * (t_stop - t_start);
  else {
    modify->clearstep_compute();
    t_target = input->variable->compute_equal(tvar);
    if (t_target < 0.0)
      error->one(FLERR, "Fix temp/berendsen variable returned negative temperature");
    modify->addstep_compute(update->ntimestep + nevery);
  }

  // rescale velocities by lamda
  double lamdaTransl =
      sqrt(1.0 + coupling_factor * update->dt / t_period * (t_target / t_current_transl - 1.0));
  double lamdaRot =
      sqrt(1.0 + coupling_factor * update->dt / t_period * (t_target / t_current_rot - 1.0));
  double efactor_rot = 0.5 * force->boltz * temperature_rot->dof;
  double efactor_transl = 0.5 * force->boltz * temperature_transl->dof;

  stored_data[0] = lamdaTransl;
  stored_data[1] = lamdaRot;
  stored_data[2] = efactor_rot;
  stored_data[3] = efactor_transl;
  stored_data[4] = t_current_transl;
  stored_data[5] = t_current_rot;

  energy += t_current_rot * (1.0 - lamdaRot * lamdaRot) * efactor_rot +
      t_current_transl * (1.0 - lamdaTransl * lamdaTransl) * efactor_transl;

  if (debug && (update->ntimestep % 1 == 0)) {
    printf("[FixTempBerendsenAsphere] t: %g (* %g) r: %g (* %g) target:%g\n", t_current_transl,
           lamdaTransl, t_current_rot, lamdaRot, t_target);
    fflush(stdout);
  }

  double **v = atom->v;
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= lamdaTransl;
        v[i][1] *= lamdaTransl;
        v[i][2] *= lamdaTransl;

        angmom[i][0] *= lamdaRot;
        angmom[i][1] *= lamdaRot;
        angmom[i][2] *= lamdaRot;
      }
    }

  } else if (which == BIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        if (temperature_transl->tempbias) temperature_transl->remove_bias(i, v[i]);

        if (temperature_rot->tempbias) temperature_rot->remove_bias(i, v[i]);

        //should discriminate trans & rot
        v[i][0] *= lamdaTransl;
        v[i][1] *= lamdaTransl;
        v[i][2] *= lamdaTransl;

        angmom[i][0] *= lamdaRot;
        angmom[i][1] *= lamdaRot;
        angmom[i][2] *= lamdaRot;

        if (temperature_transl->tempbias) temperature_transl->restore_bias(i, v[i]);

        if (temperature_rot->tempbias) temperature_rot->restore_bias(i, v[i]);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixTempBerendsenAsphere::modify_param(int narg, char **arg)
{
  int const nargs_allowed = 3;
  if (strcmp(arg[0], "temp") == 0) {    // temp mod only is allowed
    if (narg < nargs_allowed) error->all(FLERR, "Illegal fix_modify command");
    if (tflag) {
      modify->delete_compute(id_temp_transl);
      modify->delete_compute(id_temp_rot);
      tflag = 0;
    }

    // parse trasl temp compute id
    delete[] id_temp_transl;
    int n = strlen(arg[1]) + 1;
    id_temp_transl = new char[n];
    strcpy(id_temp_transl, arg[1]);

    printf("modifying fix with temp trasl id %s\n", id_temp_transl);

    int icompute_transl = modify->find_compute(id_temp_transl);
    if (icompute_transl < 0) error->all(FLERR, "Could not find fix_modify temp ID");
    temperature_transl = modify->compute[icompute_transl];

    if (temperature_transl->tempflag == 0)
      error->all(FLERR, "Fix_modify temp ID does not compute temperature");
    if (temperature_transl->igroup != igroup && comm->me == 0)
      error->warning(FLERR, "Group for fix_modify temp != fix group");

    // parse rot temp compute id
    delete[] id_temp_rot;
    n = strlen(arg[2]) + 1;
    id_temp_rot = new char[n];
    strcpy(id_temp_rot, arg[2]);

    printf("modifying fix with temp rot id %s\n", id_temp_rot);

    int icompute_rot = modify->find_compute(id_temp_rot);
    if (icompute_rot < 0) error->all(FLERR, "Could not find fix_modify temp ID");
    temperature_rot = modify->compute[icompute_rot];

    if (temperature_rot->tempflag == 0)
      error->all(FLERR, "Fix_modify temp ID does not compute temperature");
    if (temperature_rot->igroup != igroup && comm->me == 0)
      error->warning(FLERR, "Group for fix_modify temp != fix group");

    return nargs_allowed;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void FixTempBerendsenAsphere::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

double FixTempBerendsenAsphere::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
   ------------------------------------------------------------------------- */

void *FixTempBerendsenAsphere::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "t_target") == 0) { return &t_target; }
  return NULL;
}

/* ----------------------------------------------------------------------
   return either translational (n=0) or rotational (n=1) scale stored_data
   ------------------------------------------------------------------------- */

double FixTempBerendsenAsphere::compute_vector(int n)
{
  return stored_data[n];
}

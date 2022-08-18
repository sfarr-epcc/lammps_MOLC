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
                        Stephen Farr (EPCC)
   ------------------------------------------------------------------------- */

#include "pppm_molc.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "gridcomm.h"
#include "math_const.h"
#include "math_special.h"
#include "math_extra.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "remap_wrap.h"
#include "atom_vec_ellipsoid.h"

//#define DEBUG

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER 7
#define OFFSET 16384
#define LARGE 10000.0
#define SMALL 0.00001
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif


/* ---------------------------------------------------------------------- */

PPPMMolc::PPPMMolc(LAMMPS *lmp) : PPPM(lmp)
{
  part2grid=nullptr;
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"PPPM MOLC requires atom style ellipsoid");
}

/* ---------------------------------------------------------------------- */

void PPPMMolc::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal kspace_style pppm command");
  accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));


  // offcentre variant begins
  // now get these values from the compatible pair style as they will be the same
  //int nCoulSites = force->inumeric(FLERR,arg[1]);
  //int atomType[nCoulSites+1];


  // create the arrays here, we copy the values from the pair style in the init function
  //nsites = new int[atom->ntypes+1];
  //for (int t = 1; t <= atom->ntypes; ++t)
  //  nsites[t] = 0;

  //molFrameSite = new double**[atom->ntypes+1];
  //molFrameCharge = new double*[atom->ntypes+1];


  //max_nsites = 0;
  //int totsites = 0;
  //int start_sitesspec_argcount = 2;
  //int argcount = start_sitesspec_argcount;
  //for (int t = 1; t <= nCoulSites; ++t) {
  //  atomType[t] = force->inumeric(FLERR,arg[argcount++]);
  //  ++nsites[atomType[t]];
  //
  //  max_nsites = nsites[atomType[t]] > max_nsites ?
  //    nsites[atomType[t]] : max_nsites;
  //
  //  argcount += 4;
  //}

  //argcount = start_sitesspec_argcount;
  //for (int type = 1; type <= atom->ntypes; ++type) {
  //  molFrameSite[type] = new double*[nsites[type]+1];
  //  molFrameCharge[type] = new double[nsites[type]+1];
  //
  //  for (int site = 1; site <= nsites[type]; ++site) {
  //    ++argcount;
  //
  //    molFrameSite[type][site] = new double[3];
  //    molFrameSite[type][site][0] = force->numeric(FLERR,arg[argcount++]);
  //    molFrameSite[type][site][1] = force->numeric(FLERR,arg[argcount++]);
  //   molFrameSite[type][site][2] = force->numeric(FLERR,arg[argcount++]);

  //    molFrameCharge[type][site] = force->numeric(FLERR,arg[argcount++]);
  //  }

  //  totsites += nsites[type];
  //}
  // offcentre variant ends

  // check
  //if (narg > argcount) {
  //  fprintf(stderr, "number of specified charges exceeds the declared %i\n",
  //         nCoulSites);
  //  exit(1);
  //}
}

/* ----------------------------------------------------------------------
   free all memory
   ------------------------------------------------------------------------- */

PPPMMolc::~PPPMMolc()
{
  if (copymode) return;

  //delete [] factors;
  //PPPM::deallocate();
  //if (peratom_allocate_flag) deallocate_peratom();
  //if (group_allocate_flag) deallocate_groups();
  memory->destroy(part2grid);
  //memory->destroy(acons);


  // deallocate offcentre parts
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

/* ----------------------------------------------------------------------
   called once before run
   ------------------------------------------------------------------------- */

void PPPMMolc::init()
{
  if (me == 0) utils::logmesg(lmp,"PPPM/MOlC initialization ...\n");


  // error check

  triclinic_check();

  if (triclinic != domain->triclinic)
    error->all(FLERR,"Must redefine kspace_style after changing to triclinic "
               "box");

  if (domain->triclinic && differentiation_flag == 1)
    error->all(FLERR,"Cannot (yet) use PPPMMolc with triclinic box and "
               "kspace_modify diff ad");
  if (domain->triclinic && slabflag)
    error->all(FLERR,"Cannot (yet) use PPPMMolc with triclinic box and "
               "slab correction");
  if (domain->dimension == 2) error->all(FLERR, "Cannot use PPPMMolc with"
                                         "2d simulation");
  if (comm->style != 0)
    error->universe_all(FLERR,"PPPMMolc can only currently be used with "
                        "comm_style brick");

  //if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute
  //q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with PPPMMolc");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab PPPMMolc");
  }

  if (order < 2 || order > MAXORDER) {
    char str[128];
    sprintf(str,"PPPMMolc order cannot be < 2 or > than %d",MAXORDER);
    error->all(FLERR,str);
  }
  
  
  // compute two charge force

  two_charge();


  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp = 0;
  auto p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  cutoff = *p_cutoff;

  #ifdef DEBUG
  printf("pppm cutoff = %f\n", cutoff);
  #endif


  // create the arrays here, then copy the values from the pair style
  nsites = new int[atom->ntypes+1];
  for (int t = 1; t <= atom->ntypes; ++t)
    nsites[t] = 0;

  molFrameSite = new double**[atom->ntypes+1];
  molFrameCharge = new double*[atom->ntypes+1];



  // copy offcentre charge information from the pair style
  int * ref_nsites = (int *) force->pair->extract("nsites",itmp);

  if (!ref_nsites)
    error->all(FLERR,"Pair style is incompatible with KSpace style");
  
  max_nsites = 0;
  for (int t = 1; t <= atom->ntypes; ++t){
    nsites[t] = ref_nsites[t];
    max_nsites = nsites[t] > max_nsites ? nsites[t] : max_nsites;

    #ifdef DEBUG
    printf("site pppm %d: %d\n",t,nsites[t]);
    #endif
  }

  double *** ref_molFrameSite = (double ***) force->pair->extract("molFrameSite", itmp);
  double ** ref_molFrameCharge = (double **) force->pair->extract("molFrameCharge", itmp);

  for (int t = 1; t <= atom->ntypes; ++t){

    molFrameSite[t] = new double*[nsites[t]+1];
    molFrameCharge[t] = new double[nsites[t]+1];

    for(int site=1; site<=nsites[t]; ++site){
      molFrameSite[t][site] = new double[3];
      molFrameSite[t][site][0] = ref_molFrameSite[t][site][0];
      molFrameSite[t][site][1] = ref_molFrameSite[t][site][1];
      molFrameSite[t][site][2] = ref_molFrameSite[t][site][2];
      molFrameCharge[t][site] = ref_molFrameCharge[t][site];
    }
  }

  


  // compute ncharges
  int *type = atom->type;
  int nlocal = atom->nlocal;

  bigint ncharges_local(0);

#if defined(_OPENMP)
  // icpc bug: error for nlocal, type
  // solved by adding firstprivate(nlocal, type) to pragma
#pragma omp parallel for default(none) firstprivate(nlocal, type)\
  reduction(+:ncharges_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    ncharges_local += nsites[itype];
  }

  MPI_Allreduce(&ncharges_local,
                &ncharges,
                1,
                MPI_LMP_BIGINT,
                MPI_SUM,
                world);

  // search for max offset due to offcentre charges useful for qdist below
  double max_charge_offset_local = 0.0;
  double scratch = 0.0;

  for (int i = 0; i < atom->nlocal; i++) {
    int itype = type[i];
    if (nsites[itype] > 0) {
      for (int s = 1; s <= nsites[itype]; ++s) {
        scratch = fabs(molFrameSite[itype][s][0]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
        scratch = fabs(molFrameSite[itype][s][1]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
        scratch = fabs(molFrameSite[itype][s][2]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
      }
    }
  }

  MPI_Allreduce(&max_charge_offset_local,
                &max_charge_offset,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                world);

  // printf("PPPM offcentre: %i max_ch_off %f %f\n", comm->me,
  // 	 max_charge_offset, max_charge_offset_local);

  // if kspace is TIP4P, extract TIP4P params from pair style
  // bond/angle are not yet init(), so insure equilibrium request is valid

  qdist = 0.0;

  if (tip4pflag) {
    if (me == 0) {
      if (screen) fprintf(screen,"  extracting TIP4P info from pair style\n");
      if (logfile) fprintf(logfile,"  extracting TIP4P info from pair style\n");
    }

    double *p_qdist = (double *) force->pair->extract("qdist",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeA = (int *) force->pair->extract("typeA",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR,"Pair style is incompatible with TIP4P KSpace style");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    int typeB = *p_typeB;

    if (force->angle == NULL || force->bond == NULL ||
        force->angle->setflag == NULL || force->bond->setflag == NULL)
      error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
        force->angle->setflag[typeA] == 0)
      error->all(FLERR,"Bad TIP4P angle type for PPPMMolc/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
        force->bond->setflag[typeB] == 0)
      error->all(FLERR,"Bad TIP4P bond type for PPPMMolc/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5*theta) * blen);
  } else {
    qdist = max_charge_offset;
  }

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

#ifdef DEBUG
printf("accuracy: %f, absolute: %f, relative: %f, 2cf: %f\n", accuracy, accuracy_absolute, accuracy_relative, two_charge_force);
#endif

  // free all arrays previously allocated

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

GridComm *gctmp = nullptr;
  int iteration = 0;

  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,"Reducing PPPM order b/c stencil extends "
                     "beyond nearest neighbor processor");

    if (stagger_flag && !differentiation_flag) compute_gf_denom();
    set_grid_global();
    set_grid_local();
    if (overlap_allowed) break;

    gctmp = new GridComm(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                         nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                         nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);

    int tmp1,tmp2;
    gctmp->setup(tmp1,tmp2);
    if (gctmp->ghost_adjacent()) break;
    delete gctmp;

    order--;
    iteration++;
  }

  if (order < minorder) error->all(FLERR,"PPPM order < minimum allowed order");
  if (!overlap_allowed && !gctmp->ghost_adjacent())
    error->all(FLERR,"PPPM grid stencil extends beyond nearest neighbor processor");
  if (gctmp) delete gctmp;

  // adjust g_ewald

  if (!gewaldflag) adjust_gewald();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy();

  // print stats

  int ngrid_max,nfft_both_max;
  MPI_Allreduce(&ngrid,&ngrid_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nfft_both,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {
    std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    mesg += fmt::format("  grid = {} {} {}\n",nx_pppm,ny_pppm,nz_pppm);
    mesg += fmt::format("  stencil order = {}\n",order);
    mesg += fmt::format("  estimated absolute RMS force accuracy = {:.8g}\n",
                       estimated_accuracy);
    mesg += fmt::format("  estimated relative force accuracy = {:.8g}\n",
                       estimated_accuracy/two_charge_force);
    mesg += "  using " LMP_FFT_PREC " precision " LMP_FFT_LIB "\n";
    mesg += fmt::format("  3d grid and FFT values/proc = {} {}\n",
                       ngrid_max,nfft_both_max);
    utils::logmesg(lmp,mesg);
  }

  // allocate K-space dependent memory
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();


  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();
}


/* ----------------------------------------------------------------------
   compute qsum,qsqsum,q2 and give error/warning if not charge neutral
   called initially, when particle count changes, when charges are changed
   ------------------------------------------------------------------------- */

void PPPMMolc::qsum_qsq()
{
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double qsum_local(0.0), qsqsum_local(0.0);

#if defined(_OPENMP)
  // icpc bug: error for nlocal, type
  // solved by adding firstprivate(nlocal, type) to pragma
#pragma omp parallel for default(none) firstprivate(nlocal, type) \
  reduction(+:qsum_local,qsqsum_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    for (int s = 1; s <= nsites[itype]; ++s) {
      double qi = molFrameCharge[itype][s];
      qsum_local += qi;
      qsqsum_local += qi*qi;
    }
  }

  MPI_Allreduce(&qsum_local,&qsum,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&qsqsum_local,&qsqsum,1,MPI_DOUBLE,MPI_SUM,world);

  if ((qsqsum == 0.0) && (comm->me == 0) && warn_nocharge) {
    error->warning(FLERR,"Using kspace solver on system with no charge");
    warn_nocharge = 0;
  }

  q2 = qsqsum * force->qqrd2e;

  // not yet sure of the correction needed for non-neutral systems
  // so issue warning or error

  if (fabs(qsum) > SMALL) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    if (!warn_nonneutral) error->all(FLERR,str);
    if (warn_nonneutral == 1 && comm->me == 0) error->warning(FLERR,str);
    warn_nonneutral = 2;
  }
}

/* ----------------------------------------------------------------------
   compute the PPPMMolc long-range force, energy, virial
   ------------------------------------------------------------------------- */

void PPPMMolc::compute(int eflag, int vflag)
{
  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag,vflag);

  if (evflag_atom && !peratom_allocate_flag) allocate_peratom();


  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  //MATTEO probably should do something here, convert also offcntre positions
  // void Domain::x2lamda(int n) -> for offcentre just use void
  // Domain::lamda2x(double *lamda, double *x) ??? not sure maybe shoudl
  // do some transform to orientations too ? otherwise offcentre
  // position computed on the fly based on centres, which are already
  // converted

  // how are offcentre charges to be remapped among periodic replicas?
  // by now just out of box

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  // extend size of per-atom arrays if necessary

  #ifdef DEBUG
  printf("%d %d\n" ,nmax ,max_nsites);
  #endif

  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    #ifdef DEBUG
    printf("%d %d\n" ,nmax ,max_nsites);
    #endif
    memory->create(part2grid,nmax,max_nsites,3,"PPPMMolc:part2grid");
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid

  particle_map();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  gc->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  
  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1)
    gc->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  else
    gc->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,gc_buf1,gc_buf2,MPI_FFT_SCALAR);

  // extra per-atom energy/virial communication



  if (evflag_atom) fieldforce_peratom();

  // sum global energy across procs and add in volume-dependent term

  const double qscale = qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy = energy_all;

    energy *= 0.5*volume;
    energy -= g_ewald*qsqsum/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
    // for (i = 0; i < 6; i++) printf("ALL %f %f %f %f\n", virial[i], virial_all[i], qscale, volume);
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // ntotal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    int nlocal = atom->nlocal;
    int *type = atom->type;
    int ntotal = nlocal;
    if (tip4pflag) ntotal += atom->nghost;

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        int itype = type[i];
        double qi_all = 0.0;
        double qi_all_2 = 0.0;
        for (int s = 1; s <= nsites[itype]; ++s) {
          double qi = molFrameCharge[itype][s];
          qi_all_2 += qi*qi;
          qi_all += qi;
        }

        eatom[i] *= 0.5;
        eatom[i] -= g_ewald*qi_all_2/MY_PIS + MY_PI2*qi_all*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) eatom[i] *= 0.5*qscale;
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5*qscale;
    }
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();

  // convert atoms back from lamda to box coords

  //MATTEO probably should do something here, convert also offcntre positions
  if (triclinic != 0) domain->lamda2x(atom->nlocal);
}


/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   set global size of PPPMMolc grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation
   ------------------------------------------------------------------------- */

void PPPMMolc::set_grid_global()
{
  // use xprd,yprd,zprd (even if triclinic, and then scale later)
  // adjust z dimension for 2d slab PPPMMolc
  // 3d PPPMMolc just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;

  // printf("PPPMMolc::set_grid_global %f %f %f\n", xprd, yprd, zprd);

  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h;
  //bigint natoms = atom->natoms;

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");
    g_ewald = accuracy*sqrt(ncharges*cutoff*xprd*yprd*zprd) / (2.0*q2);
    //g_ewald = accuracy*sqrt(atom->natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until accuracy target is met

  // printf("PPPMMolc::set_grid_global order %i accuracy %f 1/gewald %f\n",
  // 	 order, accuracy, 1.0/g_ewald);

  if (!gridflag) {

    if (differentiation_flag == 1 || stagger_flag) {

      h = h_x = h_y = h_z = 4.0/g_ewald;
      int count = 0;
      while (1) {

        // set grid dimension
        nx_pppm = static_cast<int> (xprd/h_x);
        ny_pppm = static_cast<int> (yprd/h_y);
        nz_pppm = static_cast<int> (zprd_slab/h_z);

        if (nx_pppm <= 1) nx_pppm = 2;
        if (ny_pppm <= 1) ny_pppm = 2;
        if (nz_pppm <= 1) nz_pppm = 2;

        //set local grid dimension
        int npey_fft = -1, npez_fft = -1;
        if (nz_pppm >= nprocs) {
          npey_fft = 1;
          npez_fft = nprocs;
        } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

        int me_y = me % npey_fft;
        int me_z = me / npey_fft;

        nxlo_fft = 0;
        nxhi_fft = nx_pppm - 1;
        nylo_fft = me_y*ny_pppm/npey_fft;
        nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
        nzlo_fft = me_z*nz_pppm/npez_fft;
        nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

        double df_kspace = compute_df_kspace();

        count++;

        // break loop if the accuracy has been reached or
        // too many loops have been performed

        if (df_kspace <= accuracy) break;
        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {

      double err;
      h_x = h_y = h_z = 1.0/g_ewald;

      nx_pppm = static_cast<int> (xprd/h_x) + 1;
      ny_pppm = static_cast<int> (yprd/h_y) + 1;
      nz_pppm = static_cast<int> (zprd_slab/h_z) + 1;

      err = estimate_ik_error(h_x,xprd,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_x,xprd,ncharges);
        nx_pppm++;
        h_x = xprd/nx_pppm;
      }

      err = estimate_ik_error(h_y,yprd,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_y,yprd,ncharges);
        ny_pppm++;
        h_y = yprd/ny_pppm;
      }

      err = estimate_ik_error(h_z,zprd_slab,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_z,zprd_slab,ncharges);
        nz_pppm++;
        h_z = zprd_slab/nz_pppm;
      }
    }

    // scale grid for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = nx_pppm/xprd;
      tmp[1] = ny_pppm/yprd;
      tmp[2] = nz_pppm/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      nx_pppm = static_cast<int>(tmp[0]) + 1;
      ny_pppm = static_cast<int>(tmp[1]) + 1;
      nz_pppm = static_cast<int>(tmp[2]) + 1;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm)) nx_pppm++;
  while (!factorable(ny_pppm)) ny_pppm++;
  while (!factorable(nz_pppm)) nz_pppm++;

  // printf("PPPMMolc::set_grid_global n_ppm %i %i %i\n",
  // 	 nx_pppm, ny_pppm, nz_pppm);

  if (triclinic == 0) {
    h_x = xprd/nx_pppm;
    h_y = yprd/ny_pppm;
    h_z = zprd_slab/nz_pppm;
  } else {
    double tmp[3];
    tmp[0] = nx_pppm;
    tmp[1] = ny_pppm;
    tmp[2] = nz_pppm;
    x2lamdaT(&tmp[0],&tmp[0]);
    h_x = 1.0/tmp[0];
    h_y = 1.0/tmp[1];
    h_z = 1.0/tmp[2];
  }

  // printf("PPPMMolc::set_grid_global h %f %f %f\n",
  // 	 h_x, h_y, h_z);

  if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
    error->all(FLERR,"PPPMMolc grid is too large");
}



/* ----------------------------------------------------------------------
   compute estimated kspace force error
   ------------------------------------------------------------------------- */

double PPPMMolc::compute_df_kspace()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  //bigint natoms = atom->natoms;
  double df_kspace = 0.0;
  if (differentiation_flag == 1 || stagger_flag) {
    double qopt = compute_qopt();
    df_kspace = sqrt(qopt/ncharges)*q2/(xprd*yprd*zprd_slab);
  } else {
    double lprx = estimate_ik_error(h_x,xprd,ncharges);
    double lpry = estimate_ik_error(h_y,yprd,ncharges);
    double lprz = estimate_ik_error(h_z,zprd_slab,ncharges);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  }
  return df_kspace;
}



/* ----------------------------------------------------------------------
   calculate f(x) using Newton-Raphson solver
   ------------------------------------------------------------------------- */

double PPPMMolc::newton_raphson_f()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  //bigint natoms = atom->natoms;

  double df_rspace = 2.0*q2*exp(-g_ewald*g_ewald*cutoff*cutoff) /
    sqrt(ncharges*cutoff*xprd*yprd*zprd);
  //       sqrt(natoms*cutoff*xprd*yprd*zprd);

  double df_kspace = compute_df_kspace();

  return df_rspace - df_kspace;
}


/* ----------------------------------------------------------------------
   calculate the final estimate of the accuracy
   ------------------------------------------------------------------------- */

double PPPMMolc::final_accuracy()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  //bigint natoms = atom->natoms;

  double df_kspace = compute_df_kspace();
  double q2_over_sqrt = q2 / sqrt(ncharges*cutoff*xprd*yprd*zprd);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
                                   df_table*df_table);

  return estimated_accuracy;
}



/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
   ------------------------------------------------------------------------- */

void PPPMMolc::particle_map()
{
  int nx,ny,nz;

  double **x = atom->x;
  int *type = atom->type;
  int *tag = atom->tag;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;

  int flag = 0;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");


  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
      // current particle coord can be outside global and local box
      // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

      nx = static_cast<int> ((rsite[0]-boxlo[0])*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((rsite[1]-boxlo[1])*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((rsite[2]-boxlo[2])*delzinv+shift) - OFFSET;

      // if (triclinic != 0)
      // 	domain->lamda2x(rsite, rsite);

      // printf("PPPM offcentre %i: particle_map %f %f %f to %i %i %i of %f %f
      // %f\n", comm->me, rsite[0], rsite[1], rsite[2], nx, ny, nz, x[i][0],
      // x[i][1], x[i][2]);

      part2grid[i][s][0] = nx;
      part2grid[i][s][1] = ny;
      part2grid[i][s][2] = nz;

      // check that entire stencil around nx,ny,nz will fit in my 3d brick

      if ((nx+nlower < nxlo_out) || (nx+nupper > nxhi_out) ||
          (ny+nlower < nylo_out) || (ny+nupper > nyhi_out) ||
          (nz+nlower < nzlo_out) || (nz+nupper > nzhi_out)) {
        flag = 1;
        printf("PPPM offcentre (%2i): site at %g %g %g -> %2i %2i %2i\n"
               "PPPM offcentre (%2i): atom %i type %i\n"
               "PPPM offcentre (%2i): out of bounds x %2i < %2i%s %2i > %2i%s\n"
               "PPPM offcentre (%2i): out of bounds y %2i < %2i%s %2i > %2i%s\n"
               "PPPM offcentre (%2i): out of bounds z %2i < %2i%s %2i > %2i%s\n"
               "PPPM offcentre (%2i): with tolerance %i %i\n",
               comm->me,
               rsite[0], rsite[1], rsite[2], nx, ny, nz,
               comm->me,
               tag[i], itype,
               comm->me,
               nx+nlower, nxlo_out, (nx+nlower < nxlo_out) ? "<--" : "",
               nx+nupper, nxhi_out, (nx+nupper > nxhi_out) ? "<--" : "",
               comm->me,
               ny+nlower, nylo_out, (ny+nlower < nylo_out) ? "<--" : "",
               ny+nupper, nyhi_out, (ny+nupper > nyhi_out) ? "<--" : "",
               comm->me,
               nz+nlower, nzlo_out, (nz+nlower < nzlo_out) ? "<--" : "",
               nz+nupper, nzhi_out, (nz+nupper > nzhi_out) ? "<--" : "",
               comm->me,
               nlower,	nupper);
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);

  if (flag)
    error->one(FLERR,"Out of range atoms - cannot compute PPPMMolc");
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid
   ------------------------------------------------------------------------- */

void PPPMMolc::make_rho()
{
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  int *type = atom->type;
  double **x = atom->x;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);

      z0 = delvolinv * molFrameCharge[itype][s];
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        y0 = z0*rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          x0 = y0*rho1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            density_brick[mz][my][mx] += x0*rho1d[0][l];
          }
        }
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}





/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
   ------------------------------------------------------------------------- */

void PPPMMolc::fieldforce_ik()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];

    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};

        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0] + x[i][0],
          labFrameSite[1] + x[i][1],
          labFrameSite[2] + x[i][2]
        };

        // if triclinic all expressed in lamda coords at this point
        if (triclinic != 0)
          domain->x2lamda(rsite, rsite);

        nx = part2grid[i][s][0];
        ny = part2grid[i][s][1];
        nz = part2grid[i][s][2];
        dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
        dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
        dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

        compute_rho1d(dx,dy,dz);

        ekx = eky = ekz = ZEROF;
        for (n = nlower; n <= nupper; n++) {
          mz = n+nz;
          z0 = rho1d[2][n];
          for (m = nlower; m <= nupper; m++) {
            my = m+ny;
            y0 = z0*rho1d[1][m];
            for (l = nlower; l <= nupper; l++) {
              mx = l+nx;
              x0 = y0*rho1d[0][l];
              ekx -= x0*vdx_brick[mz][my][mx];
              eky -= x0*vdy_brick[mz][my][mx];
              ekz -= x0*vdz_brick[mz][my][mx];
            }
          }
        }

        // convert E-field to force
        const double qfactor = qqrd2e * scale * molFrameCharge[itype][s];

        double force[3] = { qfactor*ekx, qfactor*eky, 0.0 };
        if (slabflag != 2) force[2] = qfactor*ekz;

        f[i][0] += force[0];
        f[i][1] += force[1];

        if (slabflag != 2) f[i][2] += force[2];

        double torque[3];
        MathExtra::cross3(labFrameSite, force, torque);
        tor[i][0] += torque[0];
        tor[i][1] += torque[1];
        tor[i][2] += torque[2];
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
   ------------------------------------------------------------------------- */

void PPPMMolc::fieldforce_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz;
  FFT_SCALAR ekx,eky,ekz;
  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);
      compute_drho1d(dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
            eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
            ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
          }
        }
      }
      ekx *= hx_inv;
      eky *= hy_inv;
      ekz *= hz_inv;

      // convert E-field to force and substract self forces

      const double qfactor = qqrd2e * scale;
      double force[3] = { 0.0, 0.0, 0.0 };
      double torque[3] = { 0.0, 0.0, 0.0 };

      s1 = rsite[0]*hx_inv;
      s2 = rsite[1]*hy_inv;
      s3 = rsite[2]*hz_inv;
      sf = sf_coeff[0]*sin(2*MY_PI*s1);
      sf += sf_coeff[1]*sin(4*MY_PI*s1);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      force[0] = qfactor*(ekx*molFrameCharge[itype][s] - sf);
      f[i][0] += force[0];

      sf = sf_coeff[2]*sin(2*MY_PI*s2);
      sf += sf_coeff[3]*sin(4*MY_PI*s2);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      force[1] = qfactor*(eky*molFrameCharge[itype][s] - sf);
      f[i][1] += force[1];

      sf = sf_coeff[4]*sin(2*MY_PI*s3);
      sf += sf_coeff[5]*sin(4*MY_PI*s3);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      if (slabflag != 2) {
        force[2] = qfactor*(ekz*molFrameCharge[itype][s] - sf);
        f[i][2] += force[2];
      }

      MathExtra::cross3(labFrameSite, force, torque);
      tor[i][0] += torque[0];
      tor[i][1] += torque[1];
      tor[i][2] += torque[2];
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   interpolate from grid to get per-atom energy/virial
   ------------------------------------------------------------------------- */

void PPPMMolc::fieldforce_peratom()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u,v0,v1,v2,v3,v4,v5;

  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);

      u = v0 = v1 = v2 = v3 = v4 = v5 = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        z0 = rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          y0 = z0*rho1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            x0 = y0*rho1d[0][l];
            if (eflag_atom) u += x0*u_brick[mz][my][mx];
            if (vflag_atom) {
              v0 += x0*v0_brick[mz][my][mx];
              v1 += x0*v1_brick[mz][my][mx];
              v2 += x0*v2_brick[mz][my][mx];
              v3 += x0*v3_brick[mz][my][mx];
              v4 += x0*v4_brick[mz][my][mx];
              v5 += x0*v5_brick[mz][my][mx];
            }
          }
        }
      }

      if (eflag_atom) eatom[i] += molFrameCharge[itype][s]*u;
      if (vflag_atom) {
        vatom[i][0] += molFrameCharge[itype][s]*v0;
        vatom[i][1] += molFrameCharge[itype][s]*v1;
        vatom[i][2] += molFrameCharge[itype][s]*v2;
        vatom[i][3] += molFrameCharge[itype][s]*v3;
        vatom[i][4] += molFrameCharge[itype][s]*v4;
        vatom[i][5] += molFrameCharge[itype][s]*v5;
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}


/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
   ------------------------------------------------------------------------- */

void PPPMMolc::slabcorr()
{
  // compute local contribution to global dipole moment

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;
  double **tor = atom->torque;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      //
      dipole += molFrameCharge[itype][s]*rsite[2];
    }
  }

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        dipole += molFrameCharge[itype][s]*rsite[2]*rsite[2];
      }

      // sum local contributions

      double tmp;
      MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
      dipole_r2 = tmp;
    }

    // compute corrections

    const double e_slabcorr = MY_2PI*(dipole_all*dipole_all - qsum*dipole_r2 -
                                      qsum*qsum*zprd*zprd/12.0)/volume;
    const double qscale = qqrd2e * scale;

    if (eflag_global) energy += qscale * e_slabcorr;

    // per-atom energy

    if (eflag_atom) {
      double efact = qscale * MY_2PI/volume;
      for (int i = 0; i < nlocal; i++) {
        int itype = type[i];
        double rotMat[3][3];
        if (nsites[itype] > 0) {
          iquat = bonus[ellipsoid[i]].quat;
          MathExtra::quat_to_mat(iquat, rotMat);
        }

        for (int s = 1; s <= nsites[itype]; ++s) {
          double labFrameSite[3] = {0.0, 0.0, 0.0};
          if (molFrameSite[itype][s][0] != 0.0 ||
              molFrameSite[itype][s][1] != 0.0 ||
              molFrameSite[itype][s][2] != 0.0) {
            double ms[3] = {
              molFrameSite[itype][s][0],
              molFrameSite[itype][s][1],
              molFrameSite[itype][s][2]
            };

            MathExtra::matvec(rotMat, ms, labFrameSite);
          }

          double rsite[3] = {
            labFrameSite[0]+x[i][0],
            labFrameSite[1]+x[i][1],
            labFrameSite[2]+x[i][2]
          };

          eatom[i] +=
            efact * molFrameCharge[itype][s]*(rsite[2]*dipole_all -
                                              0.5*(dipole_r2 +
                                                   qsum*rsite[2]*rsite[2]) -
                                              qsum*zprd*zprd/12.0); } } }

          // add on force corrections

          double ffact = qscale * (-4.0*MY_PI/volume);
          double **f = atom->f;

          for (int i = 0; i < nlocal; i++) {
            int itype = type[i];
            double rotMat[3][3];
            if (nsites[itype] > 0) {
              iquat = bonus[ellipsoid[i]].quat;
              MathExtra::quat_to_mat(iquat, rotMat);
            }

            for (int s = 1; s <= nsites[itype]; ++s) {
              double labFrameSite[3] = {0.0, 0.0, 0.0};
              if (molFrameSite[itype][s][0] != 0.0 ||
                  molFrameSite[itype][s][1] != 0.0 ||
                  molFrameSite[itype][s][2] != 0.0) {
                double ms[3] = {
                  molFrameSite[itype][s][0],
                  molFrameSite[itype][s][1],
                  molFrameSite[itype][s][2]
                };

                MathExtra::matvec(rotMat, ms, labFrameSite);
              }

              double rsite[3] = {
                labFrameSite[0]+x[i][0],
                labFrameSite[1]+x[i][1],
                labFrameSite[2]+x[i][2]
              };

              double force[3] = {0.0, 0.0, 0.0};
              force[2] = ffact * molFrameCharge[itype][s]*(dipole_all -
                                                           qsum*rsite[2]);
              f[i][2] += force[2];

              // and torque?

              double torque[3];
              MathExtra::cross3(labFrameSite, force, torque);
              tor[i][0] += torque[0];
              tor[i][1] += torque[1];
              tor[i][2] += torque[2];
            }
          }
  }
}


/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid for group-group interactions
   ------------------------------------------------------------------------- */

void PPPMMolc::make_rho_groups(int groupbit_A,
                                    int groupbit_B,
                                    int AA_flag)
{
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density arrays

  memset(&(density_A_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  memset(&(density_B_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (int i = 0; i < nlocal; i++) {

    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        // if triclinic all expressed in lamda coords at this point
        if (triclinic != 0)
          domain->x2lamda(rsite, rsite);

        nx = part2grid[i][s][0];
        ny = part2grid[i][s][1];
        nz = part2grid[i][s][2];
        dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
        dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
        dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

        compute_rho1d(dx,dy,dz);

        z0 = delvolinv * molFrameCharge[itype][s];
        for (n = nlower; n <= nupper; n++) {
          mz = n+nz;
          y0 = z0*rho1d[2][n];
          for (m = nlower; m <= nupper; m++) {
            my = m+ny;
            x0 = y0*rho1d[1][m];
            for (l = nlower; l <= nupper; l++) {
              mx = l+nx;

              // group A

              if (mask[i] & groupbit_A)
                density_A_brick[mz][my][mx] += x0*rho1d[0][l];

              // group B

              if (mask[i] & groupbit_B)
                density_B_brick[mz][my][mx] += x0*rho1d[0][l];
            }
          }
        }
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}



/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
   ------------------------------------------------------------------------- */

void PPPMMolc::slabcorr_groups(int groupbit_A,
                                    int groupbit_B,
                                    int AA_flag)
{
  // compute local contribution to global dipole moment

  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  double **x = atom->x;
  double zprd = domain->zprd;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double qsum_A = 0.0;
  double qsum_B = 0.0;
  double dipole_A = 0.0;
  double dipole_B = 0.0;
  double dipole_r2_A = 0.0;
  double dipole_r2_B = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if (mask[i] & groupbit_A) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        qsum_A += molFrameCharge[itype][s];
        dipole_A += molFrameCharge[itype][s]*rsite[2];
        dipole_r2_A += molFrameCharge[itype][s]*rsite[2]*rsite[2];

        if (mask[i] & groupbit_B) {
          qsum_B += molFrameCharge[itype][s];
          dipole_B += molFrameCharge[itype][s]*rsite[2];
          dipole_r2_B += molFrameCharge[itype][s]*rsite[2]*rsite[2];
        }
      }
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI/volume;

  e2group += efact * (dipole_A*dipole_B - 0.5*(qsum_A*dipole_r2_B +
                                               qsum_B*dipole_r2_A)
                      - qsum_A*qsum_B*zprd*zprd/12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0*MY_PI/volume);
  f2group[2] += ffact * (qsum_A*dipole_B - qsum_B*dipole_A);
}

int PPPMMolc::getNsitesOf(int type) {
  return nsites[type];
}

double* PPPMMolc::getCharges(int type) {
  return molFrameCharge[type];
}

void PPPMMolc::setCharges(int type, int chN, double chValue) {
  molFrameCharge[type][chN] = chValue;
}

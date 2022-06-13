/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   -------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------
   Contributing author: Matteo Ricci <matteoeghirotta@gmail.com>
--------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(ellipsoid,BondEllipsoid)

#else

#ifndef LMP_BOND_ELLIPSOID_H
#define LMP_BOND_ELLIPSOID_H

#include <cstdio>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <cmath>
#include "bond_molc.h"
#include "math_extra.h"
#include "math_const.h"

namespace LAMMPS_NS {

  class BondEllipsoid : public BondMolc {
  public:
    BondEllipsoid(class LAMMPS *);
    virtual ~BondEllipsoid();
    virtual void compute(int, int);
    void settings(int, char **);
    void coeff(int, char **);
    void init_style();
    double equilibrium_distance(int);
    void write_restart(FILE *);
    void read_restart(FILE *);
    void write_data(FILE *);
    double single(int, double, int, int, double &);

    std::vector<double>
    force1_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    force2_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    torque1_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    torque2_numeric(int btype, int atom1, int atom2, double delta_move);

    class CouplerBondPotential {
    public:
      virtual std::string name() const = 0;
      virtual bool contribute_to_virial() const = 0;
      virtual std::vector<double> force1(double rnorm, double fbond,
					 double* v1, double* v2) = 0;
      virtual std::vector<double> force2(double rnorm, double fbond,
					 double* v1, double* v2) = 0;
      virtual std::vector<double> torque1(double fbond, double* v1,
					  double* v2) = 0;
      virtual std::vector<double> torque2(double fbond, double* v1,
					  double* v2) = 0;
    };

    class BondPotential {
    public:
      BondPotential(std::string n,
		    int r,
		    CouplerBondPotential *c) : name(n),
					       rank(r),
					       coupler(c)
	{}

      virtual int maxparams() const = 0;
      virtual int nparams() const = 0;
      bool contribute_to_virial() const {
	return coupler->contribute_to_virial();
      }

      std::vector<double>& get_params() { return params; }
      std::string get_name() {
	std::stringstream ss;
	ss << rank;
	return name + ss.str() + coupler->name();
      }
      virtual std::pair<double, double> compute(double x) = 0;
      virtual CouplerBondPotential *potential() {
	return coupler;
      }

    protected:
      std::string name;
      int rank;
      std::vector<double> params;
      CouplerBondPotential *coupler;
    };

    class CouplerRRBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("RR"); }
      bool contribute_to_virial() const { return true; }
      std::vector<double> force1(double rnorm,
				 double fbond,
				 double* v1,
				 double* v2) {
	std::vector<double> f(3);
	f[0] = -fbond*v1[0];
	f[1] = -fbond*v1[1];
	f[2] = -fbond*v1[2];
	return f;
      }
      std::vector<double> force2(double rnorm,
				 double fbond,
				 double* v1,
				 double* v2) {
	std::vector<double> f(3);
	f[0] = fbond*v2[0];
	f[1] = fbond*v2[1];
	f[2] = fbond*v2[2];
	return f;
      }
      std::vector<double> torque1(double fbond, double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque2(double fbond, double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
    };

    class CouplerXXBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("XX"); }
      bool contribute_to_virial() const { return false; }
      std::vector<double> force1(double rnorm, double fbond,
				 double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> force2(double rnorm, double fbond,
				 double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque1(double fbond, double* v1, double* v2) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(v1, v2, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
      std::vector<double> torque2(double fbond, double* v1, double* v2) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(v2, v1, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
    };

    class CouplerXRBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("XR"); }
      bool contribute_to_virial() const { return true; }
      std::vector<double> force1(double rnorm, double fbond,
				 double* x, double* r) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = fbond*ff[0]/rnorm;
	f[1] = fbond*ff[1]/rnorm;
	f[2] = fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> force2(double rnorm, double fbond,
				 double* x, double* r) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = -fbond*ff[0]/rnorm;
	f[1] = -fbond*ff[1]/rnorm;
	f[2] = -fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> torque1(double fbond, double* x, double* r) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(x, r, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
      std::vector<double> torque2(double fbond, double* x, double* r) {
	return std::vector<double>(3, 0.0);
      }
    };

    class CouplerRXBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("RX"); }
      bool contribute_to_virial() const { return true; }
      std::vector<double> force1(double rnorm, double fbond,
				 double* r, double* x) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = fbond*ff[0]/rnorm;
	f[1] = fbond*ff[1]/rnorm;
	f[2] = fbond*ff[2]/rnorm;
	return f;
      }

      std::vector<double> force2(double rnorm, double fbond,
				 double* r, double* x) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = -fbond*ff[0]/rnorm;
	f[1] = -fbond*ff[1]/rnorm;
	f[2] = -fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> torque1(double fbond, double* r, double* x) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque2(double fbond, double* r, double* x) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(r, x, n);
	t[0] = -fbond*n[0];
	t[1] = -fbond*n[1];
	t[2] = -fbond*n[2];
	return t;
      }
    };

    class PolynomialBondPotential : public BondPotential {
    public:
      PolynomialBondPotential(std::string n,
			      int r,
			      CouplerBondPotential * c) :
        BondPotential(n, r, c)
	{}

      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }

      std::pair<double, double> compute(double x) {
      	int const p_rank = 2;

      	double grad = 0.0;
      	double energy = params[0];
      	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];

	  double pow1 = p1*pow(x-p2, (double)(i));

      	  energy += pow1*(x-p2);

      	  grad += (double)(i+1)*pow1;
      	}

      	return std::make_pair(energy, grad);
      }
    };

    class CosinesHalfBondPotential : public BondPotential {
    public:

      CosinesHalfBondPotential(std::string n,
			       int r,
			       CouplerBondPotential * c)
	: BondPotential(n, r, c)
	{}

      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }

      std::pair<double, double> compute(double x) {
	int const p_rank = 2;

	double grad = 0.0;
	double energy = params[0];
	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];

	  energy += p1*(1.0-cos(0.5*MathConst::MY_PI*(i+1)*x-p2));
          grad   +=
            p1*0.5*(i+1)*MathConst::MY_PI*sin(0.5*MathConst::MY_PI*(i+1)*x-p2);
	}

    	return std::make_pair(energy, grad);
      }
    };

    class BondPotentialFactory {
    public:
      BondPotentialFactory() {
	if (allowed.size() > 0 || coupler.size() > 0) {
	  exit(1);
	}

	allowed.push_back("poly");
	allowed.push_back("cos");

	coupler.push_back(new CouplerRRBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
      }

      bool find(std::string name) {
	return std::find(allowed.begin(), allowed.end(), name)
	  != allowed.end();
      }

      BondPotential* create(std::string type, int rank, int index) {
	if (type == std::string("poly"))
	  return new PolynomialBondPotential("poly",
                                             rank, coupler[index]);
	else if (type == std::string("cos"))
	  return new CosinesHalfBondPotential("cos",
                                              rank, coupler[index]);
	else { printf("Unknown bond potential: %s. Abort\n", type.c_str()); exit(1); }
      }

      std::vector<std::string> allowed;
      std::vector<CouplerBondPotential*> coupler;
    };

  protected:
    BondPotentialFactory factory;
    std::vector<std::vector<BondPotential*> > bondPotSet;

    static int const pterms = 16;
    class AtomVecEllipsoid *avec;

    void allocate();
    char* hfilename(char const*);

  private:
    FILE* hdebug_val;
    FILE* hdebug_ene;
    FILE* hdebug_gra;
    FILE* hdebug_num;
    bool verbose;
    bool numeric_gradients;
    bool use_numeric_gradients;
    double delta_move;
  };
}

#endif
#endif

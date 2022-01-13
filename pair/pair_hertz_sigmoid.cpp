
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
   Contributing author: Sai Jayaraman (Sandia)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_hertz_sigmoid.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-5
#define MAXCOSH     1.e6

/* ---------------------------------------------------------------------- */

PairHertzSigmoid::PairHertzSigmoid(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairHertzSigmoid::~PairHertzSigmoid()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a);
    memory->destroy(b);
    memory->destroy(c);
    memory->destroy(d);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairHertzSigmoid::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rr, sech,_cosh;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double ehertz,fhertz,esigm,fsigm;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        rr = sqrt(rsq);
        _cosh = cosh (c[itype][jtype]*(rr-b[itype][jtype]));

        if (rr<b[itype][jtype]){
          fhertz = 2.5*a[itype][jtype]*pow((b[itype][jtype]-rr),1.5);
          ehertz = a[itype][jtype]*pow((b[itype][jtype]-rr),2.5) ;
        }
        else {
          fhertz = 0;
          ehertz = 0;
        }

        // if (rr>b[itype][jtype]-10./c[itype][jtype] && rr<b[itype][jtype]+10./c[itype][jtype])
        if (_cosh<MAXCOSH)
        { //deal with numerically problematic values of sech
          sech = 1./_cosh;
          fsigm = -c[itype][jtype]*d[itype][jtype]*sech*sech;
          esigm = d[itype][jtype] *tanh(c[itype][jtype]*(rr-b[itype][jtype]));
        }
        else{
          fsigm = 0;
          esigm = d[itype][jtype] *tanh(c[itype][jtype]*(rr-b[itype][jtype])) ;
        }

        fpair = fhertz+fsigm;

        f[i][0] += delx*fpair/rr;
        f[i][1] += dely*fpair/rr;
        f[i][2] += delz*fpair/rr;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair/rr;
          f[j][1] -= dely*fpair/rr;
          f[j][2] -= delz*fpair/rr;
        }

        if (eflag)
        evdwl = ehertz+esigm-offset[itype][jtype];

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
        }

    }

    if (vflag_fdotr) virial_fdotr_compute();
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairHertzSigmoid::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut_hertz");
  memory->create(a,n+1,n+1,"pair:a");
  memory->create(b,n+1,n+1,"pair:b");
  memory->create(c,n+1,n+1,"pair:c");
  memory->create(d,n+1,n+1,"pair:d");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairHertzSigmoid::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairHertzSigmoid::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a_one = force->numeric(FLERR,arg[2]);
  double b_one = force->numeric(FLERR,arg[3]);
  double c_one = force->numeric(FLERR,arg[4]);
  double d_one = force->numeric(FLERR,arg[5]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j<=jhi; j++) {
      a[i][j] = a_one;
      b[i][j] = b_one;
      c[i][j] = c_one;
      d[i][j] = d_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairHertzSigmoid::init_one(int i, int j)
{

   if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  // cutoff correction to energy
  if (offset_flag) {
    offset[i][j] =  d[i][j] *tanh(c[i][j]*(cut[i][j]-b[i][j]));
    }
  else offset[i][j] = 0.0;

  a[j][i] = a[i][j];
  b[j][i] = b[i][j];
  c[j][i] = c[i][j];
  d[j][i] = d[i][j];
  

  offset[j][i] = offset[i][j];
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHertzSigmoid::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a[i][j],sizeof(double),1,fp);
        fwrite(&b[i][j],sizeof(double),1,fp);
        fwrite(&c[i][j],sizeof(double),1,fp);
        fwrite(&d[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHertzSigmoid::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&a[i][j],sizeof(double),1,fp);
          fread(&b[i][j],sizeof(double),1,fp);
          fread(&c[i][j],sizeof(double),1,fp);
          fread(&d[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&c[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&d[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHertzSigmoid::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHertzSigmoid::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairHertzSigmoid::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,a[i][i],b[i][i],c[i][i],d[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairHertzSigmoid::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",i,j,a[i][j],b[i][j],c[i][j],d[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairHertzSigmoid::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  // printf("A %g B %g \n", a[itype][jtype], b[itype][jtype]);
  double phi ,r,fhertz,ehertz,fpair, esigm,fsigm,sech,_cosh;
  r = sqrt(rsq);
  
      if (rsq <= cutsq[itype][jtype]) {

        _cosh = cosh (c[itype][jtype]*(r-b[itype][jtype]));

        if (r<b[itype][jtype]){
          fhertz = 2.5*a[itype][jtype]*pow((b[itype][jtype]-r),1.5);
          ehertz = a[itype][jtype]*pow((b[itype][jtype]-r),2.5);
        }
        else {
          fhertz = 0;
          ehertz = 0;
        }

        // if (rr>b[itype][jtype]-10./c[itype][jtype] && rr<b[itype][jtype]+10./c[itype][jtype])
        if (_cosh<MAXCOSH)
        { //deal with numerically problematic values of sech
          sech = 1./_cosh;
          fsigm = -c[itype][jtype]*d[itype][jtype]*sech*sech;
          esigm = d[itype][jtype] *tanh(c[itype][jtype]*(r-b[itype][jtype]));
        }
        else{

          fsigm = 0;
          esigm = d[itype][jtype] *tanh(c[itype][jtype]*(r-b[itype][jtype])) ;
        }

        fpair = fhertz+fsigm;
        fforce = fpair/r;

        phi = ehertz + esigm-offset[itype][jtype];
  } 

  else {
    phi=0;
    fforce=0;
  }

  return phi;
}

/* ---------------------------------------------------------------------- */

void *PairHertzSigmoid::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"a") == 0) return (void *) a;
  if (strcmp(str,"b") == 0) return (void *) b;
  return NULL;
}

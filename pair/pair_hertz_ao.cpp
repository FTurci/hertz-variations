
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
#include "pair_hertz_ao.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-5
#define MAXCOSH     1.e3

/* ---------------------------------------------------------------------- */

PairHertzAo::PairHertzAo(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairHertzAo::~PairHertzAo()
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

    memory->destroy(aocoeff1);
    memory->destroy(aocoeff2);
    memory->destroy(aocoeff3);
    memory->destroy(min_eao);
  }
}

/* ---------------------------------------------------------------------- */

void PairHertzAo::compute(int eflag, int vflag)
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
  


  double ehertz,fhertz,eao,fao;

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

      rr = sqrt(rsq);

      // AO contribution

      // printf("sigmap %g \n", sigmap);
      if (rr < a[itype][jtype]+sigmap && rr >= a[itype][jtype]) {
        // printf("hey\n");
          // printf("a1 %g \n",aocoeff1[itype][jtype]);
          // printf("a2 %g \n",aocoeff2[itype][jtype]);
          // printf("a3 %g \n",aocoeff3[itype][jtype]);
        fao = aocoeff1[itype][jtype]*(aocoeff2[itype][jtype]-3*aocoeff3[itype][jtype]*rr*rr); 
        eao = aocoeff1[itype][jtype]*(1-aocoeff2[itype][jtype]*rr + aocoeff3[itype][jtype]*rr*rr*rr); 
      }
      else{
        fao = 0.0;
        eao = 0.0;
      }

      //Hertzian contribution
      if (rr<a[itype][jtype]){
          fhertz = c[itype][jtype]*b[itype][jtype]*pow((a[itype][jtype]-rr),c[itype][jtype]-1);
          ehertz = b[itype][jtype]*pow((a[itype][jtype]-rr),c[itype][jtype])+min_eao[itype][jtype] ;
        }
        else {
          fhertz = 0.0;
          ehertz = 0.0;
        }

        fpair = fao+fhertz;


        f[i][0] += delx*fpair/rr;
        f[i][1] += dely*fpair/rr;
        f[i][2] += delz*fpair/rr;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair/rr;
          f[j][1] -= dely*fpair/rr;
          f[j][2] -= delz*fpair/rr;
        }

        if (eflag)
        evdwl = eao+ehertz;

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
        }

    }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairHertzAo::allocate()
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

  memory->create(aocoeff1,n+1,n+1,"pair:aocoeff1");
  memory->create(aocoeff2,n+1,n+1,"pair:aocoeff2");
  memory->create(aocoeff3,n+1,n+1,"pair:aocoeff3");
  memory->create(min_eao,n+1,n+1,"pair:aocoeff3");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairHertzAo::settings(int narg, char **arg)
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

void PairHertzAo::coeff(int narg, char **arg)
{
  // printf("narg %d \n",narg );
  if (narg < 8 || narg > 9)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a_one = force->numeric(FLERR,arg[2]);
  double b_one = force->numeric(FLERR,arg[3]);
  double c_one = force->numeric(FLERR,arg[4]);
  double d_one = force->numeric(FLERR,arg[5]);

  zp = force->numeric(FLERR,arg[6]);
  sigmap = force->numeric(FLERR,arg[7]);

  double cut_one = cut_global;

  // if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j<=jhi; j++) {

      a[i][j] = a_one; //printf("a\n");
      b[i][j] = b_one; //printf("b\n");
      c[i][j] = c_one; //printf("c\n");
      printf("c %g\n",c[i][j]);
      d[i][j] = d_one; //printf("d\n");
      cut[i][j] = cut_one; //printf("cut\n");

      aocoeff1[i][j] = - M_PI/6* pow(sigmap,3)*zp*pow(1+a[i][j]/sigmap,3);// printf("a1 %g \n", aocoeff1[i][j]);
      aocoeff2[i][j] = 3./2./(1+sigmap/a[i][j])/a[i][j];//printf("a2 %g \n",aocoeff2[i][j]);
      aocoeff3[i][j] = 1./2./pow( a[i][j]*(1+sigmap/a[i][j]), 3);//printf("a3 %g \n",aocoeff3[i][j]);


      min_eao [i][j] = aocoeff1[i][j]*(1-aocoeff2[i][j] *a[i][j]+ aocoeff3[i][j]*a[i][j]*a[i][j]*a[i][j]); 

      setflag[i][j] = 1;
      count++;
    }
  }
  // printf("Setup done.\n");
  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairHertzAo::init_one(int i, int j)
{

   if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  // cutoff correction to energy
  if (offset_flag) {
    // double expform = exp(-c[i][j]*(cut[i][j]-b[i][j]));
    // offset[i][j] = a[i][j]*pow((b[i][j]-cut[i][j]),2.5) +d[i][j] /(1.0 + expform);
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

void PairHertzAo::write_restart(FILE *fp)
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

void PairHertzAo::read_restart(FILE *fp)
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

void PairHertzAo::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHertzAo::read_restart_settings(FILE *fp)
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

void PairHertzAo::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,a[i][i],b[i][i],c[i][i],d[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairHertzAo::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",i,j,a[i][j],b[i][j],c[i][j],d[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairHertzAo::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  // printf("A %g B %g \n", a[itype][jtype], b[itype][jtype]);
  double phi ,rr,fhertz,ehertz,fpair,fao,eao;
  rr = sqrt(rsq);
  
    // AO contribution
    if (rr < a[itype][jtype]+sigmap && rsq >= a[itype][jtype]) {
      fao = aocoeff1[itype][jtype]*(aocoeff2[itype][jtype]-3*aocoeff3[itype][jtype]*rr*rr); 
      eao = aocoeff1[itype][jtype]*(1-aocoeff2[itype][jtype]*rr + aocoeff3[itype][jtype]*rr*rr*rr); 
      
      // printf("evaluate %g %g %g \n",fao, eao,aocoeff1[itype][jtype]);

    }
    else{
      fao =0;
      eao=0;
    }
   if (rr<a[itype][jtype]){
        fhertz = c[itype][jtype]*b[itype][jtype]*pow((a[itype][jtype]-rr),c[itype][jtype]-1.0);
        ehertz = b[itype][jtype]*pow((a[itype][jtype]-rr),c[itype][jtype]) +min_eao[itype][jtype];
      }
      else {
        fhertz = 0;
        ehertz = 0;
      }

      
  fforce = fao+fhertz;
  phi = eao+ehertz;

  return phi;
}

/* ---------------------------------------------------------------------- */

void *PairHertzAo::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"a") == 0) return (void *) a;
  if (strcmp(str,"b") == 0) return (void *) b;
  return NULL;
}

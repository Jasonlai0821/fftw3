/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "threads.h"

typedef struct {
     solver super;
     int vecloop_dim;
     const int *buddies;
     size_t nbuddies;
} S;

typedef struct {
     plan_rdft super;
     plan **cldrn;
     INT its, ots;
     int nthr;
     const S *solver;
} P;

typedef struct {
     INT its, ots;
     FFTW_REAL_TYPE *I, *O;
     plan **cldrn;
} PD;

static void *spawn_apply(spawn_data *d)
{
     PD *ego = (PD *) d->data;
     int thr_num = d->thr_num;
     plan_rdft *cld = (plan_rdft *) ego->cldrn[d->thr_num];

     cld->apply((plan *) cld,
		ego->I + thr_num * ego->its, ego->O + thr_num * ego->ots);
     return 0;
}

static void apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O)
{
     const P *ego = (const P *) ego_;
     PD d;

     d.its = ego->its;
     d.ots = ego->ots;
     d.cldrn = ego->cldrn;
     d.I = I; d.O = O;

     fftw_spawn_loop(ego->nthr, ego->nthr, spawn_apply, (void*) &d);
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;
     int i;
     for (i = 0; i < ego->nthr; ++i)
	  fftw_plan_awake(ego->cldrn[i], wakefulness);
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     int i;
     for (i = 0; i < ego->nthr; ++i)
	  fftw_plan_destroy_internal(ego->cldrn[i]);
     fftw_ifree(ego->cldrn);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     const S *s = ego->solver;
     int i;
     p->print(p, "(rdft-thr-vrank>=1-x%d/%d", ego->nthr, s->vecloop_dim);
     for (i = 0; i < ego->nthr; ++i)
	  if (i == 0 || (ego->cldrn[i] != ego->cldrn[i-1] &&
			 (i <= 1 || ego->cldrn[i] != ego->cldrn[i-2])))
	       p->print(p, "%(%p%)", ego->cldrn[i]);
     p->putchr(p, ')');
}

static int pickdim(const S *ego, const tensor *vecsz, int oop, int *dp)
{
     return fftw_pickdim(ego->vecloop_dim, ego->buddies, ego->nbuddies,
		       vecsz, oop, dp);
}

static int applicable0(const solver *ego_, const problem *p_,
		       const planner *plnr, int *dp)
{
     const S *ego = (const S *) ego_;
     const problem_rdft *p = (const problem_rdft *) p_;

     return (1
	     && plnr->nthr > 1
	     && FINITE_RNK(p->vecsz->rnk)
	     && p->vecsz->rnk > 0
	     && pickdim(ego, p->vecsz, p->I != p->O, dp)
	  );
}

static int applicable(const solver *ego_, const problem *p_,
		      const planner *plnr, int *dp)
{
     const S *ego = (const S *)ego_;

     if (!applicable0(ego_, p_, plnr, dp)) return 0;

     /* fftw2 behavior */
     if (NO_VRANK_SPLITSP(plnr) && (ego->vecloop_dim != ego->buddies[0]))
	  return 0;

     return 1;
}

static plan *mkplan(const solver *ego_, const problem *p_, planner *plnr)
{
     const S *ego = (const S *) ego_;
     const problem_rdft *p;
     P *pln;
     problem *cldp;
     int vdim;
     iodim *d;
     plan **cldrn = (plan **) 0;
     int i, nthr;
     INT its, ots, block_size;
     tensor *vecsz;

     static const plan_adt padt = {
	  fftw_rdft_solve, awake, print, destroy
     };

     if (!applicable(ego_, p_, plnr, &vdim))
          return (plan *) 0;
     p = (const problem_rdft *) p_;

     d = p->vecsz->dims + vdim;

     block_size = (d->n + plnr->nthr - 1) / plnr->nthr;
     nthr = (int)((d->n + block_size - 1) / block_size);
     plnr->nthr = (plnr->nthr + nthr - 1) / nthr;
     its = d->is * block_size;
     ots = d->os * block_size;

     cldrn = (plan **)MALLOC(sizeof(plan *) * nthr, PLANS);
     for (i = 0; i < nthr; ++i) cldrn[i] = (plan *) 0;
     
     vecsz = fftw_tensor_copy(p->vecsz);
     for (i = 0; i < nthr; ++i) {
	  vecsz->dims[vdim].n =
	       (i == nthr - 1) ? (d->n - i*block_size) : block_size;
	  cldp = fftw_mkproblem_rdft(p->sz, vecsz,
				   p->I + i*its, p->O + i*ots, p->kind);
	  cldrn[i] = fftw_mkplan_d(plnr, cldp);
	  if (!cldrn[i]) goto nada;
     }
     fftw_tensor_destroy(vecsz);

     pln = MKPLAN_RDFT(P, &padt, apply);

     pln->cldrn = cldrn;
     pln->its = its;
     pln->ots = ots;
     pln->nthr = nthr;

     pln->solver = ego;
     fftw_ops_zero(&pln->super.super.ops);
     pln->super.super.pcost = 0;
     for (i = 0; i < nthr; ++i) {
	  fftw_ops_add2(&cldrn[i]->ops, &pln->super.super.ops);
	  pln->super.super.pcost += cldrn[i]->pcost;
     }

     return &(pln->super.super);

 nada:
     if (cldrn) {
	  for (i = 0; i < nthr; ++i)
	       fftw_plan_destroy_internal(cldrn[i]);
	  fftw_ifree(cldrn);
     }
     fftw_tensor_destroy(vecsz);
     return (plan *) 0;
}

static solver *mksolver(int vecloop_dim, const int *buddies, size_t nbuddies)
{
     static const solver_adt sadt = { PROBLEM_RDFT, mkplan, 0 };
     S *slv = MKSOLVER(S, &sadt);
     slv->vecloop_dim = vecloop_dim;
     slv->buddies = buddies;
     slv->nbuddies = nbuddies;
     return &(slv->super);
}

void fftw_rdft_thr_vrank_geq1_register(planner *p)
{
     size_t i;

     /* FIXME: Should we try other vecloop_dim values? */
     static const int buddies[] = { 1, -1 };

     for (i = 0; i < NELEM(buddies); ++i)
          REGISTER_SOLVER(p, mksolver(buddies[i], buddies, NELEM(buddies)));
}

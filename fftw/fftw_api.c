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

#include "fftw/fftw_api.h"
#include <string.h>
#include <stdio.h>

#include <math.h>

#define BUFSZ 256
const char fftw_cc[] = FFTW_CC;

/* fftw <= 3.2.2 had special compiler flags for codelets, which are
   not used anymore.  We keep this variable around because it is part
   of the ABI */
const char fftw_codelet_optim[] = "";

const char fftw_version[] = PACKAGE "-" PACKAGE_VERSION

#if HAVE_FMA
"-fma"
#endif

#if HAVE_SSE2
"-sse2"
#endif

/* Earlier versions of FFTW only provided 256-bit AVX, which meant
 * it was important to also enable sse2 for best performance for
 * short transforms. Since some programs check for this and warn
 * the user, we explicitly add avx_128 to the suffix to emphasize
 * that this version is more capable.
 */

#if HAVE_AVX
"-avx"
#endif

#if HAVE_AVX_128_FMA
"-avx_128_fma"
#endif

#if HAVE_AVX2
"-avx2-avx2_128"
#endif

#if HAVE_AVX512
"-avx512"
#endif

#if HAVE_KCVI
"-kcvi"
#endif

#if HAVE_ALTIVEC
"-altivec"
#endif

#if HAVE_VSX
"-vsx"
#endif

#if HAVE_NEON
"-neon"
#endif

#if defined(HAVE_GENERIC_SIMD128)
"-generic_simd128"
#endif

#if defined(HAVE_GENERIC_SIMD256)
"-generic_simd256"
#endif
;

/* a flag operation: x is either a flag, in which case xm == 0, or
   a mask, in which case xm == x; using this we can compactly code
   the various bit operations via (flags & x) ^ xm or (flags | x) ^ xm. */
typedef struct {
    unsigned x, xm;
} flagmask;

typedef struct {
    flagmask flag;
    flagmask op;
} flagop;

#define FLAGP(f, msk)(((f) & (msk).x) ^ (msk).xm)
#define OP(f, msk)(((f) | (msk).x) ^ (msk).xm)

#define YES(x) {x, 0}
#define NO(x) {x, x}
#define IMPLIES(predicate, consequence) { predicate, consequence }
#define EQV(a, b) IMPLIES(YES(a), YES(b)), IMPLIES(NO(a), NO(b))
#define NEQV(a, b) IMPLIES(YES(a), NO(b)), IMPLIES(NO(a), YES(b))

static void map_flags(unsigned *iflags, unsigned *oflags,
                      const flagop flagmap[], size_t nmap) {
    size_t i;
    for (i = 0; i < nmap; ++i)
        if (FLAGP(*iflags, flagmap[i].flag))
            *oflags = OP(*oflags, flagmap[i].op);
}

/* encoding of the planner timelimit into a BITS_FOR_TIMELIMIT-bits
   nonnegative integer, such that we can still view the integer as
   ``impatience'': higher means *lower* time limit, and 0 is the
   highest possible value (about 1 year of calendar time) */
static unsigned timelimit_to_flags(double timelimit) {
    const double tmax = 365 * 24 * 3600;
    const double tstep = 1.05;
    const int nsteps = (1 << BITS_FOR_TIMELIMIT);
    int x;

    if (timelimit < 0 || timelimit >= tmax)
        return 0;
    if (timelimit <= 1.0e-10)
        return nsteps - 1;

    x = (int) (0.5 + (log(tmax / timelimit) / log(tstep)));

    if (x < 0) x = 0;
    if (x >= nsteps) x = nsteps - 1;
    return x;
}

void fftw_mapflags(planner *plnr, unsigned flags) {
    unsigned l, u, t;

    /* map of fftw flags -> fftw flags, to implement consistency rules
       and combination flags */
    const flagop self_flagmap[] = {
            /* in some cases (notably for halfcomplex->real transforms),
               DESTROY_INPUT is the default, so we need to support
               an inverse flag to disable it.

               (PRESERVE, DESTROY)   ->   (PRESERVE, DESTROY)
                     (0, 0)                       (1, 0)
                     (0, 1)                       (0, 1)
                     (1, 0)                       (1, 0)
                     (1, 1)                       (1, 0)
            */
            IMPLIES(YES(FFTW_PRESERVE_INPUT), NO(FFTW_DESTROY_INPUT)),
            IMPLIES(NO(FFTW_DESTROY_INPUT), YES(FFTW_PRESERVE_INPUT)),

            IMPLIES(YES(FFTW_EXHAUSTIVE), YES(FFTW_PATIENT)),

            IMPLIES(YES(FFTW_ESTIMATE), NO(FFTW_PATIENT)),
            IMPLIES(YES(FFTW_ESTIMATE),
                    YES(FFTW_ESTIMATE_PATIENT
                        | FFTW_NO_INDIRECT_OP
                        | FFTW_ALLOW_PRUNING)),

            IMPLIES(NO(FFTW_EXHAUSTIVE),
                    YES(FFTW_NO_SLOW)),

            /* a canonical set of fftw2-like impatience flags */
            IMPLIES(NO(FFTW_PATIENT),
                    YES(FFTW_NO_VRECURSE
                        | FFTW_NO_RANK_SPLITS
                        | FFTW_NO_VRANK_SPLITS
                        | FFTW_NO_NONTHREADED
                        | FFTW_NO_DFT_R2HC
                        | FFTW_NO_FIXED_RADIX_LARGE_N
                        | FFTW_BELIEVE_PCOST))
    };

    /* map of (processed) fftw flags to internal problem/planner flags */
    const flagop l_flagmap[] = {
            EQV(FFTW_PRESERVE_INPUT, NO_DESTROY_INPUT),
            EQV(FFTW_NO_SIMD, NO_SIMD),
            EQV(FFTW_CONSERVE_MEMORY, CONSERVE_MEMORY),
            EQV(FFTW_NO_BUFFERING, NO_BUFFERING),
            NEQV(FFTW_ALLOW_LARGE_GENERIC, NO_LARGE_GENERIC)
    };

    const flagop u_flagmap[] = {
            IMPLIES(YES(FFTW_EXHAUSTIVE), NO(0xFFFFFFFF)),
            IMPLIES(NO(FFTW_EXHAUSTIVE), YES(NO_UGLY)),

            /* the following are undocumented, "beyond-guru" flags that
               require some understanding of FFTW internals */
            EQV(FFTW_ESTIMATE_PATIENT, ESTIMATE),
            EQV(FFTW_ALLOW_PRUNING, ALLOW_PRUNING),
            EQV(FFTW_BELIEVE_PCOST, BELIEVE_PCOST),
            EQV(FFTW_NO_DFT_R2HC, NO_DFT_R2HC),
            EQV(FFTW_NO_NONTHREADED, NO_NONTHREADED),
            EQV(FFTW_NO_INDIRECT_OP, NO_INDIRECT_OP),
            EQV(FFTW_NO_RANK_SPLITS, NO_RANK_SPLITS),
            EQV(FFTW_NO_VRANK_SPLITS, NO_VRANK_SPLITS),
            EQV(FFTW_NO_VRECURSE, NO_VRECURSE),
            EQV(FFTW_NO_SLOW, NO_SLOW),
            EQV(FFTW_NO_FIXED_RADIX_LARGE_N, NO_FIXED_RADIX_LARGE_N)
    };

    map_flags(&flags, &flags, self_flagmap, NELEM(self_flagmap));

    l = u = 0;
    map_flags(&flags, &l, l_flagmap, NELEM(l_flagmap));
    map_flags(&flags, &u, u_flagmap, NELEM(u_flagmap));

    /* enforce l <= u  */
    PLNR_L(plnr) = l;
    PLNR_U(plnr) = u | l;

    /* assert that the conversion didn't lose bits */
    A(PLNR_L(plnr) == l);
    A(PLNR_U(plnr) == (u | l));

    /* compute flags representation of the timelimit */
    t = timelimit_to_flags(plnr->timelimit);

    PLNR_TIMELIMIT_IMPATIENCE(plnr) = t;
    A(PLNR_TIMELIMIT_IMPATIENCE(plnr) == t);
}

void *fftw_malloc(size_t n) {
    return fftw_kernel_malloc(n);
}

void fftw_free(void *p) {
    fftw_kernel_free(p);
}

/* The following two routines are mainly for the convenience of
   the Fortran 2003 API, although C users may find them convienent
   as well.  The problem is that, although Fortran 2003 has a
   c_sizeof intrinsic that is equivalent to sizeof, it is broken
   in some gfortran versions, and in any case is a bit unnatural
   in a Fortran context.  So we provide routines to allocate real
   and complex arrays, which are all that are really needed by FFTW. */

FFTW_REAL_TYPE *fftw_alloc_real(size_t n) {
    return (FFTW_REAL_TYPE *) fftw_malloc(sizeof(FFTW_REAL_TYPE) * n);
}

FFTW_COMPLEX *fftw_alloc_complex(size_t n) {
    return (FFTW_COMPLEX *) fftw_malloc(sizeof(FFTW_COMPLEX) * n);
}

static planner_hook_t before_planner_hook = 0, after_planner_hook = 0;

void fftw_set_planner_hooks(planner_hook_t before, planner_hook_t after) {
    before_planner_hook = before;
    after_planner_hook = after;
}

static plan *mkplan0(planner *plnr, unsigned flags,
                     const problem *prb, unsigned hash_info,
                     wisdom_state_t wisdom_state) {
    /* map API flags into FFTW flags */
    fftw_mapflags(plnr, flags);

    plnr->flags.hash_info = hash_info;
    plnr->wisdom_state = wisdom_state;

    /* create plan */
    return plnr->adt->ifftw_mkplan(plnr, prb);
}

static unsigned force_estimator(unsigned flags) {
    flags &= ~(FFTW_MEASURE | FFTW_PATIENT | FFTW_EXHAUSTIVE);
    return (flags | FFTW_ESTIMATE);
}

static plan *mkplan(planner *plnr, unsigned flags,
                    const problem *prb, unsigned hash_info) {
    plan *pln;

    pln = mkplan0(plnr, flags, prb, hash_info, WISDOM_NORMAL);

    if (plnr->wisdom_state == WISDOM_NORMAL && !pln) {
        /* maybe the planner failed because of inconsistent wisdom;
           plan again ignoring infeasible wisdom */
        pln = mkplan0(plnr, force_estimator(flags), prb,
                      hash_info, WISDOM_IGNORE_INFEASIBLE);
    }

    if (plnr->wisdom_state == WISDOM_IS_BOGUS) {
        /* if the planner detected a wisdom inconsistency,
           forget all wisdom and plan again */
        plnr->adt->ifftw_forget(plnr, FORGET_EVERYTHING);

        A(!pln);
        pln = mkplan0(plnr, flags, prb, hash_info, WISDOM_NORMAL);

        if (plnr->wisdom_state == WISDOM_IS_BOGUS) {
            /* if it still fails, plan without wisdom */
            plnr->adt->ifftw_forget(plnr, FORGET_EVERYTHING);

            A(!pln);
            pln = mkplan0(plnr, force_estimator(flags),
                          prb, hash_info, WISDOM_IGNORE_ALL);
        }
    }

    return pln;
}

apiplan *fftw_mkapiplan(int sign, unsigned flags, problem *prb) {
    apiplan *p = 0;
    plan *pln;
    unsigned flags_used_for_planning;
    planner *plnr;
    static const unsigned int pats[] = {FFTW_ESTIMATE, FFTW_MEASURE,
                                        FFTW_PATIENT, FFTW_EXHAUSTIVE};
    int pat, pat_max;
    double pcost = 0;

    if (before_planner_hook)
        before_planner_hook();

    plnr = fftw_the_planner();

    if (flags & FFTW_WISDOM_ONLY) {
        /* Special mode that returns a plan only if wisdom is present,
           and returns 0 otherwise.  This is now documented in the manual,
           as a way to detect whether wisdom is available for a problem. */
        flags_used_for_planning = flags;
        pln = mkplan0(plnr, flags, prb, 0, WISDOM_ONLY);
    } else {
        pat_max = flags & FFTW_ESTIMATE ? 0 :
                  (flags & FFTW_EXHAUSTIVE ? 3 :
                   (flags & FFTW_PATIENT ? 2 : 1));
        pat = plnr->timelimit >= 0 ? 0 : pat_max;

        flags &= ~(FFTW_ESTIMATE | FFTW_MEASURE |
                   FFTW_PATIENT | FFTW_EXHAUSTIVE);

        plnr->start_time = fftw_get_crude_time();

        /* plan at incrementally increasing patience until we run
           out of time */
        for (pln = 0, flags_used_for_planning = 0; pat <= pat_max; ++pat) {
            plan *pln1;
            unsigned tmpflags = flags | pats[pat];
            pln1 = mkplan(plnr, tmpflags, prb, 0u);

            if (!pln1) {
                /* don't bother continuing if planner failed or timed out */
                A(!pln || plnr->timed_out);
                break;
            }

            fftw_plan_destroy_internal(pln);
            pln = pln1;
            flags_used_for_planning = tmpflags;
            pcost = pln->pcost;
        }
    }

    if (pln) {
        /* build apiplan */
        p = (apiplan *) MALLOC(sizeof(apiplan), PLANS);
        p->prb = prb;
        p->sign = sign; /* cache for execute_dft */

        /* re-create plan from wisdom, adding blessing */
        p->pln = mkplan(plnr, flags_used_for_planning, prb, BLESSING);

        /* record pcost from most recent measurement for use in fftw_cost */
        p->pln->pcost = pcost;

        if (sizeof(trigreal) > sizeof(FFTW_REAL_TYPE)) {
            /* this is probably faster, and we have enough trigreal
           bits to maintain accuracy */
            fftw_plan_awake(p->pln, AWAKE_SQRTN_TABLE);
        } else {
            /* more accurate */
            fftw_plan_awake(p->pln, AWAKE_SINCOS);
        }

        /* we don't use pln for p->pln, above, since by re-creating the
           plan we might use more patient wisdom from a timed-out mkplan */
        fftw_plan_destroy_internal(pln);
    } else
        fftw_problem_destroy(prb);

    /* discard all information not necessary to reconstruct the plan */
    plnr->adt->ifftw_forget(plnr, FORGET_ACCURSED);

#ifdef FFTW_RANDOM_ESTIMATOR
    fftw_random_estimate_seed++; /* subsequent "random" plans are distinct */
#endif

    if (after_planner_hook)
        after_planner_hook();

    return p;
}

void fftw_destroy_plan(fftw_plan p) {
    if (p) {
        if (before_planner_hook)
            before_planner_hook();

        fftw_plan_awake(p->pln, SLEEPY);
        fftw_plan_destroy_internal(p->pln);
        fftw_problem_destroy(p->prb);
        fftw_ifree(p);

        if (after_planner_hook)
            after_planner_hook();
    }
}

int fftw_alignment_of(FFTW_REAL_TYPE *p) {
    return fftw_ialignment_of(p);
}

void fftw_execute(const fftw_plan p) {
    plan *pln = p->pln;
    pln->adt->solve(pln, p->prb);
}

/* guru interface: requires care in alignment etcetera. */
void fftw_execute_dft(const fftw_plan p, FFTW_COMPLEX *in, FFTW_COMPLEX *out) {
    plan_dft *pln = (plan_dft *) p->pln;
    if (p->sign == FFT_SIGN)
        pln->apply((plan *) pln, in[0], in[0] + 1, out[0], out[0] + 1);
    else
        pln->apply((plan *) pln, in[0] + 1, in[0], out[0] + 1, out[0]);
}

/* guru interface: requires care in alignment, r - i, etcetera. */
void fftw_execute_dft_c2r(const fftw_plan p, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out) {
    plan_rdft2 *pln = (plan_rdft2 *) p->pln;
    problem_rdft2 *prb = (problem_rdft2 *) p->prb;
    pln->apply((plan *) pln, out, out + (prb->r1 - prb->r0), in[0], in[0] + 1);
}

/* guru interface: requires care in alignment, r - i, etcetera. */
void fftw_execute_dft_r2c(const fftw_plan p, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out) {
    plan_rdft2 *pln = (plan_rdft2 *) p->pln;
    problem_rdft2 *prb = (problem_rdft2 *) p->prb;
    pln->apply((plan *) pln, in, in + (prb->r1 - prb->r0), out[0], out[0] + 1);
}


/* guru interface: requires care in alignment, etcetera. */
void fftw_execute_r2r(const fftw_plan p, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out) {
    plan_rdft *pln = (plan_rdft *) p->pln;
    pln->apply((plan *) pln, in, out);
}

/* guru interface: requires care in alignment, r - i, etcetera. */
void fftw_execute_split_dft(const fftw_plan p, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro,
                            FFTW_REAL_TYPE *io) {
    plan_dft *pln = (plan_dft *) p->pln;
    pln->apply((plan *) pln, ri, ii, ro, io);
}


/* guru interface: requires care in alignment, r - i, etcetera. */
void
fftw_execute_split_dft_c2r(const fftw_plan p, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out) {
    plan_rdft2 *pln = (plan_rdft2 *) p->pln;
    problem_rdft2 *prb = (problem_rdft2 *) p->prb;
    pln->apply((plan *) pln, out, out + (prb->r1 - prb->r0), ri, ii);
}


/* guru interface: requires care in alignment, r - i, etcetera. */
void
fftw_execute_split_dft_r2c(const fftw_plan p, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
    plan_rdft2 *pln = (plan_rdft2 *) p->pln;
    problem_rdft2 *prb = (problem_rdft2 *) p->prb;
    pln->apply((plan *) pln, in, in + (prb->r1 - prb->r0), ro, io);
}


void fftw_flops(const fftw_plan p, double *add, double *mul, double *fma) {
    planner *plnr = fftw_the_planner();
    opcnt *o = &p->pln->ops;
    *add = o->add;
    *mul = o->mul;
    *fma = o->fma;
    if (plnr->cost_hook) {
        *add = plnr->cost_hook(p->prb, *add, COST_SUM);
        *mul = plnr->cost_hook(p->prb, *mul, COST_SUM);
        *fma = plnr->cost_hook(p->prb, *fma, COST_SUM);
    }
}

double fftw_estimate_cost(const fftw_plan p) {
    return fftw_iestimate_cost(fftw_the_planner(), p->pln, p->prb);
}

double fftw_cost(const fftw_plan p) {
    return p->pln->pcost;
}

rdft_kind *fftw_map_r2r_kind(int rank, const fftw_r2r_kind *kind) {
    int i;
    rdft_kind *k;

    A(FINITE_RNK(rank));
    k = (rdft_kind *) MALLOC((unsigned) rank * sizeof(rdft_kind), PROBLEMS);
    for (i = 0; i < rank; ++i) {
        rdft_kind m;
        switch (kind[i]) {
            case FFTW_R2HC:
                m = R2HC;
                break;
            case FFTW_HC2R:
                m = HC2R;
                break;
            case FFTW_DHT:
                m = DHT;
                break;
            case FFTW_REDFT00:
                m = REDFT00;
                break;
            case FFTW_REDFT01:
                m = REDFT01;
                break;
            case FFTW_REDFT10:
                m = REDFT10;
                break;
            case FFTW_REDFT11:
                m = REDFT11;
                break;
            case FFTW_RODFT00:
                m = RODFT00;
                break;
            case FFTW_RODFT01:
                m = RODFT01;
                break;
            case FFTW_RODFT10:
                m = RODFT10;
                break;
            case FFTW_RODFT11:
                m = RODFT11;
                break;
            default:
                m = R2HC; A(0);
        }
        k[i] = m;
    }
    return k;
}

fftw_plan fftw_plan_dft(int rank, const int *n,
                        FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    return fftw_plan_many_dft(rank, n, 1,
                              in, 0, 1, 1,
                              out, 0, 1, 1,
                              sign, flags);
}

fftw_plan fftw_plan_dft_1d(int n, FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    return fftw_plan_dft(1, &n, in, out, sign, flags);
}

fftw_plan fftw_plan_dft_2d(int nx, int ny, FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    int n[2];
    n[0] = nx;
    n[1] = ny;
    return fftw_plan_dft(2, n, in, out, sign, flags);
}

fftw_plan fftw_plan_dft_3d(int nx, int ny, int nz,
                           FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    int n[3];
    n[0] = nx;
    n[1] = ny;
    n[2] = nz;
    return fftw_plan_dft(3, n, in, out, sign, flags);
}

fftw_plan fftw_plan_dft_c2r(int rank, const int *n, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    return fftw_plan_many_dft_c2r(rank, n, 1,
                                  in, 0, 1, 1, out, 0, 1, 1, flags);
}

fftw_plan fftw_plan_dft_c2r_1d(int n, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    return fftw_plan_dft_c2r(1, &n, in, out, flags);
}

fftw_plan fftw_plan_dft_c2r_2d(int nx, int ny, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    int n[2];
    n[0] = nx;
    n[1] = ny;
    return fftw_plan_dft_c2r(2, n, in, out, flags);
}

fftw_plan fftw_plan_dft_c2r_3d(int nx, int ny, int nz,
                               FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    int n[3];
    n[0] = nx;
    n[1] = ny;
    n[2] = nz;
    return fftw_plan_dft_c2r(3, n, in, out, flags);
}

fftw_plan fftw_plan_dft_r2c(int rank, const int *n, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    return fftw_plan_many_dft_r2c(rank, n, 1,
                                  in, 0, 1, 1,
                                  out, 0, 1, 1,
                                  flags);
}

fftw_plan fftw_plan_dft_r2c_1d(int n, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    return fftw_plan_dft_r2c(1, &n, in, out, flags);
}

fftw_plan fftw_plan_dft_r2c_2d(int nx, int ny, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    int n[2];
    n[0] = nx;
    n[1] = ny;
    return fftw_plan_dft_r2c(2, n, in, out, flags);
}

fftw_plan fftw_plan_dft_r2c_3d(int nx, int ny, int nz,
                               FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    int n[3];
    n[0] = nx;
    n[1] = ny;
    n[2] = nz;
    return fftw_plan_dft_r2c(3, n, in, out, flags);
}

#define N0(nembed)((nembed) ? (nembed) : n)

fftw_plan fftw_plan_many_dft(int rank, const int *n,
                             int howmany,
                             FFTW_COMPLEX *in, const int *inembed,
                             int istride, int idist,
                             FFTW_COMPLEX *out, const int *onembed,
                             int ostride, int odist, int sign, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii, *ro, *io;

    if (!fftw_many_kosherp(rank, n, howmany)) return 0;

    EXTRACT_REIM(sign, in, &ri, &ii);
    EXTRACT_REIM(sign, out, &ro, &io);

    return
            fftw_mkapiplan(sign, flags,
                           fftw_mkproblem_dft_d(
                                   fftw_mktensor_rowmajor(rank, n,
                                                          N0(inembed), N0(onembed),
                                                          2 * istride, 2 * ostride),
                                   fftw_mktensor_1d(howmany, 2 * idist, 2 * odist),
                                   TAINT_UNALIGNED(ri, flags),
                                   TAINT_UNALIGNED(ii, flags),
                                   TAINT_UNALIGNED(ro, flags),
                                   TAINT_UNALIGNED(io, flags)));
}

fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n,
                                 int howmany,
                                 FFTW_COMPLEX *in, const int *inembed,
                                 int istride, int idist,
                                 FFTW_REAL_TYPE *out, const int *onembed,
                                 int ostride, int odist, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii;
    int *nfi, *nfo;
    int inplace;
    fftw_plan p;

    if (!fftw_many_kosherp(rank, n, howmany)) return 0;

    EXTRACT_REIM(FFT_SIGN, in, &ri, &ii);
    inplace = out == ri;

    if (!inplace)
        flags |= FFTW_DESTROY_INPUT;
    p = fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_rowmajor(
                            rank, n,
                            fftw_rdft2_pad(rank, n, inembed, inplace, 1, &nfi),
                            fftw_rdft2_pad(rank, n, onembed, inplace, 0, &nfo),
                            2 * istride, ostride),
                    fftw_mktensor_1d(howmany, 2 * idist, odist),
                    TAINT_UNALIGNED(out, flags),
                    TAINT_UNALIGNED(ri, flags), TAINT_UNALIGNED(ii, flags),
                    HC2R));

    fftw_ifree0(nfi);
    fftw_ifree0(nfo);
    return p;
}

fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n,
                                 int howmany,
                                 FFTW_REAL_TYPE *in, const int *inembed,
                                 int istride, int idist,
                                 FFTW_COMPLEX *out, const int *onembed,
                                 int ostride, int odist, unsigned flags) {
    FFTW_REAL_TYPE *ro, *io;
    int *nfi, *nfo;
    int inplace;
    fftw_plan p;

    if (!fftw_many_kosherp(rank, n, howmany)) return 0;

    EXTRACT_REIM(FFT_SIGN, out, &ro, &io);
    inplace = in == ro;

    p = fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_rowmajor(
                            rank, n,
                            fftw_rdft2_pad(rank, n, inembed, inplace, 0, &nfi),
                            fftw_rdft2_pad(rank, n, onembed, inplace, 1, &nfo),
                            istride, 2 * ostride),
                    fftw_mktensor_1d(howmany, idist, 2 * odist),
                    TAINT_UNALIGNED(in, flags),
                    TAINT_UNALIGNED(ro, flags), TAINT_UNALIGNED(io, flags),
                    R2HC));

    fftw_ifree0(nfi);
    fftw_ifree0(nfo);
    return p;
}

fftw_plan fftw_plan_r2r(int rank, const int *n, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                        const fftw_r2r_kind *kind, unsigned flags) {
    return fftw_plan_many_r2r(rank, n, 1, in, 0, 1, 1, out, 0, 1, 1, kind,
                              flags);
}

fftw_plan
fftw_plan_r2r_1d(int n, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out, fftw_r2r_kind kind, unsigned flags) {
    return fftw_plan_r2r(1, &n, in, out, &kind, flags);
}

fftw_plan fftw_plan_r2r_2d(int nx, int ny, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                           fftw_r2r_kind kindx, fftw_r2r_kind kindy, unsigned flags) {
    int n[2];
    fftw_r2r_kind kind[2];
    n[0] = nx;
    n[1] = ny;
    kind[0] = kindx;
    kind[1] = kindy;
    return fftw_plan_r2r(2, n, in, out, kind, flags);
}

fftw_plan fftw_plan_r2r_3d(int nx, int ny, int nz,
                           FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out, fftw_r2r_kind kindx,
                           fftw_r2r_kind kindy, fftw_r2r_kind kindz, unsigned flags) {
    int n[3];
    fftw_r2r_kind kind[3];
    n[0] = nx;
    n[1] = ny;
    n[2] = nz;
    kind[0] = kindx;
    kind[1] = kindy;
    kind[2] = kindz;
    return fftw_plan_r2r(3, n, in, out, kind, flags);
}

const int *fftw_rdft2_pad(int rnk, const int *n, const int *nembed,
                          int inplace, int cmplx, int **nfree) {
    A(FINITE_RNK(rnk));
    *nfree = 0;
    if (!nembed && rnk > 0) {
        if (inplace || cmplx) {
            int *np = (int *) MALLOC(sizeof(int) * (unsigned) rnk, PROBLEMS);
            memcpy(np, n, sizeof(int) * (unsigned) rnk);
            np[rnk - 1] = (n[rnk - 1] / 2 + 1) * (1 + !cmplx);
            nembed = *nfree = np;
        } else
            nembed = n;
    }
    return nembed;
}

static planner *plnr = 0;

/* create the planner for the rest of the API */
planner *fftw_the_planner(void) {
    if (!plnr) {
        plnr = fftw_mkplanner();
        fftw_configure_planner(plnr);
    }

    return plnr;
}

void fftw_cleanup(void) {
    if (plnr) {
        fftw_planner_destroy(plnr);
        plnr = 0;
    }
}

void fftw_set_timelimit(double tlim) {
    /* PLNR is not necessarily initialized when this function is
   called, so use fftw_the_planner() */
    fftw_the_planner()->timelimit = tlim;
}


fftw_plan fftw_plan_many_r2r(int rank, const int *n,
                             int howmany,
                             FFTW_REAL_TYPE *in, const int *inembed,
                             int istride, int idist,
                             FFTW_REAL_TYPE *out, const int *onembed,
                             int ostride, int odist,
                             const fftw_r2r_kind *kind, unsigned flags) {
    fftw_plan p;
    rdft_kind *k;

    if (!fftw_many_kosherp(rank, n, howmany)) return 0;

    k = fftw_map_r2r_kind(rank, kind);
    p = fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft_d(fftw_mktensor_rowmajor(rank, n,
                                                         N0(inembed), N0(onembed),
                                                         istride, ostride),
                                  fftw_mktensor_1d(howmany, idist, odist),
                                  TAINT_UNALIGNED(in, flags),
                                  TAINT_UNALIGNED(out, flags), k));
    fftw_ifree0(k);
    return p;
}


tensor *fftw_mktensor_rowmajor(int rnk, const int *n,
                               const int *niphys, const int *nophys,
                               int is, int os) {
    tensor *x = fftw_mktensor(rnk);

    if (FINITE_RNK(rnk) && rnk > 0) {
        int i;

        A(n && niphys && nophys);
        x->dims[rnk - 1].is = is;
        x->dims[rnk - 1].os = os;
        x->dims[rnk - 1].n = n[rnk - 1];
        for (i = rnk - 1; i > 0; --i) {
            x->dims[i - 1].is = x->dims[i].is * niphys[i];
            x->dims[i - 1].os = x->dims[i].os * nophys[i];
            x->dims[i - 1].n = n[i - 1];
        }
    }
    return x;
}

static int rowmajor_kosherp(int rnk, const int *n) {
    int i;

    if (!FINITE_RNK(rnk)) return 0;
    if (rnk < 0) return 0;

    for (i = 0; i < rnk; ++i)
        if (n[i] <= 0) return 0;

    return 1;
}

int fftw_many_kosherp(int rnk, const int *n, int howmany) {
    return (howmany >= 0) && rowmajor_kosherp(rnk, n);
}


#if defined(FFTW_SINGLE)
#  define WISDOM_NAME "wisdomf"
#elif defined(FFTW_LDOUBLE)
#  define WISDOM_NAME "wisdoml"
#else
#  define WISDOM_NAME "wisdom"
#endif

/* OS-specific configuration-file directory */
#if defined(__DJGPP__)
#  define WISDOM_DIR "/dev/env/DJDIR/etc/fftw/"
#else
#  define WISDOM_DIR "/etc/fftw/"
#endif

int fftw_import_system_wisdom(void) {
#if defined(__WIN32__) || defined(WIN32) || defined(_WINDOWS)
    return 0; /* TODO? */
#else

    FILE *f;
     f = fopen(WISDOM_DIR WISDOM_NAME, "r");
     if (f) {
          int ret = fftw_import_wisdom_from_file(f);
          fclose(f);
          return ret;
     } else
          return 0;
#endif
}

typedef struct {
    printer super;

    void (*write_char)(char c, void *);

    void *data;
} export_wisdom_P;

static void putchr_generic(printer *p_, char c) {
    export_wisdom_P *p = (export_wisdom_P *) p_;
    (p->write_char)(c, p->data);
}

void fftw_export_wisdom(void (*write_char)(char c, void *), void *data) {
    export_wisdom_P *p = (export_wisdom_P *) fftw_mkprinter(sizeof(export_wisdom_P), putchr_generic, 0);
    planner *plnr = fftw_the_planner();

    p->write_char = write_char;
    p->data = data;
    plnr->adt->ifftw_exprt(plnr, (printer *) p);
    fftw_printer_destroy((printer *) p);
}

void fftw_forget_wisdom(void) {
    planner *plnr = fftw_the_planner();
    plnr->adt->ifftw_forget(plnr, FORGET_EVERYTHING);
}


void fftw_export_wisdom_to_file(FILE *output_file) {
    printer *p = fftw_mkprinter_file(output_file);
    planner *plnr = fftw_the_planner();
    plnr->adt->ifftw_exprt(plnr, p);
    fftw_printer_destroy(p);
}

int fftw_export_wisdom_to_filename(const char *filename) {
    FILE *f = fopen(filename, "w");
    int ret;
    if (!f) return 0; /* error opening file */
    fftw_export_wisdom_to_file(f);
    ret = !ferror(f);
    if (fclose(f)) ret = 0; /* error closing file */
    return ret;
}


char *fftw_export_wisdom_to_string(void) {
    printer *p;
    planner *plnr = fftw_the_planner();
    size_t cnt;
    char *s;

    p = fftw_mkprinter_cnt(&cnt);
    plnr->adt->ifftw_exprt(plnr, p);
    fftw_printer_destroy(p);

    s = (char *) malloc(sizeof(char) * (cnt + 1));
    if (s) {
        p = fftw_mkprinter_str(s);
        plnr->adt->ifftw_exprt(plnr, p);
        fftw_printer_destroy(p);
    }

    return s;
}


typedef struct {
    printer super;
    FILE *f;
    char buf[BUFSZ];
    char *bufw;
} mkprinter_P;

static void myflush(mkprinter_P *p) {
    fwrite(p->buf, 1, p->bufw - p->buf, p->f);
    p->bufw = p->buf;
}

static void myputchr(printer *p_, char c) {
    mkprinter_P *p = (mkprinter_P *) p_;
    if (p->bufw >= p->buf + BUFSZ)
        myflush(p);
    *p->bufw++ = c;
}

static void mycleanup(printer *p_) {
    mkprinter_P *p = (mkprinter_P *) p_;
    myflush(p);
}

printer *fftw_mkprinter_file(FILE *f) {
    mkprinter_P *p = (mkprinter_P *) fftw_mkprinter(sizeof(mkprinter_P), myputchr, mycleanup);
    p->f = f;
    p->bufw = p->buf;
    return &p->super;
}

typedef struct {
    printer super;
    size_t *cnt;
} P_cnt;

static void putchr_cnt(printer *p_, char c) {
    P_cnt *p = (P_cnt *) p_;
    UNUSED(c);
    ++*p->cnt;
}

printer *fftw_mkprinter_cnt(size_t *cnt) {
    P_cnt *p = (P_cnt *) fftw_mkprinter(sizeof(P_cnt), putchr_cnt, 0);
    p->cnt = cnt;
    *cnt = 0;
    return &p->super;
}

typedef struct {
    printer super;
    char *s;
} P_str;

static void putchr_str(printer *p_, char c) {
    P_str *p = (P_str *) p_;
    *p->s++ = c;
    *p->s = 0;
}

printer *fftw_mkprinter_str(char *s) {
    P_str *p = (P_str *) fftw_mkprinter(sizeof(P_str), putchr_str, 0);
    p->s = s;
    *s = 0;
    return &p->super;
}

char *fftw_sprint_plan(const fftw_plan p) {
    size_t cnt;
    char *s;
    plan *pln = p->pln;

    printer *pr = fftw_mkprinter_cnt(&cnt);
    pln->adt->print(pln, pr);
    fftw_printer_destroy(pr);

    s = (char *) malloc(sizeof(char) * (cnt + 1));
    if (s) {
        pr = fftw_mkprinter_str(s);
        pln->adt->print(pln, pr);
        fftw_printer_destroy(pr);
    }
    return s;
}

void fftw_fprint_plan(const fftw_plan p, FILE *output_file) {
    printer *pr = fftw_mkprinter_file(output_file);
    plan *pln = p->pln;
    pln->adt->print(pln, pr);
    fftw_printer_destroy(pr);
}

void fftw_print_plan(const fftw_plan p) {
    fftw_fprint_plan(p, stdout);
}

typedef struct {
    scanner super;

    int (*read_char)(void *);

    void *data;
} wisdom_S;

static int getchr_generic(scanner *s_) {
    wisdom_S *s = (wisdom_S *) s_;
    return (s->read_char)(s->data);
}

int fftw_import_wisdom(int (*read_char)(void *), void *data) {
    wisdom_S *s = (wisdom_S *) fftw_mkscanner(sizeof(wisdom_S), getchr_generic);
    planner *plnr = fftw_the_planner();
    int ret;

    s->read_char = read_char;
    s->data = data;
    ret = plnr->adt->ifftw_imprt(plnr, (scanner *) s);
    fftw_scanner_destroy((scanner *) s);
    return ret;
}


typedef struct {
    scanner super;
    FILE *f;
    char buf[BUFSZ];
    char *bufr, *bufw;
} wisdowm_file_S;

static int getchr_file(scanner *sc_) {
    wisdowm_file_S *sc = (wisdowm_file_S *) sc_;

    if (sc->bufr >= sc->bufw) {
        sc->bufr = sc->buf;
        sc->bufw = sc->buf + fread(sc->buf, 1, BUFSZ, sc->f);
        if (sc->bufr >= sc->bufw)
            return EOF;
    }

    return *(sc->bufr++);
}

static scanner *mkscanner_file(FILE *f) {
    wisdowm_file_S *sc = (wisdowm_file_S *) fftw_mkscanner(sizeof(wisdowm_file_S), getchr_file);
    sc->f = f;
    sc->bufr = sc->bufw = sc->buf;
    return &sc->super;
}

int fftw_import_wisdom_from_file(FILE *input_file) {
    scanner *s = mkscanner_file(input_file);
    planner *plnr = fftw_the_planner();
    int ret = plnr->adt->ifftw_imprt(plnr, s);
    fftw_scanner_destroy(s);
    return ret;
}

int fftw_import_wisdom_from_filename(const char *filename) {
    FILE *f = fopen(filename, "r");
    int ret;
    if (!f) return 0; /* error opening file */
    ret = fftw_import_wisdom_from_file(f);
    if (fclose(f)) ret = 0; /* error closing file */
    return ret;
}

typedef struct {
    scanner super;
    const char *s;
} S_str;

static int getchr_str(scanner *sc_) {
    S_str *sc = (S_str *) sc_;
    if (!*sc->s)
        return EOF;
    return *sc->s++;
}

static scanner *mkscanner_str(const char *s) {
    S_str *sc = (S_str *) fftw_mkscanner(sizeof(S_str), getchr_str);
    sc->s = s;
    return &sc->super;
}

int fftw_import_wisdom_from_string(const char *input_string) {
    scanner *s = mkscanner_str(input_string);
    planner *plnr = fftw_the_planner();
    int ret = plnr->adt->ifftw_imprt(plnr, s);
    fftw_scanner_destroy(s);
    return ret;
}


fftw_plan fftw_plan_guru64_dft(int rank, const fftw_iodim64 *dims,
                               int howmany_rank, const fftw_iodim64 *howmany_dims,
                               FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii, *ro, *io;

    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(sign, in, &ri, &ii);
    EXTRACT_REIM(sign, out, &ro, &io);

    return fftw_mkapiplan(
            sign, flags,
            fftw_mkproblem_dft_d(fftw_mktensor_iodims64(rank, dims, 2, 2),
                                 fftw_mktensor_iodims64(howmany_rank, howmany_dims,
                                                        2, 2),
                                 TAINT_UNALIGNED(ri, flags),
                                 TAINT_UNALIGNED(ii, flags),
                                 TAINT_UNALIGNED(ro, flags),
                                 TAINT_UNALIGNED(io, flags)));
}


fftw_plan fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64 *dims,
                                   int howmany_rank, const fftw_iodim64 *howmany_dims,
                                   FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii;

    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(FFT_SIGN, in, &ri, &ii);

    if (out != ri)
        flags |= FFTW_DESTROY_INPUT;
    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims64(rank, dims, 2, 1),
                    fftw_mktensor_iodims64(howmany_rank, howmany_dims, 2, 1),
                    TAINT_UNALIGNED(out, flags),
                    TAINT_UNALIGNED(ri, flags),
                    TAINT_UNALIGNED(ii, flags), HC2R));
}

fftw_plan fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64 *dims,
                                   int howmany_rank,
                                   const fftw_iodim64 *howmany_dims,
                                   FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    FFTW_REAL_TYPE *ro, *io;

    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(FFT_SIGN, out, &ro, &io);

    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims64(rank, dims, 1, 2),
                    fftw_mktensor_iodims64(howmany_rank, howmany_dims, 1, 2),
                    TAINT_UNALIGNED(in, flags),
                    TAINT_UNALIGNED(ro, flags),
                    TAINT_UNALIGNED(io, flags), R2HC));
}

fftw_plan fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims,
                               int howmany_rank,
                               const fftw_iodim64 *howmany_dims,
                               FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                               const fftw_r2r_kind *kind, unsigned flags) {
    fftw_plan p;
    rdft_kind *k;

    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    k = fftw_map_r2r_kind(rank, kind);
    p = fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft_d(fftw_mktensor_iodims64(rank, dims, 1, 1),
                                  fftw_mktensor_iodims64(howmany_rank, howmany_dims,
                                                         1, 1),
                                  TAINT_UNALIGNED(in, flags),
                                  TAINT_UNALIGNED(out, flags), k));
    fftw_ifree0(k);
    return p;
}

fftw_plan fftw_plan_guru64_split_dft(int rank, const fftw_iodim64 *dims,
                                     int howmany_rank, const fftw_iodim64 *howmany_dims,
                                     FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                     unsigned flags) {
    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    return fftw_mkapiplan(
            ii - ri == 1 && io - ro == 1 ? FFT_SIGN : -FFT_SIGN, flags,
            fftw_mkproblem_dft_d(fftw_mktensor_iodims64(rank, dims, 1, 1),
                                 fftw_mktensor_iodims64(howmany_rank, howmany_dims,
                                                        1, 1),
                                 TAINT_UNALIGNED(ri, flags),
                                 TAINT_UNALIGNED(ii, flags),
                                 TAINT_UNALIGNED(ro, flags),
                                 TAINT_UNALIGNED(io, flags)));
}

fftw_plan fftw_plan_guru64_split_dft_c2r(int rank, const fftw_iodim64 *dims,
                                         int howmany_rank, const fftw_iodim64 *howmany_dims,
                                         FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out, unsigned flags) {
    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    if (out != ri)
        flags |= FFTW_DESTROY_INPUT;
    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims64(rank, dims, 1, 1),
                    fftw_mktensor_iodims64(howmany_rank, howmany_dims, 1, 1),
                    TAINT_UNALIGNED(out, flags),
                    TAINT_UNALIGNED(ri, flags),
                    TAINT_UNALIGNED(ii, flags), HC2R));
}

fftw_plan fftw_plan_guru64_split_dft_r2c(int rank, const fftw_iodim64 *dims,
                                         int howmany_rank,
                                         const fftw_iodim64 *howmany_dims,
                                         FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io, unsigned flags) {
    if (!fftw_guru64_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims64(rank, dims, 1, 1),
                    fftw_mktensor_iodims64(howmany_rank, howmany_dims, 1, 1),
                    TAINT_UNALIGNED(in, flags),
                    TAINT_UNALIGNED(ro, flags),
                    TAINT_UNALIGNED(io, flags), R2HC));
}

tensor *fftw_mktensor_iodims64(int rank, const fftw_iodim64 *dims, int is, int os) {
    int i;
    tensor *x = fftw_mktensor(rank);

    if (FINITE_RNK(rank)) {
        for (i = 0; i < rank; ++i) {
            x->dims[i].n = dims[i].n;
            x->dims[i].is = dims[i].is * is;
            x->dims[i].os = dims[i].os * os;
        }
    }
    return x;
}

static int iodims64_kosherp(int rank, const fftw_iodim64 *dims, int allow_minfty) {
    int i;

    if (rank < 0) return 0;

    if (allow_minfty) {
        if (!FINITE_RNK(rank)) return 1;
        for (i = 0; i < rank; ++i)
            if (dims[i].n < 0) return 0;
    } else {
        if (!FINITE_RNK(rank)) return 0;
        for (i = 0; i < rank; ++i)
            if (dims[i].n <= 0) return 0;
    }

    return 1;
}

int fftw_guru64_kosherp(int rank, const fftw_iodim64 *dims,
                        int howmany_rank, const fftw_iodim64 *howmany_dims) {
    return (iodims64_kosherp(rank, dims, 0) &&
            iodims64_kosherp(howmany_rank, howmany_dims, 1));
}


fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                             int howmany_rank, const fftw_iodim *howmany_dims,
                             FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii, *ro, *io;

    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(sign, in, &ri, &ii);
    EXTRACT_REIM(sign, out, &ro, &io);

    return fftw_mkapiplan(
            sign, flags,
            fftw_mkproblem_dft_d(fftw_mktensor_iodims(rank, dims, 2, 2),
                                 fftw_mktensor_iodims(howmany_rank, howmany_dims,
                                                      2, 2),
                                 TAINT_UNALIGNED(ri, flags),
                                 TAINT_UNALIGNED(ii, flags),
                                 TAINT_UNALIGNED(ro, flags),
                                 TAINT_UNALIGNED(io, flags)));
}


fftw_plan fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims,
                                 int howmany_rank, const fftw_iodim *howmany_dims,
                                 FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags) {
    FFTW_REAL_TYPE *ri, *ii;

    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(FFT_SIGN, in, &ri, &ii);

    if (out != ri)
        flags |= FFTW_DESTROY_INPUT;
    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims(rank, dims, 2, 1),
                    fftw_mktensor_iodims(howmany_rank, howmany_dims, 2, 1),
                    TAINT_UNALIGNED(out, flags),
                    TAINT_UNALIGNED(ri, flags),
                    TAINT_UNALIGNED(ii, flags), HC2R));
}

fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                 int howmany_rank,
                                 const fftw_iodim *howmany_dims,
                                 FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags) {
    FFTW_REAL_TYPE *ro, *io;

    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    EXTRACT_REIM(FFT_SIGN, out, &ro, &io);

    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims(rank, dims, 1, 2),
                    fftw_mktensor_iodims(howmany_rank, howmany_dims, 1, 2),
                    TAINT_UNALIGNED(in, flags),
                    TAINT_UNALIGNED(ro, flags),
                    TAINT_UNALIGNED(io, flags), R2HC));
}

fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims,
                             int howmany_rank,
                             const fftw_iodim *howmany_dims,
                             FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                             const fftw_r2r_kind *kind, unsigned flags) {
    fftw_plan p;
    rdft_kind *k;

    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    k = fftw_map_r2r_kind(rank, kind);
    p = fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft_d(fftw_mktensor_iodims(rank, dims, 1, 1),
                                  fftw_mktensor_iodims(howmany_rank, howmany_dims,
                                                       1, 1),
                                  TAINT_UNALIGNED(in, flags),
                                  TAINT_UNALIGNED(out, flags), k));
    fftw_ifree0(k);
    return p;
}

fftw_plan fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims,
                                   int howmany_rank, const fftw_iodim *howmany_dims,
                                   FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                   unsigned flags) {
    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    return fftw_mkapiplan(
            ii - ri == 1 && io - ro == 1 ? FFT_SIGN : -FFT_SIGN, flags,
            fftw_mkproblem_dft_d(fftw_mktensor_iodims(rank, dims, 1, 1),
                                 fftw_mktensor_iodims(howmany_rank, howmany_dims,
                                                      1, 1),
                                 TAINT_UNALIGNED(ri, flags),
                                 TAINT_UNALIGNED(ii, flags),
                                 TAINT_UNALIGNED(ro, flags),
                                 TAINT_UNALIGNED(io, flags)));
}

fftw_plan fftw_plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims,
                                       int howmany_rank, const fftw_iodim *howmany_dims,
                                       FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out, unsigned flags) {
    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    if (out != ri)
        flags |= FFTW_DESTROY_INPUT;
    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims(rank, dims, 1, 1),
                    fftw_mktensor_iodims(howmany_rank, howmany_dims, 1, 1),
                    TAINT_UNALIGNED(out, flags),
                    TAINT_UNALIGNED(ri, flags),
                    TAINT_UNALIGNED(ii, flags), HC2R));
}

fftw_plan fftw_plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims,
                                       int howmany_rank,
                                       const fftw_iodim *howmany_dims,
                                       FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io, unsigned flags) {
    if (!fftw_guru_kosherp(rank, dims, howmany_rank, howmany_dims)) return 0;

    return fftw_mkapiplan(
            0, flags,
            fftw_mkproblem_rdft2_d_3pointers(
                    fftw_mktensor_iodims(rank, dims, 1, 1),
                    fftw_mktensor_iodims(howmany_rank, howmany_dims, 1, 1),
                    TAINT_UNALIGNED(in, flags),
                    TAINT_UNALIGNED(ro, flags),
                    TAINT_UNALIGNED(io, flags), R2HC));
}

tensor *fftw_mktensor_iodims(int rank, const fftw_iodim *dims, int is, int os) {
    int i;
    tensor *x = fftw_mktensor(rank);

    if (FINITE_RNK(rank)) {
        for (i = 0; i < rank; ++i) {
            x->dims[i].n = dims[i].n;
            x->dims[i].is = dims[i].is * is;
            x->dims[i].os = dims[i].os * os;
        }
    }
    return x;
}

static int iodims_kosherp(int rank, const fftw_iodim *dims, int allow_minfty) {
    int i;

    if (rank < 0) return 0;

    if (allow_minfty) {
        if (!FINITE_RNK(rank)) return 1;
        for (i = 0; i < rank; ++i)
            if (dims[i].n < 0) return 0;
    } else {
        if (!FINITE_RNK(rank)) return 0;
        for (i = 0; i < rank; ++i)
            if (dims[i].n <= 0) return 0;
    }

    return 1;
}

int fftw_guru_kosherp(int rank, const fftw_iodim *dims,
                      int howmany_rank, const fftw_iodim *howmany_dims) {
    return (iodims_kosherp(rank, dims, 0) &&
            iodims_kosherp(howmany_rank, howmany_dims, 1));
}
static const solvtab dft_conf_s =
{
	SOLVTAB(fftw_dft_indirect_register),
	SOLVTAB(fftw_dft_indirect_transpose_register),
	SOLVTAB(fftw_dft_rank_geq2_register),
	SOLVTAB(fftw_dft_vrank_geq1_register),
	SOLVTAB(fftw_dft_buffered_register),
	SOLVTAB(fftw_dft_generic_register),
	SOLVTAB(fftw_dft_rader_register),
	SOLVTAB(fftw_dft_bluestein_register),
	SOLVTAB(fftw_dft_nop_register),
	SOLVTAB(fftw_ct_generic_register),
	SOLVTAB(fftw_ct_genericbuf_register),
	SOLVTAB_END
};

void fftw_dft_conf_standard(planner *p) {
	fftw_solvtab_exec(dft_conf_s, p);
	fftw_solvtab_exec(fftw_solvtab_dft_standard, p);
#if HAVE_SSE2
	if (fftw_have_simd_sse2())
		fftw_solvtab_exec(fftw_solvtab_dft_sse2, p);
#endif
#if HAVE_AVX
	if (fftw_have_simd_avx())
		fftw_solvtab_exec(fftw_solvtab_dft_avx, p);
#endif
#if HAVE_AVX_128_FMA
	if (fftw_have_simd_avx_128_fma())
		fftw_solvtab_exec(fftw_solvtab_dft_avx_128_fma, p);
#endif
#if HAVE_AVX2
	if (fftw_have_simd_avx2())
		fftw_solvtab_exec(fftw_solvtab_dft_avx2, p);
	if (fftw_have_simd_avx2_128())
		fftw_solvtab_exec(fftw_solvtab_dft_avx2_128, p);
#endif
#if HAVE_AVX512
	if (fftw_have_simd_avx512())
		fftw_solvtab_exec(fftw_solvtab_dft_avx512, p);
#endif
#if HAVE_KCVI
	if (fftw_have_simd_kcvi())
		fftw_solvtab_exec(fftw_solvtab_dft_kcvi, p);
#endif
#if HAVE_ALTIVEC
	if (fftw_have_simd_altivec())
		fftw_solvtab_exec(fftw_solvtab_dft_altivec, p);
#endif
#if HAVE_VSX
	if (fftw_have_simd_vsx())
		fftw_solvtab_exec(fftw_solvtab_dft_vsx, p);
#endif
#if HAVE_NEON
	if (fftw_have_simd_neon())
		fftw_solvtab_exec(fftw_solvtab_dft_neon, p);
#endif
#if HAVE_GENERIC_SIMD128
	fftw_solvtab_exec(fftw_solvtab_dft_generic_simd128, p);
#endif
#if HAVE_GENERIC_SIMD256
	fftw_solvtab_exec(fftw_solvtab_dft_generic_simd256, p);
#endif
}


typedef struct {
	solver super;
} dft_bluestein_S;

typedef struct {
	plan_dft super;
	INT n;     /* problem size */
	INT nb;    /* size of convolution */
	FFTW_REAL_TYPE *w;      /* lambda k . exp(2*pi*i*k^2/(2*n)) */
	FFTW_REAL_TYPE *W;      /* DFT(w) */
	plan *cldf;
	INT is, os;
} dft_bluestein_P;

static void dft_bluestein_sequence(enum wakefulness wakefulness, INT n, FFTW_REAL_TYPE *w) {
	INT k, ksq, n2 = 2 * n;
	triggen *t = fftw_mktriggen(wakefulness, n2);

	ksq = 0;
	for (k = 0; k < n; ++k) {
		t->cexp(t, ksq, w + 2 * k);
		/* careful with overflow */
		ksq += 2 * k + 1;
		while (ksq > n2) ksq -= n2;
	}

	fftw_triggen_destroy(t);
}

static void dft_mktwiddle(enum wakefulness wakefulness, dft_bluestein_P *p) {
	INT i;
	INT n = p->n, nb = p->nb;
	FFTW_REAL_TYPE *w, *W;
	E nbf = (E)nb;

	p->w = w = (FFTW_REAL_TYPE *)MALLOC(2 * n * sizeof(FFTW_REAL_TYPE), TWIDDLES);
	p->W = W = (FFTW_REAL_TYPE *)MALLOC(2 * nb * sizeof(FFTW_REAL_TYPE), TWIDDLES);

	dft_bluestein_sequence(wakefulness, n, w);

	for (i = 0; i < nb; ++i)
		W[2 * i] = W[2 * i + 1] = K(0.0);

	W[0] = w[0] / nbf;
	W[1] = w[1] / nbf;

	for (i = 1; i < n; ++i) {
		W[2 * i] = W[2 * (nb - i)] = w[2 * i] / nbf;
		W[2 * i + 1] = W[2 * (nb - i) + 1] = w[2 * i + 1] / nbf;
	}

	{
		plan_dft *cldf = (plan_dft *)p->cldf;
		/* cldf must be awake */
		cldf->apply(p->cldf, W, W + 1, W, W + 1);
	}
}

static void
dft_bluestein_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const dft_bluestein_P *ego = (const dft_bluestein_P *)ego_;
	INT i, n = ego->n, nb = ego->nb, is = ego->is, os = ego->os;
	FFTW_REAL_TYPE *w = ego->w, *W = ego->W;
	FFTW_REAL_TYPE *b = (FFTW_REAL_TYPE *)MALLOC(2 * nb * sizeof(FFTW_REAL_TYPE), BUFFERS);

	/* multiply input by conjugate bluestein sequence */
	for (i = 0; i < n; ++i) {
		E xr = ri[i * is], xi = ii[i * is];
		E wr = w[2 * i], wi = w[2 * i + 1];
		b[2 * i] = xr * wr + xi * wi;
		b[2 * i + 1] = xi * wr - xr * wi;
	}

	for (; i < nb; ++i) b[2 * i] = b[2 * i + 1] = K(0.0);

	/* convolution: FFT */
	{
		plan_dft *cldf = (plan_dft *)ego->cldf;
		cldf->apply(ego->cldf, b, b + 1, b, b + 1);
	}

	/* convolution: pointwise multiplication */
	for (i = 0; i < nb; ++i) {
		E xr = b[2 * i], xi = b[2 * i + 1];
		E wr = W[2 * i], wi = W[2 * i + 1];
		b[2 * i] = xi * wr + xr * wi;
		b[2 * i + 1] = xr * wr - xi * wi;
	}

	/* convolution: IFFT by FFT with real/imag input/output swapped */
	{
		plan_dft *cldf = (plan_dft *)ego->cldf;
		cldf->apply(ego->cldf, b, b + 1, b, b + 1);
	}

	/* multiply output by conjugate bluestein sequence */
	for (i = 0; i < n; ++i) {
		E xi = b[2 * i], xr = b[2 * i + 1];
		E wr = w[2 * i], wi = w[2 * i + 1];
		ro[i * os] = xr * wr + xi * wi;
		io[i * os] = xi * wr - xr * wi;
	}

	fftw_ifree(b);
}

static void dft_bluestein_awake(plan *ego_, enum wakefulness wakefulness) {
	dft_bluestein_P *ego = (dft_bluestein_P *)ego_;

	fftw_plan_awake(ego->cldf, wakefulness);

	switch (wakefulness) {
	case SLEEPY:
		fftw_ifree0(ego->w);
		ego->w = 0;
		fftw_ifree0(ego->W);
		ego->W = 0;
		break;
	default:
		A(!ego->w);
		dft_mktwiddle(wakefulness, ego);
		break;
	}
}

static int dft_bluestein_applicable(const solver *ego, const problem *p_,
	const planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	UNUSED(ego);
	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		/* FIXME: allow other sizes */
		&& fftw_is_prime(p->sz->dims[0].n)

		/* FIXME: avoid infinite recursion of bluestein with itself.
		This works because all factors in child problems are 2, 3, 5 */
		&& p->sz->dims[0].n > 16

		&& CIMPLIES(NO_SLOWP(plnr), p->sz->dims[0].n > BLUESTEIN_MAX_SLOW)
		);
}

static void dft_bluestein_destroy(plan *ego_) {
	dft_bluestein_P *ego = (dft_bluestein_P *)ego_;
	fftw_plan_destroy_internal(ego->cldf);
}

static void dft_bluestein_print(const plan *ego_, printer *p) {
	const dft_bluestein_P *ego = (const dft_bluestein_P *)ego_;
	p->print(p, "(dft-bluestein-%D/%D%(%p%))",
		ego->n, ego->nb, ego->cldf);
}

static INT dft_bluestein_choose_transform_size(INT minsz) {
	while (!fftw_factors_into_small_primes(minsz))
		++minsz;
	return minsz;
}

static plan *dft_bluestein_mkplan(const solver *ego, const problem *p_, planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	dft_bluestein_P *pln;
	INT n, nb;
	plan *cldf = 0;
	FFTW_REAL_TYPE *buf;

	static const plan_adt dft_padt = {
		fftw_dft_solve, dft_bluestein_awake, dft_bluestein_print, dft_bluestein_destroy
	};

	if (!dft_bluestein_applicable(ego, p_, plnr))
		return (plan *)0;

	n = p->sz->dims[0].n;
	nb = dft_bluestein_choose_transform_size(2 * n - 1);
	buf = (FFTW_REAL_TYPE *)MALLOC(2 * nb * sizeof(FFTW_REAL_TYPE), BUFFERS);

	cldf = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_dft_d(fftw_mktensor_1d(nb, 2, 2),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf + 1,
			buf, buf + 1),
		NO_SLOW, 0, 0);
	if (!cldf) goto nada;

	fftw_ifree(buf);

	pln = MKPLAN_DFT(dft_bluestein_P, &dft_padt, dft_bluestein_apply);

	pln->n = n;
	pln->nb = nb;
	pln->w = 0;
	pln->W = 0;
	pln->cldf = cldf;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;

	fftw_ops_add(&cldf->ops, &cldf->ops, &pln->super.super.ops);
	pln->super.super.ops.add += 4 * n + 2 * nb;
	pln->super.super.ops.mul += 8 * n + 4 * nb;
	pln->super.super.ops.other += 6 * (n + nb);

	return &(pln->super.super);

nada:
	fftw_ifree0(buf);
	fftw_plan_destroy_internal(cldf);
	return (plan *)0;
}


static solver *dft_bluestein_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, dft_bluestein_mkplan, 0 };
	dft_bluestein_S *slv = MKSOLVER(dft_bluestein_S, &sadt);
	return &(slv->super);
}

void fftw_dft_bluestein_register(planner *p) {
	REGISTER_SOLVER(p, dft_bluestein_mksolver());
}

void fftw_kdft_register(planner *p, kdft codelet, const kdft_desc *desc) {
	REGISTER_SOLVER(p, fftw_mksolver_dft_direct(codelet, desc));
	REGISTER_SOLVER(p, fftw_mksolver_dft_directbuf(codelet, desc));
}

typedef struct {
	solver super;
	size_t maxnbuf_ndx;
} dft_buffered_S;

static const INT dft_maxnbufs[] = { 8, 256 };

typedef struct {
	plan_dft super;

	plan *cld, *cldcpy, *cldrest;
	INT n, vl, nbuf, bufdist;
	INT ivs_by_nbuf, ovs_by_nbuf;
	INT roffset, ioffset;
} dft_buffered_P;

/* transform a vector input with the help of bufs */
static void
dft_buffered_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const dft_buffered_P *ego = (const dft_buffered_P *)ego_;
	INT nbuf = ego->nbuf;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * ego->bufdist * 2, BUFFERS);

	plan_dft *cld = (plan_dft *)ego->cld;
	plan_dft *cldcpy = (plan_dft *)ego->cldcpy;
	plan_dft *cldrest;
	INT i, vl = ego->vl;
	INT ivs_by_nbuf = ego->ivs_by_nbuf, ovs_by_nbuf = ego->ovs_by_nbuf;
	INT roffset = ego->roffset, ioffset = ego->ioffset;

	for (i = nbuf; i <= vl; i += nbuf) {
		/* transform to bufs: */
		cld->apply((plan *)cld, ri, ii, bufs + roffset, bufs + ioffset);
		ri += ivs_by_nbuf;
		ii += ivs_by_nbuf;

		/* copy back */
		cldcpy->apply((plan *)cldcpy, bufs + roffset, bufs + ioffset, ro, io);
		ro += ovs_by_nbuf;
		io += ovs_by_nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_dft *)ego->cldrest;
	cldrest->apply((plan *)cldrest, ri, ii, ro, io);
}


static void dft_buffered_awake(plan *ego_, enum wakefulness wakefulness) {
	dft_buffered_P *ego = (dft_buffered_P *)ego_;

	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldcpy, wakefulness);
	fftw_plan_awake(ego->cldrest, wakefulness);
}

static void dft_buffered_destroy(plan *ego_) {
	dft_buffered_P *ego = (dft_buffered_P *)ego_;
	fftw_plan_destroy_internal(ego->cldrest);
	fftw_plan_destroy_internal(ego->cldcpy);
	fftw_plan_destroy_internal(ego->cld);
}

static void dft_buffered_print(const plan *ego_, printer *p) {
	const dft_buffered_P *ego = (const dft_buffered_P *)ego_;
	p->print(p, "(dft-buffered-%D%v/%D-%D%(%p%)%(%p%)%(%p%))",
		ego->n, ego->nbuf,
		ego->vl, ego->bufdist % ego->n,
		ego->cld, ego->cldcpy, ego->cldrest);
}

static int dft_buffered_applicable0(const dft_buffered_S *ego, const problem *p_, const planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	const iodim *d = p->sz->dims;

	if (1
		&& p->vecsz->rnk <= 1
		&& p->sz->rnk == 1
		) {
		INT vl, ivs, ovs;
		fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

		if (fftw_toobig(p->sz->dims[0].n) && CONSERVE_MEMORYP(plnr))
			return 0;

		/* if this solver is redundant, in the sense that a solver
		of lower index generates the same plan, then prune this
		solver */
		if (fftw_nbuf_redundant(d[0].n, vl,
			ego->maxnbuf_ndx,
			dft_maxnbufs, NELEM(dft_maxnbufs)))
			return 0;

		/*
		In principle, the buffered transforms might be useful
		when working out of place.  However, in order to
		prevent infinite loops in the planner, we require
		that the output stride of the buffered transforms be
		greater than 2.
		*/
		if (p->ri != p->ro)
			return (d[0].os > 2);

		/*
		* If the problem is in place, the input/output strides must
		* be the same or the whole thing must fit in the buffer.
		*/
		if (fftw_tensor_inplace_strides2(p->sz, p->vecsz))
			return 1;

		if (/* fits into buffer: */
			((p->vecsz->rnk == 0)
				||
				(fftw_nbuf(d[0].n, p->vecsz->dims[0].n,
					dft_maxnbufs[ego->maxnbuf_ndx])
					== p->vecsz->dims[0].n)))
			return 1;
	}

	return 0;
}

static int dft_buffered_applicable(const dft_buffered_S *ego, const problem *p_, const planner *plnr) {
	if (NO_BUFFERINGP(plnr)) return 0;
	if (!dft_buffered_applicable0(ego, p_, plnr)) return 0;

	if (NO_UGLYP(plnr)) {
		const problem_dft *p = (const problem_dft *)p_;
		if (p->ri != p->ro) return 0;
		if (fftw_toobig(p->sz->dims[0].n)) return 0;
	}
	return 1;
}

static plan *dft_buffered_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	dft_buffered_P *pln;
	const dft_buffered_S *ego = (const dft_buffered_S *)ego_;
	plan *cld = (plan *)0;
	plan *cldcpy = (plan *)0;
	plan *cldrest = (plan *)0;
	const problem_dft *p = (const problem_dft *)p_;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)0;
	INT nbuf = 0, bufdist, n, vl;
	INT ivs, ovs, roffset, ioffset;

	static const plan_adt padt = {
		fftw_dft_solve, dft_buffered_awake, dft_buffered_print, dft_buffered_destroy
	};

	if (!dft_buffered_applicable(ego, p_, plnr))
		goto nada;

	n = fftw_tensor_sz(p->sz);

	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

	nbuf = fftw_nbuf(n, vl, dft_maxnbufs[ego->maxnbuf_ndx]);
	bufdist = fftw_bufdist(n, vl);
	A(nbuf > 0);

	/* attempt to keep real and imaginary part in the same order,
	so as to allow optimizations in the the copy plan */
	roffset = (p->ri - p->ii > 0) ? (INT)1 : (INT)0;
	ioffset = 1 - roffset;

	/* initial allocation for the purpose of planning */
	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist * 2, BUFFERS);

	/* allow destruction of input if problem is in place */
	cld = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_mktensor_1d(n, p->sz->dims[0].is, 2),
			fftw_mktensor_1d(nbuf, ivs, bufdist * 2),
			TAINT(p->ri, ivs * nbuf),
			TAINT(p->ii, ivs * nbuf),
			bufs + roffset,
			bufs + ioffset),
		0, 0, (p->ri == p->ro) ? NO_DESTROY_INPUT : 0);
	if (!cld)
		goto nada;

	/* copying back from the buffer is a rank-0 transform: */
	cldcpy = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_mktensor_0d(),
			fftw_mktensor_2d(nbuf, bufdist * 2, ovs,
				n, 2, p->sz->dims[0].os),
			bufs + roffset,
			bufs + ioffset,
			TAINT(p->ro, ovs * nbuf),
			TAINT(p->io, ovs * nbuf)));
	if (!cldcpy)
		goto nada;

	/* deallocate buffers, let apply() allocate them for real */
	fftw_ifree(bufs);
	bufs = 0;

	/* plan the leftover transforms (cldrest): */
	{
		INT id = ivs * (nbuf * (vl / nbuf));
		INT od = ovs * (nbuf * (vl / nbuf));
		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_dft_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->ri + id, p->ii + id, p->ro + od, p->io + od));
	}
	if (!cldrest)
		goto nada;

	pln = MKPLAN_DFT(dft_buffered_P, &padt, dft_buffered_apply);
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->cldrest = cldrest;
	pln->n = n;
	pln->vl = vl;
	pln->ivs_by_nbuf = ivs * nbuf;
	pln->ovs_by_nbuf = ovs * nbuf;
	pln->roffset = roffset;
	pln->ioffset = ioffset;

	pln->nbuf = nbuf;
	pln->bufdist = bufdist;

	{
		opcnt t;
		fftw_ops_add(&cld->ops, &cldcpy->ops, &t);
		fftw_ops_madd(vl / nbuf, &t, &cldrest->ops, &pln->super.super.ops);
	}

	return &(pln->super.super);

nada:
	fftw_ifree0(bufs);
	fftw_plan_destroy_internal(cldrest);
	fftw_plan_destroy_internal(cldcpy);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static solver *dft_buffered_mksolver(size_t maxnbuf_ndx) {
	static const solver_adt sadt = { PROBLEM_DFT, dft_buffered_mkplan, 0 };
	dft_buffered_S *slv = MKSOLVER(dft_buffered_S, &sadt);
	slv->maxnbuf_ndx = maxnbuf_ndx;
	return &(slv->super);
}

void fftw_dft_buffered_register(planner *p) {
	size_t i;
	for (i = 0; i < NELEM(dft_maxnbufs); ++i)
		REGISTER_SOLVER(p, dft_buffered_mksolver(i));
}


ct_solver *(*fftw_mksolver_ct_hook)(size_t, INT, int,
	ct_mkinferior, ct_force_vrecursion) = 0;

typedef struct {
	plan_dft super;
	plan *cld;
	plan *cldw;
	INT r;
} ct_P;

static void
ct_apply_dit(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const ct_P *ego = (const ct_P *)ego_;
	plan_dft *cld;
	plan_dftw *cldw;

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, ri, ii, ro, io);

	cldw = (plan_dftw *)ego->cldw;
	cldw->apply(ego->cldw, ro, io);
}

static void
ct_apply_dif(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const ct_P *ego = (const ct_P *)ego_;
	plan_dft *cld;
	plan_dftw *cldw;

	cldw = (plan_dftw *)ego->cldw;
	cldw->apply(ego->cldw, ri, ii);

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, ri, ii, ro, io);
}

static void ct_awake(plan *ego_, enum wakefulness wakefulness) {
	ct_P *ego = (ct_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldw, wakefulness);
}

static void ct_destroy(plan *ego_) {
	ct_P *ego = (ct_P *)ego_;
	fftw_plan_destroy_internal(ego->cldw);
	fftw_plan_destroy_internal(ego->cld);
}

static void ct_print(const plan *ego_, printer *p) {
	const ct_P *ego = (const ct_P *)ego_;
	p->print(p, "(dft-ct-%s/%D%(%p%)%(%p%))",
		ego->super.apply == ct_apply_dit ? "dit" : "dif",
		ego->r, ego->cldw, ego->cld);
}

static int ct_applicable0(const ct_solver *ego, const problem *p_, planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	INT r;

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1

		/* DIF destroys the input and we don't like it */
		&& (ego->dec == DECDIT ||
			p->ri == p->ro ||
			!NO_DESTROY_INPUTP(plnr))

		&& ((r = fftw_choose_radix(ego->r, p->sz->dims[0].n)) > 1)
		&& p->sz->dims[0].n > r);
}


int fftw_ct_applicable(const ct_solver *ego, const problem *p_, planner *plnr) {
	const problem_dft *p;

	if (!ct_applicable0(ego, p_, plnr))
		return 0;

	p = (const problem_dft *)p_;

	return (0
		|| ego->dec == DECDIF + TRANSPOSE
		|| p->vecsz->rnk == 0
		|| !NO_VRECURSEP(plnr)
		|| (ego->force_vrecursionp && ego->force_vrecursionp(ego, p))
		);
}


static plan *ct_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const ct_solver *ego = (const ct_solver *)ego_;
	const problem_dft *p;
	ct_P *pln = 0;
	plan *cld = 0, *cldw = 0;
	INT n, r, m, v, ivs, ovs;
	iodim *d;

	static const plan_adt padt = {
		fftw_dft_solve, ct_awake, ct_print, ct_destroy
	};

	if ((NO_NONTHREADEDP(plnr)) || !fftw_ct_applicable(ego, p_, plnr))
		return (plan *)0;

	p = (const problem_dft *)p_;
	d = p->sz->dims;
	n = d[0].n;
	r = fftw_choose_radix(ego->r, n);
	m = n / r;

	fftw_tensor_tornk1(p->vecsz, &v, &ivs, &ovs);

	switch (ego->dec) {
	case DECDIT: {
		cldw = ego->mkcldw(ego,
			r, m * d[0].os, m * d[0].os,
			m, d[0].os,
			v, ovs, ovs,
			0, m,
			p->ro, p->io, plnr);
		if (!cldw) goto nada;

		cld = fftw_mkplan_d(plnr,
			fftw_mkproblem_dft_d(
				fftw_mktensor_1d(m, r * d[0].is, d[0].os),
				fftw_mktensor_2d(r, d[0].is, m * d[0].os,
					v, ivs, ovs),
				p->ri, p->ii, p->ro, p->io)
		);
		if (!cld) goto nada;

		pln = MKPLAN_DFT(ct_P, &padt, ct_apply_dit);
		break;
	}
	case DECDIF:
	case DECDIF + TRANSPOSE: {
		INT cors, covs; /* cldw ors, ovs */
		if (ego->dec == DECDIF + TRANSPOSE) {
			cors = ivs;
			covs = m * d[0].is;
			/* ensure that we generate well-formed dftw subproblems */
			/* FIXME: too conservative */
			if (!(1
				&& r == v
				&& d[0].is == r * cors))
				goto nada;

			/* FIXME: allow in-place only for now, like in
			fftw-3.[01] */
			if (!(1
				&& p->ri == p->ro
				&& d[0].is == r * d[0].os
				&& cors == d[0].os
				&& covs == ovs
				))
				goto nada;
		}
		else {
			cors = m * d[0].is;
			covs = ivs;
		}

		cldw = ego->mkcldw(ego,
			r, m * d[0].is, cors,
			m, d[0].is,
			v, ivs, covs,
			0, m,
			p->ri, p->ii, plnr);
		if (!cldw) goto nada;

		cld = fftw_mkplan_d(plnr,
			fftw_mkproblem_dft_d(
				fftw_mktensor_1d(m, d[0].is, r * d[0].os),
				fftw_mktensor_2d(r, cors, d[0].os,
					v, covs, ovs),
				p->ri, p->ii, p->ro, p->io)
		);
		if (!cld) goto nada;

		pln = MKPLAN_DFT(ct_P, &padt, ct_apply_dif);
		break;
	}

	default:
		A(0);

	}

	pln->cld = cld;
	pln->cldw = cldw;
	pln->r = r;
	fftw_ops_add(&cld->ops, &cldw->ops, &pln->super.super.ops);

	/* inherit could_prune_now_p attribute from cldw */
	pln->super.super.could_prune_now_p = cldw->could_prune_now_p;
	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cldw);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

ct_solver *fftw_mksolver_ct(size_t size, INT r, int dec,
	ct_mkinferior mkcldw,
	ct_force_vrecursion force_vrecursionp) {
	static const solver_adt sadt = { PROBLEM_DFT, ct_mkplan, 0 };
	ct_solver *slv = (ct_solver *)fftw_mksolver(size, &sadt);
	slv->r = r;
	slv->dec = dec;
	slv->mkcldw = mkcldw;
	slv->force_vrecursionp = force_vrecursionp;
	return slv;
}

plan *fftw_mkplan_dftw(size_t size, const plan_adt *adt, dftwapply apply) {
	plan_dftw *ego;

	ego = (plan_dftw *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}

typedef struct {
	ct_solver super;
	const ct_desc *desc;
	int bufferedp;
	kdftw k;
} dftw_direct_S;

typedef struct {
	plan_dftw super;
	kdftw k;
	INT r;
	stride rs;
	INT m, ms, v, vs, mb, me, extra_iter;
	stride brs;
	twid *td;
	const dftw_direct_S *slv;
} dftw_direct_P;


/*************************************************************
Nonbuffered code
*************************************************************/
static void dftw_direct_apply(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_direct_P *ego = (const dftw_direct_P *)ego_;
	INT i;
	ASSERT_ALIGNED_DOUBLE;
	for (i = 0; i < ego->v; ++i, rio += ego->vs, iio += ego->vs) {
		INT mb = ego->mb, ms = ego->ms;
		ego->k(rio + mb * ms, iio + mb * ms, ego->td->W,
			ego->rs, mb, ego->me, ms);
	}
}

static void dftw_direct_apply_extra_iter(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_direct_P *ego = (const dftw_direct_P *)ego_;
	INT i, v = ego->v, vs = ego->vs;
	INT mb = ego->mb, me = ego->me, mm = me - 1, ms = ego->ms;
	ASSERT_ALIGNED_DOUBLE;
	for (i = 0; i < v; ++i, rio += vs, iio += vs) {
		ego->k(rio + mb * ms, iio + mb * ms, ego->td->W,
			ego->rs, mb, mm, ms);
		ego->k(rio + mm * ms, iio + mm * ms, ego->td->W,
			ego->rs, mm, mm + 2, 0);
	}
}

/*************************************************************
Buffered code
*************************************************************/
static void dftw_direct_dobatch(const dftw_direct_P *ego, FFTW_REAL_TYPE *rA, FFTW_REAL_TYPE *iA, INT mb, INT me,
	FFTW_REAL_TYPE *buf) {
	INT brs = WS(ego->brs, 1);
	INT rs = WS(ego->rs, 1);
	INT ms = ego->ms;

	fftw_cpy2d_pair_ci(rA + mb * ms, iA + mb * ms, buf, buf + 1,
		ego->r, rs, brs,
		me - mb, ms, 2);
	ego->k(buf, buf + 1, ego->td->W, ego->brs, mb, me, 2);
	fftw_cpy2d_pair_co(buf, buf + 1, rA + mb * ms, iA + mb * ms,
		ego->r, brs, rs,
		me - mb, 2, ms);
}

/* must be even for SIMD alignment; should not be 2^k to avoid
associativity conflicts */
static INT dftw_direct_compute_batchsize(INT radix) {
	/* round up to multiple of 4 */
	radix += 3;
	radix &= -4;

	return (radix + 2);
}

static void dftw_direct_apply_buf(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_direct_P *ego = (const dftw_direct_P *)ego_;
	INT i, j, v = ego->v, r = ego->r;
	INT batchsz = dftw_direct_compute_batchsize(r);
	FFTW_REAL_TYPE *buf;
	INT mb = ego->mb, me = ego->me;
	size_t bufsz = r * batchsz * 2 * sizeof(FFTW_REAL_TYPE);

	BUF_ALLOC(FFTW_REAL_TYPE *, buf, bufsz);

	for (i = 0; i < v; ++i, rio += ego->vs, iio += ego->vs) {
		for (j = mb; j + batchsz < me; j += batchsz)
			dftw_direct_dobatch(ego, rio, iio, j, j + batchsz, buf);

		dftw_direct_dobatch(ego, rio, iio, j, me, buf);
	}

	BUF_FREE(buf, bufsz);
}

/*************************************************************
common code
*************************************************************/
static void dftw_direct_awake(plan *ego_, enum wakefulness wakefulness) {
	dftw_direct_P *ego = (dftw_direct_P *)ego_;

	fftw_twiddle_awake(wakefulness, &ego->td, ego->slv->desc->tw,
		ego->r * ego->m, ego->r, ego->m + ego->extra_iter);
}

static void dftw_direct_destroy(plan *ego_) {
	dftw_direct_P *ego = (dftw_direct_P *)ego_;
	fftw_stride_destroy(ego->brs);
	fftw_stride_destroy(ego->rs);
}

static void dftw_direct_print(const plan *ego_, printer *p) {
	const dftw_direct_P *ego = (const dftw_direct_P *)ego_;
	const dftw_direct_S *slv = ego->slv;
	const ct_desc *e = slv->desc;

	if (slv->bufferedp)
		p->print(p, "(dftw-directbuf/%D-%D/%D%v \"%s\")",
			dftw_direct_compute_batchsize(ego->r), ego->r,
			fftw_twiddle_length(ego->r, e->tw), ego->v, e->nam);
	else
		p->print(p, "(dftw-direct-%D/%D%v \"%s\")",
			ego->r, fftw_twiddle_length(ego->r, e->tw), ego->v, e->nam);
}

static int dftw_direct_applicable0(const dftw_direct_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mb, INT me,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	const planner *plnr, INT *extra_iter) {
	const ct_desc *e = ego->desc;
	UNUSED(v);

	return (
		1
		&& r == e->radix
		&& irs == ors /* in-place along R */
		&& ivs == ovs /* in-place along V */

					  /* check for alignment/vector length restrictions */
		&& ((*extra_iter = 0,
			e->genus->okp(e, rio, iio, irs, ivs, m, mb, me, ms, plnr))
			||
			(*extra_iter = 1,
			(1
				/* FIXME: require full array, otherwise some threads
				may be extra_iter and other threads won't be.
				Generating the proper twiddle factors is a pain in
				this case */
				&& mb == 0 && me == m
				&& e->genus->okp(e, rio, iio, irs, ivs,
					m, mb, me - 1, ms, plnr)
				&& e->genus->okp(e, rio, iio, irs, ivs,
					m, me - 1, me + 1, ms, plnr))))

		&& (e->genus->okp(e, rio + ivs, iio + ivs, irs, ivs,
			m, mb, me - *extra_iter, ms, plnr))

		);
}

static int dftw_direct_applicable0_buf(const dftw_direct_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mb, INT me,
	const FFTW_REAL_TYPE *rio, const FFTW_REAL_TYPE *iio,
	const planner *plnr) {
	const ct_desc *e = ego->desc;
	INT batchsz;
	UNUSED(v);
	UNUSED(ms);
	UNUSED(rio);
	UNUSED(iio);

	return (
		1
		&& r == e->radix
		&& irs == ors /* in-place along R */
		&& ivs == ovs /* in-place along V */

					  /* check for alignment/vector length restrictions, both for
					  batchsize and for the remainder */
		&& (batchsz = dftw_direct_compute_batchsize(r), 1)
		&& (e->genus->okp(e, 0, ((const FFTW_REAL_TYPE *)0) + 1, 2 * batchsz, 0,
			m, mb, mb + batchsz, 2, plnr))
		&& (e->genus->okp(e, 0, ((const FFTW_REAL_TYPE *)0) + 1, 2 * batchsz, 0,
			m, mb, me, 2, plnr))
		);
}

static int dftw_direct_applicable(const dftw_direct_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mb, INT me,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	const planner *plnr, INT *extra_iter) {
	if (ego->bufferedp) {
		*extra_iter = 0;
		if (!dftw_direct_applicable0_buf(ego,
			r, irs, ors, m, ms, v, ivs, ovs, mb, me,
			rio, iio, plnr))
			return 0;
	}
	else {
		if (!dftw_direct_applicable0(ego,
			r, irs, ors, m, ms, v, ivs, ovs, mb, me,
			rio, iio, plnr, extra_iter))
			return 0;
	}

	if (NO_UGLYP(plnr) && fftw_ct_uglyp((ego->bufferedp ? (INT)512 : (INT)16),
		v, m * r, r))
		return 0;

	if (m * r > 262144 && NO_FIXED_RADIX_LARGE_NP(plnr))
		return 0;

	return 1;
}

static plan *dftw_direct_mkcldw(const ct_solver *ego_,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	planner *plnr) {
	const dftw_direct_S *ego = (const dftw_direct_S *)ego_;
	dftw_direct_P *pln;
	const ct_desc *e = ego->desc;
	INT extra_iter;

	static const plan_adt padt = {
		0, dftw_direct_awake, dftw_direct_print, dftw_direct_destroy
	};

	A(mstart >= 0 && mstart + mcount <= m);
	if (!dftw_direct_applicable(ego,
		r, irs, ors, m, ms, v, ivs, ovs, mstart, mstart + mcount,
		rio, iio, plnr, &extra_iter))
		return (plan *)0;

	if (ego->bufferedp) {
		pln = MKPLAN_DFTW(dftw_direct_P, &padt, dftw_direct_apply_buf);
	}
	else {
		pln = MKPLAN_DFTW(dftw_direct_P, &padt, extra_iter ? dftw_direct_apply_extra_iter : dftw_direct_apply);
	}

	pln->k = ego->k;
	pln->rs = fftw_mkstride(r, irs);
	pln->td = 0;
	pln->r = r;
	pln->m = m;
	pln->ms = ms;
	pln->v = v;
	pln->vs = ivs;
	pln->mb = mstart;
	pln->me = mstart + mcount;
	pln->slv = ego;
	pln->brs = fftw_mkstride(r, 2 * dftw_direct_compute_batchsize(r));
	pln->extra_iter = extra_iter;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(v * (mcount / e->genus->vl), &e->ops, &pln->super.super.ops);

	if (ego->bufferedp) {
		/* 8 load/stores * N * V */
		pln->super.super.ops.other += 8 * r * mcount * v;
	}

	pln->super.super.could_prune_now_p =
		(!ego->bufferedp && r >= 5 && r < 64 && m >= r);
	return &(pln->super.super);
}

static void dftw_direct_regone(planner *plnr, kdftw codelet,
	const ct_desc *desc, int dec, int bufferedp) {
	dftw_direct_S *slv = (dftw_direct_S *)fftw_mksolver_ct(sizeof(dftw_direct_S), desc->radix, dec,
		dftw_direct_mkcldw, 0);
	slv->k = codelet;
	slv->desc = desc;
	slv->bufferedp = bufferedp;
	REGISTER_SOLVER(plnr, &(slv->super.super));
	if (fftw_mksolver_ct_hook) {
		slv = (dftw_direct_S *)fftw_mksolver_ct_hook(sizeof(dftw_direct_S), desc->radix,
			dec, dftw_direct_mkcldw, 0);
		slv->k = codelet;
		slv->desc = desc;
		slv->bufferedp = bufferedp;
		REGISTER_SOLVER(plnr, &(slv->super.super));
	}
}

void fftw_regsolver_ct_directw(planner *plnr, kdftw codelet,
	const ct_desc *desc, int dec) {
	dftw_direct_regone(plnr, codelet, desc, dec, /* bufferedp */ 0);
	dftw_direct_regone(plnr, codelet, desc, dec, /* bufferedp */ 1);
}

typedef struct {
	ct_solver super;
	const ct_desc *desc;
	kdftwsq k;
} dftw_directsq_S;

typedef struct {
	plan_dftw super;
	kdftwsq k;
	INT r;
	stride rs, vs;
	INT m, ms, v, mb, me;
	twid *td;
	const dftw_directsq_S *slv;
} dftw_directsq_P;


static void dftw_directsq_apply(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_directsq_P *ego = (const dftw_directsq_P *)ego_;
	INT mb = ego->mb, ms = ego->ms;
	ego->k(rio + mb * ms, iio + mb * ms, ego->td->W, ego->rs, ego->vs,
		mb, ego->me, ms);
}

static void dftw_directsq_awake(plan *ego_, enum wakefulness wakefulness) {
	dftw_directsq_P *ego = (dftw_directsq_P *)ego_;

	fftw_twiddle_awake(wakefulness, &ego->td, ego->slv->desc->tw,
		ego->r * ego->m, ego->r, ego->m);
}

static void dftw_directsq_destroy(plan *ego_) {
	dftw_directsq_P *ego = (dftw_directsq_P *)ego_;
	fftw_stride_destroy(ego->rs);
	fftw_stride_destroy(ego->vs);
}

static void dftw_directsq_print(const plan *ego_, printer *p) {
	const dftw_directsq_P *ego = (const dftw_directsq_P *)ego_;
	const dftw_directsq_S *slv = ego->slv;
	const ct_desc *e = slv->desc;

	p->print(p, "(dftw-directsq-%D/%D%v \"%s\")",
		ego->r, fftw_twiddle_length(ego->r, e->tw), ego->v, e->nam);
}

static int dftw_directsq_applicable(const dftw_directsq_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mb, INT me,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	const planner *plnr) {
	const ct_desc *e = ego->desc;
	UNUSED(v);

	return (
		1
		&& r == e->radix

		/* transpose r, v */
		&& r == v
		&& irs == ovs
		&& ivs == ors

		/* check for alignment/vector length restrictions */
		&& e->genus->okp(e, rio, iio, irs, ivs, m, mb, me, ms, plnr)

		);
}

static plan *dftw_directsq_mkcldw(const ct_solver *ego_,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	planner *plnr) {
	const dftw_directsq_S *ego = (const dftw_directsq_S *)ego_;
	dftw_directsq_P *pln;
	const ct_desc *e = ego->desc;

	static const plan_adt padt = {
		0, dftw_directsq_awake, dftw_directsq_print, dftw_directsq_destroy
	};

	A(mstart >= 0 && mstart + mcount <= m);
	if (!dftw_directsq_applicable(ego,
		r, irs, ors, m, ms, v, ivs, ovs, mstart, mstart + mcount,
		rio, iio, plnr))
		return (plan *)0;

	pln = MKPLAN_DFTW(dftw_directsq_P, &padt, dftw_directsq_apply);

	pln->k = ego->k;
	pln->rs = fftw_mkstride(r, irs);
	pln->vs = fftw_mkstride(v, ivs);
	pln->td = 0;
	pln->r = r;
	pln->m = m;
	pln->ms = ms;
	pln->v = v;
	pln->mb = mstart;
	pln->me = mstart + mcount;
	pln->slv = ego;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(mcount / e->genus->vl, &e->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

static void dftw_directsq_regone(planner *plnr, kdftwsq codelet,
	const ct_desc *desc, int dec) {
	dftw_directsq_S *slv = (dftw_directsq_S *)fftw_mksolver_ct(sizeof(dftw_directsq_S), desc->radix, dec,
		dftw_directsq_mkcldw, 0);
	slv->k = codelet;
	slv->desc = desc;
	REGISTER_SOLVER(plnr, &(slv->super.super));
	if (fftw_mksolver_ct_hook) {
		slv = (dftw_directsq_S *)fftw_mksolver_ct_hook(sizeof(dftw_directsq_S), desc->radix, dec,
			dftw_directsq_mkcldw, 0);
		slv->k = codelet;
		slv->desc = desc;
		REGISTER_SOLVER(plnr, &(slv->super.super));
	}
}

void fftw_regsolver_ct_directwsq(planner *plnr, kdftwsq codelet,
	const ct_desc *desc, int dec) {
	dftw_directsq_regone(plnr, codelet, desc, dec + TRANSPOSE);
}


/* express a twiddle problem in terms of dft + multiplication by
twiddle factors */


typedef ct_solver dftw_generic_S;

typedef struct {
	plan_dftw super;

	INT r, rs, m, mb, me, ms, v, vs;

	plan *cld;

	twid *td;

	const dftw_generic_S *slv;
	int dec;
} dftw_generic_P;

static void dftw_generic_mktwiddle(dftw_generic_P *ego, enum wakefulness wakefulness) {
	static const tw_instr tw[] = { { TW_FULL, 0, 0 },
	{ TW_NEXT, 1, 0 } };

	/* note that R and M are swapped, to allow for sequential
	access both to data and twiddles */
	fftw_twiddle_awake(wakefulness, &ego->td, tw,
		ego->r * ego->m, ego->m, ego->r);
}

static void dftw_generic_bytwiddle(const dftw_generic_P *ego, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	INT iv, ir, im;
	INT r = ego->r, rs = ego->rs;
	INT m = ego->m, mb = ego->mb, me = ego->me, ms = ego->ms;
	INT v = ego->v, vs = ego->vs;
	const FFTW_REAL_TYPE *W = ego->td->W;

	mb += (mb == 0); /* skip m=0 iteration */
	for (iv = 0; iv < v; ++iv) {
		for (ir = 1; ir < r; ++ir) {
			for (im = mb; im < me; ++im) {
				FFTW_REAL_TYPE *pr = rio + ms * im + rs * ir;
				FFTW_REAL_TYPE *pi = iio + ms * im + rs * ir;
				E xr = *pr;
				E xi = *pi;
				E wr = W[2 * im + (2 * (m - 1)) * ir - 2];
				E wi = W[2 * im + (2 * (m - 1)) * ir - 1];
				*pr = xr * wr + xi * wi;
				*pi = xi * wr - xr * wi;
			}
		}
		rio += vs;
		iio += vs;
	}
}

static int dftw_generic_applicable(INT irs, INT ors, INT ivs, INT ovs,
	const planner *plnr) {
	return (1
		&& irs == ors
		&& ivs == ovs
		&& !NO_SLOWP(plnr)
		);
}

static void dftw_generic_apply_dit(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_generic_P *ego = (const dftw_generic_P *)ego_;
	plan_dft *cld;
	INT dm = ego->ms * ego->mb;

	dftw_generic_bytwiddle(ego, rio, iio);

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, rio + dm, iio + dm, rio + dm, iio + dm);
}

static void dftw_generic_apply_dif(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_generic_P *ego = (const dftw_generic_P *)ego_;
	plan_dft *cld;
	INT dm = ego->ms * ego->mb;

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, rio + dm, iio + dm, rio + dm, iio + dm);

	dftw_generic_bytwiddle(ego, rio, iio);
}

static void dftw_generic_awake(plan *ego_, enum wakefulness wakefulness) {
	dftw_generic_P *ego = (dftw_generic_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	dftw_generic_mktwiddle(ego, wakefulness);
}

static void dftw_generic_destroy(plan *ego_) {
	dftw_generic_P *ego = (dftw_generic_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void dftw_generic_print(const plan *ego_, printer *p) {
	const dftw_generic_P *ego = (const dftw_generic_P *)ego_;
	p->print(p, "(dftw-generic-%s-%D-%D%v%(%p%))",
		ego->dec == DECDIT ? "dit" : "dif",
		ego->r, ego->m, ego->v, ego->cld);
}

static plan *dftw_generic_mkcldw(const ct_solver *ego_,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	planner *plnr) {
	const dftw_generic_S *ego = ego_;
	dftw_generic_P *pln;
	plan *cld = 0;
	INT dm = ms * mstart;

	static const plan_adt padt = {
		0, dftw_generic_awake, dftw_generic_print, dftw_generic_destroy
	};

	A(mstart >= 0 && mstart + mcount <= m);
	if (!dftw_generic_applicable(irs, ors, ivs, ovs, plnr))
		return (plan *)0;

	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_mktensor_1d(r, irs, irs),
			fftw_mktensor_2d(mcount, ms, ms, v, ivs, ivs),
			rio + dm, iio + dm, rio + dm, iio + dm)
	);
	if (!cld) goto nada;

	pln = MKPLAN_DFTW(dftw_generic_P, &padt, ego->dec == DECDIT ? dftw_generic_apply_dit : dftw_generic_apply_dif);
	pln->slv = ego;
	pln->cld = cld;
	pln->r = r;
	pln->rs = irs;
	pln->m = m;
	pln->ms = ms;
	pln->v = v;
	pln->vs = ivs;
	pln->mb = mstart;
	pln->me = mstart + mcount;
	pln->dec = ego->dec;
	pln->td = 0;

	{
		double n0 = (r - 1) * (mcount - 1) * v;
		pln->super.super.ops = cld->ops;
		pln->super.super.ops.mul += 8 * n0;
		pln->super.super.ops.add += 4 * n0;
		pln->super.super.ops.other += 8 * n0;
	}
	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static void dftw_generic_regsolver(planner *plnr, INT r, int dec) {
	dftw_generic_S *slv = fftw_mksolver_ct(sizeof(dftw_generic_S), r, dec, dftw_generic_mkcldw,
		0);
	REGISTER_SOLVER(plnr, &(slv->super));
	if (fftw_mksolver_ct_hook) {
		slv = fftw_mksolver_ct_hook(sizeof(dftw_generic_S), r, dec, dftw_generic_mkcldw, 0);
		REGISTER_SOLVER(plnr, &(slv->super));
	}
}

void fftw_ct_generic_register(planner *p) {
	dftw_generic_regsolver(p, 0, DECDIT);
	dftw_generic_regsolver(p, 0, DECDIF);
}

typedef struct {
	ct_solver super;
	INT batchsz;
} dftw_genericbuf_S;

typedef struct {
	plan_dftw super;

	INT r, rs, m, ms, v, vs, mb, me;
	INT batchsz;
	plan *cld;

	triggen *t;
	const dftw_genericbuf_S *slv;
} dftw_genericbuf_P;


#define BATCHDIST(r) ((r) + 16)

/**************************************************************/
static void
dftw_genericbuf_bytwiddle(const dftw_genericbuf_P *ego, INT mb, INT me, FFTW_REAL_TYPE *buf, FFTW_REAL_TYPE *rio,
	FFTW_REAL_TYPE *iio) {
	INT j, k;
	INT r = ego->r, rs = ego->rs, ms = ego->ms;
	triggen *t = ego->t;
	for (j = 0; j < r; ++j) {
		for (k = mb; k < me; ++k)
			t->rotate(t, j * k,
				rio[j * rs + k * ms],
				iio[j * rs + k * ms],
				&buf[j * 2 + 2 * BATCHDIST(r) * (k - mb) + 0]);
	}
}

static int dftw_genericbuf_applicable0(const dftw_genericbuf_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT v,
	INT mcount) {
	return (1
		&& v == 1
		&& irs == ors
		&& mcount >= ego->batchsz
		&& mcount % ego->batchsz == 0
		&& r >= 64
		&& m >= r
		);
}

static int dftw_genericbuf_applicable(const dftw_genericbuf_S *ego,
	INT r, INT irs, INT ors,
	INT m, INT v,
	INT mcount,
	const planner *plnr) {
	if (!dftw_genericbuf_applicable0(ego, r, irs, ors, m, v, mcount))
		return 0;
	if (NO_UGLYP(plnr) && m * r < 65536)
		return 0;

	return 1;
}

static void
dftw_genericbuf_dobatch(const dftw_genericbuf_P *ego, INT mb, INT me, FFTW_REAL_TYPE *buf, FFTW_REAL_TYPE *rio,
	FFTW_REAL_TYPE *iio) {
	plan_dft *cld;
	INT ms = ego->ms;

	dftw_genericbuf_bytwiddle(ego, mb, me, buf, rio, iio);

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, buf, buf + 1, buf, buf + 1);
	fftw_cpy2d_pair_co(buf, buf + 1,
		rio + ms * mb, iio + ms * mb,
		me - mb, 2 * BATCHDIST(ego->r), ms,
		ego->r, 2, ego->rs);
}

static void dftw_genericbuf_apply(const plan *ego_, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio) {
	const dftw_genericbuf_P *ego = (const dftw_genericbuf_P *)ego_;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * 2 * BATCHDIST(ego->r) * ego->batchsz,
		BUFFERS);
	INT m;

	for (m = ego->mb; m < ego->me; m += ego->batchsz)
		dftw_genericbuf_dobatch(ego, m, m + ego->batchsz, buf, rio, iio);

	A(m == ego->me);

	fftw_ifree(buf);
}

static void dftw_genericbuf_awake(plan *ego_, enum wakefulness wakefulness) {
	dftw_genericbuf_P *ego = (dftw_genericbuf_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);

	switch (wakefulness) {
	case SLEEPY:
		fftw_triggen_destroy(ego->t);
		ego->t = 0;
		break;
	default:
		ego->t = fftw_mktriggen(AWAKE_SQRTN_TABLE, ego->r * ego->m);
		break;
	}
}

static void dftw_genericbuf_destroy(plan *ego_) {
	dftw_genericbuf_P *ego = (dftw_genericbuf_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void dftw_genericbuf_print(const plan *ego_, printer *p) {
	const dftw_genericbuf_P *ego = (const dftw_genericbuf_P *)ego_;
	p->print(p, "(dftw-genericbuf/%D-%D-%D%(%p%))",
		ego->batchsz, ego->r, ego->m, ego->cld);
}

static plan *dftw_genericbuf_mkcldw(const ct_solver *ego_,
	INT r, INT irs, INT ors,
	INT m, INT ms,
	INT v, INT ivs, INT ovs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio,
	planner *plnr) {
	const dftw_genericbuf_S *ego = (const dftw_genericbuf_S *)ego_;
	dftw_genericbuf_P *pln;
	plan *cld = 0;
	FFTW_REAL_TYPE *buf;

	static const plan_adt padt = {
		0, dftw_genericbuf_awake, dftw_genericbuf_print, dftw_genericbuf_destroy
	};

	UNUSED(ivs);
	UNUSED(ovs);
	UNUSED(rio);
	UNUSED(iio);

	A(mstart >= 0 && mstart + mcount <= m);
	if (!dftw_genericbuf_applicable(ego, r, irs, ors, m, v, mcount, plnr))
		return (plan *)0;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * 2 * BATCHDIST(r) * ego->batchsz, BUFFERS);
	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_mktensor_1d(r, 2, 2),
			fftw_mktensor_1d(ego->batchsz,
				2 * BATCHDIST(r),
				2 * BATCHDIST(r)),
			buf, buf + 1, buf, buf + 1
		)
	);
	fftw_ifree(buf);
	if (!cld) goto nada;

	pln = MKPLAN_DFTW(dftw_genericbuf_P, &padt, dftw_genericbuf_apply);
	pln->slv = ego;
	pln->cld = cld;
	pln->r = r;
	pln->m = m;
	pln->ms = ms;
	pln->rs = irs;
	pln->batchsz = ego->batchsz;
	pln->mb = mstart;
	pln->me = mstart + mcount;

	{
		double n0 = (r - 1) * (mcount - 1);
		pln->super.super.ops = cld->ops;
		pln->super.super.ops.mul += 8 * n0;
		pln->super.super.ops.add += 4 * n0;
		pln->super.super.ops.other += 8 * n0;
	}
	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static void dftw_genericbuf_regsolver(planner *plnr, INT r, INT batchsz) {
	dftw_genericbuf_S *slv = (dftw_genericbuf_S *)fftw_mksolver_ct(sizeof(dftw_genericbuf_S), r, DECDIT,
		dftw_genericbuf_mkcldw, 0);
	slv->batchsz = batchsz;
	REGISTER_SOLVER(plnr, &(slv->super.super));

	if (fftw_mksolver_ct_hook) {
		slv = (dftw_genericbuf_S *)fftw_mksolver_ct_hook(sizeof(dftw_genericbuf_S), r, DECDIT,
			dftw_genericbuf_mkcldw, 0);
		slv->batchsz = batchsz;
		REGISTER_SOLVER(plnr, &(slv->super.super));
	}

}

void fftw_ct_genericbuf_register(planner *p) {
	static const INT radices[] = { -1, -2, -4, -8, -16, -32, -64 };
	static const INT batchsizes[] = { 4, 8, 16, 32, 64 };
	unsigned i, j;

	for (i = 0; i < sizeof(radices) / sizeof(radices[0]); ++i)
		for (j = 0; j < sizeof(batchsizes) / sizeof(batchsizes[0]); ++j)
			dftw_genericbuf_regsolver(p, radices[i], batchsizes[j]);
}

/* direct DFT solver, if we have a codelet */


typedef struct {
	solver super;
	const kdft_desc *desc;
	kdft k;
	int bufferedp;
} direct_S;

typedef struct {
	plan_dft super;

	stride is, os, bufstride;
	INT n, vl, ivs, ovs;
	kdft k;
	const direct_S *slv;
} direct_P;

static void
direct_dobatch(const direct_P *ego, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
	FFTW_REAL_TYPE *buf, INT batchsz) {
	fftw_cpy2d_pair_ci(ri, ii, buf, buf + 1,
		ego->n, WS(ego->is, 1), WS(ego->bufstride, 1),
		batchsz, ego->ivs, 2);

	if (IABS(WS(ego->os, 1)) < IABS(ego->ovs)) {
		/* transform directly to output */
		ego->k(buf, buf + 1, ro, io,
			ego->bufstride, ego->os, batchsz, 2, ego->ovs);
	}
	else {
		/* transform to buffer and copy back */
		ego->k(buf, buf + 1, buf, buf + 1,
			ego->bufstride, ego->bufstride, batchsz, 2, 2);
		fftw_cpy2d_pair_co(buf, buf + 1, ro, io,
			ego->n, WS(ego->bufstride, 1), WS(ego->os, 1),
			batchsz, 2, ego->ovs);
	}
}

static INT direct_compute_batchsize(INT n) {
	/* round up to multiple of 4 */
	n += 3;
	n &= -4;

	return (n + 2);
}

static void
direct_apply_buf(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const direct_P *ego = (const direct_P *)ego_;
	FFTW_REAL_TYPE *buf;
	INT vl = ego->vl, n = ego->n, batchsz = direct_compute_batchsize(n);
	INT i;
	size_t bufsz = n * batchsz * 2 * sizeof(FFTW_REAL_TYPE);

	BUF_ALLOC(FFTW_REAL_TYPE *, buf, bufsz);

	for (i = 0; i < vl - batchsz; i += batchsz) {
		direct_dobatch(ego, ri, ii, ro, io, buf, batchsz);
		ri += batchsz * ego->ivs;
		ii += batchsz * ego->ivs;
		ro += batchsz * ego->ovs;
		io += batchsz * ego->ovs;
	}
	direct_dobatch(ego, ri, ii, ro, io, buf, vl - i);

	BUF_FREE(buf, bufsz);
}

static void
direct_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const direct_P *ego = (const direct_P *)ego_;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(ri, ii, ro, io, ego->is, ego->os, ego->vl, ego->ivs, ego->ovs);
}

static void
direct_apply_extra_iter(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro,
	FFTW_REAL_TYPE *io) {
	const direct_P *ego = (const direct_P *)ego_;
	INT vl = ego->vl;

	ASSERT_ALIGNED_DOUBLE;

	/* for 4-way SIMD when VL is odd: iterate over an
	even vector length VL, and then execute the last
	iteration as a 2-vector with vector stride 0. */
	ego->k(ri, ii, ro, io, ego->is, ego->os, vl - 1, ego->ivs, ego->ovs);

	ego->k(ri + (vl - 1) * ego->ivs, ii + (vl - 1) * ego->ivs,
		ro + (vl - 1) * ego->ovs, io + (vl - 1) * ego->ovs,
		ego->is, ego->os, 1, 0, 0);
}

static void direct_destroy(plan *ego_) {
	direct_P *ego = (direct_P *)ego_;
	fftw_stride_destroy(ego->is);
	fftw_stride_destroy(ego->os);
	fftw_stride_destroy(ego->bufstride);
}

static void direct_print(const plan *ego_, printer *p) {
	const direct_P *ego = (const direct_P *)ego_;
	const direct_S *s = ego->slv;
	const kdft_desc *d = s->desc;

	if (ego->slv->bufferedp)
		p->print(p, "(dft-directbuf/%D-%D%v \"%s\")",
			direct_compute_batchsize(d->sz), d->sz, ego->vl, d->nam);
	else
		p->print(p, "(dft-direct-%D%v \"%s\")", d->sz, ego->vl, d->nam);
}

static int direct_applicable_buf(const solver *ego_, const problem *p_,
	const planner *plnr) {
	const direct_S *ego = (const direct_S *)ego_;
	const problem_dft *p = (const problem_dft *)p_;
	const kdft_desc *d = ego->desc;
	INT vl;
	INT ivs, ovs;
	INT batchsz;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 1
		&& p->sz->dims[0].n == d->sz

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		/* UGLY if IS <= IVS */
		&& !(NO_UGLYP(plnr) &&
			fftw_iabs(p->sz->dims[0].is) <= fftw_iabs(ivs))

		&& (batchsz = direct_compute_batchsize(d->sz), 1)
		&& (d->genus->okp(d, 0, ((const FFTW_REAL_TYPE *)0) + 1, p->ro, p->io,
			2 * batchsz, p->sz->dims[0].os,
			batchsz, 2, ovs, plnr))
		&& (d->genus->okp(d, 0, ((const FFTW_REAL_TYPE *)0) + 1, p->ro, p->io,
			2 * batchsz, p->sz->dims[0].os,
			vl % batchsz, 2, ovs, plnr))


		&& (0
			/* can operate out-of-place */
			|| p->ri != p->ro

			/* can operate in-place as long as strides are the same */
			|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)

			/* can do it if the problem fits in the buffer, no matter
			what the strides are */
			|| vl <= batchsz
			)
		);
}

static int direct_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *extra_iterp) {
	const direct_S *ego = (const direct_S *)ego_;
	const problem_dft *p = (const problem_dft *)p_;
	const kdft_desc *d = ego->desc;
	INT vl;
	INT ivs, ovs;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n == d->sz

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		&& ((*extra_iterp = 0,
		(d->genus->okp(d, p->ri, p->ii, p->ro, p->io,
			p->sz->dims[0].is, p->sz->dims[0].os,
			vl, ivs, ovs, plnr)))
			||
			(*extra_iterp = 1,
			((d->genus->okp(d, p->ri, p->ii, p->ro, p->io,
				p->sz->dims[0].is, p->sz->dims[0].os,
				vl - 1, ivs, ovs, plnr))
				&&
				(d->genus->okp(d, p->ri, p->ii, p->ro, p->io,
					p->sz->dims[0].is, p->sz->dims[0].os,
					2, 0, 0, plnr)))))

		&& (0
			/* can operate out-of-place */
			|| p->ri != p->ro

			/* can always compute one transform */
			|| vl == 1

			/* can operate in-place as long as strides are the same */
			|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)
			)
		);
}


static plan *direct_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const direct_S *ego = (const direct_S *)ego_;
	direct_P *pln;
	const problem_dft *p;
	iodim *d;
	const kdft_desc *e = ego->desc;

	static const plan_adt padt = {
		fftw_dft_solve, fftw_null_awake, direct_print, direct_destroy
	};

	UNUSED(plnr);

	if (ego->bufferedp) {
		if (!direct_applicable_buf(ego_, p_, plnr))
			return (plan *)0;
		pln = MKPLAN_DFT(direct_P, &padt, direct_apply_buf);
	}
	else {
		int extra_iterp = 0;
		if (!direct_applicable(ego_, p_, plnr, &extra_iterp))
			return (plan *)0;
		pln = MKPLAN_DFT(direct_P, &padt, extra_iterp ? direct_apply_extra_iter : direct_apply);
	}

	p = (const problem_dft *)p_;
	d = p->sz->dims;
	pln->k = ego->k;
	pln->n = d[0].n;
	pln->is = fftw_mkstride(pln->n, d[0].is);
	pln->os = fftw_mkstride(pln->n, d[0].os);
	pln->bufstride = fftw_mkstride(pln->n, 2 * direct_compute_batchsize(pln->n));

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);
	pln->slv = ego;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl / e->genus->vl, &e->ops, &pln->super.super.ops);

	if (ego->bufferedp)
		pln->super.super.ops.other += 4 * pln->n * pln->vl;

	pln->super.super.could_prune_now_p = !ego->bufferedp;
	return &(pln->super.super);
}

static solver *direct_mksolver(kdft k, const kdft_desc *desc, int bufferedp) {
	static const solver_adt sadt = { PROBLEM_DFT, direct_mkplan, 0 };
	direct_S *slv = MKSOLVER(direct_S, &sadt);
	slv->k = k;
	slv->desc = desc;
	slv->bufferedp = bufferedp;
	return &(slv->super);
}

solver *fftw_mksolver_dft_direct(kdft k, const kdft_desc *desc) {
	return direct_mksolver(k, desc, 0);
}

solver *fftw_mksolver_dft_directbuf(kdft k, const kdft_desc *desc) {
	return direct_mksolver(k, desc, 1);
}

typedef struct {
	solver super;
} generic_S;

typedef struct {
	plan_dft super;
	twid *td;
	INT n, is, os;
} generic_P;


static void generic_cdot(INT n, const E *x, const FFTW_REAL_TYPE *w,
	FFTW_REAL_TYPE *or0, FFTW_REAL_TYPE *oi0, FFTW_REAL_TYPE *or1, FFTW_REAL_TYPE *oi1) {
	INT i;

	E rr = x[0], ri = 0, ir = x[1], ii = 0;
	x += 2;
	for (i = 1; i + i < n; ++i) {
		rr += x[0] * w[0];
		ir += x[1] * w[0];
		ri += x[2] * w[1];
		ii += x[3] * w[1];
		x += 4;
		w += 2;
	}
	*or0 = rr + ii;
	*oi0 = ir - ri;
	*or1 = rr - ii;
	*oi1 = ir + ri;
}

static void generic_hartley(INT n, const FFTW_REAL_TYPE *xr, const FFTW_REAL_TYPE *xi, INT xs, E *o,
	FFTW_REAL_TYPE *pr, FFTW_REAL_TYPE *pi) {
	INT i;
	E sr, si;
	o[0] = sr = xr[0];
	o[1] = si = xi[0];
	o += 2;
	for (i = 1; i + i < n; ++i) {
		sr += (o[0] = xr[i * xs] + xr[(n - i) * xs]);
		si += (o[1] = xi[i * xs] + xi[(n - i) * xs]);
		o[2] = xr[i * xs] - xr[(n - i) * xs];
		o[3] = xi[i * xs] - xi[(n - i) * xs];
		o += 4;
	}
	*pr = sr;
	*pi = si;
}

static void
generic_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const generic_P *ego = (const generic_P *)ego_;
	INT i;
	INT n = ego->n, is = ego->is, os = ego->os;
	const FFTW_REAL_TYPE *W = ego->td->W;
	E *buf;
	size_t bufsz = n * 2 * sizeof(E);

	BUF_ALLOC(E *, buf, bufsz);
	generic_hartley(n, ri, ii, is, buf, ro, io);

	for (i = 1; i + i < n; ++i) {
		generic_cdot(n, buf, W,
			ro + i * os, io + i * os,
			ro + (n - i) * os, io + (n - i) * os);
		W += n - 1;
	}

	BUF_FREE(buf, bufsz);
}

static void generic_awake(plan *ego_, enum wakefulness wakefulness) {
	generic_P *ego = (generic_P *)ego_;
	static const tw_instr half_tw[] = {
		{ TW_HALF, 1, 0 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_twiddle_awake(wakefulness, &ego->td, half_tw, ego->n, ego->n,
		(ego->n - 1) / 2);
}

static void generic_print(const plan *ego_, printer *p) {
	const generic_P *ego = (const generic_P *)ego_;

	p->print(p, "(dft-generic-%D)", ego->n);
}

static int generic_applicable(const solver *ego, const problem *p_,
	const planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	UNUSED(ego);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& (p->sz->dims[0].n % 2) == 1
		&& CIMPLIES(NO_LARGE_GENERICP(plnr), p->sz->dims[0].n < GENERIC_MIN_BAD)
		&& CIMPLIES(NO_SLOWP(plnr), p->sz->dims[0].n > GENERIC_MAX_SLOW)
		&& fftw_is_prime(p->sz->dims[0].n)
		);
}

static plan *generic_mkplan(const solver *ego, const problem *p_, planner *plnr) {
	const problem_dft *p;
	generic_P *pln;
	INT n;

	static const plan_adt padt = {
		fftw_dft_solve, generic_awake, generic_print, fftw_plan_null_destroy
	};

	if (!generic_applicable(ego, p_, plnr))
		return (plan *)0;

	pln = MKPLAN_DFT(generic_P, &padt, generic_apply);

	p = (const problem_dft *)p_;
	pln->n = n = p->sz->dims[0].n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->td = 0;

	pln->super.super.ops.add = (n - 1) * 5;
	pln->super.super.ops.mul = 0;
	pln->super.super.ops.fma = (n - 1) * (n - 1);
#if 0 /* these are nice pipelined sequential loads and should cost nothing */
	pln->super.super.ops.other = (n - 1)*(4 + 1 + 2 * (n - 1));  /* approximate */
#endif

	return &(pln->super.super);
}

static solver *generic_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, generic_mkplan, 0 };
	generic_S *slv = MKSOLVER(generic_S, &sadt);
	return &(slv->super);
}

void fftw_dft_generic_register(planner *p) {
	REGISTER_SOLVER(p, generic_mksolver());
}

/* solvers/plans for vectors of small DFT's that cannot be done
in-place directly.  Use a rank-0 plan to rearrange the data
before or after the transform.  Can also change an out-of-place
plan into a copy + in-place (where the in-place transform
is e.g. unit stride). */

/* FIXME: merge with rank-geq2.c(?), since this is just a special case
of a rank split where the first/second transform has rank 0. */


typedef problem *(*mkcld_t)(const problem_dft *p);

typedef struct {
	dftapply apply;

	problem *(*mkcld)(const problem_dft *p);

	const char *nam;
} indirect_ndrct_adt;

typedef struct {
	solver super;
	const indirect_ndrct_adt *adt;
} indirect_S;

typedef struct {
	plan_dft super;
	plan *cldcpy, *cld;
	const indirect_S *slv;
} indirect_P;

/*-----------------------------------------------------------------------*/
/* first rearrange, then transform */
static void
indirect_apply_before(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro,
	FFTW_REAL_TYPE *io) {
	const indirect_P *ego = (const indirect_P *)ego_;

	{
		plan_dft *cldcpy = (plan_dft *)ego->cldcpy;
		cldcpy->apply(ego->cldcpy, ri, ii, ro, io);
	}
	{
		plan_dft *cld = (plan_dft *)ego->cld;
		cld->apply(ego->cld, ro, io, ro, io);
	}
}

static problem *indirect_mkcld_before(const problem_dft *p) {
	return fftw_mkproblem_dft_d(fftw_tensor_copy_inplace(p->sz, INPLACE_OS),
		fftw_tensor_copy_inplace(p->vecsz, INPLACE_OS),
		p->ro, p->io, p->ro, p->io);
}

static const indirect_ndrct_adt indirect_adt_before =
{
	indirect_apply_before, indirect_mkcld_before, "dft-indirect-before"
};

/*-----------------------------------------------------------------------*/
/* first transform, then rearrange */

static void
indirect_apply_after(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const indirect_P *ego = (const indirect_P *)ego_;

	{
		plan_dft *cld = (plan_dft *)ego->cld;
		cld->apply(ego->cld, ri, ii, ri, ii);
	}
	{
		plan_dft *cldcpy = (plan_dft *)ego->cldcpy;
		cldcpy->apply(ego->cldcpy, ri, ii, ro, io);
	}
}

static problem *indirect_mkcld_after(const problem_dft *p) {
	return fftw_mkproblem_dft_d(fftw_tensor_copy_inplace(p->sz, INPLACE_IS),
		fftw_tensor_copy_inplace(p->vecsz, INPLACE_IS),
		p->ri, p->ii, p->ri, p->ii);
}

static const indirect_ndrct_adt indirect_adt_after =
{
	indirect_apply_after, indirect_mkcld_after, "dft-indirect-after"
};

/*-----------------------------------------------------------------------*/
static void indirect_destroy(plan *ego_) {
	indirect_P *ego = (indirect_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
	fftw_plan_destroy_internal(ego->cldcpy);
}

static void indirect_awake(plan *ego_, enum wakefulness wakefulness) {
	indirect_P *ego = (indirect_P *)ego_;
	fftw_plan_awake(ego->cldcpy, wakefulness);
	fftw_plan_awake(ego->cld, wakefulness);
}

static void indirect_print(const plan *ego_, printer *p) {
	const indirect_P *ego = (const indirect_P *)ego_;
	const indirect_S *s = ego->slv;
	p->print(p, "(%s%(%p%)%(%p%))", s->adt->nam, ego->cld, ego->cldcpy);
}

static int indirect_applicable0(const solver *ego_, const problem *p_,
	const planner *plnr) {
	const indirect_S *ego = (const indirect_S *)ego_;
	const problem_dft *p = (const problem_dft *)p_;
	return (1
		&& FINITE_RNK(p->vecsz->rnk)

		/* problem must be a nontrivial transform, not just a copy */
		&& p->sz->rnk > 0

		&& (0

			/* problem must be in-place & require some
			rearrangement of the data; to prevent
			infinite loops with indirect-transpose, we
			further require that at least some transform
			strides must decrease */
			|| (p->ri == p->ro
				&& !fftw_tensor_inplace_strides2(p->sz, p->vecsz)
				&& fftw_tensor_strides_decrease(
					p->sz, p->vecsz,
					ego->adt->apply == indirect_apply_after ?
					INPLACE_IS : INPLACE_OS))

			/* or problem must be out of place, transforming
			from stride 1/2 to bigger stride, for apply_after */
			|| (p->ri != p->ro && ego->adt->apply == indirect_apply_after
				&& !NO_DESTROY_INPUTP(plnr)
				&& fftw_tensor_min_istride(p->sz) <= 2
				&& fftw_tensor_min_ostride(p->sz) > 2)

			/* or problem must be out of place, transforming
			to stride 1/2 from bigger stride, for apply_before */
			|| (p->ri != p->ro && ego->adt->apply == indirect_apply_before
				&& fftw_tensor_min_ostride(p->sz) <= 2
				&& fftw_tensor_min_istride(p->sz) > 2)
			)
		);
}

static int indirect_applicable(const solver *ego_, const problem *p_,
	const planner *plnr) {
	if (!indirect_applicable0(ego_, p_, plnr)) return 0;
	{
		const problem_dft *p = (const problem_dft *)p_;
		if (NO_INDIRECT_OP_P(plnr) && p->ri != p->ro) return 0;
	}
	return 1;
}

static plan *indirect_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	const indirect_S *ego = (const indirect_S *)ego_;
	indirect_P *pln;
	plan *cld = 0, *cldcpy = 0;

	static const plan_adt padt = {
		fftw_dft_solve, indirect_awake, indirect_print, indirect_destroy
	};

	if (!indirect_applicable(ego_, p_, plnr))
		return (plan *)0;

	cldcpy =
		fftw_mkplan_d(plnr,
			fftw_mkproblem_dft_d(fftw_mktensor_0d(),
				fftw_tensor_append(p->vecsz, p->sz),
				p->ri, p->ii, p->ro, p->io));

	if (!cldcpy) goto nada;

	cld = fftw_mkplan_f_d(plnr, ego->adt->mkcld(p), NO_BUFFERING, 0, 0);
	if (!cld) goto nada;

	pln = MKPLAN_DFT(indirect_P, &padt, ego->adt->apply);
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->slv = ego;
	fftw_ops_add(&cld->ops, &cldcpy->ops, &pln->super.super.ops);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld);
	fftw_plan_destroy_internal(cldcpy);
	return (plan *)0;
}

static solver *indirect_mksolver(const indirect_ndrct_adt *adt) {
	static const solver_adt sadt = { PROBLEM_DFT, indirect_mkplan, 0 };
	indirect_S *slv = MKSOLVER(indirect_S, &sadt);
	slv->adt = adt;
	return &(slv->super);
}

void fftw_dft_indirect_register(planner *p) {
	unsigned i;
	static const indirect_ndrct_adt *const adts[] = {
		&indirect_adt_before, &indirect_adt_after
	};

	for (i = 0; i < sizeof(adts) / sizeof(adts[0]); ++i)
		REGISTER_SOLVER(p, indirect_mksolver(adts[i]));
}


/* solvers/plans for vectors of DFTs corresponding to the columns
of a matrix: first transpose the matrix so that the DFTs are
contiguous, then do DFTs with transposed output.   In particular,
we restrict ourselves to the case of a square transpose (or a
sequence thereof). */


typedef solver indirect_transpose_S;

typedef struct {
	plan_dft super;
	INT vl, ivs, ovs;
	plan *cldtrans, *cld, *cldrest;
} indirect_transpose_P;

/* initial transpose is out-of-place from input to output */
static void indirect_transpose_apply_op(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro,
	FFTW_REAL_TYPE *io) {
	const indirect_transpose_P *ego = (const indirect_transpose_P *)ego_;
	INT vl = ego->vl, ivs = ego->ivs, ovs = ego->ovs, i;

	for (i = 0; i < vl; ++i) {
		{
			plan_dft *cldtrans = (plan_dft *)ego->cldtrans;
			cldtrans->apply(ego->cldtrans, ri, ii, ro, io);
		}
		{
			plan_dft *cld = (plan_dft *)ego->cld;
			cld->apply(ego->cld, ro, io, ro, io);
		}
		ri += ivs;
		ii += ivs;
		ro += ovs;
		io += ovs;
	}
	{
		plan_dft *cldrest = (plan_dft *)ego->cldrest;
		cldrest->apply(ego->cldrest, ri, ii, ro, io);
	}
}

static void indirect_transpose_destroy(plan *ego_) {
	indirect_transpose_P *ego = (indirect_transpose_P *)ego_;
	fftw_plan_destroy_internal(ego->cldrest);
	fftw_plan_destroy_internal(ego->cld);
	fftw_plan_destroy_internal(ego->cldtrans);
}

static void indirect_transpose_awake(plan *ego_, enum wakefulness wakefulness) {
	indirect_transpose_P *ego = (indirect_transpose_P *)ego_;
	fftw_plan_awake(ego->cldtrans, wakefulness);
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldrest, wakefulness);
}

static void indirect_transpose_print(const plan *ego_, printer *p) {
	const indirect_transpose_P *ego = (const indirect_transpose_P *)ego_;
	p->print(p, "(indirect-transpose%v%(%p%)%(%p%)%(%p%))",
		ego->vl, ego->cldtrans, ego->cld, ego->cldrest);
}

static int indirect_transpose_pickdim(const tensor *vs, const tensor *s, int *pdim0, int *pdim1) {
	int dim0, dim1;
	*pdim0 = *pdim1 = -1;
	for (dim0 = 0; dim0 < vs->rnk; ++dim0)
		for (dim1 = 0; dim1 < s->rnk; ++dim1)
			if (vs->dims[dim0].n * fftw_iabs(vs->dims[dim0].is) <= fftw_iabs(s->dims[dim1].is)
				&& vs->dims[dim0].n >= s->dims[dim1].n
				&& (*pdim0 == -1
					|| (fftw_iabs(vs->dims[dim0].is) <= fftw_iabs(vs->dims[*pdim0].is)
						&& fftw_iabs(s->dims[dim1].is) >= fftw_iabs(s->dims[*pdim1].is)))) {
				*pdim0 = dim0;
				*pdim1 = dim1;
			}
	return (*pdim0 != -1 && *pdim1 != -1);
}

static int indirect_transpose_applicable0(const solver *ego_, const problem *p_,
	const planner *plnr,
	int *pdim0, int *pdim1) {
	const problem_dft *p = (const problem_dft *)p_;
	UNUSED(ego_);
	UNUSED(plnr);

	return (1
		&& FINITE_RNK(p->vecsz->rnk) && FINITE_RNK(p->sz->rnk)

		/* FIXME: can/should we relax this constraint? */
		&& fftw_tensor_inplace_strides2(p->vecsz, p->sz)

		&& indirect_transpose_pickdim(p->vecsz, p->sz, pdim0, pdim1)

		/* output should not *already* include the transpose
		(in which case we duplicate the regular indirect.c) */
		&& (p->sz->dims[*pdim1].os != p->vecsz->dims[*pdim0].is)
		);
}

static int indirect_transpose_applicable(const solver *ego_, const problem *p_,
	const planner *plnr,
	int *pdim0, int *pdim1) {
	if (!indirect_transpose_applicable0(ego_, p_, plnr, pdim0, pdim1)) return 0;
	{
		const problem_dft *p = (const problem_dft *)p_;
		INT u = p->ri == p->ii + 1 || p->ii == p->ri + 1 ? (INT)2 : (INT)1;

		/* UGLY if does not result in contiguous transforms or
		transforms of contiguous vectors (since the latter at
		least have efficient transpositions) */
		if (NO_UGLYP(plnr)
			&& p->vecsz->dims[*pdim0].is != u
			&& !(p->vecsz->rnk == 2
				&& p->vecsz->dims[1 - *pdim0].is == u
				&& p->vecsz->dims[*pdim0].is
				== u * p->vecsz->dims[1 - *pdim0].n))
			return 0;

		if (NO_INDIRECT_OP_P(plnr) && p->ri != p->ro) return 0;
	}
	return 1;
}

static plan *indirect_transpose_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	indirect_transpose_P *pln;
	plan *cld = 0, *cldtrans = 0, *cldrest = 0;
	int pdim0, pdim1;
	tensor *ts, *tv;
	INT vl, ivs, ovs;
	FFTW_REAL_TYPE *rit, *iit, *rot, *iot;

	static const plan_adt padt = {
		fftw_dft_solve, indirect_transpose_awake, indirect_transpose_print, indirect_transpose_destroy
	};

	if (!indirect_transpose_applicable(ego_, p_, plnr, &pdim0, &pdim1))
		return (plan *)0;

	vl = p->vecsz->dims[pdim0].n / p->sz->dims[pdim1].n;
	A(vl >= 1);
	ivs = p->sz->dims[pdim1].n * p->vecsz->dims[pdim0].is;
	ovs = p->sz->dims[pdim1].n * p->vecsz->dims[pdim0].os;
	rit = TAINT(p->ri, vl == 1 ? 0 : ivs);
	iit = TAINT(p->ii, vl == 1 ? 0 : ivs);
	rot = TAINT(p->ro, vl == 1 ? 0 : ovs);
	iot = TAINT(p->io, vl == 1 ? 0 : ovs);

	ts = fftw_tensor_copy_inplace(p->sz, INPLACE_IS);
	ts->dims[pdim1].os = p->vecsz->dims[pdim0].is;
	tv = fftw_tensor_copy_inplace(p->vecsz, INPLACE_IS);
	tv->dims[pdim0].os = p->sz->dims[pdim1].is;
	tv->dims[pdim0].n = p->sz->dims[pdim1].n;
	cldtrans = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(fftw_mktensor_0d(),
			fftw_tensor_append(tv, ts),
			rit, iit,
			rot, iot));
	fftw_tensor_destroy2(ts, tv);
	if (!cldtrans) goto nada;

	ts = fftw_tensor_copy(p->sz);
	ts->dims[pdim1].is = p->vecsz->dims[pdim0].is;
	tv = fftw_tensor_copy(p->vecsz);
	tv->dims[pdim0].is = p->sz->dims[pdim1].is;
	tv->dims[pdim0].n = p->sz->dims[pdim1].n;
	cld = fftw_mkplan_d(plnr, fftw_mkproblem_dft_d(ts, tv,
		rot, iot,
		rot, iot));
	if (!cld) goto nada;

	tv = fftw_tensor_copy(p->vecsz);
	tv->dims[pdim0].n -= vl * p->sz->dims[pdim1].n;
	cldrest = fftw_mkplan_d(plnr, fftw_mkproblem_dft_d(fftw_tensor_copy(p->sz), tv,
		p->ri + ivs * vl,
		p->ii + ivs * vl,
		p->ro + ovs * vl,
		p->io + ovs * vl));
	if (!cldrest) goto nada;

	pln = MKPLAN_DFT(indirect_transpose_P, &padt, indirect_transpose_apply_op);
	pln->cldtrans = cldtrans;
	pln->cld = cld;
	pln->cldrest = cldrest;
	pln->vl = vl;
	pln->ivs = ivs;
	pln->ovs = ovs;
	fftw_ops_cpy(&cldrest->ops, &pln->super.super.ops);
	fftw_ops_madd2(vl, &cld->ops, &pln->super.super.ops);
	fftw_ops_madd2(vl, &cldtrans->ops, &pln->super.super.ops);
	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cldrest);
	fftw_plan_destroy_internal(cld);
	fftw_plan_destroy_internal(cldtrans);
	return (plan *)0;
}

static solver *indirect_transpose_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, indirect_transpose_mkplan, 0 };
	indirect_transpose_S *slv = MKSOLVER(indirect_transpose_S, &sadt);
	return slv;
}

void fftw_dft_indirect_transpose_register(planner *p) {
	REGISTER_SOLVER(p, indirect_transpose_mksolver());
}

void fftw_kdft_dif_register(planner *p, kdftw codelet, const ct_desc *desc) {
	fftw_regsolver_ct_directw(p, codelet, desc, DECDIF);
}

void fftw_kdft_difsq_register(planner *p, kdftwsq k, const ct_desc *desc) {
	fftw_regsolver_ct_directwsq(p, k, desc, DECDIF);
}


void fftw_kdft_dit_register(planner *p, kdftw codelet, const ct_desc *desc) {
	fftw_regsolver_ct_directw(p, codelet, desc, DECDIT);
}

/* plans for vrank -infty DFTs (nothing to do) */


static void
nop_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	UNUSED(ego_);
	UNUSED(ri);
	UNUSED(ii);
	UNUSED(ro);
	UNUSED(io);
}

static int nop_applicable(const solver *ego_, const problem *p_) {
	const problem_dft *p = (const problem_dft *)p_;

	UNUSED(ego_);

	return 0
		/* case 1 : -infty vector rank */
		|| (!FINITE_RNK(p->vecsz->rnk))

		/* case 2 : rank-0 in-place dft */
		|| (1
			&& p->sz->rnk == 0
			&& FINITE_RNK(p->vecsz->rnk)
			&& p->ro == p->ri
			&& fftw_tensor_inplace_strides(p->vecsz)
			);
}

static void nop_print(const plan *ego, printer *p) {
	UNUSED(ego);
	p->print(p, "(dft-nop)");
}

static plan *nop_mkplan(const solver *ego, const problem *p, planner *plnr) {
	static const plan_adt padt = {
		fftw_dft_solve, fftw_null_awake, nop_print, fftw_plan_null_destroy
	};
	plan_dft *pln;

	UNUSED(plnr);

	if (!nop_applicable(ego, p))
		return (plan *)0;
	pln = MKPLAN_DFT(plan_dft, &padt, nop_apply);
	fftw_ops_zero(&pln->super.ops);

	return &(pln->super);
}

static solver *nop_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, nop_mkplan, 0 };
	return MKSOLVER(solver, &sadt);
}

void fftw_dft_nop_register(planner *p) {
	REGISTER_SOLVER(p, nop_mksolver());
}

plan *fftw_mkplan_dft(size_t size, const plan_adt *adt, dftapply apply) {
	plan_dft *ego;

	ego = (plan_dft *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}


static void problem_destroy(problem *ego_) {
	problem_dft *ego = (problem_dft *)ego_;
	fftw_tensor_destroy2(ego->vecsz, ego->sz);
	fftw_ifree(ego_);
}

static void problem_hash(const problem *p_, md5 *m) {
	const problem_dft *p = (const problem_dft *)p_;
	fftw_md5puts(m, "dft");
	fftw_md5int(m, p->ri == p->ro);
	fftw_md5INT(m, p->ii - p->ri);
	fftw_md5INT(m, p->io - p->ro);
	fftw_md5int(m, fftw_ialignment_of(p->ri));
	fftw_md5int(m, fftw_ialignment_of(p->ii));
	fftw_md5int(m, fftw_ialignment_of(p->ro));
	fftw_md5int(m, fftw_ialignment_of(p->io));
	fftw_tensor_md5(m, p->sz);
	fftw_tensor_md5(m, p->vecsz);
}

static void problem_print(const problem *ego_, printer *p) {
	const problem_dft *ego = (const problem_dft *)ego_;
	p->print(p, "(dft %d %d %d %D %D %T %T)",
		ego->ri == ego->ro,
		fftw_ialignment_of(ego->ri),
		fftw_ialignment_of(ego->ro),
		(INT)(ego->ii - ego->ri),
		(INT)(ego->io - ego->ro),
		ego->sz,
		ego->vecsz);
}

static void problem_zero(const problem *ego_) {
	const problem_dft *ego = (const problem_dft *)ego_;
	tensor *sz = fftw_tensor_append(ego->vecsz, ego->sz);
	fftw_dft_zerotens(sz, UNTAINT(ego->ri), UNTAINT(ego->ii));
	fftw_tensor_destroy(sz);
}

static const problem_adt padt =
{
	PROBLEM_DFT,
	problem_hash,
	problem_zero,
	problem_print,
	problem_destroy
};

problem *fftw_mkproblem_dft(const tensor *sz, const tensor *vecsz,
	FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	problem_dft *ego;

	/* enforce pointer equality if untainted pointers are equal */
	if (UNTAINT(ri) == UNTAINT(ro))
		ri = ro = JOIN_TAINT(ri, ro);
	if (UNTAINT(ii) == UNTAINT(io))
		ii = io = JOIN_TAINT(ii, io);

	/* more correctness conditions: */
	A(TAINTOF(ri) == TAINTOF(ii));
	A(TAINTOF(ro) == TAINTOF(io));

	A(fftw_tensor_kosherp(sz));
	A(fftw_tensor_kosherp(vecsz));

	if (ri == ro || ii == io) {
		/* If either real or imag pointers are in place, both must be. */
		if (ri != ro || ii != io || !fftw_tensor_inplace_locations(sz, vecsz))
			return fftw_mkproblem_unsolvable();
	}

	ego = (problem_dft *)fftw_mkproblem(sizeof(problem_dft), &padt);

	ego->sz = fftw_tensor_compress(sz);
	ego->vecsz = fftw_tensor_compress_contiguous(vecsz);
	ego->ri = ri;
	ego->ii = ii;
	ego->ro = ro;
	ego->io = io;

	A(FINITE_RNK(ego->sz->rnk));
	return &(ego->super);
}

/* Same as fftw_mkproblem_dft, but also destroy input tensors. */
problem *fftw_mkproblem_dft_d(tensor *sz, tensor *vecsz,
	FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	problem *p = fftw_mkproblem_dft(sz, vecsz, ri, ii, ro, io);
	fftw_tensor_destroy2(vecsz, sz);
	return p;
}

/*
* Compute transforms of prime sizes using Rader's trick: turn them
* into convolutions of size n - 1, which you then perform via a pair
* of FFTs.
*/

typedef struct {
	solver super;
} rader_S;

typedef struct {
	plan_dft super;

	plan *cld1, *cld2;
	FFTW_REAL_TYPE *omega;
	INT n, g, ginv;
	INT is, os;
	plan *cld_omega;
} rader_P;

static rader_tl *rader_omegas = 0;

static FFTW_REAL_TYPE *rader_mkomega(enum wakefulness wakefulness, plan *p_, INT n, INT ginv) {
	plan_dft *p = (plan_dft *)p_;
	FFTW_REAL_TYPE *omega;
	INT i, gpower;
	trigreal scale;
	triggen *t;

	if ((omega = fftw_rader_tl_find(n, n, ginv, rader_omegas)))
		return omega;

	omega = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (n - 1) * 2, TWIDDLES);

	scale = n - 1.0; /* normalization for convolution */

	t = fftw_mktriggen(wakefulness, n);
	for (i = 0, gpower = 1; i < n - 1; ++i, gpower = MULMOD(gpower, ginv, n)) {
		trigreal w[2];
		t->cexpl(t, gpower, w);
		omega[2 * i] = w[0] / scale;
		omega[2 * i + 1] = FFT_SIGN * w[1] / scale;
	}
	fftw_triggen_destroy(t);
	A(gpower == 1);

	p->apply(p_, omega, omega + 1, omega, omega + 1);

	fftw_rader_tl_insert(n, n, ginv, omega, &rader_omegas);
	return omega;
}

static void rader_free_omega(FFTW_REAL_TYPE *omega) {
	fftw_rader_tl_delete(omega, &rader_omegas);
}


/***************************************************************************/

/* Below, we extensively use the identity that fft(x*)* = ifft(x) in
order to share data between forward and backward transforms and to
obviate the necessity of having separate forward and backward
plans.  (Although we often compute separate plans these days anyway
due to the differing strides, etcetera.)

Of course, since the new FFTW gives us separate pointers to
the real and imaginary parts, we could have instead used the
fft(r,i) = ifft(i,r) form of this identity, but it was easier to
reuse the code from our old version. */

static void
rader_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const rader_P *ego = (const rader_P *)ego_;
	INT is, os;
	INT k, gpower, g, r;
	FFTW_REAL_TYPE *buf;
	FFTW_REAL_TYPE r0 = ri[0], i0 = ii[0];

	r = ego->n;
	is = ego->is;
	os = ego->os;
	g = ego->g;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (r - 1) * 2, BUFFERS);

	/* First, permute the input, storing in buf: */
	for (gpower = 1, k = 0; k < r - 1; ++k, gpower = MULMOD(gpower, g, r)) {
		FFTW_REAL_TYPE rA, iA;
		rA = ri[gpower * is];
		iA = ii[gpower * is];
		buf[2 * k] = rA;
		buf[2 * k + 1] = iA;
	}
	/* gpower == g^(r-1) mod r == 1 */;


	/* compute DFT of buf, storing in output (except DC): */
	{
		plan_dft *cld = (plan_dft *)ego->cld1;
		cld->apply(ego->cld1, buf, buf + 1, ro + os, io + os);
	}

	/* set output DC component: */
	{
		ro[0] = r0 + ro[os];
		io[0] = i0 + io[os];
	}

	/* now, multiply by omega: */
	{
		const FFTW_REAL_TYPE *omega = ego->omega;
		for (k = 0; k < r - 1; ++k) {
			E rB, iB, rW, iW;
			rW = omega[2 * k];
			iW = omega[2 * k + 1];
			rB = ro[(k + 1) * os];
			iB = io[(k + 1) * os];
			ro[(k + 1) * os] = rW * rB - iW * iB;
			io[(k + 1) * os] = -(rW * iB + iW * rB);
		}
	}

	/* this will add input[0] to all of the outputs after the ifft */
	ro[os] += r0;
	io[os] -= i0;

	/* inverse FFT: */
	{
		plan_dft *cld = (plan_dft *)ego->cld2;
		cld->apply(ego->cld2, ro + os, io + os, buf, buf + 1);
	}

	/* finally, do inverse permutation to unshuffle the output: */
	{
		INT ginv = ego->ginv;
		gpower = 1;
		for (k = 0; k < r - 1; ++k, gpower = MULMOD(gpower, ginv, r)) {
			ro[gpower * os] = buf[2 * k];
			io[gpower * os] = -buf[2 * k + 1];
		}
		A(gpower == 1);
	}


	fftw_ifree(buf);
}

/***************************************************************************/

static void rader_awake(plan *ego_, enum wakefulness wakefulness) {
	rader_P *ego = (rader_P *)ego_;

	fftw_plan_awake(ego->cld1, wakefulness);
	fftw_plan_awake(ego->cld2, wakefulness);
	fftw_plan_awake(ego->cld_omega, wakefulness);

	switch (wakefulness) {
	case SLEEPY:
		rader_free_omega(ego->omega);
		ego->omega = 0;
		break;
	default:
		ego->g = fftw_find_generator(ego->n);
		ego->ginv = fftw_power_mod(ego->g, ego->n - 2, ego->n);
		A(MULMOD(ego->g, ego->ginv, ego->n) == 1);

		ego->omega = rader_mkomega(wakefulness,
			ego->cld_omega, ego->n, ego->ginv);
		break;
	}
}

static void rader_destroy(plan *ego_) {
	rader_P *ego = (rader_P *)ego_;
	fftw_plan_destroy_internal(ego->cld_omega);
	fftw_plan_destroy_internal(ego->cld2);
	fftw_plan_destroy_internal(ego->cld1);
}

static void rader_print(const plan *ego_, printer *p) {
	const rader_P *ego = (const rader_P *)ego_;
	p->print(p, "(dft-rader-%D%ois=%oos=%(%p%)",
		ego->n, ego->is, ego->os, ego->cld1);
	if (ego->cld2 != ego->cld1)
		p->print(p, "%(%p%)", ego->cld2);
	if (ego->cld_omega != ego->cld1 && ego->cld_omega != ego->cld2)
		p->print(p, "%(%p%)", ego->cld_omega);
	p->putchr(p, ')');
}

static int rader_applicable(const solver *ego_, const problem *p_,
	const planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	UNUSED(ego_);
	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& CIMPLIES(NO_SLOWP(plnr), p->sz->dims[0].n > RADER_MAX_SLOW)
		&& fftw_is_prime(p->sz->dims[0].n)

		/* proclaim the solver SLOW if p-1 is not easily factorizable.
		Bluestein should take care of this case. */
		&& CIMPLIES(NO_SLOWP(plnr), fftw_factors_into_small_primes(p->sz->dims[0].n - 1))
		);
}

static int rader_mkP(rader_P *pln, INT n, INT is, INT os, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
	planner *plnr) {
	plan *cld1;
	plan *cld2 = (plan *)0;
	plan *cld_omega = (plan *)0;
	FFTW_REAL_TYPE *buf;

	/* initial allocation for the purpose of planning */
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (n - 1) * 2, BUFFERS);

	cld1 = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_dft_d(fftw_mktensor_1d(n - 1, 2, os),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf + 1, ro + os, io + os),
		NO_SLOW, 0, 0);
	if (!cld1) goto nada;

	cld2 = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_dft_d(fftw_mktensor_1d(n - 1, os, 2),
			fftw_mktensor_1d(1, 0, 0),
			ro + os, io + os, buf, buf + 1),
		NO_SLOW, 0, 0);

	if (!cld2) goto nada;

	/* plan for omega array */
	cld_omega = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_dft_d(fftw_mktensor_1d(n - 1, 2, 2),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf + 1, buf, buf + 1),
		NO_SLOW, ESTIMATE, 0);
	if (!cld_omega) goto nada;

	/* deallocate buffers; let awake() or apply() allocate them for real */
	fftw_ifree(buf);

	pln->cld1 = cld1;
	pln->cld2 = cld2;
	pln->cld_omega = cld_omega;
	pln->omega = 0;
	pln->n = n;
	pln->is = is;
	pln->os = os;

	fftw_ops_add(&cld1->ops, &cld2->ops, &pln->super.super.ops);
	pln->super.super.ops.other += (n - 1) * (4 * 2 + 6) + 6;
	pln->super.super.ops.add += (n - 1) * 2 + 4;
	pln->super.super.ops.mul += (n - 1) * 4;

	return 1;

nada:
	fftw_ifree0(buf);
	fftw_plan_destroy_internal(cld_omega);
	fftw_plan_destroy_internal(cld2);
	fftw_plan_destroy_internal(cld1);
	return 0;
}

static plan *rader_mkplan(const solver *ego, const problem *p_, planner *plnr) {
	const problem_dft *p = (const problem_dft *)p_;
	rader_P *pln;
	INT n;
	INT is, os;

	static const plan_adt padt = {
		fftw_dft_solve, rader_awake, rader_print, rader_destroy
	};

	if (!rader_applicable(ego, p_, plnr))
		return (plan *)0;

	n = p->sz->dims[0].n;
	is = p->sz->dims[0].is;
	os = p->sz->dims[0].os;

	pln = MKPLAN_DFT(rader_P, &padt, rader_apply);
	if (!rader_mkP(pln, n, is, os, p->ro, p->io, plnr)) {
		fftw_ifree(pln);
		return (plan *)0;
	}
	return &(pln->super.super);
}

static solver *rader_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, rader_mkplan, 0 };
	rader_S *slv = MKSOLVER(rader_S, &sadt);
	return &(slv->super);
}

void fftw_dft_rader_register(planner *p) {
	REGISTER_SOLVER(p, rader_mksolver());
}

/* plans for DFT of rank >= 2 (multidimensional) */


typedef struct {
	solver super;
	int spltrnk;
	const int *buddies;
	size_t nbuddies;
} rank_geq2_S;

typedef struct {
	plan_dft super;

	plan *cld1, *cld2;
	const rank_geq2_S *solver;
} rank_geq2_P;

/* Compute multi-dimensional DFT by applying the two cld plans
(lower-rnk DFTs). */
static void
rank_geq2_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const rank_geq2_P *ego = (const rank_geq2_P *)ego_;
	plan_dft *cld1, *cld2;

	cld1 = (plan_dft *)ego->cld1;
	cld1->apply(ego->cld1, ri, ii, ro, io);

	cld2 = (plan_dft *)ego->cld2;
	cld2->apply(ego->cld2, ro, io, ro, io);
}


static void rank_geq2_awake(plan *ego_, enum wakefulness wakefulness) {
	rank_geq2_P *ego = (rank_geq2_P *)ego_;
	fftw_plan_awake(ego->cld1, wakefulness);
	fftw_plan_awake(ego->cld2, wakefulness);
}

static void rank_geq2_destroy(plan *ego_) {
	rank_geq2_P *ego = (rank_geq2_P *)ego_;
	fftw_plan_destroy_internal(ego->cld2);
	fftw_plan_destroy_internal(ego->cld1);
}

static void rank_geq2_print(const plan *ego_, printer *p) {
	const rank_geq2_P *ego = (const rank_geq2_P *)ego_;
	const rank_geq2_S *s = ego->solver;
	p->print(p, "(dft-rank>=2/%d%(%p%)%(%p%))",
		s->spltrnk, ego->cld1, ego->cld2);
}

static int rank_geq2_picksplit(const rank_geq2_S *ego, const tensor *sz, int *rp) {
	A(sz->rnk > 1); /* cannot split rnk <= 1 */
	if (!fftw_pickdim(ego->spltrnk, ego->buddies, ego->nbuddies, sz, 1, rp))
		return 0;
	*rp += 1; /* convert from dim. index to rank */
	if (*rp >= sz->rnk) /* split must reduce rank */
		return 0;
	return 1;
}

static int rank_geq2_applicable0(const solver *ego_, const problem *p_, int *rp) {
	const problem_dft *p = (const problem_dft *)p_;
	const rank_geq2_S *ego = (const rank_geq2_S *)ego_;
	return (1
		&& FINITE_RNK(p->sz->rnk) && FINITE_RNK(p->vecsz->rnk)
		&& p->sz->rnk >= 2
		&& rank_geq2_picksplit(ego, p->sz, rp)
		);
}

/* TODO: revise this. */
static int rank_geq2_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *rp) {
	const rank_geq2_S *ego = (const rank_geq2_S *)ego_;
	const problem_dft *p = (const problem_dft *)p_;

	if (!rank_geq2_applicable0(ego_, p_, rp)) return 0;

	if (NO_RANK_SPLITSP(plnr) && (ego->spltrnk != ego->buddies[0])) return 0;

	/* Heuristic: if the vector stride is greater than the transform
	sz, don't use (prefer to do the vector loop first with a
	vrank-geq1 plan). */
	if (NO_UGLYP(plnr))
		if (p->vecsz->rnk > 0 &&
			fftw_tensor_min_stride(p->vecsz) > fftw_tensor_max_index(p->sz))
			return 0;

	return 1;
}

static plan *rank_geq2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rank_geq2_S *ego = (const rank_geq2_S *)ego_;
	const problem_dft *p;
	rank_geq2_P *pln;
	plan *cld1 = 0, *cld2 = 0;
	tensor *sz1, *sz2, *vecszi, *sz2i;
	int spltrnk;

	static const plan_adt padt = {
		fftw_dft_solve, rank_geq2_awake, rank_geq2_print, rank_geq2_destroy
	};

	if (!rank_geq2_applicable(ego_, p_, plnr, &spltrnk))
		return (plan *)0;

	p = (const problem_dft *)p_;
	fftw_tensor_split(p->sz, &sz1, spltrnk, &sz2);
	vecszi = fftw_tensor_copy_inplace(p->vecsz, INPLACE_OS);
	sz2i = fftw_tensor_copy_inplace(sz2, INPLACE_OS);

	cld1 = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(fftw_tensor_copy(sz2),
			fftw_tensor_append(p->vecsz, sz1),
			p->ri, p->ii, p->ro, p->io));
	if (!cld1) goto nada;

	cld2 = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_tensor_copy_inplace(sz1, INPLACE_OS),
			fftw_tensor_append(vecszi, sz2i),
			p->ro, p->io, p->ro, p->io));
	if (!cld2) goto nada;

	pln = MKPLAN_DFT(rank_geq2_P, &padt, rank_geq2_apply);

	pln->cld1 = cld1;
	pln->cld2 = cld2;

	pln->solver = ego;
	fftw_ops_add(&cld1->ops, &cld2->ops, &pln->super.super.ops);

	fftw_tensor_destroy4(sz1, sz2, vecszi, sz2i);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld2);
	fftw_plan_destroy_internal(cld1);
	fftw_tensor_destroy4(sz1, sz2, vecszi, sz2i);
	return (plan *)0;
}

static solver *rank_geq2_mksolver(int spltrnk, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_DFT, rank_geq2_mkplan, 0 };
	rank_geq2_S *slv = MKSOLVER(rank_geq2_S, &sadt);
	slv->spltrnk = spltrnk;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_dft_rank_geq2_register(planner *p) {
	static const int buddies[] = { 1, 0, -2 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, rank_geq2_mksolver(buddies[i], buddies, NELEM(buddies)));

	/* FIXME:

	Should we try more buddies?

	Another possible variant is to swap cld1 and cld2 (or rather,
	to swap their problems; they are not interchangeable because
	cld2 must be in-place).  In past versions of FFTW, however, I
	seem to recall that such rearrangements have made little or no
	difference.
	*/
}

/* use the apply() operation for DFT problems */
void fftw_dft_solve(const plan *ego_, const problem *p_) {
	const plan_dft *ego = (const plan_dft *)ego_;
	const problem_dft *p = (const problem_dft *)p_;
	ego->apply(ego_,
		UNTAINT(p->ri), UNTAINT(p->ii),
		UNTAINT(p->ro), UNTAINT(p->io));
}

/* Plans for handling vector transform loops.  These are *just* the
loops, and rely on child plans for the actual DFTs.

They form a wrapper around solvers that don't have apply functions
for non-null vectors.

vrank-geq1 plans also recursively handle the case of multi-dimensional
vectors, obviating the need for most solvers to deal with this.  We
can also play games here, such as reordering the vector loops.

Each vrank-geq1 plan reduces the vector rank by 1, picking out a
dimension determined by the vecloop_dim field of the solver. */


typedef struct {
	solver super;
	int vecloop_dim;
	const int *buddies;
	size_t nbuddies;
} vrank_geq1_S;

typedef struct {
	plan_dft super;

	plan *cld;
	INT vl;
	INT ivs, ovs;
	const vrank_geq1_S *solver;
} vrank_geq1_P;

static void
vrank_geq1_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const vrank_geq1_P *ego = (const vrank_geq1_P *)ego_;
	INT i, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	dftapply cldapply = ((plan_dft *)ego->cld)->apply;

	for (i = 0; i < vl; ++i) {
		cldapply(ego->cld,
			ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);
	}
}

static void vrank_geq1_awake(plan *ego_, enum wakefulness wakefulness) {
	vrank_geq1_P *ego = (vrank_geq1_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void vrank_geq1_destroy(plan *ego_) {
	vrank_geq1_P *ego = (vrank_geq1_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void vrank_geq1_print(const plan *ego_, printer *p) {
	const vrank_geq1_P *ego = (const vrank_geq1_P *)ego_;
	const vrank_geq1_S *s = ego->solver;
	p->print(p, "(dft-vrank>=1-x%D/%d%(%p%))",
		ego->vl, s->vecloop_dim, ego->cld);
}

static int vrank_geq1_pickdim(const vrank_geq1_S *ego, const tensor *vecsz, int oop, int *dp) {
	return fftw_pickdim(ego->vecloop_dim, ego->buddies, ego->nbuddies,
		vecsz, oop, dp);
}

static int vrank_geq1_applicable0(const solver *ego_, const problem *p_, int *dp) {
	const vrank_geq1_S *ego = (const vrank_geq1_S *)ego_;
	const problem_dft *p = (const problem_dft *)p_;

	return (1
		&& FINITE_RNK(p->vecsz->rnk)
		&& p->vecsz->rnk > 0

		/* do not bother looping over rank-0 problems,
		since they are handled via rdft */
		&& p->sz->rnk > 0

		&& vrank_geq1_pickdim(ego, p->vecsz, p->ri != p->ro, dp)
		);
}

static int vrank_geq1_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *dp) {
	const vrank_geq1_S *ego = (const vrank_geq1_S *)ego_;
	const problem_dft *p;

	if (!vrank_geq1_applicable0(ego_, p_, dp)) return 0;

	/* fftw2 behavior */
	if (NO_VRANK_SPLITSP(plnr) && (ego->vecloop_dim != ego->buddies[0]))
		return 0;

	p = (const problem_dft *)p_;

	if (NO_UGLYP(plnr)) {
		/* Heuristic: if the transform is multi-dimensional, and the
		vector stride is less than the transform size, then we
		probably want to use a rank>=2 plan first in order to combine
		this vector with the transform-dimension vectors. */
		{
			iodim *d = p->vecsz->dims + *dp;
			if (1
				&& p->sz->rnk > 1
				&& fftw_imin(fftw_iabs(d->is), fftw_iabs(d->os))
				< fftw_tensor_max_index(p->sz)
				)
				return 0;
		}

		if (NO_NONTHREADEDP(plnr)) return 0; /* prefer threaded version */
	}

	return 1;
}

static plan *vrank_geq1_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const vrank_geq1_S *ego = (const vrank_geq1_S *)ego_;
	const problem_dft *p;
	vrank_geq1_P *pln;
	plan *cld;
	int vdim;
	iodim *d;

	static const plan_adt padt = {
		fftw_dft_solve, vrank_geq1_awake, vrank_geq1_print, vrank_geq1_destroy
	};

	if (!vrank_geq1_applicable(ego_, p_, plnr, &vdim))
		return (plan *)0;
	p = (const problem_dft *)p_;

	d = p->vecsz->dims + vdim;

	A(d->n > 1);
	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_dft_d(
			fftw_tensor_copy(p->sz),
			fftw_tensor_copy_except(p->vecsz, vdim),
			TAINT(p->ri, d->is), TAINT(p->ii, d->is),
			TAINT(p->ro, d->os), TAINT(p->io, d->os)));
	if (!cld) return (plan *)0;

	pln = MKPLAN_DFT(vrank_geq1_P, &padt, vrank_geq1_apply);

	pln->cld = cld;
	pln->vl = d->n;
	pln->ivs = d->is;
	pln->ovs = d->os;

	pln->solver = ego;
	fftw_ops_zero(&pln->super.super.ops);
	pln->super.super.ops.other = 3.14159; /* magic to prefer codelet loops */
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	if (p->sz->rnk != 1 || (p->sz->dims[0].n > 64))
		pln->super.super.pcost = pln->vl * cld->pcost;

	return &(pln->super.super);
}

static solver *vrank_geq1_mksolver(int vecloop_dim, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_DFT, vrank_geq1_mkplan, 0 };
	vrank_geq1_S *slv = MKSOLVER(vrank_geq1_S, &sadt);
	slv->vecloop_dim = vecloop_dim;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_dft_vrank_geq1_register(planner *p) {
	/* FIXME: Should we try other vecloop_dim values? */
	static const int buddies[] = { 1, -1 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, vrank_geq1_mksolver(buddies[i], buddies, NELEM(buddies)));
}

/* fill a complex array with zeros. */
static void zero_recur(const iodim *dims, int rnk, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii) {
	if (rnk == RNK_MINFTY)
		return;
	else if (rnk == 0)
		ri[0] = ii[0] = K(0.0);
	else if (rnk > 0) {
		INT i, n = dims[0].n;
		INT is = dims[0].is;

		if (rnk == 1) {
			/* this case is redundant but faster */
			for (i = 0; i < n; ++i)
				ri[i * is] = ii[i * is] = K(0.0);
		}
		else {
			for (i = 0; i < n; ++i)
				zero_recur(dims + 1, rnk - 1, ri + i * is, ii + i * is);
		}
	}
}


void fftw_dft_zerotens(tensor *sz, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii) {
	zero_recur(sz->dims, sz->rnk, ri, ii);
}

typedef struct {
	solver super;
	size_t maxnbuf_ndx;
} rdft_buffered_S;

static const INT rdft_maxnbufs[] = { 8, 256 };

typedef struct {
	plan_rdft super;

	plan *cld, *cldcpy, *cldrest;
	INT n, vl, nbuf, bufdist;
	INT ivs_by_nbuf, ovs_by_nbuf;
} rdft_buffered_P;

/* transform a vector input with the help of bufs */
static void rdft_buffered_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_buffered_P *ego = (const rdft_buffered_P *)ego_;
	plan_rdft *cld = (plan_rdft *)ego->cld;
	plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
	plan_rdft *cldrest;
	INT i, vl = ego->vl, nbuf = ego->nbuf;
	INT ivs_by_nbuf = ego->ivs_by_nbuf, ovs_by_nbuf = ego->ovs_by_nbuf;
	FFTW_REAL_TYPE *bufs;

	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * ego->bufdist, BUFFERS);

	for (i = nbuf; i <= vl; i += nbuf) {
		/* transform to bufs: */
		cld->apply((plan *)cld, I, bufs);
		I += ivs_by_nbuf;

		/* copy back */
		cldcpy->apply((plan *)cldcpy, bufs, O);
		O += ovs_by_nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft *)ego->cldrest;
	cldrest->apply((plan *)cldrest, I, O);
}

/* for hc2r problems, copy the input into buffer, and then
transform buffer->output, which allows for destruction of the
buffer */
static void rdft_buffered_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_buffered_P *ego = (const rdft_buffered_P *)ego_;
	plan_rdft *cld = (plan_rdft *)ego->cld;
	plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
	plan_rdft *cldrest;
	INT i, vl = ego->vl, nbuf = ego->nbuf;
	INT ivs_by_nbuf = ego->ivs_by_nbuf, ovs_by_nbuf = ego->ovs_by_nbuf;
	FFTW_REAL_TYPE *bufs;

	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * ego->bufdist, BUFFERS);

	for (i = nbuf; i <= vl; i += nbuf) {
		/* copy input into bufs: */
		cldcpy->apply((plan *)cldcpy, I, bufs);
		I += ivs_by_nbuf;

		/* transform to output */
		cld->apply((plan *)cld, bufs, O);
		O += ovs_by_nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft *)ego->cldrest;
	cldrest->apply((plan *)cldrest, I, O);
}


static void rdft_buffered_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_buffered_P *ego = (rdft_buffered_P *)ego_;

	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldcpy, wakefulness);
	fftw_plan_awake(ego->cldrest, wakefulness);
}

static void rdft_buffered_destroy(plan *ego_) {
	rdft_buffered_P *ego = (rdft_buffered_P *)ego_;
	fftw_plan_destroy_internal(ego->cldrest);
	fftw_plan_destroy_internal(ego->cldcpy);
	fftw_plan_destroy_internal(ego->cld);
}

static void rdft_buffered_print(const plan *ego_, printer *p) {
	const rdft_buffered_P *ego = (const rdft_buffered_P *)ego_;
	p->print(p, "(rdft-buffered-%D%v/%D-%D%(%p%)%(%p%)%(%p%))",
		ego->n, ego->nbuf,
		ego->vl, ego->bufdist % ego->n,
		ego->cld, ego->cldcpy, ego->cldrest);
}

static int rdft_buffered_applicable0(const rdft_buffered_S *ego, const problem *p_, const planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	iodim *d = p->sz->dims;

	if (1
		&& p->vecsz->rnk <= 1
		&& p->sz->rnk == 1
		) {
		INT vl, ivs, ovs;
		fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

		if (fftw_toobig(d[0].n) && CONSERVE_MEMORYP(plnr))
			return 0;

		/* if this solver is redundant, in the sense that a solver
		of lower index generates the same plan, then prune this
		solver */
		if (fftw_nbuf_redundant(d[0].n, vl,
			ego->maxnbuf_ndx,
			rdft_maxnbufs, NELEM(rdft_maxnbufs)))
			return 0;

		if (p->I != p->O) {
			if (p->kind[0] == HC2R) {
				/* Allow HC2R problems only if the input is to be
				preserved.  This solver sets NO_DESTROY_INPUT,
				which prevents infinite loops */
				return (NO_DESTROY_INPUTP(plnr));
			}
			else {
				/*
				In principle, the buffered transforms might be useful
				when working out of place.  However, in order to
				prevent infinite loops in the planner, we require
				that the output stride of the buffered transforms be
				greater than 1.
				*/
				return (d[0].os > 1);
			}
		}

		/*
		* If the problem is in place, the input/output strides must
		* be the same or the whole thing must fit in the buffer.
		*/
		if (fftw_tensor_inplace_strides2(p->sz, p->vecsz))
			return 1;

		if (/* fits into buffer: */
			((p->vecsz->rnk == 0)
				||
				(fftw_nbuf(d[0].n, p->vecsz->dims[0].n,
					rdft_maxnbufs[ego->maxnbuf_ndx])
					== p->vecsz->dims[0].n)))
			return 1;
	}

	return 0;
}

static int rdft_buffered_applicable(const rdft_buffered_S *ego, const problem *p_, const planner *plnr) {
	const problem_rdft *p;

	if (NO_BUFFERINGP(plnr)) return 0;

	if (!rdft_buffered_applicable0(ego, p_, plnr)) return 0;

	p = (const problem_rdft *)p_;
	if (p->kind[0] == HC2R) {
		if (NO_UGLYP(plnr)) {
			/* UGLY if in-place and too big, since the problem
			could be solved via transpositions */
			if (p->I == p->O && fftw_toobig(p->sz->dims[0].n))
				return 0;
		}
	}
	else {
		if (NO_UGLYP(plnr)) {
			if (p->I != p->O) return 0;
			if (fftw_toobig(p->sz->dims[0].n)) return 0;
		}
	}
	return 1;
}

static plan *rdft_buffered_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	rdft_buffered_P *pln;
	const rdft_buffered_S *ego = (const rdft_buffered_S *)ego_;
	plan *cld = (plan *)0;
	plan *cldcpy = (plan *)0;
	plan *cldrest = (plan *)0;
	const problem_rdft *p = (const problem_rdft *)p_;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)0;
	INT nbuf = 0, bufdist, n, vl;
	INT ivs, ovs;
	int hc2rp;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_buffered_awake, rdft_buffered_print, rdft_buffered_destroy
	};

	if (!rdft_buffered_applicable(ego, p_, plnr))
		goto nada;

	n = fftw_tensor_sz(p->sz);
	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);
	hc2rp = (p->kind[0] == HC2R);

	nbuf = fftw_nbuf(n, vl, rdft_maxnbufs[ego->maxnbuf_ndx]);
	bufdist = fftw_bufdist(n, vl);
	A(nbuf > 0);

	/* initial allocation for the purpose of planning */
	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist, BUFFERS);

	if (hc2rp) {
		/* allow destruction of buffer */
		cld = fftw_mkplan_f_d(plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(n, 1, p->sz->dims[0].os),
				fftw_mktensor_1d(nbuf, bufdist, ovs),
				bufs, TAINT(p->O, ovs * nbuf), p->kind),
			0, 0, NO_DESTROY_INPUT);
		if (!cld) goto nada;

		/* copying input into buffer buffer is a rank-0 transform: */
		cldcpy = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_2d(nbuf, ivs, bufdist,
					n, p->sz->dims[0].is, 1),
				TAINT(p->I, ivs * nbuf), bufs));
		if (!cldcpy) goto nada;
	}
	else {
		/* allow destruction of input if problem is in place */
		cld = fftw_mkplan_f_d(plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(n, p->sz->dims[0].is, 1),
				fftw_mktensor_1d(nbuf, ivs, bufdist),
				TAINT(p->I, ivs * nbuf), bufs, p->kind),
			0, 0, (p->I == p->O) ? NO_DESTROY_INPUT : 0);
		if (!cld) goto nada;

		/* copying back from the buffer is a rank-0 transform: */
		cldcpy = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_2d(nbuf, bufdist, ovs,
					n, 1, p->sz->dims[0].os),
				bufs, TAINT(p->O, ovs * nbuf)));
		if (!cldcpy) goto nada;
	}

	/* deallocate buffers, let apply() allocate them for real */
	fftw_ifree(bufs);
	bufs = 0;

	/* plan the leftover transforms (cldrest): */
	{
		INT id = ivs * (nbuf * (vl / nbuf));
		INT od = ovs * (nbuf * (vl / nbuf));
		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->I + id, p->O + od, p->kind));
	}
	if (!cldrest) goto nada;

	pln = MKPLAN_RDFT(rdft_buffered_P, &padt, hc2rp ? rdft_buffered_apply_hc2r : rdft_buffered_apply);
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->cldrest = cldrest;
	pln->n = n;
	pln->vl = vl;
	pln->ivs_by_nbuf = ivs * nbuf;
	pln->ovs_by_nbuf = ovs * nbuf;

	pln->nbuf = nbuf;
	pln->bufdist = bufdist;

	{
		opcnt t;
		fftw_ops_add(&cld->ops, &cldcpy->ops, &t);
		fftw_ops_madd(vl / nbuf, &t, &cldrest->ops, &pln->super.super.ops);
	}

	return &(pln->super.super);

nada:
	fftw_ifree0(bufs);
	fftw_plan_destroy_internal(cldrest);
	fftw_plan_destroy_internal(cldcpy);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static solver *rdft_buffered_mksolver(size_t maxnbuf_ndx) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_buffered_mkplan, 0 };
	rdft_buffered_S *slv = MKSOLVER(rdft_buffered_S, &sadt);
	slv->maxnbuf_ndx = maxnbuf_ndx;
	return &(slv->super);
}

void fftw_rdft_buffered_register(planner *p) {
	size_t i;
	for (i = 0; i < NELEM(rdft_maxnbufs); ++i)
		REGISTER_SOLVER(p, rdft_buffered_mksolver(i));
}

/* buffering of rdft2.  We always buffer the complex array */

typedef struct {
	solver super;
	size_t maxnbuf_ndx;
} rdft_buffered2_S;


typedef struct {
	plan_rdft2 super;

	plan *cld, *cldcpy, *cldrest;
	INT n, vl, nbuf, bufdist;
	INT ivs_by_nbuf, ovs_by_nbuf;
	INT ioffset, roffset;
} rdft_buffered2_P;

/* transform a vector input with the help of bufs */
static void
rdft_buffered2_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const rdft_buffered2_P *ego = (const rdft_buffered2_P *)ego_;
	plan_rdft2 *cld = (plan_rdft2 *)ego->cld;
	plan_dft *cldcpy = (plan_dft *)ego->cldcpy;
	INT i, vl = ego->vl, nbuf = ego->nbuf;
	INT ivs_by_nbuf = ego->ivs_by_nbuf, ovs_by_nbuf = ego->ovs_by_nbuf;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * ego->bufdist, BUFFERS);
	FFTW_REAL_TYPE *bufr = bufs + ego->roffset;
	FFTW_REAL_TYPE *bufi = bufs + ego->ioffset;
	plan_rdft2 *cldrest;

	for (i = nbuf; i <= vl; i += nbuf) {
		/* transform to bufs: */
		cld->apply((plan *)cld, r0, r1, bufr, bufi);
		r0 += ivs_by_nbuf;
		r1 += ivs_by_nbuf;

		/* copy back */
		cldcpy->apply((plan *)cldcpy, bufr, bufi, cr, ci);
		cr += ovs_by_nbuf;
		ci += ovs_by_nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft2 *)ego->cldrest;
	cldrest->apply((plan *)cldrest, r0, r1, cr, ci);
}

/* for hc2r problems, copy the input into buffer, and then
transform buffer->output, which allows for destruction of the
buffer */
static void
rdft_buffered2_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const rdft_buffered2_P *ego = (const rdft_buffered2_P *)ego_;
	plan_rdft2 *cld = (plan_rdft2 *)ego->cld;
	plan_dft *cldcpy = (plan_dft *)ego->cldcpy;
	INT i, vl = ego->vl, nbuf = ego->nbuf;
	INT ivs_by_nbuf = ego->ivs_by_nbuf, ovs_by_nbuf = ego->ovs_by_nbuf;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * ego->bufdist, BUFFERS);
	FFTW_REAL_TYPE *bufr = bufs + ego->roffset;
	FFTW_REAL_TYPE *bufi = bufs + ego->ioffset;
	plan_rdft2 *cldrest;

	for (i = nbuf; i <= vl; i += nbuf) {
		/* copy input into bufs: */
		cldcpy->apply((plan *)cldcpy, cr, ci, bufr, bufi);
		cr += ivs_by_nbuf;
		ci += ivs_by_nbuf;

		/* transform to output */
		cld->apply((plan *)cld, r0, r1, bufr, bufi);
		r0 += ovs_by_nbuf;
		r1 += ovs_by_nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft2 *)ego->cldrest;
	cldrest->apply((plan *)cldrest, r0, r1, cr, ci);
}


static void rdft_buffered2_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_buffered2_P *ego = (rdft_buffered2_P *)ego_;

	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldcpy, wakefulness);
	fftw_plan_awake(ego->cldrest, wakefulness);
}

static void rdft_buffered2_destroy(plan *ego_) {
	rdft_buffered2_P *ego = (rdft_buffered2_P *)ego_;
	fftw_plan_destroy_internal(ego->cldrest);
	fftw_plan_destroy_internal(ego->cldcpy);
	fftw_plan_destroy_internal(ego->cld);
}

static void rdft_buffered2_print(const plan *ego_, printer *p) {
	const rdft_buffered2_P *ego = (const rdft_buffered2_P *)ego_;
	p->print(p, "(rdft2-buffered-%D%v/%D-%D%(%p%)%(%p%)%(%p%))",
		ego->n, ego->nbuf,
		ego->vl, ego->bufdist % ego->n,
		ego->cld, ego->cldcpy, ego->cldrest);
}

static int rdft_buffered2_applicable0(const rdft_buffered2_S *ego, const problem *p_, const planner *plnr) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	iodim *d = p->sz->dims;

	if (1
		&& p->vecsz->rnk <= 1
		&& p->sz->rnk == 1

		/* we assume even n throughout */
		&& (d[0].n % 2) == 0

		/* and we only consider these two cases */
		&& (p->kind == R2HC || p->kind == HC2R)

		) {
		INT vl, ivs, ovs;
		fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

		if (fftw_toobig(d[0].n) && CONSERVE_MEMORYP(plnr))
			return 0;

		/* if this solver is redundant, in the sense that a solver
		of lower index generates the same plan, then prune this
		solver */
		if (fftw_nbuf_redundant(d[0].n, vl,
			ego->maxnbuf_ndx,
			rdft_maxnbufs, NELEM(rdft_maxnbufs)))
			return 0;

		if (p->r0 != p->cr) {
			if (p->kind == HC2R) {
				/* Allow HC2R problems only if the input is to be
				preserved.  This solver sets NO_DESTROY_INPUT,
				which prevents infinite loops */
				return (NO_DESTROY_INPUTP(plnr));
			}
			else {
				/*
				In principle, the buffered transforms might be useful
				when working out of place.  However, in order to
				prevent infinite loops in the planner, we require
				that the output stride of the buffered transforms be
				greater than 2.
				*/
				return (d[0].os > 2);
			}
		}

		/*
		* If the problem is in place, the input/output strides must
		* be the same or the whole thing must fit in the buffer.
		*/
		if (fftw_rdft2_inplace_strides(p, RNK_MINFTY))
			return 1;

		if (/* fits into buffer: */
			((p->vecsz->rnk == 0)
				||
				(fftw_nbuf(d[0].n, p->vecsz->dims[0].n,
					rdft_maxnbufs[ego->maxnbuf_ndx])
					== p->vecsz->dims[0].n)))
			return 1;
	}

	return 0;
}

static int rdft_buffered2_applicable(const rdft_buffered2_S *ego, const problem *p_, const planner *plnr) {
	const problem_rdft2 *p;

	if (NO_BUFFERINGP(plnr)) return 0;

	if (!rdft_buffered2_applicable0(ego, p_, plnr)) return 0;

	p = (const problem_rdft2 *)p_;
	if (p->kind == HC2R) {
		if (NO_UGLYP(plnr)) {
			/* UGLY if in-place and too big, since the problem
			could be solved via transpositions */
			if (p->r0 == p->cr && fftw_toobig(p->sz->dims[0].n))
				return 0;
		}
	}
	else {
		if (NO_UGLYP(plnr)) {
			if (p->r0 != p->cr || fftw_toobig(p->sz->dims[0].n))
				return 0;
		}
	}
	return 1;
}

static plan *rdft_buffered2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	rdft_buffered2_P *pln;
	const rdft_buffered2_S *ego = (const rdft_buffered2_S *)ego_;
	plan *cld = (plan *)0;
	plan *cldcpy = (plan *)0;
	plan *cldrest = (plan *)0;
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)0;
	INT nbuf = 0, bufdist, n, vl;
	INT ivs, ovs, ioffset, roffset, id, od;

	static const plan_adt padt = {
		fftw_rdft2_solve, rdft_buffered2_awake, rdft_buffered2_print, rdft_buffered2_destroy
	};

	if (!rdft_buffered2_applicable(ego, p_, plnr))
		goto nada;

	n = fftw_tensor_sz(p->sz);
	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

	nbuf = fftw_nbuf(n, vl, rdft_maxnbufs[ego->maxnbuf_ndx]);
	bufdist = fftw_bufdist(n + 2, vl); /* complex-side rdft2 stores N+2
									   real numbers */
	A(nbuf > 0);

	/* attempt to keep real and imaginary part in the same order,
	so as to allow optimizations in the the copy plan */
	roffset = (p->cr - p->ci > 0) ? (INT)1 : (INT)0;
	ioffset = 1 - roffset;

	/* initial allocation for the purpose of planning */
	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist, BUFFERS);

	id = ivs * (nbuf * (vl / nbuf));
	od = ovs * (nbuf * (vl / nbuf));

	if (p->kind == R2HC) {
		/* allow destruction of input if problem is in place */
		cld = fftw_mkplan_f_d(
			plnr,
			fftw_mkproblem_rdft2_d(
				fftw_mktensor_1d(n, p->sz->dims[0].is, 2),
				fftw_mktensor_1d(nbuf, ivs, bufdist),
				TAINT(p->r0, ivs * nbuf), TAINT(p->r1, ivs * nbuf),
				bufs + roffset, bufs + ioffset, p->kind),
			0, 0, (p->r0 == p->cr) ? NO_DESTROY_INPUT : 0);
		if (!cld) goto nada;

		/* copying back from the buffer is a rank-0 DFT: */
		cldcpy = fftw_mkplan_d(
			plnr,
			fftw_mkproblem_dft_d(
				fftw_mktensor_0d(),
				fftw_mktensor_2d(nbuf, bufdist, ovs,
					n / 2 + 1, 2, p->sz->dims[0].os),
				bufs + roffset, bufs + ioffset,
				TAINT(p->cr, ovs * nbuf), TAINT(p->ci, ovs * nbuf)));
		if (!cldcpy) goto nada;

		fftw_ifree(bufs);
		bufs = 0;

		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft2_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->r0 + id, p->r1 + id,
				p->cr + od, p->ci + od,
				p->kind));
		if (!cldrest) goto nada;
		pln = MKPLAN_RDFT2(rdft_buffered2_P, &padt, rdft_buffered2_apply_r2hc);
	}
	else {
		/* allow destruction of buffer */
		cld = fftw_mkplan_f_d(
			plnr,
			fftw_mkproblem_rdft2_d(
				fftw_mktensor_1d(n, 2, p->sz->dims[0].os),
				fftw_mktensor_1d(nbuf, bufdist, ovs),
				TAINT(p->r0, ovs * nbuf), TAINT(p->r1, ovs * nbuf),
				bufs + roffset, bufs + ioffset, p->kind),
			0, 0, NO_DESTROY_INPUT);
		if (!cld) goto nada;

		/* copying input into buffer is a rank-0 DFT: */
		cldcpy = fftw_mkplan_d(
			plnr,
			fftw_mkproblem_dft_d(
				fftw_mktensor_0d(),
				fftw_mktensor_2d(nbuf, ivs, bufdist,
					n / 2 + 1, p->sz->dims[0].is, 2),
				TAINT(p->cr, ivs * nbuf), TAINT(p->ci, ivs * nbuf),
				bufs + roffset, bufs + ioffset));
		if (!cldcpy) goto nada;

		fftw_ifree(bufs);
		bufs = 0;

		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft2_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->r0 + od, p->r1 + od,
				p->cr + id, p->ci + id,
				p->kind));
		if (!cldrest) goto nada;

		pln = MKPLAN_RDFT2(rdft_buffered2_P, &padt, rdft_buffered2_apply_hc2r);
	}

	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->cldrest = cldrest;
	pln->n = n;
	pln->vl = vl;
	pln->ivs_by_nbuf = ivs * nbuf;
	pln->ovs_by_nbuf = ovs * nbuf;
	pln->roffset = roffset;
	pln->ioffset = ioffset;

	pln->nbuf = nbuf;
	pln->bufdist = bufdist;

	{
		opcnt t;
		fftw_ops_add(&cld->ops, &cldcpy->ops, &t);
		fftw_ops_madd(vl / nbuf, &t, &cldrest->ops, &pln->super.super.ops);
	}

	return &(pln->super.super);

nada:
	fftw_ifree0(bufs);
	fftw_plan_destroy_internal(cldrest);
	fftw_plan_destroy_internal(cldcpy);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static solver *rdft_buffered2_mksolver(size_t maxnbuf_ndx) {
	static const solver_adt sadt = { PROBLEM_RDFT2, rdft_buffered2_mkplan, 0 };
	rdft_buffered2_S *slv = MKSOLVER(rdft_buffered2_S, &sadt);
	slv->maxnbuf_ndx = maxnbuf_ndx;
	return &(slv->super);
}

void fftw_rdft2_buffered_register(planner *p) {
	size_t i;
	for (i = 0; i < NELEM(rdft_maxnbufs); ++i)
		REGISTER_SOLVER(p, rdft_buffered2_mksolver(i));
}

static const solvtab rdft_conf_s =
{
	SOLVTAB(fftw_rdft_indirect_register),
	SOLVTAB(fftw_rdft_rank0_register),
	SOLVTAB(fftw_rdft_vrank3_transpose_register),
	SOLVTAB(fftw_rdft_vrank_geq1_register),

	SOLVTAB(fftw_rdft_nop_register),
	SOLVTAB(fftw_rdft_buffered_register),
	SOLVTAB(fftw_rdft_generic_register),
	SOLVTAB(fftw_rdft_rank_geq2_register),

	SOLVTAB(fftw_dft_r2hc_register),

	SOLVTAB(fftw_rdft_dht_register),
	SOLVTAB(fftw_dht_r2hc_register),
	SOLVTAB(fftw_dht_rader_register),

	SOLVTAB(fftw_rdft2_vrank_geq1_register),
	SOLVTAB(fftw_rdft2_nop_register),
	SOLVTAB(fftw_rdft2_rank0_register),
	SOLVTAB(fftw_rdft2_buffered_register),
	SOLVTAB(fftw_rdft2_rank_geq2_register),
	SOLVTAB(fftw_rdft2_rdft_register),

	SOLVTAB(fftw_hc2hc_generic_register),

	SOLVTAB_END
};

void fftw_rdft_conf_standard(planner *p) {
	fftw_solvtab_exec(rdft_conf_s, p);
	fftw_solvtab_exec(fftw_solvtab_rdft_r2cf, p);
	fftw_solvtab_exec(fftw_solvtab_rdft_r2cb, p);
	fftw_solvtab_exec(fftw_solvtab_rdft_r2r, p);

#if HAVE_SSE2
	if (fftw_have_simd_sse2())
		fftw_solvtab_exec(fftw_solvtab_rdft_sse2, p);
#endif
#if HAVE_AVX
	if (fftw_have_simd_avx())
		fftw_solvtab_exec(fftw_solvtab_rdft_avx, p);
#endif
#if HAVE_AVX_128_FMA
	if (fftw_have_simd_avx_128_fma())
		fftw_solvtab_exec(fftw_solvtab_rdft_avx_128_fma, p);
#endif
#if HAVE_AVX2
	if (fftw_have_simd_avx2())
		fftw_solvtab_exec(fftw_solvtab_rdft_avx2, p);
	if (fftw_have_simd_avx2_128())
		fftw_solvtab_exec(fftw_solvtab_rdft_avx2_128, p);
#endif
#if HAVE_AVX512
	if (fftw_have_simd_avx512())
		fftw_solvtab_exec(fftw_solvtab_rdft_avx512, p);
#endif
#if HAVE_KCVI
	if (fftw_have_simd_kcvi())
		fftw_solvtab_exec(fftw_solvtab_rdft_kcvi, p);
#endif
#if HAVE_ALTIVEC
	if (fftw_have_simd_altivec())
		fftw_solvtab_exec(fftw_solvtab_rdft_altivec, p);
#endif
#if HAVE_VSX
	if (fftw_have_simd_vsx())
		fftw_solvtab_exec(fftw_solvtab_rdft_vsx, p);
#endif
#if HAVE_NEON
	if (fftw_have_simd_neon())
		fftw_solvtab_exec(fftw_solvtab_rdft_neon, p);
#endif
#if HAVE_GENERIC_SIMD128
	fftw_solvtab_exec(fftw_solvtab_rdft_generic_simd128, p);
#endif
#if HAVE_GENERIC_SIMD256
	fftw_solvtab_exec(fftw_solvtab_rdft_generic_simd256, p);
#endif
}

typedef struct {
	plan_rdft2 super;
	plan *cld;
	plan *cldw;
	INT r;
} ct_hc2c_P;

static void
ct_hc2c_apply_dit(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const ct_hc2c_P *ego = (const ct_hc2c_P *)ego_;
	plan_rdft *cld;
	plan_hc2c *cldw;
	UNUSED(r1);

	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, r0, cr);

	cldw = (plan_hc2c *)ego->cldw;
	cldw->apply(ego->cldw, cr, ci);
}

static void
ct_hc2c_apply_dif(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const ct_hc2c_P *ego = (const ct_hc2c_P *)ego_;
	plan_rdft *cld;
	plan_hc2c *cldw;
	UNUSED(r1);

	cldw = (plan_hc2c *)ego->cldw;
	cldw->apply(ego->cldw, cr, ci);

	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, cr, r0);
}

static void ct_hc2c_apply_dit_dft(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const ct_hc2c_P *ego = (const ct_hc2c_P *)ego_;
	plan_dft *cld;
	plan_hc2c *cldw;

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, r0, r1, cr, ci);

	cldw = (plan_hc2c *)ego->cldw;
	cldw->apply(ego->cldw, cr, ci);
}

static void ct_hc2c_apply_dif_dft(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const ct_hc2c_P *ego = (const ct_hc2c_P *)ego_;
	plan_dft *cld;
	plan_hc2c *cldw;

	cldw = (plan_hc2c *)ego->cldw;
	cldw->apply(ego->cldw, cr, ci);

	cld = (plan_dft *)ego->cld;
	cld->apply(ego->cld, ci, cr, r1, r0);
}

static void ct_hc2c_awake(plan *ego_, enum wakefulness wakefulness) {
	ct_hc2c_P *ego = (ct_hc2c_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldw, wakefulness);
}

static void ct_hc2c_destroy(plan *ego_) {
	ct_hc2c_P *ego = (ct_hc2c_P *)ego_;
	fftw_plan_destroy_internal(ego->cldw);
	fftw_plan_destroy_internal(ego->cld);
}

static void ct_hc2c_print(const plan *ego_, printer *p) {
	const ct_hc2c_P *ego = (const ct_hc2c_P *)ego_;
	p->print(p, "(rdft2-ct-%s/%D%(%p%)%(%p%))",
		(ego->super.apply == ct_hc2c_apply_dit ||
			ego->super.apply == ct_hc2c_apply_dit_dft)
		? "dit" : "dif",
		ego->r, ego->cldw, ego->cld);
}

static int ct_hc2c_applicable0(const hc2c_solver *ego, const problem *p_, planner *plnr) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	INT r;

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1

		&& (/* either the problem is R2HC, which is solved by DIT */
		(p->kind == R2HC)
			||
			/* or the problem is HC2R, in which case it is solved
			by DIF, which destroys the input */
			(p->kind == HC2R &&
			(p->r0 == p->cr || !NO_DESTROY_INPUTP(plnr))))

		&& ((r = fftw_choose_radix(ego->r, p->sz->dims[0].n)) > 0)
		&& p->sz->dims[0].n > r);
}

static int ct_hc2c_hc2c_applicable(const hc2c_solver *ego, const problem *p_,
	planner *plnr) {
	const problem_rdft2 *p;

	if (!ct_hc2c_applicable0(ego, p_, plnr))
		return 0;

	p = (const problem_rdft2 *)p_;

	return (0
		|| p->vecsz->rnk == 0
		|| !NO_VRECURSEP(plnr)
		);
}

static plan *ct_hc2c_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const hc2c_solver *ego = (const hc2c_solver *)ego_;
	const problem_rdft2 *p;
	ct_hc2c_P *pln = 0;
	plan *cld = 0, *cldw = 0;
	INT n, r, m, v, ivs, ovs;
	iodim *d;

	static const plan_adt padt = {
		fftw_rdft2_solve, ct_hc2c_awake, ct_hc2c_print, ct_hc2c_destroy
	};

	if (!ct_hc2c_hc2c_applicable(ego, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft2 *)p_;
	d = p->sz->dims;
	n = d[0].n;
	r = fftw_choose_radix(ego->r, n);
	A((r % 2) == 0);
	m = n / r;

	fftw_tensor_tornk1(p->vecsz, &v, &ivs, &ovs);

	switch (p->kind) {
	case R2HC:
		cldw = ego->mkcldw(ego, R2HC,
			r, m * d[0].os,
			m, d[0].os,
			v, ovs,
			p->cr, p->ci, plnr);
		if (!cldw) goto nada;

		switch (ego->hc2ckind) {
		case HC2C_VIA_RDFT:
			cld = fftw_mkplan_d(
				plnr,
				fftw_mkproblem_rdft_1_d(
					fftw_mktensor_1d(m, (r / 2) * d[0].is, d[0].os),
					fftw_mktensor_3d(
						2, p->r1 - p->r0, p->ci - p->cr,
						r / 2, d[0].is, m * d[0].os,
						v, ivs, ovs),
					p->r0, p->cr, R2HC)
			);
			if (!cld) goto nada;

			pln = MKPLAN_RDFT2(ct_hc2c_P, &padt, ct_hc2c_apply_dit);
			break;

		case HC2C_VIA_DFT:
			cld = fftw_mkplan_d(
				plnr,
				fftw_mkproblem_dft_d(
					fftw_mktensor_1d(m, (r / 2) * d[0].is, d[0].os),
					fftw_mktensor_2d(
						r / 2, d[0].is, m * d[0].os,
						v, ivs, ovs),
					p->r0, p->r1, p->cr, p->ci)
			);
			if (!cld) goto nada;

			pln = MKPLAN_RDFT2(ct_hc2c_P, &padt, ct_hc2c_apply_dit_dft);
			break;
		}
		break;

	case HC2R:
		cldw = ego->mkcldw(ego, HC2R,
			r, m * d[0].is,
			m, d[0].is,
			v, ivs,
			p->cr, p->ci, plnr);
		if (!cldw) goto nada;

		switch (ego->hc2ckind) {
		case HC2C_VIA_RDFT:
			cld = fftw_mkplan_d(
				plnr,
				fftw_mkproblem_rdft_1_d(
					fftw_mktensor_1d(m, d[0].is, (r / 2) * d[0].os),
					fftw_mktensor_3d(
						2, p->ci - p->cr, p->r1 - p->r0,
						r / 2, m * d[0].is, d[0].os,
						v, ivs, ovs),
					p->cr, p->r0, HC2R)
			);
			if (!cld) goto nada;

			pln = MKPLAN_RDFT2(ct_hc2c_P, &padt, ct_hc2c_apply_dif);
			break;

		case HC2C_VIA_DFT:
			cld = fftw_mkplan_d(
				plnr,
				fftw_mkproblem_dft_d(
					fftw_mktensor_1d(m, d[0].is, (r / 2) * d[0].os),
					fftw_mktensor_2d(
						r / 2, m * d[0].is, d[0].os,
						v, ivs, ovs),
					p->ci, p->cr, p->r1, p->r0)
			);
			if (!cld) goto nada;

			pln = MKPLAN_RDFT2(ct_hc2c_P, &padt, ct_hc2c_apply_dif_dft);
			break;
		}
		break;

	default:
		A(0);
	}

	pln->cld = cld;
	pln->cldw = cldw;
	pln->r = r;
	fftw_ops_add(&cld->ops, &cldw->ops, &pln->super.super.ops);

	/* inherit could_prune_now_p attribute from cldw */
	pln->super.super.could_prune_now_p = cldw->could_prune_now_p;

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cldw);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

hc2c_solver *fftw_mksolver_hc2c(size_t size, INT r,
	hc2c_kind hc2ckind,
	hc2c_mkinferior mkcldw) {
	static const solver_adt sadt = { PROBLEM_RDFT2, ct_hc2c_mkplan, 0 };
	hc2c_solver *slv = (hc2c_solver *)fftw_mksolver(size, &sadt);
	slv->r = r;
	slv->hc2ckind = hc2ckind;
	slv->mkcldw = mkcldw;
	return slv;
}

plan *fftw_mkplan_hc2c(size_t size, const plan_adt *adt, hc2capply apply) {
	plan_hc2c *ego;

	ego = (plan_hc2c *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}

typedef struct {
	hc2c_solver super;
	const hc2c_desc *desc;
	int bufferedp;
	khc2c k;
} ct_hc2c_direct_S;

typedef struct {
	plan_hc2c super;
	khc2c k;
	plan *cld0, *cldm; /* children for 0th and middle butterflies */
	INT r, m, v, extra_iter;
	INT ms, vs;
	stride rs, brs;
	twid *td;
	const ct_hc2c_direct_S *slv;
} ct_hc2c_direct_P;

/*************************************************************
Nonbuffered code
*************************************************************/
static void ct_hc2c_direct_apply(const plan *ego_, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const ct_hc2c_direct_P *ego = (const ct_hc2c_direct_P *)ego_;
	plan_rdft2 *cld0 = (plan_rdft2 *)ego->cld0;
	plan_rdft2 *cldm = (plan_rdft2 *)ego->cldm;
	INT i, m = ego->m, v = ego->v;
	INT ms = ego->ms, vs = ego->vs;

	for (i = 0; i < v; ++i, cr += vs, ci += vs) {
		cld0->apply((plan *)cld0, cr, ci, cr, ci);
		ego->k(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
			ego->td->W, ego->rs, 1, (m + 1) / 2, ms);
		cldm->apply((plan *)cldm, cr + (m / 2) * ms, ci + (m / 2) * ms,
			cr + (m / 2) * ms, ci + (m / 2) * ms);
	}
}

static void ct_hc2c_direct_apply_extra_iter(const plan *ego_, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const ct_hc2c_direct_P *ego = (const ct_hc2c_direct_P *)ego_;
	plan_rdft2 *cld0 = (plan_rdft2 *)ego->cld0;
	plan_rdft2 *cldm = (plan_rdft2 *)ego->cldm;
	INT i, m = ego->m, v = ego->v;
	INT ms = ego->ms, vs = ego->vs;
	INT mm = (m - 1) / 2;

	for (i = 0; i < v; ++i, cr += vs, ci += vs) {
		cld0->apply((plan *)cld0, cr, ci, cr, ci);

		/* for 4-way SIMD when (m+1)/2-1 is odd: iterate over an
		even vector length MM-1, and then execute the last
		iteration as a 2-vector with vector stride 0.  The
		twiddle factors of the second half of the last iteration
		are bogus, but we only store the results of the first
		half. */
		ego->k(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
			ego->td->W, ego->rs, 1, mm, ms);
		ego->k(cr + mm * ms, ci + mm * ms, cr + (m - mm) * ms, ci + (m - mm) * ms,
			ego->td->W, ego->rs, mm, mm + 2, 0);
		cldm->apply((plan *)cldm, cr + (m / 2) * ms, ci + (m / 2) * ms,
			cr + (m / 2) * ms, ci + (m / 2) * ms);
	}

}

/*************************************************************
Buffered code
*************************************************************/

/* should not be 2^k to avoid associativity conflicts */
static INT ct_hc2c_direct_compute_batchsize(INT radix) {
	/* round up to multiple of 4 */
	radix += 3;
	radix &= -4;

	return (radix + 2);
}

static void
ct_hc2c_direct_dobatch(const ct_hc2c_direct_P *ego, FFTW_REAL_TYPE *Rp, FFTW_REAL_TYPE *Ip, FFTW_REAL_TYPE *Rm,
	FFTW_REAL_TYPE *Im,
	INT mb, INT me, INT extra_iter, FFTW_REAL_TYPE *bufp) {
	INT b = WS(ego->brs, 1);
	INT rs = WS(ego->rs, 1);
	INT ms = ego->ms;
	FFTW_REAL_TYPE *bufm = bufp + b - 2;
	INT n = me - mb;

	fftw_cpy2d_pair_ci(Rp + mb * ms, Ip + mb * ms, bufp, bufp + 1,
		ego->r / 2, rs, b,
		n, ms, 2);
	fftw_cpy2d_pair_ci(Rm - mb * ms, Im - mb * ms, bufm, bufm + 1,
		ego->r / 2, rs, b,
		n, -ms, -2);

	if (extra_iter) {
		/* initialize the extra_iter element to 0.  It would be ok
		to leave it uninitialized, since we transform uninitialized
		data and ignore the result.  However, we want to avoid
		FP exceptions in case somebody is trapping them. */
		A(n < compute_batchsize(ego->r));
		fftw_zero1d_pair(bufp + 2 * n, bufp + 1 + 2 * n, ego->r / 2, b);
		fftw_zero1d_pair(bufm - 2 * n, bufm + 1 - 2 * n, ego->r / 2, b);
	}

	ego->k(bufp, bufp + 1, bufm, bufm + 1, ego->td->W,
		ego->brs, mb, me + extra_iter, 2);
	fftw_cpy2d_pair_co(bufp, bufp + 1, Rp + mb * ms, Ip + mb * ms,
		ego->r / 2, b, rs,
		n, 2, ms);
	fftw_cpy2d_pair_co(bufm, bufm + 1, Rm - mb * ms, Im - mb * ms,
		ego->r / 2, b, rs,
		n, -2, -ms);
}

static void ct_hc2c_direct_apply_buf(const plan *ego_, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const ct_hc2c_direct_P *ego = (const ct_hc2c_direct_P *)ego_;
	plan_rdft2 *cld0 = (plan_rdft2 *)ego->cld0;
	plan_rdft2 *cldm = (plan_rdft2 *)ego->cldm;
	INT i, j, ms = ego->ms, v = ego->v;
	INT batchsz = ct_hc2c_direct_compute_batchsize(ego->r);
	FFTW_REAL_TYPE *buf;
	INT mb = 1, me = (ego->m + 1) / 2;
	size_t bufsz = ego->r * batchsz * 2 * sizeof(FFTW_REAL_TYPE);

	BUF_ALLOC(FFTW_REAL_TYPE *, buf, bufsz);

	for (i = 0; i < v; ++i, cr += ego->vs, ci += ego->vs) {
		FFTW_REAL_TYPE *Rp = cr;
		FFTW_REAL_TYPE *Ip = ci;
		FFTW_REAL_TYPE *Rm = cr + ego->m * ms;
		FFTW_REAL_TYPE *Im = ci + ego->m * ms;

		cld0->apply((plan *)cld0, Rp, Ip, Rp, Ip);

		for (j = mb; j + batchsz < me; j += batchsz)
			ct_hc2c_direct_dobatch(ego, Rp, Ip, Rm, Im, j, j + batchsz, 0, buf);

		ct_hc2c_direct_dobatch(ego, Rp, Ip, Rm, Im, j, me, ego->extra_iter, buf);

		cldm->apply((plan *)cldm,
			Rp + me * ms, Ip + me * ms,
			Rp + me * ms, Ip + me * ms);

	}

	BUF_FREE(buf, bufsz);
}

/*************************************************************
common code
*************************************************************/
static void ct_hc2c_direct_awake(plan *ego_, enum wakefulness wakefulness) {
	ct_hc2c_direct_P *ego = (ct_hc2c_direct_P *)ego_;

	fftw_plan_awake(ego->cld0, wakefulness);
	fftw_plan_awake(ego->cldm, wakefulness);
	fftw_twiddle_awake(wakefulness, &ego->td, ego->slv->desc->tw,
		ego->r * ego->m, ego->r,
		(ego->m - 1) / 2 + ego->extra_iter);
}

static void ct_hc2c_direct_destroy(plan *ego_) {
	ct_hc2c_direct_P *ego = (ct_hc2c_direct_P *)ego_;
	fftw_plan_destroy_internal(ego->cld0);
	fftw_plan_destroy_internal(ego->cldm);
	fftw_stride_destroy(ego->rs);
	fftw_stride_destroy(ego->brs);
}

static void ct_hc2c_direct_print(const plan *ego_, printer *p) {
	const ct_hc2c_direct_P *ego = (const ct_hc2c_direct_P *)ego_;
	const ct_hc2c_direct_S *slv = ego->slv;
	const hc2c_desc *e = slv->desc;

	if (slv->bufferedp)
		p->print(p, "(hc2c-directbuf/%D-%D/%D/%D%v \"%s\"%(%p%)%(%p%))",
			ct_hc2c_direct_compute_batchsize(ego->r),
			ego->r, fftw_twiddle_length(ego->r, e->tw),
			ego->extra_iter, ego->v, e->nam,
			ego->cld0, ego->cldm);
	else
		p->print(p, "(hc2c-direct-%D/%D/%D%v \"%s\"%(%p%)%(%p%))",
			ego->r, fftw_twiddle_length(ego->r, e->tw),
			ego->extra_iter, ego->v, e->nam,
			ego->cld0, ego->cldm);
}

static int ct_hc2c_direct_applicable0(const ct_hc2c_direct_S *ego, rdft_kind kind,
	INT r, INT rs,
	INT m, INT ms,
	INT v, INT vs,
	const FFTW_REAL_TYPE *cr, const FFTW_REAL_TYPE *ci,
	const planner *plnr,
	INT *extra_iter) {
	const hc2c_desc *e = ego->desc;
	UNUSED(v);

	return (
		1
		&& r == e->radix
		&& kind == e->genus->kind

		/* first v-loop iteration */
		&& ((*extra_iter = 0,
			e->genus->okp(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
				rs, 1, (m + 1) / 2, ms, plnr))
			||
			(*extra_iter = 1,
			((e->genus->okp(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
				rs, 1, (m - 1) / 2, ms, plnr))
				&&
				(e->genus->okp(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
					rs, (m - 1) / 2, (m - 1) / 2 + 2, 0, plnr)))))

		/* subsequent v-loop iterations */
		&& (cr += vs, ci += vs, 1)

		&& e->genus->okp(cr + ms, ci + ms, cr + (m - 1) * ms, ci + (m - 1) * ms,
			rs, 1, (m + 1) / 2 - *extra_iter, ms, plnr)
		);
}

static int ct_hc2c_direct_applicable0_buf(const ct_hc2c_direct_S *ego, rdft_kind kind,
	INT r, INT rs,
	INT m, INT ms,
	INT v, INT vs,
	const FFTW_REAL_TYPE *cr, const FFTW_REAL_TYPE *ci,
	const planner *plnr, INT *extra_iter) {
	const hc2c_desc *e = ego->desc;
	INT batchsz, brs;
	UNUSED(v);
	UNUSED(rs);
	UNUSED(ms);
	UNUSED(vs);

	return (
		1
		&& r == e->radix
		&& kind == e->genus->kind

		/* ignore cr, ci, use buffer */
		&& (cr = (const FFTW_REAL_TYPE *)0, ci = cr + 1,
			batchsz = ct_hc2c_direct_compute_batchsize(r),
			brs = 4 * batchsz, 1)

		&& e->genus->okp(cr, ci, cr + brs - 2, ci + brs - 2,
			brs, 1, 1 + batchsz, 2, plnr)

		&& ((*extra_iter = 0,
			e->genus->okp(cr, ci, cr + brs - 2, ci + brs - 2,
				brs, 1, 1 + (((m - 1) / 2) % batchsz), 2, plnr))
			||
			(*extra_iter = 1,
				e->genus->okp(cr, ci, cr + brs - 2, ci + brs - 2,
					brs, 1, 1 + 1 + (((m - 1) / 2) % batchsz), 2, plnr)))

		);
}

static int ct_hc2c_direct_applicable(const ct_hc2c_direct_S *ego, rdft_kind kind,
	INT r, INT rs,
	INT m, INT ms,
	INT v, INT vs,
	FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
	const planner *plnr, INT *extra_iter) {
	if (ego->bufferedp) {
		if (!ct_hc2c_direct_applicable0_buf(ego, kind, r, rs, m, ms, v, vs, cr, ci, plnr,
			extra_iter))
			return 0;
	}
	else {
		if (!ct_hc2c_direct_applicable0(ego, kind, r, rs, m, ms, v, vs, cr, ci, plnr,
			extra_iter))
			return 0;
	}

	if (NO_UGLYP(plnr) && fftw_ct_uglyp((ego->bufferedp ? (INT)512 : (INT)16),
		v, m * r, r))
		return 0;

	return 1;
}

static plan *ct_hc2c_direct_mkcldw(const hc2c_solver *ego_, rdft_kind kind,
	INT r, INT rs,
	INT m, INT ms,
	INT v, INT vs,
	FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
	planner *plnr) {
	const ct_hc2c_direct_S *ego = (const ct_hc2c_direct_S *)ego_;
	ct_hc2c_direct_P *pln;
	const hc2c_desc *e = ego->desc;
	plan *cld0 = 0, *cldm = 0;
	INT imid = (m / 2) * ms;
	INT extra_iter;

	static const plan_adt padt = {
		0, ct_hc2c_direct_awake, ct_hc2c_direct_print, ct_hc2c_direct_destroy
	};

	if (!ct_hc2c_direct_applicable(ego, kind, r, rs, m, ms, v, vs, cr, ci, plnr,
		&extra_iter))
		return (plan *)0;

	cld0 = fftw_mkplan_d(
		plnr,
		fftw_mkproblem_rdft2_d(fftw_mktensor_1d(r, rs, rs),
			fftw_mktensor_0d(),
			TAINT(cr, vs), TAINT(ci, vs),
			TAINT(cr, vs), TAINT(ci, vs),
			kind));
	if (!cld0) goto nada;

	cldm = fftw_mkplan_d(
		plnr,
		fftw_mkproblem_rdft2_d(((m % 2) ?
			fftw_mktensor_0d() : fftw_mktensor_1d(r, rs, rs)),
			fftw_mktensor_0d(),
			TAINT(cr + imid, vs), TAINT(ci + imid, vs),
			TAINT(cr + imid, vs), TAINT(ci + imid, vs),
			kind == R2HC ? R2HCII : HC2RIII));
	if (!cldm) goto nada;

	if (ego->bufferedp)
		pln = MKPLAN_HC2C(ct_hc2c_direct_P, &padt, ct_hc2c_direct_apply_buf);
	else
		pln = MKPLAN_HC2C(ct_hc2c_direct_P, &padt, extra_iter ? ct_hc2c_direct_apply_extra_iter : ct_hc2c_direct_apply);

	pln->k = ego->k;
	pln->td = 0;
	pln->r = r;
	pln->rs = fftw_mkstride(r, rs);
	pln->m = m;
	pln->ms = ms;
	pln->v = v;
	pln->vs = vs;
	pln->slv = ego;
	pln->brs = fftw_mkstride(r, 4 * ct_hc2c_direct_compute_batchsize(r));
	pln->cld0 = cld0;
	pln->cldm = cldm;
	pln->extra_iter = extra_iter;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(v * (((m - 1) / 2) / e->genus->vl),
		&e->ops, &pln->super.super.ops);
	fftw_ops_madd2(v, &cld0->ops, &pln->super.super.ops);
	fftw_ops_madd2(v, &cldm->ops, &pln->super.super.ops);

	if (ego->bufferedp)
		pln->super.super.ops.other += 4 * r * m * v;

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld0);
	fftw_plan_destroy_internal(cldm);
	return 0;
}

static void ct_hc2c_direct_regone(planner *plnr, khc2c codelet,
	const hc2c_desc *desc,
	hc2c_kind hc2ckind,
	int bufferedp) {
	ct_hc2c_direct_S *slv = (ct_hc2c_direct_S *)fftw_mksolver_hc2c(sizeof(ct_hc2c_direct_S), desc->radix,
		hc2ckind, ct_hc2c_direct_mkcldw);
	slv->k = codelet;
	slv->desc = desc;
	slv->bufferedp = bufferedp;
	REGISTER_SOLVER(plnr, &(slv->super.super));
}

void fftw_regsolver_hc2c_direct(planner *plnr, khc2c codelet,
	const hc2c_desc *desc,
	hc2c_kind hc2ckind) {
	ct_hc2c_direct_regone(plnr, codelet, desc, hc2ckind, /* bufferedp */0);
	ct_hc2c_direct_regone(plnr, codelet, desc, hc2ckind, /* bufferedp */1);
}


/* Compute the complex DFT by combining R2HC RDFTs on the real
and imaginary parts.   This could be useful for people just wanting
to link to the real codelets and not the complex ones.  It could
also even be faster than the complex algorithms for split (as opposed
to interleaved) real/imag complex data. */
typedef struct {
	solver super;
} dft_r2hc_S;

typedef struct {
	plan_dft super;
	plan *cld;
	INT ishift, oshift;
	INT os;
	INT n;
} dft_r2hc_P;

static void
dft_r2hc_apply(const plan *ego_, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io) {
	const dft_r2hc_P *ego = (const dft_r2hc_P *)ego_;
	INT n;

	UNUSED(ii);

	{ /* transform vector of real & imag parts: */
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply((plan *)cld, ri + ego->ishift, ro + ego->oshift);
	}

	n = ego->n;
	if (n > 1) {
		INT i, os = ego->os;
		for (i = 1; i < (n + 1) / 2; ++i) {
			E rop, iop, iom, rom;
			rop = ro[os * i];
			iop = io[os * i];
			rom = ro[os * (n - i)];
			iom = io[os * (n - i)];
			ro[os * i] = rop - iom;
			io[os * i] = iop + rom;
			ro[os * (n - i)] = rop + iom;
			io[os * (n - i)] = iop - rom;
		}
	}
}

static void dft_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	dft_r2hc_P *ego = (dft_r2hc_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void dft_r2hc_destroy(plan *ego_) {
	dft_r2hc_P *ego = (dft_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void dft_r2hc_print(const plan *ego_, printer *p) {
	const dft_r2hc_P *ego = (const dft_r2hc_P *)ego_;
	p->print(p, "(dft-r2hc-%D%(%p%))", ego->n, ego->cld);
}


static int dft_r2hc_applicable0(const problem *p_) {
	const problem_dft *p = (const problem_dft *)p_;
	return ((p->sz->rnk == 1 && p->vecsz->rnk == 0)
		|| (p->sz->rnk == 0 && FINITE_RNK(p->vecsz->rnk))
		);
}

static int dft_r2hc_splitp(FFTW_REAL_TYPE *r, FFTW_REAL_TYPE *i, INT n, INT s) {
	return ((r > i ? (r - i) : (i - r)) >= n * (s > 0 ? s : 0 - s));
}

static int dft_r2hc_applicable(const problem *p_, const planner *plnr) {
	if (!dft_r2hc_applicable0(p_)) return 0;

	{
		const problem_dft *p = (const problem_dft *)p_;

		/* rank-0 problems are always OK */
		if (p->sz->rnk == 0) return 1;

		/* this solver is ok for split arrays */
		if (p->sz->rnk == 1 &&
			dft_r2hc_splitp(p->ri, p->ii, p->sz->dims[0].n, p->sz->dims[0].is) &&
			dft_r2hc_splitp(p->ro, p->io, p->sz->dims[0].n, p->sz->dims[0].os))
			return 1;

		return !(NO_DFT_R2HCP(plnr));
	}
}

static plan *dft_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	dft_r2hc_P *pln;
	const problem_dft *p;
	plan *cld;
	INT ishift = 0, oshift = 0;

	static const plan_adt padt = {
		fftw_dft_solve, dft_r2hc_awake, dft_r2hc_print, dft_r2hc_destroy
	};

	UNUSED(ego_);
	if (!dft_r2hc_applicable(p_, plnr))
		return (plan *)0;

	p = (const problem_dft *)p_;

	{
		tensor *ri_vec = fftw_mktensor_1d(2, p->ii - p->ri, p->io - p->ro);
		tensor *cld_vec = fftw_tensor_append(ri_vec, p->vecsz);
		int i;
		for (i = 0; i < cld_vec->rnk; ++i) { /* make all istrides > 0 */
			if (cld_vec->dims[i].is < 0) {
				INT nm1 = cld_vec->dims[i].n - 1;
				ishift -= nm1 * (cld_vec->dims[i].is *= -1);
				oshift -= nm1 * (cld_vec->dims[i].os *= -1);
			}
		}
		cld = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_1(p->sz, cld_vec,
				p->ri + ishift,
				p->ro + oshift, R2HC));
		fftw_tensor_destroy2(ri_vec, cld_vec);
	}
	if (!cld) return (plan *)0;

	pln = MKPLAN_DFT(dft_r2hc_P, &padt, dft_r2hc_apply);

	if (p->sz->rnk == 0) {
		pln->n = 1;
		pln->os = 0;
	}
	else {
		pln->n = p->sz->dims[0].n;
		pln->os = p->sz->dims[0].os;
	}
	pln->ishift = ishift;
	pln->oshift = oshift;

	pln->cld = cld;

	pln->super.super.ops = cld->ops;
	pln->super.super.ops.other += 8 * ((pln->n - 1) / 2);
	pln->super.super.ops.add += 4 * ((pln->n - 1) / 2);
	pln->super.super.ops.other += 1; /* estimator hack for nop plans */

	return &(pln->super.super);
}

/* constructor */
static solver *dft_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_DFT, dft_r2hc_mkplan, 0 };
	dft_r2hc_S *slv = MKSOLVER(dft_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_dft_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, dft_r2hc_mksolver());
}

/* Solve a DHT problem (Discrete Hartley Transform) via post-processing
of an R2HC problem. */


typedef struct {
	solver super;
} dht_r2hc_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	INT os;
	INT n;
} dht_r2hc_P;

static void dht_r2hc_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const dht_r2hc_P *ego = (const dht_r2hc_P *)ego_;
	INT os = ego->os;
	INT i, n = ego->n;

	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply((plan *)cld, I, O);
	}

	for (i = 1; i < n - i; ++i) {
		E a, b;
		a = O[os * i];
		b = O[os * (n - i)];
#if FFT_SIGN == -1
		O[os * i] = a - b;
		O[os * (n - i)] = a + b;
#else
		O[os * i] = a + b;
		O[os * (n - i)] = a - b;
#endif
	}
}

static void dht_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	dht_r2hc_P *ego = (dht_r2hc_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void dht_r2hc_destroy(plan *ego_) {
	dht_r2hc_P *ego = (dht_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void dht_r2hc_print(const plan *ego_, printer *p) {
	const dht_r2hc_P *ego = (const dht_r2hc_P *)ego_;
	p->print(p, "(dht-r2hc-%D%(%p%))", ego->n, ego->cld);
}

static int dht_r2hc_applicable0(const problem *p_, const planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	return (1
		&& !NO_DHT_R2HCP(plnr)
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& p->kind[0] == DHT
		);
}

static int dht_r2hc_applicable(const solver *ego, const problem *p, const planner *plnr) {
	UNUSED(ego);
	return (!NO_SLOWP(plnr) && dht_r2hc_applicable0(p, plnr));
}

static plan *dht_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	dht_r2hc_P *pln;
	const problem_rdft *p;
	plan *cld;

	static const plan_adt padt = {
		fftw_rdft_solve, dht_r2hc_awake, dht_r2hc_print, dht_r2hc_destroy
	};

	if (!dht_r2hc_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	/* NO_DHT_R2HC stops infinite loops with rdft-dht.c */
	cld = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_rdft_1(p->sz, p->vecsz,
			p->I, p->O, R2HC),
		NO_DHT_R2HC, 0, 0);
	if (!cld) return (plan *)0;

	pln = MKPLAN_RDFT(dht_r2hc_P, &padt, dht_r2hc_apply);

	pln->n = p->sz->dims[0].n;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;

	pln->super.super.ops = cld->ops;
	pln->super.super.ops.other += 4 * ((pln->n - 1) / 2);
	pln->super.super.ops.add += 2 * ((pln->n - 1) / 2);

	return &(pln->super.super);
}

/* constructor */
static solver *dht_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, dht_r2hc_mkplan, 0 };
	dht_r2hc_S *slv = MKSOLVER(dht_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_dht_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, dht_r2hc_mksolver());
}

typedef struct {
	solver super;
} rdft2_rdft_S;

typedef struct {
	plan_rdft2 super;

	plan *cld, *cldrest;
	INT n, vl, nbuf, bufdist;
	INT cs, ivs, ovs;
} rdft2_rdft_P;

/***************************************************************************/

/* FIXME: have alternate copy functions that push a vector loop inside
the n loops? */

/* copy halfcomplex array r (contiguous) to complex (strided) array rio/iio. */
static void rdft2_rdft_hc2c(INT n, FFTW_REAL_TYPE *r, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio, INT os) {
	INT i;

	rio[0] = r[0];
	iio[0] = 0;

	for (i = 1; i + i < n; ++i) {
		rio[i * os] = r[i];
		iio[i * os] = r[n - i];
	}

	if (i + i == n) {    /* store the Nyquist frequency */
		rio[i * os] = r[i];
		iio[i * os] = K(0.0);
	}
}

/* reverse of hc2c */
static void rdft2_rdft_c2hc(INT n, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio, INT is, FFTW_REAL_TYPE *r) {
	INT i;

	r[0] = rio[0];

	for (i = 1; i + i < n; ++i) {
		r[i] = rio[i * is];
		r[n - i] = iio[i * is];
	}

	if (i + i == n)        /* store the Nyquist frequency */
		r[i] = rio[i * is];
}

/***************************************************************************/

static void rdft2_rdft_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rdft2_rdft_P *ego = (const rdft2_rdft_P *)ego_;
	plan_rdft *cld = (plan_rdft *)ego->cld;
	INT i, j, vl = ego->vl, nbuf = ego->nbuf, bufdist = ego->bufdist;
	INT n = ego->n;
	INT ivs = ego->ivs, ovs = ego->ovs, os = ego->cs;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist, BUFFERS);
	plan_rdft2 *cldrest;

	for (i = nbuf; i <= vl; i += nbuf) {
		/* transform to bufs: */
		cld->apply((plan *)cld, r0, bufs);
		r0 += ivs * nbuf;
		r1 += ivs * nbuf;

		/* copy back */
		for (j = 0; j < nbuf; ++j, cr += ovs, ci += ovs)
			rdft2_rdft_hc2c(n, bufs + j * bufdist, cr, ci, os);
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft2 *)ego->cldrest;
	cldrest->apply((plan *)cldrest, r0, r1, cr, ci);
}

static void rdft2_rdft_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rdft2_rdft_P *ego = (const rdft2_rdft_P *)ego_;
	plan_rdft *cld = (plan_rdft *)ego->cld;
	INT i, j, vl = ego->vl, nbuf = ego->nbuf, bufdist = ego->bufdist;
	INT n = ego->n;
	INT ivs = ego->ivs, ovs = ego->ovs, is = ego->cs;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist, BUFFERS);
	plan_rdft2 *cldrest;

	for (i = nbuf; i <= vl; i += nbuf) {
		/* copy to bufs */
		for (j = 0; j < nbuf; ++j, cr += ivs, ci += ivs)
			rdft2_rdft_c2hc(n, cr, ci, is, bufs + j * bufdist);

		/* transform back: */
		cld->apply((plan *)cld, bufs, r0);
		r0 += ovs * nbuf;
		r1 += ovs * nbuf;
	}

	fftw_ifree(bufs);

	/* Do the remaining transforms, if any: */
	cldrest = (plan_rdft2 *)ego->cldrest;
	cldrest->apply((plan *)cldrest, r0, r1, cr, ci);
}

static void rdft2_rdft_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft2_rdft_P *ego = (rdft2_rdft_P *)ego_;

	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldrest, wakefulness);
}

static void rdft2_rdft_destroy(plan *ego_) {
	rdft2_rdft_P *ego = (rdft2_rdft_P *)ego_;
	fftw_plan_destroy_internal(ego->cldrest);
	fftw_plan_destroy_internal(ego->cld);
}

static void rdft2_rdft_print(const plan *ego_, printer *p) {
	const rdft2_rdft_P *ego = (const rdft2_rdft_P *)ego_;
	p->print(p, "(rdft2-rdft-%s-%D%v/%D-%D%(%p%)%(%p%))",
		ego->super.apply == rdft2_rdft_apply_r2hc ? "r2hc" : "hc2r",
		ego->n, ego->nbuf,
		ego->vl, ego->bufdist % ego->n,
		ego->cld, ego->cldrest);
}

static INT rdft2_rdft_min_nbuf(const problem_rdft2 *p, INT n, INT vl) {
	INT is, os, ivs, ovs;

	if (p->r0 != p->cr)
		return 1;
	if (fftw_rdft2_inplace_strides(p, RNK_MINFTY))
		return 1;
	A(p->vecsz->rnk == 1); /*  rank 0 and MINFTY are inplace */

	fftw_rdft2_strides(p->kind, p->sz->dims, &is, &os);
	fftw_rdft2_strides(p->kind, p->vecsz->dims, &ivs, &ovs);

	/* handle one potentially common case: "contiguous" real and
	complex arrays, which overlap because of the differing sizes. */
	if (n * fftw_iabs(is) <= fftw_iabs(ivs)
		&& (n / 2 + 1) * fftw_iabs(os) <= fftw_iabs(ovs)
		&& (((p->cr - p->ci) <= fftw_iabs(os)) ||
		((p->ci - p->cr) <= fftw_iabs(os)))
		&& ivs > 0 && ovs > 0) {
		INT vsmin = fftw_imin(ivs, ovs);
		INT vsmax = fftw_imax(ivs, ovs);
		return (((vsmax - vsmin) * vl + vsmin - 1) / vsmin);
	}

	return vl; /* punt: just buffer the whole vector */
}

static int rdft2_rdft_applicable0(const problem *p_, const rdft2_rdft_S *ego, const planner *plnr) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	UNUSED(ego);
	return (1
		&& p->vecsz->rnk <= 1
		&& p->sz->rnk == 1

		/* FIXME: does it make sense to do R2HCII ? */
		&& (p->kind == R2HC || p->kind == HC2R)

		/* real strides must allow for reduction to rdft */
		&& (2 * (p->r1 - p->r0) ==
		(((p->kind == R2HC) ? p->sz->dims[0].is : p->sz->dims[0].os)))

		&& !(fftw_toobig(p->sz->dims[0].n) && CONSERVE_MEMORYP(plnr))
		);
}

static int rdft2_rdft_applicable(const problem *p_, const rdft2_rdft_S *ego, const planner *plnr) {
	const problem_rdft2 *p;

	if (NO_BUFFERINGP(plnr)) return 0;

	if (!rdft2_rdft_applicable0(p_, ego, plnr)) return 0;

	p = (const problem_rdft2 *)p_;
	if (NO_UGLYP(plnr)) {
		if (p->r0 != p->cr) return 0;
		if (fftw_toobig(p->sz->dims[0].n)) return 0;
	}
	return 1;
}

static plan *rdft2_rdft_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rdft2_rdft_S *ego = (const rdft2_rdft_S *)ego_;
	rdft2_rdft_P *pln;
	plan *cld = (plan *)0;
	plan *cldrest = (plan *)0;
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	FFTW_REAL_TYPE *bufs = (FFTW_REAL_TYPE *)0;
	INT nbuf = 0, bufdist, n, vl;
	INT ivs, ovs, rs, id, od;

	static const plan_adt padt = {
		fftw_rdft2_solve, rdft2_rdft_awake, rdft2_rdft_print, rdft2_rdft_destroy
	};

	if (!rdft2_rdft_applicable(p_, ego, plnr))
		goto nada;

	n = p->sz->dims[0].n;
	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);

	nbuf = fftw_imax(fftw_nbuf(n, vl, 0), rdft2_rdft_min_nbuf(p, n, vl));
	bufdist = fftw_bufdist(n, vl);
	A(nbuf > 0);

	/* initial allocation for the purpose of planning */
	bufs = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * nbuf * bufdist, BUFFERS);

	id = ivs * (nbuf * (vl / nbuf));
	od = ovs * (nbuf * (vl / nbuf));

	if (p->kind == R2HC) {
		cld = fftw_mkplan_f_d(
			plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(n, p->sz->dims[0].is / 2, 1),
				fftw_mktensor_1d(nbuf, ivs, bufdist),
				TAINT(p->r0, ivs * nbuf), bufs, &p->kind),
			0, 0, (p->r0 == p->cr) ? NO_DESTROY_INPUT : 0);
		if (!cld) goto nada;
		fftw_ifree(bufs);
		bufs = 0;

		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft2_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->r0 + id, p->r1 + id,
				p->cr + od, p->ci + od,
				p->kind));
		if (!cldrest) goto nada;

		pln = MKPLAN_RDFT2(rdft2_rdft_P, &padt, rdft2_rdft_apply_r2hc);
	}
	else {
		A(p->kind == HC2R);
		cld = fftw_mkplan_f_d(
			plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(n, 1, p->sz->dims[0].os / 2),
				fftw_mktensor_1d(nbuf, bufdist, ovs),
				bufs, TAINT(p->r0, ovs * nbuf), &p->kind),
			0, 0, NO_DESTROY_INPUT); /* always ok to destroy bufs */
		if (!cld) goto nada;
		fftw_ifree(bufs);
		bufs = 0;

		cldrest = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft2_d(
				fftw_tensor_copy(p->sz),
				fftw_mktensor_1d(vl % nbuf, ivs, ovs),
				p->r0 + od, p->r1 + od,
				p->cr + id, p->ci + id,
				p->kind));
		if (!cldrest) goto nada;
		pln = MKPLAN_RDFT2(rdft2_rdft_P, &padt, rdft2_rdft_apply_hc2r);
	}

	pln->cld = cld;
	pln->cldrest = cldrest;
	pln->n = n;
	pln->vl = vl;
	pln->ivs = ivs;
	pln->ovs = ovs;
	fftw_rdft2_strides(p->kind, &p->sz->dims[0], &rs, &pln->cs);
	pln->nbuf = nbuf;
	pln->bufdist = bufdist;

	fftw_ops_madd(vl / nbuf, &cld->ops, &cldrest->ops,
		&pln->super.super.ops);
	pln->super.super.ops.other += (p->kind == R2HC ? (n + 2) : n) * vl;

	return &(pln->super.super);

nada:
	fftw_ifree0(bufs);
	fftw_plan_destroy_internal(cldrest);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

static solver *rdft2_rdft_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT2, rdft2_rdft_mkplan, 0 };
	rdft2_rdft_S *slv = MKSOLVER(rdft2_rdft_S, &sadt);
	return &(slv->super);
}

void fftw_rdft2_rdft_register(planner *p) {
	REGISTER_SOLVER(p, rdft2_rdft_mksolver());
}

/*
* Compute DHTs of prime sizes using Rader's trick: turn them
* into convolutions of size n - 1, which we then perform via a pair
* of FFTs.   (We can then do prime real FFTs via rdft-dht.c.)
*
* Optionally (determined by the "pad" field of the solver), we can
* perform the (cyclic) convolution by zero-padding to a size
* >= 2*(n-1) - 1.  This is advantageous if n-1 has large prime factors.
*
*/

typedef struct {
	solver super;
	int pad;
} dht_rader_S;

typedef struct {
	plan_rdft super;

	plan *cld1, *cld2;
	FFTW_REAL_TYPE *omega;
	INT n, npad, g, ginv;
	INT is, os;
	plan *cld_omega;
} dht_rader_P;

static rader_tl *dht_rader_omegas = 0;

/***************************************************************************/

/* If R2HC_ONLY_CONV is 1, we use a trick to perform the convolution
purely in terms of R2HC transforms, as opposed to R2HC followed by H2RC.
This requires a few more operations, but allows us to share the same
plan/codelets for both Rader children. */
#define R2HC_ONLY_CONV 1

static void dht_rader_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const dht_rader_P *ego = (const dht_rader_P *)ego_;
	INT n = ego->n; /* prime */
	INT npad = ego->npad; /* == n - 1 for unpadded Rader; always even */
	INT is = ego->is, os;
	INT k, gpower, g;
	FFTW_REAL_TYPE *buf, *omega;
	FFTW_REAL_TYPE r0;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * npad, BUFFERS);

	/* First, permute the input, storing in buf: */
	g = ego->g;
	for (gpower = 1, k = 0; k < n - 1; ++k, gpower = MULMOD(gpower, g, n)) {
		buf[k] = I[gpower * is];
	}
	/* gpower == g^(n-1) mod n == 1 */;

	A(n - 1 <= npad);
	for (k = n - 1; k < npad; ++k) /* optionally, zero-pad convolution */
		buf[k] = 0;

	os = ego->os;

	/* compute RDFT of buf, storing in buf (i.e., in-place): */
	{
		plan_rdft *cld = (plan_rdft *)ego->cld1;
		cld->apply((plan *)cld, buf, buf);
	}

	/* set output DC component: */
	O[0] = (r0 = I[0]) + buf[0];

	/* now, multiply by omega: */
	omega = ego->omega;
	buf[0] *= omega[0];
	for (k = 1; k < npad / 2; ++k) {
		E rB, iB, rW, iW, a, b;
		rW = omega[k];
		iW = omega[npad - k];
		rB = buf[k];
		iB = buf[npad - k];
		a = rW * rB - iW * iB;
		b = rW * iB + iW * rB;
#if R2HC_ONLY_CONV
		buf[k] = a + b;
		buf[npad - k] = a - b;
#else
		buf[k] = a;
		buf[npad - k] = b;
#endif
	}
	/* Nyquist component: */
	A(k + k == npad); /* since npad is even */
	buf[k] *= omega[k];

	/* this will add input[0] to all of the outputs after the ifft */
	buf[0] += r0;

	/* inverse FFT: */
	{
		plan_rdft *cld = (plan_rdft *)ego->cld2;
		cld->apply((plan *)cld, buf, buf);
	}

	/* do inverse permutation to unshuffle the output: */
	A(gpower == 1);
#if R2HC_ONLY_CONV
	O[os] = buf[0];
	gpower = g = ego->ginv;
	A(npad == n - 1 || npad / 2 >= n - 1);
	if (npad == n - 1) {
		for (k = 1; k < npad / 2; ++k, gpower = MULMOD(gpower, g, n)) {
			O[gpower * os] = buf[k] + buf[npad - k];
		}
		O[gpower * os] = buf[k];
		++k, gpower = MULMOD(gpower, g, n);
		for (; k < npad; ++k, gpower = MULMOD(gpower, g, n)) {
			O[gpower * os] = buf[npad - k] - buf[k];
		}
	}
	else {
		for (k = 1; k < n - 1; ++k, gpower = MULMOD(gpower, g, n)) {
			O[gpower * os] = buf[k] + buf[npad - k];
		}
	}
#else
	g = ego->ginv;
	for (k = 0; k < n - 1; ++k, gpower = MULMOD(gpower, g, n)) {
		O[gpower * os] = buf[k];
	}
#endif
	A(gpower == 1);

	fftw_ifree(buf);
}

static FFTW_REAL_TYPE *dht_rader_mkomega(enum wakefulness wakefulness,
	plan *p_, INT n, INT npad, INT ginv) {
	plan_rdft *p = (plan_rdft *)p_;
	FFTW_REAL_TYPE *omega;
	INT i, gpower;
	trigreal scale;
	triggen *t;

	if ((omega = fftw_rader_tl_find(n, npad + 1, ginv, dht_rader_omegas)))
		return omega;

	omega = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * npad, TWIDDLES);

	scale = npad; /* normalization for convolution */

	t = fftw_mktriggen(wakefulness, n);
	for (i = 0, gpower = 1; i < n - 1; ++i, gpower = MULMOD(gpower, ginv, n)) {
		trigreal w[2];
		t->cexpl(t, gpower, w);
		omega[i] = (w[0] + w[1]) / scale;
	}
	fftw_triggen_destroy(t);
	A(gpower == 1);

	A(npad == n - 1 || npad >= 2 * (n - 1) - 1);

	for (; i < npad; ++i)
		omega[i] = K(0.0);
	if (npad > n - 1)
		for (i = 1; i < n - 1; ++i)
			omega[npad - i] = omega[n - 1 - i];

	p->apply(p_, omega, omega);

	fftw_rader_tl_insert(n, npad + 1, ginv, omega, &dht_rader_omegas);
	return omega;
}

static void dht_rader_free_omega(FFTW_REAL_TYPE *omega) {
	fftw_rader_tl_delete(omega, &dht_rader_omegas);
}

/***************************************************************************/

static void dht_rader_awake(plan *ego_, enum wakefulness wakefulness) {
	dht_rader_P *ego = (dht_rader_P *)ego_;

	fftw_plan_awake(ego->cld1, wakefulness);
	fftw_plan_awake(ego->cld2, wakefulness);
	fftw_plan_awake(ego->cld_omega, wakefulness);

	switch (wakefulness) {
	case SLEEPY:
		dht_rader_free_omega(ego->omega);
		ego->omega = 0;
		break;
	default:
		ego->g = fftw_find_generator(ego->n);
		ego->ginv = fftw_power_mod(ego->g, ego->n - 2, ego->n);
		A(MULMOD(ego->g, ego->ginv, ego->n) == 1);

		A(!ego->omega);
		ego->omega = dht_rader_mkomega(wakefulness,
			ego->cld_omega, ego->n, ego->npad, ego->ginv);
		break;
	}
}

static void dht_rader_destroy(plan *ego_) {
	dht_rader_P *ego = (dht_rader_P *)ego_;
	fftw_plan_destroy_internal(ego->cld_omega);
	fftw_plan_destroy_internal(ego->cld2);
	fftw_plan_destroy_internal(ego->cld1);
}

static void dht_rader_print(const plan *ego_, printer *p) {
	const dht_rader_P *ego = (const dht_rader_P *)ego_;

	p->print(p, "(dht-rader-%D/%D%ois=%oos=%(%p%)",
		ego->n, ego->npad, ego->is, ego->os, ego->cld1);
	if (ego->cld2 != ego->cld1)
		p->print(p, "%(%p%)", ego->cld2);
	if (ego->cld_omega != ego->cld1 && ego->cld_omega != ego->cld2)
		p->print(p, "%(%p%)", ego->cld_omega);
	p->putchr(p, ')');
}

static int dht_rader_applicable(const solver *ego, const problem *p_, const planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego);
	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& p->kind[0] == DHT
		&& fftw_is_prime(p->sz->dims[0].n)
		&& p->sz->dims[0].n > 2
		&& CIMPLIES(NO_SLOWP(plnr), p->sz->dims[0].n > RADER_MAX_SLOW)
		/* proclaim the solver SLOW if p-1 is not easily
		factorizable.  Unlike in the complex case where
		Bluestein can solve the problem, in the DHT case we
		may have no other choice */
		&& CIMPLIES(NO_SLOWP(plnr), fftw_factors_into_small_primes(p->sz->dims[0].n - 1))
		);
}

static INT dht_rader_choose_transform_size(INT minsz) {
	static const INT primes[] = { 2, 3, 5, 0 };
	while (!fftw_factors_into(minsz, primes) || minsz % 2)
		++minsz;
	return minsz;
}

static plan *dht_rader_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const dht_rader_S *ego = (const dht_rader_S *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;
	dht_rader_P *pln;
	INT n, npad;
	INT is, os;
	plan *cld1 = (plan *)0;
	plan *cld2 = (plan *)0;
	plan *cld_omega = (plan *)0;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)0;
	problem *cldp;

	static const plan_adt padt = {
		fftw_rdft_solve, dht_rader_awake, dht_rader_print, dht_rader_destroy
	};

	if (!dht_rader_applicable(ego_, p_, plnr))
		return (plan *)0;

	n = p->sz->dims[0].n;
	is = p->sz->dims[0].is;
	os = p->sz->dims[0].os;

	if (ego->pad)
		npad = dht_rader_choose_transform_size(2 * (n - 1) - 1);
	else
		npad = n - 1;

	/* initial allocation for the purpose of planning */
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * npad, BUFFERS);

	cld1 = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(npad, 1, 1),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf,
			R2HC),
		NO_SLOW, 0, 0);
	if (!cld1) goto nada;

	cldp =
		fftw_mkproblem_rdft_1_d(
			fftw_mktensor_1d(npad, 1, 1),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf,
#if R2HC_ONLY_CONV
			R2HC
#else
			HC2R
#endif
		);
	if (!(cld2 = fftw_mkplan_f_d(plnr, cldp, NO_SLOW, 0, 0)))
		goto nada;

	/* plan for omega */
	cld_omega = fftw_mkplan_f_d(plnr,
		fftw_mkproblem_rdft_1_d(
			fftw_mktensor_1d(npad, 1, 1),
			fftw_mktensor_1d(1, 0, 0),
			buf, buf, R2HC),
		NO_SLOW, ESTIMATE, 0);
	if (!cld_omega) goto nada;

	/* deallocate buffers; let awake() or apply() allocate them for real */
	fftw_ifree(buf);
	buf = 0;

	pln = MKPLAN_RDFT(dht_rader_P, &padt, dht_rader_apply);
	pln->cld1 = cld1;
	pln->cld2 = cld2;
	pln->cld_omega = cld_omega;
	pln->omega = 0;
	pln->n = n;
	pln->npad = npad;
	pln->is = is;
	pln->os = os;

	fftw_ops_add(&cld1->ops, &cld2->ops, &pln->super.super.ops);
	pln->super.super.ops.other += (npad / 2 - 1) * 6 + npad + n + (n - 1) * ego->pad;
	pln->super.super.ops.add += (npad / 2 - 1) * 2 + 2 + (n - 1) * ego->pad;
	pln->super.super.ops.mul += (npad / 2 - 1) * 4 + 2 + ego->pad;
#if R2HC_ONLY_CONV
	pln->super.super.ops.other += n - 2 - ego->pad;
	pln->super.super.ops.add += (npad / 2 - 1) * 2 + (n - 2) - ego->pad;
#endif

	return &(pln->super.super);

nada:
	fftw_ifree0(buf);
	fftw_plan_destroy_internal(cld_omega);
	fftw_plan_destroy_internal(cld2);
	fftw_plan_destroy_internal(cld1);
	return 0;
}

/* constructors */

static solver *dht_rader_mksolver(int pad) {
	static const solver_adt sadt = { PROBLEM_RDFT, dht_rader_mkplan, 0 };
	dht_rader_S *slv = MKSOLVER(dht_rader_S, &sadt);
	slv->pad = pad;
	return &(slv->super);
}

void fftw_dht_rader_register(planner *p) {
	REGISTER_SOLVER(p, dht_rader_mksolver(0));
	REGISTER_SOLVER(p, dht_rader_mksolver(1));
}

/* direct RDFT2 R2HC/HC2R solver, if we have a codelet */


typedef struct {
	solver super;
	const kr2c_desc *desc;
	kr2c k;
} direct2_S;

typedef struct {
	plan_rdft2 super;

	stride rs, cs;
	INT vl;
	INT ivs, ovs;
	kr2c k;
	const direct2_S *slv;
	INT ilast;
} direct2_P;

static void
direct2_apply(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const direct2_P *ego = (const direct2_P *)ego_;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(r0, r1, cr, ci,
		ego->rs, ego->cs, ego->cs,
		ego->vl, ego->ivs, ego->ovs);
}

static void
direct2_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	const direct2_P *ego = (const direct2_P *)ego_;
	INT i, vl = ego->vl, ovs = ego->ovs;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(r0, r1, cr, ci,
		ego->rs, ego->cs, ego->cs,
		vl, ego->ivs, ovs);
	for (i = 0; i < vl; ++i, ci += ovs)
		ci[0] = ci[ego->ilast] = 0;
}

static void direct2_destroy(plan *ego_) {
	direct2_P *ego = (direct2_P *)ego_;
	fftw_stride_destroy(ego->rs);
	fftw_stride_destroy(ego->cs);
}

static void direct2_print(const plan *ego_, printer *p) {
	const direct2_P *ego = (const direct2_P *)ego_;
	const direct2_S *s = ego->slv;

	p->print(p, "(rdft2-%s-direct-%D%v \"%s\")",
		fftw_rdft_kind_str(s->desc->genus->kind), s->desc->n,
		ego->vl, s->desc->nam);
}

static int direct2_applicable(const solver *ego_, const problem *p_) {
	const direct2_S *ego = (const direct2_S *)ego_;
	const kr2c_desc *desc = ego->desc;
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	INT vl;
	INT ivs, ovs;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n == desc->n
		&& p->kind == desc->genus->kind

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		&& (0
			/* can operate out-of-place */
			|| p->r0 != p->cr

			/*
			* can compute one transform in-place, no matter
			* what the strides are.
			*/
			|| p->vecsz->rnk == 0

			/* can operate in-place as long as strides are the same */
			|| fftw_rdft2_inplace_strides(p, RNK_MINFTY)
			)
		);
}

static plan *direct2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const direct2_S *ego = (const direct2_S *)ego_;
	direct2_P *pln;
	const problem_rdft2 *p;
	iodim *d;
	int r2hc_kindp;

	static const plan_adt padt = {
		fftw_rdft2_solve, fftw_null_awake, direct2_print, direct2_destroy
	};

	UNUSED(plnr);

	if (!direct2_applicable(ego_, p_))
		return (plan *)0;

	p = (const problem_rdft2 *)p_;

	r2hc_kindp = R2HC_KINDP(p->kind);
	A(r2hc_kindp || HC2R_KINDP(p->kind));

	pln = MKPLAN_RDFT2(direct2_P, &padt, p->kind == R2HC ? direct2_apply_r2hc : direct2_apply);

	d = p->sz->dims;

	pln->k = ego->k;

	pln->rs = fftw_mkstride(d->n, r2hc_kindp ? d->is : d->os);
	pln->cs = fftw_mkstride(d->n, r2hc_kindp ? d->os : d->is);

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	/* Nyquist freq., if any */
	pln->ilast = (d->n % 2) ? 0 : (d->n / 2) * d->os;

	pln->slv = ego;
	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl / ego->desc->genus->vl,
		&ego->desc->ops,
		&pln->super.super.ops);
	if (p->kind == R2HC)
		pln->super.super.ops.other += 2 * pln->vl; /* + 2 stores */

	pln->super.super.could_prune_now_p = 1;
	return &(pln->super.super);
}

/* constructor */
solver *fftw_mksolver_rdft2_direct(kr2c k, const kr2c_desc *desc) {
	static const solver_adt sadt = { PROBLEM_RDFT2, direct2_mkplan, 0 };
	direct2_S *slv = MKSOLVER(direct2_S, &sadt);
	slv->k = k;
	slv->desc = desc;
	return &(slv->super);
}

/* direct RDFT solver, using r2c codelets */


typedef struct {
	solver super;
	const kr2c_desc *desc;
	kr2c k;
	int bufferedp;
} direct2_r2c_S;

typedef struct {
	plan_rdft super;

	stride rs, csr, csi;
	stride brs, bcsr, bcsi;
	INT n, vl, rs0, ivs, ovs, ioffset, bioffset;
	kr2c k;
	const direct2_r2c_S *slv;
} direct2_r2c_P;

/*************************************************************
Nonbuffered code
*************************************************************/
static void direct2_r2c_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const direct2_r2c_P *ego = (const direct2_r2c_P *)ego_;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(I, I + ego->rs0, O, O + ego->ioffset,
		ego->rs, ego->csr, ego->csi,
		ego->vl, ego->ivs, ego->ovs);
}

static void direct2_r2c_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const direct2_r2c_P *ego = (const direct2_r2c_P *)ego_;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(O, O + ego->rs0, I, I + ego->ioffset,
		ego->rs, ego->csr, ego->csi,
		ego->vl, ego->ivs, ego->ovs);
}

/*************************************************************
Buffered code
*************************************************************/
/* should not be 2^k to avoid associativity conflicts */
static INT direct2_r2c_compute_batchsize(INT radix) {
	/* round up to multiple of 4 */
	radix += 3;
	radix &= -4;

	return (radix + 2);
}

static void
direct2_r2c_dobatch_r2hc(const direct2_r2c_P *ego, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, FFTW_REAL_TYPE *buf,
	INT batchsz) {
	fftw_cpy2d_ci(I, buf,
		ego->n, ego->rs0, WS(ego->bcsr /* hack */, 1),
		batchsz, ego->ivs, 1, 1);

	if (IABS(WS(ego->csr, 1)) < IABS(ego->ovs)) {
		/* transform directly to output */
		ego->k(buf, buf + WS(ego->bcsr /* hack */, 1),
			O, O + ego->ioffset,
			ego->brs, ego->csr, ego->csi,
			batchsz, 1, ego->ovs);
	}
	else {
		/* transform to buffer and copy back */
		ego->k(buf, buf + WS(ego->bcsr /* hack */, 1),
			buf, buf + ego->bioffset,
			ego->brs, ego->bcsr, ego->bcsi,
			batchsz, 1, 1);
		fftw_cpy2d_co(buf, O,
			ego->n, WS(ego->bcsr, 1), WS(ego->csr, 1),
			batchsz, 1, ego->ovs, 1);
	}
}

static void
direct2_r2c_dobatch_hc2r(const direct2_r2c_P *ego, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, FFTW_REAL_TYPE *buf,
	INT batchsz) {
	if (IABS(WS(ego->csr, 1)) < IABS(ego->ivs)) {
		/* transform directly from input */
		ego->k(buf, buf + WS(ego->bcsr /* hack */, 1),
			I, I + ego->ioffset,
			ego->brs, ego->csr, ego->csi,
			batchsz, ego->ivs, 1);
	}
	else {
		/* copy into buffer and transform in place */
		fftw_cpy2d_ci(I, buf,
			ego->n, WS(ego->csr, 1), WS(ego->bcsr, 1),
			batchsz, ego->ivs, 1, 1);
		ego->k(buf, buf + WS(ego->bcsr /* hack */, 1),
			buf, buf + ego->bioffset,
			ego->brs, ego->bcsr, ego->bcsi,
			batchsz, 1, 1);
	}
	fftw_cpy2d_co(buf, O,
		ego->n, WS(ego->bcsr /* hack */, 1), ego->rs0,
		batchsz, 1, ego->ovs, 1);
}

static void direct2_r2c_iterate(const direct2_r2c_P *ego, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	void(*dobatch)(const direct2_r2c_P *ego, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		FFTW_REAL_TYPE *buf, INT batchsz)) {
	FFTW_REAL_TYPE *buf;
	INT vl = ego->vl;
	INT n = ego->n;
	INT i;
	INT batchsz = direct2_r2c_compute_batchsize(n);
	size_t bufsz = n * batchsz * sizeof(FFTW_REAL_TYPE);

	BUF_ALLOC(FFTW_REAL_TYPE *, buf, bufsz);

	for (i = 0; i < vl - batchsz; i += batchsz) {
		dobatch(ego, I, O, buf, batchsz);
		I += batchsz * ego->ivs;
		O += batchsz * ego->ovs;
	}
	dobatch(ego, I, O, buf, vl - i);

	BUF_FREE(buf, bufsz);
}

static void direct2_r2c_apply_buf_r2hc(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	direct2_r2c_iterate((const direct2_r2c_P *)ego_, I, O, direct2_r2c_dobatch_r2hc);
}

static void direct2_r2c_apply_buf_hc2r(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	direct2_r2c_iterate((const direct2_r2c_P *)ego_, I, O, direct2_r2c_dobatch_hc2r);
}

static void direct2_r2c_destroy(plan *ego_) {
	direct2_r2c_P *ego = (direct2_r2c_P *)ego_;
	fftw_stride_destroy(ego->rs);
	fftw_stride_destroy(ego->csr);
	fftw_stride_destroy(ego->csi);
	fftw_stride_destroy(ego->brs);
	fftw_stride_destroy(ego->bcsr);
	fftw_stride_destroy(ego->bcsi);
}

static void direct2_r2c_print(const plan *ego_, printer *p) {
	const direct2_r2c_P *ego = (const direct2_r2c_P *)ego_;
	const direct2_r2c_S *s = ego->slv;

	if (ego->slv->bufferedp)
		p->print(p, "(rdft-%s-directbuf/%D-r2c-%D%v \"%s\")",
			fftw_rdft_kind_str(s->desc->genus->kind),
			/* hack */ WS(ego->bcsr, 1), ego->n,
			ego->vl, s->desc->nam);

	else
		p->print(p, "(rdft-%s-direct-r2c-%D%v \"%s\")",
			fftw_rdft_kind_str(s->desc->genus->kind), ego->n,
			ego->vl, s->desc->nam);
}

static INT direct2_r2c_ioffset(rdft_kind kind, INT sz, INT s) {
	return (s * ((kind == R2HC || kind == HC2R) ? sz : (sz - 1)));
}

static int direct2_r2c_applicable(const solver *ego_, const problem *p_) {
	const direct2_r2c_S *ego = (const direct2_r2c_S *)ego_;
	const kr2c_desc *desc = ego->desc;
	const problem_rdft *p = (const problem_rdft *)p_;
	INT vl, ivs, ovs;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n == desc->n
		&& p->kind[0] == desc->genus->kind

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		&& (0
			/* can operate out-of-place */
			|| p->I != p->O

			/* computing one transform */
			|| vl == 1

			/* can operate in-place as long as strides are the same */
			|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)
			)
		);
}

static int direct2_r2c_applicable_buf(const solver *ego_, const problem *p_) {
	const direct2_r2c_S *ego = (const direct2_r2c_S *)ego_;
	const kr2c_desc *desc = ego->desc;
	const problem_rdft *p = (const problem_rdft *)p_;
	INT vl, ivs, ovs, batchsz;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n == desc->n
		&& p->kind[0] == desc->genus->kind

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		&& (batchsz = direct2_r2c_compute_batchsize(desc->n), 1)

		&& (0
			/* can operate out-of-place */
			|| p->I != p->O

			/* can operate in-place as long as strides are the same */
			|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)

			/* can do it if the problem fits in the buffer, no matter
			what the strides are */
			|| vl <= batchsz
			)
		);
}

static plan *direct2_r2c_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const direct2_r2c_S *ego = (const direct2_r2c_S *)ego_;
	direct2_r2c_P *pln;
	const problem_rdft *p;
	iodim *d;
	INT rs, cs, b, n;

	static const plan_adt padt = {
		fftw_rdft_solve, fftw_null_awake, direct2_r2c_print, direct2_r2c_destroy
	};

	UNUSED(plnr);

	if (ego->bufferedp) {
		if (!direct2_r2c_applicable_buf(ego_, p_))
			return (plan *)0;
	}
	else {
		if (!direct2_r2c_applicable(ego_, p_))
			return (plan *)0;
	}

	p = (const problem_rdft *)p_;

	if (R2HC_KINDP(p->kind[0])) {
		rs = p->sz->dims[0].is;
		cs = p->sz->dims[0].os;
		pln = MKPLAN_RDFT(direct2_r2c_P, &padt,
			ego->bufferedp ? direct2_r2c_apply_buf_r2hc : direct2_r2c_apply_r2hc);
	}
	else {
		rs = p->sz->dims[0].os;
		cs = p->sz->dims[0].is;
		pln = MKPLAN_RDFT(direct2_r2c_P, &padt,
			ego->bufferedp ? direct2_r2c_apply_buf_hc2r : direct2_r2c_apply_hc2r);
	}

	d = p->sz->dims;
	n = d[0].n;

	pln->k = ego->k;
	pln->n = n;

	pln->rs0 = rs;
	pln->rs = fftw_mkstride(n, 2 * rs);
	pln->csr = fftw_mkstride(n, cs);
	pln->csi = fftw_mkstride(n, -cs);
	pln->ioffset = direct2_r2c_ioffset(p->kind[0], n, cs);

	b = direct2_r2c_compute_batchsize(n);
	pln->brs = fftw_mkstride(n, 2 * b);
	pln->bcsr = fftw_mkstride(n, b);
	pln->bcsi = fftw_mkstride(n, -b);
	pln->bioffset = direct2_r2c_ioffset(p->kind[0], n, b);

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	pln->slv = ego;
	fftw_ops_zero(&pln->super.super.ops);

	fftw_ops_madd2(pln->vl / ego->desc->genus->vl,
		&ego->desc->ops,
		&pln->super.super.ops);

	if (ego->bufferedp)
		pln->super.super.ops.other += 2 * n * pln->vl;

	pln->super.super.could_prune_now_p = !ego->bufferedp;

	return &(pln->super.super);
}

/* constructor */
static solver *direct2_r2c_mksolver(kr2c k, const kr2c_desc *desc, int bufferedp) {
	static const solver_adt sadt = { PROBLEM_RDFT, direct2_r2c_mkplan, 0 };
	direct2_r2c_S *slv = MKSOLVER(direct2_r2c_S, &sadt);
	slv->k = k;
	slv->desc = desc;
	slv->bufferedp = bufferedp;
	return &(slv->super);
}

solver *fftw_mksolver_rdft_r2c_direct(kr2c k, const kr2c_desc *desc) {
	return direct2_r2c_mksolver(k, desc, 0);
}

solver *fftw_mksolver_rdft_r2c_directbuf(kr2c k, const kr2c_desc *desc) {
	return direct2_r2c_mksolver(k, desc, 1);
}

/* direct RDFT solver, using r2r codelets */


typedef struct {
	solver super;
	const kr2r_desc *desc;
	kr2r k;
} direct2_r2r_S;

typedef struct {
	plan_rdft super;

	INT vl, ivs, ovs;
	stride is, os;
	kr2r k;
	const direct2_r2r_S *slv;
} direct2_r2r_P;

static void direct2_r2r_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const direct2_r2r_P *ego = (const direct2_r2r_P *)ego_;
	ASSERT_ALIGNED_DOUBLE;
	ego->k(I, O, ego->is, ego->os, ego->vl, ego->ivs, ego->ovs);
}

static void direct2_r2r_destroy(plan *ego_) {
	direct2_r2r_P *ego = (direct2_r2r_P *)ego_;
	fftw_stride_destroy(ego->is);
	fftw_stride_destroy(ego->os);
}

static void direct2_r2r_print(const plan *ego_, printer *p) {
	const direct2_r2r_P *ego = (const direct2_r2r_P *)ego_;
	const direct2_r2r_S *s = ego->slv;

	p->print(p, "(rdft-%s-direct-r2r-%D%v \"%s\")",
		fftw_rdft_kind_str(s->desc->kind), s->desc->n,
		ego->vl, s->desc->nam);
}

static int direct2_r2r_applicable(const solver *ego_, const problem *p_) {
	const direct2_r2r_S *ego = (const direct2_r2r_S *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;
	INT vl;
	INT ivs, ovs;

	return (
		1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n == ego->desc->n
		&& p->kind[0] == ego->desc->kind

		/* check strides etc */
		&& fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs)

		&& (0
			/* can operate out-of-place */
			|| p->I != p->O

			/* computing one transform */
			|| vl == 1

			/* can operate in-place as long as strides are the same */
			|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)
			)
		);
}

static plan *direct2_r2r_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const direct2_r2r_S *ego = (const direct2_r2r_S *)ego_;
	direct2_r2r_P *pln;
	const problem_rdft *p;
	iodim *d;

	static const plan_adt padt = {
		fftw_rdft_solve, fftw_null_awake, direct2_r2r_print, direct2_r2r_destroy
	};

	UNUSED(plnr);

	if (!direct2_r2r_applicable(ego_, p_))
		return (plan *)0;

	p = (const problem_rdft *)p_;


	pln = MKPLAN_RDFT(direct2_r2r_P, &padt, direct2_r2r_apply);

	d = p->sz->dims;

	pln->k = ego->k;

	pln->is = fftw_mkstride(d->n, d->is);
	pln->os = fftw_mkstride(d->n, d->os);

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	pln->slv = ego;
	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl / ego->desc->genus->vl,
		&ego->desc->ops,
		&pln->super.super.ops);

	pln->super.super.could_prune_now_p = 1;

	return &(pln->super.super);
}

/* constructor */
solver *fftw_mksolver_rdft_r2r_direct(kr2r k, const kr2r_desc *desc) {
	static const solver_adt sadt = { PROBLEM_RDFT, direct2_r2r_mkplan, 0 };
	direct2_r2r_S *slv = MKSOLVER(direct2_r2r_S, &sadt);
	slv->k = k;
	slv->desc = desc;
	return &(slv->super);
}


typedef struct {
	solver super;
	rdft_kind kind;
} rdft_generic_S;

typedef struct {
	plan_rdft super;
	twid *td;
	INT n, is, os;
	rdft_kind kind;
} rdft_generic_P;

/***************************************************************************/

static void generic_cdot_r2hc(INT n, const E *x, const FFTW_REAL_TYPE *w, FFTW_REAL_TYPE *or0, FFTW_REAL_TYPE *oi1) {
	INT i;

	E rr = x[0], ri = 0;
	x += 1;
	for (i = 1; i + i < n; ++i) {
		rr += x[0] * w[0];
		ri += x[1] * w[1];
		x += 2;
		w += 2;
	}
	*or0 = rr;
	*oi1 = ri;
}

static void generic_hartley_r2hc(INT n, const FFTW_REAL_TYPE *xr, INT xs, E *o, FFTW_REAL_TYPE *pr) {
	INT i;
	E sr;
	o[0] = sr = xr[0];
	o += 1;
	for (i = 1; i + i < n; ++i) {
		FFTW_REAL_TYPE a, b;
		a = xr[i * xs];
		b = xr[(n - i) * xs];
		sr += (o[0] = a + b);
#if FFT_SIGN == -1
		o[1] = b - a;
#else
		o[1] = a - b;
#endif
		o += 2;
	}
	*pr = sr;
}

static void generic_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_generic_P *ego = (const rdft_generic_P *)ego_;
	INT i;
	INT n = ego->n, is = ego->is, os = ego->os;
	const FFTW_REAL_TYPE *W = ego->td->W;
	E *buf;
	size_t bufsz = n * sizeof(E);

	BUF_ALLOC(E *, buf, bufsz);
	generic_hartley_r2hc(n, I, is, buf, O);

	for (i = 1; i + i < n; ++i) {
		generic_cdot_r2hc(n, buf, W, O + i * os, O + (n - i) * os);
		W += n - 1;
	}

	BUF_FREE(buf, bufsz);
}


static void generic_cdot_hc2r(INT n, const E *x, const FFTW_REAL_TYPE *w, FFTW_REAL_TYPE *or0, FFTW_REAL_TYPE *or1) {
	INT i;

	E rr = x[0], ii = 0;
	x += 1;
	for (i = 1; i + i < n; ++i) {
		rr += x[0] * w[0];
		ii += x[1] * w[1];
		x += 2;
		w += 2;
	}
#if FFT_SIGN == -1
	* or0 = rr - ii;
	*or1 = rr + ii;
#else
	*or0 = rr + ii;
	*or1 = rr - ii;
#endif
}

static void generic_hartley_hc2r(INT n, const FFTW_REAL_TYPE *x, INT xs, E *o, FFTW_REAL_TYPE *pr) {
	INT i;
	E sr;

	o[0] = sr = x[0];
	o += 1;
	for (i = 1; i + i < n; ++i) {
		sr += (o[0] = x[i * xs] + x[i * xs]);
		o[1] = x[(n - i) * xs] + x[(n - i) * xs];
		o += 2;
	}
	*pr = sr;
}

static void generic_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_generic_P *ego = (const rdft_generic_P *)ego_;
	INT i;
	INT n = ego->n, is = ego->is, os = ego->os;
	const FFTW_REAL_TYPE *W = ego->td->W;
	E *buf;
	size_t bufsz = n * sizeof(E);

	BUF_ALLOC(E *, buf, bufsz);
	generic_hartley_hc2r(n, I, is, buf, O);

	for (i = 1; i + i < n; ++i) {
		generic_cdot_hc2r(n, buf, W, O + i * os, O + (n - i) * os);
		W += n - 1;
	}

	BUF_FREE(buf, bufsz);
}


/***************************************************************************/

static void rdft_generic_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_generic_P *ego = (rdft_generic_P *)ego_;
	static const tw_instr half_tw[] = {
		{ TW_HALF, 1, 0 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_twiddle_awake(wakefulness, &ego->td, half_tw, ego->n, ego->n,
		(ego->n - 1) / 2);
}

static void rdft_generic_print(const plan *ego_, printer *p) {
	const rdft_generic_P *ego = (const rdft_generic_P *)ego_;

	p->print(p, "(rdft-generic-%s-%D)",
		ego->kind == R2HC ? "r2hc" : "hc2r",
		ego->n);
}

static int rdft_generic_applicable(const rdft_generic_S *ego, const problem *p_,
	const planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& (p->sz->dims[0].n % 2) == 1
		&& CIMPLIES(NO_LARGE_GENERICP(plnr), p->sz->dims[0].n < GENERIC_MIN_BAD)
		&& CIMPLIES(NO_SLOWP(plnr), p->sz->dims[0].n > GENERIC_MAX_SLOW)
		&& fftw_is_prime(p->sz->dims[0].n)
		&& p->kind[0] == ego->kind
		);
}

static plan *rdft_generic_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rdft_generic_S *ego = (const rdft_generic_S *)ego_;
	const problem_rdft *p;
	rdft_generic_P *pln;
	INT n;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_generic_awake, rdft_generic_print, fftw_plan_null_destroy
	};

	if (!rdft_generic_applicable(ego, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;
	pln = MKPLAN_RDFT(rdft_generic_P, &padt,
		R2HC_KINDP(p->kind[0]) ? generic_apply_r2hc : generic_apply_hc2r);

	pln->n = n = p->sz->dims[0].n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->td = 0;
	pln->kind = ego->kind;

	pln->super.super.ops.add = (n - 1) * 2.5;
	pln->super.super.ops.mul = 0;
	pln->super.super.ops.fma = 0.5 * (n - 1) * (n - 1);
#if 0 /* these are nice pipelined sequential loads and should cost nothing */
	pln->super.super.ops.other = (n - 1)*(2 + 1 + (n - 1));  /* approximate */
#endif

	return &(pln->super.super);
}

static solver *rdft_generic_mksolver(rdft_kind kind) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_generic_mkplan, 0 };
	rdft_generic_S *slv = MKSOLVER(rdft_generic_S, &sadt);
	slv->kind = kind;
	return &(slv->super);
}

void fftw_rdft_generic_register(planner *p) {
	REGISTER_SOLVER(p, rdft_generic_mksolver(R2HC));
	REGISTER_SOLVER(p, rdft_generic_mksolver(HC2R));
}

hc2hc_solver *(*fftw_mksolver_hc2hc_hook)(size_t, INT, hc2hc_mkinferior) = 0;

typedef struct {
	plan_rdft super;
	plan *cld;
	plan *cldw;
	INT r;
} hc2hc_P;

static void hc2hc_apply_dit(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const hc2hc_P *ego = (const hc2hc_P *)ego_;
	plan_rdft *cld;
	plan_hc2hc *cldw;

	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, I, O);

	cldw = (plan_hc2hc *)ego->cldw;
	cldw->apply(ego->cldw, O);
}

static void hc2hc_apply_dif(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const hc2hc_P *ego = (const hc2hc_P *)ego_;
	plan_rdft *cld;
	plan_hc2hc *cldw;

	cldw = (plan_hc2hc *)ego->cldw;
	cldw->apply(ego->cldw, I);

	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, I, O);
}

static void hc2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	hc2hc_P *ego = (hc2hc_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldw, wakefulness);
}

static void hc2hc_destroy(plan *ego_) {
	hc2hc_P *ego = (hc2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cldw);
	fftw_plan_destroy_internal(ego->cld);
}

static void hc2hc_print(const plan *ego_, printer *p) {
	const hc2hc_P *ego = (const hc2hc_P *)ego_;
	p->print(p, "(rdft-ct-%s/%D%(%p%)%(%p%))",
		ego->super.apply == hc2hc_apply_dit ? "dit" : "dif",
		ego->r, ego->cldw, ego->cld);
}

static int hc2hc_applicable0(const hc2hc_solver *ego, const problem *p_, planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	INT r;

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1

		&& (/* either the problem is R2HC, which is solved by DIT */
		(p->kind[0] == R2HC)
			||
			/* or the problem is HC2R, in which case it is solved
			by DIF, which destroys the input */
			(p->kind[0] == HC2R &&
			(p->I == p->O || !NO_DESTROY_INPUTP(plnr))))

		&& ((r = fftw_choose_radix(ego->r, p->sz->dims[0].n)) > 0)
		&& p->sz->dims[0].n > r);
}

int fftw_hc2hc_applicable(const hc2hc_solver *ego, const problem *p_, planner *plnr) {
	const problem_rdft *p;

	if (!hc2hc_applicable0(ego, p_, plnr))
		return 0;

	p = (const problem_rdft *)p_;

	return (0
		|| p->vecsz->rnk == 0
		|| !NO_VRECURSEP(plnr)
		);
}

static plan *hc2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const hc2hc_solver *ego = (const hc2hc_solver *)ego_;
	const problem_rdft *p;
	hc2hc_P *pln = 0;
	plan *cld = 0, *cldw = 0;
	INT n, r, m, v, ivs, ovs;
	iodim *d;

	static const plan_adt padt = {
		fftw_rdft_solve, hc2hc_awake, hc2hc_print, hc2hc_destroy
	};

	if (NO_NONTHREADEDP(plnr) || !fftw_hc2hc_applicable(ego, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;
	d = p->sz->dims;
	n = d[0].n;
	r = fftw_choose_radix(ego->r, n);
	m = n / r;

	fftw_tensor_tornk1(p->vecsz, &v, &ivs, &ovs);

	switch (p->kind[0]) {
	case R2HC:
		cldw = ego->mkcldw(ego,
			R2HC, r, m, d[0].os, v, ovs, 0, (m + 2) / 2,
			p->O, plnr);
		if (!cldw) goto nada;

		cld = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(m, r * d[0].is, d[0].os),
				fftw_mktensor_2d(r, d[0].is, m * d[0].os,
					v, ivs, ovs),
				p->I, p->O, p->kind)
		);
		if (!cld) goto nada;

		pln = MKPLAN_RDFT(hc2hc_P, &padt, hc2hc_apply_dit);
		break;

	case HC2R:
		cldw = ego->mkcldw(ego,
			HC2R, r, m, d[0].is, v, ivs, 0, (m + 2) / 2,
			p->I, plnr);
		if (!cldw) goto nada;

		cld = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_d(
				fftw_mktensor_1d(m, d[0].is, r * d[0].os),
				fftw_mktensor_2d(r, m * d[0].is, d[0].os,
					v, ivs, ovs),
				p->I, p->O, p->kind)
		);
		if (!cld) goto nada;

		pln = MKPLAN_RDFT(hc2hc_P, &padt, hc2hc_apply_dif);
		break;

	default:
		A(0);
	}

	pln->cld = cld;
	pln->cldw = cldw;
	pln->r = r;
	fftw_ops_add(&cld->ops, &cldw->ops, &pln->super.super.ops);

	/* inherit could_prune_now_p attribute from cldw */
	pln->super.super.could_prune_now_p = cldw->could_prune_now_p;

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cldw);
	fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

hc2hc_solver *fftw_mksolver_hc2hc(size_t size, INT r, hc2hc_mkinferior mkcldw) {
	static const solver_adt sadt = { PROBLEM_RDFT, hc2hc_mkplan, 0 };
	hc2hc_solver *slv = (hc2hc_solver *)fftw_mksolver(size, &sadt);
	slv->r = r;
	slv->mkcldw = mkcldw;
	return slv;
}

plan *fftw_mkplan_hc2hc(size_t size, const plan_adt *adt, hc2hcapply apply) {
	plan_hc2hc *ego;

	ego = (plan_hc2hc *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}

typedef struct {
	hc2hc_solver super;
	const hc2hc_desc *desc;
	khc2hc k;
	int bufferedp;
} hc2hc_direct_S;

typedef struct {
	plan_hc2hc super;
	khc2hc k;
	plan *cld0, *cldm; /* children for 0th and middle butterflies */
	INT r, m, v;
	INT ms, vs, mb, me;
	stride rs, brs;
	twid *td;
	const hc2hc_direct_S *slv;
} hc2hc_direct_P;

/*************************************************************
Nonbuffered code
*************************************************************/
static void hc2hc_direct_apply(const plan *ego_, FFTW_REAL_TYPE *IO) {
	const hc2hc_direct_P *ego = (const hc2hc_direct_P *)ego_;
	plan_rdft *cld0 = (plan_rdft *)ego->cld0;
	plan_rdft *cldm = (plan_rdft *)ego->cldm;
	INT i, m = ego->m, v = ego->v;
	INT mb = ego->mb, me = ego->me;
	INT ms = ego->ms, vs = ego->vs;

	for (i = 0; i < v; ++i, IO += vs) {
		cld0->apply((plan *)cld0, IO, IO);
		ego->k(IO + ms * mb, IO + (m - mb) * ms,
			ego->td->W, ego->rs, mb, me, ms);
		cldm->apply((plan *)cldm, IO + (m / 2) * ms, IO + (m / 2) * ms);
	}
}

/*************************************************************
Buffered code
*************************************************************/

/* should not be 2^k to avoid associativity conflicts */
static INT hc2hc_direct_compute_batchsize(INT radix) {
	/* round up to multiple of 4 */
	radix += 3;
	radix &= -4;

	return (radix + 2);
}

static void hc2hc_direct_dobatch(const hc2hc_direct_P *ego, FFTW_REAL_TYPE *IOp, FFTW_REAL_TYPE *IOm,
	INT mb, INT me, FFTW_REAL_TYPE *bufp) {
	INT b = WS(ego->brs, 1);
	INT rs = WS(ego->rs, 1);
	INT r = ego->r;
	INT ms = ego->ms;
	FFTW_REAL_TYPE *bufm = bufp + b - 1;

	fftw_cpy2d_ci(IOp + mb * ms, bufp, r, rs, b, me - mb, ms, 1, 1);
	fftw_cpy2d_ci(IOm - mb * ms, bufm, r, rs, b, me - mb, -ms, -1, 1);

	ego->k(bufp, bufm, ego->td->W, ego->brs, mb, me, 1);

	fftw_cpy2d_co(bufp, IOp + mb * ms, r, b, rs, me - mb, 1, ms, 1);
	fftw_cpy2d_co(bufm, IOm - mb * ms, r, b, rs, me - mb, -1, -ms, 1);
}

static void hc2hc_direct_apply_buf(const plan *ego_, FFTW_REAL_TYPE *IO) {
	const hc2hc_direct_P *ego = (const hc2hc_direct_P *)ego_;
	plan_rdft *cld0 = (plan_rdft *)ego->cld0;
	plan_rdft *cldm = (plan_rdft *)ego->cldm;
	INT i, j, m = ego->m, v = ego->v, r = ego->r;
	INT mb = ego->mb, me = ego->me, ms = ego->ms;
	INT batchsz = hc2hc_direct_compute_batchsize(r);
	FFTW_REAL_TYPE *buf;
	size_t bufsz = r * batchsz * 2 * sizeof(FFTW_REAL_TYPE);

	BUF_ALLOC(FFTW_REAL_TYPE *, buf, bufsz);

	for (i = 0; i < v; ++i, IO += ego->vs) {
		FFTW_REAL_TYPE *IOp = IO;
		FFTW_REAL_TYPE *IOm = IO + m * ms;

		cld0->apply((plan *)cld0, IO, IO);

		for (j = mb; j + batchsz < me; j += batchsz)
			hc2hc_direct_dobatch(ego, IOp, IOm, j, j + batchsz, buf);

		hc2hc_direct_dobatch(ego, IOp, IOm, j, me, buf);

		cldm->apply((plan *)cldm, IO + ms * (m / 2), IO + ms * (m / 2));
	}

	BUF_FREE(buf, bufsz);
}

static void hc2hc_direct_awake(plan *ego_, enum wakefulness wakefulness) {
	hc2hc_direct_P *ego = (hc2hc_direct_P *)ego_;

	fftw_plan_awake(ego->cld0, wakefulness);
	fftw_plan_awake(ego->cldm, wakefulness);
	fftw_twiddle_awake(wakefulness, &ego->td, ego->slv->desc->tw,
		ego->r * ego->m, ego->r, (ego->m - 1) / 2);
}

static void hc2hc_direct_destroy(plan *ego_) {
	hc2hc_direct_P *ego = (hc2hc_direct_P *)ego_;
	fftw_plan_destroy_internal(ego->cld0);
	fftw_plan_destroy_internal(ego->cldm);
	fftw_stride_destroy(ego->rs);
	fftw_stride_destroy(ego->brs);
}

static void hc2hc_direct_print(const plan *ego_, printer *p) {
	const hc2hc_direct_P *ego = (const hc2hc_direct_P *)ego_;
	const hc2hc_direct_S *slv = ego->slv;
	const hc2hc_desc *e = slv->desc;
	INT batchsz = hc2hc_direct_compute_batchsize(ego->r);

	if (slv->bufferedp)
		p->print(p, "(hc2hc-directbuf/%D-%D/%D%v \"%s\"%(%p%)%(%p%))",
			batchsz, ego->r, fftw_twiddle_length(ego->r, e->tw),
			ego->v, e->nam, ego->cld0, ego->cldm);
	else
		p->print(p, "(hc2hc-direct-%D/%D%v \"%s\"%(%p%)%(%p%))",
			ego->r, fftw_twiddle_length(ego->r, e->tw), ego->v, e->nam,
			ego->cld0, ego->cldm);
}

static int hc2hc_direct_applicable0(const hc2hc_direct_S *ego, rdft_kind kind, INT r) {
	const hc2hc_desc *e = ego->desc;

	return (1
		&& r == e->radix
		&& kind == e->genus->kind
		);
}

static int hc2hc_direct_applicable(const hc2hc_direct_S *ego, rdft_kind kind, INT r, INT m, INT v,
	const planner *plnr) {
	if (!hc2hc_direct_applicable0(ego, kind, r))
		return 0;

	if (NO_UGLYP(plnr) && fftw_ct_uglyp((ego->bufferedp ? (INT)512 : (INT)16),
		v, m * r, r))
		return 0;

	return 1;
}

#define CLDMP(m, mstart, mcount) (2 * ((mstart) + (mcount)) == (m) + 2)
#define CLD0P(mstart) ((mstart) == 0)

static plan *hc2hc_direct_mkcldw(const hc2hc_solver *ego_,
	rdft_kind kind, INT r, INT m, INT ms, INT v, INT vs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *IO, planner *plnr) {
	const hc2hc_direct_S *ego = (const hc2hc_direct_S *)ego_;
	hc2hc_direct_P *pln;
	const hc2hc_desc *e = ego->desc;
	plan *cld0 = 0, *cldm = 0;
	INT imid = (m / 2) * ms;
	INT rs = m * ms;

	static const plan_adt padt = {
		0, hc2hc_direct_awake, hc2hc_direct_print, hc2hc_direct_destroy
	};

	if (!hc2hc_direct_applicable(ego, kind, r, m, v, plnr))
		return (plan *)0;

	cld0 = fftw_mkplan_d(
		plnr,
		fftw_mkproblem_rdft_1_d((CLD0P(mstart) ?
			fftw_mktensor_1d(r, rs, rs) : fftw_mktensor_0d()),
			fftw_mktensor_0d(),
			TAINT(IO, vs), TAINT(IO, vs),
			kind));
	if (!cld0) goto nada;

	cldm = fftw_mkplan_d(
		plnr,
		fftw_mkproblem_rdft_1_d((CLDMP(m, mstart, mcount) ?
			fftw_mktensor_1d(r, rs, rs) : fftw_mktensor_0d()),
			fftw_mktensor_0d(),
			TAINT(IO + imid, vs), TAINT(IO + imid, vs),
			kind == R2HC ? R2HCII : HC2RIII));
	if (!cldm) goto nada;

	pln = MKPLAN_HC2HC(hc2hc_direct_P, &padt, ego->bufferedp ? hc2hc_direct_apply_buf : hc2hc_direct_apply);

	pln->k = ego->k;
	pln->td = 0;
	pln->r = r;
	pln->rs = fftw_mkstride(r, rs);
	pln->m = m;
	pln->ms = ms;
	pln->v = v;
	pln->vs = vs;
	pln->slv = ego;
	pln->brs = fftw_mkstride(r, 2 * hc2hc_direct_compute_batchsize(r));
	pln->cld0 = cld0;
	pln->cldm = cldm;
	pln->mb = mstart + CLD0P(mstart);
	pln->me = mstart + mcount - CLDMP(m, mstart, mcount);

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(v * ((pln->me - pln->mb) / e->genus->vl),
		&e->ops, &pln->super.super.ops);
	fftw_ops_madd2(v, &cld0->ops, &pln->super.super.ops);
	fftw_ops_madd2(v, &cldm->ops, &pln->super.super.ops);

	if (ego->bufferedp)
		pln->super.super.ops.other += 4 * r * (pln->me - pln->mb) * v;

	pln->super.super.could_prune_now_p =
		(!ego->bufferedp && r >= 5 && r < 64 && m >= r);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld0);
	fftw_plan_destroy_internal(cldm);
	return 0;
}

static void hc2hc_direct_regone(planner *plnr, khc2hc codelet, const hc2hc_desc *desc,
	int bufferedp) {
	hc2hc_direct_S *slv = (hc2hc_direct_S *)fftw_mksolver_hc2hc(sizeof(hc2hc_direct_S), desc->radix,
		hc2hc_direct_mkcldw);
	slv->k = codelet;
	slv->desc = desc;
	slv->bufferedp = bufferedp;
	REGISTER_SOLVER(plnr, &(slv->super.super));
	if (fftw_mksolver_hc2hc_hook) {
		slv = (hc2hc_direct_S *)fftw_mksolver_hc2hc_hook(sizeof(hc2hc_direct_S), desc->radix,
			hc2hc_direct_mkcldw);
		slv->k = codelet;
		slv->desc = desc;
		slv->bufferedp = bufferedp;
		REGISTER_SOLVER(plnr, &(slv->super.super));
	}
}

void fftw_regsolver_hc2hc_direct(planner *plnr, khc2hc codelet,
	const hc2hc_desc *desc) {
	hc2hc_direct_regone(plnr, codelet, desc, /* bufferedp */0);
	hc2hc_direct_regone(plnr, codelet, desc, /* bufferedp */1);
}


/* express a hc2hc problem in terms of rdft + multiplication by
twiddle factors */


typedef hc2hc_solver hc2hc_generic_S;

typedef struct {
	plan_hc2hc super;

	INT r, m, s, vl, vs, mstart1, mcount1;
	plan *cld0;
	plan *cld;
	twid *td;
} hc2hc_generic_P;


/**************************************************************/
static void hc2hc_generic_mktwiddle(hc2hc_generic_P *ego, enum wakefulness wakefulness) {
	static const tw_instr tw[] = { { TW_HALF, 0, 0 },
	{ TW_NEXT, 1, 0 } };

	/* note that R and M are swapped, to allow for sequential
	access both to data and twiddles */
	fftw_twiddle_awake(wakefulness, &ego->td, tw,
		ego->r * ego->m, ego->m, ego->r);
}

static void hc2hc_generic_bytwiddle(const hc2hc_generic_P *ego, FFTW_REAL_TYPE *IO, FFTW_REAL_TYPE sign) {
	INT i, j, k;
	INT r = ego->r, m = ego->m, s = ego->s, vl = ego->vl, vs = ego->vs;
	INT ms = m * s;
	INT mstart1 = ego->mstart1, mcount1 = ego->mcount1;
	INT wrem = 2 * ((m - 1) / 2 - mcount1);

	for (i = 0; i < vl; ++i, IO += vs) {
		const FFTW_REAL_TYPE *W = ego->td->W;

		A(m % 2 == 1);
		for (k = 1, W += (m - 1) + 2 * (mstart1 - 1); k < r; ++k) {
			/* pr := IO + (j + mstart1) * s + k * ms */
			FFTW_REAL_TYPE *pr = IO + mstart1 * s + k * ms;

			/* pi := IO + (m - j - mstart1) * s + k * ms */
			FFTW_REAL_TYPE *pi = IO - mstart1 * s + (k + 1) * ms;

			for (j = 0; j < mcount1; ++j, pr += s, pi -= s) {
				E xr = *pr;
				E xi = *pi;
				E wr = W[0];
				E wi = sign * W[1];
				*pr = xr * wr - xi * wi;
				*pi = xi * wr + xr * wi;
				W += 2;
			}
			W += wrem;
		}
	}
}

static void hc2hc_generic_swapri(FFTW_REAL_TYPE *IO, INT r, INT m, INT s, INT jstart, INT jend) {
	INT k;
	INT ms = m * s;
	INT js = jstart * s;
	for (k = 0; k + k < r; ++k) {
		/* pr := IO + (m - j) * s + k * ms */
		FFTW_REAL_TYPE *pr = IO + (k + 1) * ms - js;
		/* pi := IO + (m - j) * s + (r - 1 - k) * ms */
		FFTW_REAL_TYPE *pi = IO + (r - k) * ms - js;
		INT j;
		for (j = jstart; j < jend; j += 1, pr -= s, pi -= s) {
			FFTW_REAL_TYPE t = *pr;
			*pr = *pi;
			*pi = t;
		}
	}
}

static void hc2hc_generic_reorder_dit(const hc2hc_generic_P *ego, FFTW_REAL_TYPE *IO) {
	INT i, k;
	INT r = ego->r, m = ego->m, s = ego->s, vl = ego->vl, vs = ego->vs;
	INT ms = m * s;
	INT mstart1 = ego->mstart1, mend1 = mstart1 + ego->mcount1;

	for (i = 0; i < vl; ++i, IO += vs) {
		for (k = 1; k + k < r; ++k) {
			FFTW_REAL_TYPE *p0 = IO + k * ms;
			FFTW_REAL_TYPE *p1 = IO + (r - k) * ms;
			INT j;

			for (j = mstart1; j < mend1; ++j) {
				E rp, ip, im, rm;
				rp = p0[j * s];
				im = p1[ms - j * s];
				rm = p1[j * s];
				ip = p0[ms - j * s];
				p0[j * s] = rp - im;
				p1[ms - j * s] = rp + im;
				p1[j * s] = rm - ip;
				p0[ms - j * s] = ip + rm;
			}
		}

		hc2hc_generic_swapri(IO, r, m, s, mstart1, mend1);
	}
}

static void hc2hc_generic_reorder_dif(const hc2hc_generic_P *ego, FFTW_REAL_TYPE *IO) {
	INT i, k;
	INT r = ego->r, m = ego->m, s = ego->s, vl = ego->vl, vs = ego->vs;
	INT ms = m * s;
	INT mstart1 = ego->mstart1, mend1 = mstart1 + ego->mcount1;

	for (i = 0; i < vl; ++i, IO += vs) {
		hc2hc_generic_swapri(IO, r, m, s, mstart1, mend1);

		for (k = 1; k + k < r; ++k) {
			FFTW_REAL_TYPE *p0 = IO + k * ms;
			FFTW_REAL_TYPE *p1 = IO + (r - k) * ms;
			const FFTW_REAL_TYPE half = K(0.5);
			INT j;

			for (j = mstart1; j < mend1; ++j) {
				E rp, ip, im, rm;
				rp = half * p0[j * s];
				im = half * p1[ms - j * s];
				rm = half * p1[j * s];
				ip = half * p0[ms - j * s];
				p0[j * s] = rp + im;
				p1[ms - j * s] = im - rp;
				p1[j * s] = rm + ip;
				p0[ms - j * s] = ip - rm;
			}
		}
	}
}

static int hc2hc_generic_applicable(rdft_kind kind, INT r, INT m, const planner *plnr) {
	return (1
		&& (kind == R2HC || kind == HC2R)
		&& (m % 2)
		&& (r % 2)
		&& !NO_SLOWP(plnr)
		);
}

/**************************************************************/

static void hc2hc_generic_apply_dit(const plan *ego_, FFTW_REAL_TYPE *IO) {
	const hc2hc_generic_P *ego = (const hc2hc_generic_P *)ego_;
	INT start;
	plan_rdft *cld, *cld0;

	hc2hc_generic_bytwiddle(ego, IO, K(-1.0));

	cld0 = (plan_rdft *)ego->cld0;
	cld0->apply(ego->cld0, IO, IO);

	start = ego->mstart1 * ego->s;
	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, IO + start, IO + start);

	hc2hc_generic_reorder_dit(ego, IO);
}

static void hc2hc_generic_apply_dif(const plan *ego_, FFTW_REAL_TYPE *IO) {
	const hc2hc_generic_P *ego = (const hc2hc_generic_P *)ego_;
	INT start;
	plan_rdft *cld, *cld0;

	hc2hc_generic_reorder_dif(ego, IO);

	cld0 = (plan_rdft *)ego->cld0;
	cld0->apply(ego->cld0, IO, IO);

	start = ego->mstart1 * ego->s;
	cld = (plan_rdft *)ego->cld;
	cld->apply(ego->cld, IO + start, IO + start);

	hc2hc_generic_bytwiddle(ego, IO, K(1.0));
}


static void hc2hc_generic_awake(plan *ego_, enum wakefulness wakefulness) {
	hc2hc_generic_P *ego = (hc2hc_generic_P *)ego_;
	fftw_plan_awake(ego->cld0, wakefulness);
	fftw_plan_awake(ego->cld, wakefulness);
	hc2hc_generic_mktwiddle(ego, wakefulness);
}

static void hc2hc_generic_destroy(plan *ego_) {
	hc2hc_generic_P *ego = (hc2hc_generic_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
	fftw_plan_destroy_internal(ego->cld0);
}

static void hc2hc_generic_print(const plan *ego_, printer *p) {
	const hc2hc_generic_P *ego = (const hc2hc_generic_P *)ego_;
	p->print(p, "(hc2hc-generic-%s-%D-%D%v%(%p%)%(%p%))",
		ego->super.apply == hc2hc_generic_apply_dit ? "dit" : "dif",
		ego->r, ego->m, ego->vl, ego->cld0, ego->cld);
}

static plan *hc2hc_generic_mkcldw(const hc2hc_solver *ego_,
	rdft_kind kind, INT r, INT m, INT s, INT vl, INT vs,
	INT mstart, INT mcount,
	FFTW_REAL_TYPE *IO, planner *plnr) {
	hc2hc_generic_P *pln;
	plan *cld0 = 0, *cld = 0;
	INT mstart1, mcount1, mstride;

	static const plan_adt padt = {
		0, hc2hc_generic_awake, hc2hc_generic_print, hc2hc_generic_destroy
	};

	UNUSED(ego_);

	A(mstart >= 0 && mcount > 0 && mstart + mcount <= (m + 2) / 2);

	if (!hc2hc_generic_applicable(kind, r, m, plnr))
		return (plan *)0;

	A(m % 2);
	mstart1 = mstart + (mstart == 0);
	mcount1 = mcount - (mstart == 0);
	mstride = m - (mstart + mcount - 1) - mstart1;

	/* 0th (DC) transform (vl of these), if mstart == 0 */
	cld0 = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_1_d(
			mstart == 0 ? fftw_mktensor_1d(r, m * s, m * s)
			: fftw_mktensor_0d(),
			fftw_mktensor_1d(vl, vs, vs),
			IO, IO, kind)
	);
	if (!cld0) goto nada;

	/* twiddle transforms: there are 2 x mcount1 x vl of these
	(where 2 corresponds to the real and imaginary parts) ...
	the 2 x mcount1 loops are combined if mstart=0 and mcount=(m+2)/2. */
	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_1_d(
			fftw_mktensor_1d(r, m * s, m * s),
			fftw_mktensor_3d(2, mstride * s, mstride * s,
				mcount1, s, s,
				vl, vs, vs),
			IO + s * mstart1, IO + s * mstart1, kind)
	);
	if (!cld) goto nada;

	pln = MKPLAN_HC2HC(hc2hc_generic_P, &padt, (kind == R2HC) ? hc2hc_generic_apply_dit : hc2hc_generic_apply_dif);
	pln->cld = cld;
	pln->cld0 = cld0;
	pln->r = r;
	pln->m = m;
	pln->s = s;
	pln->vl = vl;
	pln->vs = vs;
	pln->td = 0;
	pln->mstart1 = mstart1;
	pln->mcount1 = mcount1;

	{
		double n0 = 0.5 * (r - 1) * (2 * mcount1) * vl;
		pln->super.super.ops = cld->ops;
		pln->super.super.ops.mul += (kind == R2HC ? 5.0 : 7.0) * n0;
		pln->super.super.ops.add += 4.0 * n0;
		pln->super.super.ops.other += 11.0 * n0;
	}
	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld);
	fftw_plan_destroy_internal(cld0);
	return (plan *)0;
}

static void hc2hc_generic_regsolver(planner *plnr, INT r) {
	hc2hc_generic_S *slv = (hc2hc_generic_S *)fftw_mksolver_hc2hc(sizeof(hc2hc_generic_S), r,
		hc2hc_generic_mkcldw);
	REGISTER_SOLVER(plnr, &(slv->super));
	if (fftw_mksolver_hc2hc_hook) {
		slv = (hc2hc_generic_S *)fftw_mksolver_hc2hc_hook(sizeof(hc2hc_generic_S), r, hc2hc_generic_mkcldw);
		REGISTER_SOLVER(plnr, &(slv->super));
	}
}

void fftw_hc2hc_generic_register(planner *p) {
	hc2hc_generic_regsolver(p, 0);
}


/* solvers/plans for vectors of small RDFT's that cannot be done
in-place directly.  Use a rank-0 plan to rearrange the data
before or after the transform.  Can also change an out-of-place
plan into a copy + in-place (where the in-place transform
is e.g. unit stride). */

/* FIXME: merge with rank-geq2.c(?), since this is just a special case
of a rank split where the first/second transform has rank 0. */


typedef problem *(*rdft_mkcld_t)(const problem_rdft *p);

typedef struct {
	rdftapply apply;

	problem *(*rdft_mkcld)(const problem_rdft *p);

	const char *nam;
} rdft_indirect_ndrct_adt;

typedef struct {
	solver super;
	const rdft_indirect_ndrct_adt *adt;
} rdft_indirect_S;

typedef struct {
	plan_rdft super;
	plan *cldcpy, *cld;
	const rdft_indirect_S *slv;
} rdft_indirect_P;

/*-----------------------------------------------------------------------*/
/* first rearrange, then transform */
static void rdft_indirect_apply_before(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_indirect_P *ego = (const rdft_indirect_P *)ego_;

	{
		plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
		cldcpy->apply(ego->cldcpy, I, O);
	}
	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply(ego->cld, O, O);
	}
}

static problem *rdft_indirect_mkcld_before(const problem_rdft *p) {
	return fftw_mkproblem_rdft_d(fftw_tensor_copy_inplace(p->sz, INPLACE_OS),
		fftw_tensor_copy_inplace(p->vecsz, INPLACE_OS),
		p->O, p->O, p->kind);
}

static const rdft_indirect_ndrct_adt adt_before =
{
	rdft_indirect_apply_before, rdft_indirect_mkcld_before, "rdft-indirect-before"
};

/*-----------------------------------------------------------------------*/
/* first transform, then rearrange */

static void rdft_indirect_apply_after(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_indirect_P *ego = (const rdft_indirect_P *)ego_;

	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply(ego->cld, I, I);
	}
	{
		plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
		cldcpy->apply(ego->cldcpy, I, O);
	}
}

static problem *rdft_indirect_mkcld_after(const problem_rdft *p) {
	return fftw_mkproblem_rdft_d(fftw_tensor_copy_inplace(p->sz, INPLACE_IS),
		fftw_tensor_copy_inplace(p->vecsz, INPLACE_IS),
		p->I, p->I, p->kind);
}

static const rdft_indirect_ndrct_adt adt_after =
{
	rdft_indirect_apply_after, rdft_indirect_mkcld_after, "rdft-indirect-after"
};

/*-----------------------------------------------------------------------*/
static void rdft_indirect_destroy(plan *ego_) {
	rdft_indirect_P *ego = (rdft_indirect_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
	fftw_plan_destroy_internal(ego->cldcpy);
}

static void rdft_indirect_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_indirect_P *ego = (rdft_indirect_P *)ego_;
	fftw_plan_awake(ego->cldcpy, wakefulness);
	fftw_plan_awake(ego->cld, wakefulness);
}

static void rdft_indirect_print(const plan *ego_, printer *p) {
	const rdft_indirect_P *ego = (const rdft_indirect_P *)ego_;
	const rdft_indirect_S *s = ego->slv;
	p->print(p, "(%s%(%p%)%(%p%))", s->adt->nam, ego->cld, ego->cldcpy);
}

static int rdft_indirect_applicable0(const solver *ego_, const problem *p_,
	const planner *plnr) {
	const rdft_indirect_S *ego = (const rdft_indirect_S *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;
	return (1
		&& FINITE_RNK(p->vecsz->rnk)

		/* problem must be a nontrivial transform, not just a copy */
		&& p->sz->rnk > 0

		&& (0

			/* problem must be in-place & require some
			rearrangement of the data */
			|| (p->I == p->O
				&& !(fftw_tensor_inplace_strides2(p->sz, p->vecsz)))

			/* or problem must be out of place, transforming
			from stride 1/2 to bigger stride, for apply_after */
			|| (p->I != p->O && ego->adt->apply == rdft_indirect_apply_after
				&& !NO_DESTROY_INPUTP(plnr)
				&& fftw_tensor_min_istride(p->sz) <= 2
				&& fftw_tensor_min_ostride(p->sz) > 2)

			/* or problem must be out of place, transforming
			to stride 1/2 from bigger stride, for apply_before */
			|| (p->I != p->O && ego->adt->apply == rdft_indirect_apply_before
				&& fftw_tensor_min_ostride(p->sz) <= 2
				&& fftw_tensor_min_istride(p->sz) > 2)

			)
		);
}

static int rdft_indirect_applicable(const solver *ego_, const problem *p_,
	const planner *plnr) {
	if (!rdft_indirect_applicable0(ego_, p_, plnr)) return 0;

	if (NO_INDIRECT_OP_P(plnr)) {
		const problem_rdft *p = (const problem_rdft *)p_;
		if (p->I != p->O) return 0;
	}

	return 1;
}

static plan *rdft_indirect_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const problem_rdft *p = (const problem_rdft *)p_;
	const rdft_indirect_S *ego = (const rdft_indirect_S *)ego_;
	rdft_indirect_P *pln;
	plan *cld = 0, *cldcpy = 0;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_indirect_awake, rdft_indirect_print, rdft_indirect_destroy
	};

	if (!rdft_indirect_applicable(ego_, p_, plnr))
		return (plan *)0;

	cldcpy = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_0_d(
			fftw_tensor_append(p->vecsz, p->sz),
			p->I, p->O));
	if (!cldcpy) goto nada;

	cld = fftw_mkplan_f_d(plnr, ego->adt->rdft_mkcld(p), NO_BUFFERING, 0, 0);
	if (!cld) goto nada;

	pln = MKPLAN_RDFT(rdft_indirect_P, &padt, ego->adt->apply);
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->slv = ego;
	fftw_ops_add(&cld->ops, &cldcpy->ops, &pln->super.super.ops);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld);
	fftw_plan_destroy_internal(cldcpy);
	return (plan *)0;
}

static solver *rdft_indirect_mksolver(const rdft_indirect_ndrct_adt *adt) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_indirect_mkplan, 0 };
	rdft_indirect_S *slv = MKSOLVER(rdft_indirect_S, &sadt);
	slv->adt = adt;
	return &(slv->super);
}

void fftw_rdft_indirect_register(planner *p) {
	unsigned i;
	static const rdft_indirect_ndrct_adt *const adts[] = {
		&adt_before, &adt_after
	};

	for (i = 0; i < sizeof(adts) / sizeof(adts[0]); ++i)
		REGISTER_SOLVER(p, rdft_indirect_mksolver(adts[i]));
}

void fftw_khc2c_register(planner *p, khc2c codelet, const hc2c_desc *desc,
	hc2c_kind hc2ckind) {
	fftw_regsolver_hc2c_direct(p, codelet, desc, hc2ckind);
}

void fftw_khc2hc_register(planner *p, khc2hc codelet, const hc2hc_desc *desc) {
	fftw_regsolver_hc2hc_direct(p, codelet, desc);
}

void fftw_kr2c_register(planner *p, kr2c codelet, const kr2c_desc *desc) {
	REGISTER_SOLVER(p, fftw_mksolver_rdft_r2c_direct(codelet, desc));
	REGISTER_SOLVER(p, fftw_mksolver_rdft_r2c_directbuf(codelet, desc));
	REGISTER_SOLVER(p, fftw_mksolver_rdft2_direct(codelet, desc));
}

void fftw_kr2r_register(planner *p, kr2r codelet, const kr2r_desc *desc) {
	REGISTER_SOLVER(p, fftw_mksolver_rdft_r2r_direct(codelet, desc));
}

/* plans for vrank -infty RDFTs (nothing to do) */


static void rdft_nop_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	UNUSED(ego_);
	UNUSED(I);
	UNUSED(O);
}

static int rdft_nop_applicable(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);
	return 0
		/* case 1 : -infty vector rank */
		|| (p->vecsz->rnk == RNK_MINFTY)

		/* case 2 : rank-0 in-place rdft */
		|| (1
			&& p->sz->rnk == 0
			&& FINITE_RNK(p->vecsz->rnk)
			&& p->O == p->I
			&& fftw_tensor_inplace_strides(p->vecsz)
			);
}

static void rdft_nop_print(const plan *ego, printer *p) {
	UNUSED(ego);
	p->print(p, "(rdft-nop)");
}

static plan *rdft_nop_mkplan(const solver *ego, const problem *p, planner *plnr) {
	static const plan_adt padt = {
		fftw_rdft_solve, fftw_null_awake, rdft_nop_print, fftw_plan_null_destroy
	};
	plan_rdft *pln;

	UNUSED(plnr);

	if (!rdft_nop_applicable(ego, p))
		return (plan *)0;
	pln = MKPLAN_RDFT(plan_rdft, &padt, rdft_nop_apply);
	fftw_ops_zero(&pln->super.ops);

	return &(pln->super);
}

static solver *rdft_nop_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_nop_mkplan, 0 };
	return MKSOLVER(solver, &sadt);
}

void fftw_rdft_nop_register(planner *p) {
	REGISTER_SOLVER(p, rdft_nop_mksolver());
}


/* plans for vrank -infty RDFT2s (nothing to do), as well as in-place
rank-0 HC2R.  Note that in-place rank-0 R2HC is *not* a no-op, because
we have to set the imaginary parts of the output to zero. */


static void
nop2_apply(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci) {
	UNUSED(ego_);
	UNUSED(r0);
	UNUSED(r1);
	UNUSED(cr);
	UNUSED(ci);
}

static int nop2_applicable(const solver *ego_, const problem *p_) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	UNUSED(ego_);

	return (0
		/* case 1 : -infty vector rank */
		|| (p->vecsz->rnk == RNK_MINFTY)

		/* case 2 : rank-0 in-place rdft, except that
		R2HC is not a no-op because it sets the imaginary
		part to 0 */
		|| (1
			&& p->kind != R2HC
			&& p->sz->rnk == 0
			&& FINITE_RNK(p->vecsz->rnk)
			&& (p->r0 == p->cr)
			&& fftw_rdft2_inplace_strides(p, RNK_MINFTY)
			));
}

static void nop2_print(const plan *ego, printer *p) {
	UNUSED(ego);
	p->print(p, "(rdft2-nop)");
}

static plan *nop2_mkplan(const solver *ego, const problem *p, planner *plnr) {
	static const plan_adt padt = {
		fftw_rdft2_solve, fftw_null_awake, nop2_print, fftw_plan_null_destroy
	};
	plan_rdft2 *pln;

	UNUSED(plnr);

	if (!nop2_applicable(ego, p))
		return (plan *)0;
	pln = MKPLAN_RDFT2(plan_rdft2, &padt, nop2_apply);
	fftw_ops_zero(&pln->super.ops);

	return &(pln->super);
}

static solver *nop2_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT2, nop2_mkplan, 0 };
	return MKSOLVER(solver, &sadt);
}

void fftw_rdft2_nop_register(planner *p) {
	REGISTER_SOLVER(p, nop2_mksolver());
}

plan *fftw_mkplan_rdft(size_t size, const plan_adt *adt, rdftapply apply) {
	plan_rdft *ego;

	ego = (plan_rdft *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}

plan *fftw_mkplan_rdft2(size_t size, const plan_adt *adt, rdft2apply apply) {
	plan_rdft2 *ego;

	ego = (plan_rdft2 *)fftw_mkplan(size, adt);
	ego->apply = apply;

	return &(ego->super);
}

static void rdft_problem_destroy(problem *ego_) {
	problem_rdft *ego = (problem_rdft *)ego_;
#if !defined(STRUCT_HACK_C99) && !defined(STRUCT_HACK_KR)
	fftw_ifree0(ego->kind);
#endif
	fftw_tensor_destroy2(ego->vecsz, ego->sz);
	fftw_ifree(ego_);
}

static void rdft_problem_kind_hash(md5 *m, const rdft_kind *kind, int rnk) {
	int i;
	for (i = 0; i < rnk; ++i)
		fftw_md5int(m, kind[i]);
}

static void rdft_problem_hash(const problem *p_, md5 *m) {
	const problem_rdft *p = (const problem_rdft *)p_;
	fftw_md5puts(m, "rdft");
	fftw_md5int(m, p->I == p->O);
	rdft_problem_kind_hash(m, p->kind, p->sz->rnk);
	fftw_md5int(m, fftw_ialignment_of(p->I));
	fftw_md5int(m, fftw_ialignment_of(p->O));
	fftw_tensor_md5(m, p->sz);
	fftw_tensor_md5(m, p->vecsz);
}

static void rdft_problem_recur(const iodim *dims, int rnk, FFTW_REAL_TYPE *I) {
	if (rnk == RNK_MINFTY)
		return;
	else if (rnk == 0)
		I[0] = K(0.0);
	else if (rnk > 0) {
		INT i, n = dims[0].n, is = dims[0].is;

		if (rnk == 1) {
			/* this case is redundant but faster */
			for (i = 0; i < n; ++i)
				I[i * is] = K(0.0);
		}
		else {
			for (i = 0; i < n; ++i)
				rdft_problem_recur(dims + 1, rnk - 1, I + i * is);
		}
	}
}

void fftw_rdft_zerotens(tensor *sz, FFTW_REAL_TYPE *I) {
	rdft_problem_recur(sz->dims, sz->rnk, I);
}

#define KSTR_LEN 8

const char *fftw_rdft_kind_str(rdft_kind kind) {
	static const char kstr[][KSTR_LEN] = {
		"r2hc", "r2hc01", "r2hc10", "r2hc11",
		"hc2r", "hc2r01", "hc2r10", "hc2r11",
		"dht",
		"redft00", "redft01", "redft10", "redft11",
		"rodft00", "rodft01", "rodft10", "rodft11"
	};
	A(kind >= 0 && kind < sizeof(kstr) / KSTR_LEN);
	return kstr[kind];
}

static void rdft_problem_print(const problem *ego_, printer *p) {
	const problem_rdft *ego = (const problem_rdft *)ego_;
	int i;
	p->print(p, "(rdft %d %D %T %T",
		fftw_ialignment_of(ego->I),
		(INT)(ego->O - ego->I),
		ego->sz,
		ego->vecsz);
	for (i = 0; i < ego->sz->rnk; ++i)
		p->print(p, " %d", (int)ego->kind[i]);
	p->print(p, ")");
}

static void rdft_problem_zero(const problem *ego_) {
	const problem_rdft *ego = (const problem_rdft *)ego_;
	tensor *sz = fftw_tensor_append(ego->vecsz, ego->sz);
	fftw_rdft_zerotens(sz, UNTAINT(ego->I));
	fftw_tensor_destroy(sz);
}

static const problem_adt rdft_problem_padt =
{
	PROBLEM_RDFT,
	rdft_problem_hash,
	rdft_problem_zero,
	rdft_problem_print,
	rdft_problem_destroy
};

/* Dimensions of size 1 that are not REDFT/RODFT are no-ops and can be
eliminated.  REDFT/RODFT unit dimensions often have factors of 2.0
and suchlike from normalization and phases, although in principle
these constant factors from different dimensions could be combined. */
static int rdft_problem_nontrivial(const iodim *d, rdft_kind kind) {
	return (d->n > 1 || kind == R2HC11 || kind == HC2R11
		|| (REODFT_KINDP(kind) && kind != REDFT01 && kind != RODFT01));
}

problem *fftw_mkproblem_rdft(const tensor *sz, const tensor *vecsz,
	FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, const rdft_kind *kind) {
	problem_rdft *ego;
	int rnk = sz->rnk;
	int i;

	A(fftw_tensor_kosherp(sz));
	A(fftw_tensor_kosherp(vecsz));
	A(FINITE_RNK(sz->rnk));

	if (UNTAINT(I) == UNTAINT(O))
		I = O = JOIN_TAINT(I, O);

	if (I == O && !fftw_tensor_inplace_locations(sz, vecsz))
		return fftw_mkproblem_unsolvable();

	for (i = rnk = 0; i < sz->rnk; ++i) {
		A(sz->dims[i].n > 0);
		if (rdft_problem_nontrivial(sz->dims + i, kind[i]))
			++rnk;
	}

#if defined(STRUCT_HACK_KR)
	ego = (problem_rdft *)fftw_mkproblem(sizeof(problem_rdft)
		+ sizeof(rdft_kind)
		* (rnk > 0 ? rnk - 1u : 0u), &rdft_problem_padt);
#elif defined(STRUCT_HACK_C99)
	ego = (problem_rdft *)fftw_mkproblem(sizeof(problem_rdft)
		+ sizeof(rdft_kind) * (unsigned)rnk, &padt);
#else
	ego = (problem_rdft *)fftw_mkproblem(sizeof(problem_rdft), &padt);
	ego->kind = (rdft_kind *)MALLOC(sizeof(rdft_kind) * (unsigned)rnk, PROBLEMS);
#endif

	/* do compression and sorting as in fftw_tensor_compress, but take
	transform kind into account (sigh) */
	ego->sz = fftw_mktensor(rnk);
	for (i = rnk = 0; i < sz->rnk; ++i) {
		if (rdft_problem_nontrivial(sz->dims + i, kind[i])) {
			ego->kind[rnk] = kind[i];
			ego->sz->dims[rnk++] = sz->dims[i];
		}
	}
	for (i = 0; i + 1 < rnk; ++i) {
		int j;
		for (j = i + 1; j < rnk; ++j)
			if (fftw_dimcmp(ego->sz->dims + i, ego->sz->dims + j) > 0) {
				iodim dswap;
				rdft_kind kswap;
				dswap = ego->sz->dims[i];
				ego->sz->dims[i] = ego->sz->dims[j];
				ego->sz->dims[j] = dswap;
				kswap = ego->kind[i];
				ego->kind[i] = ego->kind[j];
				ego->kind[j] = kswap;
			}
	}

	for (i = 0; i < rnk; ++i)
		if (ego->sz->dims[i].n == 2 && (ego->kind[i] == REDFT00
			|| ego->kind[i] == DHT
			|| ego->kind[i] == HC2R))
			ego->kind[i] = R2HC; /* size-2 transforms are equivalent */

	ego->vecsz = fftw_tensor_compress_contiguous(vecsz);
	ego->I = I;
	ego->O = O;

	A(FINITE_RNK(ego->sz->rnk));

	return &(ego->super);
}

/* Same as fftw_mkproblem_rdft, but also destroy input tensors. */
problem *fftw_mkproblem_rdft_d(tensor *sz, tensor *vecsz,
	FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, const rdft_kind *kind) {
	problem *p = fftw_mkproblem_rdft(sz, vecsz, I, O, kind);
	fftw_tensor_destroy2(vecsz, sz);
	return p;
}

/* As above, but for rnk <= 1 only and takes a rdft_scalar kind parameter */
problem *fftw_mkproblem_rdft_1(const tensor *sz, const tensor *vecsz,
	FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, rdft_kind kind) {
	A(sz->rnk <= 1);
	return fftw_mkproblem_rdft(sz, vecsz, I, O, &kind);
}

problem *fftw_mkproblem_rdft_1_d(tensor *sz, tensor *vecsz,
	FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, rdft_kind kind) {
	A(sz->rnk <= 1);
	return fftw_mkproblem_rdft_d(sz, vecsz, I, O, &kind);
}

/* create a zero-dimensional problem */
problem *fftw_mkproblem_rdft_0_d(tensor *vecsz, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	return fftw_mkproblem_rdft_d(fftw_mktensor_0d(), vecsz, I, O,
		(const rdft_kind *)0);
}

static void problem2_destroy(problem *ego_) {
	problem_rdft2 *ego = (problem_rdft2 *)ego_;
	fftw_tensor_destroy2(ego->vecsz, ego->sz);
	fftw_ifree(ego_);
}

static void problem2_hash(const problem *p_, md5 *m) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	fftw_md5puts(m, "rdft2");
	fftw_md5int(m, p->r0 == p->cr);
	fftw_md5INT(m, p->r1 - p->r0);
	fftw_md5INT(m, p->ci - p->cr);
	fftw_md5int(m, fftw_ialignment_of(p->r0));
	fftw_md5int(m, fftw_ialignment_of(p->r1));
	fftw_md5int(m, fftw_ialignment_of(p->cr));
	fftw_md5int(m, fftw_ialignment_of(p->ci));
	fftw_md5int(m, p->kind);
	fftw_tensor_md5(m, p->sz);
	fftw_tensor_md5(m, p->vecsz);
}

static void problem2_print(const problem *ego_, printer *p) {
	const problem_rdft2 *ego = (const problem_rdft2 *)ego_;
	p->print(p, "(rdft2 %d %d %T %T)",
		(int)(ego->cr == ego->r0),
		(int)(ego->kind),
		ego->sz,
		ego->vecsz);
}

static void problem2_recur(const iodim *dims, int rnk, FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1) {
	if (rnk == RNK_MINFTY)
		return;
	else if (rnk == 0)
		I0[0] = K(0.0);
	else if (rnk > 0) {
		INT i, n = dims[0].n, is = dims[0].is;

		if (rnk == 1) {
			for (i = 0; i < n - 1; i += 2) {
				*I0 = *I1 = K(0.0);
				I0 += is;
				I1 += is;
			}
			if (i < n)
				*I0 = K(0.0);
		}
		else {
			for (i = 0; i < n; ++i)
				problem2_recur(dims + 1, rnk - 1, I0 + i * is, I1 + i * is);
		}
	}
}

static void problem2_vrecur(const iodim *vdims, int vrnk,
	const iodim *dims, int rnk, FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1) {
	if (vrnk == RNK_MINFTY)
		return;
	else if (vrnk == 0)
		problem2_recur(dims, rnk, I0, I1);
	else if (vrnk > 0) {
		INT i, n = vdims[0].n, is = vdims[0].is;

		for (i = 0; i < n; ++i)
			problem2_vrecur(vdims + 1, vrnk - 1,
				dims, rnk, I0 + i * is, I1 + i * is);
	}
}

INT fftw_rdft2_complex_n(INT real_n, rdft_kind kind) {
	switch (kind) {
	case R2HC:
	case HC2R:
		return (real_n / 2) + 1;
	case R2HCII:
	case HC2RIII:
		return (real_n + 1) / 2;
	default:
		/* can't happen */
		A(0);
		return 0;
	}
}

static void problem2_zero(const problem *ego_) {
	const problem_rdft2 *ego = (const problem_rdft2 *)ego_;
	if (R2HC_KINDP(ego->kind)) {
		/* FIXME: can we avoid the double recursion somehow? */
		problem2_vrecur(ego->vecsz->dims, ego->vecsz->rnk,
			ego->sz->dims, ego->sz->rnk,
			UNTAINT(ego->r0), UNTAINT(ego->r1));
	}
	else {
		tensor *sz;
		tensor *sz2 = fftw_tensor_copy(ego->sz);
		int rnk = sz2->rnk;
		if (rnk > 0) /* ~half as many complex outputs */
			sz2->dims[rnk - 1].n =
			fftw_rdft2_complex_n(sz2->dims[rnk - 1].n, ego->kind);
		sz = fftw_tensor_append(ego->vecsz, sz2);
		fftw_tensor_destroy(sz2);
		fftw_dft_zerotens(sz, UNTAINT(ego->cr), UNTAINT(ego->ci));
		fftw_tensor_destroy(sz);
	}
}

static const problem_adt problem2_padt =
{
	PROBLEM_RDFT2,
	problem2_hash,
	problem2_zero,
	problem2_print,
	problem2_destroy
};

problem *fftw_mkproblem_rdft2(const tensor *sz, const tensor *vecsz,
	FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
	rdft_kind kind) {
	problem_rdft2 *ego;

	A(kind == R2HC || kind == R2HCII || kind == HC2R || kind == HC2RIII);
	A(fftw_tensor_kosherp(sz));
	A(fftw_tensor_kosherp(vecsz));
	A(FINITE_RNK(sz->rnk));

	/* require in-place problems to use r0 == cr */
	if (UNTAINT(r0) == UNTAINT(ci))
		return fftw_mkproblem_unsolvable();

	/* FIXME: should check UNTAINT(r1) == UNTAINT(cr) but
	only if odd elements exist, which requires compressing the
	tensors first */

	if (UNTAINT(r0) == UNTAINT(cr))
		r0 = cr = JOIN_TAINT(r0, cr);

	ego = (problem_rdft2 *)fftw_mkproblem(sizeof(problem_rdft2), &problem2_padt);

	if (sz->rnk > 1) { /* have to compress rnk-1 dims separately, ugh */
		tensor *szc = fftw_tensor_copy_except(sz, sz->rnk - 1);
		tensor *szr = fftw_tensor_copy_sub(sz, sz->rnk - 1, 1);
		tensor *szcc = fftw_tensor_compress(szc);
		if (szcc->rnk > 0)
			ego->sz = fftw_tensor_append(szcc, szr);
		else
			ego->sz = fftw_tensor_compress(szr);
		fftw_tensor_destroy2(szc, szr);
		fftw_tensor_destroy(szcc);
	}
	else {
		ego->sz = fftw_tensor_compress(sz);
	}
	ego->vecsz = fftw_tensor_compress_contiguous(vecsz);
	ego->r0 = r0;
	ego->r1 = r1;
	ego->cr = cr;
	ego->ci = ci;
	ego->kind = kind;

	A(FINITE_RNK(ego->sz->rnk));
	return &(ego->super);

}

/* Same as fftw_mkproblem_rdft2, but also destroy input tensors. */
problem *fftw_mkproblem_rdft2_d(tensor *sz, tensor *vecsz,
	FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
	rdft_kind kind) {
	problem *p = fftw_mkproblem_rdft2(sz, vecsz, r0, r1, cr, ci, kind);
	fftw_tensor_destroy2(vecsz, sz);
	return p;
}

/* Same as fftw_mkproblem_rdft2_d, but with only one R pointer.
Used by the API. */
problem *fftw_mkproblem_rdft2_d_3pointers(tensor *sz, tensor *vecsz,
	FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
	rdft_kind kind) {
	problem *p;
	int rnk = sz->rnk;
	FFTW_REAL_TYPE *r1;

	if (rnk == 0)
		r1 = r0;
	else if (R2HC_KINDP(kind)) {
		r1 = r0 + sz->dims[rnk - 1].is;
		sz->dims[rnk - 1].is *= 2;
	}
	else {
		r1 = r0 + sz->dims[rnk - 1].os;
		sz->dims[rnk - 1].os *= 2;
	}

	p = fftw_mkproblem_rdft2(sz, vecsz, r0, r1, cr, ci, kind);
	fftw_tensor_destroy2(vecsz, sz);
	return p;
}


/* plans for rank-0 RDFTs (copy operations) */


#ifdef HAVE_STRING_H

#include <string.h>        /* for memcpy() */

#endif

#define MAXRNK 32 /* FIXME: should malloc() */

typedef struct {
	plan_rdft super;
	INT vl;
	int rnk;
	iodim d[MAXRNK];
	const char *nam;
} rank0_P;

typedef struct {
	solver super;
	rdftapply apply;

	int(*applicable)(const rank0_P *pln, const problem_rdft *p);

	const char *nam;
} rank0_S;

/* copy up to MAXRNK dimensions from problem into plan.  If a
contiguous dimension exists, save its length in pln->vl */
static int rank0_fill_iodim(rank0_P *pln, const problem_rdft *p) {
	int i;
	const tensor *vecsz = p->vecsz;

	pln->vl = 1;
	pln->rnk = 0;
	for (i = 0; i < vecsz->rnk; ++i) {
		/* extract contiguous dimensions */
		if (pln->vl == 1 &&
			vecsz->dims[i].is == 1 && vecsz->dims[i].os == 1)
			pln->vl = vecsz->dims[i].n;
		else if (pln->rnk == MAXRNK)
			return 0;
		else
			pln->d[pln->rnk++] = vecsz->dims[i];
	}

	return 1;
}

/* generic higher-rank copy routine, calls cpy2d() to do the real work */
static void rank0_copy(const iodim *d, int rnk, INT vl,
	FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	cpy2d_func cpy2d) {
	A(rnk >= 2);
	if (rnk == 2)
		cpy2d(I, O, d[0].n, d[0].is, d[0].os, d[1].n, d[1].is, d[1].os, vl);
	else {
		INT i;
		for (i = 0; i < d[0].n; ++i, I += d[0].is, O += d[0].os)
			rank0_copy(d + 1, rnk - 1, vl, I, O, cpy2d);
	}
}

/* FIXME: should be more general */
static int rank0_transposep(const rank0_P *pln) {
	int i;

	for (i = 0; i < pln->rnk - 2; ++i)
		if (pln->d[i].is != pln->d[i].os)
			return 0;

	return (pln->d[i].n == pln->d[i + 1].n &&
		pln->d[i].is == pln->d[i + 1].os &&
		pln->d[i].os == pln->d[i + 1].is);
}

/* generic higher-rank transpose routine, calls transpose2d() to do
* the real work */
static void rank0_transpose(const iodim *d, int rnk, INT vl,
	FFTW_REAL_TYPE *I,
	transpose_func transpose2d) {
	A(rnk >= 2);
	if (rnk == 2)
		transpose2d(I, d[0].n, d[0].is, d[0].os, vl);
	else {
		INT i;
		for (i = 0; i < d[0].n; ++i, I += d[0].is)
			rank0_transpose(d + 1, rnk - 1, vl, I, transpose2d);
	}
}

/**************************************************************/
/* rank 0,1,2, out of place, iterative */
static void rank0_apply_iter(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;

	switch (ego->rnk) {
	case 0:
		fftw_cpy1d(I, O, ego->vl, 1, 1, 1);
		break;
	case 1:
		fftw_cpy1d(I, O,
			ego->d[0].n, ego->d[0].is, ego->d[0].os,
			ego->vl);
		break;
	default:
		rank0_copy(ego->d, ego->rnk, ego->vl, I, O, fftw_cpy2d_ci);
		break;
	}
}

static int rank0_applicable_iter(const rank0_P *pln, const problem_rdft *p) {
	UNUSED(pln);
	return (p->I != p->O);
}

/**************************************************************/
/* out of place, write contiguous output */
static void rank0_apply_cpy2dco(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	rank0_copy(ego->d, ego->rnk, ego->vl, I, O, fftw_cpy2d_co);
}

static int rank0_applicable_cpy2dco(const rank0_P *pln, const problem_rdft *p) {
	int rnk = pln->rnk;
	return (1
		&& p->I != p->O
		&& rnk >= 2

		/* must not duplicate apply_iter */
		&& (fftw_iabs(pln->d[rnk - 2].is) <= fftw_iabs(pln->d[rnk - 1].is)
			||
			fftw_iabs(pln->d[rnk - 2].os) <= fftw_iabs(pln->d[rnk - 1].os))
		);
}

/**************************************************************/
/* out of place, tiled, no buffering */
static void rank0_apply_tiled(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	rank0_copy(ego->d, ego->rnk, ego->vl, I, O, fftw_cpy2d_tiled);
}

static int rank0_applicable_tiled(const rank0_P *pln, const problem_rdft *p) {
	return (1
		&& p->I != p->O
		&& pln->rnk >= 2

		/* somewhat arbitrary */
		&& fftw_compute_tilesz(pln->vl, 1) > 4
		);
}

/**************************************************************/
/* out of place, tiled, with buffer */
static void rank0_apply_tiledbuf(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	rank0_copy(ego->d, ego->rnk, ego->vl, I, O, fftw_cpy2d_tiledbuf);
}

#define rank0_applicable_tiledbuf rank0_applicable_tiled

/**************************************************************/
/* rank 0, out of place, using memcpy */
static void rank0_apply_memcpy(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;

	A(ego->rnk == 0);
	memcpy(O, I, ego->vl * sizeof(FFTW_REAL_TYPE));
}

static int rank0_applicable_memcpy(const rank0_P *pln, const problem_rdft *p) {
	return (1
		&& p->I != p->O
		&& pln->rnk == 0
		&& pln->vl > 2 /* do not bother memcpy-ing complex numbers */
		);
}

/**************************************************************/
/* rank > 0 vecloop, out of place, using memcpy (e.g. out-of-place
transposes of vl-tuples ... for large vl it should be more
efficient to use memcpy than the tiled stuff). */

static void rank0_memcpy_loop(size_t cpysz, int rnk, const iodim *d, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	INT i, n = d->n, is = d->is, os = d->os;
	if (rnk == 1)
		for (i = 0; i < n; ++i, I += is, O += os)
			memcpy(O, I, cpysz);
	else {
		--rnk;
		++d;
		for (i = 0; i < n; ++i, I += is, O += os)
			rank0_memcpy_loop(cpysz, rnk, d, I, O);
	}
}

static void rank0_apply_memcpy_loop(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	rank0_memcpy_loop(ego->vl * sizeof(FFTW_REAL_TYPE), ego->rnk, ego->d, I, O);
}

static int rank0_applicable_memcpy_loop(const rank0_P *pln, const problem_rdft *p) {
	return (p->I != p->O
		&& pln->rnk > 0
		&& pln->vl > 2 /* do not bother memcpy-ing complex numbers */);
}

/**************************************************************/
/* rank 2, in place, square transpose, iterative */
static void rank0_apply_ip_sq(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	UNUSED(O);
	rank0_transpose(ego->d, ego->rnk, ego->vl, I, fftw_transpose);
}


static int rank0_applicable_ip_sq(const rank0_P *pln, const problem_rdft *p) {
	return (1
		&& p->I == p->O
		&& pln->rnk >= 2
		&& rank0_transposep(pln));
}

/**************************************************************/
/* rank 2, in place, square transpose, tiled */
static void rank0_apply_ip_sq_tiled(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	UNUSED(O);
	rank0_transpose(ego->d, ego->rnk, ego->vl, I, fftw_transpose_tiled);
}

static int rank0_applicable_ip_sq_tiled(const rank0_P *pln, const problem_rdft *p) {
	return (1
		&& rank0_applicable_ip_sq(pln, p)

		/* somewhat arbitrary */
		&& fftw_compute_tilesz(pln->vl, 2) > 4
		);
}

/**************************************************************/
/* rank 2, in place, square transpose, tiled, buffered */
static void rank0_apply_ip_sq_tiledbuf(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rank0_P *ego = (const rank0_P *)ego_;
	UNUSED(O);
	rank0_transpose(ego->d, ego->rnk, ego->vl, I, fftw_transpose_tiledbuf);
}

#define rank0_applicable_ip_sq_tiledbuf rank0_applicable_ip_sq_tiled

/**************************************************************/
static int rank0_applicable(const rank0_S *ego, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	rank0_P pln;
	return (1
		&& p->sz->rnk == 0
		&& FINITE_RNK(p->vecsz->rnk)
		&& rank0_fill_iodim(&pln, p)
		&& ego->applicable(&pln, p)
		);
}

static void rank0_print(const plan *ego_, printer *p) {
	const rank0_P *ego = (const rank0_P *)ego_;
	int i;
	p->print(p, "(%s/%D", ego->nam, ego->vl);
	for (i = 0; i < ego->rnk; ++i)
		p->print(p, "%v", ego->d[i].n);
	p->print(p, ")");
}

static plan *rank0_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const problem_rdft *p;
	const rank0_S *ego = (const rank0_S *)ego_;
	rank0_P *pln;
	int retval;

	static const plan_adt padt = {
		fftw_rdft_solve, fftw_null_awake, rank0_print, fftw_plan_null_destroy
	};

	UNUSED(plnr);

	if (!rank0_applicable(ego, p_))
		return (plan *)0;

	p = (const problem_rdft *)p_;
	pln = MKPLAN_RDFT(rank0_P, &padt, ego->apply);

	retval = rank0_fill_iodim(pln, p);
	(void)retval; /* UNUSED unless DEBUG */
	A(retval);
	A(pln->vl > 0); /* because FINITE_RNK(p->vecsz->rnk) holds */
	pln->nam = ego->nam;

	/* fftw_tensor_sz(p->vecsz) loads, fftw_tensor_sz(p->vecsz) stores */
	fftw_ops_other(2 * fftw_tensor_sz(p->vecsz), &pln->super.super.ops);
	return &(pln->super.super);
}


void fftw_rdft_rank0_register(planner *p) {
	unsigned i;
	static struct {
		rdftapply apply;

		int(*applicable)(const rank0_P *, const problem_rdft *);

		const char *nam;
	} tab[] = {
		{ rank0_apply_memcpy,      rank0_applicable_memcpy,  "rdft-rank0-memcpy" },
		{ rank0_apply_memcpy_loop, rank0_applicable_memcpy_loop,
		"rdft-rank0-memcpy-loop" },
		{ rank0_apply_iter,        rank0_applicable_iter,    "rdft-rank0-iter-ci" },
		{ rank0_apply_cpy2dco,     rank0_applicable_cpy2dco, "rdft-rank0-iter-co" },
		{ rank0_apply_tiled,       rank0_applicable_tiled,   "rdft-rank0-tiled" },
		{ rank0_apply_tiledbuf, rank0_applicable_tiledbuf,   "rdft-rank0-tiledbuf" },
		{ rank0_apply_ip_sq,       rank0_applicable_ip_sq,   "rdft-rank0-ip-sq" },
		{
			rank0_apply_ip_sq_tiled,
			rank0_applicable_ip_sq_tiled,
			"rdft-rank0-ip-sq-tiled"
		},
		{
			rank0_apply_ip_sq_tiledbuf,
			rank0_applicable_ip_sq_tiledbuf,
			"rdft-rank0-ip-sq-tiledbuf"
		},
	};

	for (i = 0; i < sizeof(tab) / sizeof(tab[0]); ++i) {
		static const solver_adt sadt = { PROBLEM_RDFT, rank0_mkplan, 0 };
		rank0_S *slv = MKSOLVER(rank0_S, &sadt);
		slv->apply = tab[i].apply;
		slv->applicable = tab[i].applicable;
		slv->nam = tab[i].nam;
		REGISTER_SOLVER(p, &(slv->super));
	}
}


/* plans for rank-0 RDFT2 (copy operations, plus setting 0 imag. parts) */


#ifdef HAVE_STRING_H

#include <string.h>        /* for memcpy() */

#endif

typedef struct {
	solver super;
} rank0_rdft2_S;

typedef struct {
	plan_rdft super;
	INT vl;
	INT ivs, ovs;
	plan *cldcpy;
} rank0_rdft2_P;

static int rank0_rdft2_applicable(const problem *p_) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	return (1
		&& p->sz->rnk == 0
		&& (p->kind == HC2R
			||
			(1
				&& p->kind == R2HC

				&& p->vecsz->rnk <= 1

				&& ((p->r0 != p->cr)
					||
					fftw_rdft2_inplace_strides(p, RNK_MINFTY))))
		);
}

static void rank0_rdft2_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rank0_rdft2_P *ego = (const rank0_rdft2_P *)ego_;
	INT i, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;

	UNUSED(r1); /* rank-0 has no real odd-index elements */

	for (i = 4; i <= vl; i += 4) {
		FFTW_REAL_TYPE x0, x1, x2, x3;
		x0 = *r0;
		r0 += ivs;
		x1 = *r0;
		r0 += ivs;
		x2 = *r0;
		r0 += ivs;
		x3 = *r0;
		r0 += ivs;
		*cr = x0;
		cr += ovs;
		*ci = K(0.0);
		ci += ovs;
		*cr = x1;
		cr += ovs;
		*ci = K(0.0);
		ci += ovs;
		*cr = x2;
		cr += ovs;
		*ci = K(0.0);
		ci += ovs;
		*cr = x3;
		cr += ovs;
		*ci = K(0.0);
		ci += ovs;
	}
	for (; i < vl + 4; ++i) {
		FFTW_REAL_TYPE x0;
		x0 = *r0;
		r0 += ivs;
		*cr = x0;
		cr += ovs;
		*ci = K(0.0);
		ci += ovs;
	}
}

/* in-place r2hc rank-0: set imaginary parts of output to 0 */
static void rank0_rdft2_apply_r2hc_inplace(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rank0_rdft2_P *ego = (const rank0_rdft2_P *)ego_;
	INT i, vl = ego->vl;
	INT ovs = ego->ovs;

	UNUSED(r0);
	UNUSED(r1);
	UNUSED(cr);

	for (i = 4; i <= vl; i += 4) {
		*ci = K(0.0);
		ci += ovs;
		*ci = K(0.0);
		ci += ovs;
		*ci = K(0.0);
		ci += ovs;
		*ci = K(0.0);
		ci += ovs;
	}
	for (; i < vl + 4; ++i) {
		*ci = K(0.0);
		ci += ovs;
	}
}

/* a rank-0 HC2R rdft2 problem is just a copy from cr to r0,
so we can use a rank-0 rdft plan */
static void rank0_rdft2_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rank0_rdft2_P *ego = (const rank0_rdft2_P *)ego_;
	plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
	UNUSED(ci);
	UNUSED(r1);
	cldcpy->apply((plan *)cldcpy, cr, r0);
}

static void rank0_rdft2_awake(plan *ego_, enum wakefulness wakefulness) {
	rank0_rdft2_P *ego = (rank0_rdft2_P *)ego_;
	if (ego->cldcpy)
		fftw_plan_awake(ego->cldcpy, wakefulness);
}

static void rank0_rdft2_destroy(plan *ego_) {
	rank0_rdft2_P *ego = (rank0_rdft2_P *)ego_;
	if (ego->cldcpy)
		fftw_plan_destroy_internal(ego->cldcpy);
}

static void rank0_rdft2_print(const plan *ego_, printer *p) {
	const rank0_rdft2_P *ego = (const rank0_rdft2_P *)ego_;
	if (ego->cldcpy)
		p->print(p, "(rdft2-hc2r-rank0%(%p%))", ego->cldcpy);
	else
		p->print(p, "(rdft2-r2hc-rank0%v)", ego->vl);
}

static plan *rank0_rdft2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const problem_rdft2 *p;
	plan *cldcpy = (plan *)0;
	rank0_rdft2_P *pln;

	static const plan_adt padt = {
		fftw_rdft2_solve, rank0_rdft2_awake, rank0_rdft2_print, rank0_rdft2_destroy
	};

	UNUSED(ego_);

	if (!rank0_rdft2_applicable(p_))
		return (plan *)0;

	p = (const problem_rdft2 *)p_;

	if (p->kind == HC2R) {
		cldcpy = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_tensor_copy(p->vecsz),
				p->cr, p->r0));
		if (!cldcpy) return (plan *)0;
	}

	pln = MKPLAN_RDFT2(rank0_rdft2_P, &padt,
		p->kind == R2HC ?
		(p->r0 == p->cr ? rank0_rdft2_apply_r2hc_inplace : rank0_rdft2_apply_r2hc)
		: rank0_rdft2_apply_hc2r);

	if (p->kind == R2HC)
		fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);
	pln->cldcpy = cldcpy;

	if (p->kind == R2HC) {
		/* vl loads, 2*vl stores */
		fftw_ops_other(3 * pln->vl, &pln->super.super.ops);
	}
	else {
		pln->super.super.ops = cldcpy->ops;
	}

	return &(pln->super.super);
}

static solver *rank0_rdft2_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT2, rank0_rdft2_mkplan, 0 };
	rank0_rdft2_S *slv = MKSOLVER(rank0_rdft2_S, &sadt);
	return &(slv->super);
}

void fftw_rdft2_rank0_register(planner *p) {
	REGISTER_SOLVER(p, rank0_rdft2_mksolver());
}

/* plans for RDFT of rank >= 2 (multidimensional) */

/* FIXME: this solver cannot strictly be applied to multidimensional
DHTs, since the latter are not separable...up to rnk-1 additional
post-processing passes may be required.  See also:

R. N. Bracewell, O. Buneman, H. Hao, and J. Villasenor, "Fast
two-dimensional Hartley transform," Proc. IEEE 74, 1282-1283 (1986).

H. Hao and R. N. Bracewell, "A three-dimensional DFT algorithm
using the fast Hartley transform," Proc. IEEE 75(2), 264-266 (1987).
*/


typedef struct {
	solver super;
	int spltrnk;
	const int *buddies;
	size_t nbuddies;
} rdft_rank_geq2_S;

typedef struct {
	plan_rdft super;

	plan *cld1, *cld2;
	const rdft_rank_geq2_S *solver;
} rdft_rank_geq2_P;

/* Compute multi-dimensional RDFT by applying the two cld plans
(lower-rnk RDFTs). */
static void rdft_rank_geq2_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_rank_geq2_P *ego = (const rdft_rank_geq2_P *)ego_;
	plan_rdft *cld1, *cld2;

	cld1 = (plan_rdft *)ego->cld1;
	cld1->apply(ego->cld1, I, O);

	cld2 = (plan_rdft *)ego->cld2;
	cld2->apply(ego->cld2, O, O);
}


static void rdft_rank_geq2_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_rank_geq2_P *ego = (rdft_rank_geq2_P *)ego_;
	fftw_plan_awake(ego->cld1, wakefulness);
	fftw_plan_awake(ego->cld2, wakefulness);
}

static void rdft_rank_geq2_destroy(plan *ego_) {
	rdft_rank_geq2_P *ego = (rdft_rank_geq2_P *)ego_;
	fftw_plan_destroy_internal(ego->cld2);
	fftw_plan_destroy_internal(ego->cld1);
}

static void rdft_rank_geq2_print(const plan *ego_, printer *p) {
	const rdft_rank_geq2_P *ego = (const rdft_rank_geq2_P *)ego_;
	const rdft_rank_geq2_S *s = ego->solver;
	p->print(p, "(rdft-rank>=2/%d%(%p%)%(%p%))",
		s->spltrnk, ego->cld1, ego->cld2);
}

static int rdft_rank_geq2_picksplit(const rdft_rank_geq2_S *ego, const tensor *sz, int *rp) {
	A(sz->rnk > 1); /* cannot split rnk <= 1 */
	if (!fftw_pickdim(ego->spltrnk, ego->buddies, ego->nbuddies, sz, 1, rp))
		return 0;
	*rp += 1; /* convert from dim. index to rank */
	if (*rp >= sz->rnk) /* split must reduce rank */
		return 0;
	return 1;
}

static int rdft_rank_geq2_applicable0(const solver *ego_, const problem *p_, int *rp) {
	const problem_rdft *p = (const problem_rdft *)p_;
	const rdft_rank_geq2_S *ego = (const rdft_rank_geq2_S *)ego_;
	return (1
		&& FINITE_RNK(p->sz->rnk) && FINITE_RNK(p->vecsz->rnk)
		&& p->sz->rnk >= 2
		&& rdft_rank_geq2_picksplit(ego, p->sz, rp)
		);
}

/* TODO: revise this. */
static int rdft_rank_geq2_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *rp) {
	const rdft_rank_geq2_S *ego = (const rdft_rank_geq2_S *)ego_;

	if (!rdft_rank_geq2_applicable0(ego_, p_, rp)) return 0;

	if (NO_RANK_SPLITSP(plnr) && (ego->spltrnk != ego->buddies[0]))
		return 0;

	if (NO_UGLYP(plnr)) {
		/* Heuristic: if the vector stride is greater than the transform
		sz, don't use (prefer to do the vector loop first with a
		vrank-geq1 plan). */
		const problem_rdft *p = (const problem_rdft *)p_;

		if (p->vecsz->rnk > 0 &&
			fftw_tensor_min_stride(p->vecsz) > fftw_tensor_max_index(p->sz))
			return 0;
	}

	return 1;
}

static plan *rdft_rank_geq2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rdft_rank_geq2_S *ego = (const rdft_rank_geq2_S *)ego_;
	const problem_rdft *p;
	rdft_rank_geq2_P *pln;
	plan *cld1 = 0, *cld2 = 0;
	tensor *sz1, *sz2, *vecszi, *sz2i;
	int spltrnk;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_rank_geq2_awake, rdft_rank_geq2_print, rdft_rank_geq2_destroy
	};

	if (!rdft_rank_geq2_applicable(ego_, p_, plnr, &spltrnk))
		return (plan *)0;

	p = (const problem_rdft *)p_;
	fftw_tensor_split(p->sz, &sz1, spltrnk, &sz2);
	vecszi = fftw_tensor_copy_inplace(p->vecsz, INPLACE_OS);
	sz2i = fftw_tensor_copy_inplace(sz2, INPLACE_OS);

	cld1 = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_d(fftw_tensor_copy(sz2),
			fftw_tensor_append(p->vecsz, sz1),
			p->I, p->O, p->kind + spltrnk));
	if (!cld1) goto nada;

	cld2 = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_d(
			fftw_tensor_copy_inplace(sz1, INPLACE_OS),
			fftw_tensor_append(vecszi, sz2i),
			p->O, p->O, p->kind));
	if (!cld2) goto nada;

	pln = MKPLAN_RDFT(rdft_rank_geq2_P, &padt, rdft_rank_geq2_apply);

	pln->cld1 = cld1;
	pln->cld2 = cld2;

	pln->solver = ego;
	fftw_ops_add(&cld1->ops, &cld2->ops, &pln->super.super.ops);

	fftw_tensor_destroy4(sz2, sz1, vecszi, sz2i);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cld2);
	fftw_plan_destroy_internal(cld1);
	fftw_tensor_destroy4(sz2, sz1, vecszi, sz2i);
	return (plan *)0;
}

static solver *rdft_rank_geq2_mksolver(int spltrnk, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_rank_geq2_mkplan, 0 };
	rdft_rank_geq2_S *slv = MKSOLVER(rdft_rank_geq2_S, &sadt);
	slv->spltrnk = spltrnk;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_rdft_rank_geq2_register(planner *p) {
	static const int buddies[] = { 1, 0, -2 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, rdft_rank_geq2_mksolver(buddies[i], buddies, NELEM(buddies)));

	/* FIXME: Should we try more buddies?  See also fftw/rank-geq2. */
}

/* plans for RDFT2 of rank >= 2 (multidimensional) */

typedef struct {
	solver super;
	int spltrnk;
	const int *buddies;
	size_t nbuddies;
} rank_geq2_rdft2_S;

typedef struct {
	plan_dft super;
	plan *cldr, *cldc;
	const rank_geq2_rdft2_S *solver;
} rank_geq2_rdft2_P;

static void rank_geq2_rdft2_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rank_geq2_rdft2_P *ego = (const rank_geq2_rdft2_P *)ego_;

	{
		plan_rdft2 *cldr = (plan_rdft2 *)ego->cldr;
		cldr->apply((plan *)cldr, r0, r1, cr, ci);
	}

	{
		plan_dft *cldc = (plan_dft *)ego->cldc;
		cldc->apply((plan *)cldc, cr, ci, cr, ci);
	}
}

static void rank_geq2_rdft2_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const rank_geq2_rdft2_P *ego = (const rank_geq2_rdft2_P *)ego_;

	{
		plan_dft *cldc = (plan_dft *)ego->cldc;
		cldc->apply((plan *)cldc, ci, cr, ci, cr);
	}

	{
		plan_rdft2 *cldr = (plan_rdft2 *)ego->cldr;
		cldr->apply((plan *)cldr, r0, r1, cr, ci);
	}

}

static void rank_geq2_rdft2_awake(plan *ego_, enum wakefulness wakefulness) {
	rank_geq2_rdft2_P *ego = (rank_geq2_rdft2_P *)ego_;
	fftw_plan_awake(ego->cldr, wakefulness);
	fftw_plan_awake(ego->cldc, wakefulness);
}

static void rank_geq2_rdft2_destroy(plan *ego_) {
	rank_geq2_rdft2_P *ego = (rank_geq2_rdft2_P *)ego_;
	fftw_plan_destroy_internal(ego->cldr);
	fftw_plan_destroy_internal(ego->cldc);
}

static void rank_geq2_rdft2_print(const plan *ego_, printer *p) {
	const rank_geq2_rdft2_P *ego = (const rank_geq2_rdft2_P *)ego_;
	const rank_geq2_rdft2_S *s = ego->solver;
	p->print(p, "(rdft2-rank>=2/%d%(%p%)%(%p%))",
		s->spltrnk, ego->cldr, ego->cldc);
}

static int rank_geq2_rdft2_picksplit(const rank_geq2_rdft2_S *ego, const tensor *sz, int *rp) {
	A(sz->rnk > 1); /* cannot split rnk <= 1 */
	if (!fftw_pickdim(ego->spltrnk, ego->buddies, ego->nbuddies, sz, 1, rp))
		return 0;
	*rp += 1; /* convert from dim. index to rank */
	if (*rp >= sz->rnk) /* split must reduce rank */
		return 0;
	return 1;
}

static int rank_geq2_rdft2_applicable0(const solver *ego_, const problem *p_, int *rp,
	const planner *plnr) {
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	const rank_geq2_rdft2_S *ego = (const rank_geq2_rdft2_S *)ego_;
	return (1
		&& FINITE_RNK(p->sz->rnk) && FINITE_RNK(p->vecsz->rnk)

		/* FIXME: multidimensional R2HCII ? */
		&& (p->kind == R2HC || p->kind == HC2R)

		&& p->sz->rnk >= 2
		&& rank_geq2_rdft2_picksplit(ego, p->sz, rp)
		&& (0

			/* can work out-of-place, but HC2R destroys input */
			|| (p->r0 != p->cr &&
			(p->kind == R2HC || !NO_DESTROY_INPUTP(plnr)))

			/* FIXME: what are sufficient conditions for inplace? */
			|| (p->r0 == p->cr))
		);
}

/* TODO: revise this. */
static int rank_geq2_rdft2_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *rp) {
	const rank_geq2_rdft2_S *ego = (const rank_geq2_rdft2_S *)ego_;

	if (!rank_geq2_rdft2_applicable0(ego_, p_, rp, plnr)) return 0;

	if (NO_RANK_SPLITSP(plnr) && (ego->spltrnk != ego->buddies[0]))
		return 0;

	if (NO_UGLYP(plnr)) {
		const problem_rdft2 *p = (const problem_rdft2 *)p_;

		/* Heuristic: if the vector stride is greater than the transform
		size, don't use (prefer to do the vector loop first with a
		vrank-geq1 plan). */
		if (p->vecsz->rnk > 0 &&
			fftw_tensor_min_stride(p->vecsz)
	> fftw_rdft2_tensor_max_index(p->sz, p->kind))
			return 0;
	}

	return 1;
}

static plan *rank_geq2_rdft2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rank_geq2_rdft2_S *ego = (const rank_geq2_rdft2_S *)ego_;
	const problem_rdft2 *p;
	rank_geq2_rdft2_P *pln;
	plan *cldr = 0, *cldc = 0;
	tensor *sz1, *sz2, *vecszi, *sz2i;
	int spltrnk;
	inplace_kind k;
	problem *cldp;

	static const plan_adt padt = {
		fftw_rdft2_solve, rank_geq2_rdft2_awake, rank_geq2_rdft2_print, rank_geq2_rdft2_destroy
	};

	if (!rank_geq2_rdft2_applicable(ego_, p_, plnr, &spltrnk))
		return (plan *)0;

	p = (const problem_rdft2 *)p_;
	fftw_tensor_split(p->sz, &sz1, spltrnk, &sz2);

	k = p->kind == R2HC ? INPLACE_OS : INPLACE_IS;
	vecszi = fftw_tensor_copy_inplace(p->vecsz, k);
	sz2i = fftw_tensor_copy_inplace(sz2, k);

	/* complex data is ~half of real */
	sz2i->dims[sz2i->rnk - 1].n = sz2i->dims[sz2i->rnk - 1].n / 2 + 1;

	cldr = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft2_d(fftw_tensor_copy(sz2),
			fftw_tensor_append(p->vecsz, sz1),
			p->r0, p->r1,
			p->cr, p->ci, p->kind));
	if (!cldr) goto nada;

	if (p->kind == R2HC)
		cldp = fftw_mkproblem_dft_d(fftw_tensor_copy_inplace(sz1, k),
			fftw_tensor_append(vecszi, sz2i),
			p->cr, p->ci, p->cr, p->ci);
	else /* HC2R must swap re/im parts to get IDFT */
		cldp = fftw_mkproblem_dft_d(fftw_tensor_copy_inplace(sz1, k),
			fftw_tensor_append(vecszi, sz2i),
			p->ci, p->cr, p->ci, p->cr);
	cldc = fftw_mkplan_d(plnr, cldp);
	if (!cldc) goto nada;

	pln = MKPLAN_RDFT2(rank_geq2_rdft2_P, &padt,
		p->kind == R2HC ? rank_geq2_rdft2_apply_r2hc : rank_geq2_rdft2_apply_hc2r);

	pln->cldr = cldr;
	pln->cldc = cldc;

	pln->solver = ego;
	fftw_ops_add(&cldr->ops, &cldc->ops, &pln->super.super.ops);

	fftw_tensor_destroy4(sz2i, vecszi, sz2, sz1);

	return &(pln->super.super);

nada:
	fftw_plan_destroy_internal(cldr);
	fftw_plan_destroy_internal(cldc);
	fftw_tensor_destroy4(sz2i, vecszi, sz2, sz1);
	return (plan *)0;
}

static solver *rank_geq2_rdft2_mksolver(int spltrnk, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_RDFT2, rank_geq2_rdft2_mkplan, 0 };
	rank_geq2_rdft2_S *slv = MKSOLVER(rank_geq2_rdft2_S, &sadt);
	slv->spltrnk = spltrnk;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_rdft2_rank_geq2_register(planner *p) {
	static const int buddies[] = { 1, 0, -2 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, rank_geq2_rdft2_mksolver(buddies[i], buddies, NELEM(buddies)));

	/* FIXME: Should we try more buddies?  See also fftw/rank-geq2. */
}

/* Check if the vecsz/sz strides are consistent with the problem
being in-place for vecsz.dim[vdim], or for all dimensions
if vdim == RNK_MINFTY.  We can't just use tensor_inplace_strides
because rdft transforms have the unfortunate property of
differing input and output sizes.   This routine is not
exhaustive; we only return 1 for the most common case.  */
int fftw_rdft2_inplace_strides(const problem_rdft2 *p, int vdim) {
	INT N, Nc;
	INT rs, cs;
	int i;

	for (i = 0; i + 1 < p->sz->rnk; ++i)
		if (p->sz->dims[i].is != p->sz->dims[i].os)
			return 0;

	if (!FINITE_RNK(p->vecsz->rnk) || p->vecsz->rnk == 0)
		return 1;
	if (!FINITE_RNK(vdim)) { /* check all vector dimensions */
		for (vdim = 0; vdim < p->vecsz->rnk; ++vdim)
			if (!fftw_rdft2_inplace_strides(p, vdim))
				return 0;
		return 1;
	}

	A(vdim < p->vecsz->rnk);
	if (p->sz->rnk == 0)
		return (p->vecsz->dims[vdim].is == p->vecsz->dims[vdim].os);

	N = fftw_tensor_sz(p->sz);
	Nc = (N / p->sz->dims[p->sz->rnk - 1].n) *
		(p->sz->dims[p->sz->rnk - 1].n / 2 + 1);
	fftw_rdft2_strides(p->kind, p->sz->dims + p->sz->rnk - 1, &rs, &cs);

	/* the factor of 2 comes from the fact that RS is the stride
	of p->r0 and p->r1, which is twice as large as the strides
	in the r2r case */
	return (p->vecsz->dims[vdim].is == p->vecsz->dims[vdim].os
		&& (fftw_iabs(2 * p->vecsz->dims[vdim].os)
			>= fftw_imax(2 * Nc * fftw_iabs(cs), N * fftw_iabs(rs))));
}

/* Deal with annoyance because the tensor (is,os) applies to
(r,rio/iio) for R2HC and vice-versa for HC2R.  We originally had
(is,os) always apply to (r,rio/iio), but this causes other
headaches with the tensor functions. */
void fftw_rdft2_strides(rdft_kind kind, const iodim *d, INT *rs, INT *cs) {
	if (kind == R2HC) {
		*rs = d->is;
		*cs = d->os;
	}
	else {
		A(kind == HC2R);
		*rs = d->os;
		*cs = d->is;
	}
}

/* like fftw_tensor_max_index, but takes into account the special n/2+1
final dimension for the complex output/input of an R2HC/HC2R transform. */
INT fftw_rdft2_tensor_max_index(const tensor *sz, rdft_kind k) {
	int i;
	INT n = 0;

	A(FINITE_RNK(sz->rnk));
	for (i = 0; i + 1 < sz->rnk; ++i) {
		const iodim *p = sz->dims + i;
		n += (p->n - 1) * fftw_imax(fftw_iabs(p->is), fftw_iabs(p->os));
	}
	if (i < sz->rnk) {
		const iodim *p = sz->dims + i;
		INT is, os;
		fftw_rdft2_strides(k, p, &is, &os);
		n += fftw_imax((p->n - 1) * fftw_iabs(is), (p->n / 2) * fftw_iabs(os));
	}
	return n;
}


/* Solve an R2HC/HC2R problem via post/pre processing of a DHT.  This
is mainly useful because we can use Rader to compute DHTs of prime
sizes.  It also allows us to express hc2r problems in terms of r2hc
(via dht-r2hc), and to do hc2r problems without destroying the input. */


typedef struct {
	solver super;
} rdft_dht_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	INT is, os;
	INT n;
} rdft_dht_P;

static void rdft_dht_apply_r2hc(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_dht_P *ego = (const rdft_dht_P *)ego_;
	INT os;
	INT i, n;

	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply((plan *)cld, I, O);
	}

	n = ego->n;
	os = ego->os;
	for (i = 1; i < n - i; ++i) {
		E a, b;
		a = K(0.5) * O[os * i];
		b = K(0.5) * O[os * (n - i)];
		O[os * i] = a + b;
#if FFT_SIGN == -1
		O[os * (n - i)] = b - a;
#else
		O[os * (n - i)] = a - b;
#endif
	}
}

/* hc2r, destroying input as usual */
static void rdft_dht_apply_hc2r(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_dht_P *ego = (const rdft_dht_P *)ego_;
	INT is = ego->is;
	INT i, n = ego->n;

	for (i = 1; i < n - i; ++i) {
		E a, b;
		a = I[is * i];
		b = I[is * (n - i)];
#if FFT_SIGN == -1
		I[is * i] = a - b;
		I[is * (n - i)] = a + b;
#else
		I[is * i] = a + b;
		I[is * (n - i)] = a - b;
#endif
	}

	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply((plan *)cld, I, O);
	}
}

/* hc2r, without destroying input */
static void rdft_dht_apply_hc2r_save(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_dht_P *ego = (const rdft_dht_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;

	O[0] = I[0];
	for (i = 1; i < n - i; ++i) {
		E a, b;
		a = I[is * i];
		b = I[is * (n - i)];
#if FFT_SIGN == -1
		O[os * i] = a - b;
		O[os * (n - i)] = a + b;
#else
		O[os * i] = a + b;
		O[os * (n - i)] = a - b;
#endif
	}
	if (i == n - i)
		O[os * i] = I[is * i];

	{
		plan_rdft *cld = (plan_rdft *)ego->cld;
		cld->apply((plan *)cld, O, O);
	}
}

static void rdft_dht_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_dht_P *ego = (rdft_dht_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void rdft_dht_destroy(plan *ego_) {
	rdft_dht_P *ego = (rdft_dht_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void rdft_dht_print(const plan *ego_, printer *p) {
	const rdft_dht_P *ego = (const rdft_dht_P *)ego_;
	p->print(p, "(%s-dht-%D%(%p%))",
		ego->super.apply == rdft_dht_apply_r2hc ? "r2hc" : "hc2r",
		ego->n, ego->cld);
}

static int rdft_dht_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk == 0
		&& (p->kind[0] == R2HC || p->kind[0] == HC2R)

		/* hack: size-2 DHT etc. are defined as being equivalent
		to size-2 R2HC in problem.c, so we need this to prevent
		infinite loops for size 2 in EXHAUSTIVE mode: */
		&& p->sz->dims[0].n > 2
		);
}

static int rdft_dht_applicable(const solver *ego, const problem *p_,
	const planner *plnr) {
	return (!NO_SLOWP(plnr) && rdft_dht_applicable0(ego, p_));
}

static plan *rdft_dht_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	rdft_dht_P *pln;
	const problem_rdft *p;
	problem *cldp;
	plan *cld;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_dht_awake, rdft_dht_print, rdft_dht_destroy
	};

	if (!rdft_dht_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	if (p->kind[0] == R2HC || !NO_DESTROY_INPUTP(plnr))
		cldp = fftw_mkproblem_rdft_1(p->sz, p->vecsz, p->I, p->O, DHT);
	else {
		tensor *sz = fftw_tensor_copy_inplace(p->sz, INPLACE_OS);
		cldp = fftw_mkproblem_rdft_1(sz, p->vecsz, p->O, p->O, DHT);
		fftw_tensor_destroy(sz);
	}
	cld = fftw_mkplan_d(plnr, cldp);
	if (!cld) return (plan *)0;

	pln = MKPLAN_RDFT(rdft_dht_P, &padt, p->kind[0] == R2HC ?
		rdft_dht_apply_r2hc : (NO_DESTROY_INPUTP(plnr) ?
			rdft_dht_apply_hc2r_save : rdft_dht_apply_hc2r));
	pln->n = p->sz->dims[0].n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;

	pln->super.super.ops = cld->ops;
	pln->super.super.ops.other += 4 * ((pln->n - 1) / 2);
	pln->super.super.ops.add += 2 * ((pln->n - 1) / 2);
	if (p->kind[0] == R2HC)
		pln->super.super.ops.mul += 2 * ((pln->n - 1) / 2);
	if (pln->super.apply == rdft_dht_apply_hc2r_save)
		pln->super.super.ops.other += 2 + (pln->n % 2 ? 0 : 2);

	return &(pln->super.super);
}

/* constructor */
static solver *rdft_dht_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_dht_mkplan, 0 };
	rdft_dht_S *slv = MKSOLVER(rdft_dht_S, &sadt);
	return &(slv->super);
}

void fftw_rdft_dht_register(planner *p) {
	REGISTER_SOLVER(p, rdft_dht_mksolver());
}

/* use the apply() operation for RDFT problems */
void fftw_rdft_solve(const plan *ego_, const problem *p_) {
	const plan_rdft *ego = (const plan_rdft *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;
	ego->apply(ego_, UNTAINT(p->I), UNTAINT(p->O));
}

/* use the apply() operation for RDFT2 problems */
void fftw_rdft2_solve(const plan *ego_, const problem *p_) {
	const plan_rdft2 *ego = (const plan_rdft2 *)ego_;
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	ego->apply(ego_,
		UNTAINT(p->r0), UNTAINT(p->r1),
		UNTAINT(p->cr), UNTAINT(p->ci));
}


/* rank-0, vector-rank-3, non-square in-place transposition
(see rank0.c for square transposition)  */


#ifdef HAVE_STRING_H

#include <string.h>        /* for memcpy() */

#endif

struct P_s;

typedef struct {
	rdftapply apply;

	int(*applicable)(const problem_rdft *p, planner *plnr,
		int dim0, int dim1, int dim2, INT *nbuf);

	int(*mkcldrn)(const problem_rdft *p, planner *plnr, struct P_s *ego);

	const char *nam;
} vrank3_transpose_adt;

typedef struct {
	solver super;
	const vrank3_transpose_adt *adt;
} vrank3_S;

typedef struct P_s {
	plan_rdft super;
	INT n, m, vl; /* transpose n x m matrix of vl-tuples */
	INT nbuf; /* buffer size */
	INT nd, md, d; /* transpose-gcd params */
	INT nc, mc; /* transpose-cut params */
	plan *cld1, *cld2, *cld3; /* children, null if unused */
	const vrank3_S *slv;
} vrank3_P;


/*************************************************************************/
/* some utilities for the solvers */

static INT vrank3_gcd(INT a, INT b) {
	INT r;
	do {
		r = a % b;
		a = b;
		b = r;
	} while (r != 0);

	return a;
}

/* whether we can transpose with one of our routines expecting
contiguous Ntuples */
static int vrank3_Ntuple_transposable(const iodim *a, const iodim *b, INT vl, INT vs) {
	return (vs == 1 && b->is == vl && a->os == vl &&
		((a->n == b->n && a->is == b->os
			&& a->is >= b->n && a->is % vl == 0)
			|| (a->is == b->n * vl && b->os == a->n * vl)));
}

/* check whether a and b correspond to the first and second dimensions
of a transpose of tuples with vector length = vl, stride = vs. */
static int vrank3_transposable(const iodim *a, const iodim *b, INT vl, INT vs) {
	return ((a->n == b->n && a->os == b->is && a->is == b->os)
		|| vrank3_Ntuple_transposable(a, b, vl, vs));
}

static int vrank3_pickdim(const tensor *s, int *pdim0, int *pdim1, int *pdim2) {
	int dim0, dim1;

	for (dim0 = 0; dim0 < s->rnk; ++dim0)
		for (dim1 = 0; dim1 < s->rnk; ++dim1) {
			int dim2 = 3 - dim0 - dim1;
			if (dim0 == dim1) continue;
			if ((s->rnk == 2 || s->dims[dim2].is == s->dims[dim2].os)
				&& vrank3_transposable(s->dims + dim0, s->dims + dim1,
					s->rnk == 2 ? (INT)1 : s->dims[dim2].n,
					s->rnk == 2 ? (INT)1 : s->dims[dim2].is)) {
				*pdim0 = dim0;
				*pdim1 = dim1;
				*pdim2 = dim2;
				return 1;
			}
		}
	return 0;
}

#define MINBUFDIV 9 /* min factor by which buffer is smaller than data */
#define MAXBUF 65536 /* maximum non-ugly buffer */

/* generic applicability function */
static int vrank3_applicable(const solver *ego_, const problem *p_, planner *plnr,
	int *dim0, int *dim1, int *dim2, INT *nbuf) {
	const vrank3_S *ego = (const vrank3_S *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;

	return (1
		&& p->I == p->O
		&& p->sz->rnk == 0
		&& (p->vecsz->rnk == 2 || p->vecsz->rnk == 3)

		&& vrank3_pickdim(p->vecsz, dim0, dim1, dim2)

		/* UGLY if vecloop in wrong order for locality */
		&& (!NO_UGLYP(plnr) ||
			p->vecsz->rnk == 2 ||
			fftw_iabs(p->vecsz->dims[*dim2].is)
			< fftw_imax(fftw_iabs(p->vecsz->dims[*dim0].is),
				fftw_iabs(p->vecsz->dims[*dim0].os)))

		/* SLOW if non-square */
		&& (!NO_SLOWP(plnr)
			|| p->vecsz->dims[*dim0].n == p->vecsz->dims[*dim1].n)

		&& ego->adt->applicable(p, plnr, *dim0, *dim1, *dim2, nbuf)

		/* buffers too big are UGLY */
		&& ((!NO_UGLYP(plnr) && !CONSERVE_MEMORYP(plnr))
			|| *nbuf <= MAXBUF
			|| *nbuf * MINBUFDIV <= fftw_tensor_sz(p->vecsz))
		);
}

static void vrank3_get_transpose_vec(const problem_rdft *p, int dim2, INT *vl, INT *vs) {
	if (p->vecsz->rnk == 2) {
		*vl = 1;
		*vs = 1;
	}
	else {
		*vl = p->vecsz->dims[dim2].n;
		*vs = p->vecsz->dims[dim2].is; /* == os */
	}
}

/*************************************************************************/
/* Cache-oblivious in-place transpose of non-square matrices, based
on transposes of blocks given by the gcd of the dimensions.

This algorithm is related to algorithm V5 from Murray Dow,
"Transposing a matrix on a vector computer," Parallel Computing 21
(12), 1997-2005 (1995), with the modification that we use
cache-oblivious recursive transpose subroutines (and we derived
it independently).

For a p x q matrix, this requires scratch space equal to the size
of the matrix divided by gcd(p,q).  Alternatively, see also the
"cut" algorithm below, if |p-q| * gcd(p,q) < max(p,q). */

static void vrank3_apply_gcd(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const vrank3_P *ego = (const vrank3_P *)ego_;
	INT n = ego->nd, m = ego->md, d = ego->d;
	INT vl = ego->vl;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * ego->nbuf, BUFFERS);
	INT i, num_el = n * m * d * vl;

	A(ego->n == n * d && ego->m == m * d);
	UNUSED(O);

	/* Transpose the matrix I in-place, where I is an (n*d) x (m*d) matrix
	of vl-tuples and buf contains n*m*d*vl elements.

	In general, to transpose a p x q matrix, you should call this
	routine with d = gcd(p, q), n = p/d, and m = q/d.  */

	A(n > 0 && m > 0 && vl > 0);
	A(d > 1);

	/* treat as (d x n) x (d' x m) matrix.  (d' = d) */

	/* First, transpose d x (n x d') x m to d x (d' x n) x m,
	using the buf matrix.  This consists of d transposes
	of contiguous n x d' matrices of m-tuples. */
	if (n > 1) {
		rdftapply cldapply = ((plan_rdft *)ego->cld1)->apply;
		for (i = 0; i < d; ++i) {
			cldapply(ego->cld1, I + i * num_el, buf);
			memcpy(I + i * num_el, buf, num_el * sizeof(FFTW_REAL_TYPE));
		}
	}

	/* Now, transpose (d x d') x (n x m) to (d' x d) x (n x m), which
	is a square in-place transpose of n*m-tuples: */
	{
		rdftapply cldapply = ((plan_rdft *)ego->cld2)->apply;
		cldapply(ego->cld2, I, I);
	}

	/* Finally, transpose d' x ((d x n) x m) to d' x (m x (d x n)),
	using the buf matrix.  This consists of d' transposes
	of contiguous d*n x m matrices. */
	if (m > 1) {
		rdftapply cldapply = ((plan_rdft *)ego->cld3)->apply;
		for (i = 0; i < d; ++i) {
			cldapply(ego->cld3, I + i * num_el, buf);
			memcpy(I + i * num_el, buf, num_el * sizeof(FFTW_REAL_TYPE));
		}
	}

	fftw_ifree(buf);
}

static int vrank3_applicable_gcd(const problem_rdft *p, planner *plnr,
	int dim0, int dim1, int dim2, INT *nbuf) {
	INT n = p->vecsz->dims[dim0].n;
	INT m = p->vecsz->dims[dim1].n;
	INT d, vl, vs;
	vrank3_get_transpose_vec(p, dim2, &vl, &vs);
	d = vrank3_gcd(n, m);
	*nbuf = n * (m / d) * vl;
	return (!NO_SLOWP(plnr) /* FIXME: not really SLOW for large 1d ffts */
		&& n != m
		&& d > 1
		&& vrank3_Ntuple_transposable(p->vecsz->dims + dim0,
			p->vecsz->dims + dim1,
			vl, vs));
}

static int vrank3_mkcldrn_gcd(const problem_rdft *p, planner *plnr, vrank3_P *ego) {
	INT n = ego->nd, m = ego->md, d = ego->d;
	INT vl = ego->vl;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * ego->nbuf, BUFFERS);
	INT num_el = n * m * d * vl;

	if (n > 1) {
		ego->cld1 = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_3d(n, d * m * vl, m * vl,
					d, m * vl, n * m * vl,
					m * vl, 1, 1),
				TAINT(p->I, num_el), buf));
		if (!ego->cld1)
			goto nada;
		fftw_ops_madd(d, &ego->cld1->ops, &ego->super.super.ops,
			&ego->super.super.ops);
		ego->super.super.ops.other += num_el * d * 2;
	}

	ego->cld2 = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_0_d(
			fftw_mktensor_3d(d, d * n * m * vl, n * m * vl,
				d, n * m * vl, d * n * m * vl,
				n * m * vl, 1, 1),
			p->I, p->I));
	if (!ego->cld2)
		goto nada;
	fftw_ops_add2(&ego->cld2->ops, &ego->super.super.ops);

	if (m > 1) {
		ego->cld3 = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_3d(d * n, m * vl, vl,
					m, vl, d * n * vl,
					vl, 1, 1),
				TAINT(p->I, num_el), buf));
		if (!ego->cld3)
			goto nada;
		fftw_ops_madd2(d, &ego->cld3->ops, &ego->super.super.ops);
		ego->super.super.ops.other += num_el * d * 2;
	}

	fftw_ifree(buf);
	return 1;

nada:
	fftw_ifree(buf);
	return 0;
}

static const vrank3_transpose_adt vrank3_adt_gcd =
{
	vrank3_apply_gcd, vrank3_applicable_gcd, vrank3_mkcldrn_gcd,
	"rdft-transpose-gcd"
};

/*************************************************************************/
/* Cache-oblivious in-place transpose of non-square n x m matrices,
based on transposing a sub-matrix first and then transposing the
remainder(s) with the help of a buffer.  See also transpose-gcd,
above, if gcd(n,m) is large.

This algorithm is related to algorithm V3 from Murray Dow,
"Transposing a matrix on a vector computer," Parallel Computing 21
(12), 1997-2005 (1995), with the modifications that we use
cache-oblivious recursive transpose subroutines and we have the
generalization for large |n-m| below.

The best case, and the one described by Dow, is for |n-m| small, in
which case we transpose a square sub-matrix of size min(n,m),
handling the remainder via a buffer.  This requires scratch space
equal to the size of the matrix times |n-m| / max(n,m).

As a generalization when |n-m| is not small, we also support cutting
*both* dimensions to an nc x mc matrix which is *not* necessarily
square, but has a large gcd (and can therefore use transpose-gcd).
*/

static void vrank3_apply_cut(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const vrank3_P *ego = (const vrank3_P *)ego_;
	INT n = ego->n, m = ego->m, nc = ego->nc, mc = ego->mc, vl = ego->vl;
	INT i;
	FFTW_REAL_TYPE *buf1 = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * ego->nbuf, BUFFERS);
	UNUSED(O);

	if (m > mc) {
		((plan_rdft *)ego->cld1)->apply(ego->cld1, I + mc * vl, buf1);
		for (i = 0; i < nc; ++i)
			memmove(I + (mc * vl) * i, I + (m * vl) * i, sizeof(FFTW_REAL_TYPE) * (mc * vl));
	}

	((plan_rdft *)ego->cld2)->apply(ego->cld2, I, I); /* nc x mc transpose */

	if (n > nc) {
		FFTW_REAL_TYPE *buf2 = buf1 + (m - mc) * (nc * vl); /* FIXME: force better alignment? */
		memcpy(buf2, I + nc * (m * vl), (n - nc) * (m * vl) * sizeof(FFTW_REAL_TYPE));
		for (i = mc - 1; i >= 0; --i)
			memmove(I + (n * vl) * i, I + (nc * vl) * i, sizeof(FFTW_REAL_TYPE) * (n * vl));
		((plan_rdft *)ego->cld3)->apply(ego->cld3, buf2, I + nc * vl);
	}

	if (m > mc) {
		if (n > nc)
			for (i = mc; i < m; ++i)
				memcpy(I + i * (n * vl), buf1 + (i - mc) * (nc * vl),
				(nc * vl) * sizeof(FFTW_REAL_TYPE));
		else
			memcpy(I + mc * (n * vl), buf1, (m - mc) * (n * vl) * sizeof(FFTW_REAL_TYPE));
	}

	fftw_ifree(buf1);
}

/* only cut one dimension if the resulting buffer is small enough */
static int vrank3_cut1(INT n, INT m, INT vl) {
	return (fftw_imax(n, m) >= fftw_iabs(n - m) * MINBUFDIV
		|| fftw_imin(n, m) * fftw_iabs(n - m) * vl <= MAXBUF);
}

#define CUT_NSRCH 32 /* range of sizes to search for possible cuts */

static int vrank3_applicable_cut(const problem_rdft *p, planner *plnr,
	int dim0, int dim1, int dim2, INT *nbuf) {
	INT n = p->vecsz->dims[dim0].n;
	INT m = p->vecsz->dims[dim1].n;
	INT vl, vs;
	vrank3_get_transpose_vec(p, dim2, &vl, &vs);
	*nbuf = 0; /* always small enough to be non-UGLY (?) */
	A(MINBUFDIV <= CUT_NSRCH); /* assumed to avoid inf. loops below */
	return (!NO_SLOWP(plnr) /* FIXME: not really SLOW for large 1d ffts? */
		&& n != m

		/* Don't call transpose-cut recursively (avoid inf. loops):
		the non-square sub-transpose produced when !cut1
		should always have gcd(n,m) >= min(CUT_NSRCH,n,m),
		for which transpose-gcd is applicable */
		&& (vrank3_cut1(n, m, vl)
			|| vrank3_gcd(n, m) < fftw_imin(MINBUFDIV, fftw_imin(n, m)))

		&& vrank3_Ntuple_transposable(p->vecsz->dims + dim0,
			p->vecsz->dims + dim1,
			vl, vs));
}

static int vrank3_mkcldrn_cut(const problem_rdft *p, planner *plnr, vrank3_P *ego) {
	INT n = ego->n, m = ego->m, nc, mc;
	INT vl = ego->vl;
	FFTW_REAL_TYPE *buf;

	/* pick the "best" cut */
	if (vrank3_cut1(n, m, vl)) {
		nc = mc = fftw_imin(n, m);
	}
	else {
		INT dc, ns, ms;
		dc = vrank3_gcd(m, n);
		nc = n;
		mc = m;
		/* search for cut with largest gcd
		(TODO: different optimality criteria? different search range?) */
		for (ms = m; ms > 0 && ms > m - CUT_NSRCH; --ms) {
			for (ns = n; ns > 0 && ns > n - CUT_NSRCH; --ns) {
				INT ds = vrank3_gcd(ms, ns);
				if (ds > dc) {
					dc = ds;
					nc = ns;
					mc = ms;
					if (dc == fftw_imin(ns, ms))
						break; /* cannot get larger than this */
				}
			}
			if (dc == fftw_imin(n, ms))
				break; /* cannot get larger than this */
		}
		A(dc >= fftw_imin(CUT_NSRCH, fftw_imin(n, m)));
	}
	ego->nc = nc;
	ego->mc = mc;
	ego->nbuf = (m - mc) * (nc * vl) + (n - nc) * (m * vl);

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * ego->nbuf, BUFFERS);

	if (m > mc) {
		ego->cld1 = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_3d(nc, m * vl, vl,
					m - mc, vl, nc * vl,
					vl, 1, 1),
				p->I + mc * vl, buf));
		if (!ego->cld1)
			goto nada;
		fftw_ops_add2(&ego->cld1->ops, &ego->super.super.ops);
	}

	ego->cld2 = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_0_d(
			fftw_mktensor_3d(nc, mc * vl, vl,
				mc, vl, nc * vl,
				vl, 1, 1),
			p->I, p->I));
	if (!ego->cld2)
		goto nada;
	fftw_ops_add2(&ego->cld2->ops, &ego->super.super.ops);

	if (n > nc) {
		ego->cld3 = fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_0_d(
				fftw_mktensor_3d(n - nc, m * vl, vl,
					m, vl, n * vl,
					vl, 1, 1),
				buf + (m - mc) * (nc * vl), p->I + nc * vl));
		if (!ego->cld3)
			goto nada;
		fftw_ops_add2(&ego->cld3->ops, &ego->super.super.ops);
	}

	/* memcpy/memmove operations */
	ego->super.super.ops.other += 2 * vl * (nc * mc * ((m > mc) + (n > nc))
		+ (n - nc) * m + (m - mc) * nc);

	fftw_ifree(buf);
	return 1;

nada:
	fftw_ifree(buf);
	return 0;
}

static const vrank3_transpose_adt vrank3_adt_cut =
{
	vrank3_apply_cut, vrank3_applicable_cut, vrank3_mkcldrn_cut,
	"rdft-transpose-cut"
};

/*************************************************************************/
/* In-place transpose routine from TOMS, which follows the cycles of
the permutation so that it writes to each location only once.
Because of cache-line and other issues, however, this routine is
typically much slower than transpose-gcd or transpose-cut, even
though the latter do some extra writes.  On the other hand, if the
vector length is large then the TOMS routine is best.

The TOMS routine also has the advantage of requiring less buffer
space for the case of gcd(nx,ny) small.  However, in this case it
has been superseded by the combination of the generalized
transpose-cut method with the transpose-gcd method, which can
always transpose with buffers a small fraction of the array size
regardless of gcd(nx,ny). */

/*
* TOMS Transpose.  Algorithm 513 (Revised version of algorithm 380).
*
* These routines do in-place transposes of arrays.
*
* [ Cate, E.G. and Twigg, D.W., ACM Transactions on Mathematical Software,
*   vol. 3, no. 1, 104-110 (1977) ]
*
* C version by Steven G. Johnson (February 1997).
*/

/*
* "a" is a 1D array of length ny*nx*N which constains the nx x ny
* matrix of N-tuples to be transposed.  "a" is stored in row-major
* order (last index varies fastest).  move is a 1D array of length
* move_size used to store information to speed up the process.  The
* value move_size=(ny+nx)/2 is recommended.  buf should be an array
* of length 2*N.
*
*/

static void vrank3_transpose_toms513(FFTW_REAL_TYPE *a, INT nx, INT ny, INT N,
	char *move, INT move_size, FFTW_REAL_TYPE *buf) {
	INT i, im, mn;
	FFTW_REAL_TYPE *b, *c, *d;
	INT ncount;
	INT k;

	/* check arguments and initialize: */
	A(ny > 0 && nx > 0 && N > 0 && move_size > 0);

	b = buf;

	/* Cate & Twigg have a special case for nx == ny, but we don't
	bother, since we already have special code for this case elsewhere. */

	c = buf + N;
	ncount = 2;        /* always at least 2 fixed points */
	k = (mn = ny * nx) - 1;

	for (i = 0; i < move_size; ++i)
		move[i] = 0;

	if (ny >= 3 && nx >= 3)
		ncount += vrank3_gcd(ny - 1, nx - 1) - 1;    /* # fixed points */

	i = 1;
	im = ny;

	while (1) {
		INT i1, i2, i1c, i2c;
		INT kmi;

		/** Rearrange the elements of a loop
		and its companion loop: **/

		i1 = i;
		kmi = k - i;
		i1c = kmi;
		switch (N) {
		case 1:
			b[0] = a[i1];
			c[0] = a[i1c];
			break;
		case 2:
			b[0] = a[2 * i1];
			b[1] = a[2 * i1 + 1];
			c[0] = a[2 * i1c];
			c[1] = a[2 * i1c + 1];
			break;
		default:
			memcpy(b, &a[N * i1], N * sizeof(FFTW_REAL_TYPE));
			memcpy(c, &a[N * i1c], N * sizeof(FFTW_REAL_TYPE));
		}
		while (1) {
			i2 = ny * i1 - k * (i1 / nx);
			i2c = k - i2;
			if (i1 < move_size)
				move[i1] = 1;
			if (i1c < move_size)
				move[i1c] = 1;
			ncount += 2;
			if (i2 == i)
				break;
			if (i2 == kmi) {
				d = b;
				b = c;
				c = d;
				break;
			}
			switch (N) {
			case 1:
				a[i1] = a[i2];
				a[i1c] = a[i2c];
				break;
			case 2:
				a[2 * i1] = a[2 * i2];
				a[2 * i1 + 1] = a[2 * i2 + 1];
				a[2 * i1c] = a[2 * i2c];
				a[2 * i1c + 1] = a[2 * i2c + 1];
				break;
			default:
				memcpy(&a[N * i1], &a[N * i2],
					N * sizeof(FFTW_REAL_TYPE));
				memcpy(&a[N * i1c], &a[N * i2c],
					N * sizeof(FFTW_REAL_TYPE));
			}
			i1 = i2;
			i1c = i2c;
		}
		switch (N) {
		case 1:
			a[i1] = b[0];
			a[i1c] = c[0];
			break;
		case 2:
			a[2 * i1] = b[0];
			a[2 * i1 + 1] = b[1];
			a[2 * i1c] = c[0];
			a[2 * i1c + 1] = c[1];
			break;
		default:
			memcpy(&a[N * i1], b, N * sizeof(FFTW_REAL_TYPE));
			memcpy(&a[N * i1c], c, N * sizeof(FFTW_REAL_TYPE));
		}
		if (ncount >= mn)
			break;    /* we've moved all elements */

					  /** Search for loops to rearrange: **/

		while (1) {
			INT max = k - i;
			++i;
			A(i <= max);
			im += ny;
			if (im > k)
				im -= k;
			i2 = im;
			if (i == i2)
				continue;
			if (i >= move_size) {
				while (i2 > i && i2 < max) {
					i1 = i2;
					i2 = ny * i1 - k * (i1 / nx);
				}
				if (i2 == i)
					break;
			}
			else if (!move[i])
				break;
		}
	}
}

static void vrank3_apply_toms513(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const vrank3_P *ego = (const vrank3_P *)ego_;
	INT n = ego->n, m = ego->m;
	INT vl = ego->vl;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * ego->nbuf, BUFFERS);
	UNUSED(O);
	vrank3_transpose_toms513(I, n, m, vl, (char *)(buf + 2 * vl), (n + m) / 2, buf);
	fftw_ifree(buf);
}

static int vrank3_applicable_toms513(const problem_rdft *p, planner *plnr,
	int dim0, int dim1, int dim2, INT *nbuf) {
	INT n = p->vecsz->dims[dim0].n;
	INT m = p->vecsz->dims[dim1].n;
	INT vl, vs;
	vrank3_get_transpose_vec(p, dim2, &vl, &vs);
	*nbuf = 2 * vl
		+ ((n + m) / 2 * sizeof(char) + sizeof(FFTW_REAL_TYPE) - 1) / sizeof(FFTW_REAL_TYPE);
	return (!NO_SLOWP(plnr)
		&& (vl > 8 || !NO_UGLYP(plnr)) /* UGLY for small vl */
		&& n != m
		&& vrank3_Ntuple_transposable(p->vecsz->dims + dim0,
			p->vecsz->dims + dim1,
			vl, vs));
}

static int vrank3_mkcldrn_toms513(const problem_rdft *p, planner *plnr, vrank3_P *ego) {
	UNUSED(p);
	UNUSED(plnr);
	/* heuristic so that TOMS algorithm is last resort for small vl */
	ego->super.super.ops.other += ego->n * ego->m * 2 * (ego->vl + 30);
	return 1;
}

static const vrank3_transpose_adt vrank3_adt_toms513 =
{
	vrank3_apply_toms513, vrank3_applicable_toms513, vrank3_mkcldrn_toms513,
	"rdft-transpose-toms513"
};

/*-----------------------------------------------------------------------*/
/*-----------------------------------------------------------------------*/
/* generic stuff: */

static void vrank3_awake(plan *ego_, enum wakefulness wakefulness) {
	vrank3_P *ego = (vrank3_P *)ego_;
	fftw_plan_awake(ego->cld1, wakefulness);
	fftw_plan_awake(ego->cld2, wakefulness);
	fftw_plan_awake(ego->cld3, wakefulness);
}

static void vrank3_print(const plan *ego_, printer *p) {
	const vrank3_P *ego = (const vrank3_P *)ego_;
	p->print(p, "(%s-%Dx%D%v", ego->slv->adt->nam,
		ego->n, ego->m, ego->vl);
	if (ego->cld1) p->print(p, "%(%p%)", ego->cld1);
	if (ego->cld2) p->print(p, "%(%p%)", ego->cld2);
	if (ego->cld3) p->print(p, "%(%p%)", ego->cld3);
	p->print(p, ")");
}

static void vrank3_destroy(plan *ego_) {
	vrank3_P *ego = (vrank3_P *)ego_;
	fftw_plan_destroy_internal(ego->cld3);
	fftw_plan_destroy_internal(ego->cld2);
	fftw_plan_destroy_internal(ego->cld1);
}

static plan *vrank3_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const vrank3_S *ego = (const vrank3_S *)ego_;
	const problem_rdft *p;
	int dim0, dim1, dim2;
	INT nbuf, vs;
	vrank3_P *pln;

	static const plan_adt padt = {
		fftw_rdft_solve, vrank3_awake, vrank3_print, vrank3_destroy
	};

	if (!vrank3_applicable(ego_, p_, plnr, &dim0, &dim1, &dim2, &nbuf))
		return (plan *)0;

	p = (const problem_rdft *)p_;
	pln = MKPLAN_RDFT(vrank3_P, &padt, ego->adt->apply);

	pln->n = p->vecsz->dims[dim0].n;
	pln->m = p->vecsz->dims[dim1].n;
	vrank3_get_transpose_vec(p, dim2, &pln->vl, &vs);
	pln->nbuf = nbuf;
	pln->d = vrank3_gcd(pln->n, pln->m);
	pln->nd = pln->n / pln->d;
	pln->md = pln->m / pln->d;
	pln->slv = ego;

	fftw_ops_zero(&pln->super.super.ops); /* mkcldrn is responsible for ops */

	pln->cld1 = pln->cld2 = pln->cld3 = 0;
	if (!ego->adt->mkcldrn(p, plnr, pln)) {
		fftw_plan_destroy_internal(&(pln->super.super));
		return 0;
	}

	return &(pln->super.super);
}

static solver *vrank3_mksolver(const vrank3_transpose_adt *adt) {
	static const solver_adt sadt = { PROBLEM_RDFT, vrank3_mkplan, 0 };
	vrank3_S *slv = MKSOLVER(vrank3_S, &sadt);
	slv->adt = adt;
	return &(slv->super);
}

void fftw_rdft_vrank3_transpose_register(planner *p) {
	unsigned i;
	static const vrank3_transpose_adt *const adts[] = {
		&vrank3_adt_gcd, &vrank3_adt_cut,
		&vrank3_adt_toms513
	};
	for (i = 0; i < sizeof(adts) / sizeof(adts[0]); ++i)
		REGISTER_SOLVER(p, vrank3_mksolver(adts[i]));
}


/* Plans for handling vector transform loops.  These are *just* the
loops, and rely on child plans for the actual RDFTs.

They form a wrapper around solvers that don't have apply functions
for non-null vectors.

vrank-geq1 plans also recursively handle the case of multi-dimensional
vectors, obviating the need for most solvers to deal with this.  We
can also play games here, such as reordering the vector loops.

Each vrank-geq1 plan reduces the vector rank by 1, picking out a
dimension determined by the vecloop_dim field of the solver. */


typedef struct {
	solver super;
	int vecloop_dim;
	const int *buddies;
	size_t nbuddies;
} rdft_vrank_geq1_S;

typedef struct {
	plan_rdft super;

	plan *cld;
	INT vl;
	INT ivs, ovs;
	const rdft_vrank_geq1_S *solver;
} rdft_vrank_geq1_P;

static void rdft_vrank_geq1_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rdft_vrank_geq1_P *ego = (const rdft_vrank_geq1_P *)ego_;
	INT i, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	rdftapply cldapply = ((plan_rdft *)ego->cld)->apply;

	for (i = 0; i < vl; ++i) {
		cldapply(ego->cld, I + i * ivs, O + i * ovs);
	}
}

static void rdft_vrank_geq1_awake(plan *ego_, enum wakefulness wakefulness) {
	rdft_vrank_geq1_P *ego = (rdft_vrank_geq1_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void rdft_vrank_geq1_destroy(plan *ego_) {
	rdft_vrank_geq1_P *ego = (rdft_vrank_geq1_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void rdft_vrank_geq1_print(const plan *ego_, printer *p) {
	const rdft_vrank_geq1_P *ego = (const rdft_vrank_geq1_P *)ego_;
	const rdft_vrank_geq1_S *s = ego->solver;
	p->print(p, "(rdft-vrank>=1-x%D/%d%(%p%))",
		ego->vl, s->vecloop_dim, ego->cld);
}

static int rdft_vrank_geq1_pickdim(const rdft_vrank_geq1_S *ego, const tensor *vecsz, int oop, int *dp) {
	return fftw_pickdim(ego->vecloop_dim, ego->buddies, ego->nbuddies,
		vecsz, oop, dp);
}

static int rdft_vrank_geq1_applicable0(const solver *ego_, const problem *p_, int *dp) {
	const rdft_vrank_geq1_S *ego = (const rdft_vrank_geq1_S *)ego_;
	const problem_rdft *p = (const problem_rdft *)p_;

	return (1
		&& FINITE_RNK(p->vecsz->rnk)
		&& p->vecsz->rnk > 0

		&& p->sz->rnk >= 0

		&& rdft_vrank_geq1_pickdim(ego, p->vecsz, p->I != p->O, dp)
		);
}

static int rdft_vrank_geq1_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *dp) {
	const rdft_vrank_geq1_S *ego = (const rdft_vrank_geq1_S *)ego_;
	const problem_rdft *p;

	if (!rdft_vrank_geq1_applicable0(ego_, p_, dp)) return 0;

	/* fftw2 behavior */
	if (NO_VRANK_SPLITSP(plnr) && (ego->vecloop_dim != ego->buddies[0]))
		return 0;

	p = (const problem_rdft *)p_;

	if (NO_UGLYP(plnr)) {
		/* the rank-0 solver deals with the general case most of the
		time (an exception is loops of non-square transposes) */
		if (NO_SLOWP(plnr) && p->sz->rnk == 0)
			return 0;

		/* Heuristic: if the transform is multi-dimensional, and the
		vector stride is less than the transform size, then we
		probably want to use a rank>=2 plan first in order to combine
		this vector with the transform-dimension vectors. */
		{
			iodim *d = p->vecsz->dims + *dp;
			if (1
				&& p->sz->rnk > 1
				&& fftw_imin(fftw_iabs(d->is), fftw_iabs(d->os))
				< fftw_tensor_max_index(p->sz)
				)
				return 0;
		}

		/* prefer threaded version */
		if (NO_NONTHREADEDP(plnr)) return 0;

		/* exploit built-in vecloops of (ugly) r{e,o}dft solvers */
		if (p->vecsz->rnk == 1 && p->sz->rnk == 1
			&& REODFT_KINDP(p->kind[0]))
			return 0;
	}

	return 1;
}

static plan *rdft_vrank_geq1_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const rdft_vrank_geq1_S *ego = (const rdft_vrank_geq1_S *)ego_;
	const problem_rdft *p;
	rdft_vrank_geq1_P *pln;
	plan *cld;
	int vdim;
	iodim *d;

	static const plan_adt padt = {
		fftw_rdft_solve, rdft_vrank_geq1_awake, rdft_vrank_geq1_print, rdft_vrank_geq1_destroy
	};

	if (!rdft_vrank_geq1_applicable(ego_, p_, plnr, &vdim))
		return (plan *)0;
	p = (const problem_rdft *)p_;

	d = p->vecsz->dims + vdim;

	A(d->n > 1);

	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft_d(
			fftw_tensor_copy(p->sz),
			fftw_tensor_copy_except(p->vecsz, vdim),
			TAINT(p->I, d->is), TAINT(p->O, d->os),
			p->kind));
	if (!cld) return (plan *)0;

	pln = MKPLAN_RDFT(rdft_vrank_geq1_P, &padt, rdft_vrank_geq1_apply);

	pln->cld = cld;
	pln->vl = d->n;
	pln->ivs = d->is;
	pln->ovs = d->os;

	pln->solver = ego;
	fftw_ops_zero(&pln->super.super.ops);
	pln->super.super.ops.other = 3.14159; /* magic to prefer codelet loops */
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	if (p->sz->rnk != 1 || (p->sz->dims[0].n > 128))
		pln->super.super.pcost = pln->vl * cld->pcost;

	return &(pln->super.super);
}

static solver *rdft_vrank_geq1_mksolver(int vecloop_dim, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_RDFT, rdft_vrank_geq1_mkplan, 0 };
	rdft_vrank_geq1_S *slv = MKSOLVER(rdft_vrank_geq1_S, &sadt);
	slv->vecloop_dim = vecloop_dim;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_rdft_vrank_geq1_register(planner *p) {
	/* FIXME: Should we try other vecloop_dim values? */
	static const int buddies[] = { 1, -1 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, rdft_vrank_geq1_mksolver(buddies[i], buddies, NELEM(buddies)));
}


/* Plans for handling vector transform loops.  These are *just* the
loops, and rely on child plans for the actual RDFT2s.

They form a wrapper around solvers that don't have apply functions
for non-null vectors.

vrank-geq1-rdft2 plans also recursively handle the case of
multi-dimensional vectors, obviating the need for most solvers to
deal with this.  We can also play games here, such as reordering
the vector loops.

Each vrank-geq1-rdft2 plan reduces the vector rank by 1, picking out a
dimension determined by the vecloop_dim field of the solver. */


typedef struct {
	solver super;
	int vecloop_dim;
	const int *buddies;
	size_t nbuddies;
} vrank_geq1_rdft2_S;

typedef struct {
	plan_rdft2 super;

	plan *cld;
	INT vl;
	INT rvs, cvs;
	const vrank_geq1_rdft2_S *solver;
} vrank_geq1_rdft2_P;

static void vrank_geq1_rdft2_apply(const plan *ego_, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
	FFTW_REAL_TYPE *ci) {
	const vrank_geq1_rdft2_P *ego = (const vrank_geq1_rdft2_P *)ego_;
	INT i, vl = ego->vl;
	INT rvs = ego->rvs, cvs = ego->cvs;
	rdft2apply cldapply = ((plan_rdft2 *)ego->cld)->apply;

	for (i = 0; i < vl; ++i) {
		cldapply(ego->cld, r0 + i * rvs, r1 + i * rvs,
			cr + i * cvs, ci + i * cvs);
	}
}

static void vrank_geq1_rdft2_awake(plan *ego_, enum wakefulness wakefulness) {
	vrank_geq1_rdft2_P *ego = (vrank_geq1_rdft2_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void vrank_geq1_rdft2_destroy(plan *ego_) {
	vrank_geq1_rdft2_P *ego = (vrank_geq1_rdft2_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void vrank_geq1_rdft2_print(const plan *ego_, printer *p) {
	const vrank_geq1_rdft2_P *ego = (const vrank_geq1_rdft2_P *)ego_;
	const vrank_geq1_rdft2_S *s = ego->solver;
	p->print(p, "(rdft2-vrank>=1-x%D/%d%(%p%))",
		ego->vl, s->vecloop_dim, ego->cld);
}

static int vrank_geq1_rdft2_pickdim(const vrank_geq1_rdft2_S *ego, const tensor *vecsz, int oop, int *dp) {
	return fftw_pickdim(ego->vecloop_dim, ego->buddies, ego->nbuddies,
		vecsz, oop, dp);
}

static int vrank_geq1_rdft2_applicable0(const solver *ego_, const problem *p_, int *dp) {
	const vrank_geq1_rdft2_S *ego = (const vrank_geq1_rdft2_S *)ego_;
	const problem_rdft2 *p = (const problem_rdft2 *)p_;
	if (FINITE_RNK(p->vecsz->rnk)
		&& p->vecsz->rnk > 0
		&& vrank_geq1_rdft2_pickdim(ego, p->vecsz, p->r0 != p->cr, dp)) {
		if (p->r0 != p->cr)
			return 1;  /* can always operate out-of-place */

		return (fftw_rdft2_inplace_strides(p, *dp));
	}

	return 0;
}


static int vrank_geq1_rdft2_applicable(const solver *ego_, const problem *p_,
	const planner *plnr, int *dp) {
	const vrank_geq1_rdft2_S *ego = (const vrank_geq1_rdft2_S *)ego_;
	if (!vrank_geq1_rdft2_applicable0(ego_, p_, dp)) return 0;

	/* fftw2 behavior */
	if (NO_VRANK_SPLITSP(plnr) && (ego->vecloop_dim != ego->buddies[0]))
		return 0;

	if (NO_UGLYP(plnr)) {
		const problem_rdft2 *p = (const problem_rdft2 *)p_;
		iodim *d = p->vecsz->dims + *dp;

		/* Heuristic: if the transform is multi-dimensional, and the
		vector stride is less than the transform size, then we
		probably want to use a rank>=2 plan first in order to combine
		this vector with the transform-dimension vectors. */
		if (p->sz->rnk > 1
			&& fftw_imin(fftw_iabs(d->is), fftw_iabs(d->os))
			< fftw_rdft2_tensor_max_index(p->sz, p->kind)
			)
			return 0;

		/* Heuristic: don't use a vrank-geq1 for rank-0 vrank-1
		transforms, since this case is better handled by rank-0
		solvers. */
		if (p->sz->rnk == 0 && p->vecsz->rnk == 1) return 0;

		if (NO_NONTHREADEDP(plnr))
			return 0; /* prefer threaded version */
	}

	return 1;
}

static plan *vrank_geq1_rdft2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	const vrank_geq1_rdft2_S *ego = (const vrank_geq1_rdft2_S *)ego_;
	const problem_rdft2 *p;
	vrank_geq1_rdft2_P *pln;
	plan *cld;
	int vdim;
	iodim *d;
	INT rvs, cvs;

	static const plan_adt padt = {
		fftw_rdft2_solve, vrank_geq1_rdft2_awake, vrank_geq1_rdft2_print, vrank_geq1_rdft2_destroy
	};

	if (!vrank_geq1_rdft2_applicable(ego_, p_, plnr, &vdim))
		return (plan *)0;
	p = (const problem_rdft2 *)p_;

	d = p->vecsz->dims + vdim;

	A(d->n > 1);  /* or else, p->ri + d->is etc. are invalid */

	fftw_rdft2_strides(p->kind, d, &rvs, &cvs);

	cld = fftw_mkplan_d(plnr,
		fftw_mkproblem_rdft2_d(
			fftw_tensor_copy(p->sz),
			fftw_tensor_copy_except(p->vecsz, vdim),
			TAINT(p->r0, rvs), TAINT(p->r1, rvs),
			TAINT(p->cr, cvs), TAINT(p->ci, cvs),
			p->kind));
	if (!cld) return (plan *)0;

	pln = MKPLAN_RDFT2(vrank_geq1_rdft2_P, &padt, vrank_geq1_rdft2_apply);

	pln->cld = cld;
	pln->vl = d->n;
	pln->rvs = rvs;
	pln->cvs = cvs;

	pln->solver = ego;
	fftw_ops_zero(&pln->super.super.ops);
	pln->super.super.ops.other = 3.14159; /* magic to prefer codelet loops */
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	if (p->sz->rnk != 1 || (p->sz->dims[0].n > 128))
		pln->super.super.pcost = pln->vl * cld->pcost;

	return &(pln->super.super);
}

static solver *vrank_geq1_rdft2_mksolver(int vecloop_dim, const int *buddies, size_t nbuddies) {
	static const solver_adt sadt = { PROBLEM_RDFT2, vrank_geq1_rdft2_mkplan, 0 };
	vrank_geq1_rdft2_S *slv = MKSOLVER(vrank_geq1_rdft2_S, &sadt);
	slv->vecloop_dim = vecloop_dim;
	slv->buddies = buddies;
	slv->nbuddies = nbuddies;
	return &(slv->super);
}

void fftw_rdft2_vrank_geq1_register(planner *p) {
	/* FIXME: Should we try other vecloop_dim values? */
	static const int buddies[] = { 1, -1 };
	size_t i;

	for (i = 0; i < NELEM(buddies); ++i)
		REGISTER_SOLVER(p, vrank_geq1_rdft2_mksolver(buddies[i], buddies, NELEM(buddies)));
}


static const solvtab reodft_conf_s =
{
#if 0 /* 1 to enable "standard" algorithms with substandard accuracy;
you must also add them to Makefile.am to compile these files*/
	SOLVTAB(fftw_redft00e_r2hc_register),
	SOLVTAB(fftw_rodft00e_r2hc_register),
	SOLVTAB(fftw_reodft11e_r2hc_register),
#endif
	SOLVTAB(fftw_redft00e_r2hc_pad_register),
	SOLVTAB(fftw_rodft00e_r2hc_pad_register),
	SOLVTAB(fftw_reodft00e_splitradix_register),
	SOLVTAB(fftw_reodft010e_r2hc_register),
	SOLVTAB(fftw_reodft11e_radix2_r2hc_register),
	SOLVTAB(fftw_reodft11e_r2hc_odd_register),

	SOLVTAB_END
};

void fftw_reodft_conf_standard(planner *p) {
	fftw_solvtab_exec(reodft_conf_s, p);
}

/* Do a REDFT00 problem via an R2HC problem, with some pre/post-processing.

This code uses the trick from FFTPACK, also documented in a similar
form by Numerical Recipes.  Unfortunately, this algorithm seems to
have intrinsic numerical problems (similar to those in
reodft11e-r2hc.c), possibly due to the fact that it multiplies its
input by a cosine, causing a loss of precision near the zero.  For
transforms of 16k points, it has already lost three or four decimal
places of accuracy, which we deem unacceptable.

So, we have abandoned this algorithm in favor of the one in
redft00-r2hc-pad.c, which unfortunately sacrifices 30-50% in speed.
The only other alternative in the literature that does not have
similar numerical difficulties seems to be the direct adaptation of
the Cooley-Tukey decomposition for symmetric data, but this would
require a whole new set of codelets and it's not clear that it's
worth it at this point.  However, we did implement the latter
algorithm for the specific case of odd n (logically adapting the
split-radix algorithm); see reodft00e-splitradix.c. */


typedef struct {
	solver super;
} redft00e_r2hc_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	twid *td;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
} redft00e_r2hc_P;

static void redft00e_r2hc_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const redft00e_r2hc_P *ego = (const redft00e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;
	E csum;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[0] + I[is * n];
		csum = I[0] - I[is * n];
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb;
			a = I[is * i];
			b = I[is * (n - i)];
			csum += W[2 * i] * (amb = K(2.0) * (a - b));
			amb = W[2 * i + 1] * amb;
			apb = (a + b);
			buf[i] = apb - amb;
			buf[n - i] = apb + amb;
		}
		if (i == n - i) {
			buf[i] = K(2.0) * I[is * i];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* FIXME: use recursive/cascade summation for better stability? */
		O[0] = buf[0];
		O[os] = csum;
		for (i = 1; i + i < n; ++i) {
			INT k = i + i;
			O[os * k] = buf[i];
			O[os * (k + 1)] = O[os * (k - 1)] - buf[n - i];
		}
		if (i + i == n) {
			O[os * n] = buf[i];
		}
	}

	fftw_ifree(buf);
}

static void redft00e_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	redft00e_r2hc_P *ego = (redft00e_r2hc_P *)ego_;
	static const tw_instr redft00e_tw[] = {
		{ TW_COS,  0, 1 },
		{ TW_SIN,  0, 1 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_plan_awake(ego->cld, wakefulness);
	fftw_twiddle_awake(wakefulness,
		&ego->td, redft00e_tw, 2 * ego->n, 1, (ego->n + 1) / 2);
}

static void redft00e_r2hc_destroy(plan *ego_) {
	redft00e_r2hc_P *ego = (redft00e_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void redft00e_r2hc_print(const plan *ego_, printer *p) {
	const redft00e_r2hc_P *ego = (const redft00e_r2hc_P *)ego_;
	p->print(p, "(redft00e-r2hc-%D%v%(%p%))", ego->n + 1, ego->vl, ego->cld);
}

static int redft00e_r2hc_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->kind[0] == REDFT00
		&& p->sz->dims[0].n > 1  /* n == 1 is not well-defined */
		);
}

static int redft00e_r2hc_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && redft00e_r2hc_applicable0(ego, p));
}

static plan *redft00e_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	redft00e_r2hc_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, redft00e_r2hc_awake, redft00e_r2hc_print, redft00e_r2hc_destroy
	};

	if (!redft00e_r2hc_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n - 1;
	A(n > 0);
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	pln = MKPLAN_RDFT(redft00e_r2hc_P, &padt, redft00e_r2hc_apply);

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->td = 0;

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.other = 8 + (n - 1) / 2 * 11 + (1 - n % 2) * 5;
	ops.add = 2 + (n - 1) / 2 * 5;
	ops.mul = (n - 1) / 2 * 3 + (1 - n % 2) * 1;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *redft00e_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, redft00e_r2hc_mkplan, 0 };
	redft00e_r2hc_S *slv = MKSOLVER(redft00e_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_redft00e_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, redft00e_r2hc_mksolver());
}

/* Do a REDFT00 problem via an R2HC problem, padded symmetrically to
twice the size.  This is asymptotically a factor of ~2 worse than
redft00e-r2hc.c (the algorithm used in e.g. FFTPACK and Numerical
Recipes), but we abandoned the latter after we discovered that it
has intrinsic accuracy problems. */


typedef struct {
	solver super;
} redft00e_r2hc_pad_S;

typedef struct {
	plan_rdft super;
	plan *cld, *cldcpy;
	INT is;
	INT n;
	INT vl;
	INT ivs, ovs;
} redft00e_r2hc_pad_P;

static void redft00e_r2hc_pad_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const redft00e_r2hc_pad_P *ego = (const redft00e_r2hc_pad_P *)ego_;
	INT is = ego->is;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (2 * n), BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[0];
		for (i = 1; i < n; ++i) {
			FFTW_REAL_TYPE a = I[i * is];
			buf[i] = a;
			buf[2 * n - i] = a;
		}
		buf[i] = I[i * is]; /* i == n, Nyquist */

							/* r2hc transform of size 2*n */
		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* copy n+1 real numbers (real parts of hc array) from buf to O */
		{
			plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
			cldcpy->apply((plan *)cldcpy, buf, O);
		}
	}

	fftw_ifree(buf);
}

static void redft00e_r2hc_pad_awake(plan *ego_, enum wakefulness wakefulness) {
	redft00e_r2hc_pad_P *ego = (redft00e_r2hc_pad_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldcpy, wakefulness);
}

static void redft00e_r2hc_pad_destroy(plan *ego_) {
	redft00e_r2hc_pad_P *ego = (redft00e_r2hc_pad_P *)ego_;
	fftw_plan_destroy_internal(ego->cldcpy);
	fftw_plan_destroy_internal(ego->cld);
}

static void redft00e_r2hc_pad_print(const plan *ego_, printer *p) {
	const redft00e_r2hc_pad_P *ego = (const redft00e_r2hc_pad_P *)ego_;
	p->print(p, "(redft00e-r2hc-pad-%D%v%(%p%)%(%p%))",
		ego->n + 1, ego->vl, ego->cld, ego->cldcpy);
}

static int redft00e_r2hc_pad_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->kind[0] == REDFT00
		&& p->sz->dims[0].n > 1  /* n == 1 is not well-defined */
		);
}

static int redft00e_r2hc_pad_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && redft00e_r2hc_pad_applicable0(ego, p));
}

static plan *redft00e_r2hc_pad_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	redft00e_r2hc_pad_P *pln;
	const problem_rdft *p;
	plan *cld = (plan *)0, *cldcpy;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)0;
	INT n;
	INT vl, ivs, ovs;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, redft00e_r2hc_pad_awake, redft00e_r2hc_pad_print, redft00e_r2hc_pad_destroy
	};

	if (!redft00e_r2hc_pad_applicable(ego_, p_, plnr))
		goto nada;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n - 1;
	A(n > 0);
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (2 * n), BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(2 * n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	if (!cld)
		goto nada;

	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);
	cldcpy =
		fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_1_d(fftw_mktensor_0d(),
				fftw_mktensor_1d(n + 1, 1,
					p->sz->dims[0].os),
				buf, TAINT(p->O, ovs), R2HC));
	if (!cldcpy)
		goto nada;

	fftw_ifree(buf);

	pln = MKPLAN_RDFT(redft00e_r2hc_pad_P, &padt, redft00e_r2hc_pad_apply);

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->vl = vl;
	pln->ivs = ivs;
	pln->ovs = ovs;

	fftw_ops_zero(&ops);
	ops.other = n + 2 * n; /* loads + stores (input -> buf) */

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cldcpy->ops, &pln->super.super.ops);

	return &(pln->super.super);

nada:
	fftw_ifree0(buf);
	if (cld)
		fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

/* constructor */
static solver *redft00e_r2hc_pad_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, redft00e_r2hc_pad_mkplan, 0 };
	redft00e_r2hc_pad_S *slv = MKSOLVER(redft00e_r2hc_pad_S, &sadt);
	return &(slv->super);
}

void fftw_redft00e_r2hc_pad_register(planner *p) {
	REGISTER_SOLVER(p, redft00e_r2hc_pad_mksolver());
}

/* Do an R{E,O}DFT00 problem (of an odd length n) recursively via an
R{E,O}DFT00 problem and an RDFT problem of half the length.

This works by "logically" expanding the array to a real-even/odd DFT of
length 2n-/+2 and then applying the split-radix algorithm.

In this way, we can avoid having to pad to twice the length
(ala redft00-r2hc-pad), saving a factor of ~2 for n=2^m+/-1,
but don't incur the accuracy loss that the "ordinary" algorithm
sacrifices (ala redft00-r2hc.c).
*/



typedef struct {
	solver super;
} reodft00e_splitradix_S;

typedef struct {
	plan_rdft super;
	plan *clde, *cldo;
	twid *td;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
} reodft00e_splitradix_P;

/* redft00 */
static void reodft00e_splitradix_apply_e(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft00e_splitradix_P *ego = (const reodft00e_splitradix_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, j, n = ego->n + 1, n2 = (n - 1) / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W - 2;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n2, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		/* do size (n-1)/2 r2hc transform of odd-indexed elements
		with stride 4, "wrapping around" end of array with even
		boundary conditions */
		for (j = 0, i = 1; i < n; i += 4)
			buf[j++] = I[is * i];
		for (i = 2 * n - 2 - i; i > 0; i -= 4)
			buf[j++] = I[is * i];
		{
			plan_rdft *cld = (plan_rdft *)ego->cldo;
			cld->apply((plan *)cld, buf, buf);
		}

		/* do size (n+1)/2 redft00 of the even-indexed elements,
		writing to O: */
		{
			plan_rdft *cld = (plan_rdft *)ego->clde;
			cld->apply((plan *)cld, I, O);
		}

		/* combine the results with the twiddle factors to get output */
		{ /* DC element */
			E b20 = O[0], b0 = K(2.0) * buf[0];
			O[0] = b20 + b0;
			O[2 * (n2 * os)] = b20 - b0;
			/* O[n2*os] = O[n2*os]; */
		}
		for (i = 1; i < n2 - i; ++i) {
			E ap, am, br, bi, wr, wi, wbr, wbi;
			br = buf[i];
			bi = buf[n2 - i];
			wr = W[2 * i];
			wi = W[2 * i + 1];
#if FFT_SIGN == -1
			wbr = K(2.0) * (wr * br + wi * bi);
			wbi = K(2.0) * (wr * bi - wi * br);
#else
			wbr = K(2.0) * (wr*br - wi*bi);
			wbi = K(2.0) * (wr*bi + wi*br);
#endif
			ap = O[i * os];
			O[i * os] = ap + wbr;
			O[(2 * n2 - i) * os] = ap - wbr;
			am = O[(n2 - i) * os];
#if FFT_SIGN == -1
			O[(n2 - i) * os] = am - wbi;
			O[(n2 + i) * os] = am + wbi;
#else
			O[(n2 - i)*os] = am + wbi;
			O[(n2 + i)*os] = am - wbi;
#endif
		}
		if (i == n2 - i) { /* Nyquist element */
			E ap, wbr;
			wbr = K(2.0) * (W[2 * i] * buf[i]);
			ap = O[i * os];
			O[i * os] = ap + wbr;
			O[(2 * n2 - i) * os] = ap - wbr;
		}
	}

	fftw_ifree(buf);
}

/* rodft00 */
static void reodft00e_splitradix_apply_o(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft00e_splitradix_P *ego = (const reodft00e_splitradix_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, j, n = ego->n - 1, n2 = (n + 1) / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W - 2;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n2, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		/* do size (n+1)/2 r2hc transform of even-indexed elements
		with stride 4, "wrapping around" end of array with odd
		boundary conditions */
		for (j = 0, i = 0; i < n; i += 4)
			buf[j++] = I[is * i];
		for (i = 2 * n - i; i > 0; i -= 4)
			buf[j++] = -I[is * i];
		{
			plan_rdft *cld = (plan_rdft *)ego->cldo;
			cld->apply((plan *)cld, buf, buf);
		}

		/* do size (n-1)/2 rodft00 of the odd-indexed elements,
		writing to O: */
		{
			plan_rdft *cld = (plan_rdft *)ego->clde;
			if (I == O) {
				/* can't use I+is and I, subplan would lose in-placeness */
				cld->apply((plan *)cld, I + is, I + is);
				/* we could maybe avoid this copy by modifying the
				twiddle loop, but currently I can't be bothered. */
				A(is >= os);
				for (i = 0; i < n2 - 1; ++i)
					O[os * i] = I[is * (i + 1)];
			}
			else
				cld->apply((plan *)cld, I + is, O);
		}

		/* combine the results with the twiddle factors to get output */
		O[(n2 - 1) * os] = K(2.0) * buf[0];
		for (i = 1; i < n2 - i; ++i) {
			E ap, am, br, bi, wr, wi, wbr, wbi;
			br = buf[i];
			bi = buf[n2 - i];
			wr = W[2 * i];
			wi = W[2 * i + 1];
#if FFT_SIGN == -1
			wbr = K(2.0) * (wr * br + wi * bi);
			wbi = K(2.0) * (wi * br - wr * bi);
#else
			wbr = K(2.0) * (wr*br - wi*bi);
			wbi = K(2.0) * (wr*bi + wi*br);
#endif
			ap = O[(i - 1) * os];
			O[(i - 1) * os] = wbi + ap;
			O[(2 * n2 - 1 - i) * os] = wbi - ap;
			am = O[(n2 - 1 - i) * os];
#if FFT_SIGN == -1
			O[(n2 - 1 - i) * os] = wbr + am;
			O[(n2 - 1 + i) * os] = wbr - am;
#else
			O[(n2 - 1 - i)*os] = wbr + am;
			O[(n2 - 1 + i)*os] = wbr - am;
#endif
		}
		if (i == n2 - i) { /* Nyquist element */
			E ap, wbi;
			wbi = K(2.0) * (W[2 * i + 1] * buf[i]);
			ap = O[(i - 1) * os];
			O[(i - 1) * os] = wbi + ap;
			O[(2 * n2 - 1 - i) * os] = wbi - ap;
		}
	}

	fftw_ifree(buf);
}

static void reodft00e_splitradix_awake(plan *ego_, enum wakefulness wakefulness) {
	reodft00e_splitradix_P *ego = (reodft00e_splitradix_P *)ego_;
	static const tw_instr reodft00e_tw[] = {
		{ TW_COS,  1, 1 },
		{ TW_SIN,  1, 1 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_plan_awake(ego->clde, wakefulness);
	fftw_plan_awake(ego->cldo, wakefulness);
	fftw_twiddle_awake(wakefulness, &ego->td, reodft00e_tw,
		2 * ego->n, 1, ego->n / 4);
}

static void reodft00e_splitradix_destroy(plan *ego_) {
	reodft00e_splitradix_P *ego = (reodft00e_splitradix_P *)ego_;
	fftw_plan_destroy_internal(ego->cldo);
	fftw_plan_destroy_internal(ego->clde);
}

static void reodft00e_splitradix_print(const plan *ego_, printer *p) {
	const reodft00e_splitradix_P *ego = (const reodft00e_splitradix_P *)ego_;
	if (ego->super.apply == reodft00e_splitradix_apply_e)
		p->print(p, "(redft00e-splitradix-%D%v%(%p%)%(%p%))",
			ego->n + 1, ego->vl, ego->clde, ego->cldo);
	else
		p->print(p, "(rodft00e-splitradix-%D%v%(%p%)%(%p%))",
			ego->n - 1, ego->vl, ego->clde, ego->cldo);
}

static int reodft00e_splitradix_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& (p->kind[0] == REDFT00 || p->kind[0] == RODFT00)
		&& p->sz->dims[0].n > 1  /* don't create size-0 sub-plans */
		&& p->sz->dims[0].n % 2  /* odd: 4 divides "logical" DFT */
		&& (p->I != p->O || p->vecsz->rnk == 0
			|| p->vecsz->dims[0].is == p->vecsz->dims[0].os)
		&& (p->kind[0] != RODFT00 || p->I != p->O ||
			p->sz->dims[0].is >= p->sz->dims[0].os) /* laziness */
		);
}

static int reodft00e_splitradix_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && reodft00e_splitradix_applicable0(ego, p));
}

static plan *reodft00e_splitradix_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	reodft00e_splitradix_P *pln;
	const problem_rdft *p;
	plan *clde, *cldo;
	FFTW_REAL_TYPE *buf;
	INT n, n0;
	opcnt ops;
	int inplace_odd;

	static const plan_adt padt = {
		fftw_rdft_solve, reodft00e_splitradix_awake, reodft00e_splitradix_print, reodft00e_splitradix_destroy
	};

	if (!reodft00e_splitradix_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = (n0 = p->sz->dims[0].n) + (p->kind[0] == REDFT00 ? (INT)-1 : (INT)1);
	A(n > 0 && n % 2 == 0);
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (n / 2), BUFFERS);

	inplace_odd = p->kind[0] == RODFT00 && p->I == p->O;
	clde = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(
		fftw_mktensor_1d(n0 - n / 2, 2 * p->sz->dims[0].is,
			inplace_odd ? p->sz->dims[0].is
			: p->sz->dims[0].os),
		fftw_mktensor_0d(),
		TAINT(p->I
			+ p->sz->dims[0].is * (p->kind[0] == RODFT00),
			p->vecsz->rnk ? p->vecsz->dims[0].is : 0),
		TAINT(p->O
			+ p->sz->dims[0].is * inplace_odd,
			p->vecsz->rnk ? p->vecsz->dims[0].os : 0),
		p->kind[0]));
	if (!clde) {
		fftw_ifree(buf);
		return (plan *)0;
	}

	cldo = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(
		fftw_mktensor_1d(n / 2, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cldo)
		return (plan *)0;

	pln = MKPLAN_RDFT(reodft00e_splitradix_P, &padt,
		p->kind[0] == REDFT00 ? reodft00e_splitradix_apply_e : reodft00e_splitradix_apply_o);

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->clde = clde;
	pln->cldo = cldo;
	pln->td = 0;

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.other = n / 2;
	ops.add = (p->kind[0] == REDFT00 ? (INT)2 : (INT)0) +
		(n / 2 - 1) / 2 * 6 + ((n / 2) % 2 == 0) * 2;
	ops.mul = 1 + (n / 2 - 1) / 2 * 6 + ((n / 2) % 2 == 0) * 2;

	/* tweak ops.other so that r2hc-pad is used for small sizes, which
	seems to be a lot faster on my machine: */
	ops.other += 256;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &clde->ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cldo->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *reodft00e_splitradix_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, reodft00e_splitradix_mkplan, 0 };
	reodft00e_splitradix_S *slv = MKSOLVER(reodft00e_splitradix_S, &sadt);
	return &(slv->super);
}

void fftw_reodft00e_splitradix_register(planner *p) {
	REGISTER_SOLVER(p, reodft00e_splitradix_mksolver());
}

/* Do an R{E,O}DFT{01,10} problem via an R2HC problem, with some
pre/post-processing ala FFTPACK. */


typedef struct {
	solver super;
} reodft010e_r2hc_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	twid *td;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
	rdft_kind kind;
} reodft010e_r2hc_P;

/* A real-even-01 DFT operates logically on a size-4N array:
I 0 -r(I*) -I 0 r(I*),
where r denotes reversal and * denotes deletion of the 0th element.
To compute the transform of this, we imagine performing a radix-4
(real-input) DIF step, which turns the size-4N DFT into 4 size-N
(contiguous) DFTs, two of which are zero and two of which are
conjugates.  The non-redundant size-N DFT has halfcomplex input, so
we can do it with a size-N hc2r transform.  (In order to share
plans with the re10 (inverse) transform, however, we use the DHT
trick to re-express the hc2r problem as r2hc.  This has little cost
since we are already pre- and post-processing the data in {i,n-i}
order.)  Finally, we have to write out the data in the correct
order...the two size-N redundant (conjugate) hc2r DFTs correspond
to the even and odd outputs in O (i.e. the usual interleaved output
of DIF transforms); since this data has even symmetry, we only
write the first half of it.

The real-even-10 DFT is just the reverse of these steps, i.e. a
radix-4 DIT transform.  There, however, we just use the r2hc
transform naturally without resorting to the DHT trick.

A real-odd-01 DFT is very similar, except that the input is
0 I (rI)* 0 -I -(rI)*.  This format, however, can be transformed
into precisely the real-even-01 format above by sending I -> rI
and shifting the array by N.  The former swap is just another
transformation on the input during preprocessing; the latter
multiplies the even/odd outputs by i/-i, which combines with
the factor of -i (to take the imaginary part) to simply flip
the sign of the odd outputs.  Vice-versa for real-odd-10.

The FFTPACK source code was very helpful in working this out.
(They do unnecessary passes over the array, though.)  The same
algorithm is also described in:

John Makhoul, "A fast cosine transform in one and two dimensions,"
IEEE Trans. on Acoust. Speech and Sig. Proc., ASSP-28 (1), 27--34 (1980).

Note that Numerical Recipes suggests a different algorithm that
requires more operations and uses trig. functions for both the pre-
and post-processing passes.
*/

static void reodft010e_r2hc_apply_re01(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft010e_r2hc_P *ego = (const reodft010e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[0];
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb, wa, wb;
			a = I[is * i];
			b = I[is * (n - i)];
			apb = a + b;
			amb = a - b;
			wa = W[2 * i];
			wb = W[2 * i + 1];
			buf[i] = wa * amb + wb * apb;
			buf[n - i] = wa * apb - wb * amb;
		}
		if (i == n - i) {
			buf[i] = K(2.0) * I[is * i] * W[2 * i];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		O[0] = buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b;
			INT k;
			a = buf[i];
			b = buf[n - i];
			k = i + i;
			O[os * (k - 1)] = a - b;
			O[os * k] = a + b;
		}
		if (i == n - i) {
			O[os * (n - 1)] = buf[i];
		}
	}

	fftw_ifree(buf);
}

/* ro01 is same as re01, but with i <-> n - 1 - i in the input and
the sign of the odd output elements flipped. */
static void reodft010e_r2hc_apply_ro01(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft010e_r2hc_P *ego = (const reodft010e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[is * (n - 1)];
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb, wa, wb;
			a = I[is * (n - 1 - i)];
			b = I[is * (i - 1)];
			apb = a + b;
			amb = a - b;
			wa = W[2 * i];
			wb = W[2 * i + 1];
			buf[i] = wa * amb + wb * apb;
			buf[n - i] = wa * apb - wb * amb;
		}
		if (i == n - i) {
			buf[i] = K(2.0) * I[is * (i - 1)] * W[2 * i];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		O[0] = buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b;
			INT k;
			a = buf[i];
			b = buf[n - i];
			k = i + i;
			O[os * (k - 1)] = b - a;
			O[os * k] = a + b;
		}
		if (i == n - i) {
			O[os * (n - 1)] = -buf[i];
		}
	}

	fftw_ifree(buf);
}

static void reodft010e_r2hc_apply_re10(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft010e_r2hc_P *ego = (const reodft010e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[0];
		for (i = 1; i < n - i; ++i) {
			E u, v;
			INT k = i + i;
			u = I[is * (k - 1)];
			v = I[is * k];
			buf[n - i] = u;
			buf[i] = v;
		}
		if (i == n - i) {
			buf[i] = I[is * (n - 1)];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		O[0] = K(2.0) * buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b, wa, wb;
			a = K(2.0) * buf[i];
			b = K(2.0) * buf[n - i];
			wa = W[2 * i];
			wb = W[2 * i + 1];
			O[os * i] = wa * a + wb * b;
			O[os * (n - i)] = wb * a - wa * b;
		}
		if (i == n - i) {
			O[os * i] = K(2.0) * buf[i] * W[2 * i];
		}
	}

	fftw_ifree(buf);
}

/* ro10 is same as re10, but with i <-> n - 1 - i in the output and
the sign of the odd input elements flipped. */
static void reodft010e_r2hc_apply_ro10(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft010e_r2hc_P *ego = (const reodft010e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = I[0];
		for (i = 1; i < n - i; ++i) {
			E u, v;
			INT k = i + i;
			u = -I[is * (k - 1)];
			v = I[is * k];
			buf[n - i] = u;
			buf[i] = v;
		}
		if (i == n - i) {
			buf[i] = -I[is * (n - 1)];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		O[os * (n - 1)] = K(2.0) * buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b, wa, wb;
			a = K(2.0) * buf[i];
			b = K(2.0) * buf[n - i];
			wa = W[2 * i];
			wb = W[2 * i + 1];
			O[os * (n - 1 - i)] = wa * a + wb * b;
			O[os * (i - 1)] = wb * a - wa * b;
		}
		if (i == n - i) {
			O[os * (i - 1)] = K(2.0) * buf[i] * W[2 * i];
		}
	}

	fftw_ifree(buf);
}

static void reodft010e_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	reodft010e_r2hc_P *ego = (reodft010e_r2hc_P *)ego_;
	static const tw_instr reodft010e_tw[] = {
		{ TW_COS,  0, 1 },
		{ TW_SIN,  0, 1 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_plan_awake(ego->cld, wakefulness);

	fftw_twiddle_awake(wakefulness, &ego->td, reodft010e_tw,
		4 * ego->n, 1, ego->n / 2 + 1);
}

static void reodft010e_r2hc_destroy(plan *ego_) {
	reodft010e_r2hc_P *ego = (reodft010e_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void reodft010e_r2hc_print(const plan *ego_, printer *p) {
	const reodft010e_r2hc_P *ego = (const reodft010e_r2hc_P *)ego_;
	p->print(p, "(%se-r2hc-%D%v%(%p%))",
		fftw_rdft_kind_str(ego->kind), ego->n, ego->vl, ego->cld);
}

static int reodft010e_r2hc_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& (p->kind[0] == REDFT01 || p->kind[0] == REDFT10
			|| p->kind[0] == RODFT01 || p->kind[0] == RODFT10)
		);
}

static int reodft010e_r2hc_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && reodft010e_r2hc_applicable0(ego, p));
}

static plan *reodft010e_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	reodft010e_r2hc_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, reodft010e_r2hc_awake, reodft010e_r2hc_print, reodft010e_r2hc_destroy
	};

	if (!reodft010e_r2hc_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	switch (p->kind[0]) {
	case REDFT01:
		pln = MKPLAN_RDFT(reodft010e_r2hc_P, &padt, reodft010e_r2hc_apply_re01);
		break;
	case REDFT10:
		pln = MKPLAN_RDFT(reodft010e_r2hc_P, &padt, reodft010e_r2hc_apply_re10);
		break;
	case RODFT01:
		pln = MKPLAN_RDFT(reodft010e_r2hc_P, &padt, reodft010e_r2hc_apply_ro01);
		break;
	case RODFT10:
		pln = MKPLAN_RDFT(reodft010e_r2hc_P, &padt, reodft010e_r2hc_apply_ro10);
		break;
	default:
		A(0);
		return (plan *)0;
	}

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->td = 0;
	pln->kind = p->kind[0];

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.other = 4 + (n - 1) / 2 * 10 + (1 - n % 2) * 5;
	if (p->kind[0] == REDFT01 || p->kind[0] == RODFT01) {
		ops.add = (n - 1) / 2 * 6;
		ops.mul = (n - 1) / 2 * 4 + (1 - n % 2) * 2;
	}
	else { /* 10 transforms */
		ops.add = (n - 1) / 2 * 2;
		ops.mul = 1 + (n - 1) / 2 * 6 + (1 - n % 2) * 2;
	}

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *reodft010e_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, reodft010e_r2hc_mkplan, 0 };
	reodft010e_r2hc_S *slv = MKSOLVER(reodft010e_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_reodft010e_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, reodft010e_r2hc_mksolver());
}

/* Do an R{E,O}DFT11 problem via an R2HC problem, with some
pre/post-processing ala FFTPACK.  Use a trick from:

S. C. Chan and K. L. Ho, "Direct methods for computing discrete
sinusoidal transforms," IEE Proceedings F 137 (6), 433--442 (1990).

to re-express as an REDFT01 (DCT-III) problem.

NOTE: We no longer use this algorithm, because it turns out to suffer
a catastrophic loss of accuracy for certain inputs, apparently because
its post-processing multiplies the output by a cosine.  Near the zero
of the cosine, the REDFT01 must produce a near-singular output.
*/


typedef struct {
	solver super;
} reodft11e_r2hc_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	twid *td, *td2;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
	rdft_kind kind;
} reodft11e_r2hc_P;

static void reodft11e_r2hc_apply_re11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_r2hc_P *ego = (const reodft11e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W;
	FFTW_REAL_TYPE *buf;
	E cur;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		/* I wish that this didn't require an extra pass. */
		/* FIXME: use recursive/cascade summation for better stability? */
		buf[n - 1] = cur = K(2.0) * I[is * (n - 1)];
		for (i = n - 1; i > 0; --i) {
			E curnew;
			buf[(i - 1)] = curnew = K(2.0) * I[is * (i - 1)] - cur;
			cur = curnew;
		}

		W = ego->td->W;
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb, wa, wb;
			a = buf[i];
			b = buf[n - i];
			apb = a + b;
			amb = a - b;
			wa = W[2 * i];
			wb = W[2 * i + 1];
			buf[i] = wa * amb + wb * apb;
			buf[n - i] = wa * apb - wb * amb;
		}
		if (i == n - i) {
			buf[i] = K(2.0) * buf[i] * W[2 * i];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		W = ego->td2->W;
		O[0] = W[0] * buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b;
			INT k;
			a = buf[i];
			b = buf[n - i];
			k = i + i;
			O[os * (k - 1)] = W[k - 1] * (a - b);
			O[os * k] = W[k] * (a + b);
		}
		if (i == n - i) {
			O[os * (n - 1)] = W[n - 1] * buf[i];
		}
	}

	fftw_ifree(buf);
}

/* like for rodft01, rodft11 is obtained from redft11 by
reversing the input and flipping the sign of every other output. */
static void reodft11e_r2hc_apply_ro11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_r2hc_P *ego = (const reodft11e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W;
	FFTW_REAL_TYPE *buf;
	E cur;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		/* I wish that this didn't require an extra pass. */
		/* FIXME: use recursive/cascade summation for better stability? */
		buf[n - 1] = cur = K(2.0) * I[0];
		for (i = n - 1; i > 0; --i) {
			E curnew;
			buf[(i - 1)] = curnew = K(2.0) * I[is * (n - i)] - cur;
			cur = curnew;
		}

		W = ego->td->W;
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb, wa, wb;
			a = buf[i];
			b = buf[n - i];
			apb = a + b;
			amb = a - b;
			wa = W[2 * i];
			wb = W[2 * i + 1];
			buf[i] = wa * amb + wb * apb;
			buf[n - i] = wa * apb - wb * amb;
		}
		if (i == n - i) {
			buf[i] = K(2.0) * buf[i] * W[2 * i];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		W = ego->td2->W;
		O[0] = W[0] * buf[0];
		for (i = 1; i < n - i; ++i) {
			E a, b;
			INT k;
			a = buf[i];
			b = buf[n - i];
			k = i + i;
			O[os * (k - 1)] = W[k - 1] * (b - a);
			O[os * k] = W[k] * (a + b);
		}
		if (i == n - i) {
			O[os * (n - 1)] = -W[n - 1] * buf[i];
		}
	}

	fftw_ifree(buf);
}

static void reodft11e_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	reodft11e_r2hc_P *ego = (reodft11e_r2hc_P *)ego_;
	static const tw_instr reodft010e_tw[] = {
		{ TW_COS,  0, 1 },
		{ TW_SIN,  0, 1 },
		{ TW_NEXT, 1, 0 }
	};
	static const tw_instr reodft11e_tw[] = {
		{ TW_COS,  1, 1 },
		{ TW_NEXT, 2, 0 }
	};

	fftw_plan_awake(ego->cld, wakefulness);

	fftw_twiddle_awake(wakefulness,
		&ego->td, reodft010e_tw, 4 * ego->n, 1, ego->n / 2 + 1);
	fftw_twiddle_awake(wakefulness,
		&ego->td2, reodft11e_tw, 8 * ego->n, 1, ego->n * 2);
}

static void reodft11e_r2hc_destroy(plan *ego_) {
	reodft11e_r2hc_P *ego = (reodft11e_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void reodft11e_r2hc_print(const plan *ego_, printer *p) {
	const reodft11e_r2hc_P *ego = (const reodft11e_r2hc_P *)ego_;
	p->print(p, "(%se-r2hc-%D%v%(%p%))",
		fftw_rdft_kind_str(ego->kind), ego->n, ego->vl, ego->cld);
}

static int reodft11e_r2hc_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;

	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& (p->kind[0] == REDFT11 || p->kind[0] == RODFT11)
		);
}

static int reodft11e_r2hc_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && reodft11e_r2hc_applicable0(ego, p));
}

static plan *reodft11e_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	reodft11e_r2hc_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, reodft11e_r2hc_awake, reodft11e_r2hc_print, reodft11e_r2hc_destroy
	};

	if (!reodft11e_r2hc_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	pln = MKPLAN_RDFT(reodft11e_r2hc_P, &padt,
		p->kind[0] == REDFT11 ? reodft11e_r2hc_apply_re11 : reodft11e_r2hc_apply_ro11);
	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->td = pln->td2 = 0;
	pln->kind = p->kind[0];

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.other = 5 + (n - 1) * 2 + (n - 1) / 2 * 12 + (1 - n % 2) * 6;
	ops.add = (n - 1) * 1 + (n - 1) / 2 * 6;
	ops.mul = 2 + (n - 1) * 1 + (n - 1) / 2 * 6 + (1 - n % 2) * 3;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *reodft11e_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, reodft11e_r2hc_mkplan, 0 };
	reodft11e_r2hc_S *slv = MKSOLVER(reodft11e_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_reodft11e_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, reodft11e_r2hc_mksolver());
}

/* Do an R{E,O}DFT11 problem via an R2HC problem of the same *odd* size,
with some permutations and post-processing, as described in:

S. C. Chan and K. L. Ho, "Fast algorithms for computing the
discrete cosine transform," IEEE Trans. Circuits Systems II:
Analog & Digital Sig. Proc. 39 (3), 185--190 (1992).

(For even sizes, see reodft11e-radix2.c.)

This algorithm is related to the 8 x n prime-factor-algorithm (PFA)
decomposition of the size 8n "logical" DFT corresponding to the
R{EO}DFT11.

Aside from very confusing notation (several symbols are redefined
from one line to the next), be aware that this paper has some
errors.  In particular, the signs are wrong in Eqs. (34-35).  Also,
Eqs. (36-37) should be simply C(k) = C(2k + 1 mod N), and similarly
for S (or, equivalently, the second cases should have 2*N - 2*k - 1
instead of N - k - 1).  Note also that in their definition of the
DFT, similarly to FFTW's, the exponent's sign is -1, but they
forgot to correspondingly multiply S (the sine terms) by -1.
*/


typedef struct {
	solver super;
} reodft11e_r2hc_odd_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
	rdft_kind kind;
} reodft11e_r2hc_odd_P;

static DK(SQRT2, +1.4142135623730950488016887242096980785696718753769);

#define SGN_SET(x, i) ((i) % 2 ? -(x) : (x))

static void reodft11e_r2hc_odd_apply_re11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_r2hc_odd_P *ego = (const reodft11e_r2hc_odd_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n, n2 = n / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		{
			INT m;
			for (i = 0, m = n2; m < n; ++i, m += 4)
				buf[i] = I[is * m];
			for (; m < 2 * n; ++i, m += 4)
				buf[i] = -I[is * (2 * n - m - 1)];
			for (; m < 3 * n; ++i, m += 4)
				buf[i] = -I[is * (m - 2 * n)];
			for (; m < 4 * n; ++i, m += 4)
				buf[i] = I[is * (4 * n - m - 1)];
			m -= 4 * n;
			for (; i < n; ++i, m += 4)
				buf[i] = I[is * m];
		}

		{ /* child plan: R2HC of size n */
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* FIXME: strength-reduce loop by 4 to eliminate ugly sgn_set? */
		for (i = 0; i + i + 1 < n2; ++i) {
			INT k = i + i + 1;
			E c1, s1;
			E c2, s2;
			c1 = buf[k];
			c2 = buf[k + 1];
			s2 = buf[n - (k + 1)];
			s1 = buf[n - k];

			O[os * i] = SQRT2 * (SGN_SET(c1, (i + 1) / 2) +
				SGN_SET(s1, i / 2));
			O[os * (n - (i + 1))] = SQRT2 * (SGN_SET(c1, (n - i) / 2) -
				SGN_SET(s1, (n - (i + 1)) / 2));

			O[os * (n2 - (i + 1))] = SQRT2 * (SGN_SET(c2, (n2 - i) / 2) -
				SGN_SET(s2, (n2 - (i + 1)) / 2));
			O[os * (n2 + (i + 1))] = SQRT2 * (SGN_SET(c2, (n2 + i + 2) / 2) +
				SGN_SET(s2, (n2 + (i + 1)) / 2));
		}
		if (i + i + 1 == n2) {
			E c, s;
			c = buf[n2];
			s = buf[n - n2];
			O[os * i] = SQRT2 * (SGN_SET(c, (i + 1) / 2) +
				SGN_SET(s, i / 2));
			O[os * (n - (i + 1))] = SQRT2 * (SGN_SET(c, (i + 2) / 2) +
				SGN_SET(s, (i + 1) / 2));
		}
		O[os * n2] = SQRT2 * SGN_SET(buf[0], (n2 + 1) / 2);
	}

	fftw_ifree(buf);
}

/* like for rodft01, rodft11 is obtained from redft11 by
reversing the input and flipping the sign of every other output. */
static void reodft11e_r2hc_odd_apply_ro11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_r2hc_odd_P *ego = (const reodft11e_r2hc_odd_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n, n2 = n / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		{
			INT m;
			for (i = 0, m = n2; m < n; ++i, m += 4)
				buf[i] = I[is * (n - 1 - m)];
			for (; m < 2 * n; ++i, m += 4)
				buf[i] = -I[is * (m - n)];
			for (; m < 3 * n; ++i, m += 4)
				buf[i] = -I[is * (3 * n - 1 - m)];
			for (; m < 4 * n; ++i, m += 4)
				buf[i] = I[is * (m - 3 * n)];
			m -= 4 * n;
			for (; i < n; ++i, m += 4)
				buf[i] = I[is * (n - 1 - m)];
		}

		{ /* child plan: R2HC of size n */
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* FIXME: strength-reduce loop by 4 to eliminate ugly sgn_set? */
		for (i = 0; i + i + 1 < n2; ++i) {
			INT k = i + i + 1;
			INT j;
			E c1, s1;
			E c2, s2;
			c1 = buf[k];
			c2 = buf[k + 1];
			s2 = buf[n - (k + 1)];
			s1 = buf[n - k];

			O[os * i] = SQRT2 * (SGN_SET(c1, (i + 1) / 2 + i) +
				SGN_SET(s1, i / 2 + i));
			O[os * (n - (i + 1))] = SQRT2 * (SGN_SET(c1, (n - i) / 2 + i) -
				SGN_SET(s1, (n - (i + 1)) / 2 + i));

			j = n2 - (i + 1);
			O[os * j] = SQRT2 * (SGN_SET(c2, (n2 - i) / 2 + j) -
				SGN_SET(s2, (n2 - (i + 1)) / 2 + j));
			O[os * (n2 + (i + 1))] = SQRT2 * (SGN_SET(c2, (n2 + i + 2) / 2 + j) +
				SGN_SET(s2, (n2 + (i + 1)) / 2 + j));
		}
		if (i + i + 1 == n2) {
			E c, s;
			c = buf[n2];
			s = buf[n - n2];
			O[os * i] = SQRT2 * (SGN_SET(c, (i + 1) / 2 + i) +
				SGN_SET(s, i / 2 + i));
			O[os * (n - (i + 1))] = SQRT2 * (SGN_SET(c, (i + 2) / 2 + i) +
				SGN_SET(s, (i + 1) / 2 + i));
		}
		O[os * n2] = SQRT2 * SGN_SET(buf[0], (n2 + 1) / 2 + n2);
	}

	fftw_ifree(buf);
}

static void reodft11e_r2hc_odd_awake(plan *ego_, enum wakefulness wakefulness) {
	reodft11e_r2hc_odd_P *ego = (reodft11e_r2hc_odd_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
}

static void reodft11e_r2hc_odd_destroy(plan *ego_) {
	reodft11e_r2hc_odd_P *ego = (reodft11e_r2hc_odd_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void reodft11e_r2hc_odd_print(const plan *ego_, printer *p) {
	const reodft11e_r2hc_odd_P *ego = (const reodft11e_r2hc_odd_P *)ego_;
	p->print(p, "(%se-r2hc-odd-%D%v%(%p%))",
		fftw_rdft_kind_str(ego->kind), ego->n, ego->vl, ego->cld);
}

static int reodft11e_r2hc_odd_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n % 2 == 1
		&& (p->kind[0] == REDFT11 || p->kind[0] == RODFT11)
		);
}

static int reodft11e_r2hc_odd_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && reodft11e_r2hc_odd_applicable0(ego, p));
}

static plan *reodft11e_r2hc_odd_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	reodft11e_r2hc_odd_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, reodft11e_r2hc_odd_awake, reodft11e_r2hc_odd_print, reodft11e_r2hc_odd_destroy
	};

	if (!reodft11e_r2hc_odd_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	pln = MKPLAN_RDFT(reodft11e_r2hc_odd_P, &padt,
		p->kind[0] == REDFT11 ? reodft11e_r2hc_odd_apply_re11 : reodft11e_r2hc_odd_apply_ro11);
	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->kind = p->kind[0];

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.add = n - 1;
	ops.mul = n;
	ops.other = 4 * n;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *reodft11e_r2hc_odd_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, reodft11e_r2hc_odd_mkplan, 0 };
	reodft11e_r2hc_odd_S *slv = MKSOLVER(reodft11e_r2hc_odd_S, &sadt);
	return &(slv->super);
}

void fftw_reodft11e_r2hc_odd_register(planner *p) {
	REGISTER_SOLVER(p, reodft11e_r2hc_odd_mksolver());
}

/* Do an R{E,O}DFT11 problem of *even* size by a pair of R2HC problems
of half the size, plus some pre/post-processing.  Use a trick from:

Zhongde Wang, "On computing the discrete Fourier and cosine transforms,"
IEEE Trans. Acoust. Speech Sig. Proc. ASSP-33 (4), 1341--1344 (1985).

to re-express as a pair of half-size REDFT01 (DCT-III) problems.  Our
implementation looks quite a bit different from the algorithm described
in the paper because we combined the paper's pre/post-processing with
the pre/post-processing used to turn REDFT01 into R2HC.  (Also, the
paper uses a DCT/DST pair, but we turn the DST into a DCT via the
usual reordering/sign-flip trick.  We additionally combined a couple
of the matrices/transformations of the paper into a single pass.)

NOTE: We originally used a simpler method by S. C. Chan and K. L. Ho
that turned out to have numerical problems; see reodft11e-r2hc.c.

(For odd sizes, see reodft11e-r2hc-odd.c.)
*/



typedef struct {
	solver super;
} reodft11e_radix2_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	twid *td, *td2;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
	rdft_kind kind;
} reodft11e_radix2_P;

static void reodft11e_radix2_apply_re11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_radix2_P *ego = (const reodft11e_radix2_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n, n2 = n / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *W2;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = K(2.0) * I[0];
		buf[n2] = K(2.0) * I[is * (n - 1)];
		for (i = 1; i + i < n2; ++i) {
			INT k = i + i;
			E a, b, a2, b2;
			{
				E u, v;
				u = I[is * (k - 1)];
				v = I[is * k];
				a = u + v;
				b2 = u - v;
			}
			{
				E u, v;
				u = I[is * (n - k - 1)];
				v = I[is * (n - k)];
				b = u + v;
				a2 = u - v;
			}
			{
				E wa, wb;
				wa = W[2 * i];
				wb = W[2 * i + 1];
				{
					E apb, amb;
					apb = a + b;
					amb = a - b;
					buf[i] = wa * amb + wb * apb;
					buf[n2 - i] = wa * apb - wb * amb;
				}
				{
					E apb, amb;
					apb = a2 + b2;
					amb = a2 - b2;
					buf[n2 + i] = wa * amb + wb * apb;
					buf[n - i] = wa * apb - wb * amb;
				}
			}
		}
		if (i + i == n2) {
			E u, v;
			u = I[is * (n2 - 1)];
			v = I[is * n2];
			buf[i] = (u + v) * (W[2 * i] * K(2.0));
			buf[n - i] = (u - v) * (W[2 * i] * K(2.0));
		}


		/* child plan: two r2hc's of size n/2 */
		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		W2 = ego->td2->W;
		{ /* i == 0 case */
			E wa, wb;
			E a, b;
			wa = W2[0]; /* cos */
			wb = W2[1]; /* sin */
			a = buf[0];
			b = buf[n2];
			O[0] = wa * a + wb * b;
			O[os * (n - 1)] = wb * a - wa * b;
		}
		W2 += 2;
		for (i = 1; i + i < n2; ++i, W2 += 2) {
			INT k;
			E u, v, u2, v2;
			u = buf[i];
			v = buf[n2 - i];
			u2 = buf[n2 + i];
			v2 = buf[n - i];
			k = (i + i) - 1;
			{
				E wa, wb;
				E a, b;
				wa = W2[0]; /* cos */
				wb = W2[1]; /* sin */
				a = u - v;
				b = v2 - u2;
				O[os * k] = wa * a + wb * b;
				O[os * (n - 1 - k)] = wb * a - wa * b;
			}
			++k;
			W2 += 2;
			{
				E wa, wb;
				E a, b;
				wa = W2[0]; /* cos */
				wb = W2[1]; /* sin */
				a = u + v;
				b = u2 + v2;
				O[os * k] = wa * a + wb * b;
				O[os * (n - 1 - k)] = wb * a - wa * b;
			}
		}
		if (i + i == n2) {
			INT k = (i + i) - 1;
			E wa, wb;
			E a, b;
			wa = W2[0]; /* cos */
			wb = W2[1]; /* sin */
			a = buf[i];
			b = buf[n2 + i];
			O[os * k] = wa * a - wb * b;
			O[os * (n - 1 - k)] = wb * a + wa * b;
		}
	}

	fftw_ifree(buf);
}

#if 0

/* This version of apply_re11 uses REDFT01 child plans, more similar
to the original paper by Z. Wang.  We keep it around for reference
(it is simpler) and because it may become more efficient if we
ever implement REDFT01 codelets. */

static void reodft11e_radix2_apply_re11(const plan *ego_, R *I, R *O)
{
	const reodft11e_radix2_P *ego = (const reodft11e_radix2_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = K(2.0) * I[0];
		buf[n / 2] = K(2.0) * I[is * (n - 1)];
		for (i = 1; i + i < n; ++i) {
			INT k = i + i;
			E a, b;
			a = I[is * (k - 1)];
			b = I[is * k];
			buf[i] = a + b;
			buf[n - i] = a - b;
		}

		/* child plan: two redft01's (DCT-III) */
		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		W = ego->td2->W;
		for (i = 0; i + 1 < n / 2; ++i, W += 2) {
			{
				E wa, wb;
				E a, b;
				wa = W[0]; /* cos */
				wb = W[1]; /* sin */
				a = buf[i];
				b = buf[n / 2 + i];
				O[os * i] = wa * a + wb * b;
				O[os * (n - 1 - i)] = wb * a - wa * b;
			}
			++i;
			W += 2;
			{
				E wa, wb;
				E a, b;
				wa = W[0]; /* cos */
				wb = W[1]; /* sin */
				a = buf[i];
				b = buf[n / 2 + i];
				O[os * i] = wa * a - wb * b;
				O[os * (n - 1 - i)] = wb * a + wa * b;
			}
		}
		if (i < n / 2) {
			E wa, wb;
			E a, b;
			wa = W[0]; /* cos */
			wb = W[1]; /* sin */
			a = buf[i];
			b = buf[n / 2 + i];
			O[os * i] = wa * a + wb * b;
			O[os * (n - 1 - i)] = wb * a - wa * b;
		}
	}

	fftw_ifree(buf);
}

#endif /* 0 */

/* like for rodft01, rodft11 is obtained from redft11 by
reversing the input and flipping the sign of every other output. */
static void reodft11e_radix2_apply_ro11(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const reodft11e_radix2_P *ego = (const reodft11e_radix2_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n, n2 = n / 2;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *W2;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = K(2.0) * I[is * (n - 1)];
		buf[n2] = K(2.0) * I[0];
		for (i = 1; i + i < n2; ++i) {
			INT k = i + i;
			E a, b, a2, b2;
			{
				E u, v;
				u = I[is * (n - k)];
				v = I[is * (n - 1 - k)];
				a = u + v;
				b2 = u - v;
			}
			{
				E u, v;
				u = I[is * (k)];
				v = I[is * (k - 1)];
				b = u + v;
				a2 = u - v;
			}
			{
				E wa, wb;
				wa = W[2 * i];
				wb = W[2 * i + 1];
				{
					E apb, amb;
					apb = a + b;
					amb = a - b;
					buf[i] = wa * amb + wb * apb;
					buf[n2 - i] = wa * apb - wb * amb;
				}
				{
					E apb, amb;
					apb = a2 + b2;
					amb = a2 - b2;
					buf[n2 + i] = wa * amb + wb * apb;
					buf[n - i] = wa * apb - wb * amb;
				}
			}
		}
		if (i + i == n2) {
			E u, v;
			u = I[is * n2];
			v = I[is * (n2 - 1)];
			buf[i] = (u + v) * (W[2 * i] * K(2.0));
			buf[n - i] = (u - v) * (W[2 * i] * K(2.0));
		}


		/* child plan: two r2hc's of size n/2 */
		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		W2 = ego->td2->W;
		{ /* i == 0 case */
			E wa, wb;
			E a, b;
			wa = W2[0]; /* cos */
			wb = W2[1]; /* sin */
			a = buf[0];
			b = buf[n2];
			O[0] = wa * a + wb * b;
			O[os * (n - 1)] = wa * b - wb * a;
		}
		W2 += 2;
		for (i = 1; i + i < n2; ++i, W2 += 2) {
			INT k;
			E u, v, u2, v2;
			u = buf[i];
			v = buf[n2 - i];
			u2 = buf[n2 + i];
			v2 = buf[n - i];
			k = (i + i) - 1;
			{
				E wa, wb;
				E a, b;
				wa = W2[0]; /* cos */
				wb = W2[1]; /* sin */
				a = v - u;
				b = u2 - v2;
				O[os * k] = wa * a + wb * b;
				O[os * (n - 1 - k)] = wa * b - wb * a;
			}
			++k;
			W2 += 2;
			{
				E wa, wb;
				E a, b;
				wa = W2[0]; /* cos */
				wb = W2[1]; /* sin */
				a = u + v;
				b = u2 + v2;
				O[os * k] = wa * a + wb * b;
				O[os * (n - 1 - k)] = wa * b - wb * a;
			}
		}
		if (i + i == n2) {
			INT k = (i + i) - 1;
			E wa, wb;
			E a, b;
			wa = W2[0]; /* cos */
			wb = W2[1]; /* sin */
			a = buf[i];
			b = buf[n2 + i];
			O[os * k] = wb * b - wa * a;
			O[os * (n - 1 - k)] = wa * b + wb * a;
		}
	}

	fftw_ifree(buf);
}

static void reodft11e_radix2_awake(plan *ego_, enum wakefulness wakefulness) {
	reodft11e_radix2_P *ego = (reodft11e_radix2_P *)ego_;
	static const tw_instr reodft010e_tw[] = {
		{ TW_COS,  0, 1 },
		{ TW_SIN,  0, 1 },
		{ TW_NEXT, 1, 0 }
	};
	static const tw_instr reodft11e_tw[] = {
		{ TW_COS,  1, 1 },
		{ TW_SIN,  1, 1 },
		{ TW_NEXT, 2, 0 }
	};

	fftw_plan_awake(ego->cld, wakefulness);

	fftw_twiddle_awake(wakefulness, &ego->td, reodft010e_tw,
		2 * ego->n, 1, ego->n / 4 + 1);
	fftw_twiddle_awake(wakefulness, &ego->td2, reodft11e_tw,
		8 * ego->n, 1, ego->n);
}

static void reodft11e_radix2_destroy(plan *ego_) {
	reodft11e_radix2_P *ego = (reodft11e_radix2_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void reodft11e_radix2_print(const plan *ego_, printer *p) {
	const reodft11e_radix2_P *ego = (const reodft11e_radix2_P *)ego_;
	p->print(p, "(%se-radix2-r2hc-%D%v%(%p%))",
		fftw_rdft_kind_str(ego->kind), ego->n, ego->vl, ego->cld);
}

static int reodft11e_radix2_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->sz->dims[0].n % 2 == 0
		&& (p->kind[0] == REDFT11 || p->kind[0] == RODFT11)
		);
}

static int reodft11e_radix2_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && reodft11e_radix2_applicable0(ego, p));
}

static plan *reodft11e_radix2_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	reodft11e_radix2_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, reodft11e_radix2_awake, reodft11e_radix2_print, reodft11e_radix2_destroy
	};

	if (!reodft11e_radix2_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n / 2, 1, 1),
		fftw_mktensor_1d(2, n / 2, n / 2),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	pln = MKPLAN_RDFT(reodft11e_radix2_P, &padt,
		p->kind[0] == REDFT11 ? reodft11e_radix2_apply_re11 : reodft11e_radix2_apply_ro11);
	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->td = pln->td2 = 0;
	pln->kind = p->kind[0];

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.add = 2 + (n / 2 - 1) / 2 * 20;
	ops.mul = 6 + (n / 2 - 1) / 2 * 16;
	ops.other = 4 * n + 2 + (n / 2 - 1) / 2 * 6;
	if ((n / 2) % 2 == 0) {
		ops.add += 4;
		ops.mul += 8;
		ops.other += 4;
	}

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *reodft11e_radix2_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, reodft11e_radix2_mkplan, 0 };
	reodft11e_radix2_S *slv = MKSOLVER(reodft11e_radix2_S, &sadt);
	return &(slv->super);
}

void fftw_reodft11e_radix2_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, reodft11e_radix2_mksolver());
}

/* Do a RODFT00 problem via an R2HC problem, with some pre/post-processing.

This code uses the trick from FFTPACK, also documented in a similar
form by Numerical Recipes.  Unfortunately, this algorithm seems to
have intrinsic numerical problems (similar to those in
reodft11e-r2hc.c), possibly due to the fact that it multiplies its
input by a sine, causing a loss of precision near the zero.  For
transforms of 16k points, it has already lost three or four decimal
places of accuracy, which we deem unacceptable.

So, we have abandoned this algorithm in favor of the one in
rodft00-r2hc-pad.c, which unfortunately sacrifices 30-50% in speed.
The only other alternative in the literature that does not have
similar numerical difficulties seems to be the direct adaptation of
the Cooley-Tukey decomposition for antisymmetric data, but this
would require a whole new set of codelets and it's not clear that
it's worth it at this point.  However, we did implement the latter
algorithm for the specific case of odd n (logically adapting the
split-radix algorithm); see reodft00e-splitradix.c. */



typedef struct {
	solver super;
} rodft00e_r2hc_S;

typedef struct {
	plan_rdft super;
	plan *cld;
	twid *td;
	INT is, os;
	INT n;
	INT vl;
	INT ivs, ovs;
} rodft00e_r2hc_P;

static void rodft00e_r2hc_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rodft00e_r2hc_P *ego = (const rodft00e_r2hc_P *)ego_;
	INT is = ego->is, os = ego->os;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *W = ego->td->W;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = 0;
		for (i = 1; i < n - i; ++i) {
			E a, b, apb, amb;
			a = I[is * (i - 1)];
			b = I[is * ((n - i) - 1)];
			apb = K(2.0) * W[i] * (a + b);
			amb = (a - b);
			buf[i] = apb + amb;
			buf[n - i] = apb - amb;
		}
		if (i == n - i) {
			buf[i] = K(4.0) * I[is * (i - 1)];
		}

		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* FIXME: use recursive/cascade summation for better stability? */
		O[0] = buf[0] * (FFTW_REAL_TYPE)0.5;
		for (i = 1; i + i < n - 1; ++i) {
			INT k = i + i;
			O[os * (k - 1)] = -buf[n - i];
			O[os * k] = O[os * (k - 2)] + buf[i];
		}
		if (i + i == n - 1) {
			O[os * (n - 2)] = -buf[n - i];
		}
	}

	fftw_ifree(buf);
}

static void rodft00e_r2hc_awake(plan *ego_, enum wakefulness wakefulness) {
	rodft00e_r2hc_P *ego = (rodft00e_r2hc_P *)ego_;
	static const tw_instr rodft00e_tw[] = {
		{ TW_SIN,  0, 1 },
		{ TW_NEXT, 1, 0 }
	};

	fftw_plan_awake(ego->cld, wakefulness);

	fftw_twiddle_awake(wakefulness,
		&ego->td, rodft00e_tw, 2 * ego->n, 1, (ego->n + 1) / 2);
}

static void rodft00e_r2hc_destroy(plan *ego_) {
	rodft00e_r2hc_P *ego = (rodft00e_r2hc_P *)ego_;
	fftw_plan_destroy_internal(ego->cld);
}

static void rodft00e_r2hc_print(const plan *ego_, printer *p) {
	const rodft00e_r2hc_P *ego = (const rodft00e_r2hc_P *)ego_;
	p->print(p, "(rodft00e-r2hc-%D%v%(%p%))", ego->n - 1, ego->vl, ego->cld);
}

static int rodft00e_r2hc_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);

	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->kind[0] == RODFT00
		);
}

static int rodft00e_r2hc_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && rodft00e_r2hc_applicable0(ego, p));
}

static plan *rodft00e_r2hc_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	rodft00e_r2hc_P *pln;
	const problem_rdft *p;
	plan *cld;
	FFTW_REAL_TYPE *buf;
	INT n;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, rodft00e_r2hc_awake, rodft00e_r2hc_print, rodft00e_r2hc_destroy
	};

	if (!rodft00e_r2hc_applicable(ego_, p_, plnr))
		return (plan *)0;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n + 1;
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * n, BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	fftw_ifree(buf);
	if (!cld)
		return (plan *)0;

	pln = MKPLAN_RDFT(rodft00e_r2hc_P, &padt, rodft00e_r2hc_apply);

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->os = p->sz->dims[0].os;
	pln->cld = cld;
	pln->td = 0;

	fftw_tensor_tornk1(p->vecsz, &pln->vl, &pln->ivs, &pln->ovs);

	fftw_ops_zero(&ops);
	ops.other = 4 + (n - 1) / 2 * 5 + (n - 2) / 2 * 5;
	ops.add = (n - 1) / 2 * 4 + (n - 2) / 2 * 1;
	ops.mul = 1 + (n - 1) / 2 * 2;
	if (n % 2 == 0)
		ops.mul += 1;

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);

	return &(pln->super.super);
}

/* constructor */
static solver *rodft00e_r2hc_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, rodft00e_r2hc_mkplan, 0 };
	rodft00e_r2hc_S *slv = MKSOLVER(rodft00e_r2hc_S, &sadt);
	return &(slv->super);
}

void fftw_rodft00e_r2hc_register(planner *p) {
	REGISTER_SOLVER(p, rodft00e_r2hc_mksolver());
}

/* Do a RODFT00 problem via an R2HC problem, padded antisymmetrically to
twice the size.  This is asymptotically a factor of ~2 worse than
rodft00e-r2hc.c (the algorithm used in e.g. FFTPACK and Numerical
Recipes), but we abandoned the latter after we discovered that it
has intrinsic accuracy problems. */



typedef struct {
	solver super;
} rodft00e_r2hc_pad_S;

typedef struct {
	plan_rdft super;
	plan *cld, *cldcpy;
	INT is;
	INT n;
	INT vl;
	INT ivs, ovs;
} rodft00e_r2hc_pad_P;

static void rodft00e_r2hc_pad_apply(const plan *ego_, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O) {
	const rodft00e_r2hc_pad_P *ego = (const rodft00e_r2hc_pad_P *)ego_;
	INT is = ego->is;
	INT i, n = ego->n;
	INT iv, vl = ego->vl;
	INT ivs = ego->ivs, ovs = ego->ovs;
	FFTW_REAL_TYPE *buf;

	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (2 * n), BUFFERS);

	for (iv = 0; iv < vl; ++iv, I += ivs, O += ovs) {
		buf[0] = K(0.0);
		for (i = 1; i < n; ++i) {
			FFTW_REAL_TYPE a = I[(i - 1) * is];
			buf[i] = -a;
			buf[2 * n - i] = a;
		}
		buf[i] = K(0.0); /* i == n, Nyquist */

						 /* r2hc transform of size 2*n */
		{
			plan_rdft *cld = (plan_rdft *)ego->cld;
			cld->apply((plan *)cld, buf, buf);
		}

		/* copy n-1 real numbers (imag. parts of hc array) from buf to O */
		{
			plan_rdft *cldcpy = (plan_rdft *)ego->cldcpy;
			cldcpy->apply((plan *)cldcpy, buf + 2 * n - 1, O);
		}
	}

	fftw_ifree(buf);
}

static void rodft00e_r2hc_pad_awake(plan *ego_, enum wakefulness wakefulness) {
	rodft00e_r2hc_pad_P *ego = (rodft00e_r2hc_pad_P *)ego_;
	fftw_plan_awake(ego->cld, wakefulness);
	fftw_plan_awake(ego->cldcpy, wakefulness);
}

static void rodft00e_r2hc_pad_destroy(plan *ego_) {
	rodft00e_r2hc_pad_P *ego = (rodft00e_r2hc_pad_P *)ego_;
	fftw_plan_destroy_internal(ego->cldcpy);
	fftw_plan_destroy_internal(ego->cld);
}

static void rodft00e_r2hc_pad_print(const plan *ego_, printer *p) {
	const rodft00e_r2hc_pad_P *ego = (const rodft00e_r2hc_pad_P *)ego_;
	p->print(p, "(rodft00e-r2hc-pad-%D%v%(%p%)%(%p%))",
		ego->n - 1, ego->vl, ego->cld, ego->cldcpy);
}

static int rodft00e_r2hc_pad_applicable0(const solver *ego_, const problem *p_) {
	const problem_rdft *p = (const problem_rdft *)p_;
	UNUSED(ego_);
	return (1
		&& p->sz->rnk == 1
		&& p->vecsz->rnk <= 1
		&& p->kind[0] == RODFT00
		);
}

static int rodft00e_r2hc_pad_applicable(const solver *ego, const problem *p, const planner *plnr) {
	return (!NO_SLOWP(plnr) && rodft00e_r2hc_pad_applicable0(ego, p));
}

static plan *rodft00e_r2hc_pad_mkplan(const solver *ego_, const problem *p_, planner *plnr) {
	rodft00e_r2hc_pad_P *pln;
	const problem_rdft *p;
	plan *cld = (plan *)0, *cldcpy;
	FFTW_REAL_TYPE *buf = (FFTW_REAL_TYPE *)0;
	INT n;
	INT vl, ivs, ovs;
	opcnt ops;

	static const plan_adt padt = {
		fftw_rdft_solve, rodft00e_r2hc_pad_awake, rodft00e_r2hc_pad_print, rodft00e_r2hc_pad_destroy
	};

	if (!rodft00e_r2hc_pad_applicable(ego_, p_, plnr))
		goto nada;

	p = (const problem_rdft *)p_;

	n = p->sz->dims[0].n + 1;
	A(n > 0);
	buf = (FFTW_REAL_TYPE *)MALLOC(sizeof(FFTW_REAL_TYPE) * (2 * n), BUFFERS);

	cld = fftw_mkplan_d(plnr, fftw_mkproblem_rdft_1_d(fftw_mktensor_1d(2 * n, 1, 1),
		fftw_mktensor_0d(),
		buf, buf, R2HC));
	if (!cld)
		goto nada;

	fftw_tensor_tornk1(p->vecsz, &vl, &ivs, &ovs);
	cldcpy =
		fftw_mkplan_d(plnr,
			fftw_mkproblem_rdft_1_d(fftw_mktensor_0d(),
				fftw_mktensor_1d(n - 1, -1,
					p->sz->dims[0].os),
				buf + 2 * n - 1, TAINT(p->O, ovs), R2HC));
	if (!cldcpy)
		goto nada;

	fftw_ifree(buf);

	pln = MKPLAN_RDFT(rodft00e_r2hc_pad_P, &padt, rodft00e_r2hc_pad_apply);

	pln->n = n;
	pln->is = p->sz->dims[0].is;
	pln->cld = cld;
	pln->cldcpy = cldcpy;
	pln->vl = vl;
	pln->ivs = ivs;
	pln->ovs = ovs;

	fftw_ops_zero(&ops);
	ops.other = n - 1 + 2 * n; /* loads + stores (input -> buf) */

	fftw_ops_zero(&pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cld->ops, &pln->super.super.ops);
	fftw_ops_madd2(pln->vl, &cldcpy->ops, &pln->super.super.ops);

	return &(pln->super.super);

nada:
	fftw_ifree0(buf);
	if (cld)
		fftw_plan_destroy_internal(cld);
	return (plan *)0;
}

/* constructor */
static solver *rodft00e_r2hc_pad_mksolver(void) {
	static const solver_adt sadt = { PROBLEM_RDFT, rodft00e_r2hc_pad_mkplan, 0 };
	rodft00e_r2hc_pad_S *slv = MKSOLVER(rodft00e_r2hc_pad_S, &sadt);
	return &(slv->super);
}

void fftw_rodft00e_r2hc_pad_register(planner *p) {
	REGISTER_SOLVER(p, rodft00e_r2hc_pad_mksolver());
}

#ifdef FFTW_DEBUG
#include <stdio.h>

typedef struct {
	printer super;
	FILE *f;
} P_file;

static void putchr_file(printer *p_, char c)
{
	P_file *p = (P_file *)p_;
	fputc(c, p->f);
}

static printer *mkprinter_file(FILE *f)
{
	P_file *p = (P_file *)fftw_mkprinter(sizeof(P_file), putchr_file, 0);
	p->f = f;
	return &p->super;
}

void fftw_debug(const char *format, ...)
{
	va_list ap;
	printer *p = mkprinter_file(stderr);
	va_start(ap, format);
	p->vprint(p, format, ap);
	va_end(ap);
	fftw_printer_destroy(p);
}
#endif

/*
independent implementation of Ron Rivest's MD5 message-digest
algorithm, based on rfc 1321.

Optimized for small code size, not speed.  Works as long as
sizeof(md5uint) >= 4.
*/


/* sintab[i] = 4294967296.0 * abs(sin((double)(i + 1))) */
static const md5uint sintab[64] = {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

/* see rfc 1321 section 3.4 */
static const struct roundtab {
	char k;
	char s;
} roundtab[64] = {
	{ 0,  7 },
	{ 1,  12 },
	{ 2,  17 },
	{ 3,  22 },
	{ 4,  7 },
	{ 5,  12 },
	{ 6,  17 },
	{ 7,  22 },
	{ 8,  7 },
	{ 9,  12 },
	{ 10, 17 },
	{ 11, 22 },
	{ 12, 7 },
	{ 13, 12 },
	{ 14, 17 },
	{ 15, 22 },
	{ 1,  5 },
	{ 6,  9 },
	{ 11, 14 },
	{ 0,  20 },
	{ 5,  5 },
	{ 10, 9 },
	{ 15, 14 },
	{ 4,  20 },
	{ 9,  5 },
	{ 14, 9 },
	{ 3,  14 },
	{ 8,  20 },
	{ 13, 5 },
	{ 2,  9 },
	{ 7,  14 },
	{ 12, 20 },
	{ 5,  4 },
	{ 8,  11 },
	{ 11, 16 },
	{ 14, 23 },
	{ 1,  4 },
	{ 4,  11 },
	{ 7,  16 },
	{ 10, 23 },
	{ 13, 4 },
	{ 0,  11 },
	{ 3,  16 },
	{ 6,  23 },
	{ 9,  4 },
	{ 12, 11 },
	{ 15, 16 },
	{ 2,  23 },
	{ 0,  6 },
	{ 7,  10 },
	{ 14, 15 },
	{ 5,  21 },
	{ 12, 6 },
	{ 3,  10 },
	{ 10, 15 },
	{ 1,  21 },
	{ 8,  6 },
	{ 15, 10 },
	{ 6,  15 },
	{ 13, 21 },
	{ 4,  6 },
	{ 11, 10 },
	{ 2,  15 },
	{ 9,  21 }
};

#define rol(a, s) (((a) << (int)(s)) | ((a) >> (32 - (int)(s))))

static void doblock(md5sig state, const unsigned char *data) {
	md5uint a, b, c, d, t, x[16];
	const md5uint msk = (md5uint)0xffffffffUL;
	int i;

	/* encode input bytes into md5uint */
	for (i = 0; i < 16; ++i) {
		const unsigned char *p = data + 4 * i;
		x[i] = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
	}

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	for (i = 0; i < 64; ++i) {
		const struct roundtab *p = roundtab + i;
		switch (i >> 4) {
		case 0:
			a += (b & c) | (~b & d);
			break;
		case 1:
			a += (b & d) | (c & ~d);
			break;
		case 2:
			a += b ^ c ^ d;
			break;
		case 3:
			a += c ^ (b | ~d);
			break;
		}
		a += sintab[i];
		a += x[(int)(p->k)];
		a &= msk;
		t = b + rol(a, p->s);
		a = d;
		d = c;
		c = b;
		b = t;
	}
	state[0] = (state[0] + a) & msk;
	state[1] = (state[1] + b) & msk;
	state[2] = (state[2] + c) & msk;
	state[3] = (state[3] + d) & msk;
}


void fftw_md5begin(md5 *p) {
	p->s[0] = 0x67452301;
	p->s[1] = 0xefcdab89;
	p->s[2] = 0x98badcfe;
	p->s[3] = 0x10325476;
	p->l = 0;
}

void fftw_md5putc(md5 *p, unsigned char c) {
	p->c[p->l % 64] = c;
	if (((++p->l) % 64) == 0) doblock(p->s, p->c);
}

void fftw_md5end(md5 *p) {
	unsigned l, i;

	l = 8 * p->l; /* length before padding, in bits */

				  /* rfc 1321 section 3.1: padding */
	fftw_md5putc(p, 0x80);
	while ((p->l % 64) != 56) fftw_md5putc(p, 0x00);

	/* rfc 1321 section 3.2: length (little endian) */
	for (i = 0; i < 8; ++i) {
		fftw_md5putc(p, (unsigned char)(l & 0xFF));
		l = l >> 8;
	}

	/* Now p->l % 64 == 0 and signature is in p->s */
}

void fftw_md5putb(md5 *p, const void *d_, size_t len) {
	size_t i;
	const unsigned char *d = (const unsigned char *)d_;
	for (i = 0; i < len; ++i)
		fftw_md5putc(p, d[i]);
}

void fftw_md5puts(md5 *p, const char *s) {
	/* also hash final '\0' */
	do {
		fftw_md5putc(p, (unsigned)(*s & 0xFF));
	} while (*s++);
}

void fftw_md5int(md5 *p, int i) {
	fftw_md5putb(p, &i, sizeof(i));
}

void fftw_md5INT(md5 *p, INT i) {
	fftw_md5putb(p, &i, sizeof(i));
}

void fftw_md5unsigned(md5 *p, unsigned i) {
	fftw_md5putb(p, &i, sizeof(i));
}

#if defined(HAVE_MALLOC_H)

#  include <malloc.h>

#endif

/* ``kernel'' malloc(), with proper memory alignment */

#if defined(HAVE_DECL_MEMALIGN) && !HAVE_DECL_MEMALIGN

extern void *memalign(size_t, size_t);

#endif

#if defined(HAVE_DECL_POSIX_MEMALIGN) && !HAVE_DECL_POSIX_MEMALIGN

extern int posix_memalign(void **, size_t, size_t);

#endif

#if defined(macintosh) /* MacOS 9 */
#  include <Multiprocessing.h>
#endif

#define real_free free /* memalign and malloc use ordinary free */

#define IS_POWER_OF_TWO(n) (((n) > 0) && (((n) & ((n) - 1)) == 0))
#if defined(WITH_OUR_MALLOC) && (MIN_ALIGNMENT >= 8) && IS_POWER_OF_TWO(MIN_ALIGNMENT)
/* Our own MIN_ALIGNMENT-aligned malloc/free.  Assumes sizeof(void*) is a
power of two <= 8 and that malloc is at least sizeof(void*)-aligned.

The main reason for this routine is that, as of this writing,
Windows does not include any aligned allocation routines in its
system libraries, and instead provides an implementation with a
Visual C++ "Processor Pack" that you have to statically link into
your program.  We do not want to require users to have VC++
(e.g. gcc/MinGW should be fine).  Our code should be at least as good
as the MS _aligned_malloc, in any case, according to second-hand
reports of the algorithm it employs (also based on plain malloc). */
static void *our_malloc(size_t n)
{
	void *p0, *p;
	if (!(p0 = malloc(n + MIN_ALIGNMENT))) return (void *)0;
	p = (void *)(((uintptr_t)p0 + MIN_ALIGNMENT) & (~((uintptr_t)(MIN_ALIGNMENT - 1))));
	*((void **)p - 1) = p0;
	return p;
}
static void our_free(void *p)
{
	if (p) free(*((void **)p - 1));
}
#endif

void *fftw_kernel_malloc(size_t n) {
	void *p;

#if defined(MIN_ALIGNMENT)

#  if defined(WITH_OUR_MALLOC)
	p = our_malloc(n);
#    undef real_free
#    define real_free our_free

#  elif defined(__FreeBSD__) && (MIN_ALIGNMENT <= 16)
	/* FreeBSD does not have memalign, but its malloc is 16-byte aligned. */
	p = malloc(n);

#  elif (defined(__MACOSX__) || defined(__APPLE__)) && (MIN_ALIGNMENT <= 16)
	/* MacOS X malloc is already 16-byte aligned */
	p = malloc(n);

#  elif defined(HAVE_MEMALIGN)
	p = memalign(MIN_ALIGNMENT, n);

#  elif defined(HAVE_POSIX_MEMALIGN)
	/* note: posix_memalign is broken in glibc 2.2.5: it constrains
	the size, not the alignment, to be (power of two) * sizeof(void*).
	The bug seems to have been fixed as of glibc 2.3.1. */
	if (posix_memalign(&p, MIN_ALIGNMENT, n))
		p = (void*)0;

#  elif defined(__ICC) || defined(__INTEL_COMPILER) || defined(HAVE__MM_MALLOC)
	/* Intel's C compiler defines _mm_malloc and _mm_free intrinsics */
	p = (void *)_mm_malloc(n, MIN_ALIGNMENT);
#    undef real_free
#    define real_free _mm_free

#  elif defined(_MSC_VER)
	/* MS Visual C++ 6.0 with a "Processor Pack" supports SIMD
	and _aligned_malloc/free (uses malloc.h) */
	p = (void *)_aligned_malloc(n, MIN_ALIGNMENT);
#    undef real_free
#    define real_free _aligned_free

#  elif defined(macintosh) /* MacOS 9 */
	p = (void *)MPAllocateAligned(n,
#    if MIN_ALIGNMENT == 8
		kMPAllocate8ByteAligned,
#    elif MIN_ALIGNMENT == 16
		kMPAllocate16ByteAligned,
#    elif MIN_ALIGNMENT == 32
		kMPAllocate32ByteAligned,
#    else
#      error "Unknown alignment for MPAllocateAligned"
#    endif
		0);
#    undef real_free
#    define real_free MPFree

#  else
	/* Add your machine here and send a patch to fftw@fftw.org
	or (e.g. for Windows) configure --with-our-malloc */
#    error "Don't know how to malloc() aligned memory ... try configuring --with-our-malloc"
#  endif

#else /* !defined(MIN_ALIGNMENT) */
	p = malloc(n);
#endif

	return p;
}

void fftw_kernel_free(void *p) {
	real_free(p);
}

#if HAVE_SIMD
#  define ALGN 16
#else
/* disable the alignment machinery, because it will break,
e.g., if sizeof(R) == 12 (as in long-double/x86) */
#  define ALGN 0
#endif

/* NONPORTABLE */
int fftw_ialignment_of(FFTW_REAL_TYPE *p) {
#if ALGN == 0
	UNUSED(p);
	return 0;
#else
	return (int)(((uintptr_t)p) % ALGN);
#endif
}

INT fftw_iabs(INT a) {
	return a < 0 ? (0 - a) : a;
}

unsigned fftw_hash(const char *s) {
	unsigned h = 0xDEADBEEFu;
	do {
		h = h * 17 + (unsigned)(*s & 0xFF);
	} while (*s++);
	return h;
}

/* decompose complex pointer into real and imaginary parts.
Flip real and imaginary if there the sign does not match
FFTW's idea of what the sign should be */

void fftw_extract_reim(int sign, FFTW_REAL_TYPE *c, FFTW_REAL_TYPE **r, FFTW_REAL_TYPE **i) {
	if (sign == FFT_SIGN) {
		*r = c + 0;
		*i = c + 1;
	}
	else {
		*r = c + 1;
		*i = c + 0;
	}
}

/* common routines for Cooley-Tukey algorithms */


#define POW2P(n) (((n) > 0) && (((n) & ((n) - 1)) == 0))

/* TRUE if radix-r is ugly for size n */
int fftw_ct_uglyp(INT min_n, INT v, INT n, INT r) {
	return (n <= min_n) || (POW2P(n) && (v * (n / r)) <= 4);
}

INT fftw_imax(INT a, INT b) {
	return (a > b) ? a : b;
}

INT fftw_imin(INT a, INT b) {
	return (a < b) ? a : b;
}

void fftw_ops_zero(opcnt *dst) {
	dst->add = dst->mul = dst->fma = dst->other = 0;
}

void fftw_ops_cpy(const opcnt *src, opcnt *dst) {
	*dst = *src;
}

void fftw_ops_other(INT o, opcnt *dst) {
	fftw_ops_zero(dst);
	dst->other = o;
}

void fftw_ops_madd(INT m, const opcnt *a, const opcnt *b, opcnt *dst) {
	dst->add = m * a->add + b->add;
	dst->mul = m * a->mul + b->mul;
	dst->fma = m * a->fma + b->fma;
	dst->other = m * a->other + b->other;
}

void fftw_ops_add(const opcnt *a, const opcnt *b, opcnt *dst) {
	fftw_ops_madd(1, a, b, dst);
}

void fftw_ops_add2(const opcnt *a, opcnt *dst) {
	fftw_ops_add(a, dst, dst);
}

void fftw_ops_madd2(INT m, const opcnt *a, opcnt *dst) {
	fftw_ops_madd(m, a, dst, dst);
}


/* Given a solver which_dim, a vector sz, and whether or not the
transform is out-of-place, return the actual dimension index that
it corresponds to.  The basic idea here is that we return the
which_dim'th valid dimension, starting from the end if
which_dim < 0. */
static int really_pickdim(int which_dim, const tensor *sz, int oop, int *dp) {
	int i;
	int count_ok = 0;
	if (which_dim > 0) {
		for (i = 0; i < sz->rnk; ++i) {
			if (oop || sz->dims[i].is == sz->dims[i].os)
				if (++count_ok == which_dim) {
					*dp = i;
					return 1;
				}
		}
	}
	else if (which_dim < 0) {
		for (i = sz->rnk - 1; i >= 0; --i) {
			if (oop || sz->dims[i].is == sz->dims[i].os)
				if (++count_ok == -which_dim) {
					*dp = i;
					return 1;
				}
		}
	}
	else { /* zero: pick the middle, if valid */
		i = (sz->rnk - 1) / 2;
		if (i >= 0 && (oop || sz->dims[i].is == sz->dims[i].os)) {
			*dp = i;
			return 1;
		}
	}
	return 0;
}

/* Like really_pickdim, but only returns 1 if no previous "buddy"
which_dim in the buddies list would give the same dim. */
int fftw_pickdim(int which_dim, const int *buddies, size_t nbuddies,
	const tensor *sz, int oop, int *dp) {
	size_t i;
	int d1;

	if (!really_pickdim(which_dim, sz, oop, dp))
		return 0;

	/* check whether some buddy solver would produce the same dim.
	If so, consider this solver unapplicable and let the buddy
	take care of it.  The smallest-indexed buddy is applicable. */
	for (i = 0; i < nbuddies; ++i) {
		if (buddies[i] == which_dim)
			break;  /* found self */
		if (really_pickdim(buddies[i], sz, oop, &d1) && *dp == d1)
			return 0; /* found equivalent buddy */
	}
	return 1;
}

/* "Plan: To bother about the best method of accomplishing an
accidental result."  (Ambrose Bierce, The Enlarged Devil's
Dictionary). */

plan *fftw_mkplan(size_t size, const plan_adt *adt) {
	plan *p = (plan *)MALLOC(size, PLANS);

	A(adt->destroy);
	p->adt = adt;
	fftw_ops_zero(&p->ops);
	p->pcost = 0.0;
	p->wakefulness = SLEEPY;
	p->could_prune_now_p = 0;

	return p;
}

/*
* destroy a plan
*/
void fftw_plan_destroy_internal(plan *ego) {
	if (ego) {
		A(ego->wakefulness == SLEEPY);
		ego->adt->destroy(ego);
		fftw_ifree(ego);
	}
}

/* dummy destroy routine for plans with no local state */
void fftw_plan_null_destroy(plan *ego) {
	UNUSED(ego);
	/* nothing */
}

void fftw_plan_awake(plan *ego, enum wakefulness wakefulness) {
	if (ego) {
		A(((wakefulness == SLEEPY) ^ (ego->wakefulness == SLEEPY)));

		ego->adt->awake(ego, wakefulness);
		ego->wakefulness = wakefulness;
	}
}


/* GNU Coding Standards, Sec. 5.2: "Please write the comments in a GNU
program in English, because English is the one language that nearly
all programmers in all countries can read."

ingemisco tanquam reus
culpa rubet vultus meus
supplicanti parce [rms]
*/

#define VALIDP(solution) ((solution)->flags.hash_info & H_VALID)
#define LIVEP(solution) ((solution)->flags.hash_info & H_LIVE)
#define SLVNDX(solution) ((solution)->flags.slvndx)
#define BLISS(flags) (((flags).hash_info) & BLESSING)
#define INFEASIBLE_SLVNDX ((1U<<BITS_FOR_SLVNDX)-1)


#define MAXNAM 64  /* maximum length of registrar's name.
Used for reading wisdom.  There is no point
in doing this right */


#ifdef FFTW_DEBUG
static void check(hashtab *ht);
#endif

/* x <= y */
#define LEQ(x, y) (((x) & (y)) == (x))

/* A subsumes B */
static int subsumes(const flags_t *a, unsigned slvndx_a, const flags_t *b) {
	if (slvndx_a != INFEASIBLE_SLVNDX) {
		A(a->timelimit_impatience == 0);
		return (LEQ(a->u, b->u) && LEQ(b->l, a->l));
	}
	else {
		return (LEQ(a->l, b->l)
			&& a->timelimit_impatience <= b->timelimit_impatience);
	}
}

static unsigned addmod(unsigned a, unsigned b, unsigned p) {
	/* gcc-2.95/sparc produces incorrect code for the fast version below. */
#if defined(__sparc__) && defined(__GNUC__)
	/* slow version  */
	return (a + b) % p;
#else
	/* faster version */
	unsigned c = a + b;
	return c >= p ? c - p : c;
#endif
}

/*
slvdesc management:
*/
static void sgrow(planner *ego) {
	unsigned osiz = ego->slvdescsiz, nsiz = 1 + osiz + osiz / 4;
	slvdesc *ntab = (slvdesc *)MALLOC(nsiz * sizeof(slvdesc), SLVDESCS);
	slvdesc *otab = ego->slvdescs;
	unsigned i;

	ego->slvdescs = ntab;
	ego->slvdescsiz = nsiz;
	for (i = 0; i < osiz; ++i)
		ntab[i] = otab[i];
	fftw_ifree0(otab);
}

static void ifftw_register_solver(planner *ego, solver *s) {
	slvdesc *n;
	int kind;

	if (s) { /* add s to solver list */
		fftw_solver_use(s);

		A(ego->nslvdesc < INFEASIBLE_SLVNDX);
		if (ego->nslvdesc >= ego->slvdescsiz)
			sgrow(ego);

		n = ego->slvdescs + ego->nslvdesc;

		n->slv = s;
		n->reg_nam = ego->cur_reg_nam;
		n->reg_id = ego->cur_reg_id++;

		A(strlen(n->reg_nam) < MAXNAM);
		n->nam_hash = fftw_hash(n->reg_nam);

		kind = s->adt->problem_kind;
		n->next_for_same_problem_kind = ego->slvdescs_for_problem_kind[kind];
		ego->slvdescs_for_problem_kind[kind] = (int)/*from unsigned*/ego->nslvdesc;

		ego->nslvdesc++;
	}
}

static unsigned slookup(planner *ego, char *nam, int id) {
	unsigned h = fftw_hash(nam); /* used to avoid strcmp in the common case */
	FORALL_SOLVERS(ego, s, sp, {
		UNUSED(s);
	if (sp->reg_id == id && sp->nam_hash == h
		&& !strcmp(sp->reg_nam, nam))
		return (unsigned)/*from ptrdiff_t*/(sp - ego->slvdescs);
	});
	return INFEASIBLE_SLVNDX;
}

/* Compute a MD5 hash of the configuration of the planner.
We store it into the wisdom file to make absolutely sure that
we are reading wisdom that is applicable */
static void signature_of_configuration(md5 *m, planner *ego) {
	fftw_md5begin(m);
	fftw_md5unsigned(m, sizeof(FFTW_REAL_TYPE)); /* so we don't mix different precisions */
	FORALL_SOLVERS(ego, s, sp, {
		UNUSED(s);
	fftw_md5int(m, sp->reg_id);
	fftw_md5puts(m, sp->reg_nam);
	});
	fftw_md5end(m);
}

/*
md5-related stuff:
*/

/* first hash function */
static unsigned h1(const hashtab *ht, const md5sig s) {
	unsigned h = s[0] % ht->hashsiz;
	A(h == (s[0] % ht->hashsiz));
	return h;
}

/* second hash function (for double hashing) */
static unsigned h2(const hashtab *ht, const md5sig s) {
	unsigned h = 1U + s[1] % (ht->hashsiz - 1);
	A(h == (1U + s[1] % (ht->hashsiz - 1)));
	return h;
}

static void md5hash(md5 *m, const problem *p, const planner *plnr) {
	fftw_md5begin(m);
	fftw_md5unsigned(m, sizeof(FFTW_REAL_TYPE)); /* so we don't mix different precisions */
	fftw_md5int(m, plnr->nthr);
	p->adt->hash(p, m);
	fftw_md5end(m);
}

static int md5eq(const md5sig a, const md5sig b) {
	return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

static void sigcpy(const md5sig a, md5sig b) {
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
	b[3] = a[3];
}

/*
memoization routines :
*/

/*
liber scriptus proferetur
in quo totum continetur
unde mundus iudicetur
*/
struct solution_s {
	md5sig s;
	flags_t flags;
};

static solution *htab_lookup(hashtab *ht, const md5sig s,
	const flags_t *flagsp) {
	unsigned g, h = h1(ht, s), d = h2(ht, s);
	solution *best = 0;

	++ht->lookup;

	/* search all entries that match; select the one with
	the lowest flags.u */
	/* This loop may potentially traverse the whole table, since at
	least one element is guaranteed to be !LIVEP, but all elements
	may be VALIDP.  Hence, we stop after at the first invalid
	element or after traversing the whole table. */
	g = h;
	do {
		solution *l = ht->solutions + g;
		++ht->lookup_iter;
		if (VALIDP(l)) {
			if (LIVEP(l)
				&& md5eq(s, l->s)
				&& subsumes(&l->flags, SLVNDX(l), flagsp)) {
				if (!best || LEQ(l->flags.u, best->flags.u))
					best = l;
			}
		}
		else
			break;

		g = addmod(g, d, ht->hashsiz);
	} while (g != h);

	if (best)
		++ht->succ_lookup;
	return best;
}

static solution *hlookup(planner *ego, const md5sig s,
	const flags_t *flagsp) {
	solution *sol = htab_lookup(&ego->htab_blessed, s, flagsp);
	if (!sol) sol = htab_lookup(&ego->htab_unblessed, s, flagsp);
	return sol;
}

static void fill_slot(hashtab *ht, const md5sig s, const flags_t *flagsp,
	unsigned slvndx, solution *slot) {
	++ht->insert;
	++ht->nelem;
	A(!LIVEP(slot));
	slot->flags.u = flagsp->u;
	slot->flags.l = flagsp->l;
	slot->flags.timelimit_impatience = flagsp->timelimit_impatience;
	slot->flags.hash_info |= H_VALID | H_LIVE;
	SLVNDX(slot) = slvndx;

	/* keep this check enabled in case we add so many solvers
	that the bitfield overflows */
	CK(SLVNDX(slot) == slvndx);
	sigcpy(s, slot->s);
}

static void kill_slot(hashtab *ht, solution *slot) {
	A(LIVEP(slot)); /* ==> */ A(VALIDP(slot));

	--ht->nelem;
	slot->flags.hash_info = H_VALID;
}

static void hinsert0(hashtab *ht, const md5sig s, const flags_t *flagsp,
	unsigned slvndx) {
	solution *l;
	unsigned g, h = h1(ht, s), d = h2(ht, s);

	++ht->insert_unknown;

	/* search for nonfull slot */
	for (g = h;; g = addmod(g, d, ht->hashsiz)) {
		++ht->insert_iter;
		l = ht->solutions + g;
		if (!LIVEP(l)) break;
		A((g + d) % ht->hashsiz != h);
	}

	fill_slot(ht, s, flagsp, slvndx, l);
}

static void rehash(hashtab *ht, unsigned nsiz) {
	unsigned osiz = ht->hashsiz, h;
	solution *osol = ht->solutions, *nsol;

	nsiz = (unsigned)fftw_next_prime((INT)nsiz);
	nsol = (solution *)MALLOC(nsiz * sizeof(solution), HASHT);
	++ht->nrehash;

	/* init new table */
	for (h = 0; h < nsiz; ++h)
		nsol[h].flags.hash_info = 0;

	/* install new table */
	ht->hashsiz = nsiz;
	ht->solutions = nsol;
	ht->nelem = 0;

	/* copy table */
	for (h = 0; h < osiz; ++h) {
		solution *l = osol + h;
		if (LIVEP(l))
			hinsert0(ht, l->s, &l->flags, SLVNDX(l));
	}

	fftw_ifree0(osol);
}

static unsigned minsz(unsigned nelem) {
	return 1U + nelem + nelem / 8U;
}

static unsigned nextsz(unsigned nelem) {
	return minsz(minsz(nelem));
}

static void hgrow(hashtab *ht) {
	unsigned nelem = ht->nelem;
	if (minsz(nelem) >= ht->hashsiz)
		rehash(ht, nextsz(nelem));
}

#if 0
/* shrink the hash table, never used */
static void hshrink(hashtab *ht)
{
	unsigned nelem = ht->nelem;
	/* always rehash after deletions */
	rehash(ht, nextsz(nelem));
}
#endif

static void htab_insert(hashtab *ht, const md5sig s, const flags_t *flagsp,
	unsigned slvndx) {
	unsigned g, h = h1(ht, s), d = h2(ht, s);
	solution *first = 0;

	/* Remove all entries that are subsumed by the new one.  */
	/* This loop may potentially traverse the whole table, since at
	least one element is guaranteed to be !LIVEP, but all elements
	may be VALIDP.  Hence, we stop after at the first invalid
	element or after traversing the whole table. */
	g = h;
	do {
		solution *l = ht->solutions + g;
		++ht->insert_iter;
		if (VALIDP(l)) {
			if (LIVEP(l) && md5eq(s, l->s)) {
				if (subsumes(flagsp, slvndx, &l->flags)) {
					if (!first) first = l;
					kill_slot(ht, l);
				}
				else {
					/* It is an error to insert an element that
					is subsumed by an existing entry. */
					A(!subsumes(&l->flags, SLVNDX(l), flagsp));
				}
			}
		}
		else
			break;

		g = addmod(g, d, ht->hashsiz);
	} while (g != h);

	if (first) {
		/* overwrite FIRST */
		fill_slot(ht, s, flagsp, slvndx, first);
	}
	else {
		/* create a new entry */
		hgrow(ht);
		hinsert0(ht, s, flagsp, slvndx);
	}
}

static void hinsert(planner *ego, const md5sig s, const flags_t *flagsp,
	unsigned slvndx) {
	htab_insert(BLISS(*flagsp) ? &ego->htab_blessed : &ego->htab_unblessed,
		s, flagsp, slvndx);
}


static void invoke_hook(planner *ego, plan *pln, const problem *p,
	int optimalp) {
	if (ego->hook)
		ego->hook(ego, pln, p, optimalp);
}

#ifdef FFTW_RANDOM_ESTIMATOR
/* a "random" estimate, used for debugging to generate "random"
plans, albeit from a deterministic seed. */

unsigned fftw_random_estimate_seed = 0;

static double random_estimate(const planner *ego, const plan *pln,
	const problem *p)
{
	md5 m;
	fftw_md5begin(&m);
	fftw_md5unsigned(&m, fftw_random_estimate_seed);
	fftw_md5int(&m, ego->nthr);
	p->adt->hash(p, &m);
	fftw_md5putb(&m, &pln->ops, sizeof(pln->ops));
	fftw_md5putb(&m, &pln->adt, sizeof(pln->adt));
	fftw_md5end(&m);
	return ego->cost_hook ? ego->cost_hook(p, m.s[0], COST_MAX) : m.s[0];
}

#endif

double fftw_iestimate_cost(const planner *ego, const plan *pln, const problem *p) {
	double cost =
		+pln->ops.add
		+ pln->ops.mul

#if HAVE_FMA
		+ pln->ops.fma
#else
		+ 2 * pln->ops.fma
#endif

		+ pln->ops.other;
	if (ego->cost_hook)
		cost = ego->cost_hook(p, cost, COST_MAX);
	return cost;
}

static void evaluate_plan(planner *ego, plan *pln, const problem *p) {
	if (ESTIMATEP(ego) || !BELIEVE_PCOSTP(ego) || pln->pcost == 0.0) {
		ego->nplan++;

		if (ESTIMATEP(ego)) {
		estimate:
			/* heuristic */
#ifdef FFTW_RANDOM_ESTIMATOR
			pln->pcost = random_estimate(ego, pln, p);
			ego->epcost += fftw_iestimate_cost(ego, pln, p);
#else
			pln->pcost = fftw_iestimate_cost(ego, pln, p);
			ego->epcost += pln->pcost;
#endif
		}
		else {
			double t = fftw_measure_execution_time(ego, pln, p);

			if (t < 0) {  /* unavailable cycle counter */
						  /* Real programmers can write FORTRAN in any language */
				goto estimate;
			}

			pln->pcost = t;
			ego->pcost += t;
			ego->need_timeout_check = 1;
		}
	}

	invoke_hook(ego, pln, p, 0);
}

/* maintain dynamic scoping of flags, nthr: */
static plan *invoke_solver(planner *ego, const problem *p, solver *s,
	const flags_t *nflags) {
	flags_t flags = ego->flags;
	int nthr = ego->nthr;
	plan *pln;
	ego->flags = *nflags;
	PLNR_TIMELIMIT_IMPATIENCE(ego) = 0;
	A(p->adt->problem_kind == s->adt->problem_kind);
	pln = s->adt->ifftw_mkplan(s, p, ego);
	ego->nthr = nthr;
	ego->flags = flags;
	return pln;
}

/* maintain the invariant TIMED_OUT ==> NEED_TIMEOUT_CHECK */
static int timeout_p(planner *ego, const problem *p) {
	/* do not timeout when estimating.  First, the estimator is the
	planner of last resort.  Second, calling fftw_elapsed_since() is
	slower than estimating */
	if (!ESTIMATEP(ego)) {
		/* do not assume that fftw_elapsed_since() is monotonic */
		if (ego->timed_out) {
			A(ego->need_timeout_check);
			return 1;
		}

		if (ego->timelimit >= 0 &&
			fftw_elapsed_since(ego, p, ego->start_time) >= ego->timelimit) {
			ego->timed_out = 1;
			ego->need_timeout_check = 1;
			return 1;
		}
	}

	A(!ego->timed_out);
	ego->need_timeout_check = 0;
	return 0;
}

static plan *search0(planner *ego, const problem *p, unsigned *slvndx,
	const flags_t *flagsp) {
	plan *best = 0;
	int best_not_yet_timed = 1;

	/* Do not start a search if the planner timed out. This check is
	necessary, lest the relaxation mechanism kick in */
	if (timeout_p(ego, p))
		return 0;

	FORALL_SOLVERS_OF_KIND(p->adt->problem_kind, ego, s, sp, {
		plan *pln;

	pln = invoke_solver(ego, p, s, flagsp);

	if (ego->need_timeout_check)
		if (timeout_p(ego, p)) {
			fftw_plan_destroy_internal(pln);
			fftw_plan_destroy_internal(best);
			return 0;
		}

	if (pln) {
		/* read COULD_PRUNE_NOW_P because PLN may be destroyed
		before we use COULD_PRUNE_NOW_P */
		int could_prune_now_p = pln->could_prune_now_p;

		if (best) {
			if (best_not_yet_timed) {
				evaluate_plan(ego, best, p);
				best_not_yet_timed = 0;
			}
			evaluate_plan(ego, pln, p);
			if (pln->pcost < best->pcost) {
				fftw_plan_destroy_internal(best);
				best = pln;
				*slvndx = (unsigned)/*from ptrdiff_t*/(sp - ego->slvdescs);
			}
			else {
				fftw_plan_destroy_internal(pln);
			}
		}
		else {
			best = pln;
			*slvndx = (unsigned)/*from ptrdiff_t*/(sp - ego->slvdescs);
		}

		if (ALLOW_PRUNINGP(ego) && could_prune_now_p)
			break;
	}
	});

	return best;
}

static plan *search(planner *ego, const problem *p, unsigned *slvndx,
	flags_t *flagsp) {
	plan *pln = 0;
	unsigned i;

	/* relax impatience in this order: */
	static const unsigned relax_tab[] = {
		0, /* relax nothing */
		NO_VRECURSE,
		NO_FIXED_RADIX_LARGE_N,
		NO_SLOW,
		NO_UGLY
	};

	unsigned l_orig = flagsp->l;
	unsigned x = flagsp->u;

	/* guaranteed to be different from X */
	unsigned last_x = ~x;

	for (i = 0; i < sizeof(relax_tab) / sizeof(relax_tab[0]); ++i) {
		if (LEQ(l_orig, x & ~relax_tab[i]))
			x = x & ~relax_tab[i];

		if (x != last_x) {
			last_x = x;
			flagsp->l = x;
			pln = search0(ego, p, slvndx, flagsp);
			if (pln) break;
		}
	}

	if (!pln) {
		/* search [L_ORIG, U] */
		if (l_orig != last_x) {
			last_x = l_orig;
			flagsp->l = l_orig;
			pln = search0(ego, p, slvndx, flagsp);
		}
	}

	return pln;
}

#define CHECK_FOR_BOGOSITY                        \
     if ((ego->bogosity_hook ?                        \
      (ego->wisdom_state = ego->bogosity_hook(ego->wisdom_state, p)) \
      : ego->wisdom_state) == WISDOM_IS_BOGUS)            \
      goto wisdom_is_bogus;

static plan *ifftw_mkplan(planner *ego, const problem *p) {
	plan *pln;
	md5 m;
	unsigned slvndx;
	flags_t flags_of_solution;
	solution *sol;
	solver *s;

	ASSERT_ALIGNED_DOUBLE;
	A(LEQ(PLNR_L(ego), PLNR_U(ego)));

	if (ESTIMATEP(ego))
		PLNR_TIMELIMIT_IMPATIENCE(ego) = 0; /* canonical form */


#ifdef FFTW_DEBUG
	check(&ego->htab_blessed);
	check(&ego->htab_unblessed);
#endif

	pln = 0;

	CHECK_FOR_BOGOSITY;

	ego->timed_out = 0;

	++ego->nprob;
	md5hash(&m, p, ego);

	flags_of_solution = ego->flags;

	if (ego->wisdom_state != WISDOM_IGNORE_ALL) {
		if ((sol = hlookup(ego, m.s, &flags_of_solution))) {
			/* wisdom is acceptable */
			wisdom_state_t owisdom_state = ego->wisdom_state;

			/* this hook is mainly for MPI, to make sure that
			wisdom is in sync across all processes for MPI problems */
			if (ego->wisdom_ok_hook && !ego->wisdom_ok_hook(p, sol->flags))
				goto do_search; /* ignore not-ok wisdom */

			slvndx = SLVNDX(sol);

			if (slvndx == INFEASIBLE_SLVNDX) {
				if (ego->wisdom_state == WISDOM_IGNORE_INFEASIBLE)
					goto do_search;
				else
					return 0;   /* known to be infeasible */
			}

			flags_of_solution = sol->flags;

			/* inherit blessing either from wisdom
			or from the planner */
			flags_of_solution.hash_info |= BLISS(ego->flags);

			ego->wisdom_state = WISDOM_ONLY;

			s = ego->slvdescs[slvndx].slv;
			if (p->adt->problem_kind != s->adt->problem_kind)
				goto wisdom_is_bogus;

			pln = invoke_solver(ego, p, s, &flags_of_solution);

			CHECK_FOR_BOGOSITY;      /* catch error in child solvers */

			sol = 0; /* Paranoia: SOL may be dangling after
					 invoke_solver(); make sure we don't accidentally
					 reuse it. */

			if (!pln)
				goto wisdom_is_bogus;

			ego->wisdom_state = owisdom_state;

			goto skip_search;
		}
		else if (ego->nowisdom_hook) /* for MPI, make sure lack of wisdom */
			ego->nowisdom_hook(p);  /*   is in sync across all processes */
	}

do_search:
	/* cannot search in WISDOM_ONLY mode */
	if (ego->wisdom_state == WISDOM_ONLY)
		goto wisdom_is_bogus;

	flags_of_solution = ego->flags;
	pln = search(ego, p, &slvndx, &flags_of_solution);
	CHECK_FOR_BOGOSITY;      /* catch error in child solvers */

	if (ego->timed_out) {
		A(!pln);
		if (PLNR_TIMELIMIT_IMPATIENCE(ego) != 0) {
			/* record (below) that this plan has failed because of
			timeout */
			flags_of_solution.hash_info |= BLESSING;
		}
		else {
			/* this is not the top-level problem or timeout is not
			active: record no wisdom. */
			return 0;
		}
	}
	else {
		/* canonicalize to infinite timeout */
		flags_of_solution.timelimit_impatience = 0;
	}

skip_search:
	if (ego->wisdom_state == WISDOM_NORMAL ||
		ego->wisdom_state == WISDOM_ONLY) {
		if (pln) {
			hinsert(ego, m.s, &flags_of_solution, slvndx);
			invoke_hook(ego, pln, p, 1);
		}
		else {
			hinsert(ego, m.s, &flags_of_solution, INFEASIBLE_SLVNDX);
		}
	}

	return pln;

wisdom_is_bogus:
	fftw_plan_destroy_internal(pln);
	ego->wisdom_state = WISDOM_IS_BOGUS;
	return 0;
}

static void htab_destroy(hashtab *ht) {
	fftw_ifree(ht->solutions);
	ht->solutions = 0;
	ht->nelem = 0U;
}

static void mkhashtab(hashtab *ht) {
	ht->nrehash = 0;
	ht->succ_lookup = ht->lookup = ht->lookup_iter = 0;
	ht->insert = ht->insert_iter = ht->insert_unknown = 0;

	ht->solutions = 0;
	ht->hashsiz = ht->nelem = 0U;
	hgrow(ht);            /* so that hashsiz > 0 */
}

/* destroy hash table entries.  If FORGET_EVERYTHING, destroy the whole
table.  If FORGET_ACCURSED, then destroy entries that are not blessed. */
static void ifftw_forget(planner *ego, amnesia a) {
	switch (a) {
	case FORGET_EVERYTHING:
		htab_destroy(&ego->htab_blessed);
		mkhashtab(&ego->htab_blessed);
		/* fall through */
	case FORGET_ACCURSED:
		htab_destroy(&ego->htab_unblessed);
		mkhashtab(&ego->htab_unblessed);
		break;
	default:
		break;
	}
}

/* FIXME: what sort of version information should we write? */
#define WISDOM_PREAMBLE PACKAGE "-" VERSION " " STRINGIZE(fftw_wisdom)
static const char stimeout[] = "TIMEOUT";

/* tantus labor non sit cassus */
static void ifftw_exprt(planner *ego, printer *p) {
	unsigned h;
	hashtab *ht = &ego->htab_blessed;
	md5 m;

	signature_of_configuration(&m, ego);

	p->print(p,
		"(" WISDOM_PREAMBLE " #x%M #x%M #x%M #x%M\n",
		m.s[0], m.s[1], m.s[2], m.s[3]);

	for (h = 0; h < ht->hashsiz; ++h) {
		solution *l = ht->solutions + h;
		if (LIVEP(l)) {
			const char *reg_nam;
			int reg_id;

			if (SLVNDX(l) == INFEASIBLE_SLVNDX) {
				reg_nam = stimeout;
				reg_id = 0;
			}
			else {
				slvdesc *sp = ego->slvdescs + SLVNDX(l);
				reg_nam = sp->reg_nam;
				reg_id = sp->reg_id;
			}

			/* qui salvandos salvas gratis
			salva me fons pietatis */
			p->print(p, "  (%s %d #x%x #x%x #x%x #x%M #x%M #x%M #x%M)\n",
				reg_nam, reg_id,
				l->flags.l, l->flags.u, l->flags.timelimit_impatience,
				l->s[0], l->s[1], l->s[2], l->s[3]);
		}
	}
	p->print(p, ")\n");
}

/* mors stupebit et natura
cum resurget creatura */
static int ifftw_imprt(planner *ego, scanner *sc) {
	char buf[MAXNAM + 1];
	md5uint sig[4];
	unsigned l, u, timelimit_impatience;
	flags_t flags;
	int reg_id;
	unsigned slvndx;
	hashtab *ht = &ego->htab_blessed;
	hashtab old;
	md5 m;

	if (!sc->scan(sc,
		"(" WISDOM_PREAMBLE " #x%M #x%M #x%M #x%M\n",
		sig + 0, sig + 1, sig + 2, sig + 3))
		return 0; /* don't need to restore hashtable */

	signature_of_configuration(&m, ego);
	if (m.s[0] != sig[0] || m.s[1] != sig[1] ||
		m.s[2] != sig[2] || m.s[3] != sig[3]) {
		/* invalid configuration */
		return 0;
	}

	/* make a backup copy of the hash table (cache the hash) */
	{
		unsigned h, hsiz = ht->hashsiz;
		old = *ht;
		old.solutions = (solution *)MALLOC(hsiz * sizeof(solution), HASHT);
		for (h = 0; h < hsiz; ++h)
			old.solutions[h] = ht->solutions[h];
	}

	while (1) {
		if (sc->scan(sc, ")"))
			break;

		/* qua resurget ex favilla */
		if (!sc->scan(sc, "(%*s %d #x%x #x%x #x%x #x%M #x%M #x%M #x%M)",
			MAXNAM, buf, &reg_id, &l, &u, &timelimit_impatience,
			sig + 0, sig + 1, sig + 2, sig + 3))
			goto bad;

		if (!strcmp(buf, stimeout) && reg_id == 0) {
			slvndx = INFEASIBLE_SLVNDX;
		}
		else {
			if (timelimit_impatience != 0)
				goto bad;

			slvndx = slookup(ego, buf, reg_id);
			if (slvndx == INFEASIBLE_SLVNDX)
				goto bad;
		}

		/* inter oves locum praesta */
		flags.l = l;
		flags.u = u;
		flags.timelimit_impatience = timelimit_impatience;
		flags.hash_info = BLESSING;

		CK(flags.l == l);
		CK(flags.u == u);
		CK(flags.timelimit_impatience == timelimit_impatience);

		if (!hlookup(ego, sig, &flags))
			hinsert(ego, sig, &flags, slvndx);
	}

	fftw_ifree0(old.solutions);
	return 1;

bad:
	/* ``The wisdom of FFTW must be above suspicion.'' */
	fftw_ifree0(ht->solutions);
	*ht = old;
	return 0;
}

/*
* create a planner
*/
planner *fftw_mkplanner(void) {
	int i;

	static const ifftw_planner_adt ifftw_padt = {
		ifftw_register_solver, ifftw_mkplan, ifftw_forget, ifftw_exprt, ifftw_imprt
	};

	planner *p = (planner *)MALLOC(sizeof(planner), PLANNERS);

	p->adt = &ifftw_padt;
	p->nplan = p->nprob = 0;
	p->pcost = p->epcost = 0.0;
	p->hook = 0;
	p->cost_hook = 0;
	p->wisdom_ok_hook = 0;
	p->nowisdom_hook = 0;
	p->bogosity_hook = 0;
	p->cur_reg_nam = 0;
	p->wisdom_state = WISDOM_NORMAL;

	p->slvdescs = 0;
	p->nslvdesc = p->slvdescsiz = 0;

	p->flags.l = 0;
	p->flags.u = 0;
	p->flags.timelimit_impatience = 0;
	p->flags.hash_info = 0;
	p->nthr = 1;
	p->need_timeout_check = 1;
	p->timelimit = -1;

	mkhashtab(&p->htab_blessed);
	mkhashtab(&p->htab_unblessed);

	for (i = 0; i < PROBLEM_LAST; ++i)
		p->slvdescs_for_problem_kind[i] = -1;

	return p;
}

void fftw_planner_destroy(planner *ego) {
	/* destroy hash table */
	htab_destroy(&ego->htab_blessed);
	htab_destroy(&ego->htab_unblessed);

	/* destroy solvdesc table */
	FORALL_SOLVERS(ego, s, sp, {
		UNUSED(sp);
	fftw_solver_destroy(s);
	});

	fftw_ifree0(ego->slvdescs);
	fftw_ifree(ego); /* dona eis requiem */
}

plan *fftw_mkplan_d(planner *ego, problem *p) {
	plan *pln = ego->adt->ifftw_mkplan(ego, p);
	fftw_problem_destroy(p);
	return pln;
}

/* like fftw_mkplan_d, but sets/resets flags as well */
plan *fftw_mkplan_f_d(planner *ego, problem *p,
	unsigned l_set, unsigned u_set, unsigned u_reset) {
	flags_t oflags = ego->flags;
	plan *pln;

	PLNR_U(ego) &= ~u_reset;
	PLNR_L(ego) &= ~u_reset;
	PLNR_L(ego) |= l_set;
	PLNR_U(ego) |= u_set | l_set;
	pln = fftw_mkplan_d(ego, p);
	ego->flags = oflags;
	return pln;
}

/*
* Debugging code:
*/
#ifdef FFTW_DEBUG
static void check(hashtab *ht)
{
	unsigned live = 0;
	unsigned i;

	A(ht->nelem < ht->hashsiz);

	for (i = 0; i < ht->hashsiz; ++i) {
		solution *l = ht->solutions + i;
		if (LIVEP(l))
			++live;
	}

	A(ht->nelem == live);

	for (i = 0; i < ht->hashsiz; ++i) {
		solution *l1 = ht->solutions + i;
		int foundit = 0;
		if (LIVEP(l1)) {
			unsigned g, h = h1(ht, l1->s), d = h2(ht, l1->s);

			g = h;
			do {
				solution *l = ht->solutions + g;
				if (VALIDP(l)) {
					if (l1 == l)
						foundit = 1;
					else if (LIVEP(l) && md5eq(l1->s, l->s)) {
						A(!subsumes(&l->flags, SLVNDX(l), &l1->flags));
						A(!subsumes(&l1->flags, SLVNDX(l1), &l->flags));
					}
				}
				else
					break;
				g = addmod(g, d, ht->hashsiz);
			} while (g != h);

			A(foundit);
		}
	}
}
#endif

void *fftw_malloc_plain(size_t n) {
	void *p;
	if (n == 0)
		n = 1;
	p = fftw_kernel_malloc(n);
	CK(p);

#ifdef MIN_ALIGNMENT
	A((((uintptr_t)p) % MIN_ALIGNMENT) == 0);
#endif

	return p;
}

void fftw_ifree(void *p) {
	fftw_kernel_free(p);
}

void fftw_ifree0(void *p) {
	/* common pattern */
	if (p) fftw_ifree(p);
}

/***************************************************************************/

/* Rader's algorithm requires lots of modular arithmetic, and if we
aren't careful we can have errors due to integer overflows. */

/* Compute (x * y) mod p, but watch out for integer overflows; we must
have 0 <= {x, y} < p.

If overflow is common, this routine is somewhat slower than
e.g. using 'long long' arithmetic.  However, it has the advantage
of working when INT is 64 bits, and is also faster when overflow is
rare.  FFTW calls this via the MULMOD macro, which further
optimizes for the case of small integers.
*/

#define ADD_MOD(x, y, p) ((x) >= (p) - (y)) ? ((x) + ((y) - (p))) : ((x) + (y))

INT fftw_safe_mulmod(INT x, INT y, INT p) {
	INT r;

	if (y > x)
		return fftw_safe_mulmod(y, x, p);

	A(0 <= y && x < p);

	r = 0;
	while (y) {
		r = ADD_MOD(r, x * (y & 1), p);
		y >>= 1;
		x = ADD_MOD(x, x, p);
	}

	return r;
}

/***************************************************************************/

/* Compute n^m mod p, where m >= 0 and p > 0.  If we really cared, we
could make this tail-recursive. */

INT fftw_power_mod(INT n, INT m, INT p) {
	A(p > 0);
	if (m == 0)
		return 1;
	else if (m % 2 == 0) {
		INT x = fftw_power_mod(n, m / 2, p);
		return MULMOD(x, x, p);
	}
	else
		return MULMOD(n, fftw_power_mod(n, m - 1, p), p);
}

/* the following two routines were contributed by Greg Dionne. */
static INT get_prime_factors(INT n, INT *primef) {
	INT i;
	INT size = 0;

	A(n % 2 == 0); /* this routine is designed only for even n */
	primef[size++] = (INT)2;
	do {
		n >>= 1;
	} while ((n & 1) == 0);

	if (n == 1)
		return size;

	for (i = 3; i * i <= n; i += 2)
		if (!(n % i)) {
			primef[size++] = i;
			do {
				n /= i;
			} while (!(n % i));
		}
	if (n == 1)
		return size;
	primef[size++] = n;
	return size;
}

INT fftw_find_generator(INT p) {
	INT n, i, size;
	INT primef[16];     /* smallest number = 32589158477190044730 > 2^64 */
	INT pm1 = p - 1;

	if (p == 2)
		return 1;

	size = get_prime_factors(pm1, primef);
	n = 2;
	for (i = 0; i < size; i++)
		if (fftw_power_mod(n, pm1 / primef[i], p) == 1) {
			i = -1;
			n++;
		}
	return n;
}

/* Return first prime divisor of n  (It would be at best slightly faster to
search a static table of primes; there are 6542 primes < 2^16.)  */
INT fftw_first_divisor(INT n) {
	INT i;
	if (n <= 1)
		return n;
	if (n % 2 == 0)
		return 2;
	for (i = 3; i * i <= n; i += 2)
		if (n % i == 0)
			return i;
	return n;
}

int fftw_is_prime(INT n) {
	return (n > 1 && fftw_first_divisor(n) == n);
}

INT fftw_next_prime(INT n) {
	while (!fftw_is_prime(n)) ++n;
	return n;
}

int fftw_factors_into(INT n, const INT *primes) {
	for (; *primes != 0; ++primes)
		while ((n % *primes) == 0)
			n /= *primes;
	return (n == 1);
}

/* integer square root.  Return floor(sqrt(N)) */
INT fftw_isqrt(INT n) {
	INT guess, iguess;

	A(n >= 0);
	if (n == 0) return 0;

	guess = n;
	iguess = 1;

	do {
		guess = (guess + iguess) / 2;
		iguess = n / guess;
	} while (guess > iguess);

	return guess;
}

static INT isqrt_maybe(INT n) {
	INT guess = fftw_isqrt(n);
	return guess * guess == n ? guess : 0;
}

#define divides(a, b) (((b) % (a)) == 0)

INT fftw_choose_radix(INT r, INT n) {
	if (r > 0) {
		if (divides(r, n)) return r;
		return 0;
	}
	else if (r == 0) {
		return fftw_first_divisor(n);
	}
	else {
		/* r is negative.  If n = (-r) * q^2, take q as the radix */
		r = 0 - r;
		return (n > r && divides(r, n)) ? isqrt_maybe(n / r) : 0;
	}
}

/* return A mod N, works for all A including A < 0 */
INT fftw_modulo(INT a, INT n) {
	A(n > 0);
	if (a >= 0)
		return a % n;
	else
		return (n - 1) - ((-(a + (INT)1)) % n);
}

/* TRUE if N factors into small primes */
int fftw_factors_into_small_primes(INT n) {
	static const INT primes[] = { 2, 3, 5, 0 };
	return fftw_factors_into(n, primes);
}

#define BSZ 64

static void myputs(printer *p, const char *s) {
	char c;
	while ((c = *s++))
		p->putchr(p, c);
}

static void newline(printer *p) {
	int i;

	p->putchr(p, '\n');
	for (i = 0; i < p->indent; ++i)
		p->putchr(p, ' ');
}

static const char *digits = "0123456789abcdef";

static void putint(printer *p, INT i) {
	char buf[BSZ];
	char *f = buf;

	if (i < 0) {
		p->putchr(p, '-');
		i = -i;
	}

	do {
		*f++ = digits[i % 10];
		i /= 10;
	} while (i);

	do {
		p->putchr(p, *--f);
	} while (f != buf);
}

static void putulong(printer *p, unsigned long i, unsigned base, int width) {
	char buf[BSZ];
	char *f = buf;

	do {
		*f++ = digits[i % base];
		i /= base;
	} while (i);

	while (width > f - buf) {
		p->putchr(p, '0');
		--width;
	}

	do {
		p->putchr(p, *--f);
	} while (f != buf);
}

static void vprint(printer *p, const char *format, va_list ap) {
	const char *s = format;
	char c;
	INT ival;

	while ((c = *s++)) {
		switch (c) {
		case '%':
			switch ((c = *s++)) {
			case 'M': {
				/* md5 value */
				md5uint x = va_arg(ap, md5uint);
				putulong(p, (unsigned long)(0xffffffffUL & x),
					16u, 8);
				break;
			}
			case 'c': {
				int x = va_arg(ap, int);
				p->putchr(p, (char)x);
				break;
			}
			case 's': {
				char *x = va_arg(ap, char *);
				if (x)
					myputs(p, x);
				else
					goto putnull;
				break;
			}
			case 'd': {
				int x = va_arg(ap, int);
				ival = (INT)x;
				goto putival;
			}
			case 'D': {
				ival = va_arg(ap, INT);
				goto putival;
			}
			case 'v': {
				/* print optional vector length */
				ival = va_arg(ap, INT);
				if (ival > 1) {
					myputs(p, "-x");
					goto putival;
				}
				break;
			}
			case 'o': {
				/* integer option.  Usage: %oNAME= */
				ival = va_arg(ap, INT);
				if (ival)
					p->putchr(p, '/');
				while ((c = *s++) != '=')
					if (ival)
						p->putchr(p, c);
				if (ival) {
					p->putchr(p, '=');
					goto putival;
				}
				break;
			}
			case 'u': {
				unsigned x = va_arg(ap, unsigned);
				putulong(p, (unsigned long)x, 10u, 0);
				break;
			}
			case 'x': {
				unsigned x = va_arg(ap, unsigned);
				putulong(p, (unsigned long)x, 16u, 0);
				break;
			}
			case '(': {
				/* newline, augment indent level */
				p->indent += p->indent_incr;
				newline(p);
				break;
			}
			case ')': {
				/* decrement indent level */
				p->indent -= p->indent_incr;
				break;
			}
			case 'p': {  /* note difference from C's %p */
						 /* print plan */
				plan *x = va_arg(ap, plan *);
				if (x)
					x->adt->print(x, p);
				else
					goto putnull;
				break;
			}
			case 'P': {
				/* print problem */
				problem *x = va_arg(ap, problem *);
				if (x)
					x->adt->print(x, p);
				else
					goto putnull;
				break;
			}
			case 'T': {
				/* print tensor */
				tensor *x = va_arg(ap, tensor *);
				if (x)
					fftw_tensor_print(x, p);
				else
					goto putnull;
				break;
			}
			default:
				A(0 /* unknown format */);
				break;

			putnull:
				myputs(p, "(null)");
				break;

			putival:
				putint(p, ival);
				break;
			}
			break;
		default:
			p->putchr(p, c);
			break;
		}
	}
}

static void print(printer *p, const char *format, ...) {
	va_list ap;
	va_start(ap, format);
	vprint(p, format, ap);
	va_end(ap);
}

printer *fftw_mkprinter(size_t size,
	void(*putchr)(printer *p, char c),
	void(*cleanup)(printer *p)) {
	printer *s = (printer *)MALLOC(size, OTHER);
	s->print = print;
	s->vprint = vprint;
	s->putchr = putchr;
	s->cleanup = cleanup;
	s->indent = 0;
	s->indent_incr = 2;
	return s;
}

void fftw_printer_destroy(printer *p) {
	if (p->cleanup)
		p->cleanup(p);
	fftw_ifree(p);
}

/* constructor */
problem *fftw_mkproblem(size_t sz, const problem_adt *adt) {
	problem *p = (problem *)MALLOC(sz, PROBLEMS);

	p->adt = adt;
	return p;
}

/* destructor */
void fftw_problem_destroy(problem *ego) {
	if (ego)
		ego->adt->destroy(ego);
}

/* management of unsolvable problems */
static void unsolvable_destroy(problem *ego) {
	UNUSED(ego);
}

static void unsolvable_hash(const problem *p, md5 *m) {
	UNUSED(p);
	fftw_md5puts(m, "unsolvable");
}

static void unsolvable_print(const problem *ego, printer *p) {
	UNUSED(ego);
	p->print(p, "(unsolvable)");
}

static void unsolvable_zero(const problem *ego) {
	UNUSED(ego);
}

static const problem_adt ifftw_padt =
{
	PROBLEM_UNSOLVABLE,
	unsolvable_hash,
	unsolvable_zero,
	unsolvable_print,
	unsolvable_destroy
};

/* there is no point in malloc'ing this one */
static problem the_unsolvable_problem = { &ifftw_padt };

problem *fftw_mkproblem_unsolvable(void) {
	return &the_unsolvable_problem;
}

/* out of place 1D copy routine */


void fftw_cpy1d(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, INT n0, INT is0, INT os0, INT vl) {
	INT i0, v;

	A(I != O);
	switch (vl) {
	case 1:
		if ((n0 & 1) || is0 != 1 || os0 != 1) {
			for (; n0 > 0; --n0, I += is0, O += os0)
				*O = *I;
			break;
		}
		n0 /= 2;
		is0 = 2;
		os0 = 2;
		/* fall through */
	case 2:
		if ((n0 & 1) || is0 != 2 || os0 != 2) {
			for (; n0 > 0; --n0, I += is0, O += os0) {
				FFTW_REAL_TYPE x0 = I[0];
				FFTW_REAL_TYPE x1 = I[1];
				O[0] = x0;
				O[1] = x1;
			}
			break;
		}
		n0 /= 2;
		is0 = 4;
		os0 = 4;
		/* fall through */
	case 4:
		for (; n0 > 0; --n0, I += is0, O += os0) {
			FFTW_REAL_TYPE x0 = I[0];
			FFTW_REAL_TYPE x1 = I[1];
			FFTW_REAL_TYPE x2 = I[2];
			FFTW_REAL_TYPE x3 = I[3];
			O[0] = x0;
			O[1] = x1;
			O[2] = x2;
			O[3] = x3;
		}
		break;
	default:
		for (i0 = 0; i0 < n0; ++i0)
			for (v = 0; v < vl; ++v) {
				FFTW_REAL_TYPE x0 = I[i0 * is0 + v];
				O[i0 * os0 + v] = x0;
			}
		break;
	}
}

/* out of place 2D copy routines */


#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#  ifdef HAVE_XMMINTRIN_H
#    include <xmmintrin.h>
#    define WIDE_TYPE __m128
#  endif
#endif

#ifndef WIDE_TYPE
/* fall back to double, which means that WIDE_TYPE will be unused */
#  define WIDE_TYPE double
#endif

void fftw_cpy2d(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT vl) {
	INT i0, i1, v;

	switch (vl) {
	case 1:
		for (i1 = 0; i1 < n1; ++i1)
			for (i0 = 0; i0 < n0; ++i0) {
				FFTW_REAL_TYPE x0 = I[i0 * is0 + i1 * is1];
				O[i0 * os0 + i1 * os1] = x0;
			}
		break;
	case 2:
		if (1
			&& (2 * sizeof(FFTW_REAL_TYPE) == sizeof(WIDE_TYPE))
			&& (sizeof(WIDE_TYPE) > sizeof(double))
			&& (((size_t)I) % sizeof(WIDE_TYPE) == 0)
			&& (((size_t)O) % sizeof(WIDE_TYPE) == 0)
			&& ((is0 & 1) == 0)
			&& ((is1 & 1) == 0)
			&& ((os0 & 1) == 0)
			&& ((os1 & 1) == 0)) {
			/* copy R[2] as WIDE_TYPE if WIDE_TYPE is large
			enough to hold R[2], and if the input is
			properly aligned.  This is a win when R==double
			and WIDE_TYPE is 128 bits. */
			for (i1 = 0; i1 < n1; ++i1)
				for (i0 = 0; i0 < n0; ++i0) {
					*(WIDE_TYPE *)&O[i0 * os0 + i1 * os1] =
						*(WIDE_TYPE *)&I[i0 * is0 + i1 * is1];
				}
		}
		else if (1
			&& (2 * sizeof(FFTW_REAL_TYPE) == sizeof(double))
			&& (((size_t)I) % sizeof(double) == 0)
			&& (((size_t)O) % sizeof(double) == 0)
			&& ((is0 & 1) == 0)
			&& ((is1 & 1) == 0)
			&& ((os0 & 1) == 0)
			&& ((os1 & 1) == 0)) {
			/* copy R[2] as double if double is large enough to
			hold R[2], and if the input is properly aligned.
			This case applies when R==float */
			for (i1 = 0; i1 < n1; ++i1)
				for (i0 = 0; i0 < n0; ++i0) {
					*(double *)&O[i0 * os0 + i1 * os1] =
						*(double *)&I[i0 * is0 + i1 * is1];
				}
		}
		else {
			for (i1 = 0; i1 < n1; ++i1)
				for (i0 = 0; i0 < n0; ++i0) {
					FFTW_REAL_TYPE x0 = I[i0 * is0 + i1 * is1];
					FFTW_REAL_TYPE x1 = I[i0 * is0 + i1 * is1 + 1];
					O[i0 * os0 + i1 * os1] = x0;
					O[i0 * os0 + i1 * os1 + 1] = x1;
				}
		}
		break;
	default:
		for (i1 = 0; i1 < n1; ++i1)
			for (i0 = 0; i0 < n0; ++i0)
				for (v = 0; v < vl; ++v) {
					FFTW_REAL_TYPE x0 = I[i0 * is0 + i1 * is1 + v];
					O[i0 * os0 + i1 * os1 + v] = x0;
				}
		break;
	}
}

/* like cpy2d, but read input contiguously if possible */
void fftw_cpy2d_ci(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT vl) {
	if (IABS(is0) < IABS(is1))    /* inner loop is for n0 */
		fftw_cpy2d(I, O, n0, is0, os0, n1, is1, os1, vl);
	else
		fftw_cpy2d(I, O, n1, is1, os1, n0, is0, os0, vl);
}

/* like cpy2d, but write output contiguously if possible */
void fftw_cpy2d_co(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT vl) {
	if (IABS(os0) < IABS(os1))    /* inner loop is for n0 */
		fftw_cpy2d(I, O, n0, is0, os0, n1, is1, os1, vl);
	else
		fftw_cpy2d(I, O, n1, is1, os1, n0, is0, os0, vl);
}


/* tiled copy routines */
struct cpy2d_closure {
	FFTW_REAL_TYPE *I, *O;
	INT is0, os0, is1, os1, vl;
	FFTW_REAL_TYPE *buf;
};

static void cpy2d_dotile(INT n0l, INT n0u, INT n1l, INT n1u, void *args) {
	struct cpy2d_closure *k = (struct cpy2d_closure *) args;
	fftw_cpy2d(k->I + n0l * k->is0 + n1l * k->is1,
		k->O + n0l * k->os0 + n1l * k->os1,
		n0u - n0l, k->is0, k->os0,
		n1u - n1l, k->is1, k->os1,
		k->vl);
}

static void cpy2d_dotile_buf(INT n0l, INT n0u, INT n1l, INT n1u, void *args) {
	struct cpy2d_closure *k = (struct cpy2d_closure *) args;

	/* copy from I to buf */
	fftw_cpy2d_ci(k->I + n0l * k->is0 + n1l * k->is1,
		k->buf,
		n0u - n0l, k->is0, k->vl,
		n1u - n1l, k->is1, k->vl * (n0u - n0l),
		k->vl);

	/* copy from buf to O */
	fftw_cpy2d_co(k->buf,
		k->O + n0l * k->os0 + n1l * k->os1,
		n0u - n0l, k->vl, k->os0,
		n1u - n1l, k->vl * (n0u - n0l), k->os1,
		k->vl);
}


void fftw_cpy2d_tiled(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1, INT vl) {
	INT tilesz = fftw_compute_tilesz(vl,
		1 /* input array */
		+ 1 /* ouput array */);
	struct cpy2d_closure k;
	k.I = I;
	k.O = O;
	k.is0 = is0;
	k.os0 = os0;
	k.is1 = is1;
	k.os1 = os1;
	k.vl = vl;
	k.buf = 0; /* unused */
	fftw_tile2d(0, n0, 0, n1, tilesz, cpy2d_dotile, &k);
}

void fftw_cpy2d_tiledbuf(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1, INT vl) {
	FFTW_REAL_TYPE buf[CACHESIZE / (2 * sizeof(FFTW_REAL_TYPE))];
	/* input and buffer in cache, or
	output and buffer in cache */
	INT tilesz = fftw_compute_tilesz(vl, 2);
	struct cpy2d_closure k;
	k.I = I;
	k.O = O;
	k.is0 = is0;
	k.os0 = os0;
	k.is1 = is1;
	k.os1 = os1;
	k.vl = vl;
	k.buf = buf;
	A(tilesz * tilesz * vl * sizeof(FFTW_REAL_TYPE) <= sizeof(buf));
	fftw_tile2d(0, n0, 0, n1, tilesz, cpy2d_dotile_buf, &k);
}

void fftw_cpy2d_pair(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1) {
	INT i0, i1;

	for (i1 = 0; i1 < n1; ++i1)
		for (i0 = 0; i0 < n0; ++i0) {
			FFTW_REAL_TYPE x0 = I0[i0 * is0 + i1 * is1];
			FFTW_REAL_TYPE x1 = I1[i0 * is0 + i1 * is1];
			O0[i0 * os0 + i1 * os1] = x0;
			O1[i0 * os0 + i1 * os1] = x1;
		}
}

void fftw_zero1d_pair(FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1, INT n0, INT os0) {
	INT i0;
	for (i0 = 0; i0 < n0; ++i0) {
		O0[i0 * os0] = 0;
		O1[i0 * os0] = 0;
	}
}

/* like cpy2d_pair, but read input contiguously if possible */
void fftw_cpy2d_pair_ci(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1) {
	if (IABS(is0) < IABS(is1))    /* inner loop is for n0 */
		fftw_cpy2d_pair(I0, I1, O0, O1, n0, is0, os0, n1, is1, os1);
	else
		fftw_cpy2d_pair(I0, I1, O0, O1, n1, is1, os1, n0, is0, os0);
}

/* like cpy2d_pair, but write output contiguously if possible */
void fftw_cpy2d_pair_co(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
	INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1) {
	if (IABS(os0) < IABS(os1))    /* inner loop is for n0 */
		fftw_cpy2d_pair(I0, I1, O0, O1, n0, is0, os0, n1, is1, os1);
	else
		fftw_cpy2d_pair(I0, I1, O0, O1, n1, is1, os1, n0, is0, os0);
}

/* routines shared by the various buffered solvers */


#define DEFAULT_MAXNBUF ((INT)256)

/* approx. 512KB of buffers for complex data */
#define MAXBUFSZ (256 * 1024 / (INT)(sizeof(FFTW_REAL_TYPE)))

INT fftw_nbuf(INT n, INT vl, INT maxnbuf) {
	INT i, nbuf, lb;

	if (!maxnbuf)
		maxnbuf = DEFAULT_MAXNBUF;

	nbuf = fftw_imin(maxnbuf,
		fftw_imin(vl, fftw_imax((INT)1, MAXBUFSZ / n)));

	/*
	* Look for a buffer number (not too small) that divides the
	* vector length, in order that we only need one child plan:
	*/
	lb = fftw_imax(1, nbuf / 4);
	for (i = nbuf; i >= lb; --i)
		if (vl % i == 0)
			return i;

	/* whatever... */
	return nbuf;
}

#define SKEW 6 /* need to be even for SIMD */
#define SKEWMOD 8

INT fftw_bufdist(INT n, INT vl) {
	if (vl == 1)
		return n;
	else
		/* return smallest X such that X >= N and X == SKEW (mod SKEWMOD) */
		return n + fftw_modulo(SKEW - n, SKEWMOD);
}

int fftw_toobig(INT n) {
	return n > MAXBUFSZ;
}

/* TRUE if there exists i < which such that maxnbuf[i] and
maxnbuf[which] yield the same value, in which case we canonicalize
on the minimum value */
int fftw_nbuf_redundant(INT n, INT vl, size_t which,
	const INT *maxnbuf, size_t nmaxnbuf) {
	size_t i;
	(void)nmaxnbuf; /* UNUSED */
	for (i = 0; i < which; ++i)
		if (fftw_nbuf(n, vl, maxnbuf[i]) == fftw_nbuf(n, vl, maxnbuf[which]))
			return 1;
	return 0;
}

void fftw_null_awake(plan *ego, enum wakefulness wakefulness) {
	UNUSED(ego);
	UNUSED(wakefulness);
	/* do nothing */
}

void fftw_assertion_failed(const char *s, int line, const char *file) {
	fflush(stdout);
	fprintf(stderr, "fftw: %s:%d: assertion failed: %s\n", file, line, s);
#ifdef HAVE_ABORT
	abort();
#else
	exit(EXIT_FAILURE);
#endif
}

/*
common routines for Rader solvers
*/


/* shared twiddle and omega lists, keyed by two/three integers. */
struct rader_tls {
	INT k1, k2, k3;
	FFTW_REAL_TYPE *W;
	int refcnt;
	rader_tl *cdr;
};

void fftw_rader_tl_insert(INT k1, INT k2, INT k3, FFTW_REAL_TYPE *W, rader_tl **tl) {
	rader_tl *t = (rader_tl *)MALLOC(sizeof(rader_tl), TWIDDLES);
	t->k1 = k1;
	t->k2 = k2;
	t->k3 = k3;
	t->W = W;
	t->refcnt = 1;
	t->cdr = *tl;
	*tl = t;
}

FFTW_REAL_TYPE *fftw_rader_tl_find(INT k1, INT k2, INT k3, rader_tl *t) {
	while (t && (t->k1 != k1 || t->k2 != k2 || t->k3 != k3))
		t = t->cdr;
	if (t) {
		++t->refcnt;
		return t->W;
	}
	else
		return 0;
}

void fftw_rader_tl_delete(FFTW_REAL_TYPE *W, rader_tl **tl) {
	if (W) {
		rader_tl **tp, *t;

		for (tp = tl; (t = *tp) && t->W != W; tp = &t->cdr);

		if (t && --t->refcnt <= 0) {
			*tp = t->cdr;
			fftw_ifree(t->W);
			fftw_ifree(t);
		}
	}
}

#ifdef USE_CTYPE
#include <ctype.h>
#else
/* Screw ctype. On linux, the is* functions call a routine that gets
the ctype map in the current locale.  Because this operation is
expensive, the map is cached on a per-thread basis.  I am not
willing to link this crap with FFTW.  Not over my dead body.

Sic transit gloria mundi.
*/
#undef isspace
#define isspace(x) ((x) >= 0 && (x) <= ' ')
#undef isdigit
#define isdigit(x) ((x) >= '0' && (x) <= '9')
#undef isupper
#define isupper(x) ((x) >= 'A' && (x) <= 'Z')
#undef islower
#define islower(x) ((x) >= 'a' && (x) <= 'z')
#endif

static int mygetc(scanner *sc) {
	if (sc->ungotc != EOF) {
		int c = sc->ungotc;
		sc->ungotc = EOF;
		return c;
	}
	return (sc->getchr(sc));
}

#define GETCHR(sc) mygetc(sc)

static void myungetc(scanner *sc, int c) {
	sc->ungotc = c;
}

#define UNGETCHR(sc, c) myungetc(sc, c)

static void eat_blanks(scanner *sc) {
	int ch;
	while (ch = GETCHR(sc), isspace(ch));
	UNGETCHR(sc, ch);
}

static void mygets(scanner *sc, char *s, int maxlen) {
	char *s0 = s;
	int ch;

	A(maxlen > 0);
	while ((ch = GETCHR(sc)) != EOF && !isspace(ch)
		&& ch != ')' && ch != '(' && s < s0 + maxlen)
		*s++ = (char)(ch & 0xFF);
	*s = 0;
	UNGETCHR(sc, ch);
}

static long getlong(scanner *sc, int base, int *ret) {
	int sign = 1, ch, count;
	long x = 0;

	ch = GETCHR(sc);
	if (ch == '-' || ch == '+') {
		sign = ch == '-' ? -1 : 1;
		ch = GETCHR(sc);
	}
	for (count = 0;; ++count) {
		if (isdigit(ch))
			ch -= '0';
		else if (isupper(ch))
			ch -= 'A' - 10;
		else if (islower(ch))
			ch -= 'a' - 10;
		else
			break;
		x = x * base + ch;
		ch = GETCHR(sc);
	}
	x *= sign;
	UNGETCHR(sc, ch);
	*ret = count > 0;
	return x;
}

/* vscan is mostly scanf-like, with our additional format specifiers,
but with a few twists.  It returns simply 0 or 1 indicating whether
the match was successful. '(' and ')' in the format string match
those characters preceded by any whitespace.  Finally, if a
character match fails, it will ungetchr() the last character back
onto the stream. */
static int vscan(scanner *sc, const char *format, va_list ap) {
	const char *s = format;
	char c;
	int ch = 0;
	int fmt_len;

	while ((c = *s++)) {
		fmt_len = 0;
		switch (c) {
		case '%':
		getformat:
			switch ((c = *s++)) {
			case 's': {
				char *x = va_arg(ap, char *);
				mygets(sc, x, fmt_len);
				break;
			}
			case 'd': {
				int *x = va_arg(ap, int *);
				*x = (int)getlong(sc, 10, &ch);
				if (!ch) return 0;
				break;
			}
			case 'x': {
				int *x = va_arg(ap, int *);
				*x = (int)getlong(sc, 16, &ch);
				if (!ch) return 0;
				break;
			}
			case 'M': {
				md5uint *x = va_arg(ap, md5uint *);
				*x = (md5uint)
					(0xFFFFFFFF & getlong(sc, 16, &ch));
				if (!ch) return 0;
				break;
			}
			case '*': {
				if ((fmt_len = va_arg(ap, int)) <= 0) return 0;
				goto getformat;
			}
			default:
				A(0 /* unknown format */);
				break;
			}
			break;
		default:
			if (isspace(c) || c == '(' || c == ')')
				eat_blanks(sc);
			if (!isspace(c) && (ch = GETCHR(sc)) != c) {
				UNGETCHR(sc, ch);
				return 0;
			}
			break;
		}
	}
	return 1;
}

static int scan(scanner *sc, const char *format, ...) {
	int ret;
	va_list ap;
	va_start(ap, format);
	ret = vscan(sc, format, ap);
	va_end(ap);
	return ret;
}

scanner *fftw_mkscanner(size_t size, int(*getchr)(scanner *sc)) {
	scanner *s = (scanner *)MALLOC(size, OTHER);
	s->scan = scan;
	s->vscan = vscan;
	s->getchr = getchr;
	s->ungotc = EOF;
	return s;
}

void fftw_scanner_destroy(scanner *sc) {
	fftw_ifree(sc);
}

solver *fftw_mksolver(size_t size, const solver_adt *adt) {
	solver *s = (solver *)MALLOC(size, SOLVERS);

	s->adt = adt;
	s->refcnt = 0;
	return s;
}

void fftw_solver_use(solver *ego) {
	++ego->refcnt;
}

void fftw_solver_destroy(solver *ego) {
	if ((--ego->refcnt) == 0) {
		if (ego->adt->destroy)
			ego->adt->destroy(ego);
		fftw_ifree(ego);
	}
}

void fftw_solver_register(planner *plnr, solver *s) {
	plnr->adt->ifftw_register_solver(plnr, s);
}

void fftw_solvtab_exec(const solvtab tbl, planner *p) {
	for (; tbl->reg_nam; ++tbl) {
		p->cur_reg_nam = tbl->reg_nam;
		p->cur_reg_id = 0;
		tbl->reg(p);
	}
	p->cur_reg_nam = 0;
}


const INT fftw_an_INT_guaranteed_to_be_zero = 0;

#ifdef PRECOMPUTE_ARRAY_INDICES

stride fftw_mkstride(INT n, INT s) {
	int i;
	INT *p;

	A(n >= 0);
	p = (INT *)MALLOC((size_t)n * sizeof(INT), STRIDES);

	for (i = 0; i < n; ++i)
		p[i] = s * i;

	return p;
}

void fftw_stride_destroy(stride p) {
	fftw_ifree0(p);
}

#endif

tensor *fftw_mktensor(int rnk) {
	tensor *x;

	A(rnk >= 0);

#if defined(STRUCT_HACK_KR)
	if (FINITE_RNK(rnk) && rnk > 1)
		x = (tensor *)MALLOC(sizeof(tensor) + (unsigned)(rnk - 1) * sizeof(iodim),
			TENSORS);
	else
		x = (tensor *)MALLOC(sizeof(tensor), TENSORS);
#elif defined(STRUCT_HACK_C99)
	if (FINITE_RNK(rnk))
		x = (tensor *)MALLOC(sizeof(tensor) + (unsigned)rnk * sizeof(iodim),
			TENSORS);
	else
		x = (tensor *)MALLOC(sizeof(tensor), TENSORS);
#else
	x = (tensor *)MALLOC(sizeof(tensor), TENSORS);
	if (FINITE_RNK(rnk) && rnk > 0)
		x->dims = (iodim *)MALLOC(sizeof(iodim) * (unsigned)rnk, TENSORS);
	else
		x->dims = 0;
#endif

	x->rnk = rnk;
	return x;
}

void fftw_tensor_destroy(tensor *sz) {
#if !defined(STRUCT_HACK_C99) && !defined(STRUCT_HACK_KR)
	fftw_ifree0(sz->dims);
#endif
	fftw_ifree(sz);
}

INT fftw_tensor_sz(const tensor *sz) {
	int i;
	INT n = 1;

	if (!FINITE_RNK(sz->rnk))
		return 0;

	for (i = 0; i < sz->rnk; ++i)
		n *= sz->dims[i].n;
	return n;
}

void fftw_tensor_md5(md5 *p, const tensor *t) {
	int i;
	fftw_md5int(p, t->rnk);
	if (FINITE_RNK(t->rnk)) {
		for (i = 0; i < t->rnk; ++i) {
			const iodim *q = t->dims + i;
			fftw_md5INT(p, q->n);
			fftw_md5INT(p, q->is);
			fftw_md5INT(p, q->os);
		}
	}
}

/* treat a (rank <= 1)-tensor as a rank-1 tensor, extracting
appropriate n, is, and os components */
int fftw_tensor_tornk1(const tensor *t, INT *n, INT *is, INT *os) {
	A(t->rnk <= 1);
	if (t->rnk == 1) {
		const iodim *vd = t->dims;
		*n = vd[0].n;
		*is = vd[0].is;
		*os = vd[0].os;
	}
	else {
		*n = 1;
		*is = *os = 0;
	}
	return 1;
}

void fftw_tensor_print(const tensor *x, printer *p) {
	if (FINITE_RNK(x->rnk)) {
		int i;
		int first = 1;
		p->print(p, "(");
		for (i = 0; i < x->rnk; ++i) {
			const iodim *d = x->dims + i;
			p->print(p, "%s(%D %D %D)",
				first ? "" : " ",
				d->n, d->is, d->os);
			first = 0;
		}
		p->print(p, ")");
	}
	else {
		p->print(p, "rank-minfty");
	}
}

tensor *fftw_mktensor_0d(void) {
	return fftw_mktensor(0);
}

tensor *fftw_mktensor_1d(INT n, INT is, INT os) {
	tensor *x = fftw_mktensor(1);
	x->dims[0].n = n;
	x->dims[0].is = is;
	x->dims[0].os = os;
	return x;
}

tensor *fftw_mktensor_2d(INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1) {
	tensor *x = fftw_mktensor(2);
	x->dims[0].n = n0;
	x->dims[0].is = is0;
	x->dims[0].os = os0;
	x->dims[1].n = n1;
	x->dims[1].is = is1;
	x->dims[1].os = os1;
	return x;
}


tensor *fftw_mktensor_3d(INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT n2, INT is2, INT os2) {
	tensor *x = fftw_mktensor(3);
	x->dims[0].n = n0;
	x->dims[0].is = is0;
	x->dims[0].os = os0;
	x->dims[1].n = n1;
	x->dims[1].is = is1;
	x->dims[1].os = os1;
	x->dims[2].n = n2;
	x->dims[2].is = is2;
	x->dims[2].os = os2;
	return x;
}

/* Currently, mktensor_4d and mktensor_5d are only used in the MPI
routines, where very complicated transpositions are required.
Therefore we split them into a separate source file. */

tensor *fftw_mktensor_4d(INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT n2, INT is2, INT os2,
	INT n3, INT is3, INT os3) {
	tensor *x = fftw_mktensor(4);
	x->dims[0].n = n0;
	x->dims[0].is = is0;
	x->dims[0].os = os0;
	x->dims[1].n = n1;
	x->dims[1].is = is1;
	x->dims[1].os = os1;
	x->dims[2].n = n2;
	x->dims[2].is = is2;
	x->dims[2].os = os2;
	x->dims[3].n = n3;
	x->dims[3].is = is3;
	x->dims[3].os = os3;
	return x;
}

tensor *fftw_mktensor_5d(INT n0, INT is0, INT os0,
	INT n1, INT is1, INT os1,
	INT n2, INT is2, INT os2,
	INT n3, INT is3, INT os3,
	INT n4, INT is4, INT os4) {
	tensor *x = fftw_mktensor(5);
	x->dims[0].n = n0;
	x->dims[0].is = is0;
	x->dims[0].os = os0;
	x->dims[1].n = n1;
	x->dims[1].is = is1;
	x->dims[1].os = os1;
	x->dims[2].n = n2;
	x->dims[2].is = is2;
	x->dims[2].os = os2;
	x->dims[3].n = n3;
	x->dims[3].is = is3;
	x->dims[3].os = os3;
	x->dims[4].n = n4;
	x->dims[4].is = is4;
	x->dims[4].os = os4;
	return x;
}

INT fftw_tensor_max_index(const tensor *sz) {
	int i;
	INT ni = 0, no = 0;

	A(FINITE_RNK(sz->rnk));
	for (i = 0; i < sz->rnk; ++i) {
		const iodim *p = sz->dims + i;
		ni += (p->n - 1) * fftw_iabs(p->is);
		no += (p->n - 1) * fftw_iabs(p->os);
	}
	return fftw_imax(ni, no);
}

#define tensor_min_xstride(sz, xs) {            \
     A(FINITE_RNK((sz)->rnk));                \
     if ((sz)->rnk == 0) return 0;            \
     else {                        \
          int i;                    \
          INT s = fftw_iabs((sz)->dims[0].xs);        \
          for (i = 1; i < (sz)->rnk; ++i)            \
               s = fftw_imin(s, fftw_iabs((sz)->dims[i].xs));    \
          return s;                    \
     }                            \
}

INT fftw_tensor_min_istride(const tensor *sz) tensor_min_xstride(sz, is)

INT fftw_tensor_min_ostride(const tensor *sz) tensor_min_xstride(sz, os)

INT fftw_tensor_min_stride(const tensor *sz) {
	return fftw_imin(fftw_tensor_min_istride(sz), fftw_tensor_min_ostride(sz));
}

int fftw_tensor_inplace_strides(const tensor *sz) {
	int i;
	A(FINITE_RNK(sz->rnk));
	for (i = 0; i < sz->rnk; ++i) {
		const iodim *p = sz->dims + i;
		if (p->is != p->os)
			return 0;
	}
	return 1;
}

int fftw_tensor_inplace_strides2(const tensor *a, const tensor *b) {
	return fftw_tensor_inplace_strides(a) && fftw_tensor_inplace_strides(b);
}

/* return true (1) iff *any* strides of sz decrease when we
tensor_inplace_copy(sz, k). */
static int tensor_strides_decrease(const tensor *sz, inplace_kind k) {
	if (FINITE_RNK(sz->rnk)) {
		int i;
		for (i = 0; i < sz->rnk; ++i)
			if ((sz->dims[i].os - sz->dims[i].is)
				* (k == INPLACE_OS ? (INT)1 : (INT)-1) < 0)
				return 1;
	}
	return 0;
}

/* Return true (1) iff *any* strides of sz decrease when we
tensor_inplace_copy(k) *or* if *all* strides of sz are unchanged
but *any* strides of vecsz decrease.  This is used in indirect.c
to determine whether to use INPLACE_IS or INPLACE_OS.

Note: fftw_tensor_strides_decrease(sz, vecsz, INPLACE_IS)
|| fftw_tensor_strides_decrease(sz, vecsz, INPLACE_OS)
|| fftw_tensor_inplace_strides2(p->sz, p->vecsz)
must always be true. */
int fftw_tensor_strides_decrease(const tensor *sz, const tensor *vecsz,
	inplace_kind k) {
	return (tensor_strides_decrease(sz, k)
		|| (fftw_tensor_inplace_strides(sz)
			&& tensor_strides_decrease(vecsz, k)));
}

static void dimcpy(iodim *dst, const iodim *src, int rnk) {
	int i;
	if (FINITE_RNK(rnk))
		for (i = 0; i < rnk; ++i)
			dst[i] = src[i];
}

tensor *fftw_tensor_copy(const tensor *sz) {
	tensor *x = fftw_mktensor(sz->rnk);
	dimcpy(x->dims, sz->dims, sz->rnk);
	return x;
}

/* like fftw_tensor_copy, but makes strides in-place by
setting os = is if k == INPLACE_IS or is = os if k == INPLACE_OS. */
tensor *fftw_tensor_copy_inplace(const tensor *sz, inplace_kind k) {
	tensor *x = fftw_tensor_copy(sz);
	if (FINITE_RNK(x->rnk)) {
		int i;
		if (k == INPLACE_OS)
			for (i = 0; i < x->rnk; ++i)
				x->dims[i].is = x->dims[i].os;
		else
			for (i = 0; i < x->rnk; ++i)
				x->dims[i].os = x->dims[i].is;
	}
	return x;
}

/* Like fftw_tensor_copy, but copy all of the dimensions *except*
except_dim. */
tensor *fftw_tensor_copy_except(const tensor *sz, int except_dim) {
	tensor *x;

	A(FINITE_RNK(sz->rnk) && sz->rnk >= 1 && except_dim < sz->rnk);
	x = fftw_mktensor(sz->rnk - 1);
	dimcpy(x->dims, sz->dims, except_dim);
	dimcpy(x->dims + except_dim, sz->dims + except_dim + 1,
		x->rnk - except_dim);
	return x;
}

/* Like fftw_tensor_copy, but copy only rnk dimensions starting
with start_dim. */
tensor *fftw_tensor_copy_sub(const tensor *sz, int start_dim, int rnk) {
	tensor *x;

	A(FINITE_RNK(sz->rnk) && start_dim + rnk <= sz->rnk);
	x = fftw_mktensor(rnk);
	dimcpy(x->dims, sz->dims + start_dim, rnk);
	return x;
}

tensor *fftw_tensor_append(const tensor *a, const tensor *b) {
	if (!FINITE_RNK(a->rnk) || !FINITE_RNK(b->rnk)) {
		return fftw_mktensor(RNK_MINFTY);
	}
	else {
		tensor *x = fftw_mktensor(a->rnk + b->rnk);
		dimcpy(x->dims, a->dims, a->rnk);
		dimcpy(x->dims + a->rnk, b->dims, b->rnk);
		return x;
	}
}

static int signof(INT x) {
	if (x < 0) return -1;
	if (x == 0) return 0;
	/* if (x > 0) */ return 1;
}

/* total order among iodim's */
int fftw_dimcmp(const iodim *a, const iodim *b) {
	INT sai = fftw_iabs(a->is), sbi = fftw_iabs(b->is);
	INT sao = fftw_iabs(a->os), sbo = fftw_iabs(b->os);
	INT sam = fftw_imin(sai, sao), sbm = fftw_imin(sbi, sbo);

	/* in descending order of min{istride, ostride} */
	if (sam != sbm)
		return signof(sbm - sam);

	/* in case of a tie, in descending order of istride */
	if (sbi != sai)
		return signof(sbi - sai);

	/* in case of a tie, in descending order of ostride */
	if (sbo != sao)
		return signof(sbo - sao);

	/* in case of a tie, in ascending order of n */
	return signof(a->n - b->n);
}

static void canonicalize(tensor *x) {
	if (x->rnk > 1) {
		qsort(x->dims, (unsigned)x->rnk, sizeof(iodim),
			(int(*)(const void *, const void *)) fftw_dimcmp);
	}
}

static int compare_by_istride(const iodim *a, const iodim *b) {
	INT sai = fftw_iabs(a->is), sbi = fftw_iabs(b->is);

	/* in descending order of istride */
	return signof(sbi - sai);
}

static tensor *really_compress(const tensor *sz) {
	int i, rnk;
	tensor *x;

	A(FINITE_RNK(sz->rnk));
	for (i = rnk = 0; i < sz->rnk; ++i) {
		A(sz->dims[i].n > 0);
		if (sz->dims[i].n != 1)
			++rnk;
	}

	x = fftw_mktensor(rnk);
	for (i = rnk = 0; i < sz->rnk; ++i) {
		if (sz->dims[i].n != 1)
			x->dims[rnk++] = sz->dims[i];
	}
	return x;
}

/* Like tensor_copy, but eliminate n == 1 dimensions, which
never affect any transform or transform vector.

Also, we sort the tensor into a canonical order of decreasing
strides (see fftw_dimcmp for an exact definition).  In general,
processing a loop/array in order of decreasing stride will improve
locality.  Both forward and backwards traversal of the tensor are
considered e.g. by vrank-geq1, so sorting in increasing
vs. decreasing order is not really important. */
tensor *fftw_tensor_compress(const tensor *sz) {
	tensor *x = really_compress(sz);
	canonicalize(x);
	return x;
}

/* Return whether the strides of a and b are such that they form an
effective contiguous 1d array.  Assumes that a.is >= b.is. */
static int strides_contig(iodim *a, iodim *b) {
	return (a->is == b->is * b->n && a->os == b->os * b->n);
}

/* Like tensor_compress, but also compress into one dimension any
group of dimensions that form a contiguous block of indices with
some stride.  (This can safely be done for transform vector sizes.) */
tensor *fftw_tensor_compress_contiguous(const tensor *sz) {
	int i, rnk;
	tensor *sz2, *x;

	if (fftw_tensor_sz(sz) == 0)
		return fftw_mktensor(RNK_MINFTY);

	sz2 = really_compress(sz);
	A(FINITE_RNK(sz2->rnk));

	if (sz2->rnk <= 1) { /* nothing to compress. */
		if (0) {
			/* this call is redundant, because "sz->rnk <= 1" implies
			that the tensor is already canonical, but I am writing
			it explicitly because "logically" we need to canonicalize
			the tensor before returning. */
			canonicalize(sz2);
		}
		return sz2;
	}

	/* sort in descending order of |istride|, so that compressible
	dimensions appear contigously */
	qsort(sz2->dims, (unsigned)sz2->rnk, sizeof(iodim),
		(int(*)(const void *, const void *)) compare_by_istride);

	/* compute what the rank will be after compression */
	for (i = rnk = 1; i < sz2->rnk; ++i)
		if (!strides_contig(sz2->dims + i - 1, sz2->dims + i))
			++rnk;

	/* merge adjacent dimensions whenever possible */
	x = fftw_mktensor(rnk);
	x->dims[0] = sz2->dims[0];
	for (i = rnk = 1; i < sz2->rnk; ++i) {
		if (strides_contig(sz2->dims + i - 1, sz2->dims + i)) {
			x->dims[rnk - 1].n *= sz2->dims[i].n;
			x->dims[rnk - 1].is = sz2->dims[i].is;
			x->dims[rnk - 1].os = sz2->dims[i].os;
		}
		else {
			A(rnk < x->rnk);
			x->dims[rnk++] = sz2->dims[i];
		}
	}

	fftw_tensor_destroy(sz2);

	/* reduce to canonical form */
	canonicalize(x);
	return x;
}

/* The inverse of fftw_tensor_append: splits the sz tensor into
tensor a followed by tensor b, where a's rank is arnk. */
void fftw_tensor_split(const tensor *sz, tensor **a, int arnk, tensor **b) {
	A(FINITE_RNK(sz->rnk) && FINITE_RNK(arnk));

	*a = fftw_tensor_copy_sub(sz, 0, arnk);
	*b = fftw_tensor_copy_sub(sz, arnk, sz->rnk - arnk);
}

/* TRUE if the two tensors are equal */
int fftw_tensor_equal(const tensor *a, const tensor *b) {
	if (a->rnk != b->rnk)
		return 0;

	if (FINITE_RNK(a->rnk)) {
		int i;
		for (i = 0; i < a->rnk; ++i)
			if (0
				|| a->dims[i].n != b->dims[i].n
				|| a->dims[i].is != b->dims[i].is
				|| a->dims[i].os != b->dims[i].os
				)
				return 0;
	}

	return 1;
}

/* TRUE if the sets of input and output locations described by
(append sz vecsz) are the same */
int fftw_tensor_inplace_locations(const tensor *sz, const tensor *vecsz) {
	tensor *t = fftw_tensor_append(sz, vecsz);
	tensor *ti = fftw_tensor_copy_inplace(t, INPLACE_IS);
	tensor *to = fftw_tensor_copy_inplace(t, INPLACE_OS);
	tensor *tic = fftw_tensor_compress_contiguous(ti);
	tensor *toc = fftw_tensor_compress_contiguous(to);

	int retval = fftw_tensor_equal(tic, toc);

	fftw_tensor_destroy(t);
	fftw_tensor_destroy4(ti, to, tic, toc);

	return retval;
}

void fftw_tensor_destroy2(tensor *a, tensor *b) {
	fftw_tensor_destroy(a);
	fftw_tensor_destroy(b);
}

void fftw_tensor_destroy4(tensor *a, tensor *b, tensor *c, tensor *d) {
	fftw_tensor_destroy2(a, b);
	fftw_tensor_destroy2(c, d);
}

int fftw_tensor_kosherp(const tensor *x) {
	int i;

	if (x->rnk < 0) return 0;

	if (FINITE_RNK(x->rnk)) {
		for (i = 0; i < x->rnk; ++i)
			if (x->dims[i].n < 0)
				return 0;
	}
	return 1;
}

/* out of place 2D copy routines */


void fftw_tile2d(INT n0l, INT n0u, INT n1l, INT n1u, INT tilesz,
	void(*f)(INT n0l, INT n0u, INT n1l, INT n1u, void *args),
	void *args) {
	INT d0, d1;

	A(tilesz > 0); /* infinite loops otherwise */

tail:
	d0 = n0u - n0l;
	d1 = n1u - n1l;

	if (d0 >= d1 && d0 > tilesz) {
		INT n0m = (n0u + n0l) / 2;
		fftw_tile2d(n0l, n0m, n1l, n1u, tilesz, f, args);
		n0l = n0m;
		goto tail;
	}
	else if (/* d1 >= d0 && */ d1 > tilesz) {
		INT n1m = (n1u + n1l) / 2;
		fftw_tile2d(n0l, n0u, n1l, n1m, tilesz, f, args);
		n1l = n1m;
		goto tail;
	}
	else {
		f(n0l, n0u, n1l, n1u, args);
	}
}

INT fftw_compute_tilesz(INT vl, int how_many_tiles_in_cache) {
	return fftw_isqrt(CACHESIZE /
		(((INT) sizeof(FFTW_REAL_TYPE)) * vl * (INT)how_many_tiles_in_cache));
}

#ifdef HAVE_UNISTD_H

#  include <unistd.h>

#endif

#ifndef WITH_SLOW_TIMER

/* FFTW internal header file */
#ifndef __IFFTW_H__
#define __IFFTW_H__

#include "config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>		/* size_t */
#include <stdarg.h>		/* va_list */
#include <stddef.h>             /* ptrdiff_t */
#include <limits.h>             /* INT_MAX */

#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

#if HAVE_STDINT_H
# include <stdint.h>             /* uintptr_t, maybe */
#endif

#if HAVE_INTTYPES_H
# include <inttypes.h>           /* uintptr_t, maybe */
#endif

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

	/* Windows annoyances -- since fftw/tests/hook.c uses some internal
	FFTW functions, we need to given them the dllexport attribute
	under Windows when compiling as a DLL (see fftw/fftw3.h). */
#if defined(FFTW_EXTERN)
#  define IFFTW_EXTERN FFTW_EXTERN
#elif (defined(FFTW_DLL) || defined(DLL_EXPORT)) \
 && (defined(_WIN32) || defined(__WIN32__))
#  define IFFTW_EXTERN extern __declspec(dllexport)
#else
#  define IFFTW_EXTERN extern
#endif
	/*
	integral type large enough to contain a stride (what ``int'' should
	have been in the first place.
	*/
	typedef ptrdiff_t INT;

	/* dummy use of unused parameters to silence compiler warnings */
#define UNUSED(x) (void)x

#define NELEM(array) ((sizeof(array) / sizeof((array)[0])))

#define FFT_SIGN (-1)  /* sign convention for forward transforms */
	extern void fftw_extract_reim(int sign, FFTW_REAL_TYPE *c, FFTW_REAL_TYPE **r, FFTW_REAL_TYPE **i);

#define REGISTER_SOLVER(p, s) fftw_solver_register(p, s)

#define STRINGIZEx(x) #x
#define STRINGIZE(x) STRINGIZEx(x)
#define CIMPLIES(ante, post) (!(ante) || (post))

	/* define HAVE_SIMD if any rdft_simd extensions are supported */
#if defined(HAVE_SSE) || defined(HAVE_SSE2) || \
      defined(HAVE_AVX) || defined(HAVE_AVX_128_FMA) || \
      defined(HAVE_AVX2) || defined(HAVE_AVX512) || \
      defined(HAVE_KCVI) || \
      defined(HAVE_ALTIVEC) || defined(HAVE_VSX) || \
      defined(HAVE_MIPS_PS) || \
      defined(HAVE_GENERIC_SIMD128) || defined(HAVE_GENERIC_SIMD256)
#define HAVE_SIMD 1
#else
#define HAVE_SIMD 0
#endif

	extern int fftw_have_simd_sse2(void);
	extern int fftw_have_simd_avx(void);
	extern int fftw_have_simd_avx_128_fma(void);
	extern int fftw_have_simd_avx2(void);
	extern int fftw_have_simd_avx2_128(void);
	extern int fftw_have_simd_avx512(void);
	extern int fftw_have_simd_altivec(void);
	extern int fftw_have_simd_vsx(void);
	extern int fftw_have_simd_neon(void);

	/* forward declarations */
	typedef struct problem_s problem;
	typedef struct plan_s plan;
	typedef struct solver_s solver;
	typedef struct planner_s planner;
	typedef struct printer_s printer;
	typedef struct scanner_s scanner;

	/*-----------------------------------------------------------------------*/
	/* alloca: */
#if HAVE_SIMD
#  if defined(HAVE_KCVI) || defined(HAVE_AVX512)
#    define MIN_ALIGNMENT 64
#  elif defined(HAVE_AVX) || defined(HAVE_AVX2) || defined(HAVE_GENERIC_SIMD256)
#    define MIN_ALIGNMENT 32  /* best alignment for AVX, conservative for
	* everything else */
#  else
	/* Note that we cannot use 32-byte alignment for all SIMD.  For
	example, MacOS X malloc is 16-byte aligned, but there was no
	posix_memalign in MacOS X until version 10.6. */
#    define MIN_ALIGNMENT 16
#  endif
#endif

#if defined(HAVE_ALLOCA) && defined(FFTW_ENABLE_ALLOCA)
	/* use alloca if available */

#ifndef alloca
#ifdef __GNUC__
# define alloca __builtin_alloca
#else
# ifdef _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# else
#  if HAVE_ALLOCA_H
#   include <alloca.h>
#  else
#   ifdef _AIX
#pragma alloca
#   else
#    ifndef alloca /* predefined by HP cc +Olibcalls */
	void *alloca(size_t);
#    endif
#   endif
#  endif
# endif
#endif
#endif

#  ifdef MIN_ALIGNMENT
#    define STACK_MALLOC(T, p, n)				\
     {								\
         p = (T)alloca((n) + MIN_ALIGNMENT);			\
         p = (T)(((uintptr_t)p + (MIN_ALIGNMENT - 1)) &	\
               (~(uintptr_t)(MIN_ALIGNMENT - 1)));		\
     }
#    define STACK_FREE(n)
#  else /* HAVE_ALLOCA && !defined(MIN_ALIGNMENT) */
#    define STACK_MALLOC(T, p, n) p = (T)alloca(n)
#    define STACK_FREE(n)
#  endif

#else /* ! HAVE_ALLOCA */
	/* use malloc instead of alloca */
#  define STACK_MALLOC(T, p, n) p = (T)MALLOC(n, OTHER)
#  define STACK_FREE(n) fftw_ifree(n)
#endif /* ! HAVE_ALLOCA */

	/* allocation of buffers.  If these grow too large use malloc(), else
	use STACK_MALLOC (hopefully reducing to alloca()). */

	/* 64KiB ought to be enough for anybody */
#define MAX_STACK_ALLOC ((size_t)64 * 1024)

#define BUF_ALLOC(T, p, n)			\
{						\
     if (n < MAX_STACK_ALLOC) {			\
      STACK_MALLOC(T, p, n);		\
     } else {					\
      p = (T)MALLOC(n, BUFFERS);		\
     }						\
}

#define BUF_FREE(p, n)				\
{						\
     if (n < MAX_STACK_ALLOC) {			\
      STACK_FREE(p);			\
     } else {					\
      fftw_ifree(p);				\
     }						\
}

	/*-----------------------------------------------------------------------*/
	/* define uintptr_t if it is not already defined */

#ifndef HAVE_UINTPTR_T
#  if SIZEOF_VOID_P == 0
#    error sizeof void* is unknown!
#  elif SIZEOF_UNSIGNED_INT == SIZEOF_VOID_P
	typedef unsigned int uintptr_t;
#  elif SIZEOF_UNSIGNED_LONG == SIZEOF_VOID_P
	typedef unsigned long uintptr_t;
#  elif SIZEOF_UNSIGNED_LONG_LONG == SIZEOF_VOID_P
	typedef unsigned long long uintptr_t;
#  else
#    error no unsigned integer type matches void* sizeof!
#  endif
#endif

	/*-----------------------------------------------------------------------*/
	/* We can do an optimization for copying pairs of (aligned) floats
	when in single precision if 2*float = double. */

#define FFTW_2R_IS_DOUBLE (defined(FFTW_SINGLE) \
                           && SIZEOF_FLOAT != 0 \
                           && SIZEOF_DOUBLE == 2*SIZEOF_FLOAT)

#define DOUBLE_ALIGNED(p) ((((uintptr_t)(p)) % sizeof(double)) == 0)

	/*-----------------------------------------------------------------------*/
	/* assert.c: */
	IFFTW_EXTERN void fftw_assertion_failed(const char *s,
		int line, const char *file);

	/* always check */
#define CK(ex)						 \
      (void)((ex) || (fftw_assertion_failed(#ex, __LINE__, __FILE__), 0))

#ifdef FFTW_DEBUG
	/* check only if debug enabled */
#define A(ex)						 \
      (void)((ex) || (fftw_assertion_failed(#ex, __LINE__, __FILE__), 0))
#else
#define A(ex) /* nothing */
#endif

	extern void fftw_debug(const char *format, ...);
#define D fftw_debug

	/*-----------------------------------------------------------------------*/
	/* kalloc.c: */
	extern void *fftw_kernel_malloc(size_t n);
	extern void fftw_kernel_free(void *p);

	/*-----------------------------------------------------------------------*/
	/* alloc.c: */

	/* objects allocated by malloc, for statistical purposes */
	enum malloc_tag {
		EVERYTHING,
		PLANS,
		SOLVERS,
		PROBLEMS,
		BUFFERS,
		HASHT,
		TENSORS,
		PLANNERS,
		SLVDESCS,
		TWIDDLES,
		STRIDES,
		OTHER,
		MALLOC_WHAT_LAST		/* must be last */
	};

	IFFTW_EXTERN void fftw_ifree(void *ptr);
	extern void fftw_ifree0(void *ptr);

	IFFTW_EXTERN void *fftw_malloc_plain(size_t sz);
#define MALLOC(n, what)  fftw_malloc_plain(n)

	/*-----------------------------------------------------------------------*/
	/* low-resolution clock */

#ifdef FAKE_CRUDE_TIME
	typedef int crude_time;
#else
# if TIME_WITH_SYS_TIME
#  include <sys/time.h>
#  include <time.h>
# else
#  if HAVE_SYS_TIME_H
#   include <sys/time.h>
#  else
#   include <time.h>
#  endif
# endif

# ifdef HAVE_BSDGETTIMEOFDAY
# ifndef HAVE_GETTIMEOFDAY
# define gettimeofday BSDgettimeofday
# define HAVE_GETTIMEOFDAY 1
# endif
# endif

# if defined(HAVE_GETTIMEOFDAY)
	typedef struct timeval crude_time;
# else
	typedef clock_t crude_time;
# endif
#endif /* else FAKE_CRUDE_TIME */

	crude_time fftw_get_crude_time(void);
	double fftw_elapsed_since(const planner *plnr, const problem *p,
		crude_time t0); /* time in seconds since t0 */

						/*-----------------------------------------------------------------------*/
						/* ops.c: */
						/*
						* ops counter.  The total number of additions is add + fma
						* and the total number of multiplications is mul + fma.
						* Total flops = add + mul + 2 * fma
						*/
	typedef struct {
		double add;
		double mul;
		double fma;
		double other;
	} opcnt;

	void fftw_ops_zero(opcnt *dst);
	void fftw_ops_other(INT o, opcnt *dst);
	void fftw_ops_cpy(const opcnt *fftw, opcnt *dst);

	void fftw_ops_add(const opcnt *a, const opcnt *b, opcnt *dst);
	void fftw_ops_add2(const opcnt *a, opcnt *dst);

	/* dst = m * a + b */
	void fftw_ops_madd(INT m, const opcnt *a, const opcnt *b, opcnt *dst);

	/* dst += m * a */
	void fftw_ops_madd2(INT m, const opcnt *a, opcnt *dst);


	/*-----------------------------------------------------------------------*/
	/* minmax.c: */
	INT fftw_imax(INT a, INT b);
	INT fftw_imin(INT a, INT b);

	/*-----------------------------------------------------------------------*/
	/* iabs.c: */
	INT fftw_iabs(INT a);

	/* inline version */
#define IABS(x) (((x) < 0) ? (0 - (x)) : (x))

	/*-----------------------------------------------------------------------*/
	/* md5.c */

#if SIZEOF_UNSIGNED_INT >= 4
	typedef unsigned int md5uint;
#else
	typedef unsigned long md5uint; /* at least 32 bits as per C standard */
#endif

	typedef md5uint md5sig[4];

	typedef struct {
		md5sig s; /* state and signature */

				  /* fields not meant to be used outside md5.c: */
		unsigned char c[64]; /* stuff not yet processed */
		unsigned l;  /* total length.  Should be 64 bits long, but this is
					 good enough for us */
	} md5;

	void fftw_md5begin(md5 *p);
	void fftw_md5putb(md5 *p, const void *d_, size_t len);
	void fftw_md5puts(md5 *p, const char *s);
	void fftw_md5putc(md5 *p, unsigned char c);
	void fftw_md5int(md5 *p, int i);
	void fftw_md5INT(md5 *p, INT i);
	void fftw_md5unsigned(md5 *p, unsigned i);
	void fftw_md5end(md5 *p);

	/*-----------------------------------------------------------------------*/
	/* tensor.c: */
#define STRUCT_HACK_KR
#undef STRUCT_HACK_C99

	typedef struct {
		INT n;
		INT is;			/* input stride */
		INT os;			/* output stride */
	} iodim;

	typedef struct {
		int rnk;
#if defined(STRUCT_HACK_KR)
		iodim dims[1];
#elif defined(STRUCT_HACK_C99)
		iodim dims[];
#else
		iodim *dims;
#endif
	} tensor;

	/*
	Definition of rank -infinity.
	This definition has the property that if you want rank 0 or 1,
	you can simply test for rank <= 1.  This is a common case.

	A tensor of rank -infinity has size 0.
	*/
#define RNK_MINFTY  INT_MAX
#define FINITE_RNK(rnk) ((rnk) != RNK_MINFTY)

	typedef enum { INPLACE_IS, INPLACE_OS } inplace_kind;

	tensor *fftw_mktensor(int rnk);
	tensor *fftw_mktensor_0d(void);
	tensor *fftw_mktensor_1d(INT n, INT is, INT os);
	tensor *fftw_mktensor_2d(INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1);
	tensor *fftw_mktensor_3d(INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT n2, INT is2, INT os2);
	tensor *fftw_mktensor_4d(INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT n2, INT is2, INT os2,
		INT n3, INT is3, INT os3);
	tensor *fftw_mktensor_5d(INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT n2, INT is2, INT os2,
		INT n3, INT is3, INT os3,
		INT n4, INT is4, INT os4);
	INT fftw_tensor_sz(const tensor *sz);
	void fftw_tensor_md5(md5 *p, const tensor *t);
	INT fftw_tensor_max_index(const tensor *sz);
	INT fftw_tensor_min_istride(const tensor *sz);
	INT fftw_tensor_min_ostride(const tensor *sz);
	INT fftw_tensor_min_stride(const tensor *sz);
	int fftw_tensor_inplace_strides(const tensor *sz);
	int fftw_tensor_inplace_strides2(const tensor *a, const tensor *b);
	int fftw_tensor_strides_decrease(const tensor *sz, const tensor *vecsz,
		inplace_kind k);
	tensor *fftw_tensor_copy(const tensor *sz);
	int fftw_tensor_kosherp(const tensor *x);

	tensor *fftw_tensor_copy_inplace(const tensor *sz, inplace_kind k);
	tensor *fftw_tensor_copy_except(const tensor *sz, int except_dim);
	tensor *fftw_tensor_copy_sub(const tensor *sz, int start_dim, int rnk);
	tensor *fftw_tensor_compress(const tensor *sz);
	tensor *fftw_tensor_compress_contiguous(const tensor *sz);
	tensor *fftw_tensor_append(const tensor *a, const tensor *b);
	void fftw_tensor_split(const tensor *sz, tensor **a, int a_rnk, tensor **b);
	int fftw_tensor_tornk1(const tensor *t, INT *n, INT *is, INT *os);
	void fftw_tensor_destroy(tensor *sz);
	void fftw_tensor_destroy2(tensor *a, tensor *b);
	void fftw_tensor_destroy4(tensor *a, tensor *b, tensor *c, tensor *d);
	void fftw_tensor_print(const tensor *sz, printer *p);
	int fftw_dimcmp(const iodim *a, const iodim *b);
	int fftw_tensor_equal(const tensor *a, const tensor *b);
	int fftw_tensor_inplace_locations(const tensor *sz, const tensor *vecsz);

	/*-----------------------------------------------------------------------*/
	/* problem.c: */
	enum {
		/* a problem that cannot be solved */
		PROBLEM_UNSOLVABLE,

		PROBLEM_DFT,
		PROBLEM_RDFT,
		PROBLEM_RDFT2,

		/* for mpi/ subdirectory */
		PROBLEM_MPI_DFT,
		PROBLEM_MPI_RDFT,
		PROBLEM_MPI_RDFT2,
		PROBLEM_MPI_TRANSPOSE,

		PROBLEM_LAST
	};

	typedef struct {
		int problem_kind;
		void(*hash) (const problem *ego, md5 *p);
		void(*zero) (const problem *ego);
		void(*print) (const problem *ego, printer *p);
		void(*destroy) (problem *ego);
	} problem_adt;

	struct problem_s {
		const problem_adt *adt;
	};

	problem *fftw_mkproblem(size_t sz, const problem_adt *adt);
	void fftw_problem_destroy(problem *ego);
	problem *fftw_mkproblem_unsolvable(void);

	/*-----------------------------------------------------------------------*/
	/* print.c */
	struct printer_s {
		void(*print)(printer *p, const char *format, ...);
		void(*vprint)(printer *p, const char *format, va_list ap);
		void(*putchr)(printer *p, char c);
		void(*cleanup)(printer *p);
		int indent;
		int indent_incr;
	};

	printer *fftw_mkprinter(size_t size,
		void(*putchr)(printer *p, char c),
		void(*cleanup)(printer *p));
	IFFTW_EXTERN void fftw_printer_destroy(printer *p);

	/*-----------------------------------------------------------------------*/
	/* scan.c */
	struct scanner_s {
		int(*scan)(scanner *sc, const char *format, ...);
		int(*vscan)(scanner *sc, const char *format, va_list ap);
		int(*getchr)(scanner *sc);
		int ungotc;
	};

	scanner *fftw_mkscanner(size_t size, int(*getchr)(scanner *sc));
	void fftw_scanner_destroy(scanner *sc);

	/*-----------------------------------------------------------------------*/
	/* plan.c: */

	enum wakefulness {
		SLEEPY,
		AWAKE_ZERO,
		AWAKE_SQRTN_TABLE,
		AWAKE_SINCOS
	};

	typedef struct {
		void(*solve)(const plan *ego, const problem *p);
		void(*awake)(plan *ego, enum wakefulness wakefulness);
		void(*print)(const plan *ego, printer *p);
		void(*destroy)(plan *ego);
	} plan_adt;

	struct plan_s {
		const plan_adt *adt;
		opcnt ops;
		double pcost;
		enum wakefulness wakefulness; /* used for debugging only */
		int could_prune_now_p;
	};

	plan *fftw_mkplan(size_t size, const plan_adt *adt);
	void fftw_plan_destroy_internal(plan *ego);
	IFFTW_EXTERN void fftw_plan_awake(plan *ego, enum wakefulness wakefulness);
	void fftw_plan_null_destroy(plan *ego);

	/*-----------------------------------------------------------------------*/
	/* solver.c: */
	typedef struct {
		int problem_kind;
		plan *(*ifftw_mkplan)(const solver *ego, const problem *p, planner *plnr);
		void(*destroy)(solver *ego);
	} solver_adt;

	struct solver_s {
		const solver_adt *adt;
		int refcnt;
	};

	solver *fftw_mksolver(size_t size, const solver_adt *adt);
	void fftw_solver_use(solver *ego);
	void fftw_solver_destroy(solver *ego);
	void fftw_solver_register(planner *plnr, solver *s);

	/* shorthand */
#define MKSOLVER(type, adt) (type *)fftw_mksolver(sizeof(type), adt)

	/*-----------------------------------------------------------------------*/
	/* planner.c */

	typedef struct slvdesc_s {
		solver *slv;
		const char *reg_nam;
		unsigned nam_hash;
		int reg_id;
		int next_for_same_problem_kind;
	} slvdesc;

	typedef struct solution_s solution; /* opaque */

										/* interpretation of L and U:

										- if it returns a plan, the planner guarantees that all applicable
										plans at least as impatient as U have been tried, and that each
										plan in the solution is at least as impatient as L.

										- if it returns 0, the planner guarantees to have tried all solvers
										at least as impatient as L, and that none of them was applicable.

										The structure is packed to fit into 64 bits.
										*/

	typedef struct {
		unsigned l : 20;
		unsigned hash_info : 3;
#    define BITS_FOR_TIMELIMIT 9
		unsigned timelimit_impatience : BITS_FOR_TIMELIMIT;
		unsigned u : 20;

		/* abstraction break: we store the solver here to pad the
		structure to 64 bits.  Otherwise, the struct is padded to 64
		bits anyway, and another word is allocated for slvndx. */
#    define BITS_FOR_SLVNDX 12
		unsigned slvndx : BITS_FOR_SLVNDX;
	} flags_t;

	/* impatience flags  */
	enum {
		BELIEVE_PCOST = 0x0001,
		ESTIMATE = 0x0002,
		NO_DFT_R2HC = 0x0004,
		NO_SLOW = 0x0008,
		NO_VRECURSE = 0x0010,
		NO_INDIRECT_OP = 0x0020,
		NO_LARGE_GENERIC = 0x0040,
		NO_RANK_SPLITS = 0x0080,
		NO_VRANK_SPLITS = 0x0100,
		NO_NONTHREADED = 0x0200,
		NO_BUFFERING = 0x0400,
		NO_FIXED_RADIX_LARGE_N = 0x0800,
		NO_DESTROY_INPUT = 0x1000,
		NO_SIMD = 0x2000,
		CONSERVE_MEMORY = 0x4000,
		NO_DHT_R2HC = 0x8000,
		NO_UGLY = 0x10000,
		ALLOW_PRUNING = 0x20000
	};

	/* hashtable information */
	enum {
		BLESSING = 0x1u,   /* save this entry */
		H_VALID = 0x2u,    /* valid hastable entry */
		H_LIVE = 0x4u      /* entry is nonempty, implies H_VALID */
	};

#define PLNR_L(plnr) ((plnr)->flags.l)
#define PLNR_U(plnr) ((plnr)->flags.u)
#define PLNR_TIMELIMIT_IMPATIENCE(plnr) ((plnr)->flags.timelimit_impatience)

#define ESTIMATEP(plnr) (PLNR_U(plnr) & ESTIMATE)
#define BELIEVE_PCOSTP(plnr) (PLNR_U(plnr) & BELIEVE_PCOST)
#define ALLOW_PRUNINGP(plnr) (PLNR_U(plnr) & ALLOW_PRUNING)

#define NO_INDIRECT_OP_P(plnr) (PLNR_L(plnr) & NO_INDIRECT_OP)
#define NO_LARGE_GENERICP(plnr) (PLNR_L(plnr) & NO_LARGE_GENERIC)
#define NO_RANK_SPLITSP(plnr) (PLNR_L(plnr) & NO_RANK_SPLITS)
#define NO_VRANK_SPLITSP(plnr) (PLNR_L(plnr) & NO_VRANK_SPLITS)
#define NO_VRECURSEP(plnr) (PLNR_L(plnr) & NO_VRECURSE)
#define NO_DFT_R2HCP(plnr) (PLNR_L(plnr) & NO_DFT_R2HC)
#define NO_SLOWP(plnr) (PLNR_L(plnr) & NO_SLOW)
#define NO_UGLYP(plnr) (PLNR_L(plnr) & NO_UGLY)
#define NO_FIXED_RADIX_LARGE_NP(plnr) \
  (PLNR_L(plnr) & NO_FIXED_RADIX_LARGE_N)
#define NO_NONTHREADEDP(plnr) \
  ((PLNR_L(plnr) & NO_NONTHREADED) && (plnr)->nthr > 1)

#define NO_DESTROY_INPUTP(plnr) (PLNR_L(plnr) & NO_DESTROY_INPUT)
#define NO_SIMDP(plnr) (PLNR_L(plnr) & NO_SIMD)
#define CONSERVE_MEMORYP(plnr) (PLNR_L(plnr) & CONSERVE_MEMORY)
#define NO_DHT_R2HCP(plnr) (PLNR_L(plnr) & NO_DHT_R2HC)
#define NO_BUFFERINGP(plnr) (PLNR_L(plnr) & NO_BUFFERING)

	typedef enum { FORGET_ACCURSED, FORGET_EVERYTHING } amnesia;

	typedef enum {
		/* WISDOM_NORMAL: planner may or may not use wisdom */
		WISDOM_NORMAL,

		/* WISDOM_ONLY: planner must use wisdom and must avoid searching */
		WISDOM_ONLY,

		/* WISDOM_IS_BOGUS: planner must return 0 as quickly as possible */
		WISDOM_IS_BOGUS,

		/* WISDOM_IGNORE_INFEASIBLE: planner ignores infeasible wisdom */
		WISDOM_IGNORE_INFEASIBLE,

		/* WISDOM_IGNORE_ALL: planner ignores all */
		WISDOM_IGNORE_ALL
	} wisdom_state_t;

	typedef struct {
		void(*ifftw_register_solver)(planner *ego, solver *s);
		plan *(*ifftw_mkplan)(planner *ego, const problem *p);
		void(*ifftw_forget)(planner *ego, amnesia a);
		void(*ifftw_exprt)(planner *ego, printer *p); /* ``export'' is a reserved
													  word in C++. */
		int(*ifftw_imprt)(planner *ego, scanner *sc);
	} ifftw_planner_adt;

	/* hash table of solutions */
	typedef struct {
		solution *solutions;
		unsigned hashsiz, nelem;

		/* statistics */
		int lookup, succ_lookup, lookup_iter;
		int insert, insert_iter, insert_unknown;
		int nrehash;
	} hashtab;

	typedef enum { COST_SUM, COST_MAX } cost_kind;

	struct planner_s {
		const ifftw_planner_adt *adt;
		void(*hook)(struct planner_s *plnr, plan *pln,
			const problem *p, int optimalp);
		double(*cost_hook)(const problem *p, double t, cost_kind k);
		int(*wisdom_ok_hook)(const problem *p, flags_t flags);
		void(*nowisdom_hook)(const problem *p);
		wisdom_state_t(*bogosity_hook)(wisdom_state_t state, const problem *p);

		/* solver descriptors */
		slvdesc *slvdescs;
		unsigned nslvdesc, slvdescsiz;
		const char *cur_reg_nam;
		int cur_reg_id;
		int slvdescs_for_problem_kind[PROBLEM_LAST];

		wisdom_state_t wisdom_state;

		hashtab htab_blessed;
		hashtab htab_unblessed;

		int nthr;
		flags_t flags;

		crude_time start_time;
		double timelimit; /* elapsed_since(start_time) at which to bail out */
		int timed_out; /* whether most recent search timed out */
		int need_timeout_check;

		/* various statistics */
		int nplan;    /* number of plans evaluated */
		double pcost, epcost; /* total pcost of measured/estimated plans */
		int nprob;    /* number of problems evaluated */
	};

	planner *fftw_mkplanner(void);
	void fftw_planner_destroy(planner *ego);

	/*
	Iterate over all solvers.   Read:

	@article{ baker93iterators,
	author = "Henry G. Baker, Jr.",
	title = "Iterators: Signs of Weakness in Object-Oriented Languages",
	journal = "{ACM} {OOPS} Messenger",
	volume = "4",
	number = "3",
	pages = "18--25"
	}
	*/
#define FORALL_SOLVERS(ego, s, p, what)			\
{							\
     unsigned _cnt;					\
     for (_cnt = 0; _cnt < ego->nslvdesc; ++_cnt) {	\
      slvdesc *p = ego->slvdescs + _cnt;		\
      solver *s = p->slv;				\
      what;						\
     }							\
}

#define FORALL_SOLVERS_OF_KIND(kind, ego, s, p, what)		\
{								\
     int _cnt = ego->slvdescs_for_problem_kind[kind]; 		\
     while (_cnt >= 0) {					\
      slvdesc *p = ego->slvdescs + _cnt;			\
      solver *s = p->slv;					\
      what;							\
      _cnt = p->next_for_same_problem_kind;			\
     }								\
}


	/* make plan, destroy problem */
	plan *fftw_mkplan_d(planner *ego, problem *p);
	plan *fftw_mkplan_f_d(planner *ego, problem *p,
		unsigned l_set, unsigned u_set, unsigned u_reset);

	/*-----------------------------------------------------------------------*/
	/* stride.c: */

	/* If PRECOMPUTE_ARRAY_INDICES is defined, precompute all strides. */
#if (defined(__i386__) || defined(__x86_64__) || _M_IX86 >= 500) && !defined(FFTW_LDOUBLE)
#define PRECOMPUTE_ARRAY_INDICES
#endif

	extern const INT fftw_an_INT_guaranteed_to_be_zero;

#ifdef PRECOMPUTE_ARRAY_INDICES
	typedef INT *stride;
#define WS(stride, i)  (stride[i])
	extern stride fftw_mkstride(INT n, INT s);
	void fftw_stride_destroy(stride p);
	/* hackery to prevent the compiler from copying the strides array
	onto the stack */
#define MAKE_VOLATILE_STRIDE(nptr, x) (x) = (x) + fftw_an_INT_guaranteed_to_be_zero
#else

	typedef INT stride;
#define WS(stride, i)  (stride * i)
#define fftwf_mkstride(n, stride) stride
#define fftw_mkstride(n, stride) stride
#define fftwl_mkstride(n, stride) stride
#define fftwf_stride_destroy(p) ((void) p)
#define fftw_stride_destroy(p) ((void) p)
#define fftwl_stride_destroy(p) ((void) p)

	/* hackery to prevent the compiler from ``optimizing'' induction
	variables in codelet loops.  The problem is that for each K and for
	each expression of the form P[I + STRIDE * K] in a loop, most
	compilers will try to lift an induction variable PK := &P[I + STRIDE * K].
	For large values of K this behavior overflows the
	register set, which is likely worse than doing the index computation
	in the first place.

	If we guess that there are more than
	ESTIMATED_AVAILABLE_INDEX_REGISTERS such pointers, we deliberately confuse
	the compiler by setting STRIDE ^= ZERO, where ZERO is a value guaranteed to
	be 0, but the compiler does not know this.

	16 registers ought to be enough for anybody, or so the amd64 and ARM ISA's
	seem to imply.
	*/
#define ESTIMATED_AVAILABLE_INDEX_REGISTERS 16
#define MAKE_VOLATILE_STRIDE(nptr, x)                   \
     (nptr <= ESTIMATED_AVAILABLE_INDEX_REGISTERS ?     \
        0 :                                             \
      ((x) = (x) ^ fftw_an_INT_guaranteed_to_be_zero))
#endif /* PRECOMPUTE_ARRAY_INDICES */

	/*-----------------------------------------------------------------------*/
	/* solvtab.c */

	struct solvtab_s { void(*reg)(planner *); const char *reg_nam; };
	typedef struct solvtab_s solvtab[];
	void fftw_solvtab_exec(const solvtab tbl, planner *p);
#define SOLVTAB(s) { s, STRINGIZE(s) }
#define SOLVTAB_END { 0, 0 }

	/*-----------------------------------------------------------------------*/
	/* pickdim.c */
	int fftw_pickdim(int which_dim, const int *buddies, size_t nbuddies,
		const tensor *sz, int oop, int *dp);

	/*-----------------------------------------------------------------------*/
	/* twiddle.c */
	/* little language to express twiddle factors computation */
	enum {
		TW_COS = 0, TW_SIN = 1, TW_CEXP = 2, TW_NEXT = 3,
		TW_FULL = 4, TW_HALF = 5
	};

	typedef struct {
		unsigned char op;
		signed char v;
		short i;
	} tw_instr;

	typedef struct twid_s {
		FFTW_REAL_TYPE *W;                     /* array of twiddle factors */
		INT n, r, m;                /* transform order, radix, # twiddle rows */
		int refcnt;
		const tw_instr *instr;
		struct twid_s *cdr;
		enum wakefulness wakefulness;
	} twid;

	INT fftw_twiddle_length(INT r, const tw_instr *p);
	void fftw_twiddle_awake(enum wakefulness wakefulness,
		twid **pp, const tw_instr *instr, INT n, INT r, INT m);

	/*-----------------------------------------------------------------------*/
	/* trig.c */
#if defined(TRIGREAL_IS_LONG_DOUBLE)
	typedef long double trigreal;
#elif defined(TRIGREAL_IS_QUAD)
	typedef __float128 trigreal;
#else
	typedef double trigreal;
#endif

	typedef struct triggen_s triggen;

	struct triggen_s {
		void(*cexp)(triggen *t, INT m, FFTW_REAL_TYPE *result);
		void(*cexpl)(triggen *t, INT m, trigreal *result);
		void(*rotate)(triggen *p, INT m, FFTW_REAL_TYPE xr, FFTW_REAL_TYPE xi, FFTW_REAL_TYPE *res);

		INT twshft;
		INT twradix;
		INT twmsk;
		trigreal *W0, *W1;
		INT n;
	};

	triggen *fftw_mktriggen(enum wakefulness wakefulness, INT n);
	void fftw_triggen_destroy(triggen *p);

	/*-----------------------------------------------------------------------*/
	/* primes.c: */

#define MULMOD(x, y, p) \
   (((x) <= 92681 - (y)) ? ((x) * (y)) % (p) : fftw_safe_mulmod(x, y, p))

	INT fftw_safe_mulmod(INT x, INT y, INT p);
	INT fftw_power_mod(INT n, INT m, INT p);
	INT fftw_find_generator(INT p);
	INT fftw_first_divisor(INT n);
	int fftw_is_prime(INT n);
	INT fftw_next_prime(INT n);
	int fftw_factors_into(INT n, const INT *primes);
	int fftw_factors_into_small_primes(INT n);
	INT fftw_choose_radix(INT r, INT n);
	INT fftw_isqrt(INT n);
	INT fftw_modulo(INT a, INT n);

#define GENERIC_MIN_BAD 173 /* min prime for which generic becomes bad */

	/* thresholds below which certain solvers are considered SLOW.  These are guesses
	believed to be conservative */
#define GENERIC_MAX_SLOW     16
#define RADER_MAX_SLOW       32
#define BLUESTEIN_MAX_SLOW   24

	/*-----------------------------------------------------------------------*/
	/* rader.c: */
	typedef struct rader_tls rader_tl;

	void fftw_rader_tl_insert(INT k1, INT k2, INT k3, FFTW_REAL_TYPE *W, rader_tl **tl);
	FFTW_REAL_TYPE *fftw_rader_tl_find(INT k1, INT k2, INT k3, rader_tl *t);
	void fftw_rader_tl_delete(FFTW_REAL_TYPE *W, rader_tl **tl);

	/*-----------------------------------------------------------------------*/
	/* copy/transposition routines */

	/* lower bound to the cache size, for tiled routines */
#define CACHESIZE 8192

	INT fftw_compute_tilesz(INT vl, int how_many_tiles_in_cache);

	void fftw_tile2d(INT n0l, INT n0u, INT n1l, INT n1u, INT tilesz,
		void(*f)(INT n0l, INT n0u, INT n1l, INT n1u, void *args),
		void *args);
	void fftw_cpy1d(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, INT n0, INT is0, INT os0, INT vl);
	void fftw_zero1d_pair(FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1, INT n0, INT os0);
	void fftw_cpy2d(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);
	void fftw_cpy2d_ci(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);
	void fftw_cpy2d_co(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);
	void fftw_cpy2d_tiled(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);
	void fftw_cpy2d_tiledbuf(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);
	void fftw_cpy2d_pair(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1);
	void fftw_cpy2d_pair_ci(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1);
	void fftw_cpy2d_pair_co(FFTW_REAL_TYPE *I0, FFTW_REAL_TYPE *I1, FFTW_REAL_TYPE *O0, FFTW_REAL_TYPE *O1,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1);

	void fftw_transpose(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl);
	void fftw_transpose_tiled(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl);
	void fftw_transpose_tiledbuf(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl);

	typedef void(*transpose_func)(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl);
	typedef void(*cpy2d_func)(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
		INT n0, INT is0, INT os0,
		INT n1, INT is1, INT os1,
		INT vl);

	/*-----------------------------------------------------------------------*/
	/* misc stuff */
	void fftw_null_awake(plan *ego, enum wakefulness wakefulness);
	double fftw_iestimate_cost(const planner *, const plan *, const problem *);

#ifdef FFTW_RANDOM_ESTIMATOR
	extern unsigned fftw_random_estimate_seed;
#endif

	double fftw_measure_execution_time(const planner *plnr,
		plan *pln, const problem *p);
	IFFTW_EXTERN int fftw_ialignment_of(FFTW_REAL_TYPE *p);
	unsigned fftw_hash(const char *s);
	INT fftw_nbuf(INT n, INT vl, INT maxnbuf);
	int fftw_nbuf_redundant(INT n, INT vl, size_t which,
		const INT *maxnbuf, size_t nmaxnbuf);
	INT fftw_bufdist(INT n, INT vl);
	int fftw_toobig(INT n);
	int fftw_ct_uglyp(INT min_n, INT v, INT n, INT r);

#if HAVE_SIMD
	FFTW_REAL_TYPE *X(taint)(FFTW_REAL_TYPE *p, INT s);
	FFTW_REAL_TYPE *X(join_taint)(FFTW_REAL_TYPE *p1, FFTW_REAL_TYPE *p2);
#define TAINT(p, s) fftw_taint(p, s)
#define UNTAINT(p) ((FFTW_REAL_TYPE *) (((uintptr_t) (p)) & ~(uintptr_t)3))
#define TAINTOF(p) (((uintptr_t)(p)) & 3)
#define JOIN_TAINT(p1, p2) fftw_join_taint(p1, p2)
#else
#define TAINT(p, s) (p)
#define UNTAINT(p) (p)
#define TAINTOF(p) 0
#define JOIN_TAINT(p1, p2) p1
#endif

#define ASSERT_ALIGNED_DOUBLE  /*unused, legacy*/

	/*-----------------------------------------------------------------------*/
	/* macros used in codelets to reduce source code size */

	typedef FFTW_REAL_TYPE E;  /* internal precision of codelets. */

#if defined(FFTW_LDOUBLE)
#  define K(x) ((E) x##L)
#elif defined(FFTW_QUAD)
#  define K(x) ((E) x##Q)
#else
#  define K(x) ((E) x)
#endif
#define DK(name, value) const E name = K(value)

							   /* FMA macros */

#if defined(__GNUC__) && (defined(__powerpc__) || defined(__ppc__) || defined(_POWER))
							   /* The obvious expression a * b + c does not work.  If both x = a * b
							   + c and y = a * b - c appear in the source, gcc computes t = a * b,
							   x = t + c, y = t - c, thus destroying the fma.

							   This peculiar coding seems to do the right thing on all of
							   gcc-2.95, gcc-3.1, gcc-3.2, and gcc-3.3.  It does the right thing
							   on gcc-3.4 -fno-web (because the ``web'' pass splits the variable
							   `x' for the single-assignment form).

							   However, gcc-4.0 is a formidable adversary which succeeds in
							   pessimizing two fma's into one multiplication and two additions.
							   It does it very early in the game---before the optimization passes
							   even start.  The only real workaround seems to use fake inline asm
							   such as

							   asm ("# confuse gcc %0" : "=f"(a) : "0"(a));
							   return a * b + c;

							   in each of the FMA, FMS, FNMA, and FNMS functions.  However, this
							   does not solve the problem either, because two equal asm statements
							   count as a common subexpression!  One must use *different* fake asm
							   statements:

							   in FMA:
							   asm ("# confuse gcc for fma %0" : "=f"(a) : "0"(a));

							   in FMS:
							   asm ("# confuse gcc for fms %0" : "=f"(a) : "0"(a));

							   etc.

							   After these changes, gcc recalcitrantly generates the fma that was
							   in the source to begin with.  However, the extra asm() cruft
							   confuses other passes of gcc, notably the instruction scheduler.
							   (Of course, one could also generate the fma directly via inline
							   asm, but this confuses the scheduler even more.)

							   Steven and I have submitted more than one bug report to the gcc
							   mailing list over the past few years, to no effect.  Thus, I give
							   up.  gcc-4.0 can go to hell.  I'll wait at least until gcc-4.3 is
							   out before touching this crap again.
							   */
	static __inline__ E FMA(E a, E b, E c)
	{
		E x = a * b;
		x = x + c;
		return x;
	}

	static __inline__ E FMS(E a, E b, E c)
	{
		E x = a * b;
		x = x - c;
		return x;
	}

	static __inline__ E FNMA(E a, E b, E c)
	{
		E x = a * b;
		x = -(x + c);
		return x;
	}

	static __inline__ E FNMS(E a, E b, E c)
	{
		E x = a * b;
		x = -(x - c);
		return x;
	}
#else
#define FMA(a, b, c) (((a) * (b)) + (c))
#define FMS(a, b, c) (((a) * (b)) - (c))
#define FNMA(a, b, c) (- (((a) * (b)) + (c)))
#define FNMS(a, b, c) ((c) - ((a) * (b)))
#endif

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* __IFFTW_H__ */

#endif

#ifndef FFTW_TIME_LIMIT
#define FFTW_TIME_LIMIT 2.0  /* don't run for more than two seconds */
#endif

   /* the following code is disabled for now, because it seems to
   require that we #include <windows.h> in ifftw.h to
   typedef LARGE_INTEGER crude_time, and this pulls in the whole
   Windows universe and leads to namespace conflicts (unless
   we did some hack like assuming sizeof(LARGE_INTEGER) == sizeof(long long).
   gettimeofday is provided by MinGW, which we use to cross-compile
   FFTW for Windows, and this seems to work well enough */
#if 0 && (defined(__WIN32__) || defined(_WIN32) || defined(_WIN64))
crude_time fftw_get_crude_time(void)
{
	crude_time tv;
	QueryPerformanceCounter(&tv);
	return tv;
}

static double elapsed_since(crude_time t0)
{
	crude_time t1, freq;
	QueryPerformanceCounter(&t1);
	QueryPerformanceFrequency(&freq);
	return (((double)(t1.QuadPart - t0.QuadPart))) /
		((double)freq.QuadPart);
}

#  define TIME_MIN_SEC 1.0e-2

#elif defined(HAVE_GETTIMEOFDAY)

crude_time fftw_get_crude_time(void) {
	crude_time tv;
	gettimeofday(&tv, 0);
	return tv;
}

#define elapsed_sec(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) +        \
                (double)((t1).tv_usec - (t0).tv_usec) * 1.0E-6)

static double elapsed_since(crude_time t0) {
	crude_time t1;
	gettimeofday(&t1, 0);
	return elapsed_sec(t1, t0);
}

#  define TIME_MIN_SEC 1.0e-3

#else /* !HAVE_GETTIMEOFDAY */

   /* Note that the only system where we are likely to need to fall back
   on the clock() function is Windows, for which CLOCKS_PER_SEC is 1000
   and thus the clock wraps once every 50 days.  This should hopefully
   be longer than the time required to create any single plan! */
crude_time fftw_get_crude_time(void) { return clock(); }

#define elapsed_sec(t1,t0) ((double) ((t1) - (t0)) / CLOCKS_PER_SEC)

static double elapsed_since(crude_time t0)
{
	return elapsed_sec(clock(), t0);
}

#  define TIME_MIN_SEC 2.0e-1 /* from fftw2 */

#endif /* !HAVE_GETTIMEOFDAY */

double fftw_elapsed_since(const planner *plnr, const problem *p, crude_time t0) {
	double t = elapsed_since(t0);
	if (plnr->cost_hook)
		t = plnr->cost_hook(p, t, COST_MAX);
	return t;
}

#ifdef WITH_SLOW_TIMER
/* excruciatingly slow; only use this if there is no choice! */
typedef crude_time ticks;
#  define getticks fftw_get_crude_time
#  define elapsed(t1,t0) elapsed_sec(t1,t0)
#  define TIME_MIN TIME_MIN_SEC
#  define TIME_REPEAT 4 /* from fftw2 */
#  define HAVE_TICK_COUNTER
#endif

#ifdef HAVE_TICK_COUNTER

#  ifndef TIME_MIN
#    define TIME_MIN 100.0
#  endif

#  ifndef TIME_REPEAT
#    define TIME_REPEAT 8
#  endif

static double measure(plan *pln, const problem *p, int iter)
{
	ticks t0, t1;
	int i;

	t0 = getticks();
	for (i = 0; i < iter; ++i)
		pln->adt->solve(pln, p);
	t1 = getticks();
	return elapsed(t1, t0);
}


double fftw_measure_execution_time(const planner *plnr,
	plan *pln, const problem *p)
{
	int iter;
	int repeat;

	fftw_plan_awake(pln, AWAKE_ZERO);
	p->adt->zero(p);

start_over:
	for (iter = 1; iter; iter *= 2) {
		double tmin = 0;
		int first = 1;
		crude_time begin = fftw_get_crude_time();

		/* repeat the measurement TIME_REPEAT times */
		for (repeat = 0; repeat < TIME_REPEAT; ++repeat) {
			double t = measure(pln, p, iter);

			if (plnr->cost_hook)
				t = plnr->cost_hook(p, t, COST_MAX);
			if (t < 0)
				goto start_over;

			if (first || t < tmin)
				tmin = t;
			first = 0;

			/* do not run for too long */
			if (fftw_elapsed_since(plnr, p, begin) > FFTW_TIME_LIMIT)
				break;
		}

		if (tmin >= TIME_MIN) {
			fftw_plan_awake(pln, SLEEPY);
			return tmin / (double)iter;
		}
	}
	goto start_over; /* may happen if timer is screwed up */
}

#else /* no cycle counter */

double fftw_measure_execution_time(const planner *plnr,
	plan *pln, const problem *p) {
	UNUSED(plnr);
	UNUSED(p);
	UNUSED(pln);
	return -1.0;
}

#endif

/* in place square transposition, iterative */
void fftw_transpose(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl) {
	INT i0, i1, v;

	switch (vl) {
	case 1:
		for (i1 = 1; i1 < n; ++i1) {
			for (i0 = 0; i0 < i1; ++i0) {
				FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1];
				FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0];
				I[i1 * s1 + i0 * s0] = x0;
				I[i1 * s0 + i0 * s1] = y0;
			}
		}
		break;
	case 2:
		for (i1 = 1; i1 < n; ++i1) {
			for (i0 = 0; i0 < i1; ++i0) {
				FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1];
				FFTW_REAL_TYPE x1 = I[i1 * s0 + i0 * s1 + 1];
				FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0];
				FFTW_REAL_TYPE y1 = I[i1 * s1 + i0 * s0 + 1];
				I[i1 * s1 + i0 * s0] = x0;
				I[i1 * s1 + i0 * s0 + 1] = x1;
				I[i1 * s0 + i0 * s1] = y0;
				I[i1 * s0 + i0 * s1 + 1] = y1;
			}
		}
		break;
	default:
		for (i1 = 1; i1 < n; ++i1) {
			for (i0 = 0; i0 < i1; ++i0) {
				for (v = 0; v < vl; ++v) {
					FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1 + v];
					FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0 + v];
					I[i1 * s1 + i0 * s0 + v] = x0;
					I[i1 * s0 + i0 * s1 + v] = y0;
				}
			}
		}
		break;
	}
}

struct transpose_closure {
	FFTW_REAL_TYPE *I;
	INT s0, s1, vl, tilesz;
	FFTW_REAL_TYPE *buf0, *buf1;
};

static void transpose_dotile(INT n0l, INT n0u, INT n1l, INT n1u, void *args) {
	struct transpose_closure *k = (struct transpose_closure *) args;
	FFTW_REAL_TYPE *I = k->I;
	INT s0 = k->s0, s1 = k->s1, vl = k->vl;
	INT i0, i1, v;

	switch (vl) {
	case 1:
		for (i1 = n1l; i1 < n1u; ++i1) {
			for (i0 = n0l; i0 < n0u; ++i0) {
				FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1];
				FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0];
				I[i1 * s1 + i0 * s0] = x0;
				I[i1 * s0 + i0 * s1] = y0;
			}
		}
		break;
	case 2:
		for (i1 = n1l; i1 < n1u; ++i1) {
			for (i0 = n0l; i0 < n0u; ++i0) {
				FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1];
				FFTW_REAL_TYPE x1 = I[i1 * s0 + i0 * s1 + 1];
				FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0];
				FFTW_REAL_TYPE y1 = I[i1 * s1 + i0 * s0 + 1];
				I[i1 * s1 + i0 * s0] = x0;
				I[i1 * s1 + i0 * s0 + 1] = x1;
				I[i1 * s0 + i0 * s1] = y0;
				I[i1 * s0 + i0 * s1 + 1] = y1;
			}
		}
		break;
	default:
		for (i1 = n1l; i1 < n1u; ++i1) {
			for (i0 = n0l; i0 < n0u; ++i0) {
				for (v = 0; v < vl; ++v) {
					FFTW_REAL_TYPE x0 = I[i1 * s0 + i0 * s1 + v];
					FFTW_REAL_TYPE y0 = I[i1 * s1 + i0 * s0 + v];
					I[i1 * s1 + i0 * s0 + v] = x0;
					I[i1 * s0 + i0 * s1 + v] = y0;
				}
			}
		}
	}
}

static void transpose_dotile_buf(INT n0l, INT n0u, INT n1l, INT n1u, void *args) {
	struct transpose_closure *k = (struct transpose_closure *) args;
	fftw_cpy2d_ci(k->I + n0l * k->s0 + n1l * k->s1,
		k->buf0,
		n0u - n0l, k->s0, k->vl,
		n1u - n1l, k->s1, k->vl * (n0u - n0l),
		k->vl);
	fftw_cpy2d_ci(k->I + n0l * k->s1 + n1l * k->s0,
		k->buf1,
		n0u - n0l, k->s1, k->vl,
		n1u - n1l, k->s0, k->vl * (n0u - n0l),
		k->vl);
	fftw_cpy2d_co(k->buf1,
		k->I + n0l * k->s0 + n1l * k->s1,
		n0u - n0l, k->vl, k->s0,
		n1u - n1l, k->vl * (n0u - n0l), k->s1,
		k->vl);
	fftw_cpy2d_co(k->buf0,
		k->I + n0l * k->s1 + n1l * k->s0,
		n0u - n0l, k->vl, k->s1,
		n1u - n1l, k->vl * (n0u - n0l), k->s0,
		k->vl);
}

static void transpose_rec(FFTW_REAL_TYPE *I, INT n,
	void(*f)(INT n0l, INT n0u, INT n1l, INT n1u,
		void *args),
	struct transpose_closure *k) {
tail:
	if (n > 1) {
		INT n2 = n / 2;
		k->I = I;
		fftw_tile2d(0, n2, n2, n, k->tilesz, f, k);
		transpose_rec(I, n2, f, k);
		I += n2 * (k->s0 + k->s1);
		n -= n2;
		goto tail;
	}
}

void fftw_transpose_tiled(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl) {
	struct transpose_closure k;
	k.s0 = s0;
	k.s1 = s1;
	k.vl = vl;
	/* two blocks must be in cache, to be swapped */
	k.tilesz = fftw_compute_tilesz(vl, 2);
	k.buf0 = k.buf1 = 0; /* unused */
	transpose_rec(I, n, transpose_dotile, &k);
}

void fftw_transpose_tiledbuf(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl) {
	struct transpose_closure k;
	/* Assume that the the rows of I conflict into the same cache
	lines, and therefore we don't need to reserve cache space for
	the input.  If the rows don't conflict, there is no reason
	to use tiledbuf at all.*/
	FFTW_REAL_TYPE buf0[CACHESIZE / (2 * sizeof(FFTW_REAL_TYPE))];
	FFTW_REAL_TYPE buf1[CACHESIZE / (2 * sizeof(FFTW_REAL_TYPE))];
	k.s0 = s0;
	k.s1 = s1;
	k.vl = vl;
	k.tilesz = fftw_compute_tilesz(vl, 2);
	k.buf0 = buf0;
	k.buf1 = buf1;
	A(k.tilesz * k.tilesz * vl * sizeof(FFTW_REAL_TYPE) <= sizeof(buf0));
	A(k.tilesz * k.tilesz * vl * sizeof(FFTW_REAL_TYPE) <= sizeof(buf1));
	transpose_rec(I, n, transpose_dotile_buf, &k);
}
/*

/* trigonometric functions */
#include <math.h>

#if defined(TRIGREAL_IS_LONG_DOUBLE)
#  define COS cosl
#  define SIN sinl
#  define KTRIG(x) (x##L)
#  if defined(HAVE_DECL_SINL) && !HAVE_DECL_SINL
extern long double sinl(long double x);
#  endif
#  if defined(HAVE_DECL_COSL) && !HAVE_DECL_COSL
extern long double cosl(long double x);
#  endif
#elif defined(TRIGREAL_IS_QUAD)
#  define COS cosq
#  define SIN sinq
#  define KTRIG(x) (x##Q)
extern __float128 sinq(__float128 x);
extern __float128 cosq(__float128 x);
#else
#  define COS cos
#  define SIN sin
#  define KTRIG(x) (x)
#endif

static const trigreal K2PI =
KTRIG(6.2831853071795864769252867665590057683943388);
#define by2pi(m, n) ((K2PI * (m)) / (n))

/*
* Improve accuracy by reducing x to range [0..1/8]
* before multiplication by 2 * PI.
*/

static void real_cexp(INT m, INT n, trigreal *out) {
	trigreal theta, c, s, t;
	unsigned octant = 0;
	INT quarter_n = n;

	n += n;
	n += n;
	m += m;
	m += m;

	if (m < 0) m += n;
	if (m > n - m) {
		m = n - m;
		octant |= 4;
	}
	if (m - quarter_n > 0) {
		m = m - quarter_n;
		octant |= 2;
	}
	if (m > quarter_n - m) {
		m = quarter_n - m;
		octant |= 1;
	}

	theta = by2pi(m, n);
	c = COS(theta);
	s = SIN(theta);

	if (octant & 1) {
		t = c;
		c = s;
		s = t;
	}
	if (octant & 2) {
		t = c;
		c = -s;
		s = t;
	}
	if (octant & 4) { s = -s; }

	out[0] = c;
	out[1] = s;
}

static INT choose_twshft(INT n) {
	INT log2r = 0;
	while (n > 0) {
		++log2r;
		n /= 4;
	}
	return log2r;
}

static void cexpl_sqrtn_table(triggen *p, INT m, trigreal *res) {
	m += p->n * (m < 0);

	{
		INT m0 = m & p->twmsk;
		INT m1 = m >> p->twshft;
		trigreal wr0 = p->W0[2 * m0];
		trigreal wi0 = p->W0[2 * m0 + 1];
		trigreal wr1 = p->W1[2 * m1];
		trigreal wi1 = p->W1[2 * m1 + 1];

		res[0] = wr1 * wr0 - wi1 * wi0;
		res[1] = wi1 * wr0 + wr1 * wi0;
	}
}

/* multiply (xr, xi) by exp(FFT_SIGN * 2*pi*i*m/n) */
static void rotate_sqrtn_table(triggen *p, INT m, FFTW_REAL_TYPE xr, FFTW_REAL_TYPE xi, FFTW_REAL_TYPE *res) {
	m += p->n * (m < 0);

	{
		INT m0 = m & p->twmsk;
		INT m1 = m >> p->twshft;
		trigreal wr0 = p->W0[2 * m0];
		trigreal wi0 = p->W0[2 * m0 + 1];
		trigreal wr1 = p->W1[2 * m1];
		trigreal wi1 = p->W1[2 * m1 + 1];
		trigreal wr = wr1 * wr0 - wi1 * wi0;
		trigreal wi = wi1 * wr0 + wr1 * wi0;

#if FFT_SIGN == -1
		res[0] = xr * wr + xi * wi;
		res[1] = xi * wr - xr * wi;
#else
		res[0] = xr * wr - xi * wi;
		res[1] = xi * wr + xr * wi;
#endif
	}
}

static void cexpl_sincos(triggen *p, INT m, trigreal *res) {
	real_cexp(m, p->n, res);
}

static void cexp_zero(triggen *p, INT m, FFTW_REAL_TYPE *res) {
	UNUSED(p);
	UNUSED(m);
	res[0] = 0;
	res[1] = 0;
}

static void cexpl_zero(triggen *p, INT m, trigreal *res) {
	UNUSED(p);
	UNUSED(m);
	res[0] = 0;
	res[1] = 0;
}

static void cexp_generic(triggen *p, INT m, FFTW_REAL_TYPE *res) {
	trigreal resl[2];
	p->cexpl(p, m, resl);
	res[0] = (FFTW_REAL_TYPE)resl[0];
	res[1] = (FFTW_REAL_TYPE)resl[1];
}

static void rotate_generic(triggen *p, INT m, FFTW_REAL_TYPE xr, FFTW_REAL_TYPE xi, FFTW_REAL_TYPE *res) {
	trigreal w[2];
	p->cexpl(p, m, w);
	res[0] = xr * w[0] - xi * (FFT_SIGN * w[1]);
	res[1] = xi * w[0] + xr * (FFT_SIGN * w[1]);
}

triggen *fftw_mktriggen(enum wakefulness wakefulness, INT n) {
	INT i, n0, n1;
	triggen *p = (triggen *)MALLOC(sizeof(*p), TWIDDLES);

	p->n = n;
	p->W0 = p->W1 = 0;
	p->cexp = 0;
	p->rotate = 0;

	switch (wakefulness) {
	case SLEEPY:
		A(0 /* can't happen */);
		break;

	case AWAKE_SQRTN_TABLE: {
		INT twshft = choose_twshft(n);

		p->twshft = twshft;
		p->twradix = ((INT)1) << twshft;
		p->twmsk = p->twradix - 1;

		n0 = p->twradix;
		n1 = (n + n0 - 1) / n0;

		p->W0 = (trigreal *)MALLOC(n0 * 2 * sizeof(trigreal), TWIDDLES);
		p->W1 = (trigreal *)MALLOC(n1 * 2 * sizeof(trigreal), TWIDDLES);

		for (i = 0; i < n0; ++i)
			real_cexp(i, n, p->W0 + 2 * i);

		for (i = 0; i < n1; ++i)
			real_cexp(i * p->twradix, n, p->W1 + 2 * i);

		p->cexpl = cexpl_sqrtn_table;
		p->rotate = rotate_sqrtn_table;
		break;
	}

	case AWAKE_SINCOS:
		p->cexpl = cexpl_sincos;
		break;

	case AWAKE_ZERO:
		p->cexp = cexp_zero;
		p->cexpl = cexpl_zero;
		break;
	}

	if (!p->cexp) {
		if (sizeof(trigreal) == sizeof(FFTW_REAL_TYPE))
			p->cexp = (void(*)(triggen *, INT, FFTW_REAL_TYPE *)) p->cexpl;
		else
			p->cexp = cexp_generic;
	}
	if (!p->rotate)
		p->rotate = rotate_generic;
	return p;
}

void fftw_triggen_destroy(triggen *p) {
	fftw_ifree0(p->W0);
	fftw_ifree0(p->W1);
	fftw_ifree(p);
}

/* Twiddle manipulation */

#include <math.h>

#define HASHSZ 109

/* hash table of known twiddle factors */
static twid *twlist[HASHSZ];

static INT hash(INT n, INT r) {
	INT h = n * 17 + r;

	if (h < 0) h = -h;

	return (h % HASHSZ);
}

static int equal_instr(const tw_instr *p, const tw_instr *q) {
	if (p == q)
		return 1;

	for (;; ++p, ++q) {
		if (p->op != q->op)
			return 0;

		switch (p->op) {
		case TW_NEXT:
			return (p->v == q->v); /* p->i is ignored */

		case TW_FULL:
		case TW_HALF:
			if (p->v != q->v) return 0; /* p->i is ignored */
			break;

		default:
			if (p->v != q->v || p->i != q->i) return 0;
			break;
		}
	}
	A(0 /* can't happen */);
}

static int ok_twid(const twid *t,
	enum wakefulness wakefulness,
	const tw_instr *q, INT n, INT r, INT m) {
	return (wakefulness == t->wakefulness &&
		n == t->n &&
		r == t->r &&
		m <= t->m &&
		equal_instr(t->instr, q));
}

static twid *lookup(enum wakefulness wakefulness,
	const tw_instr *q, INT n, INT r, INT m) {
	twid *p;

	for (p = twlist[hash(n, r)];
		p && !ok_twid(p, wakefulness, q, n, r, m);
		p = p->cdr);
	return p;
}

static INT twlen0(INT r, const tw_instr *p, INT *vl) {
	INT ntwiddle = 0;

	/* compute length of bytecode program */
	A(r > 0);
	for (; p->op != TW_NEXT; ++p) {
		switch (p->op) {
		case TW_FULL:
			ntwiddle += (r - 1) * 2;
			break;
		case TW_HALF:
			ntwiddle += (r - 1);
			break;
		case TW_CEXP:
			ntwiddle += 2;
			break;
		case TW_COS:
		case TW_SIN:
			ntwiddle += 1;
			break;
		}
	}

	*vl = (INT)p->v;
	return ntwiddle;
}

INT fftw_twiddle_length(INT r, const tw_instr *p) {
	INT vl;
	return twlen0(r, p, &vl);
}

static FFTW_REAL_TYPE *compute(enum wakefulness wakefulness,
	const tw_instr *instr, INT n, INT r, INT m) {
	INT ntwiddle, j, vl;
	FFTW_REAL_TYPE *W, *W0;
	const tw_instr *p;
	triggen *t = fftw_mktriggen(wakefulness, n);

	p = instr;
	ntwiddle = twlen0(r, p, &vl);

	A(m % vl == 0);

	W0 = W = (FFTW_REAL_TYPE *)MALLOC((ntwiddle * (m / vl)) * sizeof(FFTW_REAL_TYPE), TWIDDLES);

	for (j = 0; j < m; j += vl) {
		for (p = instr; p->op != TW_NEXT; ++p) {
			switch (p->op) {
			case TW_FULL: {
				INT i;
				for (i = 1; i < r; ++i) {
					A((j + (INT)p->v) * i < n);
					A((j + (INT)p->v) * i > -n);
					t->cexp(t, (j + (INT)p->v) * i, W);
					W += 2;
				}
				break;
			}

			case TW_HALF: {
				INT i;
				A((r % 2) == 1);
				for (i = 1; i + i < r; ++i) {
					t->cexp(t, MULMOD(i, (j + (INT)p->v), n), W);
					W += 2;
				}
				break;
			}

			case TW_COS: {
				FFTW_REAL_TYPE d[2];

				A((j + (INT)p->v) * p->i < n);
				A((j + (INT)p->v) * p->i > -n);
				t->cexp(t, (j + (INT)p->v) * (INT)p->i, d);
				*W++ = d[0];
				break;
			}

			case TW_SIN: {
				FFTW_REAL_TYPE d[2];

				A((j + (INT)p->v) * p->i < n);
				A((j + (INT)p->v) * p->i > -n);
				t->cexp(t, (j + (INT)p->v) * (INT)p->i, d);
				*W++ = d[1];
				break;
			}

			case TW_CEXP:
				A((j + (INT)p->v) * p->i < n);
				A((j + (INT)p->v) * p->i > -n);
				t->cexp(t, (j + (INT)p->v) * (INT)p->i, W);
				W += 2;
				break;
			}
		}
	}

	fftw_triggen_destroy(t);
	return W0;
}

static void mktwiddle(enum wakefulness wakefulness,
	twid **pp, const tw_instr *instr, INT n, INT r, INT m) {
	twid *p;
	INT h;

	if ((p = lookup(wakefulness, instr, n, r, m))) {
		++p->refcnt;
	}
	else {
		p = (twid *)MALLOC(sizeof(twid), TWIDDLES);
		p->n = n;
		p->r = r;
		p->m = m;
		p->instr = instr;
		p->refcnt = 1;
		p->wakefulness = wakefulness;
		p->W = compute(wakefulness, instr, n, r, m);

		/* cons! onto twlist */
		h = hash(n, r);
		p->cdr = twlist[h];
		twlist[h] = p;
	}

	*pp = p;
}

static void twiddle_destroy(twid **pp) {
	twid *p = *pp;
	twid **q;

	if ((--p->refcnt) == 0) {
		/* remove p from twiddle list */
		for (q = &twlist[hash(p->n, p->r)]; *q; q = &((*q)->cdr)) {
			if (*q == p) {
				*q = p->cdr;
				fftw_ifree(p->W);
				fftw_ifree(p);
				*pp = 0;
				return;
			}
		}
		A(0 /* can't happen */);
	}
}


void fftw_twiddle_awake(enum wakefulness wakefulness, twid **pp,
	const tw_instr *instr, INT n, INT r, INT m) {
	switch (wakefulness) {
	case SLEEPY:
		twiddle_destroy(pp);
		break;
	default:
		mktwiddle(wakefulness, pp, instr, n, r, m);
		break;
	}
}

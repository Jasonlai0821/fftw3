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

/* internal API definitions */
#ifndef __API_H__
#define __API_H__

#ifndef CALLING_FFTW /* defined in hook.c, when calling internal functions */
#  define COMPILING_FFTW /* used for DLL symbol exporting in fftw3.h */
#endif

/* When compiling with GNU libtool on Windows, DLL_EXPORT is #defined
   for compiling the shared-library code.  In this case, we'll #define
   FFTW_DLL to add dllexport attributes to the specified functions in
   fftw3.h.

   If we don't specify dllexport explicitly, then libtool
   automatically exports all symbols.  However, if we specify
   dllexport explicitly for any functions, then libtool apparently
   doesn't do any automatic exporting.  (Not documented, grrr, but
   this is the observed behavior with libtool 1.5.8.)  Thus, using
   this forces us to correctly dllexport every exported symbol, or
   linking bench.exe will fail.  This has the advantage of forcing
   us to mark things correctly, which is necessary for other compilers
   (such as MS VC++). */
#ifdef DLL_EXPORT
#  define FFTW_DLL
#endif

/* just in case: force <fftw3.h> not to use C99 complex numbers
   (we need this for IBM xlc because _Complex_I is treated specially
   and is defined even if <complex.h> is not included) */
#define FFTW_NO_Complex


/* FFTW internal header file */
#ifndef __IFFTW_H__
#define __IFFTW_H__

#include "config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>        /* size_t */
#include <stdarg.h>        /* va_list */
#include <stddef.h>             /* ptrdiff_t */
#include <limits.h>             /* INT_MAX */

#include "fftw/fftw3.h"

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
#ifndef FFTW_TYPE
#define FFTW_TYPE
/* determine precision and name-mangling scheme */

#if defined(FFTW_SINGLE)
typedef float FFTW_REAL_TYPE;
typedef FFTW_REAL_TYPE FFTW_COMPLEX[2];
#elif defined(FFTW_LDOUBLE)
typedef long double FFTW_REAL_TYPE;
typedef FFTW_REAL_TYPE FFTW_COMPLEX[2];
# define TRIGREAL_IS_LONG_DOUBLE
#elif defined(FFTW_QUAD)
typedef __float128 FFTW_REAL_TYPE;
typedef FFTW_REAL_TYPE FFTW_COMPLEX[2];
# define TRIGREAL_IS_QUAD
#else
typedef double FFTW_REAL_TYPE;
typedef FFTW_REAL_TYPE FFTW_COMPLEX[2];
#endif
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

#define BUF_ALLOC(T, p, n)            \
{                        \
     if ((n) < MAX_STACK_ALLOC) {            \
      STACK_MALLOC(T, p, n);        \
     } else {                    \
      (p) = (T)MALLOC(n, BUFFERS);        \
     }                        \
}

#define BUF_FREE(p, n)                \
{                        \
     if ((n) < MAX_STACK_ALLOC) {            \
      STACK_FREE(p);            \
     } else {                    \
      fftw_ifree(p);                \
     }                        \
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
#define CK(ex)                         \
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
    MALLOC_WHAT_LAST        /* must be last */
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

void fftw_ops_cpy(const opcnt *src, opcnt *dst);

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
    INT is;            /* input stride */
    INT os;            /* output stride */
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

typedef enum {
    INPLACE_IS, INPLACE_OS
} inplace_kind;

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

    void (*hash)(const problem *ego, md5 *p);

    void (*zero)(const problem *ego);

    void (*print)(const problem *ego, printer *p);

    void (*destroy)(problem *ego);
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
    void (*print)(printer *p, const char *format, ...);

    void (*vprint)(printer *p, const char *format, va_list ap);

    void (*putchr)(printer *p, char c);

    void (*cleanup)(printer *p);

    int indent;
    int indent_incr;
};

printer *fftw_mkprinter(size_t size,
                        void (*putchr)(printer *p, char c),
                        void (*cleanup)(printer *p));

IFFTW_EXTERN void fftw_printer_destroy(printer *p);

/*-----------------------------------------------------------------------*/
/* scan.c */
struct scanner_s {
    int (*scan)(scanner *sc, const char *format, ...);

    int (*vscan)(scanner *sc, const char *format, va_list ap);

    int (*getchr)(scanner *sc);

    int ungotc;
};

scanner *fftw_mkscanner(size_t size, int (*getchr)(scanner *sc));

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
    void (*solve)(const plan *ego, const problem *p);

    void (*awake)(plan *ego, enum wakefulness wakefulness);

    void (*print)(const plan *ego, printer *p);

    void (*destroy)(plan *ego);
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

    void (*destroy)(solver *ego);
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
    unsigned l:20;
    unsigned hash_info:3;
#    define BITS_FOR_TIMELIMIT 9
    unsigned timelimit_impatience:BITS_FOR_TIMELIMIT;
    unsigned u:20;

    /* abstraction break: we store the solver here to pad the
   structure to 64 bits.  Otherwise, the struct is padded to 64
   bits anyway, and another word is allocated for slvndx. */
#    define BITS_FOR_SLVNDX 12
    unsigned slvndx:BITS_FOR_SLVNDX;
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

typedef enum {
    FORGET_ACCURSED, FORGET_EVERYTHING
} amnesia;

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
    void (*ifftw_register_solver)(planner *ego, solver *s);

    plan *(*ifftw_mkplan)(planner *ego, const problem *p);

    void (*ifftw_forget)(planner *ego, amnesia a);

    void (*ifftw_exprt)(planner *ego, printer *p); /* ``export'' is a reserved
						 word in C++. */
    int (*ifftw_imprt)(planner *ego, scanner *sc);
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

typedef enum {
    COST_SUM, COST_MAX
} cost_kind;

struct planner_s {
    const ifftw_planner_adt *adt;

    void (*hook)(struct planner_s *plnr, plan *pln,
                 const problem *p, int optimalp);

    double (*cost_hook)(const problem *p, double t, cost_kind k);

    int (*wisdom_ok_hook)(const problem *p, flags_t flags);

    void (*nowisdom_hook)(const problem *p);

    wisdom_state_t (*bogosity_hook)(wisdom_state_t state, const problem *p);

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
#define FORALL_SOLVERS(ego, s, p, what)            \
{                            \
     unsigned _cnt;                    \
     for (_cnt = 0; _cnt < (ego)->nslvdesc; ++_cnt) {    \
      slvdesc *(p) = (ego)->slvdescs + _cnt;        \
      solver *(s) = (p)->slv;                \
      what;                        \
     }                            \
}

#define FORALL_SOLVERS_OF_KIND(kind, ego, s, p, what)        \
{                                \
     int _cnt = (ego)->slvdescs_for_problem_kind[kind];        \
     while (_cnt >= 0) {                    \
      slvdesc *(p) = (ego)->slvdescs + _cnt;            \
      solver *(s) = (p)->slv;                    \
      what;                            \
      _cnt = (p)->next_for_same_problem_kind;            \
     }                                \
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
#define WS(stride, i)  ((stride)[i])

extern stride fftw_mkstride(INT n, INT s);

void fftw_stride_destroy(stride p);
/* hackery to prevent the compiler from copying the strides array
   onto the stack */
#define MAKE_VOLATILE_STRIDE(nptr, x) ((x) = (x) + fftw_an_INT_guaranteed_to_be_zero)
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

struct solvtab_s {
    void (*reg)(planner *);

    const char *reg_nam;
};

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
    void (*cexp)(triggen *t, INT m, FFTW_REAL_TYPE *result);

    void (*cexpl)(triggen *t, INT m, trigreal *result);

    void (*rotate)(triggen *p, INT m, FFTW_REAL_TYPE xr, FFTW_REAL_TYPE xi, FFTW_REAL_TYPE *res);

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
                 void (*f)(INT n0l, INT n0u, INT n1l, INT n1u, void *args),
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

typedef void (*transpose_func)(FFTW_REAL_TYPE *I, INT n, INT s0, INT s1, INT vl);

typedef void (*cpy2d_func)(FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O,
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
FFTW_REAL_TYPE *fftw_taint(FFTW_REAL_TYPE *p, INT s);
FFTW_REAL_TYPE *fftw_join_taint(FFTW_REAL_TYPE *p1, FFTW_REAL_TYPE *p2);
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
#  define K(x) ((E) (x))
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
     x = - (x + c);
     return x;
}

static __inline__ E FNMS(E a, E b, E c)
{
     E x = a * b;
     x = - (x - c);
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

#ifndef __RDFT_H__
#define __RDFT_H__


/*
 * This header file must include every file or define every
 * type or macro which is required to compile a codelet.
 */

#ifndef __RDFT_CODELET_H__
#define __RDFT_CODELET_H__

/**************************************************************
 * types of codelets
 **************************************************************/

/* FOOab, with a,b in {0,1}, denotes the FOO transform
   where a/b say whether the input/output are shifted by
   half a sample/slot. */

typedef enum {
    R2HC00, R2HC01, R2HC10, R2HC11,
    HC2R00, HC2R01, HC2R10, HC2R11,
    DHT,
    REDFT00, REDFT01, REDFT10, REDFT11, /* real-even == DCT's */
    RODFT00, RODFT01, RODFT10, RODFT11  /*  real-odd == DST's */
} rdft_kind;

/* standard R2HC/HC2R transforms are unshifted */
#define R2HC R2HC00
#define HC2R HC2R00

#define R2HCII R2HC01
#define HC2RIII HC2R10

/* (k) >= R2HC00 produces a warning under gcc because checking x >= 0
   is superfluous for unsigned values...but it is needed because other
   compilers (e.g. icc) may define the enum to be a signed int...grrr. */
#define R2HC_KINDP(k) ((k) >= R2HC00 && (k) <= R2HC11) /* uses kr2hc_genus */
#define HC2R_KINDP(k) ((k) >= HC2R00 && (k) <= HC2R11) /* uses khc2r_genus */

#define R2R_KINDP(k) ((k) >= DHT) /* uses kr2r_genus */

#define REDFT_KINDP(k) ((k) >= REDFT00 && (k) <= REDFT11)
#define RODFT_KINDP(k) ((k) >= RODFT00 && (k) <= RODFT11)
#define REODFT_KINDP(k) ((k) >= REDFT00 && (k) <= RODFT11)

/* codelets with real input (output) and complex output (input) */
typedef struct kr2c_desc_s kr2c_desc;

typedef struct {
    rdft_kind kind;
    INT vl;
} kr2c_genus;

struct kr2c_desc_s {
    INT n;    /* size of transform computed */
    const char *nam;
    opcnt ops;
    const kr2c_genus *genus;
};

typedef void (*kr2c)(FFTW_REAL_TYPE *R0, FFTW_REAL_TYPE *R1, FFTW_REAL_TYPE *Cr, FFTW_REAL_TYPE *Ci,
                     stride rs, stride csr, stride csi,
                     INT vl, INT ivs, INT ovs);

void fftw_kr2c_register(planner *p, kr2c codelet, const kr2c_desc *desc);

/* half-complex to half-complex DIT/DIF codelets: */
typedef struct hc2hc_desc_s hc2hc_desc;

typedef struct {
    rdft_kind kind;
    INT vl;
} hc2hc_genus;

struct hc2hc_desc_s {
    INT radix;
    const char *nam;
    const tw_instr *tw;
    const hc2hc_genus *genus;
    opcnt ops;
};

typedef void (*khc2hc)(FFTW_REAL_TYPE *rioarray, FFTW_REAL_TYPE *iioarray, const FFTW_REAL_TYPE *W,
                       stride rs, INT mb, INT me, INT ms);

void fftw_khc2hc_register(planner *p, khc2hc codelet, const hc2hc_desc *desc);

/* half-complex to rdft2-complex DIT/DIF codelets: */
typedef struct hc2c_desc_s hc2c_desc;

typedef enum {
    HC2C_VIA_RDFT,
    HC2C_VIA_DFT
} hc2c_kind;

typedef struct {
    int (*okp)(
            const FFTW_REAL_TYPE *Rp, const FFTW_REAL_TYPE *Ip, const FFTW_REAL_TYPE *Rm, const FFTW_REAL_TYPE *Im,
            INT rs, INT mb, INT me, INT ms,
            const planner *plnr);

    rdft_kind kind;
    INT vl;
} hc2c_genus;

struct hc2c_desc_s {
    INT radix;
    const char *nam;
    const tw_instr *tw;
    const hc2c_genus *genus;
    opcnt ops;
};

typedef void (*khc2c)(FFTW_REAL_TYPE *Rp, FFTW_REAL_TYPE *Ip, FFTW_REAL_TYPE *Rm, FFTW_REAL_TYPE *Im,
                      const FFTW_REAL_TYPE *W,
                      stride rs, INT mb, INT me, INT ms);

void fftw_khc2c_register(planner *p, khc2c codelet, const hc2c_desc *desc,
                         hc2c_kind hc2ckind);

extern const solvtab fftw_solvtab_rdft_r2cf;
extern const solvtab fftw_solvtab_rdft_r2cb;
extern const solvtab fftw_solvtab_rdft_sse2;
extern const solvtab fftw_solvtab_rdft_avx;
extern const solvtab fftw_solvtab_rdft_avx_128_fma;
extern const solvtab fftw_solvtab_rdft_avx2;
extern const solvtab fftw_solvtab_rdft_avx2_128;
extern const solvtab fftw_solvtab_rdft_avx512;
extern const solvtab fftw_solvtab_rdft_kcvi;
extern const solvtab fftw_solvtab_rdft_altivec;
extern const solvtab fftw_solvtab_rdft_vsx;
extern const solvtab fftw_solvtab_rdft_neon;
extern const solvtab fftw_solvtab_rdft_generic_simd128;
extern const solvtab fftw_solvtab_rdft_generic_simd256;

/* real-input & output DFT-like codelets (DHT, etc.) */
typedef struct kr2r_desc_s kr2r_desc;

typedef struct {
    INT vl;
} kr2r_genus;

struct kr2r_desc_s {
    INT n;    /* size of transform computed */
    const char *nam;
    opcnt ops;
    const kr2r_genus *genus;
    rdft_kind kind;
};

typedef void (*kr2r)(const FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, stride is, stride os,
                     INT vl, INT ivs, INT ovs);

void fftw_kr2r_register(planner *p, kr2r codelet, const kr2r_desc *desc);

extern const solvtab fftw_solvtab_rdft_r2r;

#endif                /* __RDFT_CODELET_H__ */


#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/* problem.c: */
typedef struct {
    problem super;
    tensor *sz, *vecsz;
    FFTW_REAL_TYPE *I, *O;
#if defined(STRUCT_HACK_KR)
    rdft_kind kind[1];
#elif defined(STRUCT_HACK_C99)
    rdft_kind kind[];
#else
     rdft_kind *kind;
#endif
} problem_rdft;

void fftw_rdft_zerotens(tensor *sz, FFTW_REAL_TYPE *I);

problem *fftw_mkproblem_rdft(const tensor *sz, const tensor *vecsz,
                             FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, const rdft_kind *kind);

problem *fftw_mkproblem_rdft_d(tensor *sz, tensor *vecsz,
                               FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, const rdft_kind *kind);

problem *fftw_mkproblem_rdft_0_d(tensor *vecsz, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O);

problem *fftw_mkproblem_rdft_1(const tensor *sz, const tensor *vecsz,
                               FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, rdft_kind kind);

problem *fftw_mkproblem_rdft_1_d(tensor *sz, tensor *vecsz,
                                 FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O, rdft_kind kind);

const char *fftw_rdft_kind_str(rdft_kind kind);

/* solve.c: */
void fftw_rdft_solve(const plan *ego_, const problem *p_);

/* plan.c: */
typedef void (*rdftapply)(const plan *ego, FFTW_REAL_TYPE *I, FFTW_REAL_TYPE *O);

typedef struct {
    plan super;
    rdftapply apply;
} plan_rdft;

plan *fftw_mkplan_rdft(size_t size, const plan_adt *adt, rdftapply apply);

#define MKPLAN_RDFT(type, adt, apply) \
  (type *)fftw_mkplan_rdft(sizeof(type), adt, apply)

/* various solvers */

solver *fftw_mksolver_rdft_r2c_direct(kr2c k, const kr2c_desc *desc);

solver *fftw_mksolver_rdft_r2c_directbuf(kr2c k, const kr2c_desc *desc);

solver *fftw_mksolver_rdft_r2r_direct(kr2r k, const kr2r_desc *desc);

void fftw_rdft_rank0_register(planner *p);

void fftw_rdft_vrank3_transpose_register(planner *p);

void fftw_rdft_rank_geq2_register(planner *p);

void fftw_rdft_indirect_register(planner *p);

void fftw_rdft_vrank_geq1_register(planner *p);

void fftw_rdft_buffered_register(planner *p);

void fftw_rdft_generic_register(planner *p);

void fftw_rdft_rader_hc2hc_register(planner *p);

void fftw_rdft_dht_register(planner *p);

void fftw_dht_r2hc_register(planner *p);

void fftw_dht_rader_register(planner *p);

void fftw_dft_r2hc_register(planner *p);

void fftw_rdft_nop_register(planner *p);

void fftw_hc2hc_generic_register(planner *p);

/****************************************************************************/
/* problem2.c: */
/*
   An RDFT2 problem transforms a 1d real array r[n] with stride is/os
   to/from an "unpacked" complex array {rio,iio}[n/2 + 1] with stride
   os/is.  R0 points to the first even element of the real array.
   R1 points to the first odd element of the real array.

   Strides on the real side of the transform express distances
   between consecutive elements of the same array (even or odd).
   E.g., for a contiguous input

     R0 R1 R2 R3 ...

   the input stride would be 2, not 1.  This convention is necessary
   for hc2c codelets to work, since they transpose even/odd with
   real/imag.

   Multidimensional transforms use complex DFTs for the
   noncontiguous dimensions.  vecsz has the usual interpretation.
*/
typedef struct {
    problem super;
    tensor *sz;
    tensor *vecsz;
    FFTW_REAL_TYPE *r0, *r1;
    FFTW_REAL_TYPE *cr, *ci;
    rdft_kind kind; /* assert(kind < DHT) */
} problem_rdft2;

problem *fftw_mkproblem_rdft2(const tensor *sz, const tensor *vecsz,
                              FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
                              rdft_kind kind);

problem *fftw_mkproblem_rdft2_d(tensor *sz, tensor *vecsz,
                                FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
                                rdft_kind kind);

problem *fftw_mkproblem_rdft2_d_3pointers(tensor *sz, tensor *vecsz,
                                          FFTW_REAL_TYPE *r, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci, rdft_kind kind);

int fftw_rdft2_inplace_strides(const problem_rdft2 *p, int vdim);

INT fftw_rdft2_tensor_max_index(const tensor *sz, rdft_kind k);

void fftw_rdft2_strides(rdft_kind kind, const iodim *d, INT *rs, INT *cs);

INT fftw_rdft2_complex_n(INT real_n, rdft_kind kind);

/* verify.c: */
void fftw_rdft2_verify(plan *pln, const problem_rdft2 *p, int rounds);

/* solve.c: */
void fftw_rdft2_solve(const plan *ego_, const problem *p_);

/* plan.c: */
typedef void (*rdft2apply)(const plan *ego, FFTW_REAL_TYPE *r0, FFTW_REAL_TYPE *r1, FFTW_REAL_TYPE *cr,
                           FFTW_REAL_TYPE *ci);

typedef struct {
    plan super;
    rdft2apply apply;
} plan_rdft2;

plan *fftw_mkplan_rdft2(size_t size, const plan_adt *adt, rdft2apply apply);

#define MKPLAN_RDFT2(type, adt, apply) \
  (type *)fftw_mkplan_rdft2(sizeof(type), adt, apply)

/* various solvers */

solver *fftw_mksolver_rdft2_direct(kr2c k, const kr2c_desc *desc);

void fftw_rdft2_vrank_geq1_register(planner *p);

void fftw_rdft2_buffered_register(planner *p);

void fftw_rdft2_rdft_register(planner *p);

void fftw_rdft2_nop_register(planner *p);

void fftw_rdft2_rank0_register(planner *p);

void fftw_rdft2_rank_geq2_register(planner *p);

/****************************************************************************/

/* configurations */
void fftw_rdft_conf_standard(planner *p);

typedef void (*hc2capply)(const plan *ego, FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci);

typedef struct hc2c_solver_s hc2c_solver;

typedef plan *(*hc2c_mkinferior)(const hc2c_solver *ego, rdft_kind kind,
                                 INT r, INT rs,
                                 INT m, INT ms,
                                 INT v, INT vs,
                                 FFTW_REAL_TYPE *cr, FFTW_REAL_TYPE *ci,
                                 planner *plnr);

typedef struct {
    plan super;
    hc2capply apply;
} plan_hc2c;

extern plan *fftw_mkplan_hc2c(size_t size, const plan_adt *adt,
                              hc2capply apply);

#define MKPLAN_HC2C(type, adt, apply) \
  (type *)fftw_mkplan_hc2c(sizeof(type), adt, apply)

struct hc2c_solver_s {
    solver super;
    INT r;

    hc2c_mkinferior mkcldw;
    hc2c_kind hc2ckind;
};

hc2c_solver *fftw_mksolver_hc2c(size_t size, INT r,
                                hc2c_kind hc2ckind,
                                hc2c_mkinferior mkcldw);

void fftw_regsolver_hc2c_direct(planner *plnr, khc2c codelet,
                                const hc2c_desc *desc,
                                hc2c_kind hc2ckind);

typedef void (*hc2hcapply)(const plan *ego, FFTW_REAL_TYPE *IO);

typedef struct hc2hc_solver_s hc2hc_solver;

typedef plan *(*hc2hc_mkinferior)(const hc2hc_solver *ego,
                                  rdft_kind kind, INT r, INT m, INT s,
                                  INT vl, INT vs, INT mstart, INT mcount,
                                  FFTW_REAL_TYPE *IO, planner *plnr);

typedef struct {
    plan super;
    hc2hcapply apply;
} plan_hc2hc;

extern plan *fftw_mkplan_hc2hc(size_t size, const plan_adt *adt,
                               hc2hcapply apply);

#define MKPLAN_HC2HC(type, adt, apply) \
  (type *)fftw_mkplan_hc2hc(sizeof(type), adt, apply)

struct hc2hc_solver_s {
    solver super;
    INT r;

    hc2hc_mkinferior mkcldw;
};

hc2hc_solver *fftw_mksolver_hc2hc(size_t size, INT r, hc2hc_mkinferior mkcldw);

extern hc2hc_solver *(*fftw_mksolver_hc2hc_hook)(size_t, INT, hc2hc_mkinferior);

void fftw_regsolver_hc2hc_direct(planner *plnr, khc2hc codelet,
                                 const hc2hc_desc *desc);

int fftw_hc2hc_applicable(const hc2hc_solver *, const problem *, planner *);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* __RDFT_H__ */

/*
 * This header file must include every file or define every
 * type or macro which is required to compile a codelet.
 */

#ifndef __DFT_CODELET_H__
#define __DFT_CODELET_H__



/**************************************************************
 * types of codelets
 **************************************************************/

/* DFT codelets */
typedef struct kdft_desc_s kdft_desc;

typedef struct {
    int (*okp)(
            const kdft_desc *desc,
            const FFTW_REAL_TYPE *ri, const FFTW_REAL_TYPE *ii, const FFTW_REAL_TYPE *ro, const FFTW_REAL_TYPE *io,
            INT is, INT os, INT vl, INT ivs, INT ovs,
            const planner *plnr);

    INT vl;
} kdft_genus;

struct kdft_desc_s {
    INT sz;    /* size of transform computed */
    const char *nam;
    opcnt ops;
    const kdft_genus *genus;
    INT is;
    INT os;
    INT ivs;
    INT ovs;
};

typedef void (*kdft)(const FFTW_REAL_TYPE *ri, const FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                     stride is, stride os, INT vl, INT ivs, INT ovs);

void fftw_kdft_register(planner *p, kdft codelet, const kdft_desc *desc);


typedef struct ct_desc_s ct_desc;

typedef struct {
    int (*okp)(
            const struct ct_desc_s *desc,
            const FFTW_REAL_TYPE *rio, const FFTW_REAL_TYPE *iio,
            INT rs, INT vs, INT m, INT mb, INT me, INT ms,
            const planner *plnr);

    INT vl;
} ct_genus;

struct ct_desc_s {
    INT radix;
    const char *nam;
    const tw_instr *tw;
    const ct_genus *genus;
    opcnt ops;
    INT rs;
    INT vs;
    INT ms;
};

typedef void (*kdftw)(FFTW_REAL_TYPE *rioarray, FFTW_REAL_TYPE *iioarray, const FFTW_REAL_TYPE *W,
                      stride ios, INT mb, INT me, INT ms);

void fftw_kdft_dit_register(planner *p, kdftw codelet, const ct_desc *desc);

void fftw_kdft_dif_register(planner *p, kdftw codelet, const ct_desc *desc);


typedef void (*kdftwsq)(FFTW_REAL_TYPE *rioarray, FFTW_REAL_TYPE *iioarray,
                        const FFTW_REAL_TYPE *W, stride is, stride vs,
                        INT mb, INT me, INT ms);

void fftw_kdft_difsq_register(planner *p, kdftwsq codelet, const ct_desc *desc);


extern const solvtab fftw_solvtab_dft_standard;
extern const solvtab fftw_solvtab_dft_sse2;
extern const solvtab fftw_solvtab_dft_avx;
extern const solvtab fftw_solvtab_dft_avx_128_fma;
extern const solvtab fftw_solvtab_dft_avx2;
extern const solvtab fftw_solvtab_dft_avx2_128;
extern const solvtab fftw_solvtab_dft_avx512;
extern const solvtab fftw_solvtab_dft_kcvi;
extern const solvtab fftw_solvtab_dft_altivec;
extern const solvtab fftw_solvtab_dft_vsx;
extern const solvtab fftw_solvtab_dft_neon;
extern const solvtab fftw_solvtab_dft_generic_simd128;
extern const solvtab fftw_solvtab_dft_generic_simd256;

#endif                /* __DFT_CODELET_H__ */


#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#ifndef __DFT_H__
#define __DFT_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/* problem.c: */
typedef struct {
    problem super;
    tensor *sz, *vecsz;
    FFTW_REAL_TYPE *ri, *ii, *ro, *io;
} problem_dft;

void fftw_dft_zerotens(tensor *sz, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii);

problem *fftw_mkproblem_dft(const tensor *sz, const tensor *vecsz,
                            FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io);

problem *fftw_mkproblem_dft_d(tensor *sz, tensor *vecsz,
                              FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io);

/* solve.c: */
void fftw_dft_solve(const plan *ego_, const problem *p_);

/* plan.c: */
typedef void (*dftapply)(const plan *ego, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro,
                         FFTW_REAL_TYPE *io);

typedef struct {
    plan super;
    dftapply apply;
} plan_dft;

plan *fftw_mkplan_dft(size_t size, const plan_adt *adt, dftapply apply);

#ifndef MKPLAN_DFT
#define MKPLAN_DFT(type, adt, apply)   (type *)fftw_mkplan_dft(sizeof(type), adt, apply)
#endif

/* various solvers */
solver *fftw_mksolver_dft_direct(kdft k, const kdft_desc *desc);

solver *fftw_mksolver_dft_directbuf(kdft k, const kdft_desc *desc);

void fftw_dft_rank0_register(planner *p);

void fftw_dft_rank_geq2_register(planner *p);

void fftw_dft_indirect_register(planner *p);

void fftw_dft_indirect_transpose_register(planner *p);

void fftw_dft_vrank_geq1_register(planner *p);

void fftw_dft_vrank2_transpose_register(planner *p);

void fftw_dft_vrank3_transpose_register(planner *p);

void fftw_dft_buffered_register(planner *p);

void fftw_dft_generic_register(planner *p);

void fftw_dft_rader_register(planner *p);

void fftw_dft_bluestein_register(planner *p);

void fftw_dft_nop_register(planner *p);

void fftw_ct_generic_register(planner *p);

void fftw_ct_genericbuf_register(planner *p);

/* configurations */
void fftw_dft_conf_standard(planner *p);

typedef void (*dftwapply)(const plan *ego, FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio);

typedef struct ct_solver_s ct_solver;

typedef plan *(*ct_mkinferior)(const ct_solver *ego,
                               INT r, INT irs, INT ors,
                               INT m, INT ms,
                               INT v, INT ivs, INT ovs,
                               INT mstart, INT mcount,
                               FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio, planner *plnr);

typedef int (*ct_force_vrecursion)(const ct_solver *ego,
                                   const problem_dft *p);

typedef struct {
    plan super;
    dftwapply apply;
} plan_dftw;

extern plan *fftw_mkplan_dftw(size_t size, const plan_adt *adt, dftwapply apply);

#define MKPLAN_DFTW(type, adt, apply) \
  (type *)fftw_mkplan_dftw(sizeof(type), adt, apply)

struct ct_solver_s {
    solver super;
    INT r;
    int dec;
#    define DECDIF 0
#    define DECDIT 1
#    define TRANSPOSE 2
    ct_mkinferior mkcldw;
    ct_force_vrecursion force_vrecursionp;
};

int fftw_ct_applicable(const ct_solver *, const problem *, planner *);

ct_solver *fftw_mksolver_ct(size_t size, INT r, int dec,
                            ct_mkinferior mkcldw,
                            ct_force_vrecursion force_vrecursionp);

extern ct_solver *(*fftw_mksolver_ct_hook)(size_t, INT, int,
                                           ct_mkinferior, ct_force_vrecursion);

void fftw_regsolver_ct_directw(planner *plnr,
                               kdftw codelet, const ct_desc *desc, int dec);

void fftw_regsolver_ct_directwbuf(planner *plnr,
                                  kdftw codelet, const ct_desc *desc, int dec);

solver *fftw_mksolver_ctsq(kdftwsq codelet, const ct_desc *desc, int dec);

void fftw_regsolver_ct_directwsq(planner *plnr, kdftwsq codelet,
                                 const ct_desc *desc, int dec);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* __DFT_H__ */

#ifndef __REODFT_H__
#define __REODFT_H__


#define REODFT_KINDP(k) ((k) >= REDFT00 && (k) <= RODFT11)

void fftw_redft00e_r2hc_register(planner *p);

void fftw_redft00e_r2hc_pad_register(planner *p);

void fftw_rodft00e_r2hc_register(planner *p);

void fftw_rodft00e_r2hc_pad_register(planner *p);

void fftw_reodft00e_splitradix_register(planner *p);

void fftw_reodft010e_r2hc_register(planner *p);

void fftw_reodft11e_r2hc_register(planner *p);

void fftw_reodft11e_radix2_r2hc_register(planner *p);

void fftw_reodft11e_r2hc_odd_register(planner *p);

/* configurations */
void fftw_reodft_conf_standard(planner *p);

#endif /* __REODFT_H__ */


/* the API ``plan'' contains both the kernel plan and problem */
struct fftw_plan_s {
    plan *pln;
    problem *prb;
    int sign;
};

/* shorthand */
typedef struct fftw_plan_s apiplan;

/* complex type for internal use */
typedef FFTW_REAL_TYPE FFTW_COMPLEX[2];

#define EXTRACT_REIM(sign, c, r, i) fftw_extract_reim(sign, (c)[0], r, i)

#define TAINT_UNALIGNED(p, flg) TAINT(p, ((flg) & FFTW_UNALIGNED) != 0)

tensor *fftw_mktensor_rowmajor(int rnk, const int *n,
                               const int *niphys, const int *nophys,
                               int is, int os);

tensor *fftw_mktensor_iodims(int rank, const fftw_iodim *dims, int is, int os);

tensor *fftw_mktensor_iodims64(int rank, const fftw_iodim64 *dims, int is, int os);

const int *fftw_rdft2_pad(int rnk, const int *n, const int *nembed,
                          int inplace, int cmplx, int **nfree);

int fftw_many_kosherp(int rnk, const int *n, int howmany);

int fftw_guru_kosherp(int rank, const fftw_iodim *dims,
                      int howmany_rank, const fftw_iodim *howmany_dims);

int fftw_guru64_kosherp(int rank, const fftw_iodim64 *dims,
                        int howmany_rank, const fftw_iodim64 *howmany_dims);

/* Note: FFTW_EXTERN is used for "internal" functions used in fftw/tests/hook.c */

FFTW_EXTERN printer *fftw_mkprinter_file(FILE *f);

printer *fftw_mkprinter_cnt(size_t *cnt);

printer *fftw_mkprinter_str(char *s);

FFTW_EXTERN planner *fftw_the_planner(void);

void fftw_configure_planner(planner *plnr);

void fftw_mapflags(planner *, unsigned);

apiplan *fftw_mkapiplan(int sign, unsigned flags, problem *prb);

rdft_kind *fftw_map_r2r_kind(int rank, const fftw_r2r_kind *kind);

typedef void (*planner_hook_t)(void);

void fftw_set_planner_hooks(planner_hook_t before, planner_hook_t after);


#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif                /* __API_H__ */

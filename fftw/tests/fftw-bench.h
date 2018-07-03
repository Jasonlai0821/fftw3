/* declarations of common subroutines, etc. for use with FFTW
   self-test/benchmark program (see bench.c). */

#include "fftw/libbench2/bench-user.h"
#include "fftw/fftw3.h"
#ifndef FFTW_TYPE
#define FFTW_TYPE
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
#define CONCAT(prefix, name) prefix ## name
#if defined(BENCHFFT_SINGLE)
#define FFTW(x) CONCAT(fftw_, x)
#elif defined(BENCHFFT_LDOUBLE)
#define FFTW(x) CONCAT(fftwl_, x)
#elif defined(BENCHFFT_QUAD)
#define FFTW(x) CONCAT(fftwq_, x)
#else
#define FFTW(x) CONCAT(fftw_, x)
#endif

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

extern FFTW(plan) mkplan(bench_problem *p, unsigned flags);
extern void initial_cleanup(void);
extern void final_cleanup(void);
extern int import_wisdom(FILE *f);
extern void export_wisdom(FILE *f);

#if defined(HAVE_THREADS) || defined(HAVE_OPENMP)
#  define HAVE_SMP
   extern int threads_ok;
#endif

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */


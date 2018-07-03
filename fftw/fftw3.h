/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * The following statement of license applies *only* to this header file,
 * and *not* to the other files distributed with FFTW or derived therefrom:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/***************************** NOTE TO USERS *********************************
 *
 *                 THIS IS A HEADER FILE, NOT A MANUAL
 *
 *    If you want to know how to use FFTW, please read the manual,
 *    online at http://www.fftw.org/doc/ and also included with FFTW.
 *    For a quick start, see the manual's tutorial section.
 *
 *   (Reading header files to learn how to use a library is a habit
 *    stemming from code lacking a proper manual.  Arguably, it's a
 *    *bad* habit in most cases, because header files can contain
 *    interfaces that are not part of the public, stable API.)
 *
 ****************************************************************************/

#ifndef FFTW3_H
#define FFTW3_H

#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */


/*
  huge second-order macro that defines prototypes for all API
  functions.  We expand this macro for each supported precision

  FFTW_REAL_TYPE: real data type
  FFTW_COMPLEX: complex data type
*/
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
/* IMPORTANT: for Windows compilers, you should add a line
        #define FFTW_DLL
   here and in kernel/ifftw.h if you are compiling/using FFTW as a
   DLL, in order to do the proper importing/exporting, or
   alternatively compile with -DFFTW_DLL or the equivalent
   command-line flag.  This is not necessary under MinGW/Cygwin, where
   libtool does the imports/exports automatically. */
#if defined(FFTW_DLL) && (defined(_WIN32) || defined(__WIN32__))
/* annoying Windows syntax for shared-library declarations */
#  if defined(COMPILING_FFTW) /* defined in fftw.h when compiling FFTW */
#    define FFTW_EXTERN extern __declspec(dllexport)
#  else /* user is calling FFTW; import symbol */
#    define FFTW_EXTERN extern __declspec(dllimport)
#  endif
#else
#  define FFTW_EXTERN extern
#endif

/* specify calling convention (Windows only) */
#if defined(_WIN32) || defined(__WIN32__)
#  define FFTW_CDECL __cdecl
#else
#  define FFTW_CDECL
#endif

enum fftw_r2r_kind_do_not_use_me {
    FFTW_R2HC = 0, FFTW_HC2R = 1, FFTW_DHT = 2,
    FFTW_REDFT00 = 3, FFTW_REDFT01 = 4, FFTW_REDFT10 = 5, FFTW_REDFT11 = 6,
    FFTW_RODFT00 = 7, FFTW_RODFT01 = 8, FFTW_RODFT10 = 9, FFTW_RODFT11 = 10
};

struct fftw_iodim_do_not_use_me {
    int n;                     /* dimension size */
    int is;            /* input stride */
    int os;            /* output stride */
};

#include <stddef.h> /* for ptrdiff_t */

struct fftw_iodim64_do_not_use_me {
    ptrdiff_t n;                     /* dimension size */
    ptrdiff_t is;            /* input stride */
    ptrdiff_t os;            /* output stride */
};

typedef void (FFTW_CDECL *fftw_write_char_func_do_not_use_me)(char c, void *);

typedef int (FFTW_CDECL *fftw_read_char_func_do_not_use_me)(void *);


typedef struct fftw_plan_s *fftw_plan;

typedef struct fftw_iodim_do_not_use_me fftw_iodim;
typedef struct fftw_iodim64_do_not_use_me fftw_iodim64;

typedef enum fftw_r2r_kind_do_not_use_me fftw_r2r_kind;

typedef fftw_write_char_func_do_not_use_me fftw_write_char_func;
typedef fftw_read_char_func_do_not_use_me fftw_read_char_func;

FFTW_EXTERN void
FFTW_CDECL fftw_execute(const fftw_plan p);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft(int rank, const int *n,
                         FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_1d(int n, FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign,
                            unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_2d(int n0, int n1,
                            FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_3d(int n0, int n1, int n2,
                            FFTW_COMPLEX *in, FFTW_COMPLEX *out, int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_many_dft(int rank, const int *n,
                              int howmany,
                              FFTW_COMPLEX *in, const int *inembed,
                              int istride, int idist,
                              FFTW_COMPLEX *out, const int *onembed,
                              int ostride, int odist,
                              int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                              int howmany_rank,
                              const fftw_iodim *howmany_dims,
                              FFTW_COMPLEX *in, FFTW_COMPLEX *out,
                              int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims,
                                    int howmany_rank,
                                    const fftw_iodim *howmany_dims,
                                    FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                    unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_dft(int rank,
                                const fftw_iodim64 *dims,
                                int howmany_rank,
                                const fftw_iodim64 *howmany_dims,
                                FFTW_COMPLEX *in, FFTW_COMPLEX *out,
                                int sign, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_split_dft(int rank,
                                      const fftw_iodim64 *dims,
                                      int howmany_rank,
                                      const fftw_iodim64 *howmany_dims,
                                      FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                      unsigned flags);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_dft(const fftw_plan p, FFTW_COMPLEX *in, FFTW_COMPLEX *out);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_split_dft(const fftw_plan p, FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii,
                                  FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_many_dft_r2c(int rank, const int *n,
                                  int howmany,
                                  FFTW_REAL_TYPE *in, const int *inembed,
                                  int istride, int idist,
                                  FFTW_COMPLEX *out, const int *onembed,
                                  int ostride, int odist,
                                  unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_r2c(int rank, const int *n,
                             FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_r2c_1d(int n, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_r2c_2d(int n0, int n1,
                                FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_r2c_3d(int n0, int n1,
                                int n2,
                                FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_many_dft_c2r(int rank, const int *n,
                                  int howmany,
                                  FFTW_COMPLEX *in, const int *inembed,
                                  int istride, int idist,
                                  FFTW_REAL_TYPE *out, const int *onembed,
                                  int ostride, int odist,
                                  unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_c2r(int rank, const int *n,
                             FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_c2r_1d(int n, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_c2r_2d(int n0, int n1,
                                FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_dft_c2r_3d(int n0, int n1,
                                int n2,
                                FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                  int howmany_rank,
                                  const fftw_iodim *howmany_dims,
                                  FFTW_REAL_TYPE *in, FFTW_COMPLEX *out,
                                  unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims,
                                  int howmany_rank,
                                  const fftw_iodim *howmany_dims,
                                  FFTW_COMPLEX *in, FFTW_REAL_TYPE *out,
                                  unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims,
                                        int howmany_rank,
                                        const fftw_iodim *howmany_dims,
                                        FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                        unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims,
                                        int howmany_rank,
                                        const fftw_iodim *howmany_dims,
                                        FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out,
                                        unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_dft_r2c(int rank,
                                    const fftw_iodim64 *dims,
                                    int howmany_rank,
                                    const fftw_iodim64 *howmany_dims,
                                    FFTW_REAL_TYPE *in, FFTW_COMPLEX *out,
                                    unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_dft_c2r(int rank,
                                    const fftw_iodim64 *dims,
                                    int howmany_rank,
                                    const fftw_iodim64 *howmany_dims,
                                    FFTW_COMPLEX *in, FFTW_REAL_TYPE *out,
                                    unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_split_dft_r2c(int rank, const fftw_iodim64 *dims,
                                          int howmany_rank,
                                          const fftw_iodim64 *howmany_dims,
                                          FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io,
                                          unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_split_dft_c2r(int rank, const fftw_iodim64 *dims,
                                          int howmany_rank,
                                          const fftw_iodim64 *howmany_dims,
                                          FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out,
                                          unsigned flags);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_dft_r2c(const fftw_plan p, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_dft_c2r(const fftw_plan p, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_split_dft_r2c(const fftw_plan p,
                                      FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_split_dft_c2r(const fftw_plan p,
                                      FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_many_r2r(int rank, const int *n,
                              int howmany,
                              FFTW_REAL_TYPE *in, const int *inembed,
                              int istride, int idist,
                              FFTW_REAL_TYPE *out, const int *onembed,
                              int ostride, int odist,
                              const fftw_r2r_kind *kind, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_r2r(int rank, const int *n, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                         const fftw_r2r_kind *kind, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_r2r_1d(int n, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                            fftw_r2r_kind kind, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_r2r_2d(int n0, int n1, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                            fftw_r2r_kind kind0, fftw_r2r_kind kind1,
                            unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_r2r_3d(int n0, int n1, int n2,
                            FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out, fftw_r2r_kind kind0,
                            fftw_r2r_kind kind1, fftw_r2r_kind kind2,
                            unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru_r2r(int rank, const fftw_iodim *dims,
                              int howmany_rank,
                              const fftw_iodim *howmany_dims,
                              FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                              const fftw_r2r_kind *kind, unsigned flags);

FFTW_EXTERN fftw_plan
FFTW_CDECL fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims,
                                int howmany_rank,
                                const fftw_iodim64 *howmany_dims,
                                FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
                                const fftw_r2r_kind *kind, unsigned flags);

FFTW_EXTERN void
FFTW_CDECL fftw_execute_r2r(const fftw_plan p, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out);

FFTW_EXTERN void
FFTW_CDECL fftw_destroy_plan(fftw_plan p);

FFTW_EXTERN void
FFTW_CDECL fftw_forget_wisdom(void);

FFTW_EXTERN void
FFTW_CDECL fftw_cleanup(void);

FFTW_EXTERN void
FFTW_CDECL fftw_set_timelimit(double t);

FFTW_EXTERN void
FFTW_CDECL fftw_plan_with_nthreads(int nthreads);

FFTW_EXTERN int
FFTW_CDECL fftw_init_threads(void);

FFTW_EXTERN void
FFTW_CDECL fftw_cleanup_threads(void);

FFTW_EXTERN void
FFTW_CDECL fftw_make_planner_thread_safe(void);

FFTW_EXTERN int
FFTW_CDECL fftw_export_wisdom_to_filename(const char *filename);

FFTW_EXTERN void
FFTW_CDECL fftw_export_wisdom_to_file(FILE *output_file);

FFTW_EXTERN char *
FFTW_CDECL fftw_export_wisdom_to_string(void);

FFTW_EXTERN void
FFTW_CDECL fftw_export_wisdom(fftw_write_char_func write_char,
                              void *data);

FFTW_EXTERN int
FFTW_CDECL fftw_import_system_wisdom(void);

FFTW_EXTERN int
FFTW_CDECL fftw_import_wisdom_from_filename(const char *filename);

FFTW_EXTERN int
FFTW_CDECL fftw_import_wisdom_from_file(FILE *input_file);

FFTW_EXTERN int
FFTW_CDECL fftw_import_wisdom_from_string(const char *input_string);

FFTW_EXTERN int
FFTW_CDECL fftw_import_wisdom(fftw_read_char_func read_char, void *data);

FFTW_EXTERN void
FFTW_CDECL fftw_fprint_plan(const fftw_plan p, FILE *output_file);

FFTW_EXTERN void
FFTW_CDECL fftw_print_plan(const fftw_plan p);

FFTW_EXTERN char *
FFTW_CDECL fftw_sprint_plan(const fftw_plan p);

FFTW_EXTERN void *
FFTW_CDECL fftw_malloc(size_t n);

FFTW_EXTERN FFTW_REAL_TYPE *
FFTW_CDECL fftw_alloc_real(size_t n);

FFTW_EXTERN FFTW_COMPLEX *
FFTW_CDECL fftw_alloc_complex(size_t n);

FFTW_EXTERN void
FFTW_CDECL fftw_free(void *p);

FFTW_EXTERN void
FFTW_CDECL fftw_flops(const fftw_plan p,
                      double *add, double *mul, double *fmas);

FFTW_EXTERN double
FFTW_CDECL fftw_estimate_cost(const fftw_plan p);

FFTW_EXTERN double
FFTW_CDECL fftw_cost(const fftw_plan p);

FFTW_EXTERN int
FFTW_CDECL fftw_alignment_of(FFTW_REAL_TYPE *p);

FFTW_EXTERN const char fftw_version[];
FFTW_EXTERN const char fftw_cc[];
FFTW_EXTERN const char fftw_codelet_optim[];


/* end of FFTW_DEFINE_API macro */


#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)

#define FFTW_NO_TIMELIMIT (-1.0)

/* documented flags */
#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3) /* NO_EXHAUSTIVE is default */
#define FFTW_PRESERVE_INPUT (1U << 4) /* cancels FFTW_DESTROY_INPUT */
#define FFTW_PATIENT (1U << 5) /* IMPATIENT is default */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

/* undocumented beyond-guru flags */
#define FFTW_ESTIMATE_PATIENT (1U << 7)
#define FFTW_BELIEVE_PCOST (1U << 8)
#define FFTW_NO_DFT_R2HC (1U << 9)
#define FFTW_NO_NONTHREADED (1U << 10)
#define FFTW_NO_BUFFERING (1U << 11)
#define FFTW_NO_INDIRECT_OP (1U << 12)
#define FFTW_ALLOW_LARGE_GENERIC (1U << 13) /* NO_LARGE_GENERIC is default */
#define FFTW_NO_RANK_SPLITS (1U << 14)
#define FFTW_NO_VRANK_SPLITS (1U << 15)
#define FFTW_NO_VRECURSE (1U << 16)
#define FFTW_NO_SIMD (1U << 17)
#define FFTW_NO_SLOW (1U << 18)
#define FFTW_NO_FIXED_RADIX_LARGE_N (1U << 19)
#define FFTW_ALLOW_PRUNING (1U << 20)

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* FFTW3_H */

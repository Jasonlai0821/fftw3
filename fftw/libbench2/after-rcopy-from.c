/* not worth copyrighting */
#include "fftw/libbench2/bench.h"

/* default routine, can be overridden by user */
void after_problem_rcopy_from(bench_problem *p, bench_real *ri)
{
     UNUSED(p);
     UNUSED(ri);
}

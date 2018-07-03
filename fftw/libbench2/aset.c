/* not worth copyrighting */

#include "fftw/libbench2/bench.h"

void aset(bench_real *A, int n, bench_real x)
{
     int i;
     for (i = 0; i < n; ++i)
	  A[i] = x;
}

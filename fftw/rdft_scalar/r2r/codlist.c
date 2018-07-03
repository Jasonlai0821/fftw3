#include "fftw/fftw_api.h"


extern void fftw_codelet_e01_8(planner *);
extern void fftw_codelet_e10_8(planner *);


extern const solvtab fftw_solvtab_rdft_r2r;
const solvtab fftw_solvtab_rdft_r2r = {
   SOLVTAB(fftw_codelet_e01_8),
   SOLVTAB(fftw_codelet_e10_8),
   SOLVTAB_END
};

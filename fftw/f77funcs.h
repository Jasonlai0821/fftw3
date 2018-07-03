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

/* Functions in the FFTW Fortran API, mangled according to the
   F77(...) macro.  This file is designed to be #included by
   f77api.c, possibly multiple times in order to support multiple
   compiler manglings (via redefinition of F77). */

FFTW_VOIDFUNC F77(execute, EXECUTE)(fftw_plan * const p)
{
     plan *pln = (*p)->pln;
     pln->adt->solve(pln, (*p)->prb);
}

FFTW_VOIDFUNC F77(destroy_plan, DESTROY_PLAN)(fftw_plan *p)
{
     fftw_destroy_plan(*p);
}

FFTW_VOIDFUNC F77(cleanup, CLEANUP)(void)
{
     fftw_cleanup();
}

FFTW_VOIDFUNC F77(forget_wisdom, FORGET_WISDOM)(void)
{
     fftw_forget_wisdom();
}

FFTW_VOIDFUNC F77(export_wisdom, EXPORT_WISDOM)(void (*f77_write_char)(char *, void *),
				       void *data)
{
     write_char_data ad;
     ad.f77_write_char = f77_write_char;
     ad.data = data;
     fftw_export_wisdom(write_char, (void *) &ad);
}

FFTW_VOIDFUNC F77(import_wisdom, IMPORT_WISDOM)(int *isuccess,
				       void (*f77_read_char)(int *, void *),
				       void *data)
{
     read_char_data ed;
     ed.f77_read_char = f77_read_char;
     ed.data = data;
     *isuccess = fftw_import_wisdom(read_char, (void *) &ed);
}

FFTW_VOIDFUNC F77(import_system_wisdom, IMPORT_SYSTEM_WISDOM)(int *isuccess)
{
     *isuccess = fftw_import_system_wisdom();
}

FFTW_VOIDFUNC F77(print_plan, PRINT_PLAN)(fftw_plan * const p)
{
     fftw_print_plan(*p);
     fflush(stdout);
}

FFTW_VOIDFUNC F77(flops,FLOPS)(fftw_plan *p, double *add, double *mul, double *fma)
{
     fftw_flops(*p, add, mul, fma);
}

FFTW_VOIDFUNC F77(estimate_cost,ESTIMATE_COST)(double *cost, fftw_plan * const p)
{
     *cost = fftw_estimate_cost(*p);
}

FFTW_VOIDFUNC F77(cost,COST)(double *cost, fftw_plan * const p)
{
     *cost = fftw_cost(*p);
}

FFTW_VOIDFUNC F77(set_timelimit,SET_TIMELIMIT)(double *t)
{
     fftw_set_timelimit(*t);
}

/******************************** DFT ***********************************/

FFTW_VOIDFUNC F77(plan_dft, PLAN_DFT)(fftw_plan *p, int *rank, const int *n,
			     FFTW_COMPLEX *in, FFTW_COMPLEX *out, int *sign, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     *p = fftw_plan_dft(*rank, nrev, in, out, *sign, *flags);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_dft_1d, PLAN_DFT_1D)(fftw_plan *p, int *n, FFTW_COMPLEX *in, FFTW_COMPLEX *out,
				   int *sign, int *flags)
{
     *p = fftw_plan_dft_1d(*n, in, out, *sign, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_2d, PLAN_DFT_2D)(fftw_plan *p, int *nx, int *ny,
				   FFTW_COMPLEX *in, FFTW_COMPLEX *out, int *sign, int *flags)
{
     *p = fftw_plan_dft_2d(*ny, *nx, in, out, *sign, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_3d, PLAN_DFT_3D)(fftw_plan *p, int *nx, int *ny, int *nz,
				   FFTW_COMPLEX *in, FFTW_COMPLEX *out,
				   int *sign, int *flags)
{
     *p = fftw_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
}

FFTW_VOIDFUNC F77(plan_many_dft, PLAN_MANY_DFT)(fftw_plan *p, int *rank, const int *n,
				       int *howmany,
				       FFTW_COMPLEX *in, const int *inembed,
				       int *istride, int *idist,
				       FFTW_COMPLEX *out, const int *onembed,
				       int *ostride, int *odist,
				       int *sign, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     int *inembedrev = reverse_n(*rank, inembed);
     int *onembedrev = reverse_n(*rank, onembed);
     *p = fftw_plan_many_dft(*rank, nrev, *howmany,
			   in, inembedrev, *istride, *idist,
			   out, onembedrev, *ostride, *odist,
			   *sign, *flags);
     fftw_ifree0(onembedrev);
     fftw_ifree0(inembedrev);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_guru_dft, PLAN_GURU_DFT)(fftw_plan *p, int *rank, const int *n,
				       const int *is, const int *os,
				       int *howmany_rank, const int *h_n,
				       const int *h_is, const int *h_os,
				       FFTW_COMPLEX *in, FFTW_COMPLEX *out, int *sign, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_dft(*rank, dims, *howmany_rank, howmany_dims,
			   in, out, *sign, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(plan_guru_split_dft, PLAN_GURU_SPLIT_DFT)(fftw_plan *p, int *rank, const int *n,
				       const int *is, const int *os,
				       int *howmany_rank, const int *h_n,
				       const int *h_is, const int *h_os,
				       FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_split_dft(*rank, dims, *howmany_rank, howmany_dims,
			   ri, ii, ro, io, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(execute_dft, EXECUTE_DFT)(fftw_plan * const p, FFTW_COMPLEX *in, FFTW_COMPLEX *out)
{
     plan_dft *pln = (plan_dft *) (*p)->pln;
     if ((*p)->sign == FFT_SIGN)
          pln->apply((plan *) pln, in[0], in[0]+1, out[0], out[0]+1);
     else
          pln->apply((plan *) pln, in[0]+1, in[0], out[0]+1, out[0]);
}

FFTW_VOIDFUNC F77(execute_split_dft, EXECUTE_SPLIT_DFT)(fftw_plan * const p,
					       FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io)
{
     plan_dft *pln = (plan_dft *) (*p)->pln;
     pln->apply((plan *) pln, ri, ii, ro, io);
}

/****************************** DFT r2c *********************************/

FFTW_VOIDFUNC F77(plan_dft_r2c, PLAN_DFT_R2C)(fftw_plan *p, int *rank, const int *n,
				     FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     *p = fftw_plan_dft_r2c(*rank, nrev, in, out, *flags);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_dft_r2c_1d, PLAN_DFT_R2C_1D)(fftw_plan *p, int *n, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out,
					   int *flags)
{
     *p = fftw_plan_dft_r2c_1d(*n, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_r2c_2d, PLAN_DFT_R2C_2D)(fftw_plan *p, int *nx, int *ny,
					   FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, int *flags)
{
     *p = fftw_plan_dft_r2c_2d(*ny, *nx, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_r2c_3d, PLAN_DFT_R2C_3D)(fftw_plan *p,
					   int *nx, int *ny, int *nz,
					   FFTW_REAL_TYPE *in, FFTW_COMPLEX *out,
					   int *flags)
{
     *p = fftw_plan_dft_r2c_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_many_dft_r2c, PLAN_MANY_DFT_R2C)(
     fftw_plan *p, int *rank, const int *n,
     int *howmany,
     FFTW_REAL_TYPE *in, const int *inembed, int *istride, int *idist,
     FFTW_COMPLEX *out, const int *onembed, int *ostride, int *odist,
     int *flags)
{
     int *nrev = reverse_n(*rank, n);
     int *inembedrev = reverse_n(*rank, inembed);
     int *onembedrev = reverse_n(*rank, onembed);
     *p = fftw_plan_many_dft_r2c(*rank, nrev, *howmany,
			       in, inembedrev, *istride, *idist,
			       out, onembedrev, *ostride, *odist,
			       *flags);
     fftw_ifree0(onembedrev);
     fftw_ifree0(inembedrev);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_guru_dft_r2c, PLAN_GURU_DFT_R2C)(
     fftw_plan *p, int *rank, const int *n,
     const int *is, const int *os,
     int *howmany_rank, const int *h_n,
     const int *h_is, const int *h_os,
     FFTW_REAL_TYPE *in, FFTW_COMPLEX *out, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_dft_r2c(*rank, dims, *howmany_rank, howmany_dims,
			       in, out, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(plan_guru_split_dft_r2c, PLAN_GURU_SPLIT_DFT_R2C)(
     fftw_plan *p, int *rank, const int *n,
     const int *is, const int *os,
     int *howmany_rank, const int *h_n,
     const int *h_is, const int *h_os,
     FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_split_dft_r2c(*rank, dims, *howmany_rank, howmany_dims,
			       in, ro, io, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(execute_dft_r2c, EXECUTE_DFT_R2C)(fftw_plan * const p, FFTW_REAL_TYPE *in, FFTW_COMPLEX *out)
{
     plan_rdft2 *pln = (plan_rdft2 *) (*p)->pln;
     problem_rdft2 *prb = (problem_rdft2 *) (*p)->prb;
     pln->apply((plan *) pln, in, in + (prb->r1 - prb->r0), out[0], out[0]+1);
}

FFTW_VOIDFUNC F77(execute_split_dft_r2c, EXECUTE_SPLIT_DFT_R2C)(fftw_plan * const p,
						       FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *ro, FFTW_REAL_TYPE *io)
{
     plan_rdft2 *pln = (plan_rdft2 *) (*p)->pln;
     problem_rdft2 *prb = (problem_rdft2 *) (*p)->prb;
     pln->apply((plan *) pln, in, in + (prb->r1 - prb->r0), ro, io);
}

/****************************** DFT c2r *********************************/

FFTW_VOIDFUNC F77(plan_dft_c2r, PLAN_DFT_C2R)(fftw_plan *p, int *rank, const int *n,
				     FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     *p = fftw_plan_dft_c2r(*rank, nrev, in, out, *flags);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_dft_c2r_1d, PLAN_DFT_C2R_1D)(fftw_plan *p, int *n, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out,
					   int *flags)
{
     *p = fftw_plan_dft_c2r_1d(*n, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_c2r_2d, PLAN_DFT_C2R_2D)(fftw_plan *p, int *nx, int *ny,
					   FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, int *flags)
{
     *p = fftw_plan_dft_c2r_2d(*ny, *nx, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_dft_c2r_3d, PLAN_DFT_C2R_3D)(fftw_plan *p,
					   int *nx, int *ny, int *nz,
					   FFTW_COMPLEX *in, FFTW_REAL_TYPE *out,
					   int *flags)
{
     *p = fftw_plan_dft_c2r_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTW_VOIDFUNC F77(plan_many_dft_c2r, PLAN_MANY_DFT_C2R)(
     fftw_plan *p, int *rank, const int *n,
     int *howmany,
     FFTW_COMPLEX *in, const int *inembed, int *istride, int *idist,
     FFTW_REAL_TYPE *out, const int *onembed, int *ostride, int *odist,
     int *flags)
{
     int *nrev = reverse_n(*rank, n);
     int *inembedrev = reverse_n(*rank, inembed);
     int *onembedrev = reverse_n(*rank, onembed);
     *p = fftw_plan_many_dft_c2r(*rank, nrev, *howmany,
			       in, inembedrev, *istride, *idist,
			       out, onembedrev, *ostride, *odist,
			       *flags);
     fftw_ifree0(onembedrev);
     fftw_ifree0(inembedrev);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_guru_dft_c2r, PLAN_GURU_DFT_C2R)(
     fftw_plan *p, int *rank, const int *n,
     const int *is, const int *os,
     int *howmany_rank, const int *h_n,
     const int *h_is, const int *h_os,
     FFTW_COMPLEX *in, FFTW_REAL_TYPE *out, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_dft_c2r(*rank, dims, *howmany_rank, howmany_dims,
			       in, out, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(plan_guru_split_dft_c2r, PLAN_GURU_SPLIT_DFT_C2R)(
     fftw_plan *p, int *rank, const int *n,
     const int *is, const int *os,
     int *howmany_rank, const int *h_n,
     const int *h_is, const int *h_os,
     FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     *p = fftw_plan_guru_split_dft_c2r(*rank, dims, *howmany_rank, howmany_dims,
			       ri, ii, out, *flags);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(execute_dft_c2r, EXECUTE_DFT_C2R)(fftw_plan * const p, FFTW_COMPLEX *in, FFTW_REAL_TYPE *out)
{
     plan_rdft2 *pln = (plan_rdft2 *) (*p)->pln;
     problem_rdft2 *prb = (problem_rdft2 *) (*p)->prb;
     pln->apply((plan *) pln, out, out + (prb->r1 - prb->r0), in[0], in[0]+1);
}

FFTW_VOIDFUNC F77(execute_split_dft_c2r, EXECUTE_SPLIT_DFT_C2R)(fftw_plan * const p,
					   FFTW_REAL_TYPE *ri, FFTW_REAL_TYPE *ii, FFTW_REAL_TYPE *out)
{
     plan_rdft2 *pln = (plan_rdft2 *) (*p)->pln;
     problem_rdft2 *prb = (problem_rdft2 *) (*p)->prb;
     pln->apply((plan *) pln, out, out + (prb->r1 - prb->r0), ri, ii);
}

/****************************** r2r *********************************/

FFTW_VOIDFUNC F77(plan_r2r, PLAN_R2R)(fftw_plan *p, int *rank, const int *n,
			     FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
			     int *kind, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     fftw_r2r_kind *k = ints2kinds(*rank, kind);
     *p = fftw_plan_r2r(*rank, nrev, in, out, k, *flags);
     fftw_ifree0(k);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_r2r_1d, PLAN_R2R_1D)(fftw_plan *p, int *n, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
				   int *kind, int *flags)
{
     *p = fftw_plan_r2r_1d(*n, in, out, (fftw_r2r_kind) *kind, *flags);
}

FFTW_VOIDFUNC F77(plan_r2r_2d, PLAN_R2R_2D)(fftw_plan *p, int *nx, int *ny,
				   FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
				   int *kindx, int *kindy, int *flags)
{
     *p = fftw_plan_r2r_2d(*ny, *nx, in, out,
			 (fftw_r2r_kind) *kindy, (fftw_r2r_kind) *kindx, *flags);
}

FFTW_VOIDFUNC F77(plan_r2r_3d, PLAN_R2R_3D)(fftw_plan *p,
				   int *nx, int *ny, int *nz,
				   FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out,
				   int *kindx, int *kindy, int *kindz,
				   int *flags)
{
     *p = fftw_plan_r2r_3d(*nz, *ny, *nx, in, out,
			 (fftw_r2r_kind) *kindz, (fftw_r2r_kind) *kindy,
			 (fftw_r2r_kind) *kindx, *flags);
}

FFTW_VOIDFUNC F77(plan_many_r2r, PLAN_MANY_R2R)(
     fftw_plan *p, int *rank, const int *n,
     int *howmany,
     FFTW_REAL_TYPE *in, const int *inembed, int *istride, int *idist,
     FFTW_REAL_TYPE *out, const int *onembed, int *ostride, int *odist,
     int *kind, int *flags)
{
     int *nrev = reverse_n(*rank, n);
     int *inembedrev = reverse_n(*rank, inembed);
     int *onembedrev = reverse_n(*rank, onembed);
     fftw_r2r_kind *k = ints2kinds(*rank, kind);
     *p = fftw_plan_many_r2r(*rank, nrev, *howmany,
			       in, inembedrev, *istride, *idist,
			       out, onembedrev, *ostride, *odist,
			       k, *flags);
     fftw_ifree0(k);
     fftw_ifree0(onembedrev);
     fftw_ifree0(inembedrev);
     fftw_ifree0(nrev);
}

FFTW_VOIDFUNC F77(plan_guru_r2r, PLAN_GURU_R2R)(
     fftw_plan *p, int *rank, const int *n,
     const int *is, const int *os,
     int *howmany_rank, const int *h_n,
     const int *h_is, const int *h_os,
     FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out, int *kind, int *flags)
{
     fftw_iodim *dims = make_dims(*rank, n, is, os);
     fftw_iodim *howmany_dims = make_dims(*howmany_rank, h_n, h_is, h_os);
     fftw_r2r_kind *k = ints2kinds(*rank, kind);
     *p = fftw_plan_guru_r2r(*rank, dims, *howmany_rank, howmany_dims,
			       in, out, k, *flags);
     fftw_ifree0(k);
     fftw_ifree0(howmany_dims);
     fftw_ifree0(dims);
}

FFTW_VOIDFUNC F77(execute_r2r, EXECUTE_R2R)(fftw_plan * const p, FFTW_REAL_TYPE *in, FFTW_REAL_TYPE *out)
{
     plan_rdft *pln = (plan_rdft *) (*p)->pln;
     pln->apply((plan *) pln, in, out);
}

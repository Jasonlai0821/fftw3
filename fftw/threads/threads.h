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

#ifndef __THREADS_H__
#define __THREADS_H__

#include "fftw/fftw_api.h"
#include "fftw/dft.h"
#include "fftw/rdft.h"

typedef struct {
     int min, max, thr_num;
     void *data;
} spawn_data;

typedef void *(*spawn_function) (spawn_data *);

void fftw_spawn_loop(int loopmax, int nthreads,
		   spawn_function proc, void *data);
int fftw_ithreads_init(void);
void fftw_threads_cleanup(void);

/* configurations */

void fftw_dft_thr_vrank_geq1_register(planner *p);
void fftw_rdft_thr_vrank_geq1_register(planner *p);
void fftw_rdft2_thr_vrank_geq1_register(planner *p);

ct_solver *fftw_mksolver_ct_threads(size_t size, INT r, int dec,
				  ct_mkinferior mkcldw,
				  ct_force_vrecursion force_vrecursionp);
hc2hc_solver *fftw_mksolver_hc2hc_threads(size_t size, INT r, hc2hc_mkinferior mkcldw);

void fftw_threads_conf_standard(planner *p);
void fftw_threads_register_hooks(void);
void fftw_threads_unregister_hooks(void);
void fftw_threads_register_planner_hooks(void);
                                      
#endif /* __THREADS_H__ */

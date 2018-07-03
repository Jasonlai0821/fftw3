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
#include "threads.h"

static int threads_inited = 0;

static void threads_register_hooks(void)
{
     fftw_mksolver_ct_hook = fftw_mksolver_ct_threads;
     fftw_mksolver_hc2hc_hook = fftw_mksolver_hc2hc_threads;
}

static void threads_unregister_hooks(void)
{
     fftw_mksolver_ct_hook = 0;
     fftw_mksolver_hc2hc_hook = 0;
}

/* should be called before all other FFTW functions! */
int fftw_init_threads(void)
{
     if (!threads_inited) {
	  planner *plnr;

          if (fftw_ithreads_init())
               return 0;

	  threads_register_hooks();

	  /* this should be the first time the_planner is called,
	     and hence the time it is configured */
	  plnr = fftw_the_planner();
	  fftw_threads_conf_standard(plnr);
	       
          threads_inited = 1;
     }
     return 1;
}


void fftw_cleanup_threads(void)
{
     fftw_cleanup();
     if (threads_inited) {
	  fftw_threads_cleanup();
	  threads_unregister_hooks();
	  threads_inited = 0;
     }
}

void fftw_plan_with_nthreads(int nthreads)
{
     planner *plnr;

     if (!threads_inited) {
	  fftw_cleanup();
	  fftw_init_threads();
     }
     A(threads_inited);
     plnr = fftw_the_planner();
     plnr->nthr = fftw_imax(1, nthreads);
}

void fftw_make_planner_thread_safe(void)
{
     fftw_threads_register_planner_hooks();
}

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

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Thu May 24 08:04:29 EDT 2018 */

#include "fftw/fftw_api.h"

#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)

/* Generated by: ../../../genfft/gen_twidsq.native -fma -compact -variables 4 -pipeline-latency 4 -reload-twiddle -dif -n 4 -name q1_4 -include fftw/rdft_scalar/q.h */

/*
 * This function contains 88 FP additions, 48 FP multiplications,
 * (or, 64 additions, 24 multiplications, 24 fused multiply/add),
 * 51 stack variables, 0 constants, and 64 memory accesses
 */
#include "fftw/rdft_scalar/q.h"

static void q1_4(FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio, const FFTW_REAL_TYPE *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + (mb * 6); m < me; m = m + 1, rio = rio + ms, iio = iio + ms, W = W + 6, MAKE_VOLATILE_STRIDE(8, rs), MAKE_VOLATILE_STRIDE(0, vs)) {
	       E T3, Tv, Tw, T6, Tc, Tf, Tx, Ts, Tm, Ti, T1H, T29, T2a, T1K, T1Q;
	       E T1T, T2b, T26, T20, T1W, TB, T13, T14, TE, TK, TN, T15, T10, TU, TQ;
	       E T19, T1B, T1C, T1c, T1i, T1l, T1D, T1y, T1s, T1o;
	       {
		    E T1, T2, Tb, Tg, Th, T8;
		    {
			 E T9, Ta, T4, T5;
			 T1 = rio[0];
			 T2 = rio[WS(rs, 2)];
			 T3 = T1 + T2;
			 T9 = iio[0];
			 Ta = iio[WS(rs, 2)];
			 Tb = T9 - Ta;
			 Tv = T9 + Ta;
			 Tg = iio[WS(rs, 1)];
			 Th = iio[WS(rs, 3)];
			 Tw = Tg + Th;
			 T4 = rio[WS(rs, 1)];
			 T5 = rio[WS(rs, 3)];
			 T6 = T4 + T5;
			 T8 = T4 - T5;
		    }
		    Tc = T8 + Tb;
		    Tf = T1 - T2;
		    Tx = Tv - Tw;
		    Ts = T3 - T6;
		    Tm = Tb - T8;
		    Ti = Tg - Th;
	       }
	       {
		    E T1F, T1G, T1P, T1U, T1V, T1M;
		    {
			 E T1N, T1O, T1I, T1J;
			 T1F = rio[WS(vs, 3)];
			 T1G = rio[WS(vs, 3) + WS(rs, 2)];
			 T1H = T1F + T1G;
			 T1N = iio[WS(vs, 3)];
			 T1O = iio[WS(vs, 3) + WS(rs, 2)];
			 T1P = T1N - T1O;
			 T29 = T1N + T1O;
			 T1U = iio[WS(vs, 3) + WS(rs, 1)];
			 T1V = iio[WS(vs, 3) + WS(rs, 3)];
			 T2a = T1U + T1V;
			 T1I = rio[WS(vs, 3) + WS(rs, 1)];
			 T1J = rio[WS(vs, 3) + WS(rs, 3)];
			 T1K = T1I + T1J;
			 T1M = T1I - T1J;
		    }
		    T1Q = T1M + T1P;
		    T1T = T1F - T1G;
		    T2b = T29 - T2a;
		    T26 = T1H - T1K;
		    T20 = T1P - T1M;
		    T1W = T1U - T1V;
	       }
	       {
		    E Tz, TA, TJ, TO, TP, TG;
		    {
			 E TH, TI, TC, TD;
			 Tz = rio[WS(vs, 1)];
			 TA = rio[WS(vs, 1) + WS(rs, 2)];
			 TB = Tz + TA;
			 TH = iio[WS(vs, 1)];
			 TI = iio[WS(vs, 1) + WS(rs, 2)];
			 TJ = TH - TI;
			 T13 = TH + TI;
			 TO = iio[WS(vs, 1) + WS(rs, 1)];
			 TP = iio[WS(vs, 1) + WS(rs, 3)];
			 T14 = TO + TP;
			 TC = rio[WS(vs, 1) + WS(rs, 1)];
			 TD = rio[WS(vs, 1) + WS(rs, 3)];
			 TE = TC + TD;
			 TG = TC - TD;
		    }
		    TK = TG + TJ;
		    TN = Tz - TA;
		    T15 = T13 - T14;
		    T10 = TB - TE;
		    TU = TJ - TG;
		    TQ = TO - TP;
	       }
	       {
		    E T17, T18, T1h, T1m, T1n, T1e;
		    {
			 E T1f, T1g, T1a, T1b;
			 T17 = rio[WS(vs, 2)];
			 T18 = rio[WS(vs, 2) + WS(rs, 2)];
			 T19 = T17 + T18;
			 T1f = iio[WS(vs, 2)];
			 T1g = iio[WS(vs, 2) + WS(rs, 2)];
			 T1h = T1f - T1g;
			 T1B = T1f + T1g;
			 T1m = iio[WS(vs, 2) + WS(rs, 1)];
			 T1n = iio[WS(vs, 2) + WS(rs, 3)];
			 T1C = T1m + T1n;
			 T1a = rio[WS(vs, 2) + WS(rs, 1)];
			 T1b = rio[WS(vs, 2) + WS(rs, 3)];
			 T1c = T1a + T1b;
			 T1e = T1a - T1b;
		    }
		    T1i = T1e + T1h;
		    T1l = T17 - T18;
		    T1D = T1B - T1C;
		    T1y = T19 - T1c;
		    T1s = T1h - T1e;
		    T1o = T1m - T1n;
	       }
	       rio[0] = T3 + T6;
	       iio[0] = Tv + Tw;
	       rio[WS(rs, 1)] = TB + TE;
	       iio[WS(rs, 1)] = T13 + T14;
	       rio[WS(rs, 2)] = T19 + T1c;
	       iio[WS(rs, 2)] = T1B + T1C;
	       iio[WS(rs, 3)] = T29 + T2a;
	       rio[WS(rs, 3)] = T1H + T1K;
	       {
		    E Tt, Ty, Tr, Tu;
		    Tr = W[2];
		    Tt = Tr * Ts;
		    Ty = Tr * Tx;
		    Tu = W[3];
		    rio[WS(vs, 2)] = FMA(Tu, Tx, Tt);
		    iio[WS(vs, 2)] = FNMS(Tu, Ts, Ty);
	       }
	       {
		    E T27, T2c, T25, T28;
		    T25 = W[2];
		    T27 = T25 * T26;
		    T2c = T25 * T2b;
		    T28 = W[3];
		    rio[WS(vs, 2) + WS(rs, 3)] = FMA(T28, T2b, T27);
		    iio[WS(vs, 2) + WS(rs, 3)] = FNMS(T28, T26, T2c);
	       }
	       {
		    E T11, T16, TZ, T12;
		    TZ = W[2];
		    T11 = TZ * T10;
		    T16 = TZ * T15;
		    T12 = W[3];
		    rio[WS(vs, 2) + WS(rs, 1)] = FMA(T12, T15, T11);
		    iio[WS(vs, 2) + WS(rs, 1)] = FNMS(T12, T10, T16);
	       }
	       {
		    E T1z, T1E, T1x, T1A;
		    T1x = W[2];
		    T1z = T1x * T1y;
		    T1E = T1x * T1D;
		    T1A = W[3];
		    rio[WS(vs, 2) + WS(rs, 2)] = FMA(T1A, T1D, T1z);
		    iio[WS(vs, 2) + WS(rs, 2)] = FNMS(T1A, T1y, T1E);
	       }
	       {
		    E Tj, Te, Tk, T7, Td;
		    Tj = Tf - Ti;
		    Te = W[5];
		    Tk = Te * Tc;
		    T7 = W[4];
		    Td = T7 * Tc;
		    iio[WS(vs, 3)] = FNMS(Te, Tj, Td);
		    rio[WS(vs, 3)] = FMA(T7, Tj, Tk);
	       }
	       {
		    E T1p, T1k, T1q, T1d, T1j;
		    T1p = T1l - T1o;
		    T1k = W[5];
		    T1q = T1k * T1i;
		    T1d = W[4];
		    T1j = T1d * T1i;
		    iio[WS(vs, 3) + WS(rs, 2)] = FNMS(T1k, T1p, T1j);
		    rio[WS(vs, 3) + WS(rs, 2)] = FMA(T1d, T1p, T1q);
	       }
	       {
		    E T23, T22, T24, T1Z, T21;
		    T23 = T1T + T1W;
		    T22 = W[1];
		    T24 = T22 * T20;
		    T1Z = W[0];
		    T21 = T1Z * T20;
		    iio[WS(vs, 1) + WS(rs, 3)] = FNMS(T22, T23, T21);
		    rio[WS(vs, 1) + WS(rs, 3)] = FMA(T1Z, T23, T24);
	       }
	       {
		    E TX, TW, TY, TT, TV;
		    TX = TN + TQ;
		    TW = W[1];
		    TY = TW * TU;
		    TT = W[0];
		    TV = TT * TU;
		    iio[WS(vs, 1) + WS(rs, 1)] = FNMS(TW, TX, TV);
		    rio[WS(vs, 1) + WS(rs, 1)] = FMA(TT, TX, TY);
	       }
	       {
		    E TR, TM, TS, TF, TL;
		    TR = TN - TQ;
		    TM = W[5];
		    TS = TM * TK;
		    TF = W[4];
		    TL = TF * TK;
		    iio[WS(vs, 3) + WS(rs, 1)] = FNMS(TM, TR, TL);
		    rio[WS(vs, 3) + WS(rs, 1)] = FMA(TF, TR, TS);
	       }
	       {
		    E Tp, To, Tq, Tl, Tn;
		    Tp = Tf + Ti;
		    To = W[1];
		    Tq = To * Tm;
		    Tl = W[0];
		    Tn = Tl * Tm;
		    iio[WS(vs, 1)] = FNMS(To, Tp, Tn);
		    rio[WS(vs, 1)] = FMA(Tl, Tp, Tq);
	       }
	       {
		    E T1v, T1u, T1w, T1r, T1t;
		    T1v = T1l + T1o;
		    T1u = W[1];
		    T1w = T1u * T1s;
		    T1r = W[0];
		    T1t = T1r * T1s;
		    iio[WS(vs, 1) + WS(rs, 2)] = FNMS(T1u, T1v, T1t);
		    rio[WS(vs, 1) + WS(rs, 2)] = FMA(T1r, T1v, T1w);
	       }
	       {
		    E T1X, T1S, T1Y, T1L, T1R;
		    T1X = T1T - T1W;
		    T1S = W[5];
		    T1Y = T1S * T1Q;
		    T1L = W[4];
		    T1R = T1L * T1Q;
		    iio[WS(vs, 3) + WS(rs, 3)] = FNMS(T1S, T1X, T1R);
		    rio[WS(vs, 3) + WS(rs, 3)] = FMA(T1L, T1X, T1Y);
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 0, 4},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 4, "q1_4", twinstr, &GENUS, {64, 24, 24, 0}, 0, 0, 0 };

void fftw_codelet_q1_4 (planner *p) {
     fftw_kdft_difsq_register (p, q1_4, &desc);
}
#else

/* Generated by: ../../../genfft/gen_twidsq.native -compact -variables 4 -pipeline-latency 4 -reload-twiddle -dif -n 4 -name q1_4 -include fftw/rdft_scalar/q.h */

/*
 * This function contains 88 FP additions, 48 FP multiplications,
 * (or, 64 additions, 24 multiplications, 24 fused multiply/add),
 * 37 stack variables, 0 constants, and 64 memory accesses
 */
#include "fftw/dft_scalar/q.h"

static void q1_4(FFTW_REAL_TYPE *rio, FFTW_REAL_TYPE *iio, const FFTW_REAL_TYPE *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + (mb * 6); m < me; m = m + 1, rio = rio + ms, iio = iio + ms, W = W + 6, MAKE_VOLATILE_STRIDE(8, rs), MAKE_VOLATILE_STRIDE(0, vs)) {
	       E T3, Te, Tb, Tq, T6, T8, Th, Tr, Tv, TG, TD, TS, Ty, TA, TJ;
	       E TT, TX, T18, T15, T1k, T10, T12, T1b, T1l, T1p, T1A, T1x, T1M, T1s, T1u;
	       E T1D, T1N;
	       {
		    E T1, T2, T9, Ta;
		    T1 = rio[0];
		    T2 = rio[WS(rs, 2)];
		    T3 = T1 + T2;
		    Te = T1 - T2;
		    T9 = iio[0];
		    Ta = iio[WS(rs, 2)];
		    Tb = T9 - Ta;
		    Tq = T9 + Ta;
	       }
	       {
		    E T4, T5, Tf, Tg;
		    T4 = rio[WS(rs, 1)];
		    T5 = rio[WS(rs, 3)];
		    T6 = T4 + T5;
		    T8 = T4 - T5;
		    Tf = iio[WS(rs, 1)];
		    Tg = iio[WS(rs, 3)];
		    Th = Tf - Tg;
		    Tr = Tf + Tg;
	       }
	       {
		    E Tt, Tu, TB, TC;
		    Tt = rio[WS(vs, 1)];
		    Tu = rio[WS(vs, 1) + WS(rs, 2)];
		    Tv = Tt + Tu;
		    TG = Tt - Tu;
		    TB = iio[WS(vs, 1)];
		    TC = iio[WS(vs, 1) + WS(rs, 2)];
		    TD = TB - TC;
		    TS = TB + TC;
	       }
	       {
		    E Tw, Tx, TH, TI;
		    Tw = rio[WS(vs, 1) + WS(rs, 1)];
		    Tx = rio[WS(vs, 1) + WS(rs, 3)];
		    Ty = Tw + Tx;
		    TA = Tw - Tx;
		    TH = iio[WS(vs, 1) + WS(rs, 1)];
		    TI = iio[WS(vs, 1) + WS(rs, 3)];
		    TJ = TH - TI;
		    TT = TH + TI;
	       }
	       {
		    E TV, TW, T13, T14;
		    TV = rio[WS(vs, 2)];
		    TW = rio[WS(vs, 2) + WS(rs, 2)];
		    TX = TV + TW;
		    T18 = TV - TW;
		    T13 = iio[WS(vs, 2)];
		    T14 = iio[WS(vs, 2) + WS(rs, 2)];
		    T15 = T13 - T14;
		    T1k = T13 + T14;
	       }
	       {
		    E TY, TZ, T19, T1a;
		    TY = rio[WS(vs, 2) + WS(rs, 1)];
		    TZ = rio[WS(vs, 2) + WS(rs, 3)];
		    T10 = TY + TZ;
		    T12 = TY - TZ;
		    T19 = iio[WS(vs, 2) + WS(rs, 1)];
		    T1a = iio[WS(vs, 2) + WS(rs, 3)];
		    T1b = T19 - T1a;
		    T1l = T19 + T1a;
	       }
	       {
		    E T1n, T1o, T1v, T1w;
		    T1n = rio[WS(vs, 3)];
		    T1o = rio[WS(vs, 3) + WS(rs, 2)];
		    T1p = T1n + T1o;
		    T1A = T1n - T1o;
		    T1v = iio[WS(vs, 3)];
		    T1w = iio[WS(vs, 3) + WS(rs, 2)];
		    T1x = T1v - T1w;
		    T1M = T1v + T1w;
	       }
	       {
		    E T1q, T1r, T1B, T1C;
		    T1q = rio[WS(vs, 3) + WS(rs, 1)];
		    T1r = rio[WS(vs, 3) + WS(rs, 3)];
		    T1s = T1q + T1r;
		    T1u = T1q - T1r;
		    T1B = iio[WS(vs, 3) + WS(rs, 1)];
		    T1C = iio[WS(vs, 3) + WS(rs, 3)];
		    T1D = T1B - T1C;
		    T1N = T1B + T1C;
	       }
	       rio[0] = T3 + T6;
	       iio[0] = Tq + Tr;
	       rio[WS(rs, 1)] = Tv + Ty;
	       iio[WS(rs, 1)] = TS + TT;
	       rio[WS(rs, 2)] = TX + T10;
	       iio[WS(rs, 2)] = T1k + T1l;
	       iio[WS(rs, 3)] = T1M + T1N;
	       rio[WS(rs, 3)] = T1p + T1s;
	       {
		    E Tc, Ti, T7, Td;
		    Tc = T8 + Tb;
		    Ti = Te - Th;
		    T7 = W[4];
		    Td = W[5];
		    iio[WS(vs, 3)] = FNMS(Td, Ti, T7 * Tc);
		    rio[WS(vs, 3)] = FMA(Td, Tc, T7 * Ti);
	       }
	       {
		    E T1K, T1O, T1J, T1L;
		    T1K = T1p - T1s;
		    T1O = T1M - T1N;
		    T1J = W[2];
		    T1L = W[3];
		    rio[WS(vs, 2) + WS(rs, 3)] = FMA(T1J, T1K, T1L * T1O);
		    iio[WS(vs, 2) + WS(rs, 3)] = FNMS(T1L, T1K, T1J * T1O);
	       }
	       {
		    E Tk, Tm, Tj, Tl;
		    Tk = Tb - T8;
		    Tm = Te + Th;
		    Tj = W[0];
		    Tl = W[1];
		    iio[WS(vs, 1)] = FNMS(Tl, Tm, Tj * Tk);
		    rio[WS(vs, 1)] = FMA(Tl, Tk, Tj * Tm);
	       }
	       {
		    E To, Ts, Tn, Tp;
		    To = T3 - T6;
		    Ts = Tq - Tr;
		    Tn = W[2];
		    Tp = W[3];
		    rio[WS(vs, 2)] = FMA(Tn, To, Tp * Ts);
		    iio[WS(vs, 2)] = FNMS(Tp, To, Tn * Ts);
	       }
	       {
		    E T16, T1c, T11, T17;
		    T16 = T12 + T15;
		    T1c = T18 - T1b;
		    T11 = W[4];
		    T17 = W[5];
		    iio[WS(vs, 3) + WS(rs, 2)] = FNMS(T17, T1c, T11 * T16);
		    rio[WS(vs, 3) + WS(rs, 2)] = FMA(T17, T16, T11 * T1c);
	       }
	       {
		    E T1G, T1I, T1F, T1H;
		    T1G = T1x - T1u;
		    T1I = T1A + T1D;
		    T1F = W[0];
		    T1H = W[1];
		    iio[WS(vs, 1) + WS(rs, 3)] = FNMS(T1H, T1I, T1F * T1G);
		    rio[WS(vs, 1) + WS(rs, 3)] = FMA(T1H, T1G, T1F * T1I);
	       }
	       {
		    E TQ, TU, TP, TR;
		    TQ = Tv - Ty;
		    TU = TS - TT;
		    TP = W[2];
		    TR = W[3];
		    rio[WS(vs, 2) + WS(rs, 1)] = FMA(TP, TQ, TR * TU);
		    iio[WS(vs, 2) + WS(rs, 1)] = FNMS(TR, TQ, TP * TU);
	       }
	       {
		    E T1e, T1g, T1d, T1f;
		    T1e = T15 - T12;
		    T1g = T18 + T1b;
		    T1d = W[0];
		    T1f = W[1];
		    iio[WS(vs, 1) + WS(rs, 2)] = FNMS(T1f, T1g, T1d * T1e);
		    rio[WS(vs, 1) + WS(rs, 2)] = FMA(T1f, T1e, T1d * T1g);
	       }
	       {
		    E T1i, T1m, T1h, T1j;
		    T1i = TX - T10;
		    T1m = T1k - T1l;
		    T1h = W[2];
		    T1j = W[3];
		    rio[WS(vs, 2) + WS(rs, 2)] = FMA(T1h, T1i, T1j * T1m);
		    iio[WS(vs, 2) + WS(rs, 2)] = FNMS(T1j, T1i, T1h * T1m);
	       }
	       {
		    E T1y, T1E, T1t, T1z;
		    T1y = T1u + T1x;
		    T1E = T1A - T1D;
		    T1t = W[4];
		    T1z = W[5];
		    iio[WS(vs, 3) + WS(rs, 3)] = FNMS(T1z, T1E, T1t * T1y);
		    rio[WS(vs, 3) + WS(rs, 3)] = FMA(T1z, T1y, T1t * T1E);
	       }
	       {
		    E TM, TO, TL, TN;
		    TM = TD - TA;
		    TO = TG + TJ;
		    TL = W[0];
		    TN = W[1];
		    iio[WS(vs, 1) + WS(rs, 1)] = FNMS(TN, TO, TL * TM);
		    rio[WS(vs, 1) + WS(rs, 1)] = FMA(TN, TM, TL * TO);
	       }
	       {
		    E TE, TK, Tz, TF;
		    TE = TA + TD;
		    TK = TG - TJ;
		    Tz = W[4];
		    TF = W[5];
		    iio[WS(vs, 3) + WS(rs, 1)] = FNMS(TF, TK, Tz * TE);
		    rio[WS(vs, 3) + WS(rs, 1)] = FMA(TF, TE, Tz * TK);
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 0, 4},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 4, "q1_4", twinstr, &GENUS, {64, 24, 24, 0}, 0, 0, 0 };

void fftw_codelet_q1_4 (planner *p) {
     fftw_kdft_difsq_register (p, q1_4, &desc);
}
#endif
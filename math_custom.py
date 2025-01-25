import pdb
import numpy as np
from math import *
import random
import scipy as scp
from scipy import optimize

import datetime
import copy
import os
import sys
import warnings
import pprint as pp


from mc_base import *
from mc_calendar import *
from mc_dec import *
from mc_gen import *
from mc_list import *
from mc_poly import *
from mc_splines import *
from mc_trig import *
from mc_vectors import *
from mc_viz import *

#warnings.filterwarnings('error')

#================this module contains custom math functions, used mainly to accelerate and/or simplify various calculations
#================specific functions are found in imported mc_ modules
#================general constants, basic and auxiliary functions, used in most other functions, ae found in mc_base module which is imported into each mc_ module
#================in math_custom.py: functions which require multiple other modules


#=============================


#sliding-window averaging for simple XY table data; 
#window = (x_central - hw, x_central + hw), xs = ys indices if xs not specified
#degree: 0 - simple averaging, 1 - linear fitting, (fast) >= 2: polynomial fitting (~Savitsky-Golay filter, but with allowed variable spacing between xs, slow)

def smoother(ys=arr_dummy, xs=arr_dummy, hw=1, deg=0, n_iter=1, ftype=''):  #(+)

 if n_iter > 1 :
  smoothie = np.copy(ys)
  for n_i in range(n_iter, 0, -1):
   smoothie = smoother(smoothie, xs=xs, hw=hw, deg=deg, n_iter=1, ftype=ftype)
  return smoothie
 n_pts = ys.size
 if xs.size != n_pts:
  xs = np.arange(n_pts)
 
 fin_mask = np.isfinite(ys) #
 nan_mask = True ^ fin_mask
 xs_f, ys_f, xs_nan = xs[fin_mask], ys[fin_mask], xs[nan_mask]
 if deg == 0 or (deg == 1 and 'poly' not in ftype):
  ys_avg = 0.5*(ys_f[1:] + ys_f[:-1])
  y_int_f = np.hstack((0.0, np.cumsum(np.diff(xs_f) * ys_avg) ))
  y_int_0 = np.full_like(ys, nan)
  y_int_0[fin_mask] = y_int_f
  y_int = nan_interp(xs, y_int_0)
  is_l = np.searchsorted(xs, xs - hw)
  is_r = np.searchsorted(xs, xs + hw) - 1
  is_m = np.arange(n_pts)
  if deg == 0:
   smoothie = (y_int[is_r] - y_int[is_l]) / (xs[is_r] - xs[is_l])
  elif deg == 1:
   xs_l, xs_r = xs[is_l], xs[is_r]
   xs_m = xs[is_m]
   mask_edge = np.where( (is_l== 0) + (is_r==n_pts-1), True, False)
   is_edge = np.arange(n_pts)[mask_edge]
   xs_m_edge = 0.5*(xs_l[is_edge] + xs_r[is_edge])
   is_m_edge = np.searchsorted(xs, xs_m_edge)
   is_m[mask_edge] = is_m_edge
   xs_m = xs[is_m]   
   y_int_l, y_int_m, y_int_r = y_int[is_l], y_int[is_m], y_int[is_r]
   xs_arr = np.array((xs_l, xs_m, xs_r)).T
   ys_arr = np.array((y_int_l, y_int_m, y_int_r)).T
   sqr_appr = sqr_calc_arr(xs_arr, ys_arr).T #c + bx + a*x**2
   ks, bs = 2*sqr_appr[2], sqr_appr[1]
   smoothie = ks*xs + bs
 else: 
  smoothie = np.zeros_like(ys)
  for i_d in range(xs.size):
   x_0 = xs[i_d]
   is_IR = np.where( (xs <= x_0 + hw) * (xs >= x_0 - hw) * fin_mask)[0]
   xs_IR, ys_IR = xs[is_IR], ys[is_IR] 
   smoothie[i_d] = Poly.fit(xs_IR, ys_IR, deg)(x_0) if is_IR.size > deg else ys[i_d]
	 
 return smoothie


#----------------solving f(x) == 0-------------------


#slow, do not use for very fast functions in long loops
#deg > 1 usually not useful because of very uneven separation between final x values
#oscillates wildly about extremum if y_val outside of it, add convergence condition
def newton_raphson(func, y_val, xs_start=(0.0, 1.0), prec=1e-4, ctr_lim=30, x_lims=(-inf, inf), ftype='', deg=1): #(+) #200k iters/s
 f_types = ftype.split(' ')
 x_l1, x_l2 = xs_start
 diff_start = abs(x_l2 - x_l1)
 y_l1, y_l2 = func(x_l1), func(x_l2)
 xs, ys = np.zeros(ctr_lim+1)*nan, np.zeros(ctr_lim+1)*nan
 xs[:2] = x_l1, x_l2
 ys[:2] = y_l1, y_l2
 cont_calc=True
 ctr=1
 while cont_calc:
  ctr += 1	 
  xn = x_l2 + (x_l1 - x_l2) * (y_val - y_l2) / (y_l1 - y_l2)
  yn = func(xn)
  xs[ctr] = xn
  ys[ctr] = yn
  x_l2, x_l1 = xs[ctr-1:ctr+1]
  y_l2, y_l1 = ys[ctr-1:ctr+1]
  prec_cur = abs((xn - x_l2) / diff_start)
  if ctr >= ctr_lim or prec_cur < prec or not inrange(xn, x_lims):
   break
 
 if deg == 1:
  return (xn, arr_stack((xs[:ctr+1], ys[:ctr+1]), 'v').T) if 'log' in ftype else xn
 
 inds_sort = np.argsort(np.abs(ys-y_val)[:ctr+1])[::-1] #sort by NR parameter, to exclude recent outlying points if present
 xs_fin = xs[inds_sort][-deg-1:] 
 ys_fin = ys[inds_sort][-deg-1:]
 y_poly = poly_calc(xs_fin, ys_fin)
 x_roots = poly_solve(y_poly, y_val, x_0=xn)
 xn = x_roots[0] if x_roots.size > 0 else xn
 return (x_n, arr_stack((xs_fin, ys_fin), 'v').T) if 'log' in ftype else xn
 
    

#binary search on interval x1...x3
#returns x value where func == y_val, with precision == thres

def bin_search(func, y_val, x_range=(0.0,1.0), prec=1e-6, ftype='', deg=1, range_expand=1.5, max_ext=1e20, ctr_lim=100): 
 x_l, x_r = x_range
 
 f_types = ftype.split(' ')
 delta_x = diff_start = abs(x_r - x_l) 

 y_l, y_r = func(x_l), func(x_r)
 yd_l, yd_r = y_l - y_val, y_r - y_val
 xs, ys = [x_l, x_r], [y_l, y_r]
 cond = True
 
 #---------------range extension if needed and indicated by ftype 
 
 if signum(yd_l) == signum(yd_r):
  if 'ext' not in f_types:
   x_val = nan
   cond = False
  else:
   diff_rel = 1.0 #relative difference, 1.0 @ the start
   while signum(yd_l) == signum(yd_r) and abs(delta_x / diff_start) < max_ext and any((yd_l != 0, yd_r != 0)): #x_l if abs(yd_l) < abs(yd_r) else x_r
    delta_x = range_expand * delta_x
    if abs(yd_l) < abs(yd_r): #move to the left
     x_l, x_r = x_l - delta_x ,x_l
     y_l, y_r = func(x_l), y_l
     xs.append(x_l)
     ys.append(y_l)
    else: # abs(yd_l) > abs(yd_r): #move to the right
     x_l, x_r = x_r, x_r + delta_x
     y_l, y_r = y_r, func(x_r)
     xs.append(x_r)
     ys.append(y_r)
    yd_l, yd_r = y_l - y_val, y_r - y_val
 
 #---------------main search ----------
 
 ctr=0
 while cond:
  x_m = 0.5*(x_l+x_r)
  y_m = func(x_m)
  xs.append(x_m)
  ys.append(y_m)
  yd_m = y_m - y_val
  if signum(yd_l) == signum(yd_m): # (pt2 -> pt1)
   x_l, y_l, yd_l = x_m, y_m, yd_m
  else: #   signum(yd_m) == signum(yd_r):
   x_r, y_r, yd_r = x_m, y_m, yd_m 
  ctr += 1
 
  #---break conditions
 
  diff_rel = abs((x_r - x_l)/diff_start)#  if 'print' in ftype:#   print(ctr, '%.6e' % x_l, '%.6e' % x_m, '%.6e' % x_r, '%.2e' % diff, '\t', '%.6e' % y_l, '%.6e' % y_m, '%.6e' % y_r) #  print(ctr, x_l, x_m, x_r, y_l, y_m, y_r)  
  if any((yd_l == 0, yd_r == 0, yd_m == 0)):
   x_val = x_l if yd_l == 0 else (x_m if yd_m == 0 else x_r)
   cond = False
  if diff_rel < prec:
   break
  if ctr > ctr_lim or signum(yd_l) == signum(yd_r) and yd_l != 0:
   x_val = nan
   cond = False
 #-------------final calculation-------------- 

 if cond:
  x_3, x_2, x_1 = xs[-3:]
  y_3, y_2, y_1 = ys[-3:]
  yd_3, yd_2, yd_1 = y_3 - y_val, y_2 - y_val, y_1 - y_val
  if deg == 0:
   x_val = x_1
  elif deg == 1:
   x_val = x_1 - yd_1 * (x_1 - x_2) / (y_1 - y_2)
  else:
   xs_fin, yds_fin = np.array((x_3, x_2, x_1)), np.array((yd_3, yd_2, yd_1))
   sqr_last = sqr_calc(xs_fin, yds_fin)
   r_1, r_2 = sqr_solve(sqr_last, 0)
   x_val = r_1 if abs(r_1 - x_m) < abs(r_2 - x_m) else r_2

 if 'log' in f_types:
  xs_out = np.array(xs)
  ys_out = np.array(ys)
  return x_val, np.array((xs_out, ys_out)).T
 else:
  return x_val



#-----------------------probability distribution function calculations------------------


def prob_int_calc(vals, ws=arr_dummy, close_thr=0.1):
 n_pts = vals.size
 ws = ws if ws.size == n_pts else np.ones_like(vals) / n_pts
 i_srt = np.argsort(vals)
 vals_srt = vals[i_srt]
 ws_srt = ws[i_srt]
 prob_int = np.cumsum(ws_srt) / ws_srt.sum() #np.arange(n_pts) / (n_pts - 1)
 mask_cl = close_pt_detect(vals_srt, thr=close_thr)
 vals_un, prob_int_un = vals_srt[mask_cl], prob_int[mask_cl]
 return vals_un, prob_int_un

#sm_probs: probability integral smoothing parameters
#use fixed number of points for calculation at edges and for close-to-unique data in the middle
#fixed hw: bad at the edges if hw < pt separation, bad in the middle if hw > ~0.3* rsd
def prob_int_smooth(vs_srt, prob_int, sm_probs = (-0.5, 1, 3)):
 hw_par, deg, n_iter = sm_probs #smooth window halfwidth parameter, smoothing polynomial degree, number of passes
 if isinstance(hw_par, int): #set smooth window with constant number of points
  vs_sm = smoother(ys=vs_srt, hw=hw_par, deg=deg, n_iter=n_iter)
  prob_int_sm = prob_int
 else:
  if hw_par < 0: #smooth window = multiple of rsd; use only if at least almost all points in source array are unique (else rsd is overestimated)
   val_rsd = vs_srt.std()
   prob_hw = hw_par * val_rsd * -1.0
  else:
   prob_hw = hw_par
  prob_int_sm = smoother(xs=vs_srt, ys=prob_int, hw=prob_hw, deg=deg, n_iter=n_iter)
  vs_sm = vs_srt
 return vs_sm, prob_int_sm

def kde_custom_test(vs_0, sm_probs=(5,1,3), step=100, ftype='clf'):
 vs = vs_0[::step]
 vs_srt, prob_int = prob_int_calc(vs)
 vs_sm = prob_int_sm(vs_srt, prob_int, sm_probs=sm_probs)
 probs_sm = np.diff(prob_int) / np.diff(vs_sm)
 vs_min, vs_max = vs.min(), vs.max()
 vs_span, vs_mid = vs_max - vs_min, 0.5*(vs_min + vs_max)
 vs_range = (vs_mid - 0.7*vs_span, vs_mid + 0.7 * vs_span)
 if 'clf' in ftype:
  plt.clf()
 h, b, p = plt.hist(vs_0, bins=100, range=vs_range, density=True)
 plt.plot(vs_sm, prob_int)
 plt.plot(vs_srt, prob_int)
 plt.plot(vs_sm[1:], probs_sm)
 plt.grid(True)
 return None

 
def prob_solve_lin(xs, prob_int, thr): 
 i_pr = max(np.where(prob_int > thr)[0][0], 1)
 x_l, x_r = xs[i_pr-1], xs[i_pr]
 pr_l, pr_r = prob_int[i_pr-1], prob_int[i_pr]
 x = x_l + (x_r - x_l) * (thr - pr_l) / (pr_r - pr_l)
 return x


def prob_solve(xs, prob_int, thr):
 if thr < 0.5:
  lthr = log10(thr)
  lpi = np.log10(prob_int)
  i_pr = max(np.where(lpi > lthr)[0][0], 1)
 else:
  lthr = log10(1 - thr)
  lpi = np.log10(1 - prob_int)
  i_pr = min(np.where(lpi > lthr)[0][-1], xs.size-2)
 x_l, x_r = xs[i_pr-1], xs[i_pr]
 pr_l, pr_r = lpi[i_pr-1], lpi[i_pr]
 x = x_l + (x_r - x_l) * (lthr - pr_l) / (pr_r - pr_l)
 return x


#------------------------math function testing-------------------------

def func_test(func_arr=None, func=None, x_range=(-10,10), n_pts=101, xs=arr_dummy, ftype='clf'):
 plt.ion()
 plt.show()
 x_l, x_r = x_range
 xs = xs if xs.size > 1 else np.linspace(x_l, x_r, n_pts)
 ys_1 = np.zeros_like(xs)*nan
 ys_2 = np.zeros_like(xs)*nan
 if func_arr != None:
  ys_1 = func_arr(xs)
 if func != None:
  ys_2 = np.array([func(x) for x in xs])  
 if 'clf' in ftype:
  plt.clf()
 plt.plot(xs, ys_1, linewidth=2)
 plt.plot(xs, ys_2, alpha=0.8)
 plt.grid(True)
 return xs, ys_1, ys_2


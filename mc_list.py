from math import *
import numpy as np
from mc_base import *

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

#functions on lists/arrays of xs and of xs and ys (1D and 2D)
#for splines, see mc_splines.py

#calculates generalized mean of vs: pwr=0:geometric, 1:arithmetic, 2:root mean sqr, -1: harmonic
def gen_mean(vs, pwr): #(+)
 vals = np.array(vs) if isinstance(vs, tuple) or isinstance(vs, list) else vs
 n_vals = vals.size
 m = (vals.prod())**(1/n_vals) if pwr==0 else ( (vals**pwr).sum()/n_vals)**(1/pwr)
 return m

#returns mask with False on non-unique or closely-spaced points
def close_pt_detect(vs_srt, thr=0.03): #(+)
 n_vs = vs_srt.size

 vs_diff = np.diff(vs_srt)
 mask_vs = np.ones(vs_srt.shape, dtype=np.bool_)
 mask_vs[np.where(vs_diff==0)[0]+1] = False
 if mask_vs[-1] == False:
  i_last_un = np.nonzero(mask_vs)[0][-1] 
  mask_vs[i_last_un:-1] = False
  mask_vs[-1] = True
 if thr<= 0:
  return mask_vs
 i_u = np.nonzero(mask_vs)[0]
 vs_u = vs_srt[i_u]
 diff_u = vs_diff[i_u[1:] - 1]

 n_u = vs_u.size

 i_close_m =  np.where(diff_u[1:-1] < thr * np.minimum(diff_u[:-2], diff_u[2:]))[0] + 1

 close_l = (1,) if diff_u[0] < thr * diff_u[1] else ()
 close_r = (n_u-2,) if diff_u[-1] < thr * diff_u[-2] else ()
 i_close = arr_append(i_close_m, vals_l = close_l, vals_r = close_r, dtype=np.int32)
 if i_close.size > 0 and i_close[-1] == n_u-1:
  i_close[-1] -= 1

 mask_vs[i_u[i_close]]=False
 return mask_vs
 

#calculate partial derivatives of vs on xs and ys coordinates (rectilinear)
def der_calc_2D(xs, ys, vs, ftype='single'): #(+) #5kcalls/s
 f_types = ftype.split(' ')
 len_x, len_y = vs.shape
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0]
 ders = np.zeros((5, len_x, len_y))
 dxs, dys = np.gradient(vs, x_ax, y_ax, edge_order=2)
 ders[i_dx], ders[i_dy] = dxs, dys
 if 'mixed' in f_types or 'double' in f_types:
  dxdys = np.gradient(dxs, y_ax, axis=1, edge_order=2) #OK!
  ders[i_dxy] = dxdys
 if 'double' in f_types:
  d2xs = np.gradient(dxs, x_ax, axis=0, edge_order=2) 
  d2ys = np.gradient(dys, y_ax, axis=1, edge_order=2)
  ders[i_d2x] = d2xs
  ders[i_d2y] = d2ys
 return ders

#finds indices where y_val is between y_left and y_right and returns 'effective index'
def inrange_ax_float(ys, y_val): #(+) 
 len_lst = len(ys)-1 #linear interpolation is used
 y_arr = np.ones(len_lst) * y_val
 is_y = np.nonzero(inrange_arr(y_arr, (ys[:-1], ys[1:])) + inrange_arr(y_arr, (ys[1:], ys[:-1])))[0]
 if is_y.size == 0:
  return is_y.astype(np.float64)
 ys_l, ys_r = ys[is_y], ys[is_y+1]
 is_frac = (y_val - ys_l) / (ys_r - ys_l)
 is_float = is_y + is_frac
 return is_float


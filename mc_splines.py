from math import *
import numpy as np
from mc_poly import *
from mc_base import *
from mc_list import *
from mc_gen import *

import copy

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats


#=====linear and cubic 1D and 2D spline calculation, evaluation and analysis
#=====all splines use absolute or relative (normalized to spline interval) coordinates
#=====spline length: number of breakpoints - 1
#=====spline values assumed linear everywhere. To calculate splines of logarithmic values, use external code.
#=====functions order: lin -> sqr -> cubic, 1D -> 2D
#=====all reduced values (x transformed with (x_l, x_r) -> (0, 1) ) are indicated by underscore. I.e. x == 2.5, (x_, x_r) == (2, 4) ->  x_ = 0.25.

spline_dummy = np.zeros((4, 4))

pwrs = np.linspace(0, 10, 11) #used in bicubic_inter, etc.

#matrices for bicubic spline calculations
a_m_inv = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.0,3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [-3.0,0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0,-3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0,-1.0, 0.0],
                    [9.0,-9.0,-9.0, 9.0, 6.0, 3.0,-6.0,-3.0, 6.0,-6.0, 3.0,-3.0, 4.0, 2.0, 2.0, 1.0],
                    [-6.0,6.0, 6.0,-6.0,-3.0,-3.0, 3.0, 3.0,-4.0, 4.0,-2.0, 2.0,-2.0,-2.0,-1.0,-1.0],
                    [2.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [-6.0,6.0, 6.0,-6.0,-4.0,-2.0, 4.0, 2.0,-3.0, 3.0,-3.0, 3.0,-2.0,-1.0,-2.0,-1.0],
                    [4.0,-4.0,-4.0, 4.0, 2.0, 2.0,-2.0,-2.0, 2.0,-2.0, 2.0,-2.0, 1.0, 1.0, 1.0, 1.0]])

a_mat = np.linalg.inv(a_m_inv)

a_m1 = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [-3.0,3.0,-2.0,-1.0],
                 [2.0,-2.0, 1.0, 1.0]])

a_m2 = np.array([[1.0, 0.0,-3.0, 2.0],
                 [0.0, 0.0, 3.0,-2.0],
                 [0.0, 1.0,-2.0, 1.0],
                 [0.0, 0.0,-1.0, 1.0]])

#hermite base polynomials, for cubic spline calculation
h_00 = np.array((1, 0, -3, 2.))
h_01 = np.array((0, 0, 3, -2.))
h_10 = np.array((0, 1, -2, 1.))
h_11 = np.array((0, 0, -1, 1.))

herm_p = np.vstack((h_00, h_01, h_10, h_11))

#===============data preparation, aux functions========================

#makes (x_l, x_r) -> (0, 1) coordinate transformation for x value(s)
#(or calculates reduced x values
def to_01(x, x_l, x_r): #(+)
 return (x - x_l) / (x_r - x_l)


#returns index and reduced value of val on (ax[i], ax[i+1]) interval
def ind_find(ax, val): #(+)
 len_ax = ax.size
 i_0 = np.searchsorted(ax, val, side='right') - 1 ##length of subarray where val < input val
 i = trunc_val(i_0, (0, len_ax-2))
 val_l, val_r = ax[i], ax[i+1]
 val_ = (val - val_l) / (val_r - val_l)
 return i, val_

#returns index and reduced value of val on (ax[i], ax[i+1]) interval
def ind_find_arr(ax, val): #(+)
 len_ax = ax.size
 i_0 = np.searchsorted(ax, val, side='right') - 1 ##length of subarray where val < input val
 i = trunc_arr(i_0, (0, len_ax-2))
 val_l, val_r = ax[i], ax[i+1]
 val_ = (val - val_l) / (val_r - val_l)
 return i, val_


#calculates indices of value (val_x, val_y) on a 2D grid of xs and ys 
#returns also reduced coordinated of (val_x, val_y) on that cell 
#2D analog of ind_find
def ind_find_2D(xs, ys, val_x, val_y, ftype=''): #(+,+)
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0,:]
 if ftype == 'i':
  i_x, i_y = ind_find(x_ax, val_x, ftype=ftype), ind_find(y_ax, val_y, ftype=ftype)
  return i_x, i_y
 else:
  i_x, x_ = ind_find(x_ax, val_x)
  i_y, y_ = ind_find(y_ax, val_y)
  return i_x, i_y, x_, y_


def ind_find_2D_arr(xs, ys, val_x, val_y, ftype=''): #(+)
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0,:]
 if ftype == 'i':
  i_x, i_y = ind_find_arr(x_ax, val_x, ftype=ftype), ind_find_arr(y_ax, val_y, ftype=ftype)
  return i_x, i_y
 else:
  i_x, x_ = ind_find_arr(x_ax, val_x)
  i_y, y_ = ind_find_arr(y_ax, val_y)
  return i_x, i_y, x_, y_


#returns dxs, dys , y1s(edge 2nd order) and y2s; ftype rel: derivatives on xs relative coordinates.
def grad(xs, ys, ftype=''): #(+) #9kcalcs/s, 2nd order more crude than np.gradient, but returns all values
 n_xs = xs.size
 x_vs, y_vs = np.zeros((n_xs, 3)), np.zeros((n_xs,3))
 dxs = np.diff(xs)
 dys = np.diff(ys)
 arr_out = np.zeros((n_xs, 4))
 y_vs[1:-1,0] = ys[:-2]
 y_vs[1:-1,1] = ys[1:-1]
 y_vs[1:-1,2] = ys[2:]
 y_vs[0], y_vs[-1] = ys[:3], ys[-3:]
 if 'rel' in ftype:
  x_vs[1:-1,0] = -dxs[:-1] / dxs[1:]
  x_vs[1:-1,2] = 1
  x_vs[0], x_vs[-1] = (xs[:3] - xs[0]) / dxs[0], (xs[-3:] - x_vs[-1]) / dxs[-1]
 else:
  x_vs[1:-1,0] = -dxs[:-1]#xs[:-2]
  x_vs[1:-1,2] = dxs[1:]#xs[2:]
  x_vs[0], x_vs[-1] = xs[:3] - xs[0], xs[-3:] - x_vs[0]
 sqrs = sqr_calc_arr(x_vs, y_vs)
 arr_out[1:-1,0] = dxs[1:]
 arr_out[1:-1,1] = dys[1:] 
 arr_out[1:-1,2] = sqrs[1:-1,1]
 arr_out[1:-1,3] = 2*sqrs[1:-1,2]
 arr_out[0] = dxs[0], dys[0], 2*sqrs[0,2]*x_vs[0,0] + sqrs[0,1], 2*sqrs[0,2]
 arr_out[-1] = dxs[-1], dys[-1], 2*sqrs[-1,2]*x_vs[-1,2] + sqrs[-1,1], 2*sqrs[-1,2]
 return arr_out


#------------------array and spline extremum calculation, used in inrange_ax etc-------------

#----on 1D arrays
#returns mask corresponding to repeating x values (i.e. [1,2,2,3,4,5,5] -> (False, True, True, False, False, True, True)
def flat_cut(xs): #(+)
 xs_diff = np.diff(xs)
 mask_flat = np.zeros(xs_0.size, dtype=np.bool_)
 i_flat = np.where(xs_diff==0)[0]
 mask_flat[i_flat] = True
 mask_flat[i_flat+1] = True
 return mask_flat

#locates extremums on xs[1:-1], filtered of nans and flat ranges
def extr_find_simple(xs): #(+)
 xs_diff = np.diff(xs)
 diff_signs = signum_arr(xs_diff)
 d2_signs = np.diff(diff_signs)
 is_max = np.where(d2_signs == -2)[0] + 1
 is_min = np.where(d2_signs == 2)[0] + 1
 return is_max, is_min


#locates extremums (including "flat" subsequences)
#must not contain nans
def extr_find(xs, ftype=0): #(+) #12kcalc/s, 2.3Mpts/s
 periodic = ftype==1
 n_vals = xs.size
 xs_diff = np.diff(xs)
 diff_signs = signum_arr(xs_diff)
 d2_signs = np.diff(diff_signs)
 extr_det = np.zeros(n_vals)
 extr_det[1:-1] = d2_signs
 is_xtr = np.nonzero(extr_det)[0]
 if is_xtr.size == 0: #all values equal
  return extr_det
 is_half = np.where((extr_det[is_xtr] == -1) + (extr_det[is_xtr] == 1))[0]
 et_vals = extr_det[is_xtr]

 mask_flat = np.zeros(n_vals, dtype=np.bool_)
 i_zd = np.where(xs_diff==0)[0]
 mask_flat[i_zd] = True
 mask_flat[i_zd+1] = True # mask_flat #OK
 flat_sum = np.cumsum(mask_flat.astype(int))
# is_flat = np.nonzero(mask_flat)[0]
# is_step = np.nonzero(np.diff(is_flat) - 1)[0]

 extr_types = np.copy(extr_det)
 for i in range(is_half.size - 1):
  i_cur, i_next = is_half[i], is_half[i+1]
  i_xtr_cur, i_xtr_next = is_xtr[i_cur], is_xtr[i_next]   #print(i_xtr_cur, i_xtr_next, '\t', extr_det[i_xtr_cur], extr_det[i_xtr_next])
  if xs[i_xtr_next] == xs[i_xtr_cur] and extr_det[i_xtr_cur] == extr_det[i_xtr_next] and flat_sum[i_xtr_next] - flat_sum[i_xtr_cur] == i_xtr_next - i_xtr_cur:
   extr_types[i_xtr_cur:i_xtr_next+1] = extr_types[i_xtr_cur] * 3

 i_first_diff = 1 if mask_flat[0]==0 else is_xtr[0] + 1
 i_last_diff = -2 if mask_flat[-1]==0 else is_xtr[-1] - 1
 x_first_diff = xs[i_first_diff]
 x_last_diff = xs[i_last_diff]

 edge_equal = periodic and xs[0]==xs[-1]
 
 if xs[0] < x_first_diff and not (periodic and (xs[0] > xs[-1] or edge_equal and xs[0] > x_last_diff)):
  extr_types[:i_first_diff] = 4
 if xs[0] > x_first_diff and not (periodic and (xs[0] < xs[-1] or edge_equal and xs[0] < x_last_diff)):
  extr_types[:i_first_diff] = -4

 if xs[-1] < x_last_diff and not (periodic and (xs[-1] > xs[0] or edge_equal and xs[-1] > x_first_diff)):
  extr_types[i_last_diff+1:] = 4
 if xs[-1] > x_last_diff and not (periodic and (xs[-1] < xs[0] or edge_equal and xs[-1] < x_first_diff)):
  extr_types[i_last_diff+1:] = -4

 return extr_types
 

def extr_find_check(xs, ftype=''):
 n_vals = xs.size
 if ftype==1:
  xs_exp = np.hstack((xs, xs, xs))
  extr_types_exp = extr_find_check(xs_exp)
  extr_types_ch = extr_types_exp[n_vals:2*n_vals]
  return extr_types_ch
 xs_diff = np.hstack((0.0, np.diff(xs)))
 extr_types_ch = np.zeros_like(xs).astype(float)
 for i_x in range(1, n_vals-1):
  x = xs[i_x]
  for i_xl in range(i_x-1, -1, -1):
   x_l = xs[i_xl]
   if x_l != x:
    break
  for i_xr in range(i_x+1, n_vals):
   x_r = xs[i_xr]
   if x_r != x:
    break
  if x < x_l and x < x_r:
   extr_types_ch[i_x] = 3 if i_xr - i_xl > 2 else 2
  if x > x_l and x > x_r:
   extr_types_ch[i_x] = -3 if i_xr - i_xl > 2 else -2

 i_first_diff = np.where(xs != xs[0])[0][0]
 i_last_diff = np.where(xs != xs[-1])[0][-1]
 
 extr_types_ch[:i_first_diff] = 4 if xs[0] < xs[i_first_diff] else -4
 extr_types_ch[i_last_diff+1:] = 4 if xs[-1] < xs[i_last_diff] else -4

 #np.vstack((np.arange(n_vals), xs, extr_types)).T
 return extr_types_ch 
  



#=======================spline calculation=============================

#add boundary conditions!
def spline_calc(xs, ys, spl_type='cube herm'):  
 spl_types = spl_type.split(' ')
 rel_type = 'rel' if 'rel' in spl_types else ''
 if 'cube' in spl_types:
  if 'herm' in spl_types:
   spl = spl_calc_cube_herm(xs, ys, spl_type=rel_type)
  else:
   spl = spl_calc_cube_nat(xs, ys, spl_type=rel_type)
 else:
  spl = spl_calc_lin(xs, ys, spl_type=rel_type)
 return spl


#A + Bx form
def spl_calc_lin(xs, ys, spl_type=''): #(+)
 x_diff = np.diff(xs)
 y_diff = np.diff(ys)
 if spl_type == 'rel':
  As = ys[:-1]
  Bs = y_diff
 else:
  Bs = y_diff / x_diff
  As = ys[:-1] - xs[:-1] * Bs
 spl = arr_stack((As, Bs), 'v').T
 return spl 

#cubic hermite spline calculation
def spl_calc_cube_herm(xs, ys, spl_type='rel', bc=('nk', 'nk')):
 dxs, dys, y1s, y2s = grad(xs, ys).T 
 ys_l, ys_r = ys[:-1], ys[1:]
 y1s_l, y1s_r = y1s[:-1], y1s[1:]
 y1s_l_, y1s_r_ = y1s_l * dxs[:-1], y1s_r * dxs[:-1]
 spl_herm_ = h_00*ys_l[:,None] + h_10*y1s_l_[:,None] + h_01*ys_r[:,None] + h_11 * y1s_r_[:,None]
 return spl_herm_ if spl_type == 'rel' else rel_to_abs(spl_herm_, xs)

#natural cubic spline, per https://en.wikipedia.org/wiki/Spline_(mathematics)
def spl_calc_cube_nat(xs, ys, spl_type=''): 
 n_pts = xs.size
 dxs, dys = np.diff(xs),  np.diff(ys)
 y1s = np.gradient(ys, xs)
 y2s = np.gradient(y1s, xs) #need more precise y2s here than "grad" function output
 spl_pr_1 = np.zeros((n_pts-1, 4)) # (x-x_l)**3, (x_r-x)**3, (x-x_l), (x_r-x) coeffs 
 spl_pr_2 = np.zeros((n_pts-1, 4))
 spl_pr_1[:,3] = y2s[1:] / (6*dxs)
 spl_pr_2[:,3] = y2s[:-1] / (6*dxs)
 spl_pr_1[:,1] = ys[1:]/dxs - y2s[1:]*dxs/6
 spl_pr_2[:,1] = ys[:-1]/dxs - y2s[:-1]*dxs/6
 ones_arr = np.ones(n_pts-1)
 spl_nat = spl_transform(spl_pr_1, -xs[:-1], ones_arr) + spl_transform(spl_pr_2, xs[1:], -1*ones_arr) 
 return spl_nat if spl_type != 'rel' else rel_to_abs(spl_nat, xs, 'rel')

#https://stackoverflow.com/questions/141422/how-can-a-transform-a-polynomial-to-another-coordinate-system
def spl_transform(spl, cs_0, cs_1): #(+)
 deg = spl.shape[-1] - 1
 if deg == 3: #(+)
  ks_0, ks_1, ks_2, ks_3 = spl.T
  ks_0_ = ks_0 + ks_1*cs_0 + ks_2*cs_0**2 + ks_3*cs_0**3
  ks_1_ = ks_1*cs_1 + 2*ks_2*cs_0*cs_1 + 3*ks_3*cs_0**2*cs_1
  ks_2_ = ks_2*cs_1**2 + 3*ks_3*cs_0*cs_1**2
  ks_3_ = ks_3*cs_1**3
  spl_trans = arr_stack((ks_0_, ks_1_, ks_2_, ks_3_), 'v').T
 elif deg == 2:
  ks_0, ks_1, ks_2 = spl.T
  ks_0_ = ks_0 + ks_1*cs_0 + ks_2*cs_0**2
  ks_1_ = ks_1*cs_1 + 2*ks_2*cs_0*cs_1
  ks_2_ = ks_2*cs_1**2
  spl_trans = arr_stack((ks_0_, ks_1_, ks_2_), 'v').T
 elif deg==1:
  ks_0, ks_1 = spl.T
  ks_0_ = ks_0 + ks_1*cs_0
  ks_1_ = ks_1*cs_1
  spl_trans = arr_stack((ks_0_, ks_1_), 'v').T
 return spl_trans
 
#transforms a spline in reduced coordinates to absolute coordinates 
def rel_to_abs(spl, xs, ftype=''): #(+)
 dxs = np.diff(xs)
 if ftype=='rel': #from absolute to relative
  cs_0, cs_1 = xs[:-1], dxs
 else: 
  cs_0, cs_1 = -1.0*xs[:-1]/dxs, 1/dxs 
 return spl_transform(spl, cs_0, cs_1)


#---------------------2D splines---------------------------

#calculates bicubic coefficients for 2x2 square vs, with values vs and derivatives dxs, dys and dxys
def bicubic_single(x_0, y_0, dx, dy, vs, dxs, dys, dxys): #(+)
 dxs_, dys_, dxys_  = dxs * dx, dys * dy, dxys * dx * dy
 vals_matrix = np.vstack(( np.hstack((vs, dys_)), np.hstack((dxs_, dxys_)) ))
 coeffs_norm = a_m1 @ vals_matrix @ a_m2
 return coeffs_norm


#calculate all spline coefficients for value val
#also calculates needed derivatives
def bicubic_spline_calc(xs, ys, vals): #(+)
 x_ax, y_ax = xs[:,0], ys[0,:]
 x_diffs, y_diffs = np.diff(x_ax), np.diff(y_ax)
 len_x, len_y = vals.shape
 splines = np.zeros((len_x, len_y, 4, 4))
 ders = der_calc_2D(xs, ys, vals, ftype = 'mixed') 
 dxs, dys, dxys = ders[i_dx], ders[i_dy], ders[i_dxy]
 for i_x in range(len_x-1):
  for i_y in range(len_y-1):
   dx, dy = x_diffs[i_x], y_diffs[i_y] 
   splines[i_x, i_y] = bicubic_single(x_ax[i_x], y_ax[i_y], dx, dy, vals[i_x:i_x+2, i_y:i_y+2], dxs[i_x:i_x+2, i_y:i_y+2], dys[i_x:i_x+2, i_y:i_y+2], dxys[i_x:i_x+2, i_y:i_y+2]) #x_0, y_0, dx, dy, vs, dxs, dys, dxys):
   if i_y == len_y - 2:
    splines[i_x, i_y + 1] = splines[i_x, i_y]
 splines[-1,:] = splines[-2,:]
 return splines


#=======================spline evaluation==============================

#evaluates spline @ x point
def spline_eval(spl, xs, x, val_type='', spl_type=''): #(+)
 len_x = xs.size
 i_x = min(max(np.searchsorted(xs, x, side='right') - 1, 0), len_x-2)
 spl_cur = spl[i_x]
 if spl_type != 'rel':
  y = poly_eval(spl_cur, x) 
 else: #splines calculated on relative (0, 1) coordinate for each interfal, used where float precision is insufficient otherwise
  x_l, x_r = xs[i_x:i_x+2]
  x_ = to_01(x, x_l, x_r) 
  y = poly_eval(spl_cur, x_)
 return 10**y if val_type=='log+' else (-1.0 * 10**y if val_type=='log-' else y)


def spline_eval_arr(spl, xs, x_vals, val_type='', spl_type=''):  #(+)
 len_x = xs.size
 spl_deg = spl.shape[-1]
 is_x_0 = np.searchsorted(xs, x_vals) - 1
 is_x = trunc_arr(is_x_0, (0, len_x-2))
 spl_coeffs = spl[is_x] #ordered against x_vals
 pwrs = np.arange(spl_deg) 
 x_vs = to_01(x_vals, xs[is_x], xs[is_x+1]) if spl_type == 'rel' else x_vals  
 x_pwrs = np.power(x_vs[:,None], pwrs)
 ys = (spl_coeffs * x_pwrs).sum(axis=1)
 return 10**ys if val_type=='log+' else (-1.0 * 10**ys if val_type=='log-' else ys)


#evaluates single bicubic spline; only on 0,1; 0,1 intervals
def bicubic_inter(coeffs, x_, y_): #(+)
 x_p = np.power(x_, pwrs[:4])
 y_p = np.power(y_, pwrs[:4])
 val = x_p @ coeffs @ y_p.T 
 return val
 

#bicubic interpolation on 2D grid #40k / second
def spline_eval_2D(spl, xs, ys, x, y): #(+)
 i_x, i_y, x_, y_ = ind_find_2D(xs, ys, x, y)
 spl_loc = spl[i_x, i_y]
 y = bicubic_inter(spl_loc, x_, y_)
 return y

def spline_eval_2D_arr(spl, xs, ys, x_vals, y_vals): #(+) #7kcalls/s, 1.6Mpts/s
 len_x, len_y = spl.shape[:2]
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0]
 pwrs = np.arange(4) 
 is_x_0 = np.searchsorted(x_ax, x_vals) - 1
 is_x = trunc_arr(is_x_0, (0, len_x-2))
 is_y_0 = np.searchsorted(y_ax, y_vals) - 1
 is_y = trunc_arr(is_y_0, (0, len_y-2))
 x_vs = to_01(x_vals, x_ax[is_x], x_ax[is_x+1])
 y_vs = to_01(y_vals, y_ax[is_y], y_ax[is_y+1])
 spl_coeffs = spl[is_x, is_y] #ordered against x_vals
 x_pwrs = np.power(x_vs[:,None], pwrs)
 y_pwrs = np.power(y_vs[:,None], pwrs)
 vs = np.einsum('ki,kij,kj->k', x_pwrs, spl_coeffs, y_pwrs)
 return vs
 
#makes cubic slice of bicubic spline @ x or y coordinate
def bicubic_to_cubic(spl, xs, ys, x=nan, y=nan): #(+)
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0]
 ax = 'x' if isfinite(x) else 'y'
 if ax=='x':
  i_ax, ax_ = ind_find(x_ax, x)
 else:
  i_ax, ax_ = ind_find(y_ax, y)  
 ax_p = np.power(ax_, pwrs[:4]) # np.array([ax_**i for i in range(4)])
 bicubics_ax = spl[:,i_ax] if ax=='x' else spl[i_ax,:]
 cubics = bicubics_ax @ ax_p if ax=='x' else ax_p @ bicubics_ax
 return cubics

#the same, single-cell
def bicubic_to_cubic_single(coeffs, ax_, ax='x'): #(+)
 spl = coeffs @ ax_p.T if ax=='x' else ax_p @ coeffs
 return spl

#=======================solve spline for y values======================


#calculates y values of a spline at break points
def spline_to_ys(spl, xs, spl_type=''): #(+)
 ys = np.zeros_like(xs)
 if spl_type=='rel':
  ys[:-1] = spl[:,0]
  ys[-1] = poly_eval(spl[-1], 1.0)
 else:
  spl_deg = spl[0].size
  pwrs = np.arange(spl_deg).astype(float)# np.indices((spls.shape[-1],))[0][::-1]
  x_pwrs = np.power(xs[:-1,np.newaxis], pwrs)
  ys[:-1] = (spl * x_pwrs).sum(axis=1)
  ys[-1] = poly_eval(spl[-1], xs[-1])
 return ys

#calculates ranges of y values for a cubic spline at each interval
def spl_y_range(spl, xs, ys): #(+)
 dxs, dys = np.diff(xs), np.diff(ys)
 y1s = dys / dxs
 ys_r = arr_append( ys[1:-1] + y1s[:-1] * dxs[1:], vals_l = (ys[0], ys[1])) #OK
 ys_l = arr_append( ys[1:-1] - y1s[1:] * dxs[:-1], vals_r = (ys[-2], ys[-1]))
 ys_all = arr_stack((ys[:-1], ys_l[:-1], ys[1:], ys_r[1:]), 'v').T
 ys_min = ys_all.min(axis=1)
 ys_max = ys_all.max(axis=1)
 return ys_min, ys_max

#returns indices of xs where y is between ys[i] and ys[i+1]
#simple form - "just between", advanced - including possible extremums, fast as possible
def inrange_ax(xs, ys, y_val, ftype='[) mask'): #(+)
 f_types = ftype.split(' ')
 len_x = xs.size
 if 'ext' not in f_types:
  if 'mask' in f_types:
   is_x = np.where(inrange_arr(y_val, (ys[:-1], ys[1:]), ftype=ftype) + inrange_arr(y_val, (ys[1:], ys[:-1]), ftype=ftype), True, False)
  else:
   is_x = np.nonzero(inrange_arr(y_val, (ys[:-1], ys[1:]), ftype=ftype) + inrange_arr(y_val, (ys[1:], ys[:-1]), ftype=ftype))[0]
 else:
  dxs, dys = np.diff(xs), np.diff(ys)
  y1s = dys / dxs
  ys_r = arr_append( ys[1:-1] + y1s[:-1] * dxs[1:], vals_l = (ys[0], ys[1])) #OK
  ys_l = arr_append( ys[1:-1] - y1s[1:] * dxs[:-1], vals_r = (ys[-2], ys[-1]))
  ys_all = arr_stack((ys[:-1], ys_l[:-1], ys[1:], ys_r[1:]), 'v').T
  ys_min = ys_all.min(axis=1)
  ys_max = ys_all.max(axis=1)
  if 'mask' in f_types:
   is_x = np.where(inrange_arr(y_val, (ys_min, ys_max), ftype=ftype), True, False)
  else:
   is_x = np.nonzero(inrange_arr(y_val, (ys_min, ys_max), ftype=ftype))[0]
 return is_x
 
 
#solves spl == y equation 
def spline_solve(spl, xs, y, spl_type='', x_range=(nan,nan), x_0 = nan, ftype=''): #(+)
 len_x = xs.size
 deg = spl.shape[-1] - 1
 
 x_l = -inf if 'oor' in ftype else (xs[0]  if isnan(x_range[0]) else x_range[0])
 x_r =  inf if 'oor' in ftype else (xs[-1] if isnan(x_range[1]) else x_range[1])
 i_l, i_r = np.searchsorted(xs, (x_l, x_r), side='right')
 xs_trunc, spl_trunc = xs[max(i_l-1,0):min(i_r+1, len_x)], spl[max(i_l-1,0):min(i_r, len_x-1)]
 
 if deg == 3:
  roots_all = spl_solve_cube(spl_trunc, xs_trunc, y, spl_type=spl_type, x_range=x_range, ftype=ftype)
 else:
  roots_all = spl_solve_lin(spl_trunc, xs_trunc, y, spl_type=spl_type, x_range=x_range)
 
 mask_IR = inrange_arr(roots_all, (x_l, x_r), ftype='[)')
 roots_0 = roots_all[mask_IR]
 i_srt = np.argsort(np.abs(roots_0 - x_0)) if isfinite(x_0) else np.argsort(roots_0)
 return roots_0[i_srt]
  
 
def spl_solve_cube(spl, xs, y, spl_type='', x_range=(nan,nan), ftype=''): #(+)
 x_l = xs[0]  if isnan(x_range[0]) else x_range[0]
 x_r = xs[-1] if isnan(x_range[1]) else x_range[1]
 ext_l, ext_r = x_l < xs[0], x_r > xs[-1]
 f_types = ftype.split(' ')
 len_x = xs.size
 ys = spline_to_ys(spl, xs, spl_type=spl_type)  

 is_x = inrange_ax(xs, ys, y, ftype='[) ext mask' if 'simple' not in f_types else '[) mask')
 calc_ext_l = ext_l and inrange(y,  poly_range(spl[0], (x_l, xs[0])))
 calc_ext_r = ext_r and inrange(y,  poly_range(spl[-1], (xs[-1], x_r)))
 is_x[0] = is_x[0] + calc_ext_l
 is_x[-1] = is_x[-1] + calc_ext_r
  
 spls_x = spl[is_x]
 n_calc = np.nonzero(is_x)[0].size
 if 'rel' not in spl_type:
  roots_0 = cube_solve_arr(spls_x, np.ones(n_calc)*y)
 else:
  dxs = np.diff(xs)
  roots_rel = cube_solve_arr(spls_x, np.ones(n_calc)*y)
  roots_0 = roots_rel * dxs[is_x,None] + xs[:-1][is_x][:,None]
 xs_l_0, xs_r_0 = xs[:-1][is_x], xs[1:][is_x]
 
 if calc_ext_l:
  xs_l_0[0] = x_l
 if calc_ext_r:
  xs_r_0[-1] = x_r
 
 xs_l = arr_stack([xs_l_0 for i in range(3)], 'v').T
 xs_r = arr_stack([xs_r_0 for i in range(3)], 'v').T
 mask_real = close_to_real_arr(roots_0, tol=100)
 roots_real, xs_l_real, xs_r_real = roots_0[mask_real].real, xs_l[mask_real], xs_r[mask_real]
 mask_IR = inrange_arr(roots_real, (xs_l_real, xs_r_real)) 
 roots = roots_real[mask_IR]
 return roots


def spl_solve_lin(spl, xs, y, spl_type='', x_range=(nan,nan)): #(+)
 x_l = xs[0]  if isnan(x_range[0]) else x_range[0]
 x_r = xs[-1] if isnan(x_range[1]) else x_range[1]
 ext_l, ext_r = x_l < xs[0], x_r > xs[-1]
 ys = spline_to_ys(spl, xs, spl_type=spl_type)  

 is_x = inrange_ax(xs, ys, y)
 calc_ext_l = ext_l and inrange(y,  poly_range(spl[0], (x_l, xs[0])))
 calc_ext_r = ext_r and inrange(y,  poly_range(spl[0], (xs[-1], x_r)))
 is_x[0] = is_x[0] + calc_ext_l
 is_x[-1] = is_x[-1] + calc_ext_r

 n_calc = np.nonzero(is_x)[0].size
 spls_x = spl[is_x]
 if 'rel' in spl_type:
  dxs = np.diff(xs)
  roots_rel = lin_solve_arr(spls_x, np.ones(n_calc)*y)
  roots = roots_rel * dxs[is_x] + xs[:-1][is_x]
 else:
  roots = lin_solve_arr(spls_x, np.ones(n_calc)*y)
 return roots


#=======================spline analysis===============================

#returns derivative of a spline
def spline_der(spl, xs, spl_type=''): #(+)
 return poly_der(spl / np.diff(xs)[:,None]) if spl_type=='rel' else poly_der(spl)


#calculate indefinite integral of a spline #4th order polynomial
#(+)
def spline_int(spl, xs, spl_type='', ftype=''): #(+)
 len_x = xs.size
 if spl_type == 'rel':
  xs_diffs = np.diff(xs)
  spl_int = poly_int(spl * xs_diffs[:,np.newaxis]) 
  ints_loc = spline_to_ys(spl_int, np.ones(len_x))[:-1] #three times higher than spl #divide by xs lingths!
 else:
  spl_int = poly_int(spl)
  ints_l = spline_to_ys(spl_int, xs)[:-1] #ints_l1 = poly_eval_arr(spl_int, xs[:-1]) 
  ints_r = spline_to_ys(spl_int[:-1], xs[1:]) #ints_r1 = poly_eval_arr(spl_int, xs[1:]) 
  ints_r[-1] = poly_eval(spl_int[-1], xs[-1])
  ints_loc = ints_r - ints_l
 
 ys_int = arr_append(np.cumsum(ints_loc), vals_l=(0.0,))
 if 'ys' in ftype:
  return ys_int
 cs_int = (ys_int[:-1] - ints_l if spl_type != 'rel' else ys_int[:-1]) #integration constants
 spl_int[:,0] = cs_int # spl_int[-1,0] = cs_int[-1] 
 return spl_int

#calculates definite integral of a spline
def spline_integrate(spl, xs, ab=(nan,nan), spl_type='', ftype=''): #(+)
 spl_int = spl if 'int' in ftype else spline_int(spl, xs, spl_type=spl_type)
 a = ab[0] if isfinite(ab[0]) else xs[0]
 b = ab[1] if isfinite(ab[1]) else xs[-1]
 int_a = spline_eval(spl_int, xs, a, spl_type=spl_type)
 int_b = spline_eval(spl_int, xs, b, spl_type=spl_type)
 val = int_b - int_a
 return val
 


#============interpolation and solving without spline calculation
#takes values of xs and ys and calculates where f(x) == y if f(xs) == ys
def list_solve(xs, ys, y_val, spl_type='herm', x_range=(nan,nan), x_0=nan, ftype=''): #(+)
 x_l = -inf if 'oor' in ftype else (xs[0]  if isnan(x_range[0]) else x_range[0])
 x_r =  inf if 'oor' in ftype else (xs[-1] if isnan(x_range[1]) else x_range[1])
 len_x = xs.size 
 if 'lin' in spl_type:
  x_roots_0 = list_solve_lin(xs, ys, y_val)
 elif len_x < 5:
  y_poly = poly_calc(xs, ys)
  x_roots_all = poly_solve(y_poly, y_val)
  x_roots_0 = x_roots_all[close_to_real_arr(x_roots_all)]
 else:
  x_roots_0 = list_solve_cube(xs, ys, y_val, spl_type=spl_type, x_range=(x_l, x_r), x_0=x_0, ftype=ftype)
 x_roots = x_roots_0[inrange_arr(x_roots_0, (x_l, x_r), ftype='[]')]
 i_srt = np.argsort(np.abs(x_roots - x_0)) if isfinite(x_0) else np.argsort(x_roots)
 return x_roots[i_srt]

def list_solve_lin(xs, ys, y): #(+)
 is_y = np.where( ((ys[:-1] <= y) * (y < ys[1:])) + ((ys[:-1] >= y) * (y > ys[1:])), True, False )     #(ys[1:] > y) * (ys[:-1] <= y), True, False) + np.where( (ys[1:] < y) * (ys[:-1] >= y), True, False)
 leftwards = (y <= ys[0] and ys[0] < ys[1] or y >= ys[0] and ys[0] > ys[1])
 rightwards = (y <= ys[-1] and ys[-1] < ys[-2] or y >= ys[-1] and ys[-1] > ys[-2])
 is_y[0] += leftwards
 is_y[-1] += rightwards
 xs_l = xs[:-1][is_y]
 xs_r = xs[1:][is_y]
 ys_l = ys[:-1][is_y]
 ys_r = ys[1:][is_y]
 x_roots = xs_l + (xs_r - xs_l) * (y - ys_l) / (ys_r - ys_l)
 return x_roots
	
def list_solve_cube(xs, ys, y_val, spl_type='herm', x_range=(nan,nan), x_0=nan, ftype=''): #(+)
 spl = spline_calc(xs, ys, spl_type=spl_type)
 x_roots = spline_solve(spl, xs, y_val, spl_type=spl_type, x_range=x_range, x_0=x_0, ftype=ftype) #(+)
 return x_roots #in ascending order

#interpolates f(xs) == (ys) at x_val
def interpolate_1D(x_ax, ys, x_val, spl_type='herm'): #(+)
 len_x = x_ax.size
 spl_types = spl_type.split(' ')
 cube = False if 'lin' in spl_types else True
 i_x, x_ = ind_find(x_ax, x_val)
 if cube:
  i_x_0 = min(max(i_x, 1), len_x-3)
  spl_loc = poly_calc(x_ax[i_x_0-1:i_x_0+3], ys[i_x_0-1:i_x_0+3])
  return poly_eval(spl_loc, x_val)
 else:
  i_x_0 = min(i_x, len_x-2)
  A = (ys[i_x_0+1] - ys[i_x_0]) / (x_ax[i_x_0+1] - x_ax[i_x_0])
  B = ys[i_x_0] - A*x_ax[i_x_0]
  return A*x_val + B
 

def interpolate_2D(xs, ys, vals, x, y):  #(+.), left and upper edge shifted
 x_ax = xs if xs.ndim == 1 else xs[:,0]
 y_ax = ys if ys.ndim == 1 else ys[0]
 len_x, len_y = x_ax.size, y_ax.size
 i_x, x_ = ind_find(x_ax, x)
 i_y, y_ = ind_find(y_ax, y)
 i_x_0, i_y_0 = min(max(i_x, 1), len_x-3), min(max(i_y, 1), len_y-3)
 i_x_, i_y_ = i_x - (i_x_0-1), i_y - (i_y_0-1)
 x_ax_loc, y_ax_loc = x_ax[i_x_0-1:i_x_0+3], y_ax[i_y_0-1:i_y_0+3]
 vs_loc = vals[i_x_0-1:i_x_0+3, i_y_0-1:i_y_0+3]
 dxs_loc, dys_loc, dxys_loc = der_calc_2D(x_ax_loc, y_ax_loc, vs_loc, ftype='mixed')[:3] #grid_loc = np.array((xs[i_x_0-1:i_x_0+3, i_y_0-1:i_y_0+3], ys[i_x_0-1:i_x_0+3, i_y_0-1:i_y_0+3], vals[i_x_0-1:i_x_0+3, i_y_0-1:i_y_0+3]))
 x_loc, y_loc = x_ax_loc[1], y_ax_loc[1]
 dx_loc, dy_loc = x_ax_loc[2] - x_loc, y_ax_loc[2] - y_loc
 spl_loc = bicubic_single(x_loc, y_loc, dx_loc, dy_loc, vs_loc[1:3,1:3], dxs_loc[1:3,1:3], dys_loc[1:3,1:3], dxys_loc[1:3,1:3])
 val = bicubic_inter(spl_loc, x_, y_)
 return val


#integrates some function on list of xs
#assumes sorted unique xs, b > a, and no singularities between a and b
#calculates values on xs, calculates integral from a to b or from xs[0] to xs[-1]
def func_integrate(func, xs=arr_dummy, ab=None, ftype='lin sph cum', n_pts=101): #(+)
 if ab != None and xs.size == 0:
  x_l, x_r = ab
  if signum(x_l) != signum(x_r): #including zeros
   xs = np.linspace(x_l, x_r, n_pts)
  elif x_l < 0:
   xs = -1.0 * np.logspace(log10(-x_l), log10(-x_r), n_pts)
  else:
   xs = np.logspace(log10(x_l), log10(x_r), n_pts)
 else:
  xs = xs
 vals = func(xs)
 xs_f, vals_f = arr_filter(xs, vals)
 f_int = integrate_lst(xs_f, vals_f, ab=ab, ftype=ftype) 
 return (xs_f, f_int) if 'cum' in ftype else f_int



def integrate_lst(xs, ys, ab=None, ftype=''): #(+)
 f_types = ftype.split(' ')
 len_x = xs.size
 (x_l, x_r) = ab if ab != None else (xs[0], xs[-1])

 if 'spl' not in f_types:
  ys_c = ys if 'sph' not in ftype else ys*4*pi*xs**2 
  ys_mid = 0.5*(ys_c[1:] + ys_c[:-1])
  xs_diff = np.diff(xs)
  y_int = np.sum(xs_diff*ys_mid) if 'cum' not in f_types else arr_append(np.cumsum(xs_diff*ys_mid), vals_l=(0.0,))
  return y_int
 
 else:
  if 'sph' in ftype:
   spl = np.zeros((len_x-1, 6))
   spl[:,2:] = spline_calc(xs, ys, spl_type='cube herm') * 4 * pi
  else:
   spl = spline_calc(xs, ys, spl_type='cube herm')  
  
  if 'cum' in ftype:
   return spline_int(spl, xs, ftype='ys')
  else:
   spl_int = spline_int(spl, xs)
   int_r, int_l = spline_eval_arr(spl_int, xs, np.array((x_r, x_l)))
   return int_r - int_l 
   



from math import *
import numpy as np
import pdb
import time

v_dummy = np.array((0.0,))
arr_dummy = np.empty(0)

x_vec = np.array((1.0, 0.0, 0.0))
y_vec = np.array((0.0, 1.0, 0.0))
z_vec = np.array((0.0, 0.0, 1.0))

factorial_table = np.hstack((1.0, np.cumprod(np.arange(1,200).astype(float)) ))

mat_def = np.vstack((x_vec, y_vec, z_vec))

machine_eps = np.finfo(float).eps

#indices of derivatives used in 2D spline calculations
i_dx, i_dy, i_dxy, i_d2x, i_d2y = 0, 1, 2, 3, 4

#-----------speed-up of simple array creation/operation (2-3 times compared to hstack/vstack/append, on small arrays. With large/multiple inputs, use default functions)


#takes tuple or list of 1D arrays and stacks them like np but faster 
def arr_stack(arrs, ftype='h', dtype=np.float64): #(+)
 if 'h' in ftype:
  return arr_stack_h(arrs, dtype=dtype)
 else:
  return arr_stack_v(arrs, dtype=dtype)


def arr_stack_h(arrs, dtype=np.float64): #(+) 
 sizes = [x.size for x in arrs]
 arr_out = np.zeros(sum(sizes), dtype=dtype)
 i_cur = 0
 for i in range(len(arrs)):
  s_cur = sizes[i]
  arr_out[i_cur:i_cur+s_cur] = arrs[i]
  i_cur += s_cur
 return arr_out

 
def arr_stack_v(arrs, dtype=np.float64):
 s = arrs[0].size
 n_arrs = len(arrs)
 arr_out = np.zeros((n_arrs, s), dtype=dtype)
 for i in range(len(arrs)):
  arr_out[i] = arrs[i]
 return arr_out

 
#append vals_r to the beginning of an array and vals_r to the end
def arr_append(arr, vals_r=(), vals_l=(), dtype=np.float64): #(+)
 vs_l = np.array(vals_l, dtype=dtype) if len(vals_l) > 0 != () else arr_dummy
 vs_r = np.array(vals_r, dtype=dtype) if len(vals_r) > 0 != () else arr_dummy
 arr_fl = arr.astype(dtype)
 return arr_stack_h((vs_l, arr_fl, vs_r), dtype=dtype)

   
#stack 1D array vertically n_rep times 
def tile_vecs(arr, n_rep): #(+) #700k/s,
 len_arr = arr.size
 arr_out = np.zeros((n_rep, len_arr), dtype=arr.dtype)
 for i_r in range(n_rep):
  arr_out[i_r] = arr
 return arr_out

#determines if a complex value is machine-precision-close to real 
def close_to_real(v, tol=100): #(+)
 v_c = complex(v)
 t = machine_eps*tol if tol > 1 else tol
 if v_c.imag == 0:
  return True
 else:
  return abs(v.imag / v.real) < t if v_c.real != 0 else False
 
 
def close_to_real_arr(v, tol=100): #(+)
 t = machine_eps*tol if tol > 1 else tol
 mask_real = np.where( np.abs(v.imag / v.real) < t, True, False)
 mask_real[np.where(v.imag==0.0)] = True
 return mask_real


#returns sign of a number
def signum(x): #(+)
 return 1 if x > 0 else (-1 if x < 0 else 0) 


def signum_arr(xs): #(+)
 signs = np.zeros(xs.size)
 signs[np.where(xs < 0)] = -1.0
 signs[np.where(xs > 0)] = 1.0
 return signs
 
#determines if a value is between x_range[0] and x_range[1]
def inrange(x, x_range, ftype='[)'): #(+), both scalars and arrays
 x_l, x_r = x_range[0], x_range[1]
 cond_l = (x >= x_l) if '[' in ftype else (x > x_l)
 cond_r = (x <= x_r) if ']' in ftype else (x < x_r)
 return cond_l and cond_r


def inrange_arr(xs, x_range, ftype='[)'): #(+) 
 xs_l, xs_r = x_range
 cond_l = np.where(xs >= xs_l, True, False) if '[' in ftype else np.where(xs > xs_l, True, False)
 cond_r = np.where(xs <= xs_r, True, False) if ']' in ftype else np.where(xs < xs_r, True, False)
 mask_IR = cond_l * cond_r
 return mask_IR


#replaces value with lower/upper limit if outside them
def trunc_val(val, lims=(-1.0, 1.0)): #(+)
 return max(min(val, lims[1]), lims[0])

#checks for out-of-range elements and replaces them with limiting values
def trunc_arr(arr, lims=(-1.0,1.0)): #(+)
 lim_lo, lim_hi = lims
 is_low  = np.where(arr < lim_lo)[0]
 is_high = np.where(arr > lim_hi)[0]
 arr_out = np.copy(arr)
 arr_out[is_low] = lim_lo
 arr_out[is_high] = lim_hi
 return arr_out

#takes arrays with infs, nans and repeated values, creates sorted and filtered xs array
def arr_filter(xs, ys=arr_dummy, ftype='srt'): #(+)
 f_types = ftype.split(' ')
 ys_present = ys.size == xs.size
 if 'no_srt' not in f_types:
  i_srt = np.argsort(xs)
  xs_s = xs[i_srt]
  ys_s = ys[i_srt] if ys_present else ys
 else:
  xs_s, ys_s = xs, ys
 xs_diff = np.diff(xs_s)
 mask_un = np.ones(xs_s.size, dtype=np.bool_)
 mask_un[np.where(xs_diff==0)[0]+1] = False
 if 'interp' in f_types:
  mask_xs = mask_un * np.isfinite(xs_s)
  xs_un, ys_un = xs_s[mask_xs], ys_s[mask_xs]
  return xs_un, nan_interp(xs_un, ys_un)
 if ys_present:
  mask_xs = mask_un * np.isfinite(xs_s) * np.isfinite(ys_s)
  return xs_s[mask_xs], ys_s[mask_xs]
 else:
  mask_xs = mask_un * np.isfinite(xs_s)
  return xs_s[mask_xs]

#finds and interpolates y nan values
def nan_interp(xs, ys): #(+)
 mask_nan = np.where(np.isnan(ys), True, False)
 mask_fin = True ^ mask_nan
 xs_nan = xs[mask_nan]
 xs_f, ys_f = xs[mask_fin], ys[mask_fin]
 ys_interp = np.interp(xs_nan, xs_f, ys_f)
 ys_out = np.copy(ys)
 ys_out[mask_nan] = ys_interp
 return ys_out

  
#works both on scalars and arrays, real and complex
#see also: np.allclose(result_rgi, result_interpn, atol=1e-15)
#compares two values or arrays and returns 0 if equal
def val_compare(v1, v2):
 diff_rel = 1 - np.minimum(np.divide(v1, v2), np.divide(v2, v1))
 return diff_rel

#binomial coefficient for k choices of n 
def bin_coeff(n, k): #(+)
 v = round(prod([i for i in range(n-k+1, n+1)]) / prod([i for i in range(1, k+1)])) if n >= k else 0
 return v


def factorial(n):
 return factorial_table[n]
  
#------------function routines-----------------  



#nested dictionary access, for plotting and printing
#i.e. 'lambda x: x['a_val']*x['b_val']'
def eval_calc(pt, val):
 func_val = eval(val)
 return(func_val(pt))

def compose(f, g):
 return lambda x: f(g(x))

def compose_l(f_list):
 return reduce(compose, f_list, lambda x: x)

def compose_list(f_list):
 def composition(x):
  for callable in f_list:
   x = callable(x)
  return x
 return composition

def f_key(key):
 return lambda x: x[key] 

def f_keys(key_list): 
 f_list = [f_key(x) for x in key_list]
 return compose_l(f_list[::-1])

#'t': timeit, 'p': profile 
def perf_test(f): 
 def wrap(*args, **kwargs):
  t_st = time.time()
  for i in range(n_iters):
   vs = func(*args, **kwargs)
  t_el = time.time() - t_st
  print(func.__name__, '%.3e' % t_el, str(n_iters), '%.3e' % (n_iters / t_el))
 return wrap
   



def print_dict(d, ptype='simple'):
 d_p = {}
 if ptype=='simple':
  cond_print = lambda x: type(d[x]) in [float, int, str, np.float64]
 for key in d.keys():
  if cond_print(key):
   d_p[key] = d[key]
 pp.pprint(d_p)
 return None

 
# returns list of keys for shortened value name
# 'dS_i_dv_fix' -> ['S_i_ders', 'dv_fix']
def val_keys(val):
 is_entropes = True if val[:3] == '_S_' else False
 key_list = [] if not is_entropes else ['isentropes',]
 val = val[3:] if is_entropes else val
 if '_d' not in val:
  key_list.append(val)
 else:
  v, d = val.split('_d')
  val_key = (v[1:] if 'd2' not in v else v[2:]) + '_ders'
  der_key = 'd' + d
  key_list = key_list + [val_key, der_key]
 return key_list

#returns single function which applies list of keys to feos_data[mat] to extract value array
#f_k = val_f('d2F_dv2_fix')
#vs = f_k(feos_data[mat])
def val_f(val):
 key_list = val_keys(val)
 f_k = f_keys(key_list)  
 return f_k 


#nested dict keys: https://rowannicholls.github.io/python/advanced/dictionaries.html
#https://www.geeksforgeeks.org/python-get-all-values-from-nested-dictionary/ 
#returns list of lists of keys for a nested dictionary, i.e. eos values or rho_calc output
#OK!
def key_structure(vals, ks_pr=[]):
 key_lists = []
 for key in vals.keys():
  if type(vals[key]) != dict:
   key_lists.append(ks_pr + [key,])
  else:
   key_lists.extend(key_structure(vals[key], ks_pr + [key,]))
 return key_lists

def val_update(vals, key_list, val): #OK
 for key in key_list[:-1]:
  if key not in vals.keys():
   vals[key] = {}  
  vals = vals[key]
 vals[key_list[-1]] = val
 return None


def func_print(f):
 f_str = str(inspect.getsourcelines(func)[0])
 return f_str.strip("['\\n']").split(" = ")[1]

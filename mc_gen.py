from math import *
import numpy as np
import bisect
from mc_base import *

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

from fractions import Fraction


#===================math simple=========================



#-------------functions on |R -> |R------------------------

#returns gaussian PDF evaluated at x(s)
def gauss(x_0, sigma, x): #(+)
 return e**( -(x - x_0)**2 / (2*sigma**2)) / (2*pi*sigma**2)**0.5


#returns exponential sigmoid function of x value 
#use: adjust ionization calculation in LIR
def sigmoid(x):  #(+)
 if x < -100: 
  return 0
 elif x > 100:
  return 1
 else:
  return e**x/(e**x+1)


def sigmoid_arr(xs): #(+), float limits e**-750, sigmoid(37)
 len_x = xs.size
 mask_IR = inrange_arr(xs, (-100, 100))
 mask_r = np.where(xs > -100, True, False)
 vs = np.zeros(len_x)
 vs[mask_r] = 1.0
 vs[mask_IR] = e**xs[mask_IR]/(e**xs[mask_IR]+1)
 return vs

#returns 'x-squared' sigmoid
def sigmoid_sqr(x): #(+)
 return x**2/(x**2+1)
 
#returns values symmetric to sqrt(x) about zero point 
def sqrtabs(x):  #(+)
 return sqrt(x) if x >= 0 else -sqrt(-x)

def sqrtabs_arr(xs):  #(+)
 vs = np.where(xs >= 0, np.sqrt(xs), -1.0* np.sqrt(-1.0* xs))
 return vs

#returns -x**2 for negative x and x**2 for positive x 
def sqrabs(x):
 return x**2 if x >= 0 else -x**2

def sqrabs_arr(xs):
 vs = np.where(xs >= 0, xs**2, -1.0*xs**2) 
 return vs

#used for log-axis visualizations
#returns log(x) if x is positive else log(-x)
def log_f(x, base=10.0):
 if x > 0:
  return log(x, base)
 elif x < 0:
  return log(-1.0*x, base)
 else:
  return nan
 
def log_f_arr(xs):
 vs = np.log10(np.absolute(xs))
 vs[np.isinf(vs)] = nan 
 return vs
 

#==================integers and rational numbers======================


#gives prime numbers up to n
def erato(n): #5M / 1s, 50M/15s #(+)
 all_nums = np.arange(n+1)
 prime_mask = np.ones(n+1, dtype=np.bool_)
 for i_n in range(2, n):
  if not prime_mask[i_n]:
   continue
  inds_mult = i_n * all_nums[2:int(n/i_n)+1]
  prime_mask[inds_mult] = False 
 return all_nums[prime_mask][2:]


#(+)
#calculates rational approximation of a number, using continuous fraction
#additional parameters, order of influence: length -> prec -> max denominator -> cont frac threshold
def cont_frac(x_in, length=30, prec=0.0, ftype='ext', den_max=inf, best=False, cf_thr=0): #50kcalc/s / 10 iters
 x = x_in
 q = floor(x)
 x = x - q
 delta = 1
 ctr=0
 n, d, num, den = 0, 1, 1, 0
 nums, dens, cf = [], [], []
 while ctr < length and den <= den_max and delta > prec and 1 / x != q:
  ctr += 1
  nums.append(num)
  dens.append(den)
  cf.append(q)
  n, d, num, den = num, den, num*q + n, den*q + d
  q = floor(1 / x)
  x = 1 / x - q
  delta = abs(nums[-1] / (dens[-1]*x_in) - 1) if ctr > 1 else 1 #(abs(rationaltoreal(cftorational(cf))/x_in - 1) if prec > 0 else 1) 
  
 if ftype=='cf':
  return np.array(cf)
 
 nums, dens, cf = np.array(nums), np.array(dens), np.array(cf)# i_0 = 2 if (nums[1], dens[1]) == (0,1) else 1
 i_cf_max = 1 + np.argmax(cf[1:])
 cf_max = cf[i_cf_max]
 if best: #choose best continued fraction if maximum above threshold
  i_fr = i_cf_max if cf_max >= cf_thr else -1
 else: #choose last cf above threshold or last value
  i_fr = -1 if cf_thr == 0 or cf_max < cf_thr else 1 + np.where(cf[1:] >= cf_thr)[0][-1] 

 num, den, cf_next = nums[i_fr], dens[i_fr], cf[i_fr] #  cf_next = (cf[i_fr+1] if i_fr < cf.size-1 else cf_next)

 if ftype=='seq':
  return np.vstack((nums, dens))
 elif ftype=='all':
  return np.vstack((nums, dens, cf))
 elif ftype=='ext':
  return np.array((num, den, cf_next))
 else:
  return np.array((num, den))


#(+)
#additional parameters: length -> prec -> max denominator -> cont frac threshold
def cont_frac_arr(x_in, length=15, prec=0.0, ftype='ext', den_max=inf, best=False, cf_thr=0):
 n_vals = x_in.size
 x = np.copy(x_in)
 q = np.floor(x)
 x = x - q
 ctr=-1
 n, d, num, den = 0, 1, 1, 0
 nums, dens, cf = np.zeros((length, n_vals)), np.zeros((length, n_vals)), np.zeros((length, n_vals))
 while ctr < length-1: # and den <= den_max and delta > prec and 1 / x != q:
  ctr += 1
  nums[ctr] = num
  dens[ctr] = den
  cf[ctr] = q
  n, d, num, den = num, den, num*q + n, den*q + d
  q = np.floor(1 / x)
  x = 1 / x - q #  delta = abs(nums[-1] / (dens[-1]*x_in) - 1) if ctr > 1 else 1 #(abs(rationaltoreal(cftorational(cf))/x_in - 1) if prec > 0 else 1) 
  
 if ftype=='cf':
  return np.array(cf)
 
 cf_sub = np.where(dens > den_max, 0, cf)# i_0 = np.where((nums[1,:] == 0) * (dens[1,:] == 1), 2, 1)
 i_cf_max = 1 + np.argmax(cf_sub[1:], axis=0) #np.where(i_0==2, 2 + np.argmax(cf_sub[2:], axis=0), 1 + np.argmax(cf_sub[1:], axis=0))
 cf_max = cf[i_cf_max, np.arange(n_vals)]
 dens_mask = np.where(dens > den_max, True, False)# i_den_max_0 = dens_mask.argmin(axis=0)
 i_den_max = length - np.argmin(dens_mask[::-1], axis=0) - 1 #np.where(i_den_max_0 != 0, i_den_max_0, length-1)
 cf_pass = np.where(cf > cf_thr, True, False)
 i_cf_pass = length - np.argmax(cf_pass[::-1], axis=0) - 1
 
 if best: #choose best continued fraction if maximum above threshold
  i_fr = np.where(cf_max >= cf_thr, i_cf_max, i_den_max)
 else: #choose last cf above threshold or last value
  i_fr = np.where(i_den_max <= i_cf_pass, i_den_max, i_cf_pass)

 num, den, cf_next = nums[i_fr, np.arange(n_vals)], dens[i_fr, np.arange(n_vals)], cf[i_fr, np.arange(n_vals)] #  cf_next = (cf[i_fr+1] if i_fr < cf.size-1 else cf_next)

 if ftype=='seq':
  return np.concatenate((nums.T[:,None,:], dens.T[:,None,:]), axis=1).astype(int)
 elif ftype=='all':
  return np.concatenate((nums.T[:,None,:], dens.T[:,None,:], cf.T[:,None,:]), axis=1).astype(int)
 elif ftype=='ext':
  return np.array((num, den, cf_next)).T.astype(int)
 else:
  return np.array((num, den)).T.astype(int)

#calculate Farey sequence of order N for x
def farey(x, N):
 a, b = 0, 1
 c, d = 1, 0
 while (b <= N and d <= N):
  mediant = (a+c)/(b+d)
  if x == mediant:
   if b + d <= N:
    return a+c, b+d
   elif d > b:
    return c, d
   else:
    return a, b
  elif x > mediant:
   a, b = a+c, b+d
  else:
   c, d = a+c, b+d
 if abs(x - c/d) < abs(x - a/b):
  return c, d
 else:
  return a, b

#-----------------math on functions------------------



# power laws: f(x) ~ k*x**n, n = d(log(f(x))/d(log(x))
def pwr_law(func, x_1, delta_x=1e-6): #takes lambda function
 d_x = 1 + delta_x
 x_2 = x_1 * d_x
 y_1, y_2 = func(x_1), func(x_2)
 return pwr_law_calc(x_1, x_2, y_1, y_2) # return (log(y_2/y_1) / log(x_2/x_1))

def pwr_law_calc(x1, x2, y1, y2): #assumes positive ys
 if y2 > 0 and y1 > 0 and x1 > 0 and x2 > 0:
  return ( (log(y2) - log(y1)) / (log(x2) - log(x1)) )
 else:
  return nan

#returns d(func)/dx @ x-val, takes lambda function
def df_dx(func, x_val, delta=1e-6, ftype='forth abs'): #mind numerical precision
 step = delta if 'abs' in ftype else x_val * delta 
 x_l = x_val-step if ('back' in ftype or 'center' in ftype) else x_val
 x_r = x_val+step if ('forth' in ftype or 'center' in ftype) else x_val
 y_l, y_r = func(x_l), func(x_r)
 d_dx = (y_r - y_l) / (x_r - x_l) 
 if 'second' in ftype:
  x_m = 0.5*(x_r + x_l)
  y_m = func(x_m)
  d2_dx = 4*(y_r - 2*y_m + y_l) / (x_r - x_l)**2
  return d_dx, d2_dx
 return d_dx 


#-------misc functions----------------------
  
#reverse operation of vs - floor(vs / range)
#assumes no adjacent increments and no jumps more than rng/2 @ truncation
def trunc_restore(vs, rng=2*pi): #(+)
 n_vals = vs.size
 vs_diff = np.diff(vs)
 vs_diff_sign = np.where(vs_diff > 0, 1, -1)
 mask_trunc = np.hstack((False, np.where((vs_diff_sign[:-2] == vs_diff_sign[2:]) * (vs_diff_sign[:-2] != vs_diff_sign[1:-1]) * (np.abs(vs_diff[1:-1]) > rng/2), True, False), False))
 i_trunc = 1 + np.where((vs_diff_sign[:-2] == vs_diff_sign[2:]) * (vs_diff_sign[:-2] != vs_diff_sign[1:-1]) * (np.abs(vs_diff[1:-1]) > rng/2))[0]
 turn_incr = np.zeros(n_vals)
 turn_incr[i_trunc + 1] = np.where(vs_diff[i_trunc] > 0, -1, 1)
 turn_incr[1] = 1 if (vs_diff[0] < -rng/2 and vs_diff[1] > 0) else (-1 if (vs_diff[0] > rng/2 and vs_diff[1] < 0) else 0) 
 turn_incr[-1] = 1 if (vs_diff[-1] < -rng/2 and vs_diff[-2] > 0) else (-1 if (vs_diff[-1] > rng/2 and vs_diff[-2] < 0) else 0) 
 n_turns = np.cumsum(turn_incr)
 vs_restore = vs + rng*n_turns
 return vs_restore



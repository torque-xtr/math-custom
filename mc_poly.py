from math import *
import numpy as np
from numpy.polynomial import Polynomial as Poly 
from mc_base import *

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats


#==========================polynomial math============================

#everywhere, reversed order is implemented, p = coeffs * x**np.arange(deg) (c[0] + c[1]*x + ... + c[n]*x**n)
#to use np.roots, reverse

#------------------------calculate polynomials from arrays of xs and ys----------------


#generalized n-1 degree polynome evaluation for n pts xs and ys #OK (+, +)
#slower than polyfit
def poly_calc(x_vals, y_vals, ftype=''): #(+)
 f_types = ftype.split(' ')
 (xs, ys) = (x_vals, y_vals) if 'check' not in f_types else arr_filter(xs, ys, ftype='srt')
 len_x = xs.size
 if len_x == 2:
  x1, x2 = xs
  y1, y2 = ys
  a = (y2-y1)/(x2-x1)
  b = y1 - a*x1
  poly = np.array((b, a))
 elif len_x == 3:
  poly = sqr_calc(xs, ys)
 else:
  poly = poly_calc_gen(xs, ys)
 return poly #from lowest to highest power
 
def lin_calc(xs, ys): #(+)
 x1, x2 = xs
 y1, y2 = ys
 A = (y2 - y1) / (x2 - x1)
 B = y1 - x1*A
 return np.array([B, A])

def lin_calc_arr(xs, ys):
 x1, x2 = xs.T
 y1, y2 = ys.T
 A = (y2 - y1) / (x2 - x1)
 B = y1 - x1*A
 return np.array([B, A]).T


#assumes unique xs
#prev version in LIR_math
def sqr_calc(xs, ys): #(+)
 x1, x2, x3 = xs
 y1, y2, y3 = ys
 denom = (x1-x2) * (x1-x3) * (x2-x3)
 if denom == 0: 
  if x1 == x2: #assumes y1 == y2
   return lin_calc(np.array((x2, x3)), np.array((y2, y3)))
  else:
   return lin_calc(np.array((x1, x2)), np.array((y1, y2)))
 else:
  A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
  B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
  C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
 return np.array([C,B,A]) #y = ax2+bx+c


#from TB_math
def sqr_calc_arr(xs, ys): #(+) 
 n_vals = xs.shape[0]
 x1, x2, x3 = xs.T
 y1, y2, y3 = ys.T
 denom = (x1-x2) * (x1-x3) * (x2-x3)
 i_deg = np.where(denom == 0, True, False)
 i_deg_1 = np.where(x1 == x2, True, False)
 i_deg_2 = np.where(x2 == x3, True, False)
 i_deg_full = i_deg_1 * i_deg_2
 
 sqrs = np.zeros((n_vals, 3))*nan
 
 As = np.where(i_deg, 0, (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom )
 Bs = np.where(i_deg, nan, (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom)
 Cs = np.where(i_deg, nan, (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom)
 
 Bs[i_deg_1] = (y3[i_deg_1] - y2[i_deg_1]) / (x3[i_deg_1] - x2[i_deg_1])
 Cs[i_deg_1] = y3[i_deg_1] - x3[i_deg_1] / Bs[i_deg_1]
 
 Bs[i_deg_2] = (y2[i_deg_2] - y1[i_deg_2]) / (x2[i_deg_2] - x1[i_deg_2])
 Cs[i_deg_2] = y2[i_deg_2] - x2[i_deg_2] / Bs[i_deg_2]

 sqrs[:,2] = As
 sqrs[:,1] = Bs
 sqrs[:,0] = Cs
 
 sqrs[i_deg_full] = nan
 
 return sqrs

#calculate polynomial of arbitrary length xs and ys
#not suitable for arrays
def poly_calc_gen(xs, ys): #(+)
 len_x = xs.size
 deg = len_x - 1
 is_m = np.linspace(0, deg, len_x)
 xs_2D = xs.repeat(len_x).reshape((len_x,len_x)).T
 pwrs_2D = is_m.repeat(len_x).reshape((len_x,len_x))
 M = np.power(xs_2D, pwrs_2D) #M = [[x**i*(-1.0)**(i*(deg+1)) for x in xs] for i in range(deg+1)]
 comp_m = np.vstack((M, ys[np.newaxis,:], M)) #np.vstack((M, ys, M)) #np.array(M + [ys] + M, dtype=np.float64)
 coeffs = np.zeros(deg+2)
 for i in range(deg+2):
  mat_cur = comp_m[i:i+deg+1] # see numpy array slicing
  coeffs[i] = np.linalg.det(mat_cur)# C = [np.linalg.det(comp_m[d:d+len(xs)]) for d in range(len(xs)+1)] # print(M) # print(C)
 pwr_coeffs = is_m[-1] + 2.0
 poly = (coeffs / coeffs[0] * (-1.0)**pwr_coeffs )[1:] #np array, [0]*x**3 + [1]*x**2 + [2]**x + [3]
 return poly


#(+)
#returns degree of a polynomial (accounting for possible zero coefficients: [1,0,3,0] -> 2)
def poly_deg(poly):
 deg_0 = np.nonzero(poly)[0]
 return deg_0[-1] if deg_0.size > 0 else 0


#(+)
def poly_deg_arr(poly):
 pwrs = np.ones(poly.shape)*np.arange(poly.shape[-1])
 nz_mask = np.where(poly != 0, True, False)
 degs = (pwrs*nz_mask).max(axis=1)
 return degs.astype(np.int64)
  

#-----evaluation, derivative, integrals-------


#(+), incl.arrays
def poly_der(poly, n=1): 
 deg = poly.shape[-1] - 1
 pwrs_0 = np.linspace(0, deg, deg+1)
 pwrs = pwrs_0 if n == 1 else (pwrs_0*(pwrs_0-1) if n==2 else (pwrs_0*(pwrs_0-1)*(pwrs_0-2) if n==3 else arr_stack([a1 - i for i in range(n)], 'v').prod(axis=0)))
 der = (poly*pwrs)[...,n:] #(p * pwrs)[:-1] if n_dim == 1 else 
 return der if poly.ndim > 1 else der[:np.where(der != 0)[0][-1]+1]# if np.any(der != 0) else np.zeros(1))


#(+), incl.arrays
def poly_int(poly):
 deg = poly.shape[-1] - 1#poly_deg(poly) 
 n_polys = 1 if poly.ndim == 1 else poly.shape[0]
 pwrs = np.linspace(0, deg, deg+1) #np.indices(p.shape)[0][::-1]
 int_p_1 = poly / (pwrs + 1)
 int_p = np.concatenate((np.zeros((n_polys, 1)), int_p_1 ), axis=1) if n_polys > 1 else  np.hstack((np.zeros(1), int_p_1 ))
 return int_p if poly.ndim > 1 else (int_p[:np.where(int_p != 0)[0][-1]+1] if np.any(int_p != 0) else np.zeros(1))

#(+)
def poly_eval(poly, x):
 if isinf(x):
  if np.any(poly!=0):
   deg = np.nonzero(poly)[0][-1]
   return poly[deg] * x * inf if deg % 2 == 1 else inf
  else:
   return nan
 deg = poly.shape[-1] - 1
 pwrs = np.linspace(0, deg, deg+1)
 y = ( poly * x**pwrs ).sum()
 return y
 

#(+)   #assumes equal numbers of polys and xs, tile if needed
def poly_eval_arr(poly, xs):
 n_polys = poly.shape[0]
 degs = poly_deg_arr(poly)
 deg = poly.shape[-1] - 1
 pwrs = np.linspace(0, deg, deg+1) #np.indices((poly.shape[-1],))[0][::-1]
 xs_1 = xs[:, None]
 xs_pwrs = np.power(xs_1, pwrs)
 ys_pwrs = poly * xs_pwrs
 ys = np.where(np.isfinite(xs), ys_pwrs.sum(axis=1), xs**degs*poly[np.arange(n_polys),degs] )
 return ys


#evaluates definite integral of poly from a to b
def poly_int_eval(poly, a, b): #(+)
 i = poly_int(poly)
 integral = poly_eval(i, b) - poly_eval(i, a)
 return integral


#----------root finding----------------

def poly_solve(poly, y_val, x_range=(-inf, inf), x_0=nan, conds='[)', ftype='real'): #(+) #OK but mind precision of np.roots (
 f_types = ftype.split(' ')
 deg = poly_deg(poly)
 if deg > 3:
  eq = arr_append(poly[1:], vals_l=(poly[0] - y_val,))[::-1]
  roots_0 = np.roots(eq)
 elif deg == 0:
  roots_0 = np.array((nan,))
 else:
  roots_0 = lin_solve(poly[:deg+1], y_val) if deg == 1 else (sqr_solve(poly[:deg+1], y_val) if deg == 2 else cube_solve(poly[:deg+1], y_val))
 roots = roots_0[close_to_real_arr(roots_0)].real if 'real' in f_types else roots_0
 if deg > 1 and 'real' in f_types and roots.size > 1:
  i_srt = np.argsort(np.abs(roots - x_0)) if isfinite(x_0) else np.argsort(roots)
  return roots[i_srt]
 else:
  return roots

  
def poly_solve_arr(polys, y_vals, x_ranges=(-inf, inf), xs_0=None, conds='[)', ftype=''): #(+) 
 f_types = ftype.split(' ')
 n_eqs = y_vals.size
 degs = poly_deg_arr(polys)
 is_m = np.arange(degs.max()+1)
 roots = np.zeros((n_eqs, degs.max())).astype(complex)*nan
 is_lin, is_sqr, is_cube, is_higher = np.where(degs==1)[0], np.where(degs==2)[0], np.where(degs==3)[0], np.where(degs>3)[0]
 if is_cube.size > 0:
  roots[is_cube, :3] = cube_solve_arr(polys[is_cube,:4], y_vals[is_cube])
 if is_sqr.size > 0:
  roots[is_sqr, :2] = sqr_solve_arr(polys[is_sqr,:3], y_vals[is_sqr])
 if is_lin.size > 0:
  roots[is_lin, 0] = lin_solve_arr(polys[is_lin,:2], y_vals[is_lin])
 for i in is_higher:
  poly = polys[i]
  eq = arr_append(poly[1:], vals_l=(poly[0] - y_vals[i],))[::-1]
  roots_cur = np.roots(eq)
  roots[i, :degs[i]+1] = roots_cur
 i_srt = np.argsort(roots.real, axis=1)
 roots_srt = roots[np.arange(n_eqs)[:,None], i_srt]
 return roots_srt


#poly == poly[0] + poly[1] * x
def lin_solve(poly, y_val): #(+)
 b, a = poly
 x = (y_val - b) / a
 return np.array((x,))


def lin_solve_arr(polys, y_vals): #(+)
 b, a = polys.T
 xs = (y_vals - b) / a
 return xs


#solves a*x**2 + b*x + c == 0
#assumes a != 0
#returns all roots, boundaries are determined in poly_solve
#poly == poly[0] + poly[1] * x + poly[2] * x**2

def sqr_solve(poly, y_val): #(+)
 c_0, b, a = poly
 c = c_0 - y_val
 discr = b**2 - 4*a*c
 sqrt_discr = float(discr)**0.5 
 x1 = 0.5 * (-b - sqrt_discr) / a
 x2 = 0.5 * (-b + sqrt_discr) / a
 return np.array([x1, x2])


def sqr_solve_arr(polys, y_vals): #(+)
 c_0, b, a = polys.T 
 c = c_0 - y_vals
 discr = b**2 - 4*a*c
 if np.nonzero(discr < 0)[0].size > 0:
  x1 = 0.5 * (-b - np.sqrt(discr.astype(np.complex128))) / a
  x2 = 0.5 * (-b + np.sqrt(discr.astype(np.complex128))) / a  
 else:
  x1 = 0.5 * (-b - np.sqrt(discr)) / a
  x2 = 0.5 * (-b + np.sqrt(discr)) / a
 xs = np.vstack((x1, x2)).T
 return xs


#https://en.wikipedia.org/wiki/Cubic_equation
def cube_solve(poly, y_val): #(+)
 eps_1 = (-1)**(-1/3) * (-1.0)
 a, b, c, d_0 = poly[::-1]
 d = d_0 - y_val
 delta_0_r = b**2 - 3*a*c
 delta_1_r = 2*b**3 - 9*a*b*c + 27*a**2*d
 if delta_0_r == 0 and delta_1_r == 0:
  return np.ones(3, dtype=np.complex128) * (-b/(3*a))
 delta_0 = complex(delta_0_r)
 delta_1 = complex(delta_1_r)
 r_1 = (delta_1**2 - 4*delta_0**3)**0.5 
 crt = (0.5*(delta_1 + r_1))**(1/3)
 if abs(crt) == 0.0:
  crt = (0.5*(delta_1 - r_1))**(1/3)
 xs = np.array([(-1/(3*a)) * (b + eps_1**k*crt + delta_0 / (crt*eps_1**k)) for k in range(3)])
 return xs[np.argsort(np.abs(xs))]
 
 
def cube_solve_arr(polys, ys): #(+) #1.3M/s
 eps_1 = (-1)**(-1/3) * (-1.0)
 d_0, c, b, a = polys.T
 degs = poly_deg_arr(polys)
 n_xs = ys.size
 xs = np.zeros((n_xs, 3)).astype(np.complex128)*nan
 d = d_0 - ys
 delta_0 = b**2 - 3*a*c
 delta_1 = 2*b**3 - 9*a*b*c + 27*a**2*d
 r_1 = (delta_1**2 - 4*delta_0**3).astype(np.complex128)**0.5
 crt = (0.5*(delta_1 + r_1))**(1/3)
 i_crt_zero = np.where(crt==0, True, False)
 crt[i_crt_zero] = ((0.5*(delta_1 - r_1))**(1/3))[i_crt_zero]
 i_crt_zero = np.where(crt==0)[0]
 delta_by_crt = np.where( (delta_0==0) * (crt==0), 0.0, delta_0/crt)
 xs = np.array([(-1/(3*a)) * (b + eps_1**k*crt + delta_by_crt * eps_1**(-k)) for k in range(3)]).T
 return xs
 

#calculates extremums of a polynomial, inclusing -inf and inf
def poly_extr(poly): #(+)
 deg = poly_deg(poly)
 signum_high = signum(poly[deg])
 xtr_l, xtr_r = (inf, inf) if deg % 2 == 0 else (-inf * signum_high, inf * signum_high)
 d1p = poly_der(poly)
 roots_d1p = poly_solve(d1p, 0)
 xs_xtr, ys_xtr = [-inf,], [xtr_l,]
 for i_r in range(roots_d1p.size):
  r = roots_d1p[i_r]
  if close_to_real(r):
   xs_xtr.append(r.real)
   ys_xtr.append(poly_eval(poly, r.real))
 xs_xtr.append(inf)
 ys_xtr.append(xtr_r)
 return np.array( (xs_xtr, ys_xtr))

#calculates range of y values of polynomial from x_range[0] to x_range[1]
#written with for cycle because no more than 2 roots are usually expected 
def poly_range(poly, x_range): #(+)
 deg = poly_deg(poly)
 x_l, x_r = x_range
 if deg > 1:
  d1p = poly_der(poly)
  roots_d1p = poly_solve(d1p, 0)
  xs, ys = [x_l,], [poly_eval(poly, x_l)]
  for r in roots_d1p:
   if inrange(r, x_range):
    xs.append(r)
    ys.append(poly_eval(poly, r))
  xs.append(x_r)
  ys.append(poly_eval(poly, x_r))
  y_min, y_max = min(ys), max(ys)
 else:
  y_l, y_r = poly[0] + x_l*poly[1], poly[0] + x_r*poly[1]
  y_min, y_max = min(y_l, y_r), max(y_l, y_r)
 return y_min, y_max


#=======================(1/poly) integrals===============

#integrates 1 / poly == 1 / (a*x + b), from x1 to x2
def integrate_lin_inv(poly, x1, x2): #(+)
 b, a = poly
 return int_lin_inv(a, b, x2) - int_lin_inv(a, b, x1)


def int_lin_inv(a, b, x): #(+)
 l = a*x + b
 return log(l)/a if l > 0 else log(-l) / a 


#https://en.wikipedia.org/wiki/List_of_integrals_of_rational_functions
#https://www.integral-calculator.com/
#integrates 1 / poly == 1 / (a*x**2 + b*x + c)
def integrate_sqr_inv(poly, x1, x2): #(+)
 c, b, a = poly
 discr = 4*a*c - b**2
 if discr > 0:
  sq_d = sqrt(discr)
  return int_sqr_inv_1(a, b, c, sq_d, x2) - int_sqr_inv_1(a, b, c, sq_d, x1)
 elif discr < 0:
  sq_d = sqrt(-discr)
  return int_sqr_inv_2(a, b, c, sq_d, x2) - int_sqr_inv_2(a, b, c, sq_d, x1)
 elif discr == 0:
  return int_sqr_inv_3(a, b, x2) - int_sqr_inv_3(a, b, x1)

 
def int_sqr_inv_1(a, b, c, sq_d, x):
 return (2 / sq_d) * atan( (2*a*x + b) / sq_d)

def int_sqr_inv_2(a, b, c, sq_d, x):
 d = 2*a*x + b
 if abs(d) < sq_d:
  return (-2 / sq_d) * atanh(d/sq_d)
 else:
  return (-2 / sq_d) * atanh(sq_d/d)

def int_sqr_inv_3(a, b, x2):
 return -2 / (2*a*x2 + b)



from math import *
import numpy as np
from mc_base import *

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

#----------------------trigonometry advanced---------------------

def arctanh(x): #(+)
 return 0.5 * np.log((1 + x) / (1 - x))

def arccoth(x): #(+) (or arctanh(1/x)
 return 0.5 * np.log((x + 1) / (x - 1))

#-------------Stumpff functions: c, s, their derivatives----------

def c(z, thr=1e-5): # trig
 if z < -thr:
  c_val = ( (cosh((-z)**0.5) - 1.0) / (-z) )
 elif z < thr:
  c_val = 0.5 - z/24
 else:
  z_mod = sqrt(z) / (2*pi)
  n_period = round(z_mod)
  frac_per = abs(z_mod - n_period) 
  if n_period >= 1 and abs(frac_per) < thr:
   c_val = (frac_per * pi) ** 2 * 2 / z 
  else:
   c_val = (1.0 - cos(z**0.5)) / z
 return c_val


def c_arr(z, thr=1e-5):
 czs = np.zeros_like(z)
 z_mod = np.sqrt(z) / (2*pi)
 n_period = np.round(z_mod)
 frac_per = abs(z_mod - n_period) 
 i_neg = np.where(z < -thr)[0]
 i_near0 = np.where(np.abs(z) <= thr)[0]
 i_pos = np.where(z > thr)[0]
 i_small = np.where((frac_per <= thr) * (n_period > 0))[0]
 czs[i_neg] = ( (np.cosh((-z[i_neg])**0.5) - 1.0) / (-z[i_neg]) )
 czs[i_near0] = 0.5 - z[i_near0]/24 
 czs[i_pos] = (1.0 - np.cos(z[i_pos]**0.5)) / z[i_pos]
 czs[i_small] = ( (frac_per[i_small] * pi) ** 2 * 2 / z[i_small] )
 return czs


def c_sum(z, prec=1e-18): #(+) # 80 kcalc/s @ 1e-18 # sum #works bad in the vicinity of z_range, apply only near zero
 sigma = 0;
 i = 0
 add = 1
 while abs(add) > prec:
  add = (-1)**i * z**i / factorial_table[2*i+2]
  sigma = sigma + add
  i += 1
 return sigma

 
def c1(z, cz=nan, sz=nan, thr=1.5e-3): #(+)
 cz = c(z) if isnan(cz) else c(z)  
 sz = s(z) if isnan(sz) else s(z)
 if z == 0:
  return -1/24
 if abs(z) < thr:
  s1_0 = -1/24
  s2 = 1/360 #by brute force!
  s1_z = s1_0 + s2 * z # -1/24 + z/360
  return s1_z
 else:
  return (1 - z * sz - 2 * cz) / (2 * z) 



def s(z, thr=1e-5): # 1Mcalc/s # trig
 if abs(z) <= thr:
  s_val = 1/6 - z/120
 elif z > thr:
  s_val = ( (z**0.5 - sin(z**0.5)) / (z**1.5) )
 elif z < -thr:
  s_val = ( (sinh((-z)**0.5) - (-z)**0.5) / (-z)**1.5 )
 return s_val

def s_arr(z, thr=1e-5): # 1Mcalc/s # trig
 szs = np.zeros_like(z) 
 i_neg = np.where(z < -thr)[0]
 i_near0 = np.where(np.abs(z) <= thr)[0]
 i_pos = np.where(z > thr)[0]
 if i_neg.size > 0:
  sqrt_z_neg = np.sqrt(-z[i_neg])
  szs[i_neg] = (np.sinh(sqrt_z_neg) - sqrt_z_neg) / (-z[i_neg])**1.5
 if i_pos.size > 0:
  sqrt_z_pos = np.sqrt(z[i_pos])
  szs[i_pos] = (sqrt_z_pos - np.sin(sqrt_z_pos)) / z[i_pos]**1.5
 szs[i_near0] = 1/6 - z[i_near0]/120
 return szs



def s_sum(z, prec=1e-18): # (+)
 if z == 0:
  return (1/6)
 else:
  precision = 1e-18
  sigma = 0;
  i = 0
  add = 1
  while abs(add) > prec:
   add = (-1)**i * z**i / factorial_table[2*i+3]
   sigma = sigma + add
   i += 1
  return sigma


 
def s1(z, cz=nan, sz=nan, thr=1.5e-3):  #(+)
 if z == 0:
  return -1/120
 if abs(z) < thr:
  c1_0 = -1/120
  c2 = 1/2520
  c1_z = c1_0 + c2 * z # -1/120 + z/2520
  return c1_z
 else:
  cz = c(z) if isnan(cz) else c(z)  
  sz = s(z) if isnan(sz) else s(z)
  return 0.5 * (cz - 3 * sz) / z

def stumpff(z): #(+)
 cz = c(z)
 sz = s(z)
 c1z = c1(z, cz=cz, sz=sz)
 s1z = s1(z, cz=cz, sz=sz)
 stvals = np.array((cz, sz, c1z, s1z)) #{'cz':cz, 'sz':sz, 'c1z':c1z, 's1z':s1z, 'yz':yz, 'y1z':y1z}
 return stvals

#--------------------------

#calculates sine and cosine  by Taylor expansion 
def sin_sum(x, prec=1e-18): #(+)
 sigma = 0
 i = 1
 add = x
 precision = 1e-18
 while abs(add) > prec: 
  sigma = sigma + add
  add = (-1)**i * x**(2*i+1) / factorial(2*i + 1)
  i += 1
 return sigma

 
def cos_sum(x, prec=1e-18): #(+)
 sigma = 0
 i = 1
 add = 1
 while abs(add) > prec: 
  sigma = sigma + add
  add = (-1)**i * x**(2*i) / factorial(2*i)
  i += 1
 return sigma

  
 

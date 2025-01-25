from math import *
import numpy as np
from mc_base import *

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats



#=====================vector math========================
#all slicing removed, perform at call if needed
#all angles - in radians

#dot product  
def vecdot(v1, v2):
 x1, y1, z1 = v1
 x2, y2, z2 = v2
 dotprod = x1*x2 + y1*y2 + z1*z2
 return dotprod

 
def vecdot_arr(vecs1, vecs2): #(+)
 vs1 = vecs1 if vecs1.ndim == 2 else tile_vecs(vecs1, vecs2.shape[0])
 vs2 = vecs2 if vecs2.ndim == 2 else tile_vecs(vecs2, vecs1.shape[0])
 xs1, ys1, zs1 = vecs1[:,0], vecs1[:,1], vecs1[:,2]
 xs2, ys2, zs2 = vecs2[:,0], vecs2[:,1], vecs2[:,2]
 dotprods = xs1*xs2 + ys1*ys2 + zs1*zs2
 return dotprods


#angle between vec1 and vec2
def vecangle(vec1, vec2): # (+) #250k
 v1, v2 = vec1, vec2
 cos_phi = vecdot(v1, v2) / (vecdot(v1, v1) * vecdot(v2, v2))**0.5 # vecdot(v1, v2) / (veclen(v1)*veclen(v2))
 phi = acos(max(-1.0, min(cos_phi, 1.0)))
 return phi

  
def vecangle_arr(vecs1, vecs2): #(+)
 vs1 = vecs1 if vecs1.ndim == 2 else tile_vecs(vecs1, vecs2.shape[0])
 vs2 = vecs2 if vecs2.ndim == 2 else tile_vecs(vecs2, vecs1.shape[0])
 cos_phis_raw = vecdot_arr(vs1, vs2) / (vecdot_arr(vs1, vs1) * vecdot_arr(vs2, vs2))**0.5
 cos_phis = trunc_arr(cos_phis_raw, (-1.0, 1.0))
 phis = np.arccos(cos_phis)
 return phis

#cross product of v1 and v2  
def veccross(v1, v2):
 x1, y1, z1 = v1
 x2, y2, z2 = v2
 v = np.zeros(3)
 x = y1*z2 - z1*y2
 y = z1*x2 - x1*z2
 z = x1*y2 - y1*x2
 v = np.array((x,y,z))
 return v

def veccross_arr(vecs1, vecs2): #(+)
 vs1 = vecs1 if vecs1.ndim == 2 else tile_vecs(vecs1, vecs2.shape[0])
 vs2 = vecs2 if vecs2.ndim == 2 else tile_vecs(vecs2, vecs1.shape[0])
 xs1, ys1, zs1 = vs1[:,0], vs1[:,1], vs1[:,2]
 xs2, ys2, zs2 = vs2[:,0], vs2[:,1], vs2[:,2]
 xs = ys1*zs2 - zs1*ys2
 ys = zs1*xs2 - xs1*zs2
 zs = xs1*ys2 - ys1*xs2
 crossprods = np.vstack((xs,ys,zs)).T
 return crossprods


#length of a vector vec  
def veclen(vec): # (+) #600k
 x, y, z = vec
 return (x*x + y*y + z*z)**0.5

def veclen_arr(vecs): #(+)
 x = vecs[:, :3]
 xsqr = x**2
 xdots = np.sum(xsqr, axis=1)
 xlens = xdots**0.5
 return xlens


#normalize vector vec (return vector pointing in the same direction, with length 1.0)
def vecnorm(vec): # (+) #70k with np.any() check, 250k without
 v = vec
 vlen = sqrt(vecdot(v,v))
 return v / vlen if vlen != 0.0 else v # if np.any(v) != 0.0 else v # ъуъ
 
def vecnorm_arr(xs): # (+)
 return xs / (veclen_arr(xs)[:, np.newaxis])



#create rotation matrix for vecrot function # https://ru.wikipedia.org/wiki/Матрица_поворота 
def rotmat(axis, theta): #70k
 cos_th = cos(theta)
 sin_th = sin(theta)
 cos_th_ = 1 - cos_th
 x, y, z = vecnorm(axis)
 mat = np.array(((cos_th + cos_th_*x**2,  cos_th_*x*y - sin_th*z, cos_th_*x*z + sin_th*y),
                 (cos_th_*y*x + sin_th*z, cos_th + cos_th_*y**2,  cos_th_*y*z - sin_th*x), 
                 (cos_th_*z*x - sin_th*y, cos_th_*z*y + sin_th*x,  cos_th + cos_th_*z**2)))

 return mat

 
def rotmat_arr(axs, ths):
 axes = axs if axs.ndim == 2 else tile_vecs(axs, ths.size)
 thetas = np.ones(axs.shape[0])*ths if ths.size == 1 else ths 
 cos_th = np.cos(thetas)
 sin_th = np.sin(thetas)
 cos_th_ = 1.0 - cos_th
 axes_norm = vecnorm_arr(axes)
 x = axes_norm[:,0]
 y = axes_norm[:,1]
 z = axes_norm[:,2]
 
 mat_1 = np.vstack((cos_th + cos_th_*x**2,  cos_th_*x*y - sin_th*z, cos_th_*x*z + sin_th*y))
 mat_2 = np.vstack((cos_th_*y*x + sin_th*z, cos_th + cos_th_*y**2,  cos_th_*y*z - sin_th*x))
 mat_3 = np.vstack((cos_th_*z*x - sin_th*y, cos_th_*z*y + sin_th*x,  cos_th + cos_th_*z**2))
 mat2_raw = np.dstack((mat_1, mat_2, mat_3))

 mat2 = mat2_raw.transpose(1,2,0)
 
 return mat2


#rotates vector v about axis by theta angle
def vecrot(v, axis, theta):  #60k
 m = rotmat(axis, theta)
 v_r = m @ v
 return v_r

 
def vecrot_arr(vecs, axes, thetas):
 n_pts = vecs.shape[0] if vecs.ndim==2 else (axes.shape[0] if axes.ndim==2 else thetas.size) 
 vs = vecs if vecs.ndim == 2 else tile_vecs(vecs, n_pts)
 axs = axes if axes.ndim == 2 else tile_vecs(axes, n_pts)
 ths = np.ones(n_pts)*thetas if thetas.size == 1 else thetas
 inds = np.arange(n_pts)
 ms = rotmat_arr(axs, ths)
 vs_o = np.zeros_like(vs)
 xs = vs[:,0]
 ys = vs[:,1]
 zs = vs[:,2]
 vs_o[:,0] = ms[:,0,0] * xs + ms[:,0,1] * ys + ms[:,0,2] * zs
 vs_o[:,1] = ms[:,1,0] * xs + ms[:,1,1] * ys + ms[:,1,2] * zs
 vs_o[:,2] = ms[:,2,0] * xs + ms[:,2,1] * ys + ms[:,2,2] * zs
 return vs_o


#-------------------------vector projections----------------------------- 
 
#calculates projection of vector vec on axis ax, or on a plane with normal ax  
def vec_proj(vec, ax, ftype=''):
 f_types = ftype.split(' ')
 x, y, z = vec 
 a, b, c = ax
 vec_ax_dot = a*x + b*y + c*z
 ax_len = sqrt(a**2 + b**2 + c**2)
 dist = vec_ax_dot / ax_len
 if 'dist' in f_types:
  return dist
 else:
  vec_on_ax = ax * dist / ax_len #vector projection on axis
  if 'ax' in f_types:
   return vec_on_ax
  else:
   return vec - vec_on_ax
 

def vec_proj_arr(vecs_0, axs_0, ftype=''):
 f_types = ftype.split(' ')
 n_pts = vecs_0.shape[0] if vecs_0.ndim==2 else axs_0.shape[0]
 vecs = vecs_0 if vecs_0.ndim==2 else tile_vecs(vecs_0, n_pts)
 axs = axs_0 if axs_0.ndim==2 else tile_vecs(axs_0, n_pts)
 vec_ax_dot = (vecs * axs).sum(axis=1)
 ax_len = np.sqrt((axs**2).sum(axis=1))
 dist = vec_ax_dot / ax_len
 if 'dist' in f_types:
  return dist
 else:
  vec_on_ax = axs * dist[:,None] / ax_len[:,None] #vector projection on axis
  if 'ax' in f_types:
   return vec_on_ax
  else:
   return vecs - vec_on_ax

 
#-------------------------random vectors----------------------------


def vecrandom(): #(-)
 return vecnorm(np.random.uniform(-1, 1, 3)) 


def vecrandom_arr(n_vecs, ftype='uniform'): #(-)
 if 'gauss' in ftype:
  vecs = np.random.normal(0.0, 1.0, (n_vecs, 3)) 
  return vecnorm_arr(vecs) if 'norm' in ftype else vecs
 else:
  return vecnorm_arr(np.random.uniform(-1, 1, (n_vecs, 3)))
	 
#returns vector v + random addition, of length == len_rel * length of v
def vecrandmod(v, len_rel=0.1, ftype=''):
 vec_mod = vecrandom()
 vec = v + vec_mod * len_rel
 return vec if 'norm' not in ftype else veclen(v)*vecnorm(vec)

def vecrandmod_arr(v, n_vecs=-1, len_rel=0.1, ftype='uniform'):
 n = v.shape[0] if n_vecs == -1 else n_vecs
 vs = v if v.ndim == 2 else tile_vecs(v, n_vecs)
 vecs_mod = vecrandom_arr(n_vecs, ftype=ftype)
 vecs = vs + vecs_mod * len_rel
 return vecs if 'norm' not in ftype else veclen_arr(vs)[:,None] * vecnorm_arr(vecs)



from math import *
import numpy as np

from math_custom import *

import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.markers import MarkerStyle
import matplotlib.colors as colors
import matplotlib.projections as projections
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


#===============visualizations================

def spline_viz(xs, spls, plt_type='log log', val_type='', spl_type='', x_lims=None, y_lims=None):
 plt.ion()
 plt.show()
 if 'clf' in plt_type:
  plt.clf()
 xs_viz = np.hstack( [np.linspace(xs[i], xs[i+1], 11)[:-1] for i in range(len(xs)-1)] )
 #pdb.set_trace()
 ys_0 = spline_eval_arr(xs, spls, xs, val_type=val_type, spl_type=spl_type)
 ys_viz = spline_eval_arr(xs, spls, xs_viz, val_type=val_type, spl_type=spl_type)
 f_x = lambda x: log_f(x) if ('log lin' in plt_type or 'log log' in plt_type) else x
 f_y = lambda y: log_f(y) if ('lin log' in plt_type or 'log log' in plt_type) else y
 plt.scatter(f_x(xs), f_y(ys_0), marker='x', s=15)
 plt.plot(f_x(xs_viz), f_y(ys_viz))
 plt.grid(True)
 if x_lims != None:
  plt.xlim(x_lims)
 if y_lims != None:
  plt.ylim(y_lims)
 return None


def viz_2D(xs_0, ys_0, vs, xlims=None, ylims=None, z_lims=None, cmap='plasma', ftype='clf', xyz_type='lin lin lin',
           figax=None, plt_name='', x_label='', y_label='', plt_title=''): #(+), s=2, g_lim=0.
 plt.ion()
 plt.show()
 (xs, ys) = np.meshgrid(xs_0, ys_0, indexing='ij') if xs_0.ndim==1 else (xs_0, ys_0)
 xyz_types = xyz_type.split(' ' )
 xs_f = np.log10(xs) if xyz_type[0] == 'log' else xs
 ys_f = np.log10(ys) if xyz_type[1] == 'lin' else ys
 vs_f = np.log10(vs) if xyz_type[2] == 'lin' else vs
 
 if 'clf' in ftype:
  plt.close('all')

 fig, ax = plt.subplots() if figax == None else figax
 
 if z_lims==None:
  z_min, z_max, z_mid = np.nanmin(vs_f), np.nanmax(vs_f), nan
 else:
  z_min, z_max = z_lims[:2]
  z_mid = z_lims[2] if len(z_lims) > 2 else nan

 clr_norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=0.9*z_max, vmax=z_max) if isfinite(z_mid) else colors.Normalize(vmin=z_min, vmax=z_max)
  
 p = ax.pcolor(xs_f, ys_f, vs_f, norm=clr_norm, cmap=cmap, shading='auto')
 cb = fig.colorbar(p, ax=ax) #, extend='max')

 ax.set_xlabel(x_label)
 ax.set_ylabel(y_label)
 ax.set_title(plt_title)
 if xlims != None:
  ax.set_xlim(xlims)
 if xlims != None:
  ax.set_ylim(ylims)

 if plt_name != '':
  plt.savefig(plt_name, dpi=200)
 
 return fig, ax


def bool_viz(vs, xs=None, ys=None, ftype='clf', plt_name='', clr_true='b', clr_false='r', thr=1.0):
 len_x, len_y = vs.shape
 if type(xs) != np.ndarray:
  xs_v, ys_v = np.indices((len_x, len_y))
 elif xs.ndim == 1:
  xs_v, ys_v = np.meshgrid(xs, ys, indexing='ij')
 else:
  xs_v, ys_v = xs, ys
 vs_v = vs if vs.dtype == bool else np.where(vs_v > thr, True, False)
 plt.ion()
 plt.show()
 if 'clf' in ftype:
  plt.clf()
 
 clrs = np.zeros((len_x, len_y, 4))
 clrs[vs] = mpl.colors.to_rgba(clr_true)
 clrs[True ^ vs] = mpl.colors.to_rgba(clr_false)
 p = plt.imshow(np.transpose(clrs, (1,0,2)), aspect='auto', interpolation='none', origin='lower', extent=[xs_v[0,0],xs_v[-1,0],ys_v[0,0],ys_v[-1,-1]]) 

 if plt_name != '':
  plt.savefig(plt_name, dpi=200)
 return None

def poly_viz(poly, n_pts=50, span=1.5, ftype='clf'):
 xs_xtr, ys_xtr = poly_extr(poly)
 if xs_xtr.size > 3:
  x_l, x_r = xs_xtr[[1,-2]]
  x_m, x_span = 0.5*(x_l + x_r), x_r - x_l
 else:
  x_m = xs_xtr[1] if xs_xtr.size == 3 else 0
  y_m = poly_eval(poly, x_m)
  x_span = poly_eval(poly_der(poly_der(poly)), x_m) / 2 if poly_deg(poly) >= 2 else abs(y_m / poly[1])
 xs = np.linspace(x_m-span*x_span/2, x_m+span*x_span/2)
 ys = poly_eval_arr(tile_vecs(poly, xs.size), xs)
 plt.ion()
 plt.show()
 if 'clf' in ftype:
  plt.clf()
 plt.plot(xs, ys)
 plt.grid(True)
 return None


#======================aux functions for visualizations================


def vs_normalize(vs_0, lims=(nan, nan, nan)):
 mask_finite = np.where(np.isfinite(vs))
 v_l = lims[0] if np.isfinite(lims[0]) else vs[mask_finite].min()
 v_r = lims[1] if np.isfinite(lims[1]) else vs[mask_finite].max()
 v_m = lim_m if len(lims) > 2 and np.isfinite(lim_m) else 0.5 * (v_l + v_r)
 vs_l_raw = (vs - v_l) / (v_m - v_l) * 0.5
 vs_r_raw = 0.5 + (vs - v_m) / (v_r - v_m) * 0.5
 vs_l = np.maximum(0.0, vs_l_raw)
 vs_r = np.minimum(1.0, vs_r_raw)
 vs_ = np.where(vs <= v_m, vs_l, vs_r) # (np.maximum(v_l, np.minimum(v_h, vs)) - v_l) / (v_h - v_l)
 return vs_

#https://stackoverflow.com/questions/12875486/what-is-the-algorithm-to-create-colors-for-a-heatmap
#vs in range of [0,1)
#use colormaps instead
def val_to_clr(vs, ftype='#clr', pal='custom', g_lim=1.0):
 if pal == '': 
  hs = 1 - vs
  ss = np.ones_like(vs)
  ls = vs*0.5 #vs*0.5 #1 - np.abs(0.5 - vs)
  rf, gf, bf = hsl_to_rgb(hs, ss, ls)
 elif pal == 'custom':
  rf = np.minimum(1.0, np.maximum(0.0, (vs-0.25)*4.0))
  gf = np.minimum(1.0, np.maximum(0.0, -1.0 * (np.abs(2.0 - vs*4.0) -2)) )
  bf = np.minimum(1.0, np.maximum(0.0, (vs- 3/4) * -4.0))
 elif pal == 'heat':
  rf, gf, bf = T_to_rgb(vs)
 ri, gi, bi = rgb_to_int(rf, gf, bf, g_lim=g_lim)
 clrs = rgb_to_clr(ri, gi, bi, ftype=ftype)  
 return clrs

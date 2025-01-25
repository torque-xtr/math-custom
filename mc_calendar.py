#check functions from trailblaser

from math import *
import numpy as np
from datetime import datetime as dt
import ephem

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

#======================calendar==========================

m_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
m_len_sums = np.hstack((0, np.cumsum(m_lengths)[:-1]))
m_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

t_noon = np.array('12:00:00')
jd_unix = 2440587.5
dt_unix = np.datetime64('1970-01-01 00:00:00')
d_current = dt.now()
yr_cur, mth_cur = d_current.timetuple()[0:2]
sm_stats = (12,1,3) #default daily stats smoothing (window half-width in days, poly degree, repeat number


#--------------polarfit----------------

#OK #https://astroconverter.com/utcmjd.html to check
def isleap(yr): #(+)
 is_leap = yr % 4 == 0 and yr % 400 not in (100, 200, 300)
 return is_leap
 
#conversion between Modified Julian Day and year float value
#@jit(nopython=True)
def jd_to_yr(jd): # (+) #200Mpts/s
 yr = (jd - jd_unix)/365.2425 + 1970
 return yr

def yr_to_jd(yr): #(+)
 jd = 365.2425 * (yr - 1970) + jd_unix
 return jd
 
#conversion of month-and-day values to day number (1 - 366) 
#@jit(nopython=True)
def md_to_daynum(m, d): #(+)
 return d + m_len_sums[m-1] #np.sum(m_lengths[:m-1])  

#daynum from 1 to 366
def daynum_to_md(dn, isleap=True): #(+)
 n_m = np.searchsorted(m_len_sums, dn) - 1 #np.where(dn > m_len_sums)[0][-1]
 n_d = dn - m_len_sums[n_m]
 return n_m + 1, n_d.astype(int)

def ymd_to_date(y, m, d): #(+)
 filler_1, filler_2 = np.array('-'), np.array(' ')
 date_strs = y.astype(int).astype(str) + filler_1 + np.char.zfill(m.astype(int).astype(str), 2) + filler_1 + np.char.zfill(d.astype(int).astype(str), 2) + filler_2 + t_noon
 dates = date_strs.astype(np.datetime64)
 return dates


#takes np.datetime64
def date_to_ymd(d): #(+)
 if d.ndim == 0:
  dt_str = str(d)
  d_str = dt_str.split('T')[0]
  ymd = np.array(d_str.split('-')[:3]).astype(int)
  return ymd 
 else:
  dt_str = d.astype(str)
  date_str = np.array([x[0] for x in np.char.split(dt_str, 'T')])
  ymds = np.char.split(date_str, '-')
  yrs_0 = np.array([int(x[-3]) for x in ymds])
  ms = np.array([int(x[-2]) for x in ymds])
  ds = np.array([int(x[-1]) for x in ymds])
  yrs = np.where(np.char.index(date_str, '-') == 0, yrs_0*-1.0, yrs_0)
  ymd = np.array((yrs, ms, ds)).T
  return ymd

#d in tuple/string/datetime/number, used to check
def date_to_jd_eph(d): #(+)
 jd = 2415020.0 + ephem.Date(d)
 return jd

#takes array [y, m, d] or array of yrs, mths, ds, returns julian day
def ymd_to_jd(d): #(+)
 yrs, ms, ds = d.T.astype(int) # t_log[:,1].astype(int), t_log[:,2].astype(int), t_log[:,3].astype(int)
 dates = ymd_to_date(yrs, ms, ds)
 jds = (dates - dt_unix) / np.timedelta64(1, 'D') + jd_unix
 return jds

def jd_to_ymd(jd): #(+) #450k/s (array)
 jd_sh = np.int64(jd - jd_unix - 0.5)
 d = jd_sh.astype('datetime64[D]')
 ymd = date_to_ymd(d)
 return ymd

def jd_to_date(jd): #(+)
 jd_sh = np.int64(jd - jd_unix - 0.5)
 d = jd_sh.astype('datetime64[D]')
 return d
  

#------------trailblazer------------


#string in, float out. zero on jan 1, 1970, 00:00:00 gregorian.
#m_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#m_len_sums = np.hstack((0, np.cumsum(m_lengths)[:-1]))
#m_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

def date_to_unix(t): #(+)
 mths = ['jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
 mth_nums = [i for i in range(1, 13)]
 mth_ds = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
 
 i_st = [x.isdigit() for x in t].index(True)
 neg_yr = i_st > 0 and t[i_st-1] == '-'
 date_in = t[i_st:].replace('"', '').replace('-', ' ').replace('??', '01').split(' ')
 yr = int(date_in[0]) * (-1.0 if neg_yr else 1.0)
 mth = date_in[1]
 d = int(date_in[2].rstrip(','))
 time_in = date_in[3].split(':') if len(date_in) > 3 else ('00','00','00')
 seconds = float(time_in[0])*3600 + float(time_in[1])*60 + float(time_in[2])
 
 i_month = mths.index(mth.lower()) if not mth.isnumeric() else int(mth)-1
 mth_days = mth_ds[i_month]
  
 days = mth_days + d - 1 # 0 on Jan 1, 364 on Dec 31; number of whole remaining days in a non-leap year;
 n_years = yr - 1970 
 
 if yr >= 1970: 

  n_leap = (yr - 1969) // 4  #all positive
  n_century_leap = (yr - 1901) // 100 
  n_quadrennial = (yr - 1601) // 400
  n_leap_total = n_leap - n_century_leap + n_quadrennial

  if mth_days >= 59 and yr % 4 == 0 and yr % 400 != 100 and yr % 400 != 200 and yr % 400 != 300:
   n_leap_total += 1  

  days_total = n_years*365 + days + n_leap_total
  
 else:

  n_leap = (yr - 1968) // 4  #all negative
  n_century_leap = (yr - 1900) // 100 
  n_quadrennial = (yr - 1600) // 400
  n_leap_total = n_leap - n_century_leap + n_quadrennial

  if days <= 58 and yr % 4 == 0 and yr % 400 != 100 and yr % 400 != 200 and yr % 400 != 300:
   n_leap_total -= 1  
   
  days_total = n_years*365 + days + n_leap_total 
 
 seconds_total = days_total * 86400 + seconds 
  
 return seconds_total

def ymd_to_unix(d_in): #valid only for yrs 1000-9999 
 
 ymdd = d_in.split('.')
 if len(ymdd[0]) == 7:
  d = d_in[:4] + '0' + d_in[4:]
 else:
  d = d_in
  
 yr = d[:4]
 m = int(d[4:6])
 d = d[6:8]
 s = 86400 * (float(d_in) % 1)
 
 month = ''
 if m == 1:
  month = 'Jan'
 elif m == 2:
  month = 'Feb'
 elif m == 3:
  month = 'Mar'
 elif m == 4:
  month = 'Apr'
 elif m == 5:
  month = 'May' 
 elif m == 6:
  month = 'Jun' 
 elif m == 7:
  month = 'Jul'
 elif m == 8:
  month = 'Aug'
 elif m == 9:
  month = 'Sep' 
 elif m == 10:
  month = 'Oct'
 elif m == 11:
  month = 'Nov'
 elif m == 12:
  month = 'Dec'
  
 date = yr + ' ' + month + ' ' + d + ' ' + '00:00:00'
# print(date)
 
 return date_to_unix(date) + s
 

def juliantounix(t):
 return ((t - 2440587.5) * 86400)
 
def unixtojulian(t):
 return (t / 86400 + 2440587.5)
  

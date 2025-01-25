from math import *
import numpy as np

from decimal import *
prec_dec = 32
getcontext().prec = prec_dec

#-----debugging and profiling
import pdb
import timeit
from cProfile import Profile
from pstats import SortKey, Stats


ints_dec = [Decimal(n) for n in range(4000)]
factorial_table_dec = [1, ]
for i in range(1, len(ints_dec)):
 factorial_table_dec.append(factorial_table_dec[i-1] * ints_dec[i])


#=================arbitrary precision math====================
 
def factorial_dec(n):
 return factorial_table_dec(n)

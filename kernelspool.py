import numpy
import os
package_path = os.path.dirname(os.path.abspath(__file__))+'/'
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.integrate import trapz, simps
from itertools import product
import multiprocessing as mp
## Define functions

#constants

#Define Q:
def Q1approx( r, x):
    return 0.25*(1 + x)**2

def Q1( r, x):
    y =  1 + r**2 - 2*r*x
    return (r**2 *(1 - x**2)**2)/y**2

def Q2approx( r, x):
    return 0.25*x*(1 + x)

def Q2( r, x):
    y =  1 + r**2 - 2*r*x
    return ((1- x**2)*r*x*(1 - r*x))/y**2

def Q3approx( r, x):
    return 0.25*x**2

def Q3( r, x):
    y =  1 + r**2 - 2*r*x
    return (x**2 *(1 - r*x)**2)/y**2

def Q5approx( r, x):
    y =  1 + r**2 - 2*r*x
    return x*(1 + x)/2.

def Q5( r, x):
    y =  1 + r**2 - 2*r*x
    return r*x*(1 - x**2.)/y

def Q8approx( r, x):
    y =  1 + r**2 - 2*r*x
    return (1 + x)/2.

def Q8( r, x):
    y =  1 + r**2 - 2*r*x
    return r**2.*(1 - x**2.)/y

def Qs2approx( r, x):
    y =  1 + r**2 - 2*r*x
    return (1 + x)*(1- 3*x)/4.

def Qs2( r, x):
    y =  1 + r**2 - 2*r*x
    return r**2.*(x**2 - 1.)*(1 - 2*r**2 + 4*r*x - 3*x**2)/y**2

dictQapprox = {1:Q1approx, 2:Q2approx, 3:Q3approx, 5:Q5approx, 8:Q8approx, -1:Qs2approx}
dictQ = {1:Q1, 2:Q2, 3:Q3, 5:Q5, 8: Q8, -1:Qs2}


#####Define double integral here
def Q_internal(k, n, r, approx = False):
        
    if approx:
        func = lambda r, x: ilpk(k*numpy.sqrt(1 + r**2 - 2*r*x))*dictQapprox[n](r, x)
    else:
        func = lambda r, x: ilpk(k*numpy.sqrt(1 + r**2 - 2*r*x))*dictQ[n](r, x)
    return (glwval*func(r, glxval)).sum(axis = -1)

def Q_external(n, k):
    fac = k**3 /(2*numpy.pi)**2
    r = (kint/k)
    absr1 = abs(r-1)
    mask = absr1 < tol
    y = numpy.zeros_like(r)
    y[~mask]  = Q_internal(k, n, r[~mask].reshape(-1, 1))
    if mask.sum():
        y[mask] = Q_internal(k, n,  r[mask].reshape(-1, 1), approx = True)
    y *= pkint
    return fac*trapz(y, r)


def Q(kv, pk, npool = 4, ns = None, kintv = None):

    global ilpk, tol, kint, pkint, glxval, glwval

    ilpk = pk
    if kintv is None:
        kint = numpy.logspace(-6, 3, 1e3)
    else:
        kint = kintv
    pkint = ilpk(kint)

    tol = 10**-5. 
    glxval, glwval = numpy.loadtxt(package_path + "/gl_128.txt", unpack = True)
    glxval = glxval.reshape(1, -1)
    glwval = glwval.reshape(1, -1)

    if ns is None:
        ns = [1, 2, 3, 5, 8, -1]

    pool = mp.Pool(npool)
    prod = product(ns, kv)
    Qv = pool.starmap(Q_external, list(prod))
    pool.close()
    pool.join()
    Qv = numpy.array(Qv).reshape(len(ns), -1)
    toret = numpy.zeros([Qv.shape[0] + 1, Qv.shape[1]])
    toret[0] = kv
    toret[1:, :] = Qv

    del ilpk, tol, kint, pkint, glxval, glwval

    return toret

###############################################################


def R1approx(r, x):
    return 0.5*(1- x)*(1 + x)**2

def R1(r, x):
    y =  1 + r**2 - 2*r*x
    return (r**2 *(1 - x**2)**2)/y

def R2approx(r, x):
    return 0.5*x*(1 + x)*(1-x)

def R2(r, x):
    y =  1 + r**2 - 2*r*x
    return ((1- x**2)*r*x*(1 - r*x))/y

listRapprox = [R1approx, R2approx]
listR = [R1, R2]


#Double integral here

def R_internal(n, r, approx = False):

    if approx:
        func = lambda r, x: listRapprox[n-1](r, x)
    else:
        func = lambda r, x: listR[n-1](r, x)
    return (glwval*func(r, glxval)).sum(axis = -1)

def R_external(n, k):

    fac = k**3 /(2*numpy.pi)**2 *ilpk(k)
    r = (kint/k)
    absr1 = abs(r-1)
    mask = absr1 < tol
    y = numpy.zeros_like(r)
    y[~mask]  = R_internal(n, r[~mask].reshape(-1, 1))
    if mask.sum():
        y[mask] = R_internal(n,  r[mask].reshape(-1, 1), approx = True)
    y *= pkint
    return fac*trapz(y, r)



def R(kv, pk, npool = 4, ns = None, kintv = None):

    global ilpk, tol, kint, pkint, glxval, glwval

    ilpk = pk
    if kintv is None:
        kint = numpy.logspace(-6, 3, 1e3)
    else:
        kint = kintv
    pkint = ilpk(kint)

    tol = 10**-5. 
    glxval, glwval = numpy.loadtxt(package_path + "/gl_128.txt", unpack = True)
    glxval = glxval.reshape(1, -1)
    glwval = glwval.reshape(1, -1)

    if ns is None:
        ns = [1, 2]

    pool = mp.Pool(npool)
    prod = product(ns, kv)
    Rv = pool.starmap(R_external, list(prod))
    pool.close()
    pool.join()
    Rv = numpy.array(Rv).reshape(len(ns), -1)
    toret = numpy.zeros([Rv.shape[0] + 1, Rv.shape[1]])
    toret[0] = kv
    toret[1:, :] = Rv

    del ilpk, tol, kint, pkint, glxval, glwval

    return toret

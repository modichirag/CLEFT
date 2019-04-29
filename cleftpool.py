import numpy
import numpy as np
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function. 
#Divide the funciton by this to get the correct form 

from scipy.integrate import romberg
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from qfuncpool import Qfunc
from itertools import repeat
from functools import partial
import multiprocessing as mp


class CLEFT():
    '''
    Class to evaluate CLEFT kernels to calculate Power Spectrum Components given a linear power spectrum
    in the form or 'k', 'p'
    or a power spectrum file - 'pfile'
    # bn and b1bn are not implemented yet
    '''
    
    def __init__(self, k = None, p = None, pfile = None, qfile = None, rfile = None, ensfile = None, \
                 npool = 4, extrapker = True, saveqfile = None, saveQRfile = None, order=2):
        if pfile is None:
            if p is None:
                print("Specify the power sepctrum file or the array")
        else:
            k, p = np.loadtxt(pfile, unpack = True)
        self.kp = k
        self.p = p
        self.order = order

        if ensfile is None:
            self.qf = Qfunc(k, p, Qfile=qfile, Rfile = rfile, npool = npool, \
                            extrapker = extrapker, saveqfile = saveqfile, order=self.order)
            print("Q & R kernels created")

            self.setup_dm()
            print("Matter q-functions created")
            self.setup_blocal()
            print("Bias(local) q-functions created")
            self.setup_bshear()
            print("Shear q-functions created")

        else:
            print('Reading ENS File')
            self.readens(ensfile)

        self.renorm = numpy.sqrt(numpy.pi/2.) #mcfit normaliztion
        self.tpi2 = 2*numpy.pi**2.
        self.jn = 20 #number of bessels to sum over
        
    def setup_dm(self):
        qf = self.qf

        #Linear
        self.xi00lin = qf.xi0lin0()
        self.qv, xi0lin = qf.xi0lin() #qv determined here
        xi2lin = qf.xi2lin()[1]
        self.Xlin = 2/3.*(self.xi00lin - xi0lin - xi2lin)
        ylinv = 2*xi2lin

        #Since we divide by ylin, check for zeros
        mask = (ylinv == 0)
        ylinv[mask] = interpolate(self.qv[~mask], ylinv[~mask])(self.qv[mask])
        self.Ylin = ylinv

        #Useful variables here
        self.XYlin = (self.Xlin + self.Ylin)
        self.sigma = self.XYlin[-1]
        self.yq = (1*self.Ylin/self.qv)

        #Loop
        if self.order == 2:
            self.xi00loop = qf.xi0loop0()
            xi0loop = qf.xi0loop()[1] 
            xi2loop = qf.xi2loop()[1]
            self.Xloop = 2/3.*(self.xi00loop - xi0loop - xi2loop)
            self.Yloop = 2*xi2loop
            self.XYloop = (self.Xloop + self.Yloop)
            self.sigmaloop = self.XYloop[-1]

            #Loop2
            self.xi1loop = qf.xi1loop(tilt = 0.5)[1]
            self.xi3loop = qf.xi3loop(tilt = 0.5)[1]
        else:
            self.Xloop, self.Yloop, self.XYloop, self.xi1loop, self.xi3loop = [np.zeros_like(self.qv) for i in range(5)]
            self.sigmaloop = 0
        
    def setup_blocal(self):
        qf = self.qf

        self.u10 = qf.u10()[1]
        self.corr = qf.corr()[1]

        if self.order == 2:
            self.x10 = qf.x10()[1]
            self.y10 = qf.y10()[1]
            self.u11 = qf.u11()[1]
            self.u20 = qf.u20()[1]
            self.u30 = qf.u3()[1]
        else:
            self.x10, self.y10, self.u11, self.u20, self.u30 = [np.zeros_like(self.qv) for i in range(5)]
                                                            
        #qil,  qih = np.where(self.qv>1e-2)[0][0], np.where(self.qv>300)[0][0]
        #tpi = qf.loginterp(self.qv[qil:qih], tp[qil:qih])(self.qv)

    def setup_bshear(self):
        qf = self.qf
        js = qf.jshear()
        js2 = qf.js2()

        self.chi = 4/3.*js2[1]**2 # this is actually \chi12 in appendix
        self.zeta = qf.zeta()[1]
        self.v12 = 4*js[1]*js2[1]
        #terms for Upsilon
        self.x20 = 4*js[2]**2
        self.y20 = 2*(3*js[1]**2 + 4*js[1]*js[2] + 2*js[1]*js[3] + 2*js[2]**2 + 4*js[2]*js[3] + js[3]**2)
        
        if self.order == 2:
            self.v10 = qf.v10()[1]
        else:
            self.v10 = np.zeros_like(self.qv)

    def readens(self, fname):

        ens = np.loadtxt(fname).T
        self.qv = ens[0]

        #Linear
        self.xi00lin = ens[1]
        xi0lin = ens[2]
        xi2lin = ens[3]
        self.Xlin = 2/3.*(self.xi00lin - xi0lin - xi2lin)
        ylinv = 2*xi2lin

        #Since we divide by ylin, check for zeros
        mask = (ylinv == 0)
        ylinv[mask] = interpolate(self.qv[~mask], ylinv[~mask])(self.qv[mask])
        self.Ylin = ylinv

        #Useful variables here
        self.XYlin = (self.Xlin + self.Ylin)
        self.sigma = self.XYlin[-1]
        self.yq = (1*self.Ylin/self.qv)

        #Loop
        self.xi00loop = ens[4] + ens[5]
        xi0loop = ens[6] + ens[7]
        xi2loop = ens[8] + ens[9]
        self.Xloop = 2/3.*(self.xi00loop - xi0loop - xi2loop)
        self.Yloop = 2*xi2loop
        self.XYloop = (self.Xloop + self.Yloop)
        self.sigmaloop = self.XYloop[-1]
        
        #Loop2
        self.xi1loop = ens[10]
        self.xi3loop = ens[11]
        
        self.corr = ens[12]
        self.zeta = ens[13]
        self.chi = ens[14]

        self.u10 = ens[15]
        self.u30 = ens[16]
        self.u11 = ens[17]
        self.u20 = ens[18]
        
        self.v10 = ens[19]
        self.v12 = ens[20]
        self.x10 = ens[21]
        self.y10 = ens[22]
        self.x20 = ens[23]
        self.y20 = ens[24]
        


#####Do Bessel Integrals Here

#Declaring global variables to be able to use pool without
#passing around huge arrays which will have to be pickled
#causing lot of overhead        


def template0(k, func, extrap = False):
    '''Template for j0 integral
    '''
    expon0 = numpy.exp(-0.5*k**2 * (XYlin - sigma))
    suppress = numpy.exp(-0.5*k**2 *sigma)
    Fq = expon0*func(k = k, l= 0)
    if abs(func(k =k, l=0)[-1] ) > 1e-7:
        #print("Subtracting large scale constant = ", func(0)[-1], k)
        sigma2 = func(k = k, l= 0)[-1]
        #print(sigma2)
        Fq -= sigma2
    ktemp, ftemp = \
            sph(qv, nu= 0, q=1.5)(Fq*renorm,extrap = extrap)
    ftemp *= suppress
    return 1* numpy.interp(k, ktemp, ftemp)*4*np.pi


def template(k, func, extrap = False):
    '''Template for higher bessel integrals
    '''
    expon = numpy.exp(-0.5*k**2 * (XYlin))
    Fq = 0
    toret = 0
    for l in range(1, jn):
        Fq = expon*func(k = k, l= l)*yq**l
        ktemp, ftemp = \
                sph(qv, nu= l, q=max(0, 1.5 - l))(Fq*renorm, extrap = extrap)
        toret += 1* k**l * numpy.interp(k, ktemp, ftemp)
    return toret*4*np.pi



def integrate(func, pool, taylor = 0):
    '''Do the \mu integrals for all j's by calling the templates
    '''

    p0  = pool.starmap(template0, zip(kv, repeat(func)))
    
    pl  = pool.starmap(template, zip(kv, repeat(func)))
    
    p0, pl = np.array(p0), np.array(pl)

    toret = p0 + pl
    if taylor:
        factorial = np.arange(1, taylor + 1)
        toret *= kv**taylor / factorial.prod()

    return toret


def make_table(cl, kmin = 1e-3, kmax = 3, nk = 100, npool = 2, z = 0, M = 0.3, order=2):
    '''Make a table of different terms of P(k) between a given
    'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k.
    Called with a CLEFT object that has all the kernels.
    The order is 
    k, ZA, A, W, b1, b1^2, b2, b2^2, b1b2, bs2, b1bs2, b2bs2, bs2^2, bn, b1bn

    Can specify a number of cores (npool), redshift(z) and Omega_m (M)
    to calculate as a function of redshift
    '''

    header = "k[h/Mpc]   P_Zel   P_A    P_W    P_d    P_dd     P_d^2    P_d^2d^2 \
 P_dd^2    P_s^2    P_ds^2    P_d^2s^2   P_s^2s^2   P_D2d     P_dD2d"

    pktable = numpy.zeros([nk, len(header.split())])
    dg2 = Dgrow(z, M)**2
    dg4 = dg2**2

    global kv
    kv = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), nk)

    global qv, XYlin, sigma, yq, renorm, jn
    qv, XYlin, sigma, yq, renorm, jn = cl.qv, cl.XYlin*dg2, cl.sigma*dg2, cl.yq*dg2, cl.renorm, cl.jn

    global Xlin, Ylin, Xloop, Yloop, xi1loop, xi3loop, x10, y10, x20, y20
    Xlin, Ylin, Xloop, Yloop, xi1loop, xi3loop, x10, y10, x20, y20 = \
                cl.Xlin*dg2, cl.Ylin*dg2, cl.Xloop*dg4, cl.Yloop*dg4, cl.xi1loop*dg4, cl.xi3loop*dg4, \
                cl.x10*dg4, cl.y10*dg4, cl.x20*dg4, cl.y20*dg4

    global u10, u30, u11, u20, v10, v12, corr, chi, zeta
    u10, u30, u11, u20, v10, v12, corr, chi, zeta = cl.u10*dg2, cl.u30*dg4, cl.u11*dg4, cl.u20*dg4, \
                                                    cl.v10*dg4, cl.v12*dg4, cl.corr*dg2, cl.chi*dg4, cl.zeta*dg4

    
    pool = mp.Pool(npool)

    pktable[:, 0] = kv[:]        
    pktable[:, 1] = integrate(func = za, pool = pool)
    if order == 1:
        pktable[:, 2], pktable[:, 3] = np.zeros_like(kv), np.zeros_like(kv)
    else:
        pktable[:, 2] = integrate(aloop, taylor = 2, pool = pool)
        pktable[:, 3] = integrate(wloop, taylor = 3, pool = pool)
    pktable[:, 4] = integrate(b1, pool = pool)
    pktable[:, 5] = integrate(b1sq, pool = pool)
    pktable[:, 6] = integrate(b2, pool = pool)
    pktable[:, 7] = integrate(b2sq, pool = pool)
    pktable[:, 8] = integrate(b1b2, pool = pool)
    pktable[:, 9] = integrate(bs2, pool = pool)
    pktable[:,10] = integrate(b1bs2, pool = pool)
    pktable[:,11] = integrate(b2bs2, pool = pool)
    pktable[:,12] = integrate(bs2sq, pool = pool)
    pktable[:,13] = np.zeros_like(kv)
    pktable[:,14] = np.zeros_like(kv)

    pool.close()
    pool.join()


    del kv

    del qv, XYlin, sigma, yq, renorm, jn

    del Xlin, Ylin, Xloop, Yloop, xi1loop, xi3loop, x10, y10, x20, y20

    del u10, u30, u11, u20, v10, v12, corr, chi, zeta

    return pktable



###################################################################################
#Functions from table in Appendix B
def za( k, l):
    return np.ones_like(qv)

def aloop( k, l):
    return  -(Xloop + Yloop - 2*l*Yloop/Ylin/k**2.)  

def wloop( k, l):
    return  bool(l)*(6/5.*xi1loop + 6/5.*xi3loop  -
                    6*(l-1)*xi3loop/(k**2 *Ylin)) *1/(yq *k)

def b1( k, l):
    return  -k**2 *((x10*1.) + (y10*1.))  + 2*l* (y10*1.)/Ylin -2*qv* (u10+u30) * bool(l)/Ylin
    #Ad-hoc factor of 2 to match ZV files
    #return  -k**2 *((x10*2.) + (y10*2)) + 2*l* (y10*2)/Ylin -2*qv* bool(l)*(u10+u30)/Ylin

def b1sq(  k, l):
    return  corr - k**2 *u10**2 + 2*l*u10**2/Ylin -qv* bool(l)*u11/Ylin

def b2(  k, l):
    return   -k**2 *u10**2 + 2*l*u10**2/Ylin -qv* bool(l)*u20/Ylin

def b2sq(  k, l):
    return   corr**2 /2.

def b1b2(  k, l):
    return   -2 *bool(l)*qv*u10*corr/Ylin

def bs2(  k, l):
    return  - k**2 *(x20 + y20) + 2*l*y20/Ylin -2*qv* bool(l)*v10/Ylin

def b1bs2(  k, l):
    return   -1*qv* bool(l)*v12/Ylin

def b2bs2( k, l):
    return   chi

def bs2sq(  k, l):
    return  zeta


#Function to calculate growth factor

def Dgrow(z, M, L = None, H0 = 100):
    """return D(a)/D(1.)"""
    a = 1/(1+z)
    if L is None:
        L = 1-M
    logamin = -20.
    Np = 1000
    logx = numpy.linspace(logamin, 0, Np)
    x = numpy.exp(logx)

    Ea = lambda a: (a ** -1.5 * (M + (1 - M - L) * a + L * a ** 3) ** 0.5)

    def kernel(loga):
        a = numpy.exp(loga)
        return (a * Ea(a = a)) ** -3 * a # da = a * d loga                                                                                            

    y = Ea(a = x) * numpy.array([ romberg(kernel, logx.min(), loga, vec_func=True) for loga in logx])
    
    Da = interpolate(x, y)
    return Da(a)/Da(1.)



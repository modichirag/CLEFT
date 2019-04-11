import numpy
import kernelspool as ksp
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function. 
#Divide the funciton by this to get the correct form 

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import inspect
from statsmodels.nonparametric.smoothers_lowess import lowess

class Qfunc:

    def __init__(self, k, p, Qfile = None, Rfile = None, npool = 4, extrapker = True, saveqfile = None):
        
        self.kp = k
        self.p = p
        self.ipk = interpolate(k, p)
        self.ilpk = self.loginterp(k, p)
        self.renorm = numpy.sqrt(numpy.pi/2.)
        self.tpi2 = 2*numpy.pi**2.
        self.kint = numpy.logspace(-5, 6, 1e4)
        self.npool = npool
        self.extrapker = extrapker #If the kernels should be extrapolated. True recommended

        if Qfile is None:
            self.kq, self.Q1, self.Q2, self.Q3, self.Q5, self.Q8, self.Qs2 = self.calc_Q()
        else:
            self.kq, self.Q1, self.Q2, self.Q3, self.Q5, self.Q8, self.Qs2  = numpy.loadtxt(Qfile, unpack=True)
        self.ilQ1 = self.loginterp(self.kq, self.Q1, rp = -5)
        self.ilQ2 = self.loginterp(self.kq, self.Q2, rp = -5)
        self.ilQ3 = self.loginterp(self.kq, self.Q3, rp = -5)
        self.ilQ5 = self.loginterp(self.kq, self.Q5, rp = -5)
        self.ilQ8 = self.loginterp(self.kq, self.Q8, rp = -5)
        self.ilQs2 = self.loginterp(self.kq, self.Qs2, rp = -5)

        if Rfile is None:
            self.kr, self.R1, self.R2 = self.calc_R()
        else:
            self.kr, self.R1, self.R2  = numpy.loadtxt(Rfile, unpack=True)
        self.ilR1 = self.loginterp(self.kr, self.R1)
        self.ilR2 = self.loginterp(self.kr, self.R2)

        if saveqfile is not None:
            self.save_qfunc(saveqfile)
            


    def calc_Q(self):
        print('Evaluating Q integrals. Recommend saving them')
        k = numpy.logspace(-4, 4, 2e3)
        #k = numpy.logspace(-4, 2, 5e2)
        ns = [1, 2, 3, 5, 8, -1]
        Qv = ksp.Q(kv = k, pk = self.ilpk, npool = self.npool, ns = ns)
        return Qv[0], Qv[1], Qv[2], Qv[3], Qv[4], Qv[5], Qv[6]

    def calc_R(self):
        print('Evaluating R integrals. Recommend saving them')
        k = numpy.logspace(-4, 4, 2e3)
        #k = numpy.logspace(-4, 2, 5e2)
        ns = [1, 2]
        Rv = ksp.R(kv = k, pk = self.ilpk, npool = self.npool, ns = ns)
        return Rv[0], Rv[1], Rv[2]


    def dosph(self, n, x, f, tilt = 1.5, extrap = True , q1 = 1e-3, q2 = 5e2):
        f = f*self.renorm
        if not self.extrapker:
            return sph(x, nu = n, q = tilt)(f, extrap = extrap)
        else:
            q, p = sph(x, nu = n, q = tilt)(f, extrap = extrap)
            q1, q2 = numpy.where(q>q1)[0][0], numpy.where(q>q2)[0][0] 
            qv = numpy.logspace(-6, 7, 1e4) 
            ip = self.loginterp(q[q1:q2], p[q1:q2])
            return (qv, ip(qv))

    #correlatin function
    def corr(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (1.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 400)

    #Xi integrals from 1506.05264; ZV LEFT paper
    #0 lag
    def xi0lin0(self, kmin = 1e-6, kmax = 1e3):
        val = quad(self.ilpk, kmin, kmax, limit = 200)[0]/self.tpi2
        return val

    def xi0loop0(self, kmin = 1e-6, kmax = 1e3):
        val = self.xi0loop013() + self.xi0loop022()
        return val

    def xi0loop013(self, kmin = 1e-6, kmax = 1e3):
        kint = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), 1e4)
        integrand = 10./21.*self.ilR1(kint)
        linterp = self.loginterp(kint, integrand)
        val = quad(linterp, kmin, kmax, limit = 200)[0]/self.tpi2
        return val

    def xi0loop022(self, kmin = 1e-6, kmax = 1e3):
        kint = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), 1e4)
        integrand = 9./98.*self.ilQ1(kint)
        linterp = self.loginterp(kint, integrand)
        val = quad(linterp, kmin, kmax, limit = 200)[0]/self.tpi2
        return val

 
   #j0
    def xi0lin(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-3, q2 = 2000)
    
    def xi0loop(self, kint = None, tilt = 1.5):
        toret13 = self.xi0loop13()
        toret22 = self.xi0loop22()
        toret = ((toret13[0] + toret22[0])/2., toret13[1] + toret22[1])
        return toret

    def xi0loop13(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = 10./21.*self.ilR1(kint)
        integrand /= (kint**2 *self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-4, q2 =500)

    def xi0loop22(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = 9./98.*self.ilQ1(kint)
        integrand /= (kint**2 *self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-4, q2 =500)

    def xi0eft(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (self.tpi2)        
        return self.dosph(0, kint, integrand, tilt = tilt)
    
    #j2
    def xi2lin(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-1, q2 =2000)

    def xi2loop(self, kint = None, tilt = 1.5):
        toret13 = self.xi2loop13()
        toret22 = self.xi2loop22()
        toret = ((toret13[0] + toret22[0])/2., toret13[1] + toret22[1])
        return toret

    def xi2loop13(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = 10./21.*self.ilR1(kint)
        integrand /= (kint**2 *self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =1000)

    def xi2loop22(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = 9./98.*self.ilQ1(kint)
        integrand /= (kint**2 *self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =1000)

    def xi2eft(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (self.tpi2)        
        return self.dosph(2, kint, integrand, tilt = tilt)
        
    #j1
    def xi1loop(self, kint = None, tilt = 1.5):
        toret13 = self.xi1loop13()
        toret22 = self.xi1loop22()
        toret = ((toret13[0] + toret22[0])/2., toret13[1] + toret22[1])
        return toret

    def xi1loop13(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint        
        integrand = 2.*self.ilR1(kint) - 6*self.ilR2(kint)
        integrand *= (-3./7.)/(kint**3.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =2000)

    def xi1loop22(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint        
        integrand = self.ilQ1(kint) - 3*self.ilQ2(kint)
        integrand *= (-3./7.)/(kint**3.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =2000)

    def xi1eft(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand *= (-3./7.)/(kint *self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)

    #j3
    def xi3loop(self, kint = None, tilt = 1.5):
        #T112
        toret13 = self.xi3loop13()
        toret22 = self.xi3loop22()
        toret = ((toret13[0] + toret22[0])/2., toret13[1] + toret22[1])
        return toret

    def xi3loop13(self, kint = None, tilt = 1.5):
        #T112
        if kint is None:
            kint = self.kint
        integrand = 2.*self.ilR1(kint) + 4*self.ilR2(kint)
        integrand *= (-3./7.)/(kint**3. *self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt, q1 = 1e-2, q2 =2000)

    def xi3loop22(self, kint = None, tilt = 1.5):
        #T112
        if kint is None:
            kint = self.kint
        integrand = self.ilQ1(kint) + 2*self.ilQ2(kint)
        integrand *= (-3./7.)/(kint**3. *self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt, q1 = 1e-2, q2 =2000)

    def xi3eft(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand *= (-3./7.)/(kint *self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt)


    #Integrals from Apprndix B3 of 1209.0780; Carlson CLPT paper
    #V
    def v1(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilR1(kint)
        integrand *= (-3./7.)/(kint**3 *self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)

    def v3(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilQ1(kint)
        integrand *= (-3./7.)/(kint**3 *self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)


    #U
    def u10(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = -1*kint*self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-4, q2 = 500)
        
    
    def u3(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        #U3 in Martin's code
        integrand = (-5./21)*kint*self.ilR1(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)

    def u11(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = (-6./7)* kint* (self.ilR1(kint) + self.ilR2(kint))
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 300)

    def u20(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = (-3./7)* kint* self.ilQ8(kint) 
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)

    #x10&y10 loop
    #These are off by factor of 2 from ZV files so that factor is added in
    def x10(self, kint = None):
        lag = self.x100lag()
        j0 =  self.x10j0(kint = kint)
        j2 = self.x10j2(kint = kint)
        #Addhoc factor of 2 that I seem to be missing
        value = 2.*(lag + j0[1] + j2[1])
        return (j0[0], value)

    def x100lag(self, kmin = 1e-6, kmax = 1e3, kint = None):
        if kint is None:
            kint = self.kint
        integrand = 1/7.*(self.ilR1(kint) - self.ilR2(kint))
        linterp = self.loginterp(kint, integrand)
        val = quad(linterp, kmin, kmax, limit = 200)[0]/self.tpi2
        return val

    def x10j0(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = (-1./14)*( 4*self.ilR2(kint) + 2*self.ilQ5(kint))
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-5, q2 =500)

    def x10j2(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = (-1./14)*(3*self.ilR1(kint) + 4*self.ilR2(kint) + 2*self.ilQ5(kint))
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =1000)
    
    def y10(self, kint = None, tilt = 1.5):    
        #This is simply -3*x10j2?
        if kint is None:
            kint = self.kint
        integrand = (3./14)*(3*self.ilR1(kint) + 4*self.ilR2(kint) + 2*self.ilQ5(kint))
        integrand /= (kint**2.*self.tpi2)
        #Addhoc factor of 2 that I seem to be missing
        integrand *= 2.
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =1000)
                          
    #T
    def t112(self, kint = None, tilt = 1.5):
        #same as xi3loop
        if kint is None:
            kint = self.kint
        qt = self.ilQ1(kint) + 2*self.ilQ2(kint)
        rt = 2.*self.ilR1(kint) + 4*self.ilR2(kint)
        integrand = qt + rt 
        integrand *= (-3./7.)/(kint**3. *self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =1000)

    #SHEAR integrals
    #from 1609.02908; ZV, EC; CLEFT-GSM, Appendix D
    #Jshear
    def jshear(self, kint = None, tilt = 1.5):    
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (kint**1.*self.tpi2)
        j1 = self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =500)
        j3 = self.dosph(3, kint, integrand, tilt = tilt, q1 = 1e-3, q2 =500)
        q = (j1[0] + j3[0])/2.
        js2 = 2*j1[1]/15. - j3[1]/5.
        js3 = -j1[1]/5. - j3[1]/5.
        js4 = j3[1].copy()
        return (q, js2, js3, js4)


    def js2(self, kint = None, tilt = 1.5):    
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (1.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)

    def v10(self, kint = None, tilt = 1.5):    
        if kint is None:
            kint = self.kint
        integrand = (-2/7.)*self.ilQs2(kint)
        integrand /= (kint * self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt, q1 = 1e-5, q2 =600)


    def zeta(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (1.*self.tpi2)
        q,l0 = self.dosph(0, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)
        q,l2 = self.dosph(2, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)
        q,l4 = self.dosph(4, kint, integrand, tilt = tilt, q1 = 1e-5, q2 = 500)
        toret = 8/35.*l4**2 + 8/63.*l2**2 + 4/45.*l0**2
        return (q, 2*toret)
    


    ### Enterpolate functions in log-sapce beyond the limits


    def loginterp(self, x, y, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6):
        '''Extrapolate function by evaluating a log-index of left & right side
        '''
        if yint is None:
            yint = interpolate(x, y, k = 5)
        if side == "both":
            side = "lr"
            l =lp
            r =rp
        lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
        rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
        if lneff < 0:
            print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
            print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
        if rneff > 0:
            print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
            print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

##        while (lneff <= 0) or (lneff > 1):
##            lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
##            l +=1
##            if niter > 100: continue
##        print('Left slope = %0.3f at point '%lneff, l)
##        niter = 0
##        while (rneff < -3) or (rneff > -2):
##            rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
##            r -= 1
##            if niter > 100: continue
##
        xl = numpy.logspace(-12, numpy.log10(x[l]), 10**5.)
        xr = numpy.logspace(numpy.log10(x[r]), 12., 10**5.)
        yl = y[l]*(xl/x[l])**lneff
        yr = y[r]*(xr/x[r])**rneff

        xint = x[l+1:r].copy()
        yint = y[l+1:r].copy()
        if side.find("l") > -1:
            xint = numpy.concatenate((xl, xint))
            yint = numpy.concatenate((yl, yint))
        if side.find("r") > -1:
            xint = numpy.concatenate((xint, xr))
            yint = numpy.concatenate((yint, yr))
        yint2 = interpolate(xint, yint, k = 5)

        return yint2


    
    def save_qfunc(self, fname):

        #Linear
        xi0lin0 = self.xi0lin0()
        qv, xi0lin = self.xi0lin() #qv determined here
        xi2lin = self.xi2lin()[1]

        #Loop
        xi0loop013 = self.xi0loop013()
        xi0loop022 = self.xi0loop022()
        xi0loop13 = self.xi0loop13()[1] 
        xi2loop13 = self.xi2loop13()[1]
        xi0loop22 = self.xi0loop22()[1] 
        xi2loop22 = self.xi2loop22()[1]

        
        #Loop2
        xi1loop = self.xi1loop(tilt = 0.5)[1]
        xi3loop = self.xi3loop(tilt = 0.5)[1]

        #
        js = self.jshear()
        js2 = self.js2()

        #
        corr = self.corr()[1]
        zeta = self.zeta()[1]
        chi = 4/3.*js2[1]**2
        
        #
        u10 = self.u10()[1]
        u3 = self.u3()[1]
        u11 = self.u11()[1]
        u20 = self.u20()[1]

        v10 = self.v10()[1]
        v12 = 4*js[1]*js2[1]

        #
        x10 = self.x10()[1]
        y10 = self.y10()[1]
        x20 = 4*js[2]**2
        y20 = 2*(3*js[1]**2 + 4*js[1]*js[2] + 2*js[1]*js[3] + 2*js[2]**2 + 4*js[2]*js[3] + js[3]**2)

        
        header = '{0: ^13}'.format('q') +  '{0: ^13}'.format('E0lin') + '{0: ^13}'.format('Elin') + '{0: ^13}'.format('E2lin') + \
                 '{0: ^13}'.format('E0loop220') +  '{0: ^13}'.format('E0loop130') + '{0: ^13}'.format('E0loop22') + '{0: ^13}'.format('E0loop13') + \
                 '{0: ^13}'.format('E2loop22') + '{0: ^13}'.format('E2loop13') + '{0: ^13}'.format('E1loop') + '{0: ^13}'.format('E3loop') + \
                 '{0: ^13}'.format('Xi') + '{0: ^13}'.format('Zeta') + '{0: ^13}'.format('Chi') + '{0: ^13}'.format('U10lin') + '{0: ^13}'.format('U3') + \
                 '{0: ^13}'.format('U11') + '{0: ^13}'.format('U20') + '{0: ^13}'.format('V10') + '{0: ^13}'.format('V12') + \
                 '{0: ^13}'.format('X10') + '{0: ^13}'.format('Y10') + '{0: ^13}'.format('X20') + '{0: ^13}'.format('Y20') 

        tosave = numpy.array([qv, xi0lin0 + qv*0, xi0lin, xi2lin, xi0loop022 + qv*0, xi0loop013 + qv*0, xi0loop22, xi0loop13, 
                              xi2loop22, xi2loop13, xi1loop, xi3loop, corr, zeta, chi, u10, u3, u11, u20, v10, v12, x10, y10, x20, y20]).T

        numpy.savetxt(fname, tosave, header = header, fmt = '%0.6e')




###
###    def loginterp2(self, xx, yy, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -1, \
###                  ldx = 1e-6, rdx = 1e-6):
###        '''Extrapolate function by evaluating a log-index of left & right side
###        '''
###        x, y = xx.copy(), yy.copy()
###        lp, rp = 10, 10
###        #x[-1*rp:] = (x[-1*rp-1:-1] + x[-1*rp:])*0.5
###        #y[-1*rp:] = (y[-1*rp-1:-1] + y[-1*rp:])*0.5
###        #x[:lp] = (x[1:lp+1] + x[:lp])*0.5
###        #y[:lp] = (y[1:lp+1] + y[:lp])*0.5
###        xl = numpy.log10(x)
###        yl = numpy.log10(abs(y))
###        sgn = numpy.sign(y)
###        #func = interp1d(xl, yl, fill_value='extrapolate')
###        func = interpolate(xl, yl)
###        sgnf = interp1d(xl, sgn, fill_value='extrapolate')
###        toret = lambda xx : sgnf(numpy.log10(xx))*10**(func(numpy.log10(xx))) 
###        if  toret(x[lp]) - toret(x[lp+1]) > 0:
###            deriv = (toret(x[lp+1]) - toret(x[lp]))/(x[lp+1]- x[lp])
###            if deriv < -1e-5:
###                print('ERROR: Runaway index on left side, bad interpolation',deriv)
###        if  toret(x[-1*rp]) - toret(x[-1*rp+1]) < 0:
###            deriv = (toret(x[-1*rp+1]) - toret(x[-1*rp]))/( x[-1*rp+1]- x[-1*rp])
###            if deriv > 1e-5:
###                print('ERROR: Runaway index on right side, bad interpolation',  toret(x[-1*rp]) , toret(x[-1*rp+1]))
###        
###        return toret

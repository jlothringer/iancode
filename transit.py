"""
-------------------------------------------------------
The Mandel & Agol (2002) transit light curve equations.
-------------------------------------------------------

:FUNCTIONS:
   :func:`occultuniform` -- uniform-disk transit light curve

   :func:`occultquad` -- quadratic limb-darkening

   :func:`occultnonlin` -- full (4-parameter) nonlinear limb-darkening

   :func:`occultnonlin_small` -- small-planet approximation with full
                                 nonlinear limb-darkening.

   :func:`t2z` -- convert dates to transiting z-parameter for circular
                  orbits.


:REQUIREMENTS:
   `numpy <http://www.numpy.org/>`_

   `scipy.special <http://www.scipy.org/>`_


:NOTES:
    Certain values of p (<0.09, >0.5) cause some routines to hang;
    your mileage may vary.  If you find out why, please let me know!

    Cursory testing suggests that the Python routines contained within
     are slower than the corresponding IDL code by a factor of 5-10.

    For :func:`occultquad` I relied heavily on the IDL code of E. Agol
    and J. Eastman.  

    Function :func:`appellf1` comes from the mpmath compilation, and
    is adopted (with modification) for use herein in compliance with
    its BSD license (see function documentation for more details).

:REFERENCE:
    The main reference is that seminal work by `Mandel and Agol (2002)
    <http://adsabs.harvard.edu/abs/2002ApJ...580L.171M>`_.

:LICENSE:
    Created by `Ian Crossfield <http://www.astro.ucla.edu/~ianc/>`_ at
    UCLA.  The code contained herein may be reused, adapted, or
    modified so long as proper attribution is made to the original
    authors.

:REVISIONS:
   2011-04-22 11:08 IJMC: Finished, renamed occultation functions.
                          Cleaned up documentation. Published to
                          website.
                          
   2011-04-25 17:32 IJMC: Fixed bug in :func:`ellpic_bulirsch`.

   2012-03-09 08:38 IJMC: Several major bugs fixed, courtesy of
                          S. Aigrain at Oxford U.

   2012-03-20 14:12 IJMC: Fixed modeleclipse_simple based on new
                          format of :func:`occultuniform.  `

"""

import numpy as np
from scipy import special, misc
import pdb
import os

eps = np.finfo(float).eps
zeroval = eps*1e6

try:
    import _integral_smallplanet_nonlinear
    c_integral_smallplanet_nonlinear = True
except:
    c_integral_smallplanet_nonlinear = False




def appellf1(a,b1,b2,c,z1,z2,**kwargs):
    """Give the Appell hypergeometric function of two variables.

    :INPUTS:
       six parameters, all scalars.

    :OPTIONS:
       eps -- scalar, machine tolerance precision.  Defaults to 1e-10.

    :NOTES:
       Adapted from the `mpmath <http://code.google.com/p/mpmath/>`_
       module, but using the scipy (instead of mpmath) Gauss
       hypergeometric function speeds things up.
       
    :LICENSE:
       MPMATH Copyright (c) 2005-2010 Fredrik Johansson and mpmath
       contributors.  All rights reserved.

       Redistribution and use in source and binary forms, with or
       without modification, are permitted provided that the following
       conditions are met:

       a. Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.

       b. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.  
     
       c. Neither the name of mpmath nor the names of its contributors
          may be used to endorse or promote products derived from this
          software without specific prior written permission.


       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
       CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
       INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
       MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
       DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
       LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
       EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
       TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
       ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
       IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
       THE POSSIBILITY OF SUCH DAMAGE.
    """
    #2011-04-22 10:15 IJMC: Adapted from mpmath, but using scipy Gauss
    #   hypergeo. function
    # 2013-03-11 13:34 IJMC: Added a small error-trap for 'nan' hypgf values

    if kwargs.has_key('eps'):
        eps = kwargs['eps']
    else:
        eps = 1e-9

    # Assume z1 smaller
    # We will use z1 for the outer loop
    if abs(z1) > abs(z2):
        z1, z2 = z2, z1
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # IJMC: Ignore the finite cases for now....
    ## Finite cases
    #if ctx.isnpint(a):
    #    pass
    #elif ctx.isnpint(b1):
    #    pass
    #elif ctx.isnpint(b2):
    #    z1, z2, b1, b2 = z2, z1, b2, b1
    #else:
    #    #print z1, z2
    #    # Note: ok if |z2| > 1, because
    #    # 2F1 implements analytic continuation
    if not ok(z1):
        u1 = (z1-z2)/(z1-1)
        if not ok(u1):
            raise ValueError("Analytic continuation not implemented")
        #print "Using analytic continuation"
        return (1-z1)**(-b1)*(1-z2)**(c-a-b2)*\
            appellf1(c-a,b1,c-b1-b2,c,u1,z2,**kwargs)

    #print "inner is", a, b2, c
    ##one = ctx.one
    s = 0
    t = 1
    k = 0
    
    while 1:
        #h = ctx.hyp2f1(a,b2,c,z2,zeroprec=ctx.prec,**kwargs)
        #print a.__class__, b2.__class__, c.__class__, z2.__class__
        h = special.hyp2f1(float(a), float(b2), float(c), float(z2))
        if not np.isfinite(h):
            break
        term = t * h 
        if abs(term) < eps and abs(h) > 10*eps:
            break
        s += term
        k += 1
        t = (t*a*b1*z1) / (c*k)
        c += 1 # one
        a += 1 # one 
        b1 += 1 # one
        #print k, h, term, s


    return s

def ellke2(k, tol=100*eps, maxiter=100):
    """Compute complete elliptic integrals of the first kind (K) and
    second kind (E) using the series expansions."""
    # 2011-04-24 21:14 IJMC: Created

    k = np.array(k, copy=False)
    ksum = 0*k
    kprevsum = ksum.copy()
    kresidual = ksum + 1
    #esum = 0*k
    #eprevsum = esum.copy()
    #eresidual = esum + 1
    n = 0
    sqrtpi = np.sqrt(np.pi)
    #kpow = k**0
    #ksq = k*k

    while (np.abs(kresidual) > tol).any() and n <= maxiter:
        #kpow *= (ksq)
        #print kpow==(k**(2*n))
        ksum += ((misc.factorial2(2*n - 1)/misc.factorial2(2*n))**2) * k**(2*n)
        #ksum += (special.gamma(n + 0.5)/special.gamma(n + 1) / sqrtpi) * k**(2*n)
        kresidual = ksum - kprevsum
        kprevsum = ksum.copy()
        n += 1
        #print n, kresidual

    return ksum * (np.pi/2.)




def ellke(k):
    """Compute Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) kind.

    :INPUTS:
       k -- scalar or Numpy array
      
    :OUTPUTS:
       ek, kk

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJC: Adapted from J. Eastman's IDL code.
    
    m1 = 1. - k**2
    logm1 = np.log(m1)

    # First kind:
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1. + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1 * (b1 + m1*(b2 + m1*(b3 + m1*b4))) * (-logm1)
    
    # Second kind:     
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))) * logm1

    return ee1 + ee2, ek1 - ek2
         

def ellpic_bulirsch(n, k, tol=1000*eps, maxiter=1e4):
    """Compute the complete elliptical integral of the third kind
    using the algorithm of Bulirsch (1965).

    :INPUTS:
       n -- scalar or Numpy array

       k-- scalar or Numpy array

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJMC: Adapted from J. Eastman's IDL code.
    # 2011-04-25 11:40 IJMC: Set a more stringent tolerance (from 1e-8
    #                  to 1e-14), and fixed tolerance flag to the
    #                  maximum of all residuals.
    # 2013-04-13 21:31 IJMC: Changed 'max' call to 'any'; minor speed boost.
    
    # Make p, k into vectors:
    #if not hasattr(n, '__iter__'):
    #    n = array([n])
    #if not hasattr(k, '__iter__'):
    #    k = array([k])

    if not hasattr(n,'__iter__'):
        n = np.array([n])
    if not hasattr(k,'__iter__'):
        k = np.array([k])

    if len(n)==0 or len(k)==0:
        return np.array([])

    kc = np.sqrt(1. - k**2)
    p = n + 1.
    
    if min(p) < 0:
        print "Negative p"
        
    # Initialize:
    m0 = np.array(1.)
    c = np.array(1.)
    p = np.sqrt(p)
    d = 1./p
    e = kc.copy()

    outsideTolerance = True
    iter = 0
    while outsideTolerance and iter<maxiter:
        f = c.copy()
        c = d/p + c
        g = e/p
        d = 2. * (f*g + d)
        p = g + p; 
        g = m0.copy()
        m0 = kc + m0
        if ((np.abs(1. - kc/g)) > tol).any():
            kc = 2. * np.sqrt(e)
            e = kc * m0
            iter += 1
        else:
            outsideTolerance = False
        #if (iter/10.) == (iter/10):
        #    print iter, (np.abs(1. - kc/g))
        ## For debugging:
        #print min(np.abs(1. - kc/g)) > tol
        #print 'tolerance>>', tol
        #print 'minimum>>  ', min(np.abs(1. - kc/g))  
        #print 'maximum>>  ', max(np.abs(1. - kc/g)) #, (np.abs(1. - kc/g))

    return .5 * np.pi * (c*m0 + d) / (m0 * (m0 + p))

def z2dt_circular(per, inc, ars, z):
    """ Convert transit crossing parameter z to a time offset for circular orbits.

    :INPUTS:
        per --  scalar. planetary orbital period

        inc -- scalar. orbital inclination (in degrees)

        ars -- scalar.  ratio a/Rs,  orbital semimajor axis over stellar radius

        z -- scalar or array; transit crossing parameter z.

    :RETURNS:
        |dt| -- magnitude of the time offset from transit center at
                which specified z occurs.
        """
    # 2011-06-14 11:26 IJMC: Created.

    numer = (z / ars)**2 - 1.
    denom = np.cos(inc*np.pi/180.)**2 - 1.
    dt = (per / (2*np.pi)) * np.arccos(np.sqrt(numer / denom))

    return dt


def t2z(tt, per, inc, hjd, ars, ecc=0, longperi=0, transitonly=False, occultationonly=False):
    """Convert HJD (time) to transit crossing parameter z.
    
    :INPUTS:
        tt --  scalar. transit ephemeris

        per --  scalar. planetary orbital period (in days)

        inc -- scalar. orbital inclination (in degrees)

        hjd -- scalar or array of times, typically heliocentric or
               barycentric julian date.

        ars -- scalar.  ratio a/Rs,  orbital semimajor axis over stellar radius

        ecc -- scalar.  orbital eccentricity.

        longperi=0 scalar.  longitude of periapse (in radians)

        transitonly : bool
          If False, both transits and occultations have z=0.  But this
          means that other routines (e.g., :func:`occultuniform`)
          model two eclipse events per orbit. As a kludge, set
          transitonly=True -- it sets all values of 'z' near
          occultations to be very large, so that they are not modeled
          as eclipses.

        occultationonly : bool
          Same as transitonly, but for occultations (secondary eclipses)

    :ALGORITHM:
       At zero eccentricity, z relates to physical quantities by:

       z = (a/Rs) * sqrt(sin[w*(t-t0)]**2+[cos(i)*cos(w*[t-t0])]**2)
       """
    # 2010-01-11 18:18 IJC: Created
    # 2011-04-19 15:20 IJMC: Updated documentation.
    # 2011-04-22 11:27 IJMC: Updated to avoid reliance on planet objects.
    # 2011-05-22 16:51 IJMC: Temporarily removed eccentricity
    #                        dependence... I'll deal with that later.
    # 2013-10-12 22:58 IJMC: Added transitonly, occultationonly options


    #if not p.transit:
    #    print "Must use a transiting exoplanet!"
    #    return False
    from analysis import trueanomaly


    if ecc <> 0:
        ecc = 0
        print "WARNING: setting ecc=0 for now until I get this function working"


    if ecc==0:
        #omega_orb = 2*np.pi/per
        omega_tdiff = (2*np.pi/per) * (hjd - tt)
        cosom = np.cos(omega_tdiff)
        z = ars * np.sqrt(np.sin(omega_tdiff)**2 + \
                              (np.cos(np.deg2rad(inc))*cosom)**2)
        if transitonly:
            z[cosom<0] = 100.
        if occultationonly:
            z[cosom>0] = 100.

    else:
        print "WARNING: this hasn't been tested!!! Get it working someday."
        if longperi is None:
            longperi = 180.
        f = trueanomaly(ecc, (2*np.pi/per) * (hjd - tt))
        z = ars * (1. - ecc**2) * np.sqrt(1. - (np.sin(longperi + f) * np.sin(inc*np.pi/180.))**2) / \
            (1. + ecc * np.cos(f)) 

    return z

def uniform(*arg, **kw):
    """Placeholder for my old code; the new function is called
    :func:`occultuniform`.
    """
    # 2011-04-19 15:06 IJMC: Created
    print "The function 'transit.uniform()' is deprecated."
    print "Please use transit.occultuniform() in the future."
    return occultuniform(*arg, **kw)


def occultuniform(z, p, complement=False, verbose=False):
    """Uniform-disk transit light curve (i.e., no limb darkening).

    :INPUTS:
       z -- scalar or sequence; positional offset values of planet in
            units of the stellar radius.

       p -- scalar;  planet/star radius ratio.

       complement : bool
         If True, return (1 - occultuniform(z, p))

    :SEE ALSO:  :func:`t2z`, :func:`occultquad`, :func:`occultnonlin_small`
    """
    # 2011-04-15 16:56 IJC: Added a tad of documentation
    # 2011-04-19 15:21 IJMC: Cleaned up documentation.
    # 2011-04-25 11:07 IJMC: Can now handle scalar z input.
    # 2011-05-15 10:20 IJMC: Fixed indexing check (size, not len)
    # 2012-03-09 08:30 IJMC: Added "complement" argument for backwards
    #                        compatibility, and fixed arccos error at
    #                        1st/4th contact point (credit to
    #                        S. Aigrain @ Oxford)
    # 2013-04-13 21:28 IJMC: Some code optimization; ~20% gain.

    z = np.abs(np.array(z,copy=True))
    fsecondary = np.zeros(z.shape,float)
    if p < 0:
        pneg = True
        p = np.abs(p)
    else:
        pneg = False

    p2 = p*p

    if len(z.shape)>0: # array entered
        i1 = (1+p)<z
        i2 = (np.abs(1-p) < z) * (z<= (1+p))
        i3 = z<= (1-p)
        i4 = z<=(p-1)

        any2 = i2.any()
        any3 = i3.any()
        any4 = i4.any()
        #print i1.sum(),i2.sum(),i3.sum(),i4.sum()

        if any2:
            zi2 = z[i2]
            zi2sq = zi2*zi2
            arg1 = 1 - p2 + zi2sq
            acosarg1 = (p2+zi2sq-1)/(2.*p*zi2)
            acosarg2 = arg1/(2*zi2)
            acosarg1[acosarg1 > 1] = 1.  # quick fix for numerical precision errors
            acosarg2[acosarg2 > 1] = 1.  # quick fix for numerical precision errors
            k0 = np.arccos(acosarg1)
            k1 = np.arccos(acosarg2)
            k2 = 0.5*np.sqrt(4*zi2sq-arg1*arg1)
            fsecondary[i2] = (1./np.pi)*(p2*k0 + k1 - k2)

        fsecondary[i1] = 0.
        if any3: fsecondary[i3] = p2
        if any4: fsecondary[i4] = 1.

        if verbose:
            if not (i1+i2+i3+i4).all():
                print "warning -- some input values not indexed!"
            if (i1.sum()+i2.sum()+i3.sum()+i4.sum() <> z.size):
                print "warning -- indexing didn't get the right number of values"

        

    else:  # scalar entered
        if (1+p)<=z:
            fsecondary = 0.
        elif (np.abs(1-p) < z) * (z<= (1+p)):
            z2 = z*z
            k0 = np.arccos((p2+z2-1)/(2.*p*z))
            k1 = np.arccos((1-p2+z2)/(2*z))
            k2 = 0.5*np.sqrt(4*z2-(1+z2-p2)**2)
            fsecondary = (1./np.pi)*(p2*k0 + k1 - k2)
        elif z<= (1-p):
            fsecondary = p2
        elif z<=(p-1):
            fsecondary = 1.
        
    if pneg:
        fsecondary *= -1

    if complement:
        return fsecondary
    else:
        return 1. - fsecondary
    

def depthchisq(z, planet, data, ddepth=[-.1,.1], ndepth=20, w=None):
    #z = transit.t2z(planet, planet.i, hjd, 0.211)
    nobs = z.size
    depths = np.linspace(ddepth[0],ddepth[1], ndepth)
    print depths
    chisq = np.zeros(ndepth, float)
    for ii in range(ndepth):
        tr = -(transit.occultuniform(z, np.sqrt(planet.depth))/depths[ii])
        if w is None:
            w = np.ones(nobs,float)/data[tr==0].std()
        print 'w>>',w[0]
        baseline = np.ones(nobs,float) * an.wmean(data[tr==0], w[tr==0])
        print 'b>>',baseline[0]
        print 'd[ii]>>',depths[ii]
        model = baseline + tr*depths[ii]
        plot(model)
        chisq[ii] = (w*(model-data)**2).sum()
    return depths, chisq




def integral_smallplanet_nonlinear(z, p, cn, lower, upper):
    """Return the integral in I*(z) in Eqn. 8 of Mandel & Agol (2002).
    -- Int[I(r) 2r dr]_{z-p}^{1}, where:
    
    :INPUTS:
         z = scalar or array.  Distance between center of star &
             planet, normalized by the stellar radius.

         p = scalar.  Planet/star radius ratio.

         cn = 4-sequence.  Nonlinear limb-darkening coefficients,
              e.g. from Claret 2000.

         lower, upper -- floats. Limits of integration in units of mu

    :RETURNS:
         value of the integral at specified z.

         """
    # 2010-11-06 14:12 IJC: Created
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero

    #import pdb

    #z = np.array(z, copy=True)
    #z[z==0] = zeroval
    #a = (z - p)**2
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    return eval_int_at_limit(upper, cn) - eval_int_at_limit(lower, cn) 

def eval_int_at_limit(limit, cn):
    """Evaluate the integral at a specified limit (upper or lower)"""
    # 2013-04-17 22:27 IJMC: Implemented some speed boosts; added a
    #                        bug; fixed it again.

    # The old way:
    #term1 = cn[0] * (1. - 0.8 * np.sqrt(limit))
    #term2 = cn[1] * (1. - (2./3.) * limit)
    #term3 = cn[2] * (1. - (4./7.) * limit**1.5)
    #term4 = cn[3] * (1. - 0.5 * limit**2)
    #goodret = -(limit**2) * (1. - term1 - term2 - term3 - term4)

    # The new, somewhat faster, way:
    sqrtlimit = np.sqrt(limit)
    sqlimit = limit*limit
    total = 1. - cn[0] * (1. - 0.8 * sqrtlimit)
    total -= cn[1] * (1. - (2./3.) * limit)
    total -= cn[2] * (1. - (4./7.) * limit*sqrtlimit)
    total -= cn[3] * (1. - 0.5 * sqlimit)
    ret = -(sqlimit) * total

    return ret

        

def smallplanet_nonlinear(*arg, **kw):
    """Placeholder for backwards compatibility with my old code.  The
     function is now called :func:`occultnonlin_small`.
    """
    # 2011-04-19 15:10 IJMC: Created

    print "The function 'transit.smallplanet_nonlinear()' is deprecated."
    print "Please use transit.occultnonlin_small() in the future."

    return occultnonlin_small(*arg, **kw)


def occultnonlin_small(z,p, cn):
    """Nonlinear limb-darkening light curve in the small-planet
    approximation (section 5 of Mandel & Agol 2002).

    :INPUTS:
        z -- sequence of positional offset values

        p -- planet/star radius ratio

        cn -- four-sequence nonlinear limb darkening coefficients.  If
              a shorter sequence is entered, the later values will be
              set to zero.

    :NOTE: 
       I had to divide the effect at the near-edge of the light curve
       by pi for consistency; this factor was not in Mandel & Agol, so
       I may have coded something incorrectly (or there was a typo).

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = transit.occultnonlin_small(z, 0.1, coef)
             plot(z, f, '--')

    :SEE ALSO:
       :func:`t2z`
    """
    # 2010-11-06 14:23 IJC: Created
    # 2011-04-19 15:22 IJMC: Updated documentation.  Renamed.
    # 2011-05-24 14:00 IJMC: Now check the size of cn.
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero
    # 2013-04-17 10:51 IJMC: Mild code optimization

    #import pdb

    cn = np.array([cn], copy=False).ravel()
    if cn.size < 4:
        cn = np.concatenate((cn, [0.]*(4-cn.size)))

    z = np.array(z, copy=False)
    F = np.ones(z.shape, float)

    z[z==0] = zeroval # cheat!

    a = (z - p)**2
    b = (z + p)**2
    c0 = 1. - np.sum(cn)
    Omega = 0.25 * c0 + np.sum( cn / np.arange(5., 9.) )

    ind1 = ((1. - p) < z) * ((1. + p) > z)
    ind2 = z <= (1. - p)

    # Need to specify limits of integration in terms of mu (not r)
    aind1 = 1. - a[ind1]
    zind1m1 = z[ind1] - 1.
    if c_integral_smallplanet_nonlinear:
        #print 'do it the C way'
        Istar_edge = _integral_smallplanet_nonlinear.integral_smallplanet_nonlinear(cn, np.sqrt(aind1), np.array([0.])) / aind1
        Istar_inside = _integral_smallplanet_nonlinear.integral_smallplanet_nonlinear(cn, np.sqrt(1. - a[ind2]), np.sqrt(1. - b[ind2])) / z[ind2]
    else:
        Istar_edge = integral_smallplanet_nonlinear(None, p, cn, \
                                                    np.sqrt(aind1), np.array([0.])) / aind1
        
        Istar_inside = integral_smallplanet_nonlinear(None, p, cn, \
                                              np.sqrt(1. - a[ind2]), \
                                              np.sqrt(1. - b[ind2])) / \
                                              (z[ind2])


    term1 = 0.25 * Istar_edge / (np.pi * Omega)
    term2 = p*p * np.arccos((zind1m1) / p)
    term3 = (zind1m1) * np.sqrt(p*p - (zind1m1*zind1m1))

    F[ind1] = 1. - term1 * (term2 - term3)
    F[ind2] = 1. - 0.0625 * p * Istar_inside / Omega

    return F


def occultquad(z,p0, gamma, retall=False, verbose=False):
    """Quadratic limb-darkening light curve; cf. Section 4 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        gamma -- two-sequence.
           quadratic limb darkening coefficients.  (c1=c3=0; c2 =
           gamma[0] + 2*gamma[1], c4 = -gamma[1]).  If only a single
           gamma is used, then you're assuming linear limb-darkening.

    :OPTIONS:
        retall -- bool.  
           If True, in addition to the light curve return the
           uniform-disk light curve, lambda^d, and eta^d parameters.
           Using these quantities allows for quicker model generation
           with new limb-darkening coefficients -- the speed boost is
           roughly a factor of 50.  See the second example below.

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         gammavals = [[0., 0.], [1., 0.], [2., -1.]]
         figure()
         for gammas in gammavals:
             f = transit.occultquad(z, 0.1, gammas)
             plot(z, f)

       ::

         # Calculate the same geometric transit with two different
         #    sets of limb darkening coefficients:
         from pylab import *
         import transit
         p, b = 0.1, 0.5
         x = (arange(300.)/299. - 0.5)*2.
         z = sqrt(x**2 + b**2)
         gammas = [.25, .75]
         F1, Funi, lambdad, etad = transit.occultquad(z, p, gammas, retall=True)

         gammas = [.35, .55]
         F2 = 1. - ((1. - gammas[0] - 2.*gammas[1])*(1. - F1) + 
            (gammas[0] + 2.*gammas[1])*(lambdad + 2./3.*(p > z)) + gammas[1]*etad) / 
            (1. - gammas[0]/3. - gammas[1]/6.)
         figure()
         plot(x, F1, x, F2)
         legend(['F1', 'F2'])
         

    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`

    :NOTES:
       In writing this I relied heavily on the occultquad IDL routine
       by E. Agol and J. Eastman, especially for efficient computation
       of elliptical integrals and for identification of several
       apparent typographic errors in the 2002 paper (see comments in
       the source code).

       From some cursory testing, this routine appears about 9 times
       slower than the IDL version.  The difference drops only
       slightly when using precomputed quantities (i.e., retall=True).
       A large portion of time is taken up in :func:`ellpic_bulirsch`
       and :func:`ellke`, but at least as much is taken up by this
       function itself.  More optimization (or a C wrapper) is desired!
    """
    # 2011-04-15 15:58 IJC: Created; forking from smallplanet_nonlinear
    # 2011-05-14 22:03 IJMC: Now linear-limb-darkening is allowed with
    #                        a single parameter passed in.
    # 2013-04-13 21:06 IJMC: Various code tweaks; speed increased by
    #                        ~20% in some cases.
    #import pdb

    # Initialize:
    z = np.array(z, copy=False)
    lambdad = np.zeros(z.shape, float)
    etad = np.zeros(z.shape, float)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input

    # Define limb-darkening coefficients:
    if len(gamma) < 2 or not hasattr(gamma, '__iter__'):  # Linear limb-darkening
        gamma = np.concatenate([gamma.ravel(), [0.]])
        c2 = gamma[0]
    else:
        c2 = gamma[0] + 2 * gamma[1]

    c4 = -gamma[1]



    # Test the simplest case (a zero-sized planet):
    if p==0:
        if retall:
            ret = np.ones(z.shape, float), np.ones(z.shape, float), \
                  np.zeros(z.shape, float), np.zeros(z.shape, float)
        else:
            ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    fourOmega = 1. - gamma[0]/3. - gamma[1]/6. # Actually 4*Omega
    a = (z - p)*(z - p)
    b = (z + p)*(z + p)
    k = 0.5 * np.sqrt((1. - a) / (z * p))  # 8%
    p2 = p*p
    z2 = z*z
    ninePi = 9*np.pi

    # Define the many necessary indices for the different cases:
    pgt0 = p > 0
    
    i01 = pgt0 * (z >= (1. + p))
    i02 = pgt0 * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = pgt0 * (p < 0.5) * (z > p) * (z < (1. - p))
    i04 = pgt0 * (p < 0.5) * (z == (1. - p))
    i05 = pgt0 * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i09 = pgt0 * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = pgt0 * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))
    #any01 = i01.any()
    #any02 = i02.any()
    #any03 = i03.any()
    any04 = i04.any()
    any05 = i05.any()
    any06 = i06.any()
    any07 = i07.any()
    #any08 = i08.any()
    #any09 = i09.any()
    any10 = i10.any()
    any11 = i11.any()
    #print n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11
    if verbose:
        allind = i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08 + i09 + i10 + i11 
        nused = (i01.sum() + i02.sum() + i03.sum() + i04.sum() + \
                     i05.sum() + i06.sum() + i07.sum() + i08.sum() + \
                     i09.sum() + i10.sum() + i11.sum()) 

        print "%i/%i indices used" % (nused, i01.size)
        if not allind.all():
            print "Some indices not used!"



    # Lambda^e and eta^d are more tricky:
    # Simple cases:
    lambdad[i01] = 0.
    etad[i01] = 0.

    if any06:
        lambdad[i06] = 1./3. - 4./ninePi
        etad[i06] = 0.09375 # = 3./32.

    if any11:
        lambdad[i11] = 1.
        # etad[i11] = 1.  # This is what the paper says
        etad[i11] = 0.5 # Typo in paper (according to J. Eastman)


    # Lambda_1:
    ilam1 = i02 + i08
    q1 = p2 - z2[ilam1]
    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - 1./a[ilam1], k[ilam1])
    # ellipe, ellipk = ellke(k[ilam1])

    # This is what J. Eastman's code has:

    # 2011-04-24 20:32 IJMC: The following codes act funny when
    #                        sqrt((1-a)/(b-a)) approaches unity.
    qq = np.sqrt((1. - a[ilam1]) / (b[ilam1] - a[ilam1]))
    ellippi = ellpic_bulirsch(1./a[ilam1] - 1., qq)
    ellipe, ellipk = ellke(qq)
    lambdad[ilam1] = (1./ (ninePi*np.sqrt(p*z[ilam1]))) * \
        ( ((1. - b[ilam1])*(2*b[ilam1] + a[ilam1] - 3) - \
               3*q1*(b[ilam1] - 2.)) * ellipk + \
              4*p*z[ilam1]*(z2[ilam1] + 7*p2 - 4.) * ellipe - \
              3*(q1/a[ilam1])*ellippi)

    # Lambda_2:
    ilam2 = i03 + i09
    q2 = p2 - z2[ilam2]

    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - b[ilam2]/a[ilam2], 1./k[ilam2])
    # ellipe, ellipk = ellke(1./k[ilam2])

    # This is what J. Eastman's code has:
    ailam2 = a[ilam2] # Pre-cached for speed
    bilam2 = b[ilam2] # Pre-cached for speed
    omailam2 = 1. - ailam2 # Pre-cached for speed
    ellippi = ellpic_bulirsch(bilam2/ailam2 - 1, np.sqrt((bilam2 - ailam2)/(omailam2)))
    ellipe, ellipk = ellke(np.sqrt((bilam2 - ailam2)/(omailam2)))

    lambdad[ilam2] = (2. / (ninePi*np.sqrt(omailam2))) * \
        ((1. - 5*z2[ilam2] + p2 + q2*q2) * ellipk + \
             (omailam2)*(z2[ilam2] + 7*p2 - 4.) * ellipe - \
             3*(q2/ailam2)*ellippi)


    # Lambda_3:
    #ellipe, ellipk = ellke(0.5/ k)  # This is what the paper says
    if any07:
        ellipe, ellipk = ellke(0.5/ p)  # Corrected typo (1/2k -> 1/2p), according to J. Eastman
        lambdad[i07] = 1./3. + (16.*p*(2*p2 - 1.)*ellipe - 
                                (1. - 4*p2)*(3. - 8*p2)*ellipk / p) / ninePi


    # Lambda_4
    #ellipe, ellipk = ellke(2. * k)  # This is what the paper says
    if any05:
        ellipe, ellipk = ellke(2. * p)  # Corrected typo (2k -> 2p), according to J. Eastman
        lambdad[i05] = 1./3. + (2./ninePi) * (4*(2*p2 - 1.)*ellipe + (1. - 4*p2)*ellipk)

    # Lambda_5
    ## The following line is what the 2002 paper says:
    #lambdad[i04] = (2./(3*np.pi)) * (np.arccos(1 - 2*p) - (2./3.) * (3. + 2*p - 8*p2))
    # The following line is what J. Eastman's code says:
    if any04:
        lambdad[i04] = (2./3.) * (np.arccos(1. - 2*p)/np.pi - \
                                      (6./ninePi) * np.sqrt(p * (1.-p)) * \
                                      (3. + 2*p - 8*p2) - \
                                      float(p > 0.5))

    # Lambda_6
    if any10:
        lambdad[i10] = -(2./3.) * (1. - p2)**1.5

    # Eta_1:
    ilam3 = ilam1 + i07 # = i02 + i07 + i08
    z2ilam3  = z2[ilam3]    # pre-cache for better speed
    twoZilam3  = 2*z[ilam3] # pre-cache for better speed
    #kappa0 = np.arccos((p2+z2ilam3-1)/(p*twoZilam3))
    #kappa1 = np.arccos((1-p2+z2ilam3)/(twoZilam3))
    #etad[ilam3] = \
    #    (0.5/np.pi) * (kappa1 + kappa0*p2*(p2 + 2*z2ilam3) - \
    #                    0.25*(1. + 5*p2 + z2ilam3) * \
    #                    np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.))) 
    etad[ilam3] = \
        (0.5/np.pi) * ((np.arccos((1-p2+z2ilam3)/(twoZilam3))) + (np.arccos((p2+z2ilam3-1)/(p*twoZilam3)))*p2*(p2 + 2*z2ilam3) - \
                        0.25*(1. + 5*p2 + z2ilam3) * \
                        np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.))) 


    # Eta_2:
    etad[ilam2 + i04 + i05 + i10] = 0.5 * p2 * (p2 + 2. * z2[ilam2 + i04 + i05 + i10])
    

    # We're done!


    ## The following are handy for debugging:
    #term1 = (1. - c2) * lambdae
    #term2 = c2*lambdad
    #term3 = c2*(2./3.) * (p>z).astype(float)
    #term4 = c4 * etad
    # Lambda^e is easy:
    lambdae = 1. - occultuniform(z, p)  # 14%
    F = 1. - ((1. - c2) * lambdae + \
                  c2 * (lambdad + (2./3.) * (p > z)) - \
                  c4 * etad) / fourOmega  # 13%

    if retall:
        ret = F, lambdae, lambdad, etad
    else:
        ret = F

    return ret

def occultnonlin(z,p0, cn):
    """Nonlinear limb-darkening light curve; cf. Section 3 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        cn -- four-sequence. nonlinear limb darkening coefficients

    :EXAMPLE:
        ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 50)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = transit.occultnonlin(z, 0.1, coef)
             plot(z, f)

    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`, :func:`occultquad`

    :NOTES: 
        Scipy is much faster than mpmath for computing the Beta and
        Gauss hypergeometric functions.  However, Scipy does not have
        the Appell hypergeometric function -- the current version is
        not vectorized.
    """
    # 2011-04-15 15:58 IJC: Created; forking from occultquad
    #import pdb

    # Initialize:
    cn0 = np.array(cn, copy=True)
    z = np.array(z, copy=True)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input


    # Test the simplest case (a zero-sized planet):
    if p==0:
        ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    c0 = 1. - np.sum(cn0)
    # Row vectors:
    c = np.concatenate(([c0], cn0))
    n = np.arange(5, dtype=float)
    # Column vectors:
    cc = c.reshape(5, 1)
    nn = n.reshape(5,1)  
    np4 = n + 4.
    nd4 = n / 4.
    twoOmega = 0.5*c[0] + 0.4*c[1] + c[2]/3. + 2.*c[3]/7. + 0.25*c[4]

    a = (z - p)**2
    b = (z + p)**2
    am1 = a - 1.
    bma = b - a
    
    k = 0.5 * np.sqrt(-am1 / (z * p))
    p2 = p**2
    z2 = z**2


    # Define the many necessary indices for the different cases:
    i01 = (p > 0) * (z >= (1. + p))
    i02 = (p > 0) * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = (p > 0) * (p < 0.5) * (z > p) * (z <= (1. - p))  # also contains Case 4
    #i04 = (z==(1. - p))
    i05 = (p > 0) * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i08a = (p == 1) * (z == 0)
    i09 = (p > 0) * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = (p > 0) * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))

    iN = i02 + i08
    iM = i03 + i09

    # Compute N and M for the appropriate indices:
    #  (Use the slow, non-vectorized appellf1 function:)
    myappellf1 = np.frompyfunc(appellf1, 6, 1)
    N = np.zeros((5, z.size), float)
    M = np.zeros((3, z.size), float)
    if iN.any():
        termN = myappellf1(0.5, 1., 0.5, 0.25*nn + 2.5, am1[iN]/a[iN], -am1[iN]/bma[iN])
        N[:, iN] = ((-am1[iN])**(0.25*nn + 1.5)) / np.sqrt(bma[iN]) * \
            special.beta(0.25*nn + 2., 0.5) * \
            (((z2[iN] - p2) / a[iN]) * termN - \
                 special.hyp2f1(0.5, 0.5, 0.25*nn + 2.5, -am1[iN]/bma[iN]))

    if iM.any():
        termM = myappellf1(0.5, -0.25*nn[1:4] - 1., 1., 1., -bma[iM]/am1[iM], -bma[iM]/a[iM]) 
        M[:, iM] = ((-am1[iM])**(0.25*nn[1:4] + 1.)) * \
            (((z2[iM] - p2)/a[iM]) * termM - \
                 special.hyp2f1(-0.25*nn[1:4] - 1., 0.5, 1., -bma[iM]/am1[iM]))


    # Begin going through all the cases:

    # Case 1:
    F[i01] = 1.

    # Case 2: (Gauss and Appell hypergeometric functions)
    F[i02] = 1. - (1. / (np.pi*twoOmega)) * \
        (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 3 : (Gauss and Appell hypergeometric functions)
    F[i03] = 1. - (0.5/twoOmega) * \
        (c0*p2 + 2*(M[:, i03] * cc[1:4]/(nn[1:4] + 4.)).sum(0) + \
             c[-1]*p2*(1. - 0.5*p2 - z2[i03]))

    #if i04.any():
    #    F[i04] = occultnonlin_small(z[i04], p, cn)
    #    print "Value found for z = 1-p: using small-planet approximation "
    #    print "where Appell F2 function will not otherwise converge."

    #F[i04] = 0.5 * (occultnonlin(z[i04]+p/2., p, cn) + occultnonlin(z[i04]-p/2., p, cn))

    # Case 5: (Gauss hypergeometric function)
    F[i05] = 0.5 + \
        ((c/np4) * special.hyp2f1(0.5, -nd4 - 1., 1., 4*p2)).sum() / twoOmega

    # Case 6:  Gamma function
    F[i06] = 0.5 + (1./(np.sqrt(np.pi) * twoOmega)) * \
        ((c/np4) * special.gamma(1.5 + nd4) / special.gamma(2. + nd4)).sum()

    # Case 7: Gauss hypergeometric function, beta function
    F[i07] = 0.5 + (0.5/(p * np.pi * twoOmega)) * \
        ((c/np4) * special.beta(0.5, nd4 + 2.) * \
             special.hyp2f1(0.5, 0.5, 2.5 + nd4, 0.25/p2)).sum()

    # Case 8: (Gauss and Appell hypergeometric functions)
    F[i08a] = 0.
    F[i08] =  -(1. / (np.pi*twoOmega)) * (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 9: (Gauss and Appell hypergeometric functions)
    F[i09] = (0.5/twoOmega) * \
        (c0 * (1. - p2) + c[-1] * (0.5 - p2*(1. - 0.5*p2 - z2[i09])) - \
             2*(M[:, i09] * cc[1:4] / (nn[1:4] + 4.)).sum(0))

    # Case 10: 
    F[i10] = (2. / twoOmega) * ((c/np4) * (1. - p2)**(nd4 + 1.)).sum()

    # Case 11:
    F[i11] = 0.


    # We're done!

    return F


def modeltransit(params, func, per, t):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.

    :INPUTS:
      params -- (5+N)-sequence with the following:
        the time of conjunction for each individual transit (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        stellar flux (F0),

        the limb-darkening parameters u1 and u2:
             
          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients

          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients

          OR:
            Nothing at all (i.e., only 5 parameters).

      func -- function to fit to data, e.g. transit.occultquad

      per -- float.  Orbital period, in days.

      t -- numpy array.  Time of observations.
    """
    # 2011-05-22 16:14 IJMC: Created.
    # 2011-05-24 10:52 IJMC: Inserted a check for cos(i) > 1

    ecc = 0.
    nparam = len(params)


    if (params[1] * params[2]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], per, (180./np.pi)*np.arccos(params[1]*params[2]), t, 1./params[2], 0., transitonly=True)

    # Mask out secondary eclipses:
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.

    
    if len(params)>5:
        model = params[4] * func(z, params[3], params[5::])
    try:  # Limb-darkened
        model = params[4] * func(z, params[3], params[5::])
    except:  # Uniform-disk
        model = params[4] * (1. - func(z, params[3]))

    return model

def modeltransit_general(params, t, NL, NP=1, errscale=1, smallplanet=True, svs=None):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity.

    :INPUTS:
      params -- (5 + NP + NL + NS)-sequence with the following:
        Tc, the time of conjunction for each individual transit,

        P, the orbital period (in units of "t")

        i, the orbital inclination (in degrees; 90 is edge-on)

        R*/a, the stellar radius in units of orbital distance,

        Rp/R*, planet-to-star radius ratio, 

        the NP polynomial coefficients to normalize the data.
             
          EITHER:
            F0 -- stellar flux _ONLY_ (set NP=1)

          OR:
            [p_1, p_2, ..., p_(NP)] -- coefficients for polyval, to be
            used as: numpy.polyval([p_1, ...], t)

        the NL limb-darkening parameters (cf. Claret+2011):
             
          EITHER:
            u      -- linear limb-darkening (set NL=1)

          OR:
            a, b   -- quadratic limb-darkening (set NL=2)

          OR:
            c,  d  -- root-square limb-darkening (set NL= -2)
              where
            I(mu) = 1 - c * (1 - mu) - d * (1 - mu^0.5)

          OR:
            a1, a2, a3, a4 -- nonlinear limb-darkening  (set NL=4)
              where
            I(mu) = 1 - a1 * (1 - mu^0.5) - a2 * (1 - mu) - \
                        a3 * (1 - mu^1.5) - a4 * (1 - mu^2)

          OR:
            Nothing at all -- uniform limb-darkening (set NL=0)

        multiplicative factors for the NS state vectors (passed in as 'svs')

      t -- numpy array.  Time of observations.  

      smallplanet : bool
        This only matters for root-square and four-parameter nonlinear
        limb-darkening.  If "smallplanet" is True, use
        :func:`occultnonlin_small`.  Otherwise, use
        :func:`occultnonlin`

      errscale : int
        If certain conditions (see below) are met, the resulting
        transit light curve is scaled by this factor.  When fitting,
        set errscale to a very large value (e.g., 1e6) to use as an
        extremely crude hard-edged filter.

        A better way would be to incorporate constrained fitting...

      svs : None or list of 1D NumPy Arrays
        State vectors, applied with coefficients as defined above. To
        avoid degeneracies with the NP polynomial terms (especially
        the constant offset term), it is preferable that the state
        vectors are all mean-subtracted.

    :NOTES:      

      If quadratic or linear limb-darkening (L.D.) is used, the sum of
      the L.D. coefficients cannot exceed 1.  If they do, this routine
      normalizes the coefficients [g1,g2] to:  g_i = g_i / (g1 + g2).

      If "Rp/R*", or "R*/a" are < 0, they will be set to zero.

      If "P" < 0.01, it will be set to 0.01.

      If "inc" > 90, it will be set to 90.
    """
    # 2012-04-30 03:41 IJMC: Created
    # 2012-05-08 16:59 IJMC: NL can be negative (for root-square profiles)
    # 2013-01-31 18:21 IJMC: Added 'errscale' option
    # 2013-04-02 09:07 IJMC: Added 'svs' option
    # 2013-04-18 10:22 IJMC: Apply penalty scaling in 'sqrt' case if
    #                        coefficients give nonphysical intensity values.
    # 2013-04-22 17:43 IJMC: Fixed a few errors in the documentation.


    ecc = 0.
    longperi = 0.

    if svs is None:
        nsvs = 0
    else:
        if isinstance(svs, np.ndarray) and svs.ndim==1:
            svs = svs.reshape(1, svs.size)
        nsvs = len(svs)
    nparam = len(params) - nsvs

    verbose = False

    tc, per, inc, ra, k = params[0:5]
    if NP>0:
        poly_params = params[5:5+NP]
    else:
        poly_params = [1]

    pNL = np.abs(NL)
    if pNL>0:
        ld_params = params[5+NP:5+NP+pNL]

    penalty_factor = 1.
    # Enforce various normalization constraints:
    if inc > 90:
        inc = 90.
        penalty_factor *= errscale

    if per < 0.01:
        per = 0.01
        penalty_factor *= errscale

    if k < 0:
        k = 0.
        penalty_factor *= errscale

    if ra <= 0:
        ra = 1e-6
        penalty_factor *= errscale

    # Enforce the constraint that cos(i) <= 1.
    #print ("%1.5f "*4) % (tc, per, inc, ra)
    z = t2z(tc, per, inc, t, 1./ra, ecc=ecc, longperi=longperi, transitonly=True)


    if NL<>0 and (sum(ld_params)>1 or sum(ld_params)<0):
        penalty_factor *= errscale

    if NL==0:  # Uniform 
        model = occultuniform(z, k)

    elif NL==2 or NL==1:  # Quadratic or Linear
        model = occultquad(z, k, ld_params, verbose=verbose)

    elif NL==-2: # Root-square (i.e., nonlinear)
        new_ld_params = [ld_params[1], ld_params[0], 0., 0.]
        if smallplanet:
            model = occultnonlin_small(z, k, new_ld_params)
        else:
            model = occultnonlin(z, k, new_ld_params)

    elif NL==4 or NL==3: # Nonlinear
        if smallplanet:
            model = occultnonlin_small(z, k, ld_params)
        else:
            model = occultnonlin(z, k, ld_params)

    model *= np.polyval(poly_params, t)
    for ii in xrange(nsvs): 
        model += params[-ii-1] * svs[-ii-1]

    model *= penalty_factor
    return model


def modeleclipse(params, func, per, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.

    :INPUTS:
      params -- (6-or-7)-sequence with the following:
        the time of conjunction for each individual eclipse (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        eclipse depth (dimensionless),

        stellar flux (F0),

        orbital period (OPTIONAL!)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      per -- float.  
        Orbital period,  OR

        None, if period is included in params

      t -- numpy array.  
         Time of observations (same units as Tc and per)
    """
    # 2011-05-30 16:56 IJMC: Created from modeltransit()
    # 2012-01-31 22:14 IJMC: Period can be included in parameters, for
    #                        fitting purposes.

    ecc = 0.
    nparam = len(params)

    if per is None:
        per = params[6]

    if (params[1] * params[2]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], per, (180./np.pi)*np.arccos(params[1]*params[2]), t, 1./params[2], 0.)

#    if len(params)>6:
#        model = params[4] * func(z, params[3], params[6::])
    try:  # Limb-darkened
        TLC = func(z, params[3], params[6::])
    except:  # Uniform-disk
        TLC =  (func(z, params[3]))

    # Appropriately scale eclipse depth:
    model = params[5] * (1. - params[4] * (TLC ) / params[3]**2)

    return model


def modellightcurve(params, t, tfunc=occultuniform, nlimb=0, nchan=0):
    """Model a full planetary light curve: transit, eclipse, and
    (sinusoidal) phase variation. Accept independent eclipse and
    transit times-of-center, but otherwise assume a circular orbit
    (and thus symmetric transits and eclipses).

    :INPUTS:
      params -- (M+10+N)-sequence with the following:

        OPTIONALLY:

          sensitivity variations for each of M channels (e.g.,
          SST/MIPS).  This assumes the times in 't' are in the order
          (T_{1,0}, ... T_{1,M-1}, ... T_{2,0}, ...).  The parameters
          affect the data multiplicatively as (1 + c_i), with the
          constraint that Prod_i(1+c_i) = 1.
             
        the time of conjunction for each individual transit (T_t),

        the time of conjunction for each individual eclipse (T_e),

        the orbital period (P),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        stellar flux (F0),

        maximum (hot-side) planet flux (Fbright),

        minimum (cold-side) planet flux (Fdark),

        phase curve offset (phi_0; 0 implies maximum flux near eclipse) 

        OPTIONALLY:
          limb-darkening parameters (depending on tfunc):
             
          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients

          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients

      t -- numpy array.  Time of observations; same units as orbital
                         period and ephemerides.  If nchan>0, t should
                         be of shape (nchan, L), or a .ravel()ed
                         version of that.

    :OPTIONS:
      tfunc : model transit function
          One of :func:`occultuniform`, :func:`occultnonlin_small`,
          :func:`occultquad`, or :func:`occultnonlin`

      nlimb : int
         number of limb-darkening parameters; these should be the last
         values of params.

      nchan : int
         number of photometric channel sensitivity perturbations;
         these should be the first 'nchan' values of params.

    :EXAMPLE:
       TBW

    :NOTES:
      This should be updated to use the new 'transitonly' options in :func:`t2z`

    """
    # 2011-06-10 11:10 IJMC: Created.
    # 2011-06-14 13:18 IJMC: Sped up with creation of z2dt()
    # 2011-06-30 21:00 IJMC: Fixed functional form of phase curve.
    from scipy import optimize
    import pdb

    ecc = 0.
    if nchan>0:
        cparams = params[0:nchan].copy()
        params = params[nchan::].copy()
        cparams[0] = 1./(1. + cparams[1::]).prod() - 1.

    nparam = len(params)
    tt, te, per, b, ra, k, fstar, fbright, fdark, phi = params[0:10]
    if nparam > 10:
        limbdarkening = params[10::]
    else:
        limbdarkening = None

    cosi = b * ra
    if (cosi) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        zt = t2z(tt, per, (180./np.pi)*np.arccos(b * ra), t, 1./ra, ecc)
        ze = t2z(te, per, (180./np.pi)*np.arccos(b * ra), t, 1./ra, ecc)
        inc = np.arccos(cosi)
        

    # Solve for t given z0, such that  z0 - t2z(t) = 0.
    def residualz(tguess, z0):
        return z0 - t2z(te, per, (180./np.pi)*np.arccos(b * ra), tguess, 1./ra, ecc)

    # Mask out secondary eclipses:
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.

    sep = (tt - te) % per
    transit_times = (((t - tt) % per) < sep/2.) + (((t - tt) % per) > (per - sep/2.))
    eclipse_times = (((t - te) % per) < sep/2.) + (((t - te) % per) > (per - sep/2.))

    
    # Model phase curve flux
    def phaseflux(time):
        return 0.5*(fbright + fdark) + \
            0.5*(fbright - fdark) * np.cos((2*np.pi*(time - tt))/per + phi)

    phas = phaseflux(t)

    # Model transit:
    trans = np.ones(zt.shape, dtype=float)
    if limbdarkening is None:
        trans[transit_times] = (1. - tfunc(zt[transit_times], k))
    else:
        trans[transit_times] = tfunc(zt[transit_times], k, limbdarkening)

    transit_curve = trans*fstar + phas

    # Model eclipse:
    feclip = phaseflux(te) / (fstar + phaseflux(te))
    eclip = np.ones(ze.shape, dtype=float)
    eclip[eclipse_times] =  (1. - occultuniform(ze[eclipse_times], k))
    eclip = 1. + feclip * (eclip - 1.) / k**2


    # A lot of hokey cheats to keep the eclipse bottom flat, but
    #    ingress and egress continuous:

    ## The following code is deprecated with the creation of z2dt()
    #t14 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. + k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t23 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. - k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t12 = 0.5 * (t14 - t23) 

    #zzz = [t2z(tt, per, (180./np.pi)*np.arccos(b * ra), thist, 1./ra, ecc) for thist in [te-t14, te, te+t14]]
    #aaa,bbb = residualz(te-t14, 1. + k), residualz(te, 1. + k)
    #ccc,ddd = residualz(te-t14, 1. - k), residualz(te, 1. - k)
    #if (aaa >= 0 and bbb >= 0) or (aaa <= 0 and bbb <= 0):
    #    print aaa, bbb
    #    print te, t14, t23, k, ra, b, per
    #if (ccc >= 0 and ddd >= 0) or (ccc <= 0 and ddd <= 0):
    #    print ccc, ddd
    #    print te, t14, t23, k, ra, b, per
    #t5 = optimize.bisect(residualz, te - 2*t14, te + t14, args=(1. + k,))
    ##t5 = optimize.newton(residualz, te - t23 - t12, args=(1. + k,))
    ##t6 = optimize.newton(residualz, te - t23 + t12, args=(1. - k,))
    #t6 = optimize.bisect(residualz, te - 2*t14, te + t14, args=(1. - k,))

    t5 = te - z2dt_circular(per, inc*180./np.pi, 1./ra, 1. + k)
    t6 = te - z2dt_circular(per, inc*180./np.pi, 1./ra, 1. - k)
    t7 = te + (te - t6)
    t8 = te + (te - t5)
    #z58 = [t2z(tt, per, (180./np.pi)*np.arccos(b * ra), thist, 1./ra, ecc) for thist in [t5,t6,t7,t8]]

    eclipse_ingress = eclipse_times * (((t - t5) % per) < (t6 - t5))
    if eclipse_ingress.any():
        inscale = np.zeros(ze.shape, dtype=float)
        tei = t[eclipse_ingress]
        inscale[eclipse_ingress] = ((fstar + phaseflux(t6)) * (1. - feclip) - fstar) * \
            ((tei - tei.min()) / (tei.max() - tei.min())) 
    else:
        inscale = 0.

    eclipse_egress = eclipse_times * (((t - t7) % per) < (t8 - t7))
    if eclipse_egress.any():
        egscale = np.zeros(ze.shape, dtype=float)
        tee = t[eclipse_egress]
        egscale[eclipse_egress] = ((fstar + phaseflux(t7)) * (1. - feclip) - fstar) * \
            ((tee - tee.max()) / (tee.max() - tee.min())) 
    else:
        egscale = 0.

    # Now compute the full light curve:
    full_curve = transit_curve * eclip
    full_curve[eclipse_times * (ze < (1. - k))] = fstar
    full_curve = full_curve - inscale + egscale 

    if nchan>0:
        if len(t.shape)==2:  # Data entered as 2D
            full_curve *= (1. + cparams.reshape(nchan, 1))
        else: # Data entered as 1D
            full_curve = (full_curve.reshape(nchan, full_curve.size/nchan) * \
                (1. + cparams.reshape(nchan, 1))).ravel()

    return full_curve


def modeleclipse_simple(params, tparams, func, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN orbit.

    :INPUTS:
      params -- (3)-sequence with eclipse parameters to FIT:
        the time of conjunction for each individual eclipse (Tc),

        eclipse depth (dimensionless),

        stellar flux (F0),

      tparams -- (4)-sequence of transit parameters to HOLD FIXED:
        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        orbital period (same units as Tc and t)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      t -- numpy array.  Time of observations.
    """
    # 2011-05-31 08:35 IJMC: Created anew, specifically for eclipses.

    ecc = 0.
   

    if (tparams[0] * tparams[1]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], tparams[3], (180./np.pi)*np.arccos(tparams[0]*tparams[1]), \
                    t, 1./tparams[1], ecc=ecc)

    # Uniform-disk occultation:
    TLC =  (func(z, tparams[2]))

    # Appropriately scale eclipse depth:
    model = params[2] * (1. + params[1] * (TLC - 1.) / (tparams[2]*tparams[2]))

    return model



def modeleclipse_simple14(params, tparams, func, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN orbit.

    :INPUTS:
      params -- (14+3)-sequence with eclipse parameters to FIT:
        the multiplicative sensitivity effects (c0, ..., c13), which
        affect each bit of data as (1. + c_j) * ...  HOWEVER, to keep
        these from becoming degenerate with the overall stellar flux
        level, only 13 of these are free parameters: the first (c0)
        will always be set such that the product PROD_j(1 + c_j) = 1.

        the time of conjunction for each individual eclipse (Tc),

        eclipse depth (dimensionless),

        stellar flux (F0),

      tparams -- (4)-sequence of transit parameters to HOLD FIXED:
        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        orbital period (same units as Tc and t)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      t -- numpy array.  Time of observations.  
         Must either be of size (14xN), or if a 1D vector then
         t.reshape(14, N) must correctly reformat the data into data
         streams at 14 separate positions.
    """
    # 2011-05-31 08:35 IJMC: Created anew, specifically for eclipses.


    # Separate the c (sensitivity) and t (transit) parameters:
    cparams = params[0:14].reshape(14, 1)
    params = params[14::]

    cparams[0] = 1./(1.+cparams[1::]).prod() - 1.
    
    tis1D = False # we want "t" to be 2D

    if len(t.shape)==1:
        t = t.reshape(14, t.size/14)
        tis1D = True  # "t" is 1D
    elif len(t.shape)>2:
        print "t is of too high a dimension (>2)"
        return -1

    # Get the vanilla transit light curve:
    model = modeleclipse_simple(params, tparams, func, t)
    
    # Apply sensitivity calibrations:
    model *= (1. + cparams)
    
    if tis1D:
        model = model.ravel()

    return model



def modeltransit14(params, func, per, t):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period, and
    assuming MIPS-type data with 14 separate sensitivity dependencies.

    :INPUTS:
      params -- (14+5+N)-sequence with the following:

        the multiplicative sensitivity effects (c0, ..., c13), which
        affect each bit of data as (1. + c_j) * ...  HOWEVER, to keep
        these from becoming degenerate with the overall stellar flux
        level, only 13 of these are free parameters: the first (c0)
        will always be set such that the product PROD_j(1 + c_j) = 1.

        the time of conjunction for each individual transit (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar), 

        stellar flux (F0),

        the limb-darkening parameters u1 and u2:
             
          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients

          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients

          OR:
            Nothing at all (i.e., only 5 parameters).

      func -- function to fit to data, e.g. transit.occultquad

      per -- float.  Orbital period, in days.

      t -- numpy array.  Time of observations.  
         Must either be of size (14xN), or if a 1D vector then
         t.reshape(14, N) must correctly reformat the data into data
         streams at 14 separate positions.

    :SEE ALSO:
      :func:`modeltransit`
    """
    # 2011-05-26 13:37 IJMC: Created, from the 'vanilla' modeltransit.

    # Separate the c (sensitivity) and t (transit) parameters:
    cparams = params[0:14].reshape(14, 1)
    tparams = params[14::]

    cparams[0] = 1./(1.+cparams[1::]).prod() - 1.
    
    tis1D = False # we want "t" to be 2D

    if len(t.shape)==1:
        t = t.reshape(14, t.size/14)
        tis1D = True  # "t" is 1D
    elif len(t.shape)>2:
        print "t is of too high a dimension (>2)"
        return -1

    # Get the vanilla transit light curve:
    model = modeltransit(tparams, func, per, t)
    if np.sum(model + 1)==0:
        model = -np.ones(t.shape, dtype=float)
    
    # Apply sensitivity calibrations:
    model *= (1. + cparams)
    
    if tis1D:
        model = model.ravel()

    return model







def mcmc_transit_single(flux, sigma, t, per, func, params, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 5-parameter eclipse function of transit with KNOWN period

    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        per : scalar
                    Known orbital period (same units as t)

        func : function
                    Function to model transit (e.g., transit.occultuniform)

        params : 5+N parameters to be fit
          [T_center, b, Rstar/a, Rp/Rstar, Fstar] + (limb-darkening parameters?)
              #[Fstar, t_center, b, v (in Rstar/day), p (Rp/Rs)]

        stepsize :  1D or 2D array
                if 1D: array of 1-sigma change in parameter per iteration
                if 2D: array of covariances for new parameters

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step
    
    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia    
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work, and from K. Stevenson's MCMC
    #                        example implementation.
    # 2011-05-24 15:52 IJMC: testing the new 'holdfixed' option.
    # 2011-11-02 22:08 IJMC: Now cast numit as int

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    params     = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be positive definite:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modeltransit(params, func, per, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            if len(stepsize.shape)==1:
                nextp    = np.random.normal(params,stepsize)
            else:
                nextp = np.random.multivariate_normal(params, stepsize)

            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modeltransit(nextp, func, per, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum() 

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi


def mcmc_transit_single14(flux, sigma, t, per, func, params, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 5-parameter eclipse function of transit with KNOWN period

    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        per : scalar
                    Known orbital period (same units as t)

        func : function
                    Function to model transit (e.g., transit.occultuniform)

        params : 14+5+N parameters to be fit
          [c0,...,c13] + [T_center, b, Rstar/a, Rp/Rstar, Fstar] + (limb-darkening parameters?)

        stepsize :  1D or 2D array
                    If 1D: 1-sigma change in parameter per iteration
                    If 2D: covariance matrix for parameter changes.

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step
    
    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia    
    """
    # 2011-05-27 13:46 IJMC: Created
    # 2011-06-23 13:26 IJMC: Now accepts 2D covariance stepsize inputs.
    # 2011-11-02 22:09 IJMC: Cast numit as int

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    params     = np.copy(params)
    #params[0] = 1./(1.+params[1:14]).prod() - 1.
    bestp      = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be held fixed:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modeltransit14(params, func, per, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq
    print "zmodel [0,1,2]=", zmodel.ravel()[0:3]
    print "Initial chisq is %5.1f" % currchisq

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            if len(stepsize.shape)==1:
                nextp    = np.random.normal(params,stepsize)
            else:
                nextp = np.random.multivariate_normal(params, stepsize)

            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #nextp[0] = 1./(1. + nextp[1:14]).prod() - 1.
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modeltransit14(nextp, func, per, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum() 

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi

def mcmc_eclipse(flux, sigma, t, func, params, tparams, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 3-parameter eclipse function with KNOWN orbit
    
    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        func : function
                    Function to model eclipse (e.g., :func:`transit.occultuniform`)

        params : parameters to be fit:  
          EITHER:
             [T_center, depth, Fstar]
          OR:
             [c0, ..., c13, T_center, depth, Fstar]

        params : 4 KNOWN, CONSTANT orbital parameters
          [b, Rstar/a, Rp/Rstar, period]

        stepsize :  1D array
                Array of 1-sigma change in parameter per iteration

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step
    
    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia    
    """
    # 2011-05-31 10:48 IJMC: Created from mcmc_transit
    # 2011-11-02 17:14 IJMC: Now cast numit as int

    import numpy as np

    #Initial setup
    if len(params) > 14:
        modelfunc = modeleclipse_simple14
    else:
        modelfunc = modeleclipse_simple

    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    params     = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be positive definite:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modelfunc(params, tparams, func, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            nextp    = np.random.normal(params,stepsize)
            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modelfunc(nextp, tparams, func, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum() 

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi


#def t14(per, ars, p0, 
#
    #t14 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. + k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t23 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. - k**2) - b**2)/np.sin(np.arccos(cosi)))


def fiteclipse(data, sv, ords, tlc, edata=None, index=None, dotransit=True, dopb=True):
    """data: time series to fit using least-squares.

      sv:  state vectors (e.g., various instrumental parameters)

      ords: orders to raise each sv vector to: e.g., [1, [1,2], 3]

      tlc:  eclipse light curve

      edata: error on the data (for chisq ONLY! No weighted fits.)

      index: array index to apply to data, sv, and tlc

      dopb: do prayer-bead uncertainty analysis

      dotransit: include tlc in the fitting; otherwise, leave it out.
      """
    # 2012-01-05 11:25 IJMC: Created

    import analysis as an

    #Simple prayer-bead analysis routine (using matrix multiplication):
    def pbanal(data, xmatrix):
        nobs, ncoef = xmatrix.shape
        solns = np.zeros((nobs, ncoef), float)
        solns[0] = np.dot(np.linalg.pinv(xmatrix), data)
        model = np.dot(xmatrix, solns[0])
        residual = data - model
        for ii in range(1, nobs):
            fakedata = model + np.concatenate((residual[ii::], residual[0:ii]))
            solns[ii] = np.dot(np.linalg.pinv(xmatrix), fakedata)
        return solns



    nobs = len(data)
    if sv is None:
        sv = np.ones((0, nobs))
    else:
        sv = np.array(sv, copy=False)
    if sv.size>0 and sv.size==sv.shape[0]:
        sv = sv.reshape(1, len(sv))
    nsv = sv.shape[0]
    if index is None:
        index = np.ones(nobs, dtype=bool)
    else:
        index = np.array(index, copy=False)

    if edata is None:
        edata = np.ones(nobs)
    elif not hasattr(edata, '__iter__'):
        edata = np.tile(edata, nobs)
    else:
        edata = np.array(edata, copy=False)
    if len(edata.shape)==1:
        weights = np.diag(1./edata**2)
    elif len(edata.shape)==2:
        weights = 1./edata**2

    xmat = np.ones((1, nobs), float)
    if dotransit:
        xmat = np.vstack((xmat, tlc))
    for jj in range(nsv):
        if hasattr(ords[jj], '__iter__'):
            for ord in ords[jj]:
                xmat = np.vstack((xmat, sv[jj]**ord))
        else:
            xmat = np.vstack((xmat, sv[jj]**ords[jj]))

    xmat = xmat.transpose()
    nparam = xmat.shape[1]
    prayerbead = pbanal(np.log(data[index]), xmat[index])
    if dotransit:
        depth = prayerbead[0,1]
        udepth = an.dumbconf(prayerbead[:, 1], .683, type='central', mid=prayerbead[0,1])[0]
    else:
        edepth, udepth = 0., 0.
    model = np.exp(np.dot(xmat, prayerbead[0,:]))
    mods = np.exp(np.array([(np.dot(xmat, prayerbead[ii,:])) for ii in range(index.sum())]))
    chisq = ((np.diag(weights) * (data - model)**2)[index]).sum()
    bic   = chisq + nparam * np.log(index.sum())
    return (depth, udepth), (chisq, bic), prayerbead, model, mods



def analyzetransit_general(params, time, data, limb_dark=None, NP=1, weights=None, dopb=False, domcmc=False, gaussprior=None, ngaussprior=None, uniformprior=None, nsigma=5, maxiter=10, parinfo=None, nthread=1, nstep=2000, nwalker_factor=8, GRmetric=1.03, xtol=1e-12, ftol=1e-10, errscale=1e6, svs=None, verbose=False, savefile=None, numint=None, ninterval=None):
    """
    Fit transit to data, and estimate uncertainties on the fit.

    :INPUTS:
     params : sequence
       A guess at the best-fit model transit parameters, to be passed
       to :func:`modeltransit_batman`.  

       If 'svs' are passed in (see below), then params should have one
       additional value concatenated on the end as the coefficient per
       state vector.

     time : sequence
       time values (e.g., if time = BJD_TDB - BJD_0 then you should set params = [BJD_0, ....]

     data : sequence
       photometric values (i.e., the transit light curve) to be fit to.

     NP : int
       number of normalizing polynomial coefficients
       (cf. :func:`modeltransit_general`)

     limb_dark : set to:
          uniform, linear, quadratic, square-root, logarithmic, nonlinear...
          (uniform/0, linear/1, quadratic/2, sqrt/-2, nonlinear/4)

     weights : sequence
       weights to the photometric values.  If None, weights will be
       set equal to the inverse square of the residuals to the
       best-fit model.  In either case, extreme outliers will be
       de-weighted in the fitting process.  This will not change the
       values of the input 'weights'.

     nsigma : scalar
       Residuals beyond this value of sigma-clipped standard
       deviations will be de-weighted.

     maxiter : int > 0
       Maximum number of times to loop through and remove outliers.

     nthread : int >0
       Number of multiprocessing cores/threads to use

     parinfo:
       Optional input for Prayed-bead function: :func:`analysis.prayerbead`

     dopb : bool
       If True, run prayer-bead (residual permutation) error analysis.

     domcmc : bool
       If True, run Markov Chain Monte Carlo error analysis (requires EmCee)

     GRmetric : scalar > 1
       When Gelman-Rubin metric reaches this value or less, MCMC
       analysis terminates.

     nstep : int
       Number of steps for EmCee MCMC run.  This should be *at least*
       several thousand.

     errscale: scalar
       See :func:`modeltransit_general`

     svs : None, or sequence of 1D NumPy arrays
       State vectors, for additional decorrelation of data in a
       least-squares sense. See :func:`modeltransit_general`

    :OUTPUTS:
      (eventually, some object with useful fields)

    :SEE_ALSO:
       :func:`modeltransit_general`
    """
    # 2012-05-03 11:42 IJMC: Created
    # 2012-05-08 16:59 IJMC: Added ngaussprior option; NL can be negative.
    # 2013-04-01 09:38 IJMC: Added 'svs' option; rejiggered errscale, smallplanet
    # 2013-04-18 12:16 IJMC: Made this a bit smarter; now if MCMC
    #                        sampler finds a new best fit,
    #                        optimization is re-run and the sampler is
    #                        restarted.
    # 2013-10-09 06:51 IJMC: Added uniformprior option.
    # 2015-11-18 17:58 IJMC: Updated; also now uses BATMAN instead.

    import emcee
    #from kapteyn import kmpfit
    import analysis as an
    import tools
    from scipy import optimize
    import transit
    import phasecurves as pc
    import pylab as py
    from scipy import signal

    # Parse inputs:
    inputparams = np.array(params, copy=True)
    bestparams = np.array(params, copy=True)
    nparams = len(params)
    ndim = len(params)
    nobs = len(data)
    nwalkers = nwalker_factor * ndim
    if nwalkers % 2: nwalkers += 1
    limb_dark, NL = get_ldtype(limb_dark)

    # Set up State Vectors:
    if svs is None:
        nsvs = 0
    else:
        if isinstance(svs, np.ndarray) and svs.ndim==1:
            svs = svs.reshape(1, svs.size)
        nsvs = len(svs)

    labs = np.array(['Tt', 'Per', 'inc', 'a/R*', 'Rp/Rs', 'ecc', 'omega', 'lg10(dilut)'] + \
                        ['p%i' % val for val in range(NP)] + \
                        ['LD%i' % val for val in range(np.abs(NL))] + \
                        ['SV%i' % val for val in range(nsvs)])

    if parinfo is None:
        parinfo = [None] * nparams
    if gaussprior is None:
        gaussprior = [None] * nparams
    if uniformprior is None:
        uniformprior = [None] * nparams


    fitkw = dict(gaussprior=gaussprior, ngaussprior=ngaussprior, uniformprior=uniformprior, \
                     nans_allowed=False)

    # Run through an initial fitting routine, to flag and de-weight outliers:
    d5 = signal.medfilt(data, 5)
    filtDat = data/d5
    goodind = (data > 0) * (np.abs(filtDat - 1.) <= (250 * an.dumbconf(filtDat, .683)[0]))
    goodind *= (np.abs(data / np.median(data)-1) <= (250 * (data/np.median(data)).std()))
    #goodind = np.isfinite(data)

    prev_ngood = goodind.sum()
    newBadPixels = True
    niter = 0
    if weights is None:
        weights = np.ones(nobs) / data[goodind].var()
        weights[True - goodind] = 1e-18
        scaleWeights = True
    else:
        weights = np.array(weights, copy=True)
        scaleWeights = False

    ph = time % bestparams[1]
    while newBadPixels and niter <= maxiter:
        fitargs = (modeltransit_batman, time, NL, NP, None, numint, ninterval, data, weights, fitkw)
        mod = modeltransit_batman(bestparams, *fitargs[1:-3])
        lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        lightcurveFit = an.fmin(pc.errfunc, bestparams, args=fitargs, full_output=True, retall=True)
        inputFit = an.fmin(pc.errfunc, inputparams, args=fitargs, full_output=True, retall=True)
        lsq_fitchisq = pc.errfunc(lsq_fit[0], *fitargs)
        LCfitchisq = pc.errfunc(lightcurveFit[0], *fitargs)
        inputchisq = pc.errfunc(inputparams, *fitargs)
        inputfitchisq = pc.errfunc(inputFit[0], *fitargs)
        # Select best fit, and pick walker starting positions, excluding bad points:
            
        if verbose:
            print "lsq_chi = %1.1f, LC_chi = %1.1f, ifit_chi=%1.1f, inp_chi = %1.1f" % \
                (lsq_fitchisq, LCfitchisq, inputfitchisq, inputchisq)

        if lsq_fitchisq < LCfitchisq and lsq_fitchisq < inputchisq and lsq_fitchisq < inputfitchisq:
            covar = lsq_fit[1]
        elif LCfitchisq < inputchisq and LCfitchisq < inputfitchisq:
            lsq_fit = lightcurveFit
            covar = None
        elif inputchisq < inputfitchisq:
            lsq_fit = [inputparams.copy()]
            covar = None
        else:
            lsq_fit = inputFit
            covar = None

        bestparams = lsq_fit[0]
        p0 = np.array(lightcurveFit[-1])[-nwalkers:]
        if p0.shape[0]<nwalkers:  p0 = np.tile(p0, (nwalkers, 1))[0:nwalkers]
        if verbose:
            print '   Best-fit parameters are currently:'
            for jjj in xrange(nparams):
                print '%17s, %1.7f' % (labs[jjj], bestparams[jjj])


       
        model = fitargs[0](bestparams, *fitargs[1:-3])
        residuals = data - model
        new_goodind = (True - (np.abs((residuals-np.median(residuals)) / (an.dumbconf(residuals[goodind], .683)[0])) > nsigma))

        goodind = goodind * new_goodind
        ngood = goodind.sum()
        if scaleWeights:
            weights *= nobs/pc.errfunc(bestparams, *fitargs)
        weights[True - goodind] = 1e-18
        if ngood==prev_ngood:
            newBadPixels = False
        else:
            newBadPixels = True
            prev_ngood = ngood
        niter += 1
        #pdb.set_trace()

    # Redefine fitting arguments, with outliers de-weighted:
    fitargs = (modeltransit_batman, time, NL, NP, None, numint, ninterval, data, weights, fitkw)
    fitchisq = pc.errfunc(bestparams, *fitargs)

    # Now run a prayer-bead analysis.
    if dopb:
        print "Starting prayer-bead analysis"
        pb_fits = an.prayerbead(bestparams, *fitargs, parinfo=parinfo, xtol=xtol)
        bestparams = pb_fits[0].copy()
        
    else:
        pb_fits = None

    # Now run an MCMC analysis:
    if domcmc:
        print "Starting MCMC analysis"
        # Initialize sampler:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pc.lnprobfunc, args=fitargs, threads=nthread)

        if verbose:
            print '   Initial positions for chains are approximately:'
            for jjj in xrange(nparams):
                print "%17s:  %1.7f +/- %1.7f" % (labs[jjj], np.median(p0[:,jjj]), np.std(p0[:,jjj]))

        # Run burn-in
        pos1, prob1, state1 = sampler.run_mcmc(p0, nstep) #min(2000, nstep))
        if verbose:
            print '   Positions after burn-in is approximately:'
            for jjj in xrange(nparams):
                print "%17s:  %1.7f +/- %1.7f" % (labs[jjj], np.median(sampler.flatchain[:,jjj]), np.std(sampler.flatchain[:,jjj]))
        #for ii in range(2):
        #    pos, prob, state = sampler.run_mcmc(pos, max(1, (ii+1)*nstep/5))
        #    if (bestchisq + 2*max(prob)) > ftol: #-2*prob < bestchisq).any(): # Found a better fit! Optimize:
        #        if verbose: print "Found a better fit; re-starting MCMC... (%1.8f, %1.8f)" % (bestchisq, (bestchisq + 2*max(prob)))
        #        bestparams = pos[(prob==prob.max()).nonzero()[0][0]].copy()
        #        lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        #        bestparams = lsq_fit[0]
        #        covar = lsq_fit[1]
        #        bestchisq = pc.errfunc(bestparams, *fitargs)
        #        ii = 0 # Re-start the burn-in.
        #        #pos[prob < np.median(prob)] = bestparams

        #    pos[pos[:,2]>90,2] = pos[pos[:,2]>90,2] % 360.
        #    pos[pos[:,2]>90,2] = 180 - pos[pos[:,2]>90,2] # reset inclination
        #    badpos = (prob < np.median(prob))
        #    if (pos[True-badpos].std(0) <= 0).any():
        #        goodpos_unc = np.vstack((np.abs(bestparams / 100.), pos[True-badpos].std(0))).max(0)
        #        pos[badpos] = np.random.normal(bestparams, goodpos_unc, (badpos.sum(), ndim))
        #    else:
        #        pos[badpos]  = np.random.multivariate_normal(bestparams, np.cov(pos[True-badpos].transpose()), badpos.sum())
        #
        #pos[badpos] = bestparams

        # Run main MCMC run:
        sampler.reset()
        iter = 0
        converged = False
        pos2 = pos1
        while iter < 10 and not converged:
            pos2, prob2, state2 = sampler.run_mcmc(pos2, nstep)
            if verbose:
                print '   Positions after MCMC-ing are approximately:'
                for jjj in xrange(nparams):
                    print "%17s:  %1.7f +/- %1.7f" % (labs[jjj], np.median(sampler.flatchain[:,jjj]), np.std(sampler.flatchain[:,jjj]))
            if (np.abs(sampler.lnprobability)>1e6).sum() > (0.5*sampler.lnprobability.size): stop
            GRvalue = tools.gelman_rubin(sampler.chain).max()
            converged = GRvalue < GRmetric
            iter += 1
            mcmc_params = sampler.chain[sampler.lnprobability==sampler.lnprobability.max()][0]
            mc_chisq1 = pc.errfunc(mcmc_params, *fitargs)
            mc_chisq2 = pc.errfunc(np.median(sampler.flatchain,0), *fitargs)
            if fitchisq==min(fitchisq, mc_chisq1, mc_chisq2):
                print "Iterating MCMC, %i steps done and Gelman-Rubin metric=%1.4f" % (sampler.chain.size/nwalkers/ndim, GRvalue)
                pass
            else: # Reset the sampler, re-fit, and restart:
                print "Found a better fit; re-starting MCMC... (%1.8f)" % fitchisq
                                                                                  
                if mc_chisq1==min(fitchisq, mc_chisq1, mc_chisq2):
                    lightcurveFit = an.fmin(pc.errfunc, mcmc_params, args=fitargs, full_output=True, retall=True)            
                elif mc_chisq2==min(fitchisq, mc_chisq1, mc_chisq2):
                    lightcurveFit = an.fmin(pc.errfunc, np.median(sampler.flatchain,0), args=fitargs, full_output=True, retall=True)
                lightcurveModel = fitargs[0](lightcurveFit[0], *fitargs[1:-3])
                weights *= nobs/pc.errfunc(lightcurveFit[0], *fitargs)
                fitkw['nans_allowed'] = False
                fitargs = (modeltransit_batman, time, NL, NP, None, numint, ninterval, data, weights, fitkw)
                fitchisq = pc.errfunc(lightcurveFit[0], *fitargs)
                pos2 = np.array(lightcurveFit[-1])[-nwalkers:]
                if pos2.shape[0]<nwalkers:  pos2 = np.tile(p0, (nwalkers, 1))[0:nwalkers]
                sampler.reset()
                pos2, prob2, state2 = sampler.run_mcmc(pos2, nstep) 
                sampler.reset()
                iter = 0
                converged = False

        ## Run main MCMC run:
        #pos, prob, state = sampler.run_mcmc(pos, max(1, (ii+1)*nstep/4))
        #sampler.reset()
        #steps_taken = 0
        #while steps_taken < nstep:
        #    pos, prob, state = sampler.run_mcmc(pos, 1)
        #    steps_taken += 1
        #    if (bestchisq + 2*max(prob)) > ftol: # Found a better fit! Optimize:
        #        if verbose: print "Found a better fit; re-starting MCMC... (%1.8f, %1.8f)" % (bestchisq, (bestchisq + 2*max(prob)))
        #        bestparams = pos[(prob==prob.max()).nonzero()[0][0]].copy()
        #        lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=1e-10)
        #        bestparams = lsq_fit[0]
        #        covar = lsq_fit[1]
        #        bestchisq = pc.errfunc(bestparams, *fitargs)
        #        steps_taken = 0 # Re-start the MCMC
        #        sampler.reset()
        #        pos[prob<np.median(prob)] = bestparams
        #
        ## Test if MCMC found a better chi^2 region of parameter space:
        #mcmc_params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        #mcmc_fit = optimize.leastsq(pc.devfunc, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=1e-10)
        #if pc.errfunc(mcmc_fit[0], *fitargs) < pc.errfunc(bestparams, *fitargs):
        #    bestparams = mcmc_fit[0]
        #else:
        #    pass
        
    else:
        sampler = None

    print "Finished MCMC! Doing final fits and tidying up loose ends..."
    mcmc_params = sampler.chain[sampler.lnprobability==sampler.lnprobability.max()][0]
    mc_chisq1 = pc.errfunc(mcmc_params, *fitargs)
    mc_chisq2 = pc.errfunc(np.median(sampler.flatchain,0), *fitargs)     
    # Run a second fit with scipy.optimize.leastsq:
    lightcurveFit1 = an.fmin(pc.errfunc, bestparams, args=fitargs, full_output=True, retall=True)
    lightcurveFit2 = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
    LCfitchisq1 = pc.errfunc(lightcurveFit1[0], *fitargs)
    LCfitchisq2 = pc.errfunc(lightcurveFit2[0], *fitargs)
    # Select best fit, and pick walker starting positions, excluding bad points:
    if LCfitchisq1>LCfitchisq2:
        lightcurveFit = lightcurveFit2
    else:
        lightcurveFit = lightcurveFit1
        

    lightcurveFit = an.fmin(pc.errfunc, lightcurveFit[0], args=fitargs, full_output=True, retall=True)
    bestparams = lightcurveFit[0]
    bestchisq = pc.errfunc(bestparams, *fitargs)
    
    lightcurveModel = fitargs[0](bestparams, *fitargs[1:-3])
    rms = an.dumbconf(data - lightcurveModel, .683)[0]
    
    if savefile is not None:
        tt, per = lightcurveFit[0][0:2]
        timeplot = np.linspace(tt-0.5, tt+0.5, 1000)
        plotargs = list(fitargs)
        plotargs[1] = timeplot
        plotmodel = plotargs[0](lightcurveFit[0], *tuple(plotargs)[1:-3])
        timephase = (time - tt + per/2.) % per - per/2.
        timeplotphase = (timeplot - tt + per/2.) % per - per/2.
        py.figure()
        ax=py.subplot(111, position=[.15, .15, .8, .8])
        py.plot(24*timephase, data, 'ok')
        py.plot(24*timeplotphase, plotmodel, '-r', linewidth=2)
        py.xlim([-3, 3])
        py.ylim([1. - lightcurveFit[0][4]**2 - rms*5, 1. + rms*5])
        py.xlabel('Hours From Mid-Transit')
        py.ylabel('Normalized Flux')
        py.minorticks_on()
        py.text(.9, .9, os.path.split(savefile)[1], horizontalalignment='right', transform=ax.transAxes)

        valid = (sampler.lnprobability<0).prod(0).astype(bool)
        tools.hparams(np.vstack(sampler.chain[:,valid,:]), 100, labs=labs, plotmid=lightcurveFit[0])
        tools.plotcorrs(np.vstack(sampler.chain[:,valid,:]), labs=labs, getdist=True, plotmid=bestparams)

        mode = 'batman'
        output_vals = 'sampler.chain', 'sampler.lnprobability', 'lightcurveFit', 'lightcurveModel', 'timeplot', 'plotmodel', 'time', 'data', 'weights', 'labs', 'fitkw', 'mode', 'savefile'
        output = dict()
        for val in output_vals:
            exec('output["%s"] = %s' % (val, val))
        tools.savepickle(output, savefile + '.pickle')
        tools.printfigs(savefile + '.pdf', pdfmode='gs')
        py.close('all')

    
    output_vals = 'lightcurveFit', 'pb_fits', 'sampler', 'weights', 'lightcurveModel'
    ret = dict()
    for val in output_vals:
        exec('ret["%s"] = %s' % (val, val))
    return ret




def bls_simple(times, flux, prange, dlogper=0.0004, nbins=150, maxwid=10, nthreads=1):
    """Implement a simple/stupid version of the Box-Least-Squares algorithm.

    :INPUTS:
      times : 1D NumPy array.
        timestamps of the input time series

      flux : 1D NumPy array.
        input time series

      prange : 2-sequence
        Minimum and maximum periods to search.

      dlogper : positive scalar
        Test periods will be uniformly separated in log-space, with
        this spacing in log10.

      nbins : positive int
        Number of bins to use when binning the phase-folded data.  The
        width of each bin will depend on the test period considered,
        as (P/nbins)

      maxwid : positive int
        Maximum width of the transit signal sought, in terms of the
        number of phase bins.  

      nthreads : positive int
        Number of multiprocessing threads to use.

    :OUTPUTS:
        (test_periods, reduction_in_dispersion)

    :NOTES:
        Transits shorter than (prange[0]/nbins) and longer than
        (maxwid*prange[1]/nbins) may not be correctly modelled.

        This routine is my own crude attempt at a box-fitting
        least-squares algorithm.  Note however that this is *NOT* the
        well-publicized (and more rigorous) version of Kovacs et
        al. (2002).  Here for each trial period, I bin the data and
        construct simple box-shaped transit models for combinations of
        ingress and egress times. For each model the reduction in the
        standard deviation of (binned data - model) is calculated; for
        each period, the greatest reduction value is reported.
    """
    #2013-10-11 14:05 IJMC: Created

    #prange = [0.43, .47]
    #nbins = 200
    #dlogper = 0.0002
    #maxwid = 10

    if nthreads>1:
        from multiprocessing import Pool
        pool = Pool(processes=nthreads)

    nper = int(np.ceil(np.log10(prange[1] / prange[0]) / np.log10(1. + dlogper)))
    periods = prange[0] * (1.0+dlogper)**np.arange(nper)
    phasebins = np.linspace(0, 1, nbins+1)

    if nthreads>1:
        reduction = pool.map(get_reduction_factor, [[periods[ii], times, flux, phasebins, maxwid] for ii in xrange(nper)])
    else:
        reduction = np.array([get_reduction_factor([periods[ii], times, flux, phasebins, maxwid]) for ii in xrange(nper)])

    return periods, reduction

def get_reduction_factor(args):
    """ Helper function for bls_simple.

    args = thisperiod, times, flux, phasebins, maxwid):"""
    # 2013-10-11 15:27 IJMC: Created
    from tools import errxy

    thisperiod, times, flux, phasebins, maxwid = args

    tbin, fbin, junk1, junk2 = errxy((times / thisperiod) % 1.0, flux, phasebins, xerr=None, yerr=None)
    nbins = tbin.size
    model = np.zeros(nbins, dtype=float)
    myvals = np.ones((nbins, nbins))
    for i1 in range(0, nbins-1):
      for i2 in range(i1+1, min(nbins, i1+1+maxwid)):
        model[i1:i2] = np.mean(fbin[i1:i2])
        oot_val = np.mean(np.concatenate((fbin[0:i1], fbin[i2:])))
        model[0:i1] = oot_val
        model[i2:] = oot_val
        myvals[i1,i2] = (fbin - model).std()
    return (fbin.std()/myvals).max()


def modhaze_radspec_simple(params, wmod, rmod, rstar, retspec=False, filter_splines=None, w_star=None, f_star=None):
    """Add a simple ad-hoc haze model to a planet's radius spectrum.

    :INPUTS:
      params : 3-sequence
        params[0]: offset value for haze model
        params[1]: slope value (= alpha * H) for haze model
        params[2]: scaling factor for rmod

        haze = params[1] * np.log(wmod) + params[0]
        newmod = vstack((rmod*params[2], haze/(planet.rstar*an.rsun))).max(0)

      wmod : 1D NumPy array
        Wavelength grid of model spectrum 'rmod'
 
      rmod : 1D NumPy array
        Planet radius at each wavelength specified in 'wmod'
 
      rstar : scalar
        Stellar radius, in units of solar radii.
 
      retspec : bool
        If True, just return the new model spectrum. 

        If False, return the filter-averaged value of the resulting
        spectrum for each filter profile specified in 'filts'.
 
      w_star : 1D NumPy array
        Wavelength grid of stellar flux, in same units as wmod (but
        not necessary of same size!). 

      f_star : 1D NumPy array
        Photon flux density of stellar flux at wavelengths specified
        in w_star.

    :NOTES:
      Because this routine makes use of the numpy.interp function,
      using a smaller input model grid can significantly speed things
      up.
    """
    # 2013-12-20 16:37 IJMC: Created 2-3 weeks before this for GJ3470b
    #                        analysis.
    
    from analysis import rsun

    offset, hscale, mscale = params
    haze = (hscale * np.log(wmod) + offset)
    newmod = np.vstack((rmod*mscale, haze/(rstar*rsun))).max(0)
    if retspec:
        ret = newmod
    else:
        ret = np.zeros(len(filter_splines))
        for jj, spline in enumerate(filter_splines):
            spline_interp = spline(wmod) * np.interp(wmod, w_star, f_star)
            ret[jj] = (newmod * spline_interp).sum() / spline_interp.sum()
    return ret

def createJKTEBOPinput(*args, **kw):
    """Create an input file suitable for running JKTEBOP.

    :INPUT:
      A long sequence of numerical & string inputs, defined as
      follows.  If simulating a transiting planet, object "A"
      indicates the star and object "B" indicates the planet.

        JKTEBOP "Task" to do (from 1 to 9) -- for now, only 2 is valid.

        Integrating ring size (deg) -- between 0.1 and 10.0 degrees   

        Sum of the radii   (R_A/a + R_B/a)

        Ratio of the radii (R_B / R_A)	      

        Orbital inclination (deg)  

        Mass ratio of system   (M_B / M_A)  

        e*cos(omega) or orbital eccentricity -- see NOTES below.

        e*sin(omega) or periastron longitude (deg) -- see NOTES below.

        Gravity darkening (object A; typically set to 1.0)

        Grav darkening (object B; typically set to 1.0)

        Surface brightness ratio  (typically set to 0.0)

        Amount of third light    

        Limb-darkening law type for star A:
           one of ('lin', 'log', 'sqrt', 'quad', 'cub', '4par')    

        Limb-darkening law type for star A:
           (any of the preceding options, or 'same' for same) 

        LD star A: 
          if LD law is '4par' then one after another enter the 4 values
          (coef1, coef2, coef3, coef).  Otherwise, (linear coeff, nonlincoef)   

        LD star B: 
          Same as described above.

        Reflection effect star A   

        Reflection effect star B 

        Orbital phase of primary eclipse   

        Light scale factor : scalar
            Apparent magnitude of the target star

    :OPTIONS:
      period : scalar
        Orbital period of the binary in days, used to compute the
        output timestamps. If not entered, output 'time' will be
        orbital phase.

      t0 : scalar
        Time index of mid-transit. 

      infile : str
        Name of the 'input file' to be created by this routine.

      outfile : str
        Name of the 'output file' to be created by JKTEBOP.

      datfile : str
        Name of the 'data file' to be input; has 2-3 columns of
        timestamps, photometry, and (optional) uncertainties,
        respectively.

      clobber : bool
        If true, overwrite any existing file. Otherwise: don't!


    :OUTPUT:
      If no 'filename' option was passed in, returns a list
      of strings, suitable to writing to disk via the usual:
        ::

          f = open(filename, 'w')
          f.writelines(data)
          f.close()

    :EXAMPLE:
      ::

        vals = [2, 1, 0.21, 0.15, 88.5, 0.0013, 0, 0, 1, 1, 0, 0, \
             'quad', 'lin', 0.3, 0, 0.3, 0, 0, 0, 0, 0.6]
        output = transit.createJKTEBOPinput(vals, filename='test.in', clobber=False)

    :NOTES:
      Put a negative number for the mass ratio to force the stars to
      be spherical.  The mass ratio will then be irrelevant (it is
      only used to get deformations).

      To input R_A/a and R_B/a (instead of [R_A+R_B]/a and R_B/R_A),
      give a negative value for [R_A+R_B]/a. Then it will be
      interpreted to mean R_A/a, and R_B/R_A will be interpreted as
      R_B/a.  

      If eccentricity < 10 then e and omega will be assumed to be
      e*cos(omega) and e*sin(omega). If e >= 10 then e and omega will
      be assumed to be (e+10) and omega (degrees).  The first option
      is often better unless eccentricity is larger or fixed.

      See the JKTEBOP documentation for more details on all these
      parameters.  JKTEBOP is currently available online at
      http://www.astro.keele.ac.uk/~jkt/codes/jktebop.html

    :TO_DO:
      Add other optional parameters: TMIN, LRAT, THDL, ECSW, ENSW,
      SINE, POLY, NUMI, RV1 & RV2, orbital period, reference epoch...

      Allow fitting (i.e., enable Tasks 3-9).
      """
    # 2014-08-08 09:25 IJMC: Created.

    # Parse inputs
    if len(args)==0:
        inputs = None
    elif len(args)==1:
        inputs = args[0]
    else:
        inputs = args

    if kw.has_key('clobber'):
        clobber = kw['clobber']
    else:
        clobber=False

    if kw.has_key('period'):
        period = kw['period']
    else:
        period=None

    if kw.has_key('t0'):
        t0 = kw['t0']
    else:
        t0=None

    if kw.has_key('infile'):
        infile = kw['infile']
    else:
        infile=None

    if kw.has_key('outfile'):
        outfile = kw['outfile']
    else:
        outfile=None

    if kw.has_key('datfile'):
        datfile = kw['datfile']
    else:
        datfile=None

    # Prepare explanatory text:
    textlines1 = [ \
        'Task to do (from 1 to 9)   Integ. ring size (deg)',    
        'Sum of the radii           Ratio of the radii',	       
        'Orbital inclination (deg)  Mass ratio of system',      
        'Orbital eccentricity       Periastron longitude deg',  
        'Gravity darkening (star A) Grav darkening (star B)',   
        'Surface brightness ratio   Amount of third light',     
        'LD law type for star A     LD law type for star B']
    textlines_LD2 = [ \
        'LD star A (linear coeff)   LD star B (linear coeff)',  
        'LD star A (nonlin coeff)   LD star B (nonlin coeff)']
    textlines_LD4 = [ \
                   'LD star A (coefficient 1) LD star B (coefficient 1)',
                   'LD star A (coefficient 2) LD star B (coefficient 2)',
                   'LD star A (coefficient 3) LD star B (coefficient 3)',
                   'LD star A (coefficient 4) LD star B (coefficient 4)']
    textlines2 = [ \
        'Reflection effect star A   Reflection effect star B',  
        'Phase shift of primary min Light scale factor (mag)']

    # Construct the list of output strings:
    fmtstr = '%1.16f  %1.16f  %s\n'
    outputlines = []
    outputlines.append('%i  %1.16f  %s\n' % (inputs[0], inputs[1], textlines1[0]))

    for ii in xrange(1,6):
        outputlines.append(fmtstr % (inputs[ii*2], inputs[ii*2+1], textlines1[ii]))
    outputlines.append('%s  %s  %s\n' % (inputs[12], inputs[13], textlines1[6]))

    if inputs[14]=='4par' or inputs[15]=='4par':
        NL = 4
        LDlines = textlines_LD4
    else:
        NL = 2
        LDlines = textlines_LD2

    for ii in xrange(NL):
        outputlines.append(fmtstr % (inputs[14+ii], inputs[14+ii+NL], LDlines[ii]))

    for ii in xrange(2):
        outputlines.append(fmtstr % (inputs[14+2*NL+ii*2], inputs[15+2*NL+ii*2], textlines2[ii]))

    # Handle Options
    if period is not None:
        outputlines.append('%1.10f \n' % period)

    if t0 is not None:
        outputlines.append('%8.10f \n' % t0)

    for filename in [datfile, outfile]:
        if filename is not None:
            outputlines.append('%s  \n' % filename)

    if infile is not None:
        if os.path.isfile(infile) and not clobber:
            print "File '%s' exists and 'clobber' set to False... cannot overwrite!" % infile
            ret = -1
        else:
            f = open(infile, 'w')
            f.writelines(outputlines)
            ret = f.close()
    else:
        ret = outputlines

    return ret

def runJKTEBOP(*args, **kw):
    """Run JKTEBOP simulation from the command line, and return results.

    :INPUTS:
      See :func:`createJKTEBOPinput` for a description of the inputs.
      For now, only "Task 2" can be run through this Python interface.

    :OPTIONS:
      exe : string
        Full path to the JKTEBOP executable.

      period : scalar
        Orbital period of the binary in days, used to compute the
        output timestamps. If not entered, output 'time' will be
        orbital phase.

      infile : str
        Name of the 'input file' to be created by this routine. If no
        value is passed, the routine will create a file named
        'testX.in', where "X" is a random, long, integer.

      outfile : str
        Name of the 'output file' to be created by JKTEBOP. If no
        value is passed, the routine will create a file named
        "textX.out"

      clobber : bool
        If true, overwrite any and all existing files. Otherwise: don't!

    :NOTES:
      See the JKTEBOP documentation for more details.  JKTEBOP is
      currently available online at
      http://www.astro.keele.ac.uk/~jkt/codes/jktebop.html

    :EXAMPLE:
      ::

        vals = [2, 1, 0.21, 0.15, 88.5, 0.0013, 0, 0, 1, 1, 0, 0, \
             'quad', 'lin', 0.3, 0, 0.3, 0, 0, 0, 0, 0.6]
        inputFile = 'JKTEBOP_test.in'
        outputFile = 'JKTEBOP_test.out'
        period = 9.8 
        exe = os.path.expanduser('~')+'/proj/transit/jktebop/jktebop_orig'
        time, mag = transit.runJKTEBOP(vals, infile=inputFile, outfile=outputFile, period=period, clobber=True, exe=exe)

      ::
      
        ## THIS IS A SPECIAL EXAMPLE FOR MY CODE ONLY --
        ##    I HAVE MODIFIED 'jktebop' SO DON'T EXPECT THE FOLLOWING TO
        ##    WORK PROPERLY ON YOUR COMPUTER!
        import transit
        import pylab as py

        vals = [2, 1, 0.211, 0.154, 88.59, 0.0013, 0, 0, 1, 1, 0, 0, \
             'quad', 'lin', 0.3, 0., 0.3, 0, 0, 0, 0, 0.0]
        inputFile = 'JKTEBOP_test_mod.in'
        outputFile = 'JKTEBOP_test_mod.out'
        dataFile    = 'wasp4.dat'   # data file; only its timestamps are used!
        period = 1.3382320363      # Orbital period, in days
        t0 = 54740.62  # Time of central transit, in days.
        exe = os.path.expanduser('~')+'/proj/transit/jktebop/jktebop_mod'
        time, mag = transit.runJKTEBOP(vals, infile=inputFile, outfile=outputFile, datfile=dataFile, period=period, t0=t0, clobber=True, exe=exe)

        exampleData = py.loadtxt(dataFile)
        examplePhase = (exampleData[:,0] - t0) / period

        py.figure()
        py.plot(examplePhase, 10**(-0.4*exampleData[:,1]), 'ob')
        py.plot(time/period, 10**(-0.4*mag), '--r', linewidth=2)
        py.xlabel('Orbital Phase')
        py.ylabel('Normalized Flux')
        py.minorticks_on()
        py.legend(['Sample Observations', 'JKTEBOP Model'], 4)
        
    :SEE_ALSO:
      :func:`createJKTEBOPinput`

    :TO_DO:
      Enable light-curve simulations (Tasks 3-9).
    """
    # 2014-08-08 12:16 IJMC: Created.


    # Parse Inputs
    if kw.has_key('exe'):
        exe = kw.pop('exe')
    else:
        exe = os.path.expanduser('~')+'/proj/transit/jktebop/jktebop'

    if kw.has_key('period'):
        period = kw.pop('period')
    else:
        period = None

    if kw.has_key('clobber'):
        clobber = kw.pop('clobber')
    else:
        clobber=False

    if kw.has_key('infile'):
        infile = kw.pop('infile')
    else:
        inputFileExists = True
        while inputFileExists:
            randInt = np.random.uniform(0, 1e9, 1).astype(int)
            infile = 'PyJKTEBOP_%08i.in' % randInt
            inputFileExists = os.path.isfile(infile)

    if kw.has_key('outfile'):
        outfile = kw.pop('outfile')
    else:
        if infile[-3:]=='.in':
            outfile = infile[0:-3]+'.out'
        else:
            outfile = infile+'.out'

    # Create the Input File:
    createJKTEBOPinput(*args, infile=infile, clobber=clobber, outfile=outfile, period=period, **kw)

    # Run JKTEBOP
    if clobber and os.path.isfile(outfile):
        os.remove(outfile)
    os.system('%s %s' % (exe, infile))

    # Load & parse the output
    if not os.path.isfile(outfile):
        print "Output file '%s' not found -- something went wrong. Exiting..." % outfile
        ret = -1

    else:
        BOPout = np.loadtxt(outfile)
        time     = BOPout[:,0]
        magnitude = BOPout[:,1]
        if period is not None:
            time *= period
        ret = time, magnitude

    return ret


def JKTEBOP_lightcurve(v, vary, ldtype, nsine, psine, npoly, ppoly, time, dtype1, la, lb, numint, ninterval, nthreads=1):
    """Generate a JKTEBOP light curve using the F2Py-compiled library.

    :INPUTS:
      The inputs to the Fortran function 'jktebop.getmodel' are many,
      and their formatting is complicated. You may want to examine the
      Fortran source code for more insight.

        v : 138-sequence of floats
          Photometric parameters. In full, these are:

            V(0) = central surface brightness ratio (starB/starA)

            V(1) = sum of the fractional radii (radii divided by semimajor axis)
               
            V(2) = ratio of stellar radii (starB/starA)
               
            V(3) = linear limb darkening coefficient for star A
               
            V(4) = linear limb darkening coefficient for star B
               
            V(5) = orbital inclination (degrees)
               
            V(6) = e cos(omega) OR ecentricity
               
            V(7) = e sin(omega) OR omega(degrees)
               
            V(8) = gravity darkening of star A
               
            V(9) = gravity darkening of star B

            V(10) = reflected light for star A

            V(11) = reflected light for star A
                
            V(12) = mass ratio (starB/starA) for the light curve calculation
                
            V(13) = tidal lead/lag angle (degrees)
                
            V(14) = third light in units where (LA + LB + LC = 1)
                
            V(15) = phase correction factor (i.e. phase of primary eclipse)
                
            V(16) = light scaling factor (magnitudes)
                
            V(17) = integration ring size (degrees)
                
            V(18) = orbital period (days)
                
            V(19) = ephemeris timebase (days)
                
            V(20) = limb darkening coefficient 2 for star A
                
            V(21) = limb darkening coefficient 3 for star A
                
            V(22) = limb darkening coefficient 4 for star A
                
            V(23) = limb darkening coefficient 2 for star B
                
            V(24) = limb darkening coefficient 3 for star B
                
            V(25) = limb darkening coefficient 4 for star B
                
            V(26) = velocity amplitude of star A (km/s)
                
            V(27) = velocity amplitude of star B (km/s)
                
            V(28) = systemic velocity of star A (km/s)
                
            V(29) = systemic velocity of star B (km/s)
                
            V(30-56) nine lots of sine curve parameters [T0,period,amplitude]

            V(57-137) nine lots of polynomial parameters [pivot,Tstart,Tend,const,x,x2,x3,x4,x5]

        vary : 138-sequence of ints
          "Which parameters are fitted."  But for directly called
          jktebop.getmodel, the only possible relevant values are [a]
          all values zero (normal operations) or [b] all values zero,
          but vary[29]=-1 (if dtype1=8).

        ldtype : 2-sequence of ints
          LD law type for the stars A and B, respectively:

            1 --> "lin"  -- linear 
            2 --> "log"  -- linear + log
            3 --> "sqrt" -- linear + sqrt
            4 --> "quad" -- linear + quadratic
            5 --> "cub"  -- linear + cubic
            6 --> "4par" -- Claret's 4-parameter form.

        nsine, npoly : ints
          The number of sines and polynomials, respectively.

        psine, ppoly : 9-sequences
          The parameters for sines and polynomials, respectively.

        time 
          The given TIME, PHASE or CYCLE

        dtype1 : int, 1-8 inclusive.
          Precise meaning of the output value depends on DTYPE.
            1  it outputs an EBOP magnitude for given time
            2  it outputs a light ratio for the given time
            3  outputs a time of eclipse for the given =CYCLE=
            4  it simply outputs the third light value
            5  it outputs e or e*cos(omega)
            6  it outputs omega or e*sin(omega)
            7  it outputs the RV of star A
            8  it outputs the RV of star B

        la, lb : floats
          Light produced by each star    (??)

        numint : int, >= 1.
          Number of numerical integrations. Long exposure times can be
          split up into NUMINT points.

        ninterval : float
          Time interval for integrations.  If numint>1, each point in
          the numerical integration occupies a total time interval of
          NINTERVAL seconds.


    :OPTIONS:
      None (so far!)

    :NOTES:
      See the JKTEBOP documentation for more details.  JKTEBOP is
      currently available online at
      http://www.astro.keele.ac.uk/~jkt/codes/jktebop.html

      To compile JKTEBOP (v34) with F2Py, I had to rename the source
      file to be "jktebop_orig.f90", and then I ran the following command:

        f2py-2.7 -c --debug -m jktebop_f2py jktebop_orig.f90
          OR
        f2py-2.7 -c --debug -m jktebop_mod jktebop.f90

    :EXAMPLE:
      ::

        import pylab as py
        import jktebop_f2py
        import transit

        v = np.zeros(138, dtype=float)
        v[[1,2,3,4,5]] = [0.211, 0.154, 0.3, 0.3, 88.59]
        v[12] = 0.0013    # Mass ratio
        v[17] = 1         # Integration ring size; cannot be zero.
        v[18] = 1.3382320363  # Orbital period
        v[19] = 54740.62  # Transit ephemeris
        vary = np.zeros(138, dtype=int)
        ldtype = [4, 1]   # Quadratic for star, linear for planet.
        nsine, npoly = 0, 0
        psine, ppoly = np.zeros(9), np.zeros(9)
        times = py.linspace(v[19]-0.1, v[19]+0.1, 100)
        dtype1 = 1   # To compute a light curve
        la, lb = 0, 0  # (????)
        numint = 1    # Cannot be zero
        ninterval = 0 # Irrelevant, because numint=1

        magout_direct = py.array([jktebop_f2py.getmodel(v, vary, ldtype, nsine, psine, npoly, ppoly, time, dtype1, la, lb, numint, ninterval) for time in times])

        magout_alt = transit.JKTEBOP_lightcurve(v, vary, ldtype, nsine, psine, npoly, ppoly, times, dtype1, la, lb, numint, ninterval) 

    :SEE_ALSO:
      :func:`runJKTEBOP`

      """
    # 2014-08-09 09:49 IJMC: Created.
    

    # Do some basic error-trapping:
    try:
        import jktebop_f2py
    except:
        jktebop_f2py = -1

    if not hasattr(jktebop_f2py, 'getmodel'):
        print "Could not load F2Py-compiled library 'jktebop_f2py', or it does"
        print "  not contain the necessary function 'getmodel.'  Aborting..."
        return -1

    if len(v)==13:
        v, vary, ldtype, nsine, psine, npoly, ppoly, time, dtype1, la, lb, numint, ninterval = v[0:13]

    if numint<1:
        print "Numint is set to %s, but it must be >=1. Aborting..." % numint
        return -1
    
    if not hasattr(time, '__iter__'):
        time = np.array([time])

    if nthreads>1:
        from multiprocessing import Pool
        pool = Pool(processes=nthreads)


    # Define a helper function:
    def helper(all_args):
        return jktebop_f2py.getmodel(*all_args)

    # Now, run the code:
    magout = np.zeros(time.size, dtype=float)
    #magout2 = np.zeros(time.size, dtype=float)
    for ii in xrange(time.size):
        magout[ii] = jktebop_f2py.getmodel(v, vary, ldtype, nsine, psine, npoly, ppoly, time[ii], dtype1, la, lb, numint, ninterval)
        #magout2[ii] = helper((v, vary, ldtype, nsine, psine, npoly, ppoly, time[ii], dtype1, la, lb, numint, ninterval))
        
    
    #magout4 = np.array(pool.map(helper, [(v, vary, ldtype, nsine, psine, npoly, ppoly, time0, dtype1, la, lb, numint, ninterval) for time0 in time]))
    

    return magout#, magout2, magout3

def JKTEBOP_lightcurve_helper(all_args):
    """


    vv = ((v, vary, ldtype, nsine, psine, npoly, ppoly, bjd, 1, 0, 0, numint, ninterval))

    out0 = np.array([jktebop_f2py.getmodel(v, vary, ldtype, nsine, psine, npoly, ppoly, time, 1, 0, 0, numint, ninterval) for time in bjd])
    out00 = jktebop_mod.getmodelarray(v, vary, ldtype, nsine, psine, npoly, ppoly, bjd, 1, 0, 0, numint, ninterval, bjd.size)
    out1 = transit.JKTEBOP_lightcurve(*vv)
    out2 = transit.JKTEBOP_lightcurve_helper(vv)
    
    out3 = np.array(map(transit.JKTEBOP_lightcurve_helper, [(v, vary, ldtype, nsine, psine, npoly, ppoly, time0, 1, 0, 0, numint, ninterval) for time0 in bjd])).squeeze()
    out4 = np.array(pool.map(transit.JKTEBOP_lightcurve_helper, [(v, vary, ldtype, nsine, psine, npoly, ppoly, time0, 1, 0, 0, numint, ninterval) for time0 in bjd])).squeeze()


   1e4    1e5    1e6
0  0.104  0.984   9.762
1  0.103  0.979   9.878
2  0.101  0.999   9.801
3  0.252  3.028  31.8
4  0.228  3.006  30.7


    This method doesn't seem any faster!

    """
    # 2014-08-10 10:30 IJMC: Created.
    return JKTEBOP_lightcurve(*all_args)



def computeInTransitIndex(time, period, tt, t14):
    """REturn a boolean mask, True wherever the planet is in transit."""
    # 2014-08-11 17:39 IJMC: Created.
    modT = (time - tt) % period
    return (modT < t14/2.) + (modT > (period - t14/2.))


def pldEclipse(params, tparams, time, vecs):
    """Simple toy model for Pixel-Level-Decorrelation testing.

    tparams and time are for :func:`modeleclipse_simple`

    vecs are: np.vstack((phat.T, othervecs.T)); of shape (Nvec x Nobs)

    params are [eclipseCenter, ... then all the PLD coefs]

    """
     # 2014-10-04 13:35 IJMC: Created

    eparams = [params[0], 1., 1.]
    occ = modeleclipse_simple(eparams, tparams, occultuniform, time) - 1. 
    newvecs = np.vstack((occ, vecs))    
    return np.dot(params[1:], newvecs)
    #occ = modeleclipse_simple(eparams, tparams, occultuniform, time) 
    #newvecs = np.vstack((occ, vecs))    
    #return np.dot(params[1:], vecs) * occ


def analyzeeclipse_simple(guess_params, tparams, times, data2fit, weights=None, nwalkers=None, maxSteps=20000, gr_cutoff=1.1, priors=None, pool=None):
    """
    Determine best fit and uncertainties on secondary eclipse data.

    

    :SEE_ALSO:
      :doc:`modeleclipse_simple`

    returns:  bestfit, sampler, weights, bestmod
    """
    # 2015-03-04 23:20 IJMC: Created

    from analysis import fmin
    from phasecurves import errfunc, lnprobfunc
    import emcee
    import tools

    #mod0 = transit.modeleclipse_simple(guess_params, tparams, transit.occultuniform, times)

    if weights is None:
        rms = data2fit.std()
        weights = np.ones(data2fit.size, float)/rms**2


    fitkw = dict(gaussprior=priors)
    for ii in [0,1]:
        mcargs = (modeleclipse_simple, tparams, occultuniform, times, data2fit, weights, fitkw)
        bestfit = fmin(errfunc, guess_params, args=mcargs, full_output=True, disp=False, retall=True)
        weights *= data2fit.size / bestfit[1]
        bestmod = modeleclipse_simple(bestfit[0], tparams, occultuniform, times)


    ndim = bestfit[0].size
    if nwalkers is None:
        nwalkers = 20*ndim

    pos0 = np.array(bestfit[-1])
    if pos0.shape[0] < nwalkers:
        pos0 = np.tile(pos0, (np.ceil(1.0*nwalkers/pos0.shape[0]), 1))

    pos0 = pos0[-nwalkers:]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfunc, args=mcargs, pool=pool)
    pos1, prob1, state1 = sampler.run_mcmc(pos0, 2000)
    sampler.reset()
    
    keepRunning = True
    while keepRunning:
        pos2, prob2, state2 = sampler.run_mcmc(pos1, 2000)
        if (tools.gelman_rubin(sampler.chain) < gr_cutoff).all(): 
            keepRunning = False
        if sampler.chain.shape[1] > maxSteps:
            keepRunning = False

    return bestfit, sampler, weights, bestmod



def analyze_jktebop(guess_params, times, NL, NP, data2fit, weights=None, nwalkers=None, maxSteps=20000, gr_cutoff=1.1, priors=None, pool=None, verbose=False, dchi=0.01):
    """
    Determine best fit and uncertainties on transits, eclipses, phasecurves.

    

    :SEE_ALSO:
      :doc:`blender.modeltransit_jktebop`

    returns:  bestfit, sampler, weights, bestmod
    """
    # 2015-03-04 23:20 IJMC: Created

    from analysis import fmin
    from phasecurves import errfunc, lnprobfunc
    import emcee
    import tools
    from blender import modeltransit_jktebop

    dstep = min(2000, maxSteps)
    if verbose: print "Iterating by %i steps for total of %i steps" % (dstep, maxSteps)

    #mod0 = transit.modeleclipse_simple(guess_params, tparams, transit.occultuniform, times)

    if weights is None:
        rms = data2fit.std()
        weights = np.ones(data2fit.size, float)/rms**2

        
    def getfitargs(params, weights):
        for ii in [0,1]:
            mcargs = (modeltransit_jktebop, times, NL, NP, data2fit, weights, fitkw)
            fit = fmin(errfunc, params, args=mcargs, full_output=True, disp=False, retall=True)
            weights *= data2fit.size / fit[1]
            #bestmod = modeleclipse_simple(fit[0], tparams, occultuniform, times)
        return mcargs, fit

    fitkw = dict(gaussprior=priors)
    mcargs, bestfit = getfitargs(guess_params, weights)
    ndim = bestfit[0].size
    if nwalkers is None:
        nwalkers = 20*ndim

    def getinitialstates(params):
        states = np.array(params)
        if states.shape[0] < nwalkers:
            states = np.tile(states, (np.ceil(1.0*nwalkers/states.shape[0]), 1))
        states = states[-nwalkers:]
        stuck_index = np.abs(states.std(0)/states.mean(0)) <= 1e-6
        if stuck_index.any():
            new_dispersions = [1e-4, 1e-5, 1e-4, 1e-4, 1e-5, 1e-6, 1e-6, 1e-8] + \
                list(np.abs(states.mean(0)[8:8+NP+np.abs(NL)])/1e4)
        return states


    pos0 = getinitialstates(bestfit[-1])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfunc, args=mcargs, pool=pool)
    pos1, prob1, state1 = sampler.run_mcmc(pos0, dstep)
    sampler.reset()
    
    keepRunning = True
    iter = 0
    while keepRunning:
        iter += 1
        if verbose: print "MCMC/fit loop, iteration %i" % iter
        pos2, prob2, state2 = sampler.run_mcmc(pos1, dstep)
        if (np.abs(pos2.std(0) / pos2.mean(0)) < 1e-10).all():
            print "Your  method didn't work!"
        if (tools.gelman_rubin(sampler.chain) < gr_cutoff).all(): 
            keepRunning = False
        if sampler.chain.shape[1] > maxSteps:
            keepRunning = False
        else:
            if verbose: print "Number of completed MCMC steps: %i" % sampler.chain.shape[1]

        maxprob = sampler.lnprobability.ravel().max()
        if (-2*maxprob) < (bestfit[1] - dchi):
            if verbose: print "MCMC found a better fit (dchisq=%1.1f); restarting." % (bestfit[1] +2* maxprob)
            params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==maxprob)[0][0]]
            mcargs, bestfit = getfitargs(params, weights)
            pos1 = getinitialstates(bestfit[-1])
            if (np.abs(pos1.std(0) / pos1.mean(0)) < 1e-10).all():
                print "Your  method didn't work!"

            sampler.reset()
            keepRunning = True
        if verbose: print "Best chisq=%1.3f (vs. %1.3f)" % (bestfit[1], -2*maxprob)

    bestmod = mcargs[0](bestfit[0], *mcargs[1:-3])

    return bestfit, sampler, mcargs[-2], bestmod

def photoeccentric_maxprob(rho_circ, err_rho_circ, rho_star, err_rho_star, rho_star_max=20., npts=120, nsamp=0, plotfig=False, retvals=False, verbose=False):
    """Determine e, omega, and rho_* via a Photoeccentric analysis.

    :INPUTS:
      rho_circ, err_rho_circ : scalars
        Best-fit stellar density and uncertainty from a circular-orbit
        fit to the transit light curve.

      rho_star, err_rho_star : scalars
        Best-fit stellar density and uncertainty from other
        information, e.g. spectroscopic or asteroseismic analysis.

      rho_star_max : scalar
        Maximum density of star, in same units as above.

      npts : positive int
        Number of points in finite Maximimum-likelihood analysis.

      nsamp : int
        Number of samples to draw from 3D posterior
        distribution. Don't set it too high without testing it first;
        maybe ~10000 at most. If negative, directly display posteriors
        but don't draw samples.

      retvals : bool
        If True, return the (eccentricity, omega, g, rho_star) samples.
        If True and nsamp<1, return (nsamp, nsamp, nsamp, nsamp)

    :EXAMPLE:
      ::

        import transit

        # Quick:
        transit.photoeccentric_maxprob(1.5, 1, 4.18, 2.9,nsamp=-1,npts=240, plotfig=True)

        # Slow, but return samples:
        ecc, om, g, rho = transit.photoeccentric_maxprob(1.5, 1, 4.18, 2.9,nsamp=3000,npts=240, plotfig=True, retvals=True)


    :NOTES:
      Follows Sec. 3.4 of Dawson & Johnson (2012)
        """
    # 2015-07-21 13:57 IJMC: Created

    import tools
    if plotfig:
        import corner
    import pylab as py
    import analysis as an

    inplines = ['', 'Inputs:']
    inplines.append('')

    if hasattr(rho_circ, '__iter__') and len(rho_circ)>1 and nsamp>0:
        inplines.append("rho*_circ:  (posterior input)")
        inplines.append("rho*_meas:  %1.3f +/- %1.3f" % (rho_star, err_rho_star))
        ecc0,om0,gs0,rho0 = photoeccentric_maxprob(2.0, 999999, rho_star, err_rho_star, nsamp=nsamp,npts=npts, plotfig=False, retvals=True, verbose=False)
        bins = np.linspace(0, 30, npts*2)
        #bincens = np.vstack((bins[1:], bins[0:-1])).mean(0)
        #rhostar_spec_prob, junk  = np.histogram(rho0, bins, normed=True)
        #rhostar_circ_prob, junk  = np.histogram(rho_circ, bins, normed=True)

        if rho_circ.size >= rho0.size:
            nstep = (1.0*rho_circ.size/rho0.size)
            nsamp0 = (np.arange(rho0.size)*nstep).astype(int)
        else:
            nsamp0 = np.random.uniform(0, rho_circ.size, rho0.size).astype(int)
        rhostar_fake = rho_circ[nsamp0]/gs0**3 
        MC = np.exp(-((rhostar_fake - rho_star) / err_rho_star)**2) > np.random.uniform(size=rhostar_fake.size)
        samp_ecc, samp_om, samp_gs, samp_rhostar = ecc0[MC], om0[MC], gs0[MC], rho0[MC]
        best_ewp = np.median(samp_ecc), np.median(samp_om), np.median(samp_rhostar)
        best_g = np.median(samp_gs)

    else:
        inplines.append("rho*_circ:  %1.3f +/- %1.3f" % (rho_circ, err_rho_circ))        
        inplines.append("rho*_meas:  %1.3f +/- %1.3f" % (rho_star, err_rho_star))

        om = np.linspace(0,2*np.pi, npts-1)
        ecc = np.linspace(0, .99, npts)
        rhostar = np.linspace(0, 20, npts+1)
        #om1, ecc1 = np.meshgrid(om, ecc)
        #g1 = (1. + ecc1 * np.sin(om1)) / (1. - ecc1**2)**0.5
        om2, ecc2, rhostar2 = np.meshgrid(om, ecc, rhostar)
        g2 = (1. + ecc2 * np.sin(om2)) / (1. - ecc2**2)**0.5
        loglik = -0.5 * (((g2**3 * rhostar2 - rho_circ)/err_rho_circ)**2 + ((rhostar2-rho_star)/err_rho_star)**2)
        best_indices = (loglik==loglik.max()).nonzero()
        best_ewp = ecc[best_indices[0]][0], om[best_indices[1]][0], rhostar[best_indices[2]][0]
        best_g = g2[best_indices[0],best_indices[1],best_indices[2]][0]

        if nsamp>0:
            samp_ecc, samp_om, samp_rhostar = np.array(tools.sample_3dcdf(np.exp(loglik), ecc, om, rhostar, nsamp=nsamp))

    tlines = ['', 'Max-Likelihood Outputs:','']
    tlines.append("eccentricity is: %1.2f" % best_ewp[0])
    tlines.append("omega is:        %1.2f " % best_ewp[1])
    tlines.append("g(e, omega) is:  %1.2f" % best_g)
    tlines.append("stellar density: %1.2f" % best_ewp[2])
    tlines.append("")

        
    if nsamp>0:
        nsamp1 = samp_ecc.size
        samp_g = (1. + samp_ecc * np.sin(samp_om)) / (1. - samp_ecc**2)**0.5
        lohi = np.round(nsamp1*np.array([.1587, .8413])).astype(int)
        sigma_ecc = np.diff(np.sort(samp_ecc)[lohi])
        sigma_om = np.diff(np.sort(samp_om)[lohi])
        sigma_g = np.diff(np.sort(samp_g)[lohi])
        sigma_rhostar = np.diff(np.sort(samp_rhostar)[lohi])
        vals = np.concatenate((best_ewp[0:2], (best_g, best_ewp[-1]))).ravel()
        limits = np.sort(np.vstack((samp_ecc, samp_om, samp_g, samp_rhostar)), axis=1)[:,lohi]
        lowers = vals - limits[:,0]
        uppers = limits[:,1] - vals

        for kk in range(4):
            tlines[kk+3] += '$,^{+%1.2f}_{-%1.2f}$' % (uppers[kk], lowers[kk])

        if plotfig:
            labs = ['$e$', '$\omega$', '$g(e, \omega)$', '$\\rho_{star}$']
            fig=corner.corner(np.vstack((samp_ecc, samp_om, samp_g, samp_rhostar)).T, plot_datapoints=False, labels=labs)
            for child in fig.get_children():
                if hasattr(child, 'get_xaxis'):
                    child.get_xaxis().get_label().set_fontsize(16)
                    child.get_yaxis().get_label().set_fontsize(16)

    else:
        samp_ecc, samp_om, samp_g, samp_rhostar = (nsamp, nsamp, nsamp, nsamp)
        cmap = py.cm.cubehelix
        cmap_r = py.cm.cubehelix_r
        linecol = 'k'
        linewid = 2

        prob = np.exp(loglik)
        conflevels = [[an.confmap(prob.sum(ii), val) for val in [.6827,.9545,.9973]] for ii in range(3)]

        if plotfig:
            fig = py.figure()
            ax_ecc = py.subplot(3,3,1)
            py.plot(ecc, prob.sum(2).sum(1), color=linecol, linewidth=linewid)
            ax_eccom = py.subplot(3,3,4)
            py.contourf(ecc, om, prob.sum(2).T, cmap=cmap)
            py.contour(ecc, om, prob.sum(2).T, conflevels[2], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_om = py.subplot(3,3,5)
            py.plot(om, prob.sum(2).sum(0), color=linecol, linewidth=linewid)
            ax_eccrho = py.subplot(3,3,7)
            py.contourf(ecc, rhostar, prob.sum(1).T, cmap=cmap)
            py.contour(ecc, rhostar, prob.sum(1).T, conflevels[1], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_omrho = py.subplot(3,3,8)
            py.contourf(om, rhostar, prob.sum(0).T, cmap=cmap)
            py.contour(om, rhostar, prob.sum(0).T, conflevels[0], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_rho = py.subplot(3,3,9)
            py.plot(rhostar, prob.sum(1).sum(0), color=linecol, linewidth=linewid)

            rho_cumsum  = np.cumsum(prob.sum(1).sum(0)/prob.sum())
            rho_5sighi = py.find(rho_cumsum>0.9999994)
            rho_5siglo = py.find(rho_cumsum<(1.-0.9999994))
            if rho_5sighi.size>0:
                rho_upper = rhostar[rho_5sighi[0]]
            else:
                rho_upper = rhostar.max()
            if rho_5siglo.size>0:
                rho_lower = rhostar[rho_5siglo[-1]]
            else:
                rho_lower = rhostar.min()


            omlim = 0, 2*np.pi
            [ax.set_xlim(omlim) for ax in [ax_om, ax_omrho]]
            ax_eccom.set_ylim(omlim)
            [ax.set_ylim([rho_lower, rho_upper]) for ax in [ax_eccrho, ax_omrho]]
            ax_rho.set_xlim([rho_lower, rho_upper])

            [ax.set_xticklabels([]) for ax in [ax_ecc, ax_eccom, ax_om]]
            [ax.set_yticklabels([]) for ax in [ax_ecc, ax_om, ax_omrho, ax_rho]]
            [[tick.set_rotation(45) for tick in ax.get_xaxis().get_ticklabels()] \
                 for ax in [ax_eccrho, ax_omrho, ax_rho]]
            [[tick.set_rotation(45) for tick in ax.get_yaxis().get_ticklabels()] \
                 for ax in [ax_eccrho, ax_eccom, ax_ecc]]
            ax_rho.set_xlabel('$\\rho_*$')
            ax_omrho.set_xlabel('$\omega$')
            ax_eccrho.set_xlabel('$e$')
            ax_eccrho.set_ylabel('$\\rho_*$')
            ax_eccom.set_ylabel('$\omega$')


    if plotfig:
        ax = fig.add_axes([.4, .8, .57, .16])
        tools.textfig(inplines, ax=ax, fig=fig, fontsize=12)
        ax = fig.add_axes([.57, .57, .4, .2])
        tools.textfig(tlines, ax=ax, fig=fig, fontsize=14)


        


    if verbose:
        for line in inplines: print line
        for line in tlines: print line

    if retvals:
        ret = samp_ecc, samp_om, samp_g, samp_rhostar
    else:
        ret = None

    return ret

def modeltransit_batman(params, time, limb_dark, NP=1, svs=None, numint=1, ninterval=0.):
    """
    :INPUTS:
      params -- (8 + NP + NL + NS)-sequence with the following:
        Tc, the time of conjunction for each individual transit,

        P, the orbital period (in units of "t")

        i, the orbital inclination (in degrees; 90 is edge-on)

        a/R*, the semimajor axis-to-stellar radius ratio,

        Rp/R*, planet-to-star radius ratio, 

        ecc, the orbital eccentricity,

        omega, the longitude of periastron (in degrees),

        log10(dilution) -- the "second-light" dilution present. If
           none, set to highly negative number (e.g., -16); if 0.0, a
           diluting star of equal brightness to the primary; etc.

        the NP polynomial coefficients to normalize the data.

          EITHER:
            F0 -- stellar flux _ONLY_ (set NP=1)

          OR:
            [p_1, p_2, ..., p_(NP)] -- coefficients for polyval, to be
            used as: numpy.polyval([p_1, ...], t)

        the limb-darkening parameters (cf. Claret+2011): set limb_dark to:
          uniform, linear, quadratic, square-root, logarithmic, nonlinear...
          (uniform/0, linear/1, quadratic/2, sqrt/-2, nonlinear/4)
          

        multiplicative factors for the NS state vectors (passed in as 'svs')

      time -- numpy array.  Time of observations.  

      svs : None or list of 1D NumPy Arrays
        State vectors, applied with coefficients as defined above. To
        avoid degeneracies with the NP polynomial terms (especially
        the constant offset term), it is preferable that the state
        vectors are all mean-subtracted.

    numint : int, >= 1.
      Number of numerical integrations. Long exposure times can be
      split up into NUMINT points.

    ninterval : float, >= 0
      Time interval for integrations.  If numint>1, each point in
      the numerical integration occupies a total time interval of
      NINTERVAL seconds.

    """
    # 2015-10-29 16:19 IJMC: Created from my previous JKTEBOP version.
    # 2015-11-05 17:06 IJMC: Added second-light constraints
    # 2015-11-18 13:08 IJMC: Moved into `transit.py`
    # 2015-11-19 10:02 IJMC: dilution is now entered as log10
    import batman

    nbasic = 8
    batParams = batman.TransitParams()
    batParams.t0  = params[0]       #time of inferior conjunction
    batParams.per = params[1]       #orbital period
    batParams.inc = params[2]       #orbital inclination (in degrees)
    batParams.a   = params[3]       #semi-major axis (units of stellar radii)
    batParams.rp  = params[4]       #planet radius (in units of stellar radii)
    batParams.ecc = params[5]       #eccentricity
    batParams.w   = params[6]       #longitude of periastron (in degrees)
    secondLight   = 10**params[7]       # diluting "second light"
    # Set up polynomial terms:
    if NP>0:
        poly_params = params[nbasic:nbasic+NP]
    else:
        poly_params = [1]

    
    limb_dark, NL = get_ldtype(limb_dark)

    pNL = np.abs(NL)
    batParams.u = params[nbasic+NP:nbasic+NP+pNL]
    batParams.limb_dark = limb_dark     #limb darkening model
    

    # Set up State Vectors:
    if svs is None:
        nsvs = 0
    else:
        if isinstance(svs, np.ndarray) and svs.ndim==1:
            svs = svs.reshape(1, svs.size)
        nsvs = len(svs)
    nparam = len(params) - nsvs

    try:
        m = batman.TransitModel(batParams, time, supersample_factor=numint, exp_time=ninterval/86400.)
        lightCurve = m.light_curve(batParams)
    except:
        lightCurve = -np.ones(time.shape)

    model = (lightCurve + secondLight) / (1. + secondLight)
    model *= np.polyval(poly_params, time)
    if not np.isfinite(model).all():
        print "Model light curve contains NaN -- something is wrong with your "
        print "  BATMAN model inputs. Returning a model that is -1 everywhere!"
        model = -np.ones(time.shape)


    for ii in xrange(nsvs): 
        model += params[-ii-1] * svs[-ii-1]

        

    return model

def get_ldtype(limb_dark):
    """Convert input into types of limb darkening.  Input a string or
    an integer. Valid values are:

    uniform / 0
    linear / 1
    quadratic / 2
    square-root / -2
    nonlinear / 4

    Returns (string, int)
    """
    # 2015-11-18 14:20 IJMC: Created
    # 2016-06-16 14:45 IJMC: Added NL==4 case

    # Set up limb-darkening terms:
    if limb_dark=='uniform' or limb_dark==0:
        NL = 0
        limb_dark = 'uniform'
    elif limb_dark=='linear' or limb_dark==1:
        NL = 1
        limb_dark = 'linear'
    elif limb_dark=='quadratic' or limb_dark=='square-root' or \
            limb_dark=='logarithmic' or limb_dark=='exponential':
        NL = 2
    elif limb_dark==2:
        NL = 2
        limb_dark = 'quadratic'
    elif limb_dark==-2:
        NL = -2
        limb_dark = 'square-root'
    elif limb_dark=='nonlinear' or limb_dark==4:
        NL = 4
        limb_dark = 'nonlinear'
    else:
        print "Invalid limb-darkening type (you set limb_dark=%s)" % limb_dark
        NL = -1
        limb_dark = 'invalid'

    return limb_dark, NL

    
def quickEclipseLimit(time, data, t0, per, t14, ntrials=100, transitModel=None):
    """Quick 'n' Dirty computation of 3-sigma upper limit to eclipse depth.
    """
    # 2015-12-07 17:00 IJMC: Created
    from analysis import lsq, dumbconf

    phase = ((time - t0) % per ) 
    oot = (phase > (1.5*t14)) * (phase < (per-1.5*t14))
    if transitModel is None: transitModel = np.ones(data.size)*np.median(data[oot])
    valid = oot * ((time - time.min()) > (t14/2.)) * ((time.max() - time) > (t14/2.)) * \
            (np.abs(data / transitModel - 1.) <= (5 * dumbconf(data/transitModel, .683)[0]))
    nvalid  = valid.sum()
    baseline = np.ones(nvalid, float)
    vdata = (data/transitModel)[valid]

    t0trials = np.linspace(phase.min(), phase.max(), ntrials)
    edepths = np.zeros(ntrials, float)

    for ii in xrange(ntrials):
        model = np.zeros(nvalid, float)
        model[(np.abs(time[valid] - t0trials[ii]) % per) <= t14/2.] = -1.
        out, junk = lsq(np.vstack((baseline, model)).T, vdata, checkvals=False)
        edepths[ii] = out[1]

    return dumbconf(edepths, .00135, 'lower')[0]

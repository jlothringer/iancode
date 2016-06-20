"""Functions for fitting phase curves.

Use of 'errfunc' is encouraged.

:REQUIREMENTS:
  :doc:`transit`
"""
# 2009-12-15 13:26 IJC: Created
# 2011-04-21 16:13 IJMC: Added tsin function and nsin function
# 2013-04-30 16:01 IJMC: Added simplest possible C-functionality w/chi2.

#try:
#    import psyco
#    psyco.full()
#except ImportError:
#    print 'Psyco not installed, the program will just run slower'



from numpy import *
import numpy as np
import transit
import pdb

try:
    import _chi2
    c_chisq = True
except:
    c_chisq = False

def tsin(param, x):
    """Compute phase function with unknown period, assuming a sinusoid:

    p(x) = param[0] - param[1] * cos(2*pi*x/param[2] + param[3])
    """
    return param[0]- abs(param[1]) *cos(2*pi*x/param[2] +param[3])

def nsin(param, x):
    """Compute phase function with unknown period, assuming n sinusoids:

    p(x) = param[0] - \
           param[i+1] * cos(2*pi*x/param[i+2] + param[i+3]) - \
           param[i+3+1] * cos(2*pi*x/param[i+3+2] + param[i+3+3]) - ...
    """

    nparam = len(param)
    nsinusoids = (nparam - 1) / 3
    ret = param[0]
    for ii in range(nsinusoids):
        ret -= abs(param[1+ii*3]) *cos(2*pi*x/param[2+ii*3] +param[3+ii*3])

    return ret

def phasesin(param, x):
    """compute phase function with a fixed period=1, assuming a sinusoid:

    p(x) = param[0] - param[1] * cos(2*pi*x + param[2])
    """
    return param[0]- abs(param[1]) *cos(2*pi*x +param[2])

def phasesin2(param, phaseoffset, x):
    """compute phase function with a fixed period=1 and phase offset (in radians):

    p(x) = param[0] - param[1] * cos(2*pi*x + phaseoffset)
    """
    return param[0]- abs(param[1]) *cos(2*pi*x +phaseoffset)

def phase2sin(param, x, absamp=True):
    """compute double-sinusoid with a fixed period=1 offset (in radians):

    p(x) = param[0] - param[1] * cos(2*pi*x + param[2]) - \
                      param[3] * cos(4*pi*x + param[4])

    ampsamp: if True, take the absolute value of param[1] & param[3]
    """
    # 2013-02-09 18:48 IJMC: Fixed a typo.

    if absamp:
        ret = param[0]- np.abs(param[1]) *cos(2*pi*x +param[2]) - \
            np.abs(param[3]) *cos(4*pi*x +param[4]) 
    else:
        ret = param[0]- param[1] *cos(2*pi*x +param[2]) - \
            param[3] *cos(4*pi*x +param[4]) 

    return  ret

def phasesin14(param, x):
    """compute phase function with a fixed period=1, assuming a
    sinusoid, and account for 14 different possible flux offsets.

    Input data 'x' must therefore be of size (14xN); if not, it will
    be reshaped into that.

    p[i,j] = (1. + param[3+j]) * (param[0] - param[1]*cos(2*pi*x[i,j]+param[2]))

    [note that the first c-parameter (individual offset) will be
     constrained such that: prod(1. + param[3::]) = 1.]
    """
    # 2010-04-27 11:49 IJC: Created.
    # 2011-06-09 16:40 IJMC: Definition slightly changed to be a
    #                        multiplicative factor.
    cparam = array(param[3::], copy=True)
    cparam[0] = 1. / prod(1. + cparam[1::]) - 1.

    param[2] = param[2] % (2*pi)

    if len(x.shape)==1:
        was1d = True
        x = x.reshape(14, len(x)/14.)
    else:
        was1d = False

    ret =  param[0] - abs(param[1]) *cos(2*pi*x +param[2])
    #print 'param[3::]>>',param[3::]
    #print 'x.shape>>',x.shape
    ret *= (1. + cparam.reshape(14,1))

    if was1d:
        ret = ret.ravel()

    return ret

def phasepoly14(param, x):
    """compute phase function, assuming a polynomial, and account for
    14 different possible flux offsets.

    Input data 'x' must be in units of orbital phase, and must be of
    size (14xN); if not, it will be reshaped into that.

    For an order-N polynomial:
       p[i,j] = (1. + param[N+j]) * (numpy.polyval(param[0:N], x))

    [note that the first c-parameter (individual offset) will be
     constrained such that: prod(1. + param[N::]) = 1.]
    """
    # 2011-09-26 10:42 IJMC: Created from phaselamb14
    N = len(param) - 14
    cparam = array(param[N::], copy=True)
    cparam[0] = 1. / prod(1. + cparam[1::]) - 1.

    if len(x.shape)==1:
        was1d = True
        x = x.reshape(14, len(x)/14.)
    else:
        was1d = False

    ret =  polyval(param[0:N], x)
    ret *= (1. + cparam.reshape(14,1))

    if was1d:
        ret = ret.ravel()

    return ret

def phaselamb14(param, x):
    """compute phase function with a fixed period=1, assuming a
    sinusoid, and account for 14 different possible flux offsets.

    Input data 'x' must therefore be of size (14xN); if not, it will
    be reshaped into that.

    p[i,j] = (1. + param[3+j]) * (param[0] + param[1]*lambertian(2*pi*x[i,j]+param[2]))

    [note that the first c-parameter (individual offset) will be
     constrained such that: prod(1. + param[3::]) = 1.]
    """
    # 2011-09-25 22:24 IJMC: Created from phasesin14
    cparam = array(param[3::], copy=True)
    cparam[0] = 1. / prod(1. + cparam[1::]) - 1.

    param[2] = param[2] % (2*pi)

    if len(x.shape)==1:
        was1d = True
        x = x.reshape(14, len(x)/14.)
    else:
        was1d = False

    ret =  param[0] + abs(param[1]) *lambertian(2*pi*x +param[2])
    #print 'param[3::]>>',param[3::]
    #print 'x.shape>>',x.shape
    ret *= (1. + cparam.reshape(14,1))

    if was1d:
        ret = ret.ravel()

    return ret

def phasesinsin14(param, x):
    """compute phase function with a fixed period=1, assuming a
    sinusoid and first harmonic, and account for 14 different possible
    flux offsets.

    Input data 'x' must therefore be of size (14xN); if not, it will
    be reshaped into that.

    p[i,j] = (1. + param[5+j]) * \
             [  param[0] - param[1]*cos(2*pi*x[i,j]+param[2]) + \
                param[3]*cos(4*pi*x[i,j]+param[4])  ]

    [note that the first c-parameter (individual offset) will be
     constrained such that: prod(1. + param[5::]) = 1.]

    :NOTE: 
      The magnitude of the amplitudes will always be taken; they
      cannot be negative.
    """
    # 2011-09-16 09:14 IJMC: Created from phasesin14
    cparam = array(param[5::], copy=True)
    cparam[0] = 1. / prod(1. + cparam[1::]) - 1.

    param[2] = param[2] % (2*pi)
    param[4] = param[4] % (2*pi)

    if len(x.shape)==1:
        was1d = True
        x = x.reshape(14, len(x)/14.)
    else:
        was1d = False

    ret =  param[0] - abs(param[1]) *cos(2*pi*x +param[2]) + \
        abs(param[3]) *cos(4*pi*x +param[4])
    #print 'param[3::]>>',param[3::]
    #print 'x.shape>>',x.shape
    ret *= (1. + cparam.reshape(14,1))

    if was1d:
        ret = ret.ravel()

    return ret

def phasesinsin14_2(param, x):
    """compute phase function with a fixed period=1, assuming a
    sinusoid and first harmonic, and account for 14 different possible
    flux offsets.

    Input data 'x' must therefore be of size (14xN); if not, it will
    be reshaped into that.

    p[i,j] = (1. + param[5+j]) * \
             [  param[0] - param[1]*cos(2*pi*x[i,j]) - param[2]*sin(2*pi*x[i,j]) -
                param[3]*cos(4*pi*x[i,j]) - param[4]*sin(4*pi*x[i,j])  ]

    [note that the first c-parameter (individual offset) will be
     constrained such that: prod(1. + param[5::]) = 1.]
    """
    # 2011-09-16 09:14 IJMC: Created from phasesinsin14
    cparam = array(param[5::], copy=True)
    cparam[0] = 1. / prod(1. + cparam[1::]) - 1.

    if len(x.shape)==1:
        was1d = True
        x = x.reshape(14, len(x)/14.)
    else:
        was1d = False

    ret =  param[0] - param[1]*cos(2*pi*x) - param[2]*sin(2*pi*x) - \
        param[3]*cos(4*pi*x) - param[4]*sin(4*pi*x)
    #print 'param[3::]>>',param[3::]
    #print 'x.shape>>',x.shape
    ret *= (1. + cparam.reshape(14,1))

    if was1d:
        ret = ret.ravel()

    return ret


def phasesin14xymult(param, xyord,crossord,t, x, y):
    """compute phase function with a fixed period=1, assuming a
    sinusoid, and account for 14 different possible flux offsets and
    X/Y positional motions.

    Input data 't','x','y' must therefore be of size (14xN); if not,
    it will be reshaped into that.

    "xyord" determines the linear order of the polynomial in x and y.
    If xyord==1, then:
    f[i,j] = p[0] - p[1]*cos(2*pi*x[i,j]+p[2]) + p[3+i] +p[17+i]*x + p[31+i]*y

    If xyord==2, then:
    f[i,j] = p[0] + p[3+i] +p[17+i]*x + p[31+i]*y +p[45+i]*x**2 + p[59+i]*y**2
                  - p[1]*cos(2*pi*x[i,j]+p[2])

    If crossord==1, then the cross-terms (x*y) will be included using
         14 coefficients. crossord>1 still just uses the single
         cross-terms; I haven't generalized it yet to higher orders.

    [note that the the individual offsets will be subjected to the
    constraint: param[3::] -= param[3::].mean() ]
    """
    # 2010-04-27 11:49 IJC: Created
    # 2010-05-28 15:42 IJC: Added x*y cross-terms

    param = array(param,copy=True)
    x = array(x,copy=True)
    y = array(y,copy=True)
    t = array(t,copy=True)

    xparam = zeros((0,14),float)
    yparam = zeros((0,14),float)
    crossparam = zeros((0,14),float)

    cparam = param[3:17]
    if xyord>=1:
        for ii in range(xyord):
            xparam = vstack((xparam,param[17+ii*28:31+ii*28]))
            yparam = vstack((yparam,param[31+ii*28:45+ii*28]))

    lastxyparamind = 45+(xyord-1)*28
    if crossord>=1:
        for ii in [0]: #range(crossparam):
            crossparam = vstack((crossparam,param[lastxyparamind:lastxyparamind+(ii+1)*14]))

    #cparam -= mean(cparam)
    param[2] = param[2] % (2*pi)
    
    if len(t.shape)==1:
        was1d = True
        t = t.reshape(14, len(t)/14.)
        x = x.reshape(14, len(x)/14.)
        y = y.reshape(14, len(y)/14.)
    else:
        was1d = False

    # Subtract the mean from the X and Y data
    x -= x.mean(1).reshape(14,1)
    y -= y.mean(1).reshape(14,1)

    # Zeroth-order model:
    ret =  param[0] - abs(param[1]) *cos(2*pi*t +param[2])

    # Apply constant offsets:
    ret *= (1. + tile(cparam, (t.shape[1],1)).transpose())
    if xyord>=1:
        for ii in range(xyord):
            ret *= (1. + tile(xparam[ii], (t.shape[1],1)).transpose()*x**(ii+1))
            ret *= (1. + tile(yparam[ii], (t.shape[1],1)).transpose()*y**(ii+1))

    if crossord>=1:
        for ii in [0]: 
            ret *= (1. + tile(crossparam[ii], (t.shape[1],1)).transpose()*x*y)

    if was1d:
        ret = ret.ravel()

    return ret

def subfit_kw(params, input, i0, i1):
    """Parse keyword inputs (for :func:`errfunc`, etc.) for multiple
    concatenated inputs (e.g., with input key 'npars' set).

    :INPUTS:
      params : 1D NumPy array of parameters

      input : dict of keywords

      i0 : int, first index of current parameters

      i1 : int, last index of current parameters (e.g.: params[i0:i1])
    """
    # 2013-04-19 16:18 IJMC: Created
    # 2013-04-20 17:54 IJMC: Fixed a small bug in the 'ngaussprior' check.
    # 2013-04-30 20:46 IJMC: Now accept 'wrapped_joint_params' keyword

    i0 =int(i0)
    i1 = int(i1)
    valid_indices = range(i0, i1)
    sub_input = input.copy()

    if 'wrapped_joint_params' in sub_input:
        wrap_indices = sub_input.pop('wrapped_joint_params')
        params = unwrap_joint_params(params, wrap_indices)
        # Handle various fitting keywords appropriately:
        if 'jointpars' in sub_input: junk = sub_input.pop('jointpars')
        #for key in ['uniformprior', 'gaussprior']:
        #    if key in sub_input and sub_input[key] is not None:
        #        sub_input[key] = unwrap_joint_params(sub_input[key], wrap_indices)
        #if 'ngaussprior' in sub_input and sub_input['ngaussprior'] is not None:
            

    sub_params = params[i0:i1]

    # Now check and clean all the possible sub-keywords:
    if 'npars' in sub_input and sub_input['npars'] is not None:
        junk = sub_input.pop('npars')

    if 'uniformprior' in sub_input and sub_input['uniformprior'] is not None:
        sub_input['uniformprior'] = sub_input['uniformprior'][i0:i1]

    if 'gaussprior' in sub_input and sub_input['gaussprior'] is not None:
        sub_input['gaussprior'] = sub_input['gaussprior'][i0:i1]

    if 'jointpars' in sub_input and sub_input['jointpars'] is not None:
        new_jointpars = []
        for these_jointpars in sub_input['jointpars']:
            if (these_jointpars[0] in valid_indices and \
                    these_jointpars[1] in valid_indices):
                # Account for the fact that, in a sub-fit, the index value
                #    is different from the index of the ensemble fit:
                new_jointpars.append((these_jointpars[0]-i0, these_jointpars[1]-i0))

        if new_jointpars==[]:
            junk = sub_input.pop('jointpars')
        else:
            sub_input['jointpars'] = new_jointpars

    if 'ngaussprior' in sub_input and sub_input['ngaussprior'] is not None:
        new_ngaussprior = []
        for this_ngp in sub_input['ngaussprior']:
            all_indices_valid = True
            for ngp_index in this_ngp[0]:
                all_indices_valid = all_indices_valid and ngp_index in valid_indices
            # Account for the fact that, in a sub-fit, the index value
            #    is different from the index of the ensemble fit:
            if all_indices_valid:
                new_ngaussprior.append([this_ngp[0]-i0, \
                                            this_ngp[1], this_ngp[2]])

        if new_ngaussprior==[]:
            junk = sub_input.pop('ngaussprior')
        else:
            sub_input['ngaussprior'] = new_ngaussprior
                
    return sub_params, sub_input

def eclipse14_single(param, tparam, t, xyord=None, x=None, y=None):
    """compute 3-parameter eclipse function of a single event, and
    account for 14 different possible flux offsets and X/Y positional
    motions.

    param : parameters to be fit
      [Fstar, Fplanet, t_center, c0, c1, ... c13]

    tparam : parameters to be held constant (from transit)
      [b, v (in Rstar/day), p (Rp/Rs)]
    
    Input data 't','x','y' must therefore be of size (14xN); if not,
    it will be reshaped into that.

    If xyord==None or 0:
      f[i, j] = (1. + p[3+i]) * [eclipse light curve]

    "xyord" determines the linear order of the polynomial in x and y.
    If xyord==1, then:
    f[i,j] = (...) * (1.+ p[3+i] +p[17+i]*x + p[31+i]*y)

    [note that the the individual offsets will be subjected to the
    constraint: param[3] = 1./(1.+param[4:17]).prod() - 1. ]
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work.

    param = array(param,copy=True)
    tparam = array(tparam,copy=True)
    x = array(x,copy=True)
    y = array(y,copy=True)
    t = array(t,copy=True)
    if xyord >= 1:
        xparam = zeros((0,14),float)
        yparam = zeros((0,14),float)


    # Set up 14 sensitivity perturbations:
    cparam = param[3:17].copy()
    # Ensure that prod(1.+cparam) equals unity:
    cparam[0] = 1./(1.+cparam[1::]).prod() - 1.
    if xyord>=1:
        for ii in range(xyord):
            xparam = vstack((xparam,param[17+ii*28:31+ii*28]))
            yparam = vstack((yparam,param[31+ii*28:45+ii*28]))

    if len(t.shape)==1:
        was1d = True
        t = t.reshape(14, len(t)/14.)
        if xyord >= 1:
            x = x.reshape(14, len(x)/14.)
            y = y.reshape(14, len(y)/14.)
    else:
        was1d = False

    # Set up eclipse light curve:
    b, v, p = tparam[0:3]
    z = sqrt(b**2 + (v * (t - param[2]))**2)
    tr = param[0] - param[1] * transit.occultuniform(z, p)/p**2
    


    # Subtract the mean from the X and Y data
    if xyord >= 1:
        x -= x.mean(1).reshape(14,1)
        y -= y.mean(1).reshape(14,1)

    # Apply constant and X/Y offsets:
    #ret *= (1. + tile(cparam, (t.shape[1],1)).transpose())
    offset_term = (1. + tile(cparam, (t.shape[1],1)).transpose())
    if xyord>=1:
        for ii in range(xyord):
            offset_term += tile(xparam[ii], (t.shape[1],1)).transpose()*x**(ii+1)
            offset_term += tile(yparam[ii], (t.shape[1],1)).transpose()*y**(ii+1)

    # Apply the (1+c+dx+ey) term:
    tr *= offset_term

    if was1d:
        tr = tr.ravel()

    return tr




def eclipse_single(param, tparam, t):
    """compute 3-parameter eclipse function of a single event

    param : 3 parameters to be fit
      [Fstar, Fplanet, t_center]

    tparam : 3 parameters to be held constant (from transit)
      [b, v (in Rstar/day), p (Rp/Rs)]
    
    Input data 't' must be of size (14xN); if not, it will be reshaped
       into that.
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work.

    param = array(param,copy=True)
    tparam = array(tparam,copy=True)
    t = array(t,copy=True)


    if len(t.shape)==1:
        was1d = True
        t = t.reshape(14, len(t)/14.)
    else:
        was1d = False

    # Set up eclipse light curve:
    b, v, p = tparam[0:3]
    z = sqrt(b**2 + (v * (t - param[2]))**2)
    tr = param[0] - param[1] * transit.occultuniform(z, p)/p**2

    if was1d:
        tr = tr.ravel()

    return tr

def transit_single(param, t):
    """compute 6+L-parameter eclipse function of a single event

velocity, impact parameter, stellar flux,
planet/star radius ratio, time of center transit, period, limb darkening

    param : parameters to be fit: 
      [Fstar, t_center, b, v (in Rstar/day), p (Rp/Rs), per (days)]
    
      Up to two additional parameter can be concatenated onto the end,
      respectively representing linear and quadratic limb-darkening.

    Input data 't' must be of size (14xN); if not, it will be reshaped
       into that.
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work.
    # 2011-05-16 10:43 IJC: Adding period as a free parameter

    param = array(param,copy=True)
    t = array(t,copy=True)


    if len(t.shape)==1:
        was1d = True
        t = t.reshape(14, len(t)/14.)
    else:
        was1d = False

    # Set up eclipse light curve:
    z = sqrt(param[2]**2 + (param[3] * (((t - param[1] - param[5]*.5) % param[5]) - param[5]*.5) )**2)
    if param.size==6:
        tr = param[0] *(1. - transit.occultuniform(z, param[4]))
    elif param.size==7:  # Linear limb-darkening:
        tr = param[0] * transit.occultquad(z, param[4], [param[6], 0.])
    elif param.size>=8:  # Linear limb-darkening:
        tr = param[0] * transit.occultquad(z, param[4], [param[6], param[7]])

    if was1d:
        tr = tr.ravel()

    return tr

def mcmc_eclipse_single(z, t, sigma, params, tparam, stepsize, numit, nstep=1):
    """MCMC for 3-parameter eclipse function of a single event


    :INPUTS:
        z : 1D array
                    Contains dependent data

        t : 1D array
                    Contains independent data: phase, x- and y-positions

        sigma : 1D array
                    Contains standard deviation of dependent (z) data

        params : 3 parameters to be fit
          [Fstar, Fplanet, t_center]

        tparam : 3 parameters to be held constant (from transit)
          [b, v (in Rstar/day), p (Rp/Rs)]

        stepsize :  1D array
                Array of 1-sigma change in parameter per iteration

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step
    
    :REFERENCE:
        Numerical Recipes, 3rd Edition (Section 15.8); Wikipedia

    
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work, and from K. Stevenson's MCMC
    #                        example implementation.

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel     = eclipse_single(params, tparam, t)

    currchisq  = (((zmodel - z)/sigma)**2).ravel().sum()

    bestchisq  = currchisq
#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            nextp    = np.random.normal(params,stepsize)
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = eclipse_single(nextp, tparam, t)

            nextchisq  = (((zmodel - z)/sigma)**2).ravel().sum() 

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

def mcmc_eclipse14_single(z, t, x, y, sigma, params, tparam, stepsize, numit, nstep=1, xyord=None):
    """MCMC for 17-parameter eclipse function of a single event

    :INPUTS:
        z : 1D array
                    Contains dependent data

        t,x,y : 1D array
                    Contains independent data: phase, x- and y-positions

        sigma : 1D array
                    Contains standard deviation of dependent (z) data

        params : 17 parameters to be fit
          [Fstar, Fplanet, t_center, c0, ... , c13]

        tparam : 3 parameters to be held constant (from transit)
          [b, v (in Rstar/day), p (Rp/Rs)]

        stepsize :  1D array
                Array of 1-sigma change in parameter per iteration

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        xyord : int
                Highest order in x/y motions

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
        Numerical Recipes, 3rd Edition (Section 15.8); Wikipedia

    
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work, and from K. Stevenson's MCMC
    #                        example implementation.

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel     = eclipse14_single(params, tparam, t, xyord=xyord, x=x, y=y)

    currchisq  = (((zmodel - z)/sigma)**2).ravel().sum()

    bestchisq  = currchisq
#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            nextp    = np.random.normal(params,stepsize)
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = eclipse14_single(nextp, tparam, t, xyord=xyord, x=x, y=y)

            nextchisq  = (((zmodel - z)/sigma)**2).ravel().sum() 

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


def phasesin14xymult_cfix(param, xyord,crossord,t, x, y):
    """compute phase function with a fixed period=1, assuming a
    sinusoid, and account for 14 different possible flux offsets and
    X/Y positional motions.

    Input data 't','x','y' must therefore be of size (14xN); if not,
    it will be reshaped into that.

    "xyord" determines the linear order of the polynomial in x and y.
    If xyord==1, then:
    f[i,j] = (p[0] - p[1]*cos(2*pi*x[i,j]+p[2])) * (1.+ p[3+i] +p[17+i]*x + p[31+i]*y)

    If xyord==2, then:
    f[i,j] = (p[0] - p[1]*cos(2*pi*x[i,j]+p[2])) * (1.+ p[3+i] +p[17+i]*x + p[31+i]*y)

    f[i,j] = (p[0] - p[1]*cos(2*pi*x[i,j]+p[2])) * \
                (1.+ p[3+i] +p[17+i]*x + p[31+i]*y +p[45+i]*x**2 + p[59+i]*y**2

    If crossord==1, then the cross-terms (x*y) will be included using
         14 coefficients. crossord>1 still just uses the single
         cross-terms; I haven't generalized it yet to higher orders.

    [note that the the individual offsets will be subjected to the
    constraint: param[3] = 1./(1.+param[4:17]).prod() - 1. ]
    """
    # 2010-04-27 11:49 IJC: Created
    # 2010-05-28 15:42 IJC: Added x*y cross-terms
    # 2010-07-21 13:02 IJC: switched to a mostly-additive model

    param = array(param,copy=True)
    x = array(x,copy=True)
    y = array(y,copy=True)
    t = array(t,copy=True)

    xparam = zeros((0,14),float)
    yparam = zeros((0,14),float)
    crossparam = zeros((0,14),float)

    cparam = param[3:17].copy()
    # Ensure that prod(1.+cparam) equals zero
    cparam[0] = 1./(1.+cparam[1::]).prod() - 1.
    if xyord>=1:
        for ii in range(xyord):
            xparam = vstack((xparam,param[17+ii*28:31+ii*28]))
            yparam = vstack((yparam,param[31+ii*28:45+ii*28]))

    lastxyparamind = 45+(xyord-1)*28
    if crossord>=1:
        for ii in [0]: #range(crossparam):
            crossparam = vstack((crossparam,param[lastxyparamind:lastxyparamind+(ii+1)*14]))

    #cparam -= mean(cparam)
    param[2] = param[2] % (2*pi)
    
    if len(t.shape)==1:
        was1d = True
        t = t.reshape(14, len(t)/14.)
        x = x.reshape(14, len(x)/14.)
        y = y.reshape(14, len(y)/14.)
    else:
        was1d = False

    # Subtract the mean from the X and Y data
    x -= x.mean(1).reshape(14,1)
    y -= y.mean(1).reshape(14,1)

    # Zeroth-order model:
    ret =  param[0] - abs(param[1]) *cos(2*pi*t +param[2])

    # Apply constant and X/Y offsets:
    #ret *= (1. + tile(cparam, (t.shape[1],1)).transpose())
    offset_term = (1. + tile(cparam, (t.shape[1],1)).transpose())
    if xyord>=1:
        for ii in range(xyord):
            offset_term += tile(xparam[ii], (t.shape[1],1)).transpose()*x**(ii+1)
            offset_term += tile(yparam[ii], (t.shape[1],1)).transpose()*y**(ii+1)

    if crossord>=1:
        for ii in [0]: 
            offset_term += tile(crossparam[ii], (t.shape[1],1)).transpose()*x*y

    # Apply the (1+c+dx+ey) term:
    ret *= offset_term

    if was1d:
        ret = ret.ravel()

    return ret

def phaselinsin(param, x):
    """compute phase function with a linear drift and  period=1:

    p(x) = param[0]+param[1]*x - param[2]*cos(2*pi*x + param[3])
    """
    return param[0]+ abs(param[1])*x - param[2] *cos(2*pi*x +param[3])

def phaselinsin2(param, phaseoffset, x):
    """compute phase function with a linear drift and fixed period=1
    and fixed phase offset (in radians):

    p(x) = param[0]+param[1]*x - param[2]*cos(2*pi*x + phaseoffset)
    """
    return param[0]+ abs(param[1])*x - param[2] *cos(2*pi*x + phaseoffset)

def lam4fit(param, x):
    """Compute labertian phase function with a fixed period=1, and x
    in units of orbital cycles.

    param = [DC pedestal, AC amplitude, inclination (radians)]
    """
    pedestal, amplitude, inc = param
    ophase = x*2*pi
    return pedestal + abs(amplitude)*lambertian(ophase, inc=inc)

def lam4fit2(param, x):
    """Compute labertian phase function with a fixed period=1, and x
    in units of orbital cycles -- this time with a variable phase offset.

    param = [DC pedestal, AC amplitude, inclination (radians), phase offset (radians)]
    """
    pedestal, amplitude, inc, poff = param
    ophase = x*2*pi + poff
    return pedestal + abs(amplitude)*lambertian(ophase, inc=inc)

def lam4fit_noinc(param, inc, x):
    """Compute labertian phase function with a fixed period=1, and x
    in units of orbital cycles -- a variable phase offset but FIXED
    inclination (in radians)

    param = [DC pedestal, AC amplitude, phase offset (radians)]
    """
    pedestal, amplitude, poff = param
    ophase = x*2*pi + poff
    return pedestal + abs(amplitude)*lambertian(ophase, inc=inc)

def lambertian(ophase, inc=pi/2):
    """
    Return a lambertian phase function with peak-to-valley amplitude unity.

    INPUTS:
      ophase (seq) an orbital phase (in radians).  Secondary eclipse
           (or 'opposition' for non-transiting planets) occurs at pi;
           Primary transit (or 'conjuction') occurs at 0 or 2*pi

      inc (float) system inclination angle (also in radians).  Edge-on
           is pi/2, face-on is 0.
    """
    # 2009-12-16 10:07 IJC: Created based on Hansen 2008 (ApJS 179:484) Eq. 43
    #  and Barnes et al. 2007 (MNRAS 379:1097) Eq. 2.
    #
    # 2011-09-25 22:36 IJMC: Added test to speed up inc=pi/2 case.
    # 2011-10-11 17:28 IJMC: Removed that test.
    apparentphase = arccos(-sin(inc)*cos(ophase))

    ret = cos(apparentphase)-(apparentphase*cos(apparentphase)-sin(apparentphase))/pi
    
    return ret

def lambertian_mean(inc,n=5000):
    """Return mean of a nominally unity-amplitude lambertian with a given
    inclination angle, using function 'lambertian'.  inc is in radians."""
    # 2010-03-23 16:56 IJC: Created
    phase = linspace(0,2*pi,n)
    if hasattr(inc,'__iter__'):
        ret = zeros(len(inc),float)
        for ii in range(len(inc)):
            ret[ii] = lambertian(phase,inc=inc[ii]).mean()

    else:
        ret= lambertian(phase,inc=inc).mean()

    return ret

def lambertian_amplitude(inc,n=5000):
    """Return amplitude of a nominally unity-amplitude lambertian with a given
    inclination angle, using function 'lambertian'.  inc is in radians."""
    # 2010-03-23 16:56 IJC: Created
    phase = [0,pi]
    if hasattr(inc,'__iter__'):
        ret = zeros(len(inc),float)
        for ii in range(len(inc)):
            ret[ii] = diff(lambertian(phase,inc=inc[ii]))[0]

    else:
        ret= diff(lambertian(phase,inc=inc))[0]

    return ret

def slicemodel(param, xi):
    """Compute a slice model via Cowan & Agol (2008). 

    xi is from 0 to 2*pi"""
    # 2009-12-15 15:14 IJC: Created, but doesn't work correctly yet.
    const = param[0]
    phi0 = param[1]
    nslices = len(param)-2
    Jcoef = param[2::]
    dphi = 2*pi/nslices
    phi = phi0+arange(nslices)*dphi

#    Gmatrix0 = zeros((len(xi),nslices),float)
#    for ii in range(len(xi)):
#        for jj in range(nslices):
#            alphaplus0  = arccos(max(cos(xi[ii] + phi[jj] + dphi/2.), 0))
#            alphaminus0 = arccos(max(cos(xi[ii] + phi[jj] - dphi/2.), 0))
#            Gmatrix0[ii,jj] = sin(alphaplus0) - sin(alphaminus0)

    phi_j, xi_i = meshgrid(phi, xi)
    tempaplus  = cos(xi_i + phi_j + dphi/2.)
    tempaminus = cos(xi_i + phi_j - dphi/2.)
    tempaplus[tempaplus<0] = 0.
    tempaminus[tempaminus<0] = 0.
    alphaplus = arccos(tempaplus)
    alphaminus = arccos(tempaminus)

    Gmatrix = sin(alphaplus) - sin(alphaminus)
    flux = const + dot(Gmatrix, Jcoef)
    Gmatrix = cos(alphaplus) - cos(alphaminus)
    flux2 = const + dot(Gmatrix, Jcoef)
    
    print "doesn't seem to work quite right yet...  2009-12-15 15:14 IJC: "
    return flux


def lnprobfunc(*arg, **kw):
    """Return natural logarithm of posterior probability: i.e., -chisq/2.

    Inputs are the same as for :func:`errfunc`.

    :SEE ALSO:
      :func:`gaussianprocess.negLogLikelihood`
    """
    # 2012-03-23 18:17 IJMC: Created for use with :doc:`emcee` module.
    # 2015-11-05 17:50 IJMC: Added 'no_nans_allowed' option
    ret = -0.5 * errfunc(*arg, **kw)
    if kw.has_key('nans_allowed') and (not kw.pop('nans_allowed')) or not (np.isfinite(ret)):
        print "Whoops -- nan detected, but nans NOT ALLOWED in lnprobfunc!"
        ret = 9e99

    return ret



def errfunc14xymult_cfix(*arg,**kw):
    """Generic function to give the chi-squared error on a generic function:

    INPUTS:
       (fitparams, arg1, arg2, ... indepvar, depvar, weights)

    """
    # 2010-07-16 10:30 IJC: Created to try multi-threading w/PP
    chisq = resfunc(arg[0], phasesin14xymult_cfix, *arg[1::]).sum()

    return chisq



def devfunc(*arg, **kw):
    """Generic function to give the weighted residuals on a function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

      OR:
       
       (fitparams, function, arg1, arg2, ... , depvar, weights, kw)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.

      If the last argument is of type dict, it is assumed to be a set
      of keyword arguments: this will be added to resfunc's direct
      keyword arguments, and will then be passed to the fitting
      function **kw.  This is necessary for use with various fitting
      and sampling routines (e.g., kapteyn.kmpfit and emcee.sampler)
      which do not allow keyword arguments to be explicitly passed.
      So, we cheat!  Note that any keyword arguments passed in this
      way will overwrite keywords of the same names passed in the
      standard, Pythonic, way.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      gaussprior -- list of 2-tuples (or None values), same length as "fitparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  Here in
                   :func:`devfunc`, we _scale_ the error-weighted
                   deviates such that the resulting chi-squared will
                   increase by the desired amount.

      uniformprior -- list of 2-tuples (or 'None's), same length as "fitparams."
                   The i^th tuple (lo_i, hi_i) imposes a uniform prior
                   on the i^th parameter p_i by requiring that it lie
                   within the specified "high" and "low" limits.  We
                   do this (imprecisely) by multiplying the resulting
                   deviates by 1e9 for each parameter outside its
                   limits.

      ngaussprior -- list of 3-tuples of Numpy arrays.
                   Each tuple (j_ind, mu, cov) imposes a multinormal
                   Gaussian prior on the parameters indexed by
                   'j_ind', with mean values specified by 'mu' and
                   covariance matrix 'cov.' This is the N-dimensional
                   generalization of the 'gaussprior' option described
                   above. Here in :func:`devfunc`, we _scale_ the
                   error-weighted deviates such that the resulting
                   chi-squared will increase by the desired amount.

                   For example, if parameters 0 and 3 are to be
                   jointly constrained (w/unity means), set: 
                     jparams = np.array([0, 3])
                     mu = np.array([1, 1])
                     cov = np.array([[1, .9], [9., 1]])
                     ngaussprior=[[jparams, mu, cov]]  # Double brackets are key!


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))
    """
    # 2009-12-15 13:39 IJC: Created
    # 2010-11-23 16:25 IJMC: Added 'testfinite' flag keyword
    # 2011-06-06 10:52 IJMC: Added 'useindepvar' flag keyword
    # 2011-06-24 15:03 IJMC: Added multi-function (npars) and
    #                        jointpars support.
    # 2011-06-27 14:34 IJMC: Flag-catching for multifunc calling
    # 2012-03-23 18:32 IJMC: testfinite and useindepvar are now FALSE
    #                        by default.
    # 2012-05-01 01:04 IJMC: Adding surreptious keywords, and GAUSSIAN
    #                        PRIOR capability.
    # 2012-05-08 16:31 IJMC: Added NGAUSSIAN option.
    # 2012-10-16 09:07 IJMC: Added 'uniformprior' option.
    # 2013-02-26 11:19 IJMC: Reworked return & concatenation in 'npars' cases.
    # 2013-03-08 12:54 IJMC: Added check for chisq=0 in penalty-factor cases.
    # 2013-10-12 23:47 IJMC: Added 'jointpars1' keyword option.

    import pdb
    #pdb.set_trace()
    params = np.array(arg[0], copy=False)

    if isinstance(arg[-1], dict): 
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass


    if len(arg)==2:
        residuals = devfunc(params, *arg[1], **kw)

    else:
        if kw.has_key('testfinite'):
            testfinite =  kw['testfinite']
        else:
            testfinite = False
        if not kw.has_key('useindepvar'):
            kw['useindepvar'] = False

        # Keep fixed pairs of joint parameters:
        if kw.has_key('jointpars1'):
            jointpars1 = kw['jointpars1']
            for jointpar1 in jointpars1:
                params[jointpar1[1]] = params[jointpar1[0]]


        if kw.has_key('gaussprior') and kw['gaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_gaussprior =  kw['gaussprior']
            gaussprior = []
            for pair in temp_gaussprior:
                if pair is None:
                    gaussprior.append([0, np.inf])
                else:
                    gaussprior.append(pair)
        else:
            gaussprior = None

        if kw.has_key('uniformprior'):
            # If any priors are None, redefine them:
            temp_uniformprior =  kw['uniformprior']
            uniformprior = []
            for pair in temp_uniformprior:
                if pair is None:
                    uniformprior.append([-np.inf, np.inf])
                else:
                    uniformprior.append(pair)
        else:
            uniformprior = None

        if kw.has_key('ngaussprior') and kw['ngaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_ngaussprior =  kw['ngaussprior']
            ngaussprior = []
            for triplet in temp_ngaussprior:
                if triplet is not None and len(triplet)==3:
                    ngaussprior.append(triplet)
        else:
            ngaussprior = None


        #print "len(arg)>>", len(arg),

        if kw.has_key('npars'):
            npars = kw['npars']
            residuals = np.array([])
            # Excise "npars" kw for recursive calling:
            lower_kw = kw.copy()
            junk = lower_kw.pop('npars')

            # Keep fixed pairs of joint parameters:
            if kw.has_key('jointpars'):
                jointpars = kw['jointpars']
                for jointpar in jointpars:
                    params[jointpar[1]] = params[jointpar[0]]
                #pdb.set_trace()

            for ii in range(len(npars)):
                i0 = sum(npars[0:ii])
                i1 = i0 + npars[ii]
                these_params = arg[0][i0:i1]
                #ret.append(devfunc(these_params, *arg[1][ii], **lower_kw))
                these_params, lower_kw = subfit_kw(arg[0], kw, i0, i1)
                #pdb.set_trace()
                residuals = np.concatenate((residuals, devfunc(these_params, *arg[ii+1], **lower_kw).ravel()))
                #pdb.set_trace()

            return residuals

        else: # Single function-fitting
            depvar = arg[-2]
            weights = arg[-1]

            if not kw['useindepvar']:
                functions = arg[1]
                helperargs = arg[2:len(arg)-2]
            else:
                functions = arg[1]
                helperargs = arg[2:len(arg)-3]
                indepvar = arg[-3]



        if testfinite:
            finiteind = isfinite(indepvar) * isfinite(depvar) * isfinite(weights)
            indepvar = indepvar[finiteind]
            depvar = depvar[finiteind]
            weights = weights[finiteind]


        if not kw['useindepvar'] or arg[1].__name__=='multifunc' or arg[1].__name__=='sumfunc':
            if params.std()==0 or not (np.isfinite(params).all()): 
                #print "BAD!"
                model = -np.ones(len(weights))
            else:
                model = functions(*((params,)+helperargs))
        else:  # (i.e., if useindepvar is True!)
            model = functions(*((params,)+helperargs + (indepvar,)))

        # Compute the weighted residuals:
        residuals = np.sqrt(weights)*(model-depvar)


        # Compute 1D and N-D gaussian, and uniform, prior penalties:
        additionalChisq = 0.
        if gaussprior is not None:
            additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for \
                                   param0, gprior in zip(params, gaussprior)])

        if ngaussprior is not None:
            for ind, mu, cov in ngaussprior:
                dvec = params[ind] - mu
                additionalChisq += \
                    np.dot(dvec.transpose(), np.dot(np.linalg.inv(cov), dvec))

        if uniformprior is not None:
            for param0, uprior in zip(params, uniformprior):
                if (param0 < uprior[0]) or (param0 > uprior[1]):
                    residuals *= 1e9

        # Scale up the residuals so as to impose priors in chi-squared
        # space:
        if additionalChisq<>0:
            thisChisq = np.sum(weights * (model - depvar)**2)
            scaleFactor = 1. + additionalChisq / thisChisq
            residuals *= np.sqrt(scaleFactor)
    
    return residuals

def errfunc(*arg, **kw):
    """Generic function to give the chi-squared error on a generic
        function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

      OR:
       
       (fitparams, function, arg1, arg2, ... , depvar, weights, kw)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.

      If the last argument is of type dict, it is assumed to be a set
      of keyword arguments: this will be added to errfunc2's direct
      keyword arguments, and will then be passed to the fitting
      function **kw.  This is necessary for use with various fitting
      and sampling routines (e.g., kapteyn.kmpfit and emcee.sampler)
      which do not allow keyword arguments to be explicitly passed.
      So, we cheat!  Note that any keyword arguments passed in this
      way will overwrite keywords of the same names passed in the
      standard, Pythonic, way.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      gaussprior -- list of 2-tuples (or None values), same length as "fitparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  Here in
                   :func:`devfunc`, we _scale_ the error-weighted
                   deviates such that the resulting chi-squared will
                   increase by the desired amount.

      uniformprior -- list of 2-tuples (or 'None's), same length as "fitparams."
                   The i^th tuple (lo_i, hi_i) imposes a uniform prior
                   on the i^th parameter p_i by requiring that it lie
                   within the specified "high" and "low" limits.  We
                   do this (imprecisely) by multiplying the resulting
                   deviates by 1e9 for each parameter outside its
                   limits.

      ngaussprior -- list of 3-tuples of Numpy arrays.
                   Each tuple (j_ind, mu, cov) imposes a multinormal
                   Gaussian prior on the parameters indexed by
                   'j_ind', with mean values specified by 'mu' and
                   covariance matrix 'cov.' This is the N-dimensional
                   generalization of the 'gaussprior' option described
                   above. Here in :func:`devfunc`, we _scale_ the
                   error-weighted deviates such that the resulting
                   chi-squared will increase by the desired amount.

                   For example, if parameters 0 and 3 are to be
                   jointly constrained (w/unity means), set: 
                     jparams = np.array([0, 3])
                     mu = np.array([1, 1])
                     cov = np.array([[1, .9], [9., 1]])
                     ngaussprior=[[jparams, mu, cov]]  # Double brackets are key!

      scaleErrors -- bool
                   If True, instead of chi^2 we return:
                     chi^2 / s^2  +  2N ln(s)
                   Where 's' is the first input parameter (pre-pended
                   to those used for the specified function) and N the
                   number of datapoints.
   

                   In this case, the first element of 'fitparams'
                   ("s") is used to rescale the measurement
                   uncertainties. Thus weights --> weights/s^2, and
                   chi^2 --> 2 N log(s) + chi^2/s^2 (for N data points).  


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))

    """
    # 2009-12-15 13:39 IJC: Created
    # 2010-11-23 16:25 IJMC: Added 'testfinite' flag keyword
    # 2011-06-06 10:52 IJMC: Added 'useindepvar' flag keyword
    # 2011-06-24 15:03 IJMC: Added multi-function (npars) and
    #                        jointpars support.
    # 2011-06-27 14:34 IJMC: Flag-catching for multifunc calling
    # 2012-03-23 18:32 IJMC: testfinite and useindepvar are now FALSE
    #                        by default.
    # 2012-05-01 01:04 IJMC: Adding surreptious keywords, and GAUSSIAN
    #                        PRIOR capability.
    # 2012-05-08 16:31 IJMC: Added NGAUSSIAN option.
    # 2012-10-16 09:07 IJMC: Added 'uniformprior' option.
    # 2013-02-26 11:19 IJMC: Reworked return & concatenation in 'npars' cases.
    # 2013-03-08 12:54 IJMC: Added check for chisq=0 in penalty-factor cases.
    # 2013-04-30 15:33 IJMC: Added C-based chi-squared calculator;
    #                        made this function separate from devfunc.
    # 2013-07-23 18:32 IJMC: Now 'ravel' arguments for C-based function.
    # 2013-10-12 23:47 IJMC: Added 'jointpars1' keyword option.
    # 2014-05-02 11:45 IJMC: Added 'scaleErrors' keyword option..

    import pdb
    #pdb.set_trace()
    params = np.array(arg[0], copy=False)
    #if 'wrapped_joint_params' in kw:
    #    params = unwrap_joint_params(params, kw['wrapped_joint_params'])

    if isinstance(arg[-1], dict): 
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass


    if len(arg)==2:
        chisq = errfunc(params, *arg[1], **kw)

    else:
        testfinite = ('testfinite' in kw) and kw['testfinite']
        if not kw.has_key('useindepvar'):
            kw['useindepvar'] = False

        # Keep fixed pairs of joint parameters:
        if kw.has_key('jointpars1'):
            jointpars1 = kw['jointpars1']
            for jointpar1 in jointpars1:
                params[jointpar1[1]] = params[jointpar1[0]]


        if kw.has_key('gaussprior') and kw['gaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_gaussprior =  kw['gaussprior']
            gaussprior = []
            for pair in temp_gaussprior:
                if pair is None:
                    gaussprior.append([0, np.inf])
                else:
                    gaussprior.append(pair)
        else:
            gaussprior = None

        if kw.has_key('uniformprior'):
            # If any priors are None, redefine them:
            temp_uniformprior =  kw['uniformprior']
            uniformprior = []
            for pair in temp_uniformprior:
                if pair is None:
                    uniformprior.append([-np.inf, np.inf])
                else:
                    uniformprior.append(pair)
        else:
            uniformprior = None

        if kw.has_key('ngaussprior') and kw['ngaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_ngaussprior =  kw['ngaussprior']
            ngaussprior = []
            for triplet in temp_ngaussprior:
                if len(triplet)==3:
                    ngaussprior.append(triplet)
        else:
            ngaussprior = None


        #print "len(arg)>>", len(arg),

        #pdb.set_trace()
        if kw.has_key('npars'):
            npars = kw['npars']
            chisq = 0.0
            # Excise "npars" kw for recursive calling:
            lower_kw = kw.copy()
            junk = lower_kw.pop('npars')

            # Keep fixed pairs of joint parameters:
            if kw.has_key('jointpars'):
                jointpars = kw['jointpars']
                for jointpar in jointpars:
                    params[jointpar[1]] = params[jointpar[0]]
                #pdb.set_trace()

            for ii in range(len(npars)):
                i0 = sum(npars[0:ii])
                i1 = i0 + npars[ii]
                these_params = arg[0][i0:i1]
                #ret.append(devfunc(these_params, *arg[1][ii], **lower_kw))
                these_params, lower_kw = subfit_kw(arg[0], kw, i0, i1)
                #if 'wrapped_joint_params' in lower_kw:
                #    junk = lower_kw.pop('wrapped_joint_params')
                chisq  += errfunc(these_params, *arg[ii+1], **lower_kw)
                #pdb.set_trace()
            return chisq

        else: # Single function-fitting
            depvar = arg[-2]
            weights = arg[-1]

            if not kw['useindepvar']:  # Standard case:
                functions = arg[1]
                helperargs = arg[2:len(arg)-2]
            else:                      # Obsolete, deprecated case:
                functions = arg[1] 
                helperargs = arg[2:len(arg)-3]
                indepvar = arg[-3]



        if testfinite:
            finiteind = isfinite(indepvar) * isfinite(depvar) * isfinite(weights)
            indepvar = indepvar[finiteind]
            depvar = depvar[finiteind]
            weights = weights[finiteind]

        doScaleErrors = 'scaleErrors' in kw and kw['scaleErrors']==True
        if doScaleErrors:
            #pdb.set_trace()
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params[1:],)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params[1:],)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            if c_chisq:
                chisq = _chi2.chi2(model.ravel(), depvar.ravel(), \
                                       weights.ravel())
            else:
                chisq = (weights*((model-depvar))**2).sum()
            chisq = chisq/params[0]**2 + 2*depvar.size*np.log(np.abs(params[0]))

        else:
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params,)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params,)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            if c_chisq:
                chisq = _chi2.chi2(model.ravel(), depvar.ravel(), \
                                       weights.ravel())
            else:
                chisq = (weights*(model-depvar)**2).sum()
            

        # Compute 1D and N-D gaussian, and uniform, prior penalties:
        additionalChisq = 0.
        if gaussprior is not None:
            #pdb.set_trace()
            additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for \
                                   param0, gprior in zip(params, gaussprior)])

        if ngaussprior is not None:
            for ind, mu, cov in ngaussprior:
                dvec = params[ind] - mu
                additionalChisq += \
                    np.dot(dvec.transpose(), np.dot(np.linalg.inv(cov), dvec))

        if uniformprior is not None:
            for param0, uprior in zip(params, uniformprior):
                if (param0 < uprior[0]) or (param0 > uprior[1]):
                    chisq *= 1e9

        # Scale up the residuals so as to impose priors in chi-squared
        # space:
        chisq += additionalChisq
    
    return chisq

def resfunc(*arg, **kw):
    """Generic function to give the error-weighted deviates on a function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... depvar, errs)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

    EXAMPLE: 

    SEE ALSO:
      :func:`resfunc`
    """
    # 2011-11-10 09:09 IJMC: Created by copying resfunc
    # 2012-04-28 05:01 IJMC: Simplified -- now just call resfunc.

    #phasecurves.resfunc():
    if len(arg)==2:
        ret = devfunc(arg[0], *arg[1])
    else:
        ret = devfunc(*arg, **kw)

    return ret**2







def domodfit(profile, fdatc, wmat, xmat):
    """Helper function for fitsin

    Generates takes a matrix of variables and adds (1) a phase profile
    to the first row and (2) a flux conservation constraint to the
    last column.
    """
    # 2010-03-04 14:31 IJC: Created.
    # 2010-03-15 16:42 IJC: Added coefficient-covariance matrix calculation

    xmat = array(xmat,copy=True)

    if  (profile==0).all():
        xmatc = xmat.copy()
        # Add constraint of flux conservation.
        xmatc = hstack((xmat, array([0,1,0]*14+[0]).reshape(43,1)))    
    else:
        xmat = vstack((profile, xmat))
        # Add constraint of flux conservation.
        xmatc = hstack((xmat, array([0]+[0,1,0]*14+[0]).reshape(44,1)))    


    xmatc = xmatc.transpose()
    xtw = dot(xmatc.transpose(), wmat)
    coef = dot(linalg.inv(dot(xtw,xmatc)),dot(xtw,fdatc))
    ccov = linalg.inv(dot(xtw,xmatc))
    
    
    return coef, xmat.transpose(), ccov



def fitsin(pdat0,fdat0, efdat0, i2, xmat, phi=30, bsind=None):
    """Decorrelate data with 14 XYs, 14 offsets, and a sinusoid model.

    INPUTS:
    phase data (14xN)
    flux data (14xN)
    error on flux data (14xN)
    boolean time index array (14)
    matrix generated for testing
    
    phi -- either phi values to test, or number of evenly space phi
            values to test.
    bsind -- (N). bootstrap indices.  Indices, 0-N inclusive and with
            repetitions allowed.
    """
    # 2010-03-04 14:31 IJC: Created

    if (not hasattr(phi,'__iter__')):
        phi = linspace(0,pi,phi)

    if bsind==None:
        bsind = arange(i2.sum())

    nsets,nper = pdat0.shape
    ind3 = (bsind+i2.sum()*arange(nsets).reshape(nsets,1)).ravel().astype(int)

    nphi = len(phi)
    i3 = tile(i2,(14,1)).ravel()

    sinchi = 1e6+zeros(nphi,float)
    bestmod = 0; bestcoef=0; bestphi = -1
    thisfdatc = concatenate((fdat0[:,i2].ravel(), [0]))
    thisefdatc = concatenate((efdat0[:,i2].ravel(), [0.000001]))
    thiswmatc = diag(1./thisefdatc**2)

    # Use least-squares to test each phase offset:
    for ii in range(nphi):
        profile = -cos(2*pi*pdat0.ravel()-phi[ii])
        thiscoef, thisxmat, thisccov = domodfit(profile[i3], thisfdatc, thiswmatc, xmat[:,i3])
        thismodel = dot(thisxmat, thiscoef)
        #thismodAll = thismodel.reshape(nsets,i2.sum())
        residualErr = (thismodel-fdat0[:,i2].ravel())/efdat0[:,i2].ravel()
        sinchi[ii] = sum((residualErr[ind3])**2)
        if sinchi[ii]==min(sinchi):
            bestmod = thismodel
            bestcoef = thiscoef
            bestphi = phi[ii]

    return bestphi, bestcoef, min(sinchi), thismodel

def fithelper(param, func, pdat0, fdat0, efdat0, fdatc, wmatc, xmat, i2, i3, bsind=None, nsigma=5, retall=False):
    """Helper funtion for fitcurve -- XXX

    Param -- either [phi_0] or [phi_0, inclination]
    """
    # 2010-03-29 20:50 IJC: 

    if bsind==None:
        bsind = arange(i2.sum())

    #print "bsind>>", bsind

    nsets,nper = pdat0.shape
    #print "nsets,nper>>", nsets, nper

    # Select which time indices to use (same from all dither positions)
    #print bsind.min(), bsind.max(), i2.sum(), nsets, nper
    ind3 = (bsind+i2.sum()*arange(nsets).reshape(nsets,1)).ravel().astype(int)

    if not hasattr(param,'__iter__') or len(param)==1:
        phi = param
        nfit = 1
    elif len(param)>1:
        phi, inc = param[0:2]
        nfit = 2
    
    if nfit==1:
        profile = func((2*pi*pdat0.ravel()-phi) % (2*pi))
    else:
        profile = func((2*pi*pdat0.ravel()-phi) % (2*pi), inc=inc)

    thiscoef, thisxmat, thisccov = domodfit(profile[i3], fdatc, wmatc, xmat)

    #plot(2*pi*pdat0.ravel()-phi[ii], profile, '.')
    thismodel = dot(thisxmat, thiscoef)
    #thismodAll = thismodel.reshape(nsets,i2.sum())
    residualErr = (thismodel-fdat0[:,i2].ravel())/efdat0[:,i2].ravel()
    keepIndex =  abs(residualErr[ind3])<nsigma
    chi = sum((residualErr[ind3][keepIndex])**2)

    if retall:
        ret = chi, profile, thismodel, thiscoef
    else:
        ret = chi

    return ret

def fitcurve(pdat0,fdat0, efdat0, i2, xmat, phi=30, bsind=None, func=cos, retall=False, args=None, nsigma=Inf):
    """Decorrelate data with 14 XYs, 14 offsets, and a curve model of
    arbitrary amplitude.

    INPUTS:
    phase data (14xN)
    flux data (14xN)
    error on flux data (14xN)
    boolean time index array (14)
    matrix generated for testing
     
    phi --  phi values to test
    bsind -- (N). bootstrap indices.  Indices, 0-N inclusive and with
            repetitions allowed.
    func -- phase function, returning amplitude for an input phase
            value in radians.
    """
    # 2010-03-04 14:31 IJC: Created
    # 2010-03-30 09:20 IJC: Now accepts single-valued phi input as a starting point to guessing!
 
    from scipy import optimize

    if bsind==None:
        bsind = arange(i2.sum())

    listofphi = hasattr(phi,'__iter__')

    nsets,nper = pdat0.shape
    # Select which time indices to use (same from all dither positions)
    ind3 = (bsind+i2.sum()*arange(nsets).reshape(nsets,1)).ravel().astype(int)

    if listofphi:
        nphi = len(phi)
        sinchi = 1e6+zeros(nphi,float)

    i3 = tile(i2,(14,1)).ravel()
    bestmod = 0; bestcoef=0; bestccov = 0; bestphi = -1; bestprofile = 0
    thisfdatc = concatenate((fdat0[:,i2].ravel(), [0]))
    thisefdatc = concatenate((efdat0[:,i2].ravel(), [0.000001]))
    thiswmatc = diag(1./thisefdatc**2)

    if listofphi:
        # Use least-squares to test each phase offset:
        #figure()
        for ii in range(nphi):
            if args.__class__==dict and args.has_key('inc'):
                param = [phi[ii], args['inc']]
            else:
                param = [phi[ii]]

            sinchi[ii], profile, thismodel, thiscoef = fithelper(param, func, pdat0, fdat0, efdat0, thisfdatc, thiswmatc, xmat[:,i3], i2, i3, nsigma=nsigma,bsind=bsind, retall=True)

            if sinchi[ii]==min(sinchi):
                bestprofile = profile
                bestmod = thismodel
                bestcoef = thiscoef
                bestphi = phi[ii]
                
        if retall:
            ret =  bestphi, bestcoef, min(sinchi), sinchi, bestmod, bestprofile*bestcoef[0]+bestcoef[1],xmat
        else:
            ret =  bestphi, bestcoef, min(sinchi)

    else:  # just a single phi value passed in -- it's a guess!
        if args.__class__==dict and args.has_key('inc'):
            guess = [phi, args['inc']]
        else:
            guess = [phi]

        # Perform the optimization over phi_0 (and inc, if specified)
        thisfit = optimize.fmin(fithelper, guess, args=(func, pdat0, fdat0, efdat0, thisfdatc, thiswmatc, xmat[:,i3], i2, i3,bsind, nsigma, False),retall=True, disp=False)

        # Get the least-squares coefficients and best-fit model parameters.
        thischi, profile, thismodel, thiscoef = fithelper(thisfit[0], func, pdat0, fdat0, efdat0, thisfdatc, thiswmatc, xmat[:,i3], i2, i3,bsind=bsind, nsigma=nsigma, retall=True)

        if retall:
            ret = thisfit[0], thiscoef, thischi, [thischi], thismodel, profile*thiscoef[0]+thiscoef[1],xmat
        else:
            ret = thisfit[0], thiscoef, thischi

    return ret


def makexmat(xpos, ypos, constraint=True):
    """Generate matrix for least-squares MIPS photometric detrending.

    Generate the LSq dependent-variable matrix.  The rows are:
    0--constant pedestal, (ii*3+1)--dither position-dependent pedestal
    offset, (ii*3+2)--dither position-dependent x-position
    correlation, (ii*3+3)--dither position-dependent y-position
    correlation.

    EXAMPLE:
       xmat = makexmat()
       cleanedphot = dot(coef, xmat)

       """
    # 2010-04-01 17:38 IJC: Created
    if xpos.shape != ypos.shape:
        print "positional offset arrays must be of the same size!"
        return -1

    ndat = xpos.size
    nsets, nper = xpos.shape

    xmat = ones((ndat),float)
    for ii in range(nsets):
        thisIvec = zeros(ndat,float)
        thisIvec[nper*ii:(nper*(ii+1))] = 1.
        thisXvec = zeros(ndat,float)
        thisXvec[nper*ii:(nper*(ii+1))] = xpos[ii]-xpos[ii].mean()
        thisYvec = zeros(ndat,float)
        thisYvec[nper*ii:(nper*(ii+1))] = ypos[ii]-ypos[ii].mean()
        xmat = vstack((xmat, thisIvec,thisXvec,thisYvec))

    if constraint:
        xmat = hstack((xmat, array([0,1,0]*14+[0]).reshape(43,1)))

    return xmat


def mcmc14xy(z, t, x, y, sigma, params, stepsize, numit,nstep=1,sumtol=1e20, xyord=1, prodtol=1e-6, fixc0=False):
	"""
 	Applies Markov Chain Monte Carlo model fitting using the
 	Metropolis-Hastings algorithm.

    :INPUTS:
        z : 1D array
		Contains dependent data

	t,x,y : 1D array
		Contains independent data: phase, x- and y-positions

	sigma :	1D array
		Standard deviation of dependent (y) data

        params : 1D array 
		Initial guesses for parameters

	stepsize :  1D array
		Array of 1-sigma change in parameter per iteration

	numit :	int
		Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        sumtol : float
                Tolerance error, used as:  abs(params[3:17].sum()/sumtol)**2

        xyord : int
               "xyord" determines the linear order of the polynomial in x and y.

    :Returns:
	allparams : 2D array
		Contains all parameters at each step

	bestp : 1D array
		Contains best paramters as determined by lowest Chi^2

	numaccept: int
		Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step

    :REFERENCES:
	Numerical Recipes, 3rd Edition (Section 15.8); 	Wikipedia

    :REVISIONS:
	2008-05-02	Kevin Stevenson, UCF  	
				kevin218@knights.ucf.edu
				Started converting MCMC from IDL to Python while making upgrades 

 	2008-06-21	Kevin Stevenson	
				Finished updating, current stable version

	2008-11-17	Kevin Stevenson
				Updated docstring
				Simplified routine for AST3110 project and AST5765 class demo

        2010-06-12 14:21 IJC: Modified for phasecurve test

        2010-06-30 11:43 IJC: Added nstep option

        2010-07-01 16:47 IJC: Added sum constraint for parameters 3-17.

        2010-07-14 11:26 IJC: Added product constraints for parameters 3-17
	
	"""
        import numpy as np

	#Initial setup
	numaccept  = 0
        nout = numit/nstep
  	bestp      = np.copy(params)
   	allparams  = np.zeros((len(params), nout))
        allchi     = np.zeros(nout,float)

   	#Calc chi-squared for model type using current params
        if fixc0:
            zmodel     = phasesin14xymult_cfix(params, xyord, 0, t, x, y)
        else:
            zmodel     = phasesin14xymult(params, xyord, 0, t, x, y)
        #print zmodel.shape, z.shape, sigma.shape

        sum_offsetlevel = params[3:17].sum()
        prod_offsetlevel = (1.+params[3:17]).prod() -1.
   	currchisq  = (((zmodel - z)/sigma)**2).ravel().sum()
        currchisq +=  (sum_offsetlevel/sumtol)**2 + (prod_offsetlevel/prodtol)**2

   	bestchisq  = currchisq
    #Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
   	for j in range(numit):
    	#Take step in random direction for adjustable parameters
		nextp    = np.random.normal(params,stepsize)
		#COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
                if fixc0:
                    zmodel     = phasesin14xymult_cfix(nextp, xyord, 0, t, x, y)
                else:
                    zmodel     = phasesin14xymult(nextp, xyord, 0, t, x, y)
                #ACCOUNT FOR individual pedestal offset levels:

                sum_offsetlevel = nextp[3:17].sum()
                prod_offsetlevel = (1.+nextp[3:17]).prod() -1.
		nextchisq  = (((zmodel - z)/sigma)**2).ravel().sum() 
                nextchisq +=  (sum_offsetlevel/sumtol)**2 +  (prod_offsetlevel/prodtol)**2

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


def singleexp(params, t):
    """Model a simple, single-exponential function.

    params: 2- or 3-sequence, defining the function as shown below.

    t: sequence.  Input time (presumed: since start of observations)

    Functional form:
      if len(params)==2:
        F'/F = 1 - p0*exp(-t/p1)
      elif len(params)>2:
        F'/F = p2 * (1 - p0*exp(-t/p1))

    """
    # 2011-05-18 20:58 IJMC: Created
    # 2011-06-03 11:49 IJMC: Normalized to unity.

    if len(params)==2:
        return 1. - params[0] * exp(-t/params[1]) 
    else:
        return params[2] * (1. - params[0] * exp(-t/params[1]) )
    

def singleexp14(params, t):
    """Model a simple, single-exponential function.

    params: 16- or 17-sequence, defining the function as shown below.

    t: sequence.  Input time (presumed: since start of observations)

    Functional form:
      if len(params)==2:
        F'/F = 1 - p0*exp(-t/p1)
      elif len(params)>2:
        F'/F = p2 * (1 - p0*exp(-t/p1))

        ... with fourteen additional sensitivity parameters.

    """
    # 2011-10-31 15:51 IJMC: Created from singleexp

    t = array(t, copy=False)
    tsh = t.shape
    if len(tsh)==1:
        t_is_1d = True
    else:
        t_is_1d = False

    if len(params)==2:
        ret =  1. - params[0] * exp(-t/params[1]) 
    else:
        ret =  params[2] * (1. - params[0] * exp(-t/params[1]) )


    if t_is_1d:
        return (ret.reshape(14, tsh[0]/14) * (1 + params[-14::]).reshape(14,1)).ravel()
    else:
        return ret * (1 + params[-14::]).reshape(14,1)

    

def doubleexp(params, t):
    """ Model Agol et al. 2010's double-exponential IRAC ramp
    function.

    params - 4- or 5-sequence, defining the function as shown below.

    t - sequence.  Input time (presumed: since start of observations)

    Functional form:
      if len(params)==4:
        F'/F = 1 - p0*exp(-t/p2) - p1*exp(-t/p3)
      elif len(params)>4:
        F'/F = p4 * (1 - p0*exp(-t/p2) - p1*exp(-t/p3))

        """
    # 2011-05-18 20:58 IJMC: Created

    if len(params)==4:
        return 1. - params[0] * exp(-t/params[2]) - \
            params[1] * exp(-t/params[3])
    else:
        return params[4] * (1. - params[0] * exp(-t/params[2]) - \
                                params[1] * exp(-t/params[3]))



def doubleexp2(params, t):
    """ Model a different double-exponential ramp function.

   :INPUTS:

    params : 3- or 4-sequence
       Parameters that define the function as shown below.

    t : sequence. 
       Input time (presumed: since start of observations)

   :Functional_form:
      if len(params)==3:
        F'/F = (1 - p0*exp(-t/p1)) * exp(-t/p2)
      elif len(params)>3:
        F'/F = (1 - p0*exp(-t/p1)) * exp(-t/p2) * p3

        """
    # 2011-10-27 15:50 IJMC: Created

    if len(params)==3:
        return (1. - params[0] * exp(-t/params[1])) * exp(-t/params[2])
    else:
        return (1. - params[0] * exp(-t/params[1])) * exp(-t/params[2]) * params[3]

def doubleexp214(params, t):
    """ Model a different double-exponential ramp function w/14 positional offsets.

   :INPUTS:

    params : sequence of length [(3 or 4) + 14]
       Parameters that define the function as shown below.

    t : sequence. 
       Input time (presumed: since start of observations).  If not of
       shape (14 x N), will be reshaped to that.

   :Functional_form:
      if len(params)==3:
        F = (1 - p0*exp(-t/p1)) * exp(-t/p2)
      elif len(params)>3:
        F = (1 - p0*exp(-t/p1)) * exp(-t/p2) * p3
      return (F * (1 + p[3/4::]).reshape(14,1) )
        """
    # 2011-10-27 15:50 IJMC: Created

    from numpy import array
    t = array(t, copy=False)
    tsh = t.shape
    if len(tsh)==1:
        t_is_1d = True
    else:
        t_is_1d = False

    if len(params)==17:
        ret = (1. - params[0] * exp(-t/params[1])) * exp(-t/params[2])
    elif len(params)==18:
        ret = (1. - params[0] * exp(-t/params[1])) * exp(-t/params[2]) * params[3]

    if t_is_1d:
        return (ret.reshape(14, tsh[0]/14) * (1 + params[-14::]).reshape(14,1)).ravel()
    else:
        return ret * (1 + params[-14::]).reshape(14,1)


def sin2_errs(params, eparams, nphi=1e4, ntrials=1e4):
    """Estimate the uncertainties from a double-sinusoid fit.

    :FUNCTION:
       p[0] - p[1]*cos(phi + p[2]) + p[3]*cos(2*phi + p[4])

    :INPUTS:
       params : 5-sequence
          parameters for the function, as defined immediately above.
       
       eparams : 5-sequence
          1-sigma uncertainties on parameters

    :OPTIONS:
       ntrials : float
          number of Monte Carlo trials to run

       nphi : float
          number of points in phase curve (0-1, inclusive)
       
    :RETURNS:
       (visibilities, peak_offset (rad), trough_offset (rad), true_vals)

    :SEE_ALSO:
       :func:`phasesinsin14`
    """
    # 2011-10-17 14:48 IJMC: Created
    phi = np.linspace(0,1, nphi)[:-2]
    vis = np.zeros(ntrials, float)
    ploc = np.zeros(ntrials, float)
    tloc = np.zeros(ntrials, float)

    a,b,c,d,e = params
    flux = a - b*np.cos(phi*2*np.pi + c)        + d*np.cos(4*phi*np.pi + e)
    peak_loc   = (2*np.pi*phi[flux==flux.max()].mean())
    trough_loc = (2*np.pi*phi[flux==flux.min()].mean())
    visibility = (flux.max() - flux.min())/a
    #pdb.set_trace()

    for ii in range(ntrials):
        a,b,c,d,e = np.random.normal(params, eparams)
        flux = a - b*np.cos(phi*2*np.pi + c)        + d*np.cos(4*phi*np.pi + e)
        ploc[ii] = (2*np.pi*phi[flux==flux.max()].mean())
        tloc[ii] = (2*np.pi*phi[flux==flux.min()].mean())
        vis[ii] = (flux.max() - flux.min())/a

    return vis, ploc, tloc, (visibility ,peak_loc, trough_loc)

def model_fixed_param(varparam, fixedparam, fixedindex, func, *arg, **kw):
    """Allow modeling with some parameters held constant.

    :INPUTS:
      varparam : sequence
         Primary parameters (which can be varied).

      fixedparam : sequence
         Secondary parameters (which should be held fixed.)

      fixedindex : sequence of ints

         Indices of parameters which should be held fixed, when passed to 

      func : function
         Modeling function.  Arguments

      Thus if param = [10, 20, 50, 70] and holdfixed = [2], one would
      set varparam = [10, 50, 70] and fixedparam = [20].

         
    :OPTIONS:
      *arg : tuple
         Arguments to be passed to `func`

      **kw : dict
         Keyword (optional) arguments to be passed to `func`

    :OUTPUTS:
      func(param, *arg, **kw)

    """
    # 2012-04-17 16:03 IJMC: Created

    #nvar = len(varparam)
    #nfix = len(fixedparam)
    #nparam = nvar + nfix

    #param = np.zeros(nparam, float)
    param = list(varparam)
    for fparam, findex in zip(fixedparam, fixedindex):
        param.insert(findex, fparam)

    return func(param, *arg, **kw)


def rotmod(param, airmass, rotang, phase=None):
    """Model the Bean & Seifert rotation angle effect.

    :INPUTS:
      param : 3- or 4-sequence

      airmass : NumPy array

      rotang : NumPy array
        Instrument rotator angle, in radians.

      phase : NumPy array, or None
        Orbital phase (or time, etc.)

    :OUTPUTS:
      param[0] + param[1] * airmass * np.cos(param[2] + rotang) + \
         param[3] * (phase - phase.mean())
         """
    # 2012-04-18 10:44 IJMC: Created
    a, b, offset = param[0:3]
    mod = a + b*airmass * np.cos(offset + rotang)
    if phase is not None:
        mod += param[3] * (phase - phase.mean())
    return mod


def prolatesize(f, phase):
    """The projected size of a prolate ellipsoidal planet, viewed
    edge-on, at a given orbital phase.

    :INPUTS:
      f : scalar
        Ratio of large and small axes of the ellipsoid

      phase : scalar or NumPy array
        Orbital phase, in radians.  0 = transit, 1.57 ~ quadrature, etc.

     :OUTPUTS:
      a_scale 
        Value by which to scale up the area of a circle (f=1).

     :REFERENCE:
       Vickers 1996: http://dx.doi.org/10.1016/0032-5910(95)03049-2
        """
    # 2012-05-04 11:19 IJMC: Created.

    return np.sqrt(np.cos(phase)**2 + f**2 * np.sin(phase)**2)


if False:
    import analysis
    import numpy as np
    import transit
    import phasecurves as pc
    import analysis as an

    planet = analysis.getobj('WASP-12 b')
    npts = 200
    hjd = np.linspace(planet.tt - .15, planet.tt + .15, npts)
    phase_rad = planet.phase(hjd) * 2*np.pi

    k_0 = (planet.r / planet.rstar) * (analysis.rjup / analysis.rsun)
    k_eff = k_0 * np.sqrt(pc.prolatesize(1.8, phase_rad))
    z = transit.t2z(planet.tt, planet.per, planet.i, hjd, planet.ar)
    tlc_sphere = transit.occultuniform(z, k_0)
    tlc_sphereLD = transit.occultquad(z, k_0, [.161])
    tlc_prolate = np.array([transit.occultuniform(z[ii], k_eff[ii]) for ii in range(npts)]).ravel()
    tlc_prolateLD = np.array([transit.occultquad([z[ii]], [k_eff[ii]], [.162]) for ii in range(npts)]).ravel()

    #Find the most degenerate scaling (i.e., best-fit to spherical transit):
    scales = np.linspace(.95,1.05,10)
    ooo=[(np.array([transit.occultuniform(z[ii], scale*k_eff[ii]) for ii in range(npts)]).ravel() - tlc_sphereLD) for scale in scales]
    y = [(o**2).sum() for o in ooo]
    scalefit = polyfit(scales, y, 2)
    bestscale = -scalefit[1]/(2. * scalefit[0])
    tlc_prolate = np.array([transit.occultuniform(z[ii], bestscale*k_eff[ii]) for ii in range(npts)]).ravel()

    ooo=[(np.array([transit.occultquad([z[ii]], [scale*k_eff[ii]], [.162]) for ii in range(npts)]).ravel() - tlc_sphereLD) for scale in scales]
    y = [(o**2).sum() for o in ooo]
    scalefit = polyfit(scales, y, 2)
    bestscaleLD = -scalefit[1]/(2. * scalefit[0])
    tlc_prolateLD = np.array([transit.occultquad([z[ii]], [bestscaleLD*k_eff[ii]], [.162]) for ii in range(npts)]).ravel()




    #fvals = [1., 1.2, 1.4, 1.6, 1.8, 2.0]
    fvals = np.arange(0.9, 2.1, .1)
    fvals = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    prolateModels = np.zeros((len(fvals), npts), dtype=float)
    for jj, ff in enumerate(fvals):
        k_eff = k_0 * np.sqrt(pc.prolatesize(ff, phase_rad))
        ooo=[(np.array([transit.occultquad([z[ii]], [scale*k_eff[ii]], [.162]) for ii in range(npts)]).ravel() - tlc_sphereLD) for scale in scales]
        y = [(o**2).sum() for o in ooo]
        scalefit = polyfit(scales, y, 2)
        bestscale_thisLD = -scalefit[1]/(2. * scalefit[0])
        prolateModels[jj] = np.array([transit.occultquad([z[ii]], [bestscale_thisLD*k_eff[ii]], [.162]) for ii in range(npts)]).ravel()

        
    snr = 600 # per-point SNR
    ntransit = 20
    niter = 20
    allchisq = []
    for ii in range(niter):
        onevec = np.ones(npts, dtype=float)
        noise = np.random.normal(0., 1., npts) / snr / np.sqrt(ntransit)
        evec = np.ones(npts, dtype=float) / snr / np.sqrt(ntransit)
        chisq = np.zeros((len(fvals), len(fvals)), dtype=float)
        for jj in range(len(fvals)):  # iterate over models to fit to
            for kk in range(len(fvals)): # iterate over candidate models
                thisfit = an.lsq((onevec, prolateModels[kk]), prolateModels[jj])
                thismodel = thisfit[0][0] + thisfit[0][1] * prolateModels[kk]
                chisq[jj, kk] = (((prolateModels[jj] - thismodel + noise)/evec)**2).sum()
        allchisq.append(chisq)

    allchisq = np.array(allchisq)
    figure()
    ylabel('$\Delta\chi^2$', fontsize=18)
    xlabel('$R_{long}/R_P$', fontsize=18)
    for jj in range(len(fvals)):
        c,s,l = tools.plotstyle(jj)
        dif = (allchisq[:,:,jj] - allchisq[:,:,jj].min(1).reshape(niter,1))
        plot(fvals, dif.mean(0), 'o-'+c, label='%1.1f' % (fvals[jj]))
        plot(fvals, dif.mean(0) + dif.std(0), '.--'+c)
        plot(fvals, dif.mean(0) - dif.std(0), '.--'+c)

    plot(xlim(), [9.21]*2, ':k', linewidth=3)
    ylim([0,10])
    title('%i Transits Observed' % ntransit, fontsize=20)
    legend()

    noiselevel = 0.002
    ntrials = 100
    jj = 4
    bestFitRatio = np.zeros(ntrials, dtype=float)
    for ii in range(ntrials):
        observation = prolateModels[jj] + np.random.normal(0, noiselevel, npts)
        thesechisq = np.zeros(len(fvals), dtype=float)
        for kk in range(len(fvals)): # iterate over candidate models
            thisfit = an.lsq((onevec, prolateModels[kk]), observation)
            thismodel = thisfit[0][0] + thisfit[0][1] * prolateModels[kk]
            thesechisq[kk] = (((prolateModels[jj] - thismodel)/noiselevel)**2).sum()
        #thisfit = np.polyfit(fvals, thesechisq, 2)
        bestFitRatio[ii] = fvals[thesechisq==thesechisq.min()][0]


    # f, depth, scale, LLD
    def modelprolate(params, LLD=.16): 
        f, depth, scale = params[0:3]
        k_eff = np.sqrt(depth * pc.prolatesize(f, phase_rad))
        return scale * np.array([transit.occultquad([zz], [kk], [LLD]) for zz, kk in zip(z, k_eff)])


    pc.errfunc([1.2, .0126, 1.], modelprolate, observation, np.ones(npts)/noiselevel**2)


    guess = [1.1, .0127, 1.]
    fitargs = (modelprolate, observation, np.ones(npts)/noiselevel**2)
    ndim = len(guess)
    nwalkers = 20 * ndim
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pc.lnprobfunc, args=fitargs, threads=4)

    p0 = np.random.normal(guess, np.abs(guess)/100., (nwalkers*5, len(guess)))
    badp0 = ((p0[:,0] < 1) + (p0[:,1] < 0))
    p0 = p0[np.nonzero(True-badp0)[0][0:nwalkers]]
    #chisq = [pc.errfunc(par0, *fitargs) for par0 in p0]
    #models = [fitargs[0](par0, *fitargs[1:-2]) for par0 in p0]


    pos, prob, state = sampler.run_mcmc(p0, 1) # Burn-in


def prepEclipseMap(nslice, npix0, k, b, z=None):
    """Set up variables necessary for (DISCRETE!) eclipse-mapping.

    :INPUTS:
      nslice : int
        Number of slices in model

      npix : int
        Number of pixels across the digitized planet face.

      k : scalar
        Planet/star radius ratio.

      b : scalar
        Transit impact parameter [always < (1+k) ]

      z : 1D Numpy array
        Transit crossing parameter (i.e., separation between geometric
        planet and stellar centers) at epochs of interest.

    :RETURNS: 
      slicemasks : Numpy array
        [N x M x M] boolean array, one for each map cell.

      timemasks : Numpy array
        [N x T] boolean array.  Equals true for the data indices which
        contribute to the corresponding slice mask.

      cumulative_ingress_masks, cumulative_egress_masks : 3D Numpy arrays
        [nslice x M x M] boolean maps of planet occultation at various
        stages, from :func:`prepEclipseMap`

      ingress_zmasks, egress_zmasks : 2D Numpy arrays
        [nslice x N] boolean maps of which slices correspond to which
        z-indices, from :func:`prepEclipseMap`
        """
    # 2012-07-29 22:45 IJMC: Created


    #npix0 = 200
    #nslice = 10
    #planet = an.getobj('WASP-12 b')
    #k = (planet.r / planet.rstar) * (an.rjup / an.rsun)
    #nobs = 200
    #bjd = np.linspace(-planet.t14/1.8, planet.t14/1.8, nobs) + planet.tt
    #z = transit.t2z(planet.tt, planet.per, planet.i, bjd, planet.ar)
    import pdb 
    from pylab import *

    # Define some useful constants.  Note that we define everything in
    # terms of planetary radii.
    ik = 1./k
    bik = b * ik
    x0 = np.linspace(-1, 1, npix0)
    pmask = np.abs(x0 + 1j*x0.reshape(npix0, 1)) <= 1

    # Define z-points of contact points.  
    dx_contact1 = np.sqrt((1 + ik)**2 - (bik)**2)
    dx_contact2 = np.sqrt((1 - ik)**2 - (bik)**2)
    dxs = np.linspace(dx_contact1, dx_contact2, nslice+1)
    dzs = np.abs(dxs + 1j*bik)

    if z is not None:
        nobs = len(z)
        zk = z * ik
        firsthalf = np.arange(nobs) <= (zk==zk.min()).nonzero()[0][0]
        secondhalf = True - firsthalf


    # Create the maps:
    stary = x0.reshape(npix0, 1) - bik

    egress_masks = np.zeros((nslice, npix0, npix0), dtype=np.bool)
    egress_zmasks = []
    for ii in range(nslice):
        starx = x0 - dxs[ii+1]
        srr = np.abs(starx + 1j*stary) 
        egress_masks[ii] = (srr <= ik) - egress_masks[0:ii].sum(0)
        #pdb.set_trace()
        if z is not None:
            temporal_mask = (zk < dzs[ii]) * (zk >= dzs[ii+1])
            egress_zmasks.append(temporal_mask * firsthalf)
        else:
            egress_zmasks.append(None)

    ingress_masks = np.zeros((nslice, npix0, npix0), dtype=np.bool)
    ingress_zmasks = []
    for ii in range(nslice):
        starx = x0 + dxs[ii+1]
        srr = np.abs(starx + 1j*stary) 
        ingress_masks[ii] = (srr <= ik) - ingress_masks[0:ii].sum(0)
        if z is not None:
            temporal_mask = (zk < dzs[ii]) * (zk >= dzs[ii+1])
            ingress_zmasks.append(temporal_mask * secondhalf)
        else:
            ingress_zmasks.append(None)

    ingress_masks = ingress_masks[::-1] * pmask
    egress_masks = egress_masks*pmask # place in correct order

    cumulative_ingress_masks = np.cumsum(ingress_masks, axis=0)[::-1]
    cumulative_egress_masks  = np.cumsum(egress_masks[::-1],  axis=0)

    ingress_masks = ingress_masks[::-1]
    egress_masks = egress_masks[::-1]

    # Switch into correct frame (sloppy of me!!!):
    temp = egress_zmasks
    egress_zmasks = ingress_zmasks#[::-1]
    ingress_zmasks = temp#[::-1]
    egress_zmasks = egress_zmasks[::-1]

    # For large slice number, it's significantly faster to instantiate
    # a too-large array, then trim it, than to append to a growing list.
    all_masks = np.zeros((nslice**2, npix0, npix0), dtype=bool)
    all_zmasks = []
    all_binzmasks = np.zeros((nslice**2, nslice*2), dtype=int)
    iter = 0
    eg_iter = 0
    gress_id = np.zeros((nslice**2,2), dtype=int)
    for mask1, zmask1 in zip(egress_masks, egress_zmasks):
        in_iter = 0
        for mask2, zmask2 in zip(ingress_masks, ingress_zmasks):
            temp = mask1 * mask2
            if temp.any():
                #all_masks.append(temp)
                all_masks[iter] = temp
                if z is not None:
                    all_zmasks.append(zmask1 - zmask2.astype(int))
                    all_binzmasks[iter, in_iter] = -1
                    all_binzmasks[iter, nslice + eg_iter] = +1
                    gress_id[iter] = in_iter, eg_iter
                iter += 1
            in_iter += 1
        eg_iter += 1

    # Trim the excess slices:
    all_masks = all_masks[0:iter]
    all_binzmasks = all_binzmasks[0:iter]
    gress_id = gress_id[0:iter]


    if z is None:
        ret = all_masks
    else:
        all_zmasks = np.array(all_zmasks)
        ret = all_masks, all_zmasks, all_binzmasks, ingress_masks, egress_masks, ingress_zmasks, egress_zmasks, gress_id

    return ret


def map2lightcurve(map, z, k, cumulative_ingress_masks, cumulative_egress_masks, ingress_zmasks, egress_zmasks, alt=False):
    """Take a 2D planet map, and convert it into an eclipse light curve.

    :INPUTS:
      map : 2D Numpy array
        [M x M] square map of planet surface brightness
        distribution. Note that map.sum() corresponds to the
        fractional eclipse depth.

      z : 1D Numpy array
        length-N planet crossing parameter z (i.e., distance between
        planet and stellar geocenters in units of stellar radii).

      k : scalar
        planet/star radius ratio

      cumulative_ingress_masks, cumulative_egress_masks : 3D Numpy arrays
        [nslice x M x M] boolean maps of planet occultation at various
        stages, from :func:`prepEclipseMap`

      ingress_zmasks, egress_zmasks : 2D Numpy arrays
        [nslice x N] boolean maps of which slices correspond to which
        z-indices, from :func:`prepEclipseMap`

     :RETURNS:
       lightcurve : 1D Numpy array
         length-N light curve, normalized to unity in eclipse.
    """
    # 2012-07-30 10:57 IJMC: Created

    npix0 = map.shape[0]
    nslice = cumulative_ingress_masks.shape[0]
    nobs = z.size

    ingress = (cumulative_ingress_masks * map.reshape(1,npix0, npix0)).reshape(nslice, npix0**2).sum(1)
    egress = (cumulative_egress_masks * map.reshape(1,npix0, npix0)).reshape(nslice, npix0**2).sum(1)

    if alt:
        ret = np.concatenate((ingress, egress))
    else:
        lc = np.zeros(nobs, dtype=np.float32)
        lc[z >= (1.+k)] = map.sum()
        for ii in range(nslice):
            lc[ingress_zmasks[ii]] = ingress[ii]
            lc[egress_zmasks[ii]] = egress[ii]
        ret = lc+1

    return ret


def visit_offsets(visitcoef, masks):
    """
    :INPUTS:
       visitcoef : 1D NumPy array
         offsets from unity for each of N HST visits

       masks : 2D NumPy array, N x M
         Set of boolean masks for each of N visits (not orbit!),
         assuming M total observations.

     :NOTES:
       Note that visitcoef[0] will be set so that the quantity
       (1. + visitcoef).prod() always equals unity.
       """
    # 2012-12-21 11:16 IJMC: Created

    nvisits = masks.shape[0]
    visitcoef[0] = 1./(1. + visitcoef[1:]).prod() - 1. # enforce constraint
    mod = (masks * visitcoef.reshape(nvisits, 1)).sum(0)
    #mod = (masks * visitcoef.reshape(nvisits, 1))[masks] # <--- slower!!
    return 1. + mod




def unwrap_joint_params(params, jfw_indices):
    """Unwrap parameters that are jointly constrained.

    :INPUTS:
      params -- 1D NumPy array
        The P *non-redundant* parameters to the input function --
         i.e., any parameters which are to be jointly fit (i.e., both
         held to the same, floating, value) are included only once.

      jfw_indices : sequence of scalars and sequences
        A length-P sequence of scalars and sequences.  Each element
        jfw_indices[i] indicates the indices in the unwrapped set of
        parameters that will be assigned the value of params[i]. 

        The final value of jfw_indices should be an integer equal to
        the length of the final set of unwrapped parameters.

    :EXAMPLE:
     ::
       
       import tools
       import numpy as np

       npts = 100
       snr = 20.
       params = [1, 0.5, 1, 1]
       x = np.linspace(0, 1, npts)
       y = np.polyval(params, x) + np.random.randn(npts)/snr
       jointpars = [(0, 2), (0, 3)]

       joint_guess = np.array([1, 0.5])
       jfw_indices = [[0, 2, 3], [1], 4]
       full_params = tools.unwrap_joint_params(joint_guess, jfw_indices)
       
    :SEE_ALSO:
      :func:`wrap_joint_params`

    """
    # 2013-04-30 17:06 IJMC: Created

    njfw = len(jfw_indices) - 1
    n_jfw_params = jfw_indices[-1]
    if hasattr(params, 'dtype'):
        dtype = params.dtype
    else:
        dtype = np.object
    
    new_params = np.zeros(n_jfw_params, dtype=dtype)
    #pdb.set_trace()
    for ii in xrange(njfw):
        ind = jfw_indices[ii]
        if hasattr(ind, '__iter__') and len(ind)>1:
            for subind in ind:
                new_params[subind] = params[ii]
        else:
            new_params[ind] = params[ii]

    return new_params


def wrap_joint_params(params, jointpars):
    """Wrap parameters that are jointly constrained.

    :INPUTS:
      params -- 1D NumPy array
        All parameters, some of which may be jointly constrained.

      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

    :EXAMPLE:
     ::
       
       import tools
       import numpy as np

       npts = 100
       snr = 20.
       params = [1, 0.5, 1, 1]
       x = np.linspace(0, 1, npts)
       y = np.polyval(params, x) + np.random.randn(npts)/snr
       jointpars = [(0, 2), (0, 3)]

       all_params, joint_indices = tools.wrap_joint_params(full_params, jointpars)
       wrapped_params = tools.unwrap_joint_params(all_params, joint_indices)
       
       
    :SEE_ALSO:
      :func:`unwrap_joint_params`
      """
    nparam = len(params)
    njoint = len(jointpars)
    ret_ind = []
    ret_par = []
    already_joint = []
    all_joint = []

    for joint_constraint in jointpars:
        j0 = joint_constraint[0]
        if j0 in already_joint:
            ind = already_joint.index(j0)
            ret_ind[ind].append(joint_constraint[1])
            all_joint += list(joint_constraint[1:])
        else:
            ret_par.append(params[j0])
            already_joint.append(j0)
            all_joint += list(joint_constraint)
            ret_ind.append(list(joint_constraint))

    for ii in xrange(nparam):
        if ii in all_joint:
            pass
        else:
            ret_par.append(params[ii])
            ret_ind.append([ii])

    ret_ind.append(nparam)
    return ret_par, ret_ind

def ramp2p(params, phase, args=dict(n=3, guess=[1, -0.16, 4.2])):
    """Model Ramp Eq. 2 (positive) from Stevenson et al. (2011).

    params: 3-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + np.exp(-r[1]*phase + r[2]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + np.exp(-params[1]*phase + params[2]))
    
def ramp2n(params, phase, args=dict(n=3, guess=[1, 26.6, 7.8])):
    """Model Ramp Eq. 2 (negative) from Stevenson et al. (2011).

    params: 3-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. - np.exp(-r[1]*phase + r[2]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. - np.exp(-params[1]*phase + params[2]))
    
def ramp3p(params, phase, args=dict(n=4, guess=[1, -0.16, 4.2, 0.1])):
    """Model Ramp Eq. 3 (positive) from Stevenson et al. (2011).

    params: 4-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + np.exp(-r[1]*t + r[2]) + r[3] * (t - 0.5))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + np.exp(-params[1]*phase + params[2]) + \
                             params[3] * (phase - 0.5))
    

def ramp3n(params, phase, args=dict(n=4, guess=[1, 141, 57.7, 0.123])):
    """Model Ramp Eq. 3 (negative) from Stevenson et al. (2011).

    params: 4-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. - np.exp(-r[1]*t + r[2]) + r[3] * (t - 0.5))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. - np.exp(-params[1]*phase + params[2]) + \
                             params[3] * (phase - 0.5))
    
    
def ramp4p(params, phase, args=dict(n=5, guess=[1, -0.068, 2.33, 0.933, -20.5])):
    """Model Ramp Eq. 4 (positive) from Stevenson et al. (2011).

    params: 5-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + np.exp(-r[1]*phase + r[2]) + r[3] * (phase - 0.5) + r[4] * (phase - 0.5)**2)

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + np.exp(-params[1]*phase + params[2]) + \
                             params[3] * (phase - 0.5) + \
                             params[4] * (phase - 0.5)**2)
    
def ramp4n(params, phase, args=dict(n=5, guess=[1, -3.7e-4, -0.94, 0.087, -1.08])):
    """Model Ramp Eq. 4 (negative) from Stevenson et al. (2011).

    params: 5-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. - np.exp(-r[1]*phase + r[2]) + r[3] * (phase - 0.5) + r[4] * (phase - 0.5)**2)

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. - np.exp(-params[1]*phase + params[2]) + \
                             params[3] * (phase - 0.5) + \
                             params[4] * (phase - 0.5)**2)
    
    
def ramp5p(params, phase, args=dict(n=5, guess=[1, -0.32, 2, -0.08, 2])):
    """Model Ramp Eq. 5 (positive) from Stevenson et al. (2011).

    params: 5-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + np.exp(-r[1]*phase + r[2]) + np.exp(-r[3]*phase + r[4]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + np.exp(-params[1]*phase + params[2]) + \
                             np.exp(-params[3]*phase + params[4]))
    
def ramp5n(params, phase, args=dict(n=5, guess=[1., 20, 83, 8.1, -0.1])): #-0.16, 4.4, -0.16, 0.43])):
    """Model Ramp Eq. 5 (negative) from Stevenson et al. (2011).

    params: 5-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. - np.exp(-r[1]*phase + r[2]) - np.exp(-r[3]*phase + r[4]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. - np.exp(-params[1]*phase + params[2]) - \
                             np.exp(-params[3]*phase + params[4]))
    

def ramp6(params, phase, args=dict(n=4, guess=[1, 0.053,  0.0040 ,  0.4])):
    """Model Ramp Eq. 6 from Stevenson et al. (2011).

    params: 4-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * (phase - 0.5) + r[2] * np.log(phase - r[3]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    if params[3]>=phase.min():
        params[3] = phase.min() - np.diff(phase).mean()/1e6
    
    return params[0] * (1. + params[1] * (phase - 0.5) + params[2] * np.log(phase - params[3]))
    
def ramp7(params, phase, args=dict(n=5, guess=[1, 0.034, 0.35, 0.005, 0.35])):
    """Model Ramp Eq. 7 from Stevenson et al. (2011).

    params: 5-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * (phase - 0.5) + \
                            r[2] * (phase - 0.5)**2 + \
                            r[3] * np.log(phase - r[4]))

    """
    # 2013-12-07 14:08 IJMC: Created.

    if params[4]>=phase.min():
        params[4] = phase.min() - np.diff(phase).mean()/1e6
    
    return params[0] * (1. + params[1] * (phase - 0.5) + \
                        params[2] * (phase - 0.5)**2 + \
                        params[3] * np.log(phase - params[4]))

    
def ramp8(params, phase, args=dict(n=4, guess=[1, 0.0096, 0.35, 5.3e-4])):
    """Model Ramp Eq. 8 from Stevenson et al. (2011).

    params: 4-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * np.log(phase - r[2]) + \
                            r[3] * np.log(phase - r[2])**2)

    """
    # 2013-12-07 14:08 IJMC: Created.

    if params[2]>=phase.min():
        params[2] = phase.min() - np.diff(phase).mean()/1e6
    
    return params[0] * (1. + params[1] * np.log(phase - params[2]) + \
                            params[3] * np.log(phase - params[2])**2)

    
def ramp9(params, phase, args=dict(n=6, guess=[1, 0.003, 0.6, 0.009, 0.35, 4e-4])):
    """Model Ramp Eq. 9 from Stevenson et al. (2011).

    params: 6-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * (phase - 0.5) + \
                            r[2] * (phase - 0.5)**2 + \
                            r[3] * np.log(phase - r[4]) + \
                            r[5] * np.log(phase - r[4])**2)

    """
    # 2013-12-07 14:08 IJMC: Created.

    if params[4]>=phase.min():
        params[4] = phase.min() - np.diff(phase).mean()/1e6
    
    return params[0] * (1. + params[1] * (phase - 0.5) + \
                        params[2] * (phase - 0.5)**2 + \
                        params[3] * np.log(phase - params[4]) + \
                        params[5] * np.log(phase - params[4])**2)

    
def ramp10(params, phase, args=dict(n=2, guess=[1, 0.2])):
    """Model Ramp Eq. 10 from Stevenson et al. (2011).

    params: 2-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * (phase - 0.5))
    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + params[1] * (phase - 0.5))

def ramp11(params, phase, args=dict(n=3, guess=[1, 0.14, -1.9])):
    """Model Ramp Eq. 11 from Stevenson et al. (2011).

    params: 3-sequence
      parameters that define the function, as shown below.

    phase: NumPy array. 
      Orbital phase (or more generally, 'time')

    Functional form:
        ramp = r[0] * (1. + r[1] * (phase - 0.5) + r[2] * (phase - 0.5)**2)
    """
    # 2013-12-07 14:08 IJMC: Created.

    return params[0] * (1. + params[1] * (phase - 0.5) + params[2] * (phase - 0.5)**2)


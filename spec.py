"""
Suite to reduce spectroscopic data.

subfunctions:
   calibrate
      setheaders -- exptime, gain, readnoise, etc.
      makeflat -- make median flat and noisy pixel map
      makedark -- make median dark, and estimate  noise in each pixel.
   clean -- clean and replace bad pixels
   extract
      trace -- trace spectral orders
      makeprofile -- compute mean spectral PSF (a spline) for an order
      fitprofile -- fit given spline-PSF to a spectral cross-section

Utilities:
   pickloc   
   fitPSF
"""
# 2010-07-02 10:56 IJC: Began the great endeavor.



try:
    from astropy.io import fits as pyfits
except:
    import pyfits

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, interpolate
import pdb


class baseObject:
    """Empty object container.
    """
    def __init__(self):
        return


########
# Utilities to add in a separate cohesive package at a later date:
from analysis import polyfitr, stdr, binarray, gaussian, egaussian
import analysis as an
from nsdata import bfixpix

########
# Parameters to put in a GUI:
gain = 5 # e-/ADU
readnoise = 25 # e-


#########

def message(text):
    """Display a message; for now, with text."""
    from sys import stdout

    print text
    stdout.flush()

def pickloc(ax=None, zoom=10):
    """
    :INPUTS:
       ax   : (axes instance) -- axes in which to pick a location

       zoom : int -- zoom radius for target confirmation
            : 2-tuple -- (x,y) radii for zoom confirmation.
    """
    # 2011-04-29 19:26 IJC: 
    # 2011-09-03 20:59 IJMC: Zoom can now be a tuple; x,y not cast as int.

    pickedloc = False
    if ax is None:
        ax = plt.gca()

    axlimits = ax.axis()

    if hasattr(zoom, '__iter__') and len(zoom)>1:
        xzoom, yzoom = zoom
    else:
        xzoom = zoom
        yzoom = zoom

    while not pickedloc:
        ax.set_title("click to select location")
        ax.axis(axlimits)

        x = None
        while x is None:  
            selectevent = plt.ginput(n=1,show_clicks=False)
            if len(selectevent)>0: # Prevent user from cancelling out.
                x,y = selectevent[0]

        #x = x.astype(int)
        #y = y.astype(int)

        
        if zoom is not None:
            ax.axis([x-xzoom,x+xzoom,y-yzoom,y+yzoom])

        ax.set_title("you selected xy=(%i,%i)\nclick again to confirm, or press Enter/Return to try again" %(x,y)  )
        plt.draw()

        confirmevent = plt.ginput(n=1,show_clicks=False)
        if len(confirmevent)>0:  
            pickedloc = True
            loc = confirmevent[0]

    return loc

def fitTophat(vec, err=None, verbose=False, guess=None):
    """Fit a 1D tophat function to an input data vector.

    Return the fit, and uncertainty estimates on that fit.
    
    SEE ALSO: :func:`analysis.gaussian`"""
    xtemp = np.arange(1.0*len(vec))

    if guess is None: # Make some educated guesses as to the parameters:
        pedestal = 0.5 * (0.8*np.median(vec) + 0.2*(vec[0:2].sum()))
        area = (vec-pedestal).sum()
        centroid = (vec*xtemp).sum()/vec.sum()
        if centroid<0:
            centroid = 1.
        elif centroid>len(vec):
            centroid = len(vec)-2.

        sigma = area/vec[int(centroid)]/np.sqrt(2*np.pi)
        if sigma<=0:
            sigma = 0.01

        guess = [area,sigma,centroid,pedestal]

    if verbose: 
        print 'Gaussian guess parameters>>', guess
    if err is None:
        fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec), full_output=True)[0:2]
        pc.resfunc


        fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec), full_output=True)[0:2]
    else:
        fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec, err), full_output=True)[0:2]

    if fitcov is None: # The fitting was really bad!
        fiterr = np.abs(fit)
    else:
        fiterr = np.sqrt(np.diag(fitcov))

    if verbose:
        print 'Best-fit parameters>>', fit
        f = plt.figure()
        ax = plt.axes()
        plt.plot(xtemp, vec, 'o', \
                     xtemp, gaussian(fit, xtemp), '-', \
                     xtemp, gaussian(guess, xtemp), '--')

    return fit, fiterr

def fitGaussian(vec, err=None, verbose=False, guess=None, sigmaguess=None):
    """Fit a Gaussian function to an input data vector.

    Return the fit, and uncertainty estimates on that fit.
    
    SEE ALSO: :func:`analysis.gaussian`"""
    # 2012-12-20 13:28 IJMC: Make a more robust guess for the centroid.
    # 2016-04-20 09:13 IJMC: Added 'sigmaguess' option
    xtemp = np.arange(1.0*len(vec))

    if guess is None: # Make some educated guesses as to the parameters:
        pedestal = (0.8*np.median(vec) + 0.2*(vec[0:2].sum()))
        area = (vec-pedestal).sum()
        centroid = ((vec-pedestal)**2*xtemp).sum()/((vec-pedestal)**2).sum()
        if np.isnan(centroid):
            centroid = min(1., vec.size-1)
        elif centroid<0:
            centroid = 1.
        elif centroid>len(vec):
            centroid = len(vec)-2.
        #pdb.set_trace()
        sigma = area/vec[int(centroid)]/np.sqrt(2*np.pi)
        if sigma<=0:
            sigma = .01
        if sigmaguess is not None:
            sigma = sigmaguess

        guess = [area,sigma,centroid,pedestal]

    if err is None:
        err = np.ones(vec.shape, dtype=float)

    badvals = True - (np.isfinite(xtemp) * np.isfinite(err) * np.isfinite(vec))
    vec[badvals] = np.median(vec[True - badvals])
    err[badvals] = vec[True - badvals].max() * 1e9

    if verbose: 
        print 'Gaussian guess parameters>>', guess

    if not np.isfinite(xtemp).all():
        pdb.set_trace()
    if not np.isfinite(vec).all():
        pdb.set_trace()
    if not np.isfinite(err).all():
        pdb.set_trace()
    try:
        fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec, err), full_output=True)[0:2]
    except:
        pdb.set_trace()

    if fitcov is None: # The fitting was really bad!
        fiterr = np.abs(fit)
    else:
        fiterr = np.sqrt(np.diag(fitcov))

    #pdb.set_trace()
    if verbose:
        print 'Best-fit parameters>>', fit
        f = plt.figure()
        ax = plt.axes()
        plt.plot(xtemp, vec, 'o', \
                     xtemp, gaussian(fit, xtemp), '-', \
                     xtemp, gaussian(guess, xtemp), '--')

    return fit, fiterr



def fitGaussiann(vec, err=None, verbose=False, guess=None, holdfixed=None):
    """Fit a Gaussian function to an input data vector.

    Return the fit, and uncertainty estimates on that fit.
    
    SEE ALSO: :func:`analysis.gaussian`"""
    from phasecurves import errfunc
    from analysis import fmin

    xtemp = np.arange(1.0*len(vec))

    if guess is None: # Make some educated guesses as to the parameters:
        holdfixed = None

        pedestal = 0.5 * (0.8*np.median(vec) + 0.2*(vec[0:2].sum()))
        area = (vec-pedestal).sum()
        centroid = (vec*xtemp).sum()/vec.sum()
        if centroid<0:
            centroid = 1.
        elif centroid>len(vec):
            centroid = len(vec)-2.

        sigma = area/vec[int(centroid)]/np.sqrt(2*np.pi)
        if sigma<=0:
            sigma = 0.01

        guess = [area,sigma,centroid,pedestal]

    if err is None:
        err = np.ones(vec.shape, dtype=float)

    badvals = True - (np.isfinite(xtemp) * np.isfinite(err) * np.isfinite(vec))
    vec[badvals] = np.median(vec[True - badvals])
    err[badvals] = vec[True - badvals].max() * 1e9

    if verbose: 
        print 'Gaussian guess parameters>>', guess

    if not np.isfinite(xtemp).all():
        pdb.set_trace()
    if not np.isfinite(vec).all():
        pdb.set_trace()
    if not np.isfinite(err).all():
        pdb.set_trace()

    try:
        #fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec, err), full_output=True)[0:2]
        fitargs = (gaussian, xtemp, vec, 1./err**2)
        fit = fmin(errfunc, guess, args=fitargs, full_output=True, disp=False, holdfixed=holdfixed)[0]
        fitcov = None
    except:
        pdb.set_trace()

    if fitcov is None: # The fitting was really bad!
        fiterr = np.abs(fit)
    else:
        fiterr = np.sqrt(np.diag(fitcov))

    if verbose:
        print 'Best-fit parameters>>', fit
        f = plt.figure()
        ax = plt.axes()
        plt.plot(xtemp, vec, 'o', \
                     xtemp, gaussian(fit, xtemp), '-', \
                     xtemp, gaussian(guess, xtemp), '--')

    return fit, fiterr

def fit2Gaussian(vec, err=None, verbose=False, guess=None, holdfixed=None):
    """Fit two Gaussians simultaneously to an input data vector.

    :INPUTS:
      vec : sequence
        1D array or list of values to fit to

      err : sequence
        uncertainties on vec

      guess : sequence
        guess parameters: [area1, sigma1, cen1, area2, sig2, cen2, constant].
        Note that parameters are in pixel units.

      holdfixed : sequence
        parameters to hold fixed in analysis, _IF_ guess is passed in.


    SEE ALSO: :func:`analysis.gaussian`, :func:`fitGaussian`"""

    # 2012-03-14 14:33 IJMC: Created

    from tools import sumfunc
    from phasecurves import errfunc
    from analysis import fmin  # Don't use SciPy, because I want keywords!


    xtemp = np.arange(1.0*len(vec))
    if err is None:
        err = np.ones(xtemp.shape)
    else:
        err = np.array(err, copy=False)
    
    if guess is None: # Make some semi-educated guesses as to the parameters:
        holdfixed = None

        pedestal = 0.5 * (0.8*np.median(vec) + 0.2*(vec[0:2].sum()))
        area = (vec-pedestal).sum()
        centroid = (vec*xtemp).sum()/vec.sum()
        if centroid<0:
            centroid = 1.
        elif centroid>len(vec):
            centroid = len(vec)-2.

        sigma = area/vec[int(centroid)]/np.sqrt(2*np.pi)
        if sigma<=0:
            sigma = 0.01

        guess1 = [area/2.,sigma/1.4,centroid+xtemp.size/5.]
        guess2 = [area/2.,sigma/1.4,centroid-xtemp.size/5.]
        guess = guess1 + guess2 + [pedestal]


    ## Testing:
    #mod = sumfunc(guess, gaussian, gaussian, 3, 4, args1=(xtemp,), args2=(xtemp,)) 
    #fitargs = (sumfunc, gaussian, gaussian, 3, 4, (xtemp,), (xtemp,), None, vec, 1./err**2)
    #fitkw = dict(useindepvar=False, testfinite=False)
    #chisq = errfunc(guess, *fitargs, **fitkw)
    #thisfit = fmin(errfunc, guess, args=fitargs, kw=fitkw, full_output=True)
    #mod2 = sumfunc(thisfit[0], gaussian, gaussian, 3, 4, args1=(xtemp,), args2=(xtemp,)) 

    if verbose: 
        print '2-Gaussian guess parameters>>', guess

    fitargs = (sumfunc, gaussian, gaussian, 3, 4, (xtemp,), (xtemp,), vec, 1./err**2)
    fitkw = dict(useindepvar=False, testfinite=False)
    fit = fmin(errfunc, guess, args=fitargs, kw=fitkw, full_output=True, disp=False, holdfixed=holdfixed)

    if verbose:
        model = sumfunc(fit[0], gaussian, gaussian, 3, 4, args1=(xtemp,), args2=(xtemp,)) 
        model1 = gaussian(fit[0][0:3], xtemp)
        model2 = gaussian(fit[0][3:6], xtemp)
        print 'Best-fit parameters>>', fit[0]
        f = plt.figure()
        ax = plt.axes()
        plt.plot(xtemp, vec, 'o', \
                     xtemp, model, '--')
        plt.plot(xtemp, model1+fit[0][6], ':')
        plt.plot(xtemp, model2+fit[0][6], ':')

    return fit[0]

def fitPSF(ec, guessLoc, fitwidth=20, verbose=False, sigma=5, medwidth=6, err_ec=None):
    """
    Helper function to fit 1D PSF near a given region.  Assumes
    spectrum runs horizontally across the frame!

    ec : 2D numpy array
        echellogram array, with horizontal dispersion direction
    guessLoc : 2-tuple
       A slight misnomer for this (x,y) tuple: y is a guess and will
         be fit, but x is the coordinate at which the fitting takes
         place.
    fitwidth : int
       width of cross-dispersion direction to use in fitting
    medwidth : int
       number of columns to average over when fitting a profile
    verbose : bool
       verbosity/debugging printout flag

    sigma : scalar
       sigma scale for clipping bad values
    """
    # 2010-08-24 22:00 IJC: Added sigma option
    # 2010-11-29 20:54 IJC: Added medwidth option
    # 2011-11-26 23:17 IJMC: Fixed bug in computing "badval"
    # 2012-04-27 05:15 IJMC: Now allow error estimates to be passed in.
    # 2012-04-28 08:48 IJMC: Added better guessing for initial case.

    if verbose<0:
        verbose = False

    ny, nx = ec.shape

    x = guessLoc[0].astype(int)
    y = guessLoc[1].astype(int)

    # Fit the PSF profile at the initial, selected location:
    ymin = min(ny-1, max(y-fitwidth/2, 0))
    ymax = max(1, min(y+fitwidth/2, ny))
    xmin = min(nx-1, max(x-medwidth/2, 0))
    xmax = max(1, min(x+medwidth/2, nx))
    if verbose:
        message("Sampling: ec[%i:%i,%i:%i]"%(ymin,ymax,xmin,xmax))

    firstSeg = np.median(ec[ymin:ymax,xmin:xmax],1)
    if verbose:
        print firstSeg

    if err_ec is None:
        ey = stdr(firstSeg, sigma)
        badval = abs((firstSeg-np.median(firstSeg))/ey) > sigma
        err = np.ones(firstSeg.shape, float)
        err[badval] = 1e9
    else:
        err = np.sqrt(np.abs(err_ec[ymin:ymax,xmin:xmax]**2).mean(1))
        err[True - np.isfinite(err)] = err[np.isfinite(err)].max() * 1e9

    guessAmp = (an.wmean(firstSeg, 1./err**2) - np.median(firstSeg)) * fitwidth
    if not np.isfinite(guessAmp):
        pdb.set_trace()

    #fit, efit = fitGaussian(firstSeg, verbose=verbose, err=err, guess=[guessAmp[0], 5, fitwidth/2., np.median(firstSeg)])
    fit, efit = fitGaussian(firstSeg, verbose=verbose, err=err, guess=None)
    newY = ymin+fit[2]
    err_newY = efit[2]
    if verbose:
        message("Initial position: (%3.2f,%3.2f)"%(x,newY))

    return x, newY, err_newY


def traceorders(filename, pord=5, dispaxis=0, nord=1, verbose=False, ordlocs=None, stepsize=20, fitwidth=20, plotalot=False, medwidth=6, xylims=None, uncertainties=None, g=gain, rn=readnoise, badpixelmask=None, retsnr=False, retfits=False):
    """
    Trace spectral orders for a specified filename.

    filename : str OR 2D array
        full path and filename to a 2D echelleogram FITS file, _OR_
        a 2D numpy array representing such a file.

    OPTIONAL INPUTS:
    pord : int
        polynomial order of spectral order fit

    dispaxis : int
        set dispersion axis: 0 = horizontal and  = vertical
    
    nord : int
        number of spectral orders to trace.

    ordlocs : (nord x 2) numpy array
        Location to "auto-click" and 

    verbose: int
        0,1,2; whether (and how much) to print various verbose debugging text

    stepsize : int
        number of pixels to step along the spectrum while tracing

    fitwidth : int
        number of pixels to use when fitting a spectrum's cross-section
        
    medwidth : int
        number of columns to average when fitting profiles to echelleograms
        
    plotalot : bool
        Show pretty plot?  If running in batch mode (using ordlocs)
        default is False; if running in interactive mode (ordlocs is
        None) default is True.

    xylims : 4-sequence
        Extract the given subset of the data array: [xmin, xmax, ymin, ymax]

    uncertainties : str OR 2D array
        full path and filename to a 2D uncertainties FITS file, _OR_
        a 2D numpy array representing such a file.
        
        If this is set, 'g' and 'rn' below are ignored.  This is
        useful if, e.g., you are analyzing data which have already
        been sky-subtracted, nodded on slit, or otherwise altered.
        But note that this must be the same size as the input data!

    g : scalar > 0
      Detector gain, electrons per ADU (for setting uncertainties)
    
    rn : scalar > 0
      Detector read noise, electrons (for setting uncertainties)
    
    retsnr : bool
      If true, also return the computed S/N of the position fit at
      each stepped location.
    
    retfits : bool
      If true, also return the X,Y positions at each stepped location.
    

    :RETURNS:
       (nord, pord) shaped numpy array representing the polynomial
         coefficients for each order (suitable for use with np.polyval)

    :NOTES:

      If tracing fails, a common reason can be that fitwidth is too
      small.  Try increasing it!
    """
    # 2010-08-31 17:00 IJC: If best-fit PSF location goes outside of
    #     the 'fitwidth' region, nan values are returned that don't
    #     cause the routine to bomb out quite so often.
    # 2010-09-08 13:54 IJC: Updated so that going outside the
    #     'fitwidth' region is determined only in the local
    #     neighborhood, not in relation to the (possibly distant)
    #     initial guess position.
    # 2010-11-29 20:56 IJC: Added medwidth option
    # 2010-12-17 09:32 IJC: Changed color scaling options; added xylims option.

    # 2012-04-27 04:43 IJMC: Now perform _weighted_ fits to the measured traces!
    # 2015-04-23 15:21 IJMC: Not take modulus of square root

    global gain
    global readnoise

    if verbose < 0:
        verbose = 0

    if not g==gain:
        message("Setting gain to " + str(g))
        gain = g

    if not rn==readnoise:
        message("Setting readnoise to " + str(rn))
        readnoise = rn

    if ordlocs is not None:
        ordlocs = np.array(ordlocs, copy=False)
        if ordlocs.ndim==1:
            ordlocs = ordlocs.reshape(1, ordlocs.size)
        autopick = True
    else:
        autopick = False


    plotalot = (not autopick) or plotalot
    if isinstance(filename, np.ndarray):
        ec = filename.copy()
    else:
        try:
            ec = pyfits.getdata(filename)
        except:
            message("Could not open filename %s" % filename)
            return -1


    if isinstance(uncertainties, np.ndarray):
        err_ec = uncertainties.copy()
    else:
        try:
            err_ec = pyfits.getdata(uncertainties)
        except:
            err_ec = np.sqrt(np.abs(ec * gain + readnoise**2))

    if dispaxis<>0:
        ec = ec.transpose()
        err_ec = err_ec.transpose()
        if verbose: message("Took transpose of echelleogram to rotate dispersion axis")
    else:
        pass

    if xylims is not None:
        try:
            ec = ec[xylims[0]:xylims[1], xylims[2]:xylims[3]]
            err_ec = err_ec[xylims[0]:xylims[1], xylims[2]:xylims[3]]
        except:
            message("Could not extract subset: ", xylims)
            return -1

    if badpixelmask is None:
        badpixelmask = np.zeros(ec.shape, dtype=bool)
    else:
        if not hasattr(badpixelmask, 'shape'):
            badpixelmask = pyfits.getdata(badpixelmask)
        if xylims is not None:
            badpixelmask = badpixelmask[xylims[0]:xylims[1], xylims[2]:xylims[3]]

        
    err_ec[badpixelmask.nonzero()] = err_ec[np.isfinite(err_ec)].max() * 1e9

    try:
        ny, nx = ec.shape
    except:
        message("Echellogram file %s does not appear to be 2D: exiting." % filename)
        return -1

    if plotalot:
        f = plt.figure()
        ax = plt.axes()
        plt.imshow(ec, interpolation='nearest',aspect='auto')
        sortedvals = np.sort(ec.ravel())
        plt.clim([sortedvals[nx*ny*.01], sortedvals[nx*ny*.99]])
        #plt.imshow(np.log10(ec-ec.min()+np.median(ec)),interpolation='nearest',aspect='auto')
        ax.axis([0, nx, 0, ny])

    orderCoefs = np.zeros((nord, pord+1), float)
    position_SNRs = []
    xyfits = []
    if not autopick:

        ordlocs = np.zeros((nord, 2),float)
        for ordernumber in range(nord):
            message('Selecting %i orders; please click on order %i now.' % (nord, 1+ordernumber))
            plt.figure(f.number)
            guessLoc = pickloc(ax, zoom=fitwidth)
            ordlocs[ordernumber,:] = guessLoc
            ax.plot([guessLoc[0]],[guessLoc[1]], '*k')
            ax.axis([0, nx, 0, ny])
            plt.figure(f.number)
            plt.draw()
            if verbose:
                message("you picked the location: ")
                message(guessLoc)

    for ordernumber in range(nord):
        guessLoc = ordlocs[ordernumber,:]
        xInit, yInit, err_yInit = fitPSF(ec, guessLoc, fitwidth=fitwidth,verbose=verbose, medwidth=medwidth, err_ec=err_ec)
        if plotalot:
            ax.plot([xInit],[yInit], '*k')
            ax.axis([0, nx, 0, ny])
            plt.figure(f.number)
            plt.draw()

        if verbose:
            message("Initial (fit) position: (%3.2f,%3.2f)"%(xInit,yInit))

        # Prepare to fit PSFs at multiple wavelengths.

        # Determine the other positions at which to fit:
        xAbove = np.arange(1, np.ceil(1.0*(nx-xInit)/stepsize))*stepsize + xInit
        xBelow = np.arange(-1,-np.ceil((1.+xInit)/stepsize),-1)*stepsize + xInit
        nAbove = len(xAbove)
        nBelow = len(xBelow)
        nToMeasure = nAbove + nBelow + 1
        iInit = nBelow

        if verbose:
            message("Going to measure PSF at the following %i locations:"%nToMeasure )
            message(xAbove)
            message(xBelow)
        
        # Measure all positions "above" the initial selection:
        yAbove = np.zeros(nAbove,float)
        err_yAbove = np.zeros(nAbove,float)
        lastY = yInit
        for i_meas in range(nAbove):
            guessLoc = xAbove[i_meas], lastY

            thisx, thisy, err_thisy = fitPSF(ec, guessLoc, fitwidth=fitwidth, verbose=verbose-1, medwidth=medwidth, err_ec=err_ec)
            if abs(thisy - yInit)>fitwidth/2:
                thisy = yInit
                err_thisy = yInit
                lastY = yInit
            else:
                lastY = thisy.astype(int)
            yAbove[i_meas] = thisy
            err_yAbove[i_meas] = err_thisy
            if verbose:
                print thisx, thisy
            if plotalot and not np.isnan(thisy):
                #ax.plot([thisx], [thisy], 'xk')
                ax.errorbar([thisx], [thisy], [err_thisy], fmt='xk')

        # Measure all positions "below" the initial selection:
        yBelow = np.zeros(nBelow,float)
        err_yBelow = np.zeros(nBelow,float)
        lastY = yInit
        for i_meas in range(nBelow):
            guessLoc = xBelow[i_meas], lastY
            thisx, thisy, err_thisy = fitPSF(ec, guessLoc, fitwidth=fitwidth, verbose=verbose-1, medwidth=medwidth, err_ec=err_ec)
            if abs(thisy-lastY)>fitwidth/2:
                thisy = np.nan
            else:
                lastY = thisy.astype(int)
            yBelow[i_meas] = thisy
            err_yBelow[i_meas] = err_thisy
            if verbose:
                print thisx, thisy
            if plotalot and not np.isnan(thisy):
                ax.errorbar([thisx], [thisy], [err_thisy], fmt='xk')
    
        # Stick all the fit positions together:
        yPositions = np.concatenate((yBelow[::-1], [yInit], yAbove))
        err_yPositions = np.concatenate((err_yBelow[::-1], [err_yInit], err_yAbove))
        xPositions = np.concatenate((xBelow[::-1], [xInit], xAbove))

        if verbose:
            message("Measured the following y-positions:")
            message(yPositions)

        theseTraceCoefs = polyfitr(xPositions, yPositions, pord, 3, \
                                       w=1./err_yPositions**2, verbose=verbose)
        orderCoefs[ordernumber,:] = theseTraceCoefs
        # Plot the traces
        if plotalot:
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx)), '-k')
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx))+fitwidth/2, '--k')
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx))-fitwidth/2, '--k')
            ax.axis([0, nx, 0, ny])   
            plt.figure(f.number)
            plt.draw()

        if retsnr:
            position_SNRs.append(yPositions / err_yPositions)
        if retfits:
            xyfits.append((xPositions, yPositions))
    
    # Prepare for exit and return:
    ret = (orderCoefs,)
    if retsnr:
        ret = ret + (position_SNRs,)
    if retfits:
        ret = ret + (xyfits,)
    if len(ret)==1:
        ret = ret[0]
    return  ret


def makeprofile(filename, trace, **kw): #dispaxis=0, fitwidth=20, verbose=False, oversamp=4, nsigma=20, retall=False, neg=False, xylims=None, rn=readnoise, g=gain, extract_radius=10, bkg_radii=[15, 20], bkg_order=0, badpixelmask=None, interp=False):
    """
    Make a spatial profile from a spectrum, given its traced location.
    We interpolate the PSF at each pixel to a common reference frame,
    and then average them.

    filename : str _OR_ 2D numy array
       2D echellogram

    trace : 1D numpy array
        set of polynomial coeficients of order (P-1)

    dispaxis : int
        set dispersion axis: 0 = horizontal and  = vertical
    
    fitwidth : int
        Total width of extracted spectral sub-block. WIll always be
        increased to at least twice the largest value of bkg_radii.

    neg : bool scalar
        set True for a negative spectral trace

    nsigma : scalar
        Sigma-clipping cut for bad pixels (beyond read+photon
        noise). Set it rather high, and feel free to experiment with
        this parameter!

    xylims : 4-sequence
        Extract the given subset of the data array: [xmin, xmax, ymin, ymax]

    retall : bool
        Set True to output several additional parameters (see below)

    rn : scalar
        Read noise (electrons)

    g : scalar
        Detector gain (electrons per data unit)

    bkg_radii : 2-sequence
        Inner and outer radius for background computation and removal;
        measured in pixels from the center of the profile.

    bkg_order : int > 0
        Polynomial order of background trend computed in master spectral profile

    interp : bool
        Whether to (bi-linearly) interpolate each slice to produce a
        precisely-centered spectral profile (according to the input
        'trace').  If False, slices will only be aligned to the
        nearest pixel.


    OUTPUT:
       if retall:
         a spline-function that interpolates pixel locations onto the mean profile

         a stack of data slices

         estimates of the uncertainties

         good pixel flag

         list of splines

       else:
         a spline-function that interpolates pixel locations onto the mean profile    
    """
    from scipy import signal
    #import numpy as np
    # 2010-12-17 10:22 IJC: Added xylims option
    # 2012-03-15 06:34 IJMC: Updated documentation.
    # 2012-04-24 16:43 IJMC: Now properly update gain and readnoise,
    #                        as neccessary.  Much better flagging of
    #                        bad pixels.  Fixed interpolation on RHS.
    # 2012-08-19 17:53 IJMC: Added dispaxis option.
    # 2012-09-05 10:17 IJMC: Made keyword options into a dict.

    global gain
    global readnoise


    # Parse inputs:
    names = ['dispaxis', 'fitwidth', 'verbose', 'oversamp', 'nsigma', 'retall', 'neg', 'xylims', 'rn', 'g', 'extract_radius', 'bkg_radii', 'bkg_order', 'badpixelmask', 'interp']
    defaults = [0, 20, False, 4, 20, False, False, None, readnoise, gain, 10, [15, 20], 0, None, False]
    for n,d in zip(names, defaults):
        #exec('%s = [d, kw["%s"]][kw.has_key("%s")]' % (n, n, n))
        exec('%s = kw["%s"] if kw.has_key("%s") else d' % (n, n, n))

    if fitwidth < (bkg_radii[1]*2):
        fitwidth = bkg_radii[1]*2


    if not g==gain:
        message("Setting gain to " + str(g))
        gain = g

    if not rn==readnoise:
        message("Setting readnoise to " + str(rn))
        readnoise = rn


    if verbose:
        f = plt.figure()
        ax = plt.axes()

    # Check whether we have a filename, or an array:
    if isinstance(filename, np.ndarray):
        ec = filename.copy()
    else:
        try:
            ec = pyfits.getdata(filename)
        except:
            message("Could not open filename %s" % filename)
            return -1

    if neg:
        ec *= -1
        
    if xylims is not None:
        try:
            ec = ec[xylims[0]:xylims[1], xylims[2]:xylims[3]]
            if verbose:
                message("Extracted subset: " + str(xylims))
        except:
            message("Could not extract subset: " + str(xylims))
            return -1

    if badpixelmask is None:
        badpixelmask = np.zeros(ec.shape, dtype=bool)
    else:
        if not hasattr(badpixelmask, 'shape'):
            badpixelmask = pyfits.getdata(badpixelmask)
        if xylims is not None:
            badpixelmask = badpixelmask[xylims[0]:xylims[1], xylims[2]:xylims[3]]




    trace = np.array(trace,copy=True)
    if len(trace.shape)>1:
        if verbose:
            message("Multi-order spectrum input...")
        #rets = []
        rets = [makeprofile(filename, thistrace, dispaxis=dispaxis, fitwidth=fitwidth,verbose=verbose,oversamp=oversamp,nsigma=nsigma,retall=retall, neg=neg, xylims=xylims, rn=rn, g=g, extract_radius=extract_radius, bkg_radii=bkg_radii, bkg_order=bkg_order,interp=interp, badpixelmask=badpixelmask) for thistrace in trace]

        return rets

    if dispaxis==1:
        if verbose: message("Transposing spectra and bad pixel mask...")
        ec = ec.transpose()
        badpixelmask = badpixelmask.transpose()

    if verbose:
        message("Making a spatial profile...")

    ny, nx = ec.shape
    pord = len(trace) - 1
    if verbose:
        message("Echellogram of size (%i,%i)" % (nx,ny))
        message("Polynomial of order %i (i.e., %i coefficients)" % (pord, pord+1))

    xPositions = np.arange(nx)
    yPositions = np.polyval(trace, xPositions)
    yPixels = yPositions.astype(int)

    # Placeholder for a more comprehensive workaround:
    if (yPixels.max()+fitwidth/2)>=ny:
        message("Spectrum may be too close to the upper boundary; try decreasing fitwidth")
    if (yPixels.min()-fitwidth/2)<0:
        message("Spectrum may be too close to the lower boundary; try decreasing fitwidth")
    
    yProfile = np.linspace(-fitwidth/2, fitwidth/2-1,fitwidth*oversamp)
    #yProfile2 = np.linspace(-fitwidth/2, fitwidth/2-1,(fitwidth+1)*oversamp)
    profileStack = np.zeros((nx,fitwidth),float)
    badpixStack  = np.zeros((nx,fitwidth),bool)
    profileSplineStack = np.zeros((nx,(fitwidth)*oversamp),float)
    xProf = np.arange(fitwidth, dtype=float) - fitwidth/2.
    xProf2 = np.arange(ny, dtype=float)


    # Loop through to extract the spectral cross-sections at each pixel:
    for i_pix in range(nx):
        xi = xPositions[i_pix]
        yi = yPixels[i_pix]

        ymin = max(yi-fitwidth/2, 0)
        ymax = min(yi+fitwidth/2, ny)
        # Ensure that profile will always be "fitwidth" wide:
        if (ymax-ymin)<fitwidth and ymin==0:
            ymax = fitwidth
        elif (ymax-ymin)<fitwidth and ymax==ny:
            ymin = ymax - fitwidth

        profile = ec[ymin:ymax,xi]
        if interp:
            profile = np.interp(xProf + yPositions[i_pix], xProf2[ymin:ymax], profile)

        if verbose:
            print 'i_pix, xi, yi, y0, ymin, ymax', i_pix, xi, yi, yPositions[i_pix], ymin, ymax

        try:
            profileStack[i_pix] = profile.copy() * gain
            badpixStack[i_pix] = badpixelmask[ymin:ymax, xi]
        except:
            print "Busted!!!!"
            print "profileStack[i_pix].shape, profile.shape, gain", \
                profileStack[i_pix].shape, profile.shape, gain
            stop

    # Use the profile stack to flag bad pixels in the interpolation process:
    medianProfile = np.median(profileStack,0)
    scaledProfileStack = 0*profileStack
    myMatrix = np.linalg.pinv(np.vstack((medianProfile, np.ones(profileStack.shape[1]))).transpose())
    for ii in range(profileStack.shape[0]):
        these_coefs = np.dot(myMatrix, profileStack[ii])
        scaledProfileStack[ii] = (profileStack[ii] - these_coefs[1]) / these_coefs[0]

    #import pylab as py
    #pdb.set_trace()


    #########
    # Flag all the bad pixels:
    #########

    # Hot Pixels should be 1-pixel wide:
    nFilt = 3
    filteredProfileStack = signal.medfilt2d(profileStack, nFilt)
    errorFiltStack = np.sqrt(filteredProfileStack + readnoise**2)
    errorFiltStack[badpixStack] = errorFiltStack.max() * 1e9

    goodPixels = (True - badpixStack) * np.abs((profileStack - filteredProfileStack) / errorFiltStack) < nsigma
    badPixels = True - goodPixels
    #cleaned_profileStack  = ns.bfixpix(profStack, badpix, retdat=True)

    if False:  # Another (possibly worse?) way to do it:
        stdProfile = stdr(scaledProfileStack, nsigma=nsigma, axis=0)  #profileStack.std(0)
        goodPixels = (scaledProfileStack-medianProfile.reshape(1,fitwidth)) /  \
            stdProfile.reshape(1,fitwidth)<=nsigma
        goodPixels *= profileStack>=0

    goodColumns = np.ones(goodPixels.shape[0],bool)
    errorStack = np.sqrt(np.abs(profileStack + readnoise**2))
    errorStack[badPixels] = profileStack[goodPixels].max() * 1e9
    
    # Loop through and fit weighted splines to each cross-section
    for i_pix in range(nx):
        xi = xPositions[i_pix]
        yi = yPixels[i_pix]
        ymin = max(yi-fitwidth/2, 0)
        ymax = min(yi+fitwidth/2, ny)
        # Ensure that profile will always be "fitwidth" wide:
        if (ymax-ymin)<fitwidth and ymin==0:
            ymax = fitwidth
        elif (ymax-ymin)<fitwidth and ymax==ny:
            ymin = ymax - fitwidth

        profile = profileStack[i_pix]
        index = goodPixels[i_pix]

        ytemp = np.arange(ymin,ymax)-yPositions[i_pix]
        if verbose>1: 
            print "ii, index.sum()>>",i_pix,index.sum()
            print "xi, yi>>",xi,yi
        if verbose: 
            print "ymin, ymax>>",ymin,ymax
            print "ytemp.shape, profile.shape, index.shape>>",ytemp.shape, profile.shape, index.shape
        if index.sum()>fitwidth/2:
            ydata_temp = np.concatenate((ytemp[index], ytemp[index][-1] + np.arange(1,6)))
            pdata_temp = np.concatenate((profile[index], [profile[index][-1]]*5))
            profileSpline = interpolate.UnivariateSpline(ydata_temp, pdata_temp, k=3.0,s=0.0)
            profileSplineStack[i_pix,:] = profileSpline(yProfile)
        else:
            if verbose: message("not enough good pixels in segment %i" % i_pix)
            profileSplineStack[i_pix,:] = 0.0
            goodColumns[i_pix] = False

    finalYProfile = np.arange(fitwidth)-fitwidth/2
    finalSplineProfile = np.median(profileSplineStack[goodColumns,:],0)
    ## Normalize the absolute scaling of the final Profile
    #xProfile = np.arange(-50.*fitwidth, 50*fitwidth)/100.
    #profileParam = fitGaussian(finalSplineProfile, verbose=True)
    #profileScaling = profileParam[0]*.01
    #print 'profileScaling>>', profileScaling

    backgroundAperture = (np.abs(yProfile) > bkg_radii[0]) * (np.abs(yProfile) < bkg_radii[1])
    #pdb.set_trace()
    backgroundFit = np.polyfit(yProfile[backgroundAperture], finalSplineProfile[backgroundAperture], bkg_order)

    extractionAperture = (np.abs(yProfile) < extract_radius)
    normalizedProfile = finalSplineProfile - np.polyval(backgroundFit, yProfile)
    normalizedProfile *= oversamp / normalizedProfile[extractionAperture].sum()

    finalSpline = interpolate.UnivariateSpline(yProfile, 
                      normalizedProfile,   k=3.0, s=0.0)

    

    if verbose:
        ax.plot(yProfile, finalSplineProfile, '--',linewidth=2)
        plt.draw()
    
    if retall==True:
        ret =  finalSpline, profileStack, errorStack, goodPixels,profileSplineStack
    else:
        ret = finalSpline

    return ret

def makeFitModel(param, spc,profile, xtemp=None):
    scale, background, shift = param[0:3]
    npix = len(spc)
    if xtemp is None:
        xtemp = np.arange(-npix/2,npix/2,dtype=float)
    model = scale*profile(xtemp-shift)+background
    return model


def profileError(param, spc, profile, w, xaxis=None, retresidual=False):
    "Compute the chi-squared error on a spectrum vs. a profile "
    # 2012-04-25 12:59 IJMC: Slightly optimised (if retresidual is False)

        #scale, background, shift = param[0:3]
        #npix = len(spc)
        #xtemp = np.arange(-npix/2,npix/2,dtype=float)
    interpolatedProfile = makeFitModel(param,spc,profile, xtemp=xaxis) #scale*profile(xtemp-shift)+background
    #wresiduals = 
    #wresiduals = (w*((spc-interpolatedProfile))**2).sum()
    if retresidual:
        ret = (w**0.5) * (spc - interpolatedProfile)
    else:
        ret = (w * (spc - interpolatedProfile)**2).sum()

    return ret


def fitprofile(spc, profile, w=None,verbose=False, guess=None, retall=False):
    """Fit a spline-class spatial profile to a spectrum cross-section
    """
    # 2010-11-29 14:14 IJC: Added poor attempt at catching bad pixels

    import analysis as an
    import numpy as np

    spc = np.array(spc)
    try:
        npix = len(spc)
    except:
        npix = 0

    if w is None:
        w = np.ones(spc.shape)
    else:
        w[True - np.isfinite(w)] = 0.

    if guess is None:
        guessParam = [1., 0., 0.]
    else:
        guessParam = guess

    if verbose>1:
        #message("profileError>>"+str(profileError))
        #message("guessParam>>"+str(guessParam))
        #message("spc>>"+str(spc))
        message("w>>"+str(w))

    good_index = w<>0.0
    xaxis = np.arange(-npix/2,npix/2,dtype=float) # take outside of fitting loop
    bestFit = optimize.fmin_powell(profileError, guessParam, args=(spc,profile,w * good_index, xaxis),disp=verbose, full_output=True)
    best_bic = bestFit[1] + len(bestFit[0]) * np.log(good_index.sum())
    keepCleaning = True

    if False:
        while keepCleaning is True:
            if verbose: print "best bic is: ", best_bic
            #w_residuals = profileError(bestFit[0], spc, profile, w, retresidual=True)
            #good_values = (abs(w_residuals)<>abs(w_residuals).max())
            diffs = np.hstack(([0], np.diff(spc)))
            d2 = diffs * (-np.ones(len(spc)))**np.arange(len(spc))
            d3 = np.hstack((np.vstack((d2[1::], d2[0:len(spc)-1])).mean(0), [0]))
            good_values = abs(d3)<>abs(d3).max()
            good_index *= good_values
            xaxis = np.arange(-npix/2,npix/2,dtype=float) # take outside of fitting loop
            latestFit = optimize.fmin_powell(profileError, guessParam, args=(spc,profile,w * good_index, xaxis),disp=verbose, full_output=True)
            latest_bic = latestFit[1] + len(latestFit[0]) * np.log(good_index.sum())
            if latest_bic < best_bic:
                best_bic = latest_bic
                bestFit = latestFit
                keepCleaning = True
            else:
                keepCleaning = False

            if good_index.any() is False:
                keepCleaning = False
            #good_index = good_index * an.removeoutliers(w_residuals, 5, retind=True)[1]

    if verbose:
        message("initial guess chisq>>%3.2f" % profileError(guessParam, spc, profile,w,xaxis))
        message("final fit chisq>>%3.2f" % profileError(bestFit[0], spc, profile,w, xaxis))

    if retall:
        ret = bestFit
    else:
        ret = bestFit[0]

    return ret

    

def fitProfileSlices(splineProfile, profileStack, stdProfile, goodPixels,verbose=False, bkg_radii=None, extract_radius=None):
    """Fit a given spatial profile to a spectrum

    Helper function for :func:`extractSpectralProfiles`

    """

    npix, fitwidth = profileStack.shape
    stdProfile = stdProfile.copy()
    goodPixels[np.isnan(stdProfile) + stdProfile==0] = False
    stdProfile[True - goodPixels] = 1e9
    #varPData = stdProfile**2

    if extract_radius is None:
        extract_radius = fitwidth

    x2 = np.arange(npix)
    x = np.arange(-fitwidth/2, fitwidth/2)
    extractionAperture = np.abs(x) < extract_radius
    nextract = extractionAperture.sum()

    fitparam = np.zeros((npix, 3),float)
    fitprofiles = np.zeros((npix,nextract),float)
    tempx = np.arange(-fitwidth/2,fitwidth/2,dtype=float)
    
    # Start out by getting a rough estimate of any additional bad
    # pixels (to flag):

    #backgroundAperture = (np.abs(x) > bkg_radii[0]) * (np.abs(x) < bkg_radii[1])
    #background = an.wmean(profileStack[:, backgroundAperture], (goodPixels/varPData)[:, backgroundAperture], axis=1)
    #badBackground = True - np.isfinite(background)
    #background[badBackground] = 0.

    #subPData = profileStack - background
    #standardSpectrum = an.wmean(subPData[:, extractionAperture], (goodPixels/varPData)[:,extractionAperture], axis=1) * extractionAperture.sum()

    if False:
        varStandardSpectrum = an.wmean(varPData[:, extractionAperture], goodPixels[:, extractionAperture], axis=1) * extractionAperture.sum()

        badSpectrum = True - np.isfinite(standardSpectrum)
        standardSpectrum[badSpectrum] = 1.
        varStandardSpectrum[badSpectrum] = varStandardSpectrum[True - badSpectrum].max() * 1e9

        mod = background + standardSpectrum * splineProfile(x) * (splineProfile(x).sum() / splineProfile(x[extractionAperture]).sum())




    anyNewBadPixels = True
    nbp0 = goodPixels.size - goodPixels.sum()
    gp0 = goodPixels.copy()

    while anyNewBadPixels is True:
        for ii in range(npix):
            if verbose>1:
                print "*****In fitProfileSlices, pixel %i/%i" % (ii+1, npix)
            if ii>0: # take the median of the last several guesses
                gindex = [max(0,ii-2), min(npix, ii+2)]
                guess = np.median(fitparam[gindex[0]:gindex[1]], 0)
            else:
                guess = None

            if verbose>1:
                message("ii>>%i"%ii)
                message("goodPixels>>"+str(goodPixels.astype(float)[ii,:]))
                message("stdProfile>>"+str(stdProfile[ii,:]))

            thisfit = fitprofile(profileStack[ii,extractionAperture], splineProfile, (goodPixels.astype(float)/stdProfile**2)[ii,extractionAperture],verbose=verbose, guess=guess, retall=True)
            bestfit = thisfit[0]
            bestchi = thisfit[1]

            if verbose>1:
                print "this fit: ",bestfit
            fitparam[ii,:] = bestfit
            fitprofiles[ii,:] = makeFitModel(bestfit, profileStack[ii,extractionAperture],splineProfile) 
            if verbose>1:
                print "finished pixel %i/%i" % (ii+1,npix)
                
            
        deviations = (fitprofiles - profileStack[:, extractionAperture])/stdProfile[:, extractionAperture]
        for kk in range(nextract):
            kk0 = extractionAperture.nonzero()[0][kk]
            thisdevfit = an.polyfitr(x2, deviations[:, kk], 1, 3)
            theseoutliers = np.abs((deviations[:, kk] - np.polyval(thisdevfit, x2))/an.stdr(deviations, 3)) > 5
            goodPixels[theseoutliers, kk0] = False
            
        nbp = goodPixels.size - goodPixels.sum()
        if nbp <= nbp0:
            anyNewBadPixels = False
        else:
            nbp0 = nbp

    for ii in range(fitparam.shape[1]):
        fitparam[:,ii] = bfixpix(fitparam[:,ii], goodPixels[:,extractionAperture].sum(1) <= (nextract/2.), retdat=True)

    return fitparam, fitprofiles






def extractSpectralProfiles(args, **kw):
    """
    Extract spectrum 

    :INPUTS:
       args : tuple or list
           either a tuple of (splineProfile, profileStack, errorStack,
           profileMask), or a list of such tuples (from makeprofile).
             
    :OPTIONS:
       bkg_radii : 2-sequence
         inner and outer radii to use in computing background

       extract_radius : int
         radius to use for both flux normalization and extraction

    :RETURNS:
       3-tuple:
          [0] -- spectrum flux (in electrons)

          [1] -- background flux

          [2] -- best-fit pixel shift

    :EXAMPLE:
      ::
        out = spec.traceorders('aug16s0399.fits',nord=7)

        out2 = spec.makeprofile('aug16s0399.fits',out,retall=True)

        out3 = spec.extractSpectralProfiles(out2)

    :SEE_ALSO:
      :func:`optimalExtract`

    :NOTES:
      Note that this is non-optimal with highly tilted or curved
      spectra, for the reasons described by Marsh (1989) and Mukai
      (1990).
    """
    # 2010-08-24 21:24 IJC: Updated comments

    if kw.has_key('verbose'):
        verbose = kw['verbose']
    else:
        verbose = False

    if kw.has_key('bkg_radii'):
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if kw.has_key('extract_radius'):
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10


    #print 'starting...', args[0].__class__, isinstance(args[0],tuple)#, len(*arg)
    fitprofiles = []
    if isinstance(args,list): # recurse with each individual set of arguments
        #nord = len(arg[0])
        spectrum = []
        background = []
        pixshift = []
        for ii, thesearg in enumerate(args):
            if verbose: print "----------- Iteration %i/%i" % (ii+1, len(args)) 
            #print 'recursion loop', thesearg.__class__, verbose, kw
            tempout = extractSpectralProfiles(thesearg, **kw)
            spectrum.append(tempout[0])
            background.append(tempout[1])
            pixshift.append(tempout[2])
            fitprofiles.append(tempout[3])
        spectrum = np.array(spectrum)
        background = np.array(background)
        pixshift = np.array(pixshift)
    else:
        #print 'actual computation', args.__class__, verbose, kw
        splineProfile = args[0]
        profileStack = args[1]
        errorStack = args[2]
        profileMask = args[3]


        #print splineProfile.__class__, profileStack.__class__, errorStack.__class__, profileMask.__class__
        #print profileStack.shape, errorStack.shape, profileMask.shape
        fps = dict()
        if kw.has_key('verbose'): fps['verbose'] = kw['verbose']
        if kw.has_key('bkg_radii'): fps['bkg_radii'] = kw['bkg_radii']
        if kw.has_key('extract_radius'): fps['extract_radius'] = kw['extract_radius']
        fitparam, fitprofile = fitProfileSlices(splineProfile, profileStack, errorStack, profileMask, **fps)
        spectrum = fitparam[:,0]
        background = fitparam[:,1]
        pixshift = fitparam[:,2]
        fitprofiles.append(fitprofile)

    ret = (spectrum, background, pixshift, fitprofiles)

    return  ret



def gaussint(x):
    """ 
    :PURPOSE:
        Compute the integral from -inf to x of the normalized Gaussian

    :INPUTS:
        x : scalar
            upper limit of integration

    :NOTES:
        Designed to copy the IDL function of the same name.
        """
    # 2011-10-07 15:41 IJMC: Created

    from scipy.special import erf

    scalefactor = 1./np.sqrt(2)
    return 0.5 + 0.5 * erf(x * scalefactor)

def slittrans(*varargin):
    """+
     :NAME:
         slittrans

     :PURPOSE:
         Compute flux passing through a slit assuming a gaussian PSF.

     :CATEGORY:
         Spectroscopy

     :CALLING SEQUENCE:
         result = slittrans(width,height,fwhm,xoffset,yoffset,CANCEL=cancel)

     :INPUTS:
         width   - Width of slit.
         height  - Height of slit.
         fwhm    - Full-width at half-maximum of the gaussian image.
         xoffset - Offset in x of the image from the center of the slit.
         yoffset - Offset in y of the image from the center of the slit.

         Note: the units are arbitrary but they must be the same across
               all of the input quantities.  

     KEYWORD PARAMETERS:
         CANCEL - Set on return if there is a problem

     OUTPUTS:
         Returned is the fraction of the total gaussian included in the slit.

     EXAMPLE:
         result = slittrans(0.3,15,0.6,0,0)

         Computes the fraction of the flux transmitted through a slit 
         0.3x15 arcseconds with a PSF of 0.6 arcseconds FWHM.  The PSF is
         centered on the slit.

     MODIFICATION HISTORY:
         Based on M Buie program,  1991 Mar., Marc W. Buie, Lowell Observatory
         Modified 2000 Apr., M. Cushing to include y offsets.
         2011-10-07 15:45 IJMC: Converted to Python
         2011-11-14 16:29 IJMC: Rewrote to use :func:`erf` rather than
                                :func:`gaussint`
         """
    #function slittrans,width,height,fwhm,xoffset,yoffset,CANCEL=cancel

    from scipy.special import erf

    cancel = 0
    n_params = len(varargin)

    if n_params <> 5:
        print 'Syntax - result = slittrans(width,height,fwhm,xoffset,yoffset,',
        print '                            CANCEL=cancel)'
        cancel = 1
        return cancel

    width, height, fwhm, xoffset, yoffset = varargin[0:5]

    #cancel = cpar('slittrans',width,1,'Width',[2,3,4,5],0)
    #if cancel then return,-1
    #cancel = cpar('slittrans',height,2,'Height',[2,3,4,5],0)
    #if cancel then return,-1
    #cancel = cpar('slittrans',fwhm,3,'FWHM',[2,3,4,5],0)
    #if cancel then return,-1
    #cancel = cpar('slittrans',xoffset,4,'Xoffset',[2,3,4,5],0)
    #if cancel then return,-1
    #cancel = cpar('slittrans',yoffset,5,'Yoffset',[2,3,4,5],0)
    #if cancel then return,-1

    #  Go ahead

    a = 0.5 * width 
    b = 0.5 * height 

    #s = fwhm/(np.sqrt(8.0*np.log(2.0)))  
    #slit = ( 1.0 - gaussint( -(a+xoffset)/s ) - gaussint( -(a-xoffset)/s ) ) * \
    #       ( 1.0 - gaussint( -(b+yoffset)/s ) - gaussint( -(b-yoffset)/s ) )

    invs2 = (np.sqrt(4.0*np.log(2.0)))/fwhm    # sigma * sqrt(2)
    
    slit4 = 0.25 * (  (erf( -(a+xoffset)*invs2) + erf( -(a-xoffset)*invs2) ) ) * \
                   (  (erf( -(b+yoffset)*invs2) + erf( -(b-yoffset)*invs2) ) )
       

    #print 'smin/max,',slit.min(), slit5.max()
    #print 'smin/max,',slit4.min(), slit4.max()

     
    return slit4


def atmosdisp(wave, wave_0, za, pressure, temp, water=2., fco2=0.0004, obsalt=0.):
    """:NAME:
         atmosdisp

     PURPOSE:
         Compute the atmosperic dispersion relative to lambda_0.     

     CATEGORY:
         Spectroscopy

     CALLING SEQUENCE:
         result = atmosdisp(wave,wave_0,za,pressure,temp,[water],[obsalt],$
                            CANCEL=cancel)

     INPUTS:
         wave     - wavelength in microns
         wave_0   - reference wavelength in microns
         za       - zenith angle of object [in degrees]
         pressure - atmospheric pressure in mm of Hg
         temp     - atmospheric temperature in degrees C

     OPTIONAL INPUTS:
         water    - water vapor pressure in mm of Hg.
         fco2     - relative concentration of CO2 (by pressure)
         obsalt    - The observatory altitude in km.

     KEYWORD PARAMETERS:
         CANCEL - Set on return if there is a problem

     OUTPUTS:
         Returns the atmospheric disperion in arcseconds.      

     PROCEDURE:
         Computes the difference between the dispersion at two
         wavelengths.  The dispersion for each wavelength is derived from
         Section 4.3 of Green's "Spherical Astronomy" (1985).

     EXAMPLE:



     MODIFICATION HISTORY:
         2000-04-05 - written by M. Cushing, Institute for Astronomy, UH
         2002-07-26 - cleaned up a bit.
         2003-10-20 - modified formula - WDV
         2011-10-07 15:51 IJMC: Converted to Python, with some unit conversions
    -"""

    #function atmosdisp,wave,wave_0,za,pressure,temp,water,obsalt,CANCEL=cancel
    from nsdata import nAir

    # Constants

    mmHg2pa = 101325./760.      # Pascals per Torr (i.e., per mm Hg)
    rearth = 6378.136e6 #6371.03	# mean radius of earth in km [Allen's]
    hconst = 2.926554e-2	# R/(mu*g) in km/deg K,  R=gas const=8.3143e7
                                    # mu=mean mol wght. of atm=28.970, g=980.665
    tempk  = temp + 273.15
    pressure_pa = pressure * mmHg2pa
    water_pp = water/pressure   # Partial pressure
    hratio = (hconst * tempk)/(rearth + obsalt)

    # Compute index of refraction

    nindx  = nAir(wave,P=pressure_pa,T=tempk,pph2o=water_pp, fco2=fco2)
    nindx0 = nAir(wave_0,P=pressure_pa,T=tempk,pph2o=water_pp, fco2=fco2)

    # Compute dispersion

    acoef  = (1. - hratio)*(nindx - nindx0)
    bcoef  = 0.5*(nindx*nindx - nindx0*nindx0) - (1. + hratio)*(nindx - nindx0)

    tanz   = np.tan(np.deg2rad(za))
    disp   = 206265.*tanz*(acoef + bcoef*tanz*tanz)

    #print nindx
    #print nindx0
    #print acoef
    #print bcoef
    #print tanz
    #print disp
    return disp



def parangle(HA, DEC, lat):
    """
    +
     NAME:
         parangle

     PURPOSE:
         To compute the parallactic angle at a given position on the sky.

     CATEGORY:
         Spectroscopy    

     CALLING SEQUENCE:
         eta, za = parangle(HA, DEC, lat)

     INPUTS:
         HA  - Hour angle of the object, in decimal hours (0,24)
         DEC - Declination of the object, in degrees
         lat - The latitude of the observer, in degrees

     KEYWORD PARAMETERS:
         CANCEL - Set on return if there is a problem

     OUTPUTS:
         eta - The parallactic angle
         za  - The zenith angle

     PROCEDURE:
         Given an objects HA and DEC and the observers latitude, the
         zenith angle and azimuth are computed.  The law of cosines
         then gives the parallactic angle.  

     EXAMPLE:
         NA


     MODIFICATION HISTORY:
         2000-04-05 - written by M. Cushing, Institute for Astronomy,UH
         2002-08-15 - cleaned up a bit.
         2003-10-21 - changed to pro; outputs zenith angle as well - WDV
         2011-10-07 17:58 IJMC: Converted to Python
-"""

  #pro parangle, HA, DEC, lat, eta, za, CANCEL=cancel

    cancel = 0
    d2r = np.deg2rad(1.)
    r2d = np.rad2deg(1.)

    #  If HA equals zero then it is easy.
    HA = HA % 24
    #  Check to see if HA is greater than 12.
    if hasattr(HA, '__iter__'):
        HA = np.array(HA, copy=False)
        HAind = HA > 12
        if HAind.any():
            HA[HAind] = 24. - HA[HAind]
    else:
        if HA>12.:
            HA = 24. - HA

    HA = HA*15.

    #  Determine Zenith angle and Azimuth
    cos_za = np.sin(lat*d2r) * np.sin(DEC*d2r) + \
             np.cos(lat*d2r) * np.cos(DEC*d2r) * np.cos(HA*d2r)
    za     = np.arccos(cos_za) * r2d
    cos_az = (np.sin(DEC*d2r) - np.sin(lat*d2r)*np.cos(za*d2r)) / \
             (np.cos(lat*d2r) * np.sin(za*d2r))
    az     = np.arccos(cos_az)*r2d

    if hasattr(az, '__iter__'):
        azind = az==0
        if azind.any() and DEC<lat:
            az[azind] = 180.
    else:
        if az==0. and DEC<lat:
            az = 180.

    tan_eta = np.sin(HA*d2r)*np.cos(lat*d2r) / \
              (np.cos(DEC*d2r)*np.sin(lat*d2r) - \
               np.sin(DEC*d2r)*np.cos(lat*d2r)*np.cos(HA*d2r))
    eta     = np.arctan(tan_eta)*r2d

    if hasattr(eta, '__iter__'):
        etaind = eta < 0
        ezind = (eta==0) * (az==0)
        zaind = za > 90
        if etaind.any():
            eta[etaind] += 180.
        elif ezind.any():
            eta[ezind] = 180.
        if zaind.any():
            eta[zaind] = np.nan
    else:
        if eta < 0:
            eta += 180.
        elif eta==0 and az==0:
            eta = 180.
        if za>90:
            eta = np.nan

    HA = HA/15.0

    return eta, za

def lightloss(objfile, wguide, seeing, press=None, water=None, temp=None, fco2=None, wobj=None, dx=0, dy=0, retall=False):
    """    +
     NAME:
         lightloss

     PURPOSE:
         To determine the slit losses from a spectrum.

     CATEGORY:
         Spectroscopy     

     CALLING SEQUENCE:
         ### TBD lightloss, obj, std, wguide, seeing, out, CANCEL=cancel

     INPUTS:
         obj    - FITS file of the object spectrum
         wguide - wavelength at which guiding was done
         seeing - seeing FWHM at the guiding wavelength

     OPTIONAL INPUTS:
         press -  mm Hg typical value (615 for IRTF, unless set)
         water -  mm Hg typical value (2 for IRTF, unless set)
         temp  -  deg C typical value (0 for IRTF, unless set)
         fco2  -  relative concentration of CO2 (0.004, unless set)
         wobj  -  wavelength scale for data
         dx    -  horizontal offset of star from slit center
         dy    -  vertical offset of star from slit center
         retall-  whether to return much diagnostic info, or just lightloss.

     NOTES:
         'seeing', 'dx', and 'dy' should all be in the same units, and
         also the same units used to define the slit dimensions in the
         obj FITS file header

     KEYWORD PARAMETERS:
         CANCEL - Set on return if there is a problem

     OUTPUTS:
         array : fractional slit loss at each wavelength value
        OR:
         tuple of arrays: (slitloss, disp_obj, diff, fwhm, dx_obj, dy_obj)


     PROCEDURE:
         Reads a Spextool FITS file.

     EXAMPLE:
         None

     REQUIREMENTS:
         :doc:`phot`
         :doc:`pyfits`

     MODIFICATION HISTORY:
         2003-10-21 - Written by W D Vacca 
         2011-10-07 20:19 IJMC: Converted to Python, adapted for single objects
         2011-10-14 14:01 IJMC: Added check for Prism mode (has
                    different slit dimension keywords) and different
                    pyfits header read mode.
         2011-11-07 15:53 IJMC: Added 'retall' keyword
         
    -
    """
    #pro lightloss, objfile, stdfile, wguide, seeing, outfile,CANCEL=cancel


    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from phot import hms, dms

    r2d = np.rad2deg(1)

    # --- Open input files

    obj    = pyfits.getdata(objfile)
    objhdu = pyfits.open(objfile)
    #objhdr = pyfits.getheader(objfile)
    objhdu.verify('silentfix')
    objhdr = objhdu[0].header
    if wobj is None:
        try:
            wobj   = obj[:,0,:]
            #fobj   = obj[:,1,:]
            #eobj   = obj[:,2,:]
        except:
            wobj   = obj[0,:]
            #fobj   = obj[1,:]
            #eobj   = obj[2,:]

    # --- Read header keywords

    tele       = objhdr['TELESCOP']
    
    try:
        slitwd     = objhdr['SLTW_ARC']
        slitht     = objhdr['SLTH_ARC']
    except: # for SpeX prism mode:
        slitwd, slitht = map(float, objhdr['SLIT'].split('x'))

    #xaxis      = objhdr['XAXIS']
    #xunits     = objhdr['XUNITS']
    #yaxis      = objhdr['YAXIS']
    #yunits     = objhdr['YUNITS']

    posang_obj = objhdr['POSANGLE']
    HA_objstr  = objhdr['HA']
    DEC_objstr = objhdr['DEC']

    # --- Process keywords

    #coord_str  = HA_objstr + ' ' + DEC_objstr
    #get_coords, coords, InString=coord_str
    HA_obj, DEC_obj = dms(HA_objstr), dms(DEC_objstr)

    if posang_obj<0.0:
        posang_obj += 180.
    if posang_obj >= 180.0:
        posang_obj -= 180.0

    if tele=='NASA IRTF':
       obsdeg  = 19.0
       obsmin  = 49.0
       obssec  = 34.39
       obsalt  = 4.16807    # observatory altitude in km
       teldiam = 3.0        # diameter of primary in meters
       if press is None:
           press   = 615.0 	    # mm Hg typical value
       if water is None:
           water   = 2.0        # mm Hg typical value
       if temp is None:
           temp    = 0.0        # deg C typical value   
    else:
       print 'Unknown Telescope - stopping!'
       return

    if fco2 is None:
        fco2 = 0.0004

    obslat  = dms('%+02i:%02i:%02i' % (obsdeg, obsmin, obssec))

    # --- Compute Parallactic Angle
    pa_obj, za_obj = parangle(HA_obj, DEC_obj, obslat)
    dtheta_obj = posang_obj - pa_obj
    #print posang_obj, pa_obj, dtheta_obj, za_obj

    #print posang_obj, pa_obj, dtheta_obj, HA_obj

    # --- Compute Differential Atmospheric Dispersion
    disp_obj = atmosdisp(wobj, wguide, za_obj, press, temp, \
                             water=water, obsalt=obsalt, fco2=fco2)

    # --- Compute FWHM at each input wavelength
    diff     = 2.0*1.22e-6*3600.0*r2d*(wobj/teldiam)	# arcsec
    fwhm     = (seeing*(wobj/wguide)**(-0.2))
    fwhm[fwhm < diff] = diff[fwhm < diff]

    # --- Compute Relative Fraction of Light contained within the slit
    dx_obj   = (disp_obj*np.sin(dtheta_obj/r2d)) + dx
    dy_obj   = (disp_obj*np.cos(dtheta_obj/r2d)) + dy

    slitloss  = slittrans(slitwd,slitht,fwhm,dx_obj,dy_obj)

    #debug_check = lightloss2(wobj, slitwd, slitht, posang_obj/57.3, pa_obj/57.3, za_obj/57.3, wguide, seeing, retall=retall)

    if retall:
        return  (slitloss, disp_obj, diff, fwhm, dx_obj, dy_obj)
    else:
        return slitloss



def lightloss2(wobj, slitwd, slitht, slitPA, targetPA, zenith_angle, wguide, seeing, press=615., water=2., temp=0., fco2=0.004, obsalt=4.16807, teldiam=3., dx=0, dy=0, retall=False, ydisp=None, xdisp=None, fwhm=None):
    """    +
     NAME:
         lightloss2

     PURPOSE:
         To determine the slit losses from an observation (no FITS file involved)

     CATEGORY:
         Spectroscopy     

     CALLING SEQUENCE:
         ### TBD lightloss, obj, std, wguide, seeing, out, CANCEL=cancel

     INPUTS:
         wobj  -  wavelength scale for data
         slitwd - width of slit, in arcsec
         slitht - height of slit, in arcsec
         slitPA - slit Position Angle, in radians
         targetPA - Parallactic Angle at target, in radians
         zenith_angle - Zenith Angle, in radians
         wguide - wavelength at which guiding was done
         seeing - seeing FWHM at the guiding wavelength

     OPTIONAL INPUTS:
         press -  mm Hg typical value (615, unless set)
         water -  mm Hg typical value (2 , unless set)
         temp  -  deg C typical value (0 , unless set)
         fco2  -  relative concentration of CO2 (0.004, unless set)
         obsalt-  observatory altitude, in km
         teldiam- observatory limiting aperture diameter, in m
         dx    -  horizontal offset of star from slit center
         dy    -  vertical offset of star from slit center
         retall-  whether to return much diagnostic info, or just lightloss.

         ydisp - The position of the spectrum in the slit at all
                 values of wobj.  This should be an array of the same
                 size as wobj, with zero corresponding to the vertical
                 middle of the slit and positive values tending toward
                 zenith.  In this case xdisp will be computed as XXXX
                 rather than from the calculated atmospheric
                 dispersion; dx and dy will also be ignored.

         fwhm - Full-width at half-maximum of the spectral trace, at
                all values of wobj.  This should be an array of the
                same size as wobj, measured in arc seconds.


     NOTES:
         'slitwidth', 'slitheight', 'seeing', 'dx', 'dy', and 'fwhm'
         (if used) should all be in the same units: arc seconds.

     OUTPUTS:
         array : fractional slit loss at each wavelength value
        OR:
         tuple of arrays: (slitloss, disp_obj, diff, fwhm, dx_obj, dy_obj)


     PROCEDURE:
         All input-driven.  For the SpeXTool-version analogue, see
         :func:`lightloss`

     EXAMPLE:
         import numpy as np
         import spec
         w = np.linspace(.5, 2.5, 100) # Wavelength, in microns
         d2r = np.deg2rad(1.)
         #targetPA, za = spec.parangle(1.827, 29.67*d2r, lat=20.*d2r)
         targetPA, za = 105.3, 27.4
         slitPA = 90. * d2r

         spec.lightloss2(w, 3., 15., slitPA, targetPA*d2r, za*d2r, 2.2, 1.0)

     REQUIREMENTS:
         :doc:`phot`

     MODIFICATION HISTORY:
         2003-10-21 - Written by W D Vacca 
         2011-10-07 20:19 IJMC: Converted to Python, adapted for single objects
         2011-10-14 14:01 IJMC: Added check for Prism mode (has
                    different slit dimension keywords) and different
                    pyfits header read mode.
         2011-11-07 15:53 IJMC: Added 'retall' keyword
         2011-11-07 21:17 IJMC: Cannibalized from SpeXTool version
         2011-11-25 15:06 IJMC: Added ydisp and fwhm options.
    -
    """
    #pro lightloss, objfile, stdfile, wguide, seeing, outfile,CANCEL=cancel


    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from phot import hms, dms

    r2d = np.rad2deg(1)
    d2r = np.deg2rad(1.)

    if slitPA<0.0:
        slitPA += np.pi
    if slitPA >= np.pi:
        slitPA -= np.pi

    # --- Compute Parallactic Angle
    dtheta_obj = slitPA - targetPA
    #print slitPA, targetPA, dtheta_obj, zenith_angle

    # --- Compute FWHM at each input wavelength
    diff     = 2.0*1.22e-6*3600.0*r2d*(wobj/teldiam)	# arcsec
    if fwhm is None:
        fwhm     = (seeing*(wobj/wguide)**(-0.2))

    fwhm[fwhm < diff] = diff[fwhm < diff]

    if ydisp is None or xdisp is None:
        # --- Compute Differential Atmospheric Dispersion
        disp_obj = atmosdisp(wobj, wguide, zenith_angle*r2d, press, temp, \
                                 water=water, obsalt=obsalt, fco2=fco2)


        if ydisp is None:
            dy_obj   = (disp_obj*np.cos(dtheta_obj)) + dy
        else:
            dy_obj = ydisp

        if xdisp is None:
            dx_obj   = (disp_obj*np.sin(dtheta_obj)) + dx
        else:
            dx_obj = xdisp

    else:
        dx_obj = np.array(xdisp, copy=False)
        dy_obj = np.array(ydisp, copy=False)
        if retall:
            disp_obj = (dy_obj - dy) / np.cos(dtheta_obj)

#    if xdisp is None and ydisp is None:
#        guide_index = np.abs(wobj - wguide).min() == np.abs(wobj-wguide)
#        dy = ydisp[guide_index].mean()
#        dy_obj = ydisp
#        dx_obj = (dy_obj - dy) * np.tan(dtheta_obj)
        
    # --- Compute Relative Fraction of Light contained within the slit
    slitloss  = slittrans(slitwd, slitht, fwhm, dx_obj, dy_obj)

    if retall:
        return slitloss, disp_obj, diff, fwhm, dx_obj, dy_obj
    else:
        return slitloss


def humidpressure(RH, T):
    """    +
     NAME:
         humidpressure

     PURPOSE:
         To convert relative humidity into a H2O vapor partial pressure

     CATEGORY:
         Spectroscopy     

     CALLING SEQUENCE:
         humidpressure(RH, 273.15)

     INPUTS:
         RH  - relative humidity, in percent
         T   - temperature, in Kelvin

     OUTPUTS:
         h2o_pp : water vapor partial pressure, in Pascals

     PROCEDURE:
         As outlined in Butler (1998): "Precipitable Water at KP", MMA
         Memo 238 (which refers in turn to Liebe 1989, "MPM - An
         Atmospheric Millimeter-Wave Propagation Model").  Liebe
         claims that this relation has an error of <0.2% from -40 C to
         +40 C.

     EXAMPLE:
         None

     MODIFICATION HISTORY:
         2011-10-08 17:08 IJMC: Created.
    -
    """

    # units of Pa
    #theta =  300./T
    #return 2.408e11 * RH * np.exp(-22.64*theta) * (theta*theta*theta*theta)

    theta_mod = 6792./T  # theta * 22.64
    return 9.1638 * RH * np.exp(-theta_mod) * \
        (theta_mod*theta_mod*theta_mod*theta_mod)


def runlblrtm(lamrange, pwv=2., zang=0., alt=4.2, co2=390, res=200, dotrans=True, dorad=False,
              pwv_offset=4.,
              verbose=False, _save='/Users/ianc/temp/testatmo.mat',
              _wd='/Users/ianc/proj/atmo/aerlbl_v12.0_package/caltech_wrapper_v02/',
              scriptname='runtelluric.m', 
              command = '/Applications/Octave.app/Contents/Resources/bin/octave'):
    """
    Run LBLRTM to compute atmospheric transmittance and/or radiance.

    :INPUTS:
      lamrange : 2-sequence
        approximate minimum and maximum wavelengths

    :OPTIONS:
      pwv : float
        mm of Precipitable Water Vapor above observation site.  If
        negative, then the value abs(pwv_offset-pwv) will be used instead.

      pwv_offset : float
        Only used if (pwv < 0); see above for description.
        
      zang : float
        observation angle from zenith, in degrees

      alt : float
        Observation elevation, in km.

      co2 : float
        CO2 concentration in ppm by volume.  Concentration is assumed
        to be uniform throughout the atmosphere.

      res : float
        approximate spectral resolution desired
      
      dotrans : bool
        compute atmospheric transmittance

      dorad : bool
        compute atmospheric radiance.  NOT CURRENTLY WORKING

      _save : str
        path where temporary MAT save file will be stored

      _wd : str
        path where MATLAB wrapper scripts for LBLRTM are located

      scriptname : str
        filename for temporary MATLAB/OCTAVE script (saved after exit)

      command : str
        path to MATLAB/OCTAVE executable

    :OUTPUTS:
       A 2- or 3-tuple: First element is wavelength in microns, second
       element is transmittance (if requested). Radiance will (if
       requested) always be the last element, and in f_nu units: W/cm2/sr/(cm^-1)

    :REQUIREMENTS:
       SciPy

       OCTAVE or MATLAB

       `LBLRTM  <http://rtweb.aer.com/lblrtm_code.html>`_

       D. Feldman's set of MATLAB `wrapper scripts
       <http://www.mathworks.com/matlabcentral/fileexchange/6461-lblrtm-wrapper-version-0-2/>`_
    """
    # 2011-10-13 13:59 IJMC: Created.
    # 2011-10-19 23:35 IJMC: Added scipy.__version__ check for MAT IO
    # 2011-10-25 17:00 IJMC: Added pwv_offset option.
    # 2011-11-07 08:47 IJMC: Now path(path..) uses _wd input option; call --wd option.
    # 2012-07-20 21:25 IJMC: Added 'alt' option for altitude.
    # 2012-09-16 15:13 IJMC: Fixed for 'dotrans' and 'dorad'
    # 2014-02-17 18:48 IJMC: Specified approximate units of radiance output.

    import os
    from scipy.io import loadmat
    from scipy import __version__

    # Define variables:
    
    def beforev8(ver):
        v1, v2, v3 = ver.split('.')
        if v1==0 and v2<8:
            before = True
        else:
            before = False
        return before

    if pwv<0:
        pwv = np.abs(pwv_offset - pwv)
    #lamrange = [1.12, 2.55] # microns
    #res = 200;  # lambda/dlambda
    

    # Try to delete old files, if possible.
    if os.path.isfile(_save):
        #try: 
        #    os.remove(_save)
        #except:
        while os.path.isfile(_save):
            _save = _save.replace('.mat', '0.mat')

    if os.path.isfile(scriptname):
        #try: 
        #    os.remove(scriptname)
        #except:
        while os.path.isfile(scriptname):
            scriptname = scriptname.replace('.m', '0.m')

    aerlbl_dir = '/'.join(_wd.split('/')[0:-2]) + '/'
            
    # Make the matlab script:
    matlines = []
    matlines.append("_path = '%s';\n" % _wd )
    matlines.append("_save = '%s';\n" % _save )
    matlines.append("_dir0 = pwd;\n")
    matlines.append("path(path, '%s')\n" % _wd )
    matlines.append("lamrange = [%1.5f, %1.5f];\n" % (lamrange[0], lamrange[1]))
    if verbose:
        matlines.append("pwd\n")
        matlines.append("wt = which('telluric_simulator')\n")
        matlines.append("strfind(path, 'caltech')\n")
    matlines.append("tran_out = telluric_simulator(lamrange, '--dotran %i', '--dorad %i', '--R %s', '--alt %1.4f', '--pwv %1.4f', '--co2 %1.1f', '--zang %1.3f', '--verbose %i', '--wd %s');\n" % (int(dotrans), int(dorad), res, alt, pwv, co2, zang, verbose, aerlbl_dir) )
    matlines.append("save('-v6', _save, 'tran_out');\n")

    # Write script to disk, and execute it:
    f = open(scriptname, 'w')
    f.writelines(matlines)
    f.close()
    os.system('%s %s' % (command, scriptname))

    # Open the MAT file and extract the desired output:
#    try:
    if os.path.isfile(_save):
        mat = loadmat(_save)
    else:
        trycount = 1
        print "Saved file '%s' could not be loaded..." % _save
        while trycount < 5:
            os.system('%s %s' % (command, scriptname))
            try:
                mat = loadmat(_save)
                trycount = 10
            except:
                trycount += 1
        if trycount < 10: # never successfully loaded the file.
            pdb.set_trace()

    if beforev8(__version__):
        w = mat['tran_out'][0][0].wavelength.ravel()
    else:
        w = mat['tran_out']['wavelength'][0][0].ravel()
    if dotrans:
        if beforev8(__version__):
            t = mat['tran_out'][0][0].transmittance.ravel()
        else:
            t = mat['tran_out']['transmittance'][0][0].ravel()
    if dorad:
        if beforev8(__version__):
            r = mat['tran_out'][0][0].radiance.ravel()
        else:
            r = mat['tran_out']['radiance'][0][0].ravel()

    os.remove(scriptname)
    os.remove(_save)

    if dotrans:
        ret = (w, t)
    else:
        ret = (w,)
    if dorad:
        ret = ret + (r,)

    return ret

def zenith(ra, dec, ha, obs):
    """ Compute zenith angle (in degrees) for an observation.

    :INPUTS:
      ra : str
         Right Ascension of target, in format: HH:MM:SS.SS

      dec : str
         Declination of target, in format: +ddd:mm:ss

      ha : str
         Hour Angle of target, in format: +HH:MM:SS.SS

      obs : str
         Name of observatory site (keck, irtf, lick, lapalma, ctio,
         andersonmesa, mtgraham, kpno) or a 3-tuple containing
         (longitude_string, latitude_string, elevation_in_meters)

         
    :OUTPUTS:
      Zenith angle, in degrees, for the specified observation

    :REQUIREMENTS:
      :doc:`phot`

      Numpy
         """
    # 2011-10-14 09:43 IJMC: Created

    import phot

    #observer = ephem.Observer()
    if obs=='lick':
        obs_long, obs_lat = '-121:38.2','37:20.6'
        obs_elev = 1290
    elif obs=='keck':
        obs_long, obs_lat = '-155:28.7','19:49.7'
	obs_elev = 4160
    elif obs=='irtf':
        obs_long, obs_lat = '-155:28:21.3', '19:49:34.8'
	obs_elev = 4205
    elif obs=='lapalma':
        obs_long, obs_lat = '17:53.6','28:45.5'
	obs_elev = 4160
    elif obs=='ctio':
        obs_long, obs_lat = '-70:48:54','-30:9.92'
	obs_elev = 2215
    elif obs=='andersonmesa':  # 
        obs_long, obs_lat = '-111:32:09', '30:05:49'
	obs_elev = 2163
    elif obs=='mtgraham':
        obs_long, obs_lat = '-109:53:23', '32:42:05'
        obs_elev = 3221
    elif obs=='kpno':
        obs_long, obs_lat = '-111:25:48', '31:57:30'
        obs_elev = 2096
    elif len(obs)==3:
        obs_long, obs_lat, obs_elev = obs
    else:
        print "Unknown or imparseable observatory site."
        return -1

    lat = phot.dms(obs_lat) * np.pi/180.
    long = phot.dms(obs_long) * np.pi/180.
    ra = (phot.hms(ra))  * np.pi/180.
    dec= (phot.dms(dec)) * np.pi/180.

    # Compute terms for coordinate conversion
    if hasattr(ha, '__iter__'):
        zang = []
        for ha0 in ha:
            har = (phot.hms(ha0))  * np.pi/180.
            term1 = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(har)
            term2 = np.cos(lat)*np.sin(dec)-np.sin(lat)*np.cos(dec)*np.cos(har)
            term3 = -np.cos(dec)*np.sin(har)
            rad = np.abs(term3 +1j*term2)
            az = np.arctan2(term3,term2)
            alt = np.arctan2(term1, rad)
            zang.append(90. - (alt*180./np.pi))
    else:
        har = (phot.hms(ha0))  * np.pi/180.
        term1 = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(har)
        term2 = np.cos(lat)*np.sin(dec)-np.sin(lat)*np.cos(dec)*np.cos(har)
        term3 = -np.cos(dec)*np.sin(har)
        rad = np.abs(term3 +1j*term2)
        az = np.arctan2(term3,term2)
        alt = np.arctan2(term1, rad)
        zang = 90. - (alt*180./np.pi)
        
    ## Compute airmass
    #z = pi/2. - alt
    #airmass = 1./(np.cos(z) + 0.50572*(96.07995-z*180./pi)**(-1.6364))
    
    return zang




def spexsxd_scatter_model(dat, halfwid=48, xlims=[470, 1024], ylims=[800, 1024], full_output=False, itime=None):
    """Model the scattered light seen in SpeX/SXD K-band frames.

    :INPUTS:
       dat : str or numpy array
         filename of raw SXD frame to be corrected, or a Numpy array
         containing its data.

    :OPTIONS:
       halfwid : int
         half-width of the spectral orders.  Experience shows this is
         approximately 48 pixels.  This value is not fit!

       xlims : list of length 2
         minimum and maximum x-pixel values to use in the fitting

       ylims : list of length 2
         minimum and maximum y-pixel values to use in the fitting

       full_output : bool
         whether to output only model, or the tuple (model, fits, chisq, nbad)

       itime : float
         integration time, in seconds, with which to scale the initial
         guesses

    :OUTPUT:
       scatter_model : numpy array
         Model of the scattered light component, for subtraction or saving.

      OR:

       scatter_model, fits, chis, nbad

    :REQUIREMENTS:
       :doc:`pyfits`, :doc:`numpy`, :doc:`fit_atmo`, :doc:`analysis`, :doc:`phasecurves`

    :TO_DO_LIST:
       I could stand to be more clever in modeling the scattered light
       components -- perhaps fitting for the width, or at least
       allowing the width to be non-integer.
    """
    # 2011-11-10 11:10 IJMC: Created

    import analysis as an
    import phasecurves as pc

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    

    ############################################################
    # Define some helper functions:
    ############################################################



    def tophat(param, x):
        """Grey-pixel tophat function with set width
        param: [cen_pix, amplitude, background]
        x : must be array of ints, arange(0, size-1)
        returns the model."""     
        # 2011-11-09 21:37 IJMC: Created
        intpix, fracpix = int(param[0]), param[0] % 1
        th = param[1] * ((-halfwid <= (x - intpix)) * ((x - intpix) < halfwid))
        #        th =  * th.astype(float)
        if (intpix >= halfwid) and ((intpix - halfwid) < x.size):
            th[intpix - halfwid] = param[1]*(1. - fracpix)
        if (intpix < (x.size - halfwid)) and ((intpix + halfwid) >= 0):
            th[intpix + halfwid] = param[1]*fracpix
        return th + param[2]

    def tophat2g(param, x, p0prior=None):
        """Grey-pixel double-tophat plus gaussian
        param: [cen_pix1, amplitude1, cen_pix2, amplitude2, g_area, g_sigma, g_center, background]
        x : must be ints, arange(0, size-1)
        returns the model."""     # 2011-11-09 21:37 IJMC: Created
        #th12 = 
        #th2 = 
        #gauss = 
#        if p0prior is not None:
#            penalty = 
        return tophat([param[0], param[1], 0], x) + \
            tophat([param[2], param[3], 0], x) + \
            gaussian(param[4:7], x) + param[7]

    ############################################################
    # Parse inputs
    ############################################################
    halfwid = int(halfwid)
    if isinstance(dat, np.ndarray):
        if itime is None: 
            itime = 1.
    else:
        if itime is None:
            try:
                itime = pyfits.getval(dat, 'ITIME')
            except:
                itime = 1.
        dat = pyfits.getdata(dat)

    nx, ny = dat.shape

    scatter_model = np.zeros((nx, ny), dtype=float)
    chis, fits, nbad = [], [], []
    iivals = np.arange(xlims[1]-1, xlims[0], -1, dtype=int)

    position_offset = 850 - ylims[0]
    est_coefs = np.array([ -5.02509772e-05,   2.97212397e-01,  -7.65702234e+01])
    estimated_position = np.polyval(est_coefs, iivals) + position_offset
    estimated_error = 0.5

    # to hold scattered light position fixed, rather than fitting for
    # that position, uncomment the following line:
    #holdfixed = [0] 
    holdfixed = None

    ############################################################
    # Start fitting
    ############################################################
    for jj, ii in enumerate(iivals):
        col = dat[ylims[0]:ylims[1], ii]
        ecol = np.ones(col.size, dtype=float)
        x = np.arange(col.size, dtype=float)
        if len(fits)==0:
            all_guess = [175 + position_offset, 7*itime, \
                             70 + position_offset, 7*itime, \
                             250*itime, 5, 89 + position_offset, 50]
        else:
            all_guess = fits[-1]
        all_guess[0] = estimated_position[jj]
        model_all = tophat2g(all_guess, x)
        res = (model_all - col)
        badpix = np.abs(res) > (4*an.stdr(res, nsigma=4))
        ecol[badpix] += 1e9

        fit = an.fmin(pc.errfunc, all_guess, args=(tophat2g, x, col, 1./ecol**2), full_output=True, maxiter=1e4, maxfun=1e4, disp=False, kw=dict(testfinite=False), holdfixed=holdfixed)
        best_params = fit[0].copy()
        res = tophat2g(best_params, x) - col
        badpix = np.abs(res) > (4*an.stdr(res, nsigma=4))
        badpix[((np.abs(np.abs(x - best_params[0]) - 48.)) < 2) + \
                   ((np.abs(np.abs(x - best_params[2]) - 48.)) < 2)] = False
        badpix += (np.abs(res) > (20*an.stdr(res, nsigma=4)))
        ecol = np.ones(col.size, dtype=float)
        ecol[badpix] += 1e9
        best_chisq = pc.errfunc(best_params, tophat2g, x, col, 1./ecol**2)

        # Make sure you didn't converge on the wrong model:
        for this_offset in ([-2, 0, 2]):
            this_guess = fit[0].copy()
            this_guess[2] += this_offset
            this_guess[0] = estimated_position[jj]
            #pc.errfunc(this_guess, tophat2g, x, col, 1./ecol**2)
            this_fit = an.fmin(pc.errfunc, this_guess, args=(tophat2g, x, col, 1./ecol**2), full_output=True, maxiter=1e4, maxfun=1e4, disp=False, kw=dict(testfinite=False), holdfixed=holdfixed)
            #print this_offset1, this_offset2, this_fit[1]
            if this_fit[1] < best_chisq:
                best_chisq = this_fit[1]
                best_params = this_fit[0].copy()

        fits.append(best_params)
        chis.append(best_chisq)
        nbad.append(badpix.sum())
        mod2 = tophat2g(best_params, x)
        scatter_model[ylims[0]:ylims[1], ii]  = tophat(list(best_params[0:2])+[0], x)

    if full_output:
        return scatter_model, fits, chis, nbad
    else:
        return scatter_model



def spexsxd_scatter_fix(fn1, fn2, **kw):
    """ Fix scattered light in SpeX/SXD K-band and write a new file.

    :INPUTS:
       fn1 : str
         file to be fixed

       fn2 : str
         new filename of fixed file.

    :OPTIONS:
       clobber : bool
          whether to overwrite existing FITS files
       
       Other options will be passed to :func:`spexsxd_scatter_model`

    :OUTPUTS:
       status : int
         0 if a problem, 1 if everything is Okay
    """
    # 2011-11-10 13:34 IJMC: Created


    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    

    if kw.has_key('clobber'):
        clobber = kw.pop('clobber')
    else:
        clobber = False

    try:
        dat0 = pyfits.getdata(fn1)
        hdr0 = pyfits.getheader(fn1)
    except:
        print "Could not read one of data or header from %s" % fn1
        return 0

    try:
        hdr0.update('SCAT_FIX', 1, 'SXD Scattered light fixed by spexsxd_scatter_fix()')
    except:
        print "Could not update header properly."
        return 0

    try:
        if hdr0.has_key('ITIME') and not kw.has_key('itime'):
            kw['itime'] = hdr0['ITIME']
        mod = spexsxd_scatter_model(dat0, **kw)
    except:
        print "Could not not model scattered light with spexsxd_scatter_model()"
        return 0

    try:
        if not isinstance(mod, np.ndarray):
            mod = mod[0]
        pyfits.writeto(fn2, dat0 - mod, header=hdr0, clobber=clobber, output_verify='ignore')
    except:
        print "Could not write updated file to %s; clobber is %s" % (fn2, clobber)
        return 0

    return 1
    
        
    
def tophat2(param, x):
    """Grey-pixel tophat function with set width
    param: [cen_pix, amplitude, background]
    newparam: [amplitude, full width, cen_pix, background]
    x : must be array of ints, arange(0, size-1)
    returns the model."""     
    # 2011-11-09 21:37 IJMC: Created
    import analysis as an
    
    oversamp = 100.

    x2 = np.linspace(x[0], x[-1], (x.size - 1)*oversamp + 1)
    halfwid = np.round(param[1] / 2.).astype(int)
    halfwid = param[1] / 2.
    intpix, fracpix = int(param[2]), param[2] % 1
    intwid, fracwid = int(halfwid), halfwid % 1
    th2 = 0.0 + param[0] * ((-halfwid <= (x2 - param[2])) * ((x2 - param[2]) < halfwid))

    th = an.binarray(np.concatenate((th2, np.zeros(oversamp-1))), oversamp)

    return th + param[3]


def tophat_alt(param, x):
    """Standard tophat function (alternative version).

    :INPUTS:
      p : sequence
        p[0] -- Amplitude
        p[1] -- full width dispersion
        p[2] -- central offset (mean location)
        p[3] -- vertical offset (OPTIONAL)

      x : scalar or sequence
        values at which to evaluate function

    :OUTPUTS:
      y : scalar or sequence
        1.0 where |x| < 0.5, 0.5 where |x| = 0.5, 0.0 otherwise.
        """
    # 2012-04-11 12:50 IJMC: Created
    if len(param)==3:
        vertical_offset = 0.
    elif len(param)>3:
        vertical_offset = param[3]
    else:
        print "Input `param` to function `tophat` requires length > 3."
        return -1

    amplitude, width, center = param[0:3]

    absx = np.abs((x - center) )
    if hasattr(x, '__iter__'):
        ret = np.zeros(x.shape, float)

        #ind1 = absx < (0.5*width)
        #ret[ind1] = 1.0

        i0 = np.searchsorted(x, center-width/2)
        i1 = np.searchsorted(x, center+width/2)
        ret[i0:i1] = 1.

        #ind2 = absx ==(0.5*width)
        #ret[ind2] = 0.5

    else:
        if absx < 0.5:
            ret = 1.0
        elif absx==0.5:
            ret = 0.5
        else:
            ret = 0.

    return amplitude * ret + vertical_offset


def model_resel(param, x):
    """Model a spectral resolution element.

    :INPUTS:
      param : sequence
      
        param[0, 1, 2] - amplitude, sigma, and central location of
        Gaussian line profile (cf. :func:`analysis.gaussian`).

        param[3, 4, 5] - amplitude, width, and central location of
        top-hat-like background (cf. :func:`tophat`).

        param[6::] - additional (constant or polynomial) background
        components, for evaluation with :func:`numpy.polyval`

      x : sequence
        Values at which to evaluate model function (i.e., pixels).
        Typically 1D.
    
    :OUTPUTS:
      line : NumPy array
         model of the resolution element, of same shape as `x`.

    :DESCRIPTION:
      
      Model a spectral resolution element along the spatial
      direction.  This consists of a (presumably Gaussian) line
      profile superimposed on the spectral trace's top-hat-like
      background, with an additional constant (or polynomial)
      out-of-echelle-order background component.
    """
    # 2012-04-11 12:57 IJMC: Created
    # 2012-04-12 13:29 IJMC: Try to be more clever and save time in
    #                        the back2 polyval call.

    lineprofile = gaussian(param[0:3], x)
    back1 = tophat(param[3:6], x)

    if hasattr(param[6::], '__iter__') and len(param[6::])>1:
        back2 = np.polyval(param[6::], x)
    else:
        back2 = param[6::]

    return lineprofile + back1 + back2

def add_partial_pixel(x0, y0, z0, z):
    """
    :INPUTS:
      x0 : int or sequence of ints
        first index of z at which data will be added

      y0 : int or sequence of ints
        second index of z at which data will be added

      z0 : scalar or sequence
        values which will be added to z

      z : 2D NumPy array
        initial data, to which partial-pixels will be added
    """
    # 2012-04-11 14:24 IJMC: Created
    nx, ny = z.shape[0:2]
    x = np.arange(nx)
    y = np.arange(ny)
    dx, dy = 1, 1
    ret = np.array(z, copy=True)

    if hasattr(x0, '__iter__'):
        if len(x0)==len(y0) and len(x0)==len(z0):
            for x00, y00, z00 in zip(x0, y0, z0):
                ret = add_partial_pixel(x00, y00, z00, ret)
        else:
            print "Inputs x0, y0, and z0 must have the same length!"
            ret = -1

            if False:
                if hasattr(x0, '__iter__'):
                    x0 = np.array(x0, copy=False)
                    y0 = np.array(y0, copy=False)
                    z0 = np.array(z0, copy=False)
                else:
                    x0 = np.array([x0])
                    y0 = np.array([y0])
                    z0 = np.array([z0])

                ix0 = np.tile(np.vstack((np.floor(x0), np.floor(x0)+1)).astype(int), (2,1))
                iy0 = np.tile(np.vstack((np.floor(y0), np.floor(y0)+1)).astype(int), (2,1))

                xfrac = x0 % 1
                yfrac = y0 % 1

                weights0 = np.vstack([(1. - xfrac) * (1. - yfrac), \
                               xfrac * (1. - yfrac), \
                               (1. - xfrac) *  yfrac, \
                               xfrac * yfrac])

                for ii in range(ix0.shape[1]):
                    print ii, weights0[:,ii]*z0[ii]
                    ret[ix0[:,ii], iy0[:,ii]] = ret[ix0[:,ii], iy0[:,ii]] + weights0[:,ii]*z0[ii]

    else:
        ix0 = map(int, [np.floor(x0), np.floor(x0)+1]*2)
        iy0 = map(int, [np.floor(y0)]*2 + [np.floor(y0)+1]*2)

        xfrac = x0 % 1
        yfrac = y0 % 1
        weights0 = [(1. - xfrac) * (1. - yfrac), \
                       xfrac * (1. - yfrac), \
                       (1. - xfrac) *  yfrac, \
                       xfrac * yfrac]
        ix = []
        iy = []
        weights = []
        for ii in range(4):
            #print ix0[ii], iy0[ii], weights0[ii]
            if ix0[ii]>=0 and ix0[ii]<nx and iy0[ii]>=0 and iy0[ii]<ny:
                ix.append(ix0[ii]), iy.append(iy0[ii])
                weights.append(weights0[ii])

        npix = len(ix)
        if npix>0:
            sumweights = sum(weights)
            for ii in range(npix):
                #print ix[ii], iy[ii], weights[ii]
                ret[ix[ii], iy[ii]] += weights[ii] * z0


    return ret


def modelSpectralTrace(param, shape=None, nscat=None, npw=None, npy=None, noy=None, now=None, ndist=None, x=None, y=None, transpose=False):
    """Model a raw spectral trace!

    :INPUTS:

    NOTE that most inputs should be in the _rectified_ frame.
    
      Trace background pedestal level : 1D array

      Width of background pedestarl level : scalar (for now)

      Center of trace : 1D array

      Offset of object spectrum, relative to center : scalar

      width of 1D PSF : scalar

      Area of 1D psf : 1D array

      Distortion (x and y, somehow???)

      Scattered light background : scalar (for now)
    """
    # 2012-04-11 17:50 IJMC: Created

    ########################################
    # Parse various inputs:
    ########################################
    param = np.array(param, copy=False)
    nx, ny = shape

    # Construct pixel vectors:
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)

    def makevec(input):
        if not hasattr(input, '__iter__'):
            ret = [input]*ny 
            #ret = np.polyval([input], y)
        elif len(input)==1:
            ret = [input[0]]*ny 
            #ret = np.polyval(input, y)
        else:
            ret = np.polyval(input, y)
        return ret

    # Define all the parameters:
    pedestal_level = param[0:ny]
    obj_flux = param[ny:ny*2]
    pedestal_width = param[ny*2:ny*2+npw]
    obj_fwhm = param[ny*2+npw:ny*2+npw+now]
    spectrum_yparam = param[ny*2+npw+now : ny*2+npw+now+npy]
    obj_yparam = param[ny*2+npw+now+npy : ny*2+npw+now+npy+noy]
    disp_distort = param[ny*2+npw+now+npy+noy : ny*2+npw+now+npy+noy+ndist]
    scat_param = list(param[-ndist:])

    
    pedestal_width = makevec(pedestal_width)
    obj_fwhm = makevec(obj_fwhm)
    obj_yloc = makevec(obj_yparam) #, mode='cheby')
    spectrum_yloc = makevec(spectrum_yparam) #, mode='cheby')

    ########################################
    # Generate spectral model
    ########################################

    rectified_model = np.zeros((nx, ny), float)

    for ii in range(ny):
        obj_param = [obj_flux[ii], obj_fwhm[ii], obj_yloc[ii]]
        bkg_param = [pedestal_level[ii], pedestal_width[ii], spectrum_yloc[ii]]
        oneDparam = obj_param + bkg_param + scat_param
        rectified_model[:, ii] =  model_resel(oneDparam, x)

        #ypts = x
        #xpts = ii + np.polyval(disp_distort, x-obj_yloc[ii])
        #distorted_model = spec.add_partial_pixel(ypts, xpts, oneD, model)

    interp_model = 0*rectified_model

    if False:
        # One way to do it:
        newxcoords = (y - np.polyval(disp_distort, (x.reshape(nx,1)-obj_yloc.reshape(1,ny))))

        meshmod = interpolate.RectBivariateSpline(x, y, rectified_model, kx=1, ky=1, s=0.)
        for ii in range(nx):
            interp_model[ii,:] = meshmod(x[ii], newxcoords[ii,:])

    else:
        # An alternate (not entirely equivalanet) approach, about equally fast:
        shifts = np.polyval(disp_distort, x)
        intshifts = np.floor(shifts)
        minshift = intshifts.min()
        shifts -= minshift
        intshifts -= minshift

        for ii in range(nx):
            kern = np.zeros(intshifts.max() - intshifts.min()+2, float)
            kern[intshifts[ii]] += 1. - (shifts[ii] - intshifts[ii])
            kern[intshifts[ii]+1] += (shifts[ii] - intshifts[ii])
            interp_model[ii] = np.convolve(rectified_model[ii], kern, 'same')


    return interp_model


def makeSpexSlitlessSky(skyfns, scatcen=[980, 150], scatdim=[60, 300]):
    """
    Generate a normalized Sky frame from SpeX slitless spectroscopy data.

    :INPUTS:
      skyfns : list of strs
        Filenames of slitless-spectroscopy sky frames
        
      scatcen : 2-sequence
        center of region to use in median-normalizing the frame.

      scatdim : 2-sequence
        full width of region to use in median-normalizing the frame.

    :OUTPUTS:
      (sky, skyHeader)
    """
    # 2012-04-19 09:38 IJMC: Created
    import phot

    nsky = len(skyfns)

    skyscat = phot.subreg2(skyfns, center=scatcen, dim=scatdim)
    normfact = np.median(skyscat.reshape(nsky, np.prod(scatdim)), 1)

    hdr = pyfits.getheader(skyfns[0])

    skyframe = np.zeros((hdr['naxis1'], hdr['naxis2']), float)
    for skyfn, factor in zip(skyfns, normfact):
        skyframe += (pyfits.getdata(skyfn) / factor / nsky)

    hdr.update('SKYRGCEN', str(scatcen))
    hdr.update('SKYRGDIM', str(scatdim))
    hdr.update('SKYNOTE', 'sky frame, median=1 in SKYRG')

    return skyframe, hdr
    


def resamplespec(w1, w0, spec0, oversamp=100):
    """
    Resample a spectrum while conserving flux density.

    :INPUTS:
      w1 : sequence
        new wavelength grid (i.e., center wavelength of each pixel)

      w0 : sequence
        old wavelength grid (i.e., center wavelength of each pixel)

      spec0 : sequence
        old spectrum (e.g., flux density or photon counts)

      oversamp : int
        factor by which to oversample input spectrum prior to
        rebinning.  The worst fractional precision you achieve is
        roughly 1./oversamp.

    :NOTE: 
      Format is the same as :func:`numpy.interp`
      
    :REQUIREMENTS:
      :doc:`tools`

    """
    from tools import errxy

    # 2012-04-25 18:40 IJMC: Created
    nlam = len(w0)
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.)*oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, w0)
    spec0int = np.interp(w0int, w0, spec0)/oversamp

    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(w1).max()
    w1bins = np.concatenate(([w1[0] - maxdiffw1], 
                             .5*(w1[1::] + w1[0:-1]), \
                                 [w1[-1] + maxdiffw1]))
    # Bin down the interpolated spectrum:
    junk, spec1, junk2, junk3 = errxy(w0int, spec0int, w1bins, xmode=None, ymode='sum', xerr=None, yerr=None)

    return spec1




#wcoords = spec.wavelengthMatch(mastersky0, wsky, thissky, ethissky, guess=wcoef1, order=None)

def wavelengthMatch(spectrum, wtemplate, template, etemplate, guess=None, nthread=1):
    """
    Determine dispersion solution for a spectrum, from a template.

    :INPUTS:
      spectrum : 1D sequence
        Spectrum for which a wavelength solution is desired.

      wtemplate : 1D sequence
        Known wavelength grid of a template spectrum.

      template : 1D sequence
        Flux (e.g.) levels of the template spectrum with known
        wavelength solution.

      etemplate : 1D sequence
        Uncertainties on the template values.  This can be important
        in a weighted fit!

    :OPTIONS:
      guess : sequence
        Initial guess for the wavelength solution.  This is very
        helpful, if you have it!  The guess should be a sequence
        containing the set of Chebychev polynomial coefficients,
        followed by a scale factor and DC offset (to help in scaling
        the template).

        If guess is None, attempt to fit a simple linear dispersion relation.

      order : int > 0
        NOT YET IMPLEMENTED! BUT EVENTUALLY: if guess is None, this
        sets the polynomial order of the wavelength solution.

        FOR THE MOMENT: if guess is None, return a simple linear
        solution.  This is likely to fail entirely for strongly
        nonlinear dispersion solutions or poorly mismatched template
        and spectrum.

      nthread : int > 0
        Number of processors to use for MCMC searching.

    :RETURNS:
      (wavelength, wavelength_polynomial_coefficients, full_parameter_set)

    :NOTES:
      This implementation uses a rather crude MCMC sampling approach
      to sample parameters space and 'home in' on better solutions.
      There is probably a way to do this that is both faster and more
      optimal...

      Note that if 'spectrum' and 'template' are of different lengths,
      the longer one will be trimmed at the end to make the lengths match.

    :REQUIREMENTS:
      `emcee  <http://TBD>`_
      
    """
    #2012-04-25 20:53 IJMC: Created
    # 2012-09-23 20:17 IJMC: Now spectrum & template can be different length.
    # 2013-03-09 17:23 IJMC: Added nthread option

    import emcee
    import phasecurves as pc

    nlam_s = len(spectrum)
    nlam_t = len(template)

    if nlam_s <= nlam_t:
        template = np.array(template, copy=True)[0:nlam_s]
        etemplate = np.array(etemplate, copy=True)[0:nlam_s]
        nlam = nlam_s
        spectrum_trimmed = False        
    else: # nlam_s > nlam_t:
        spectrum0 = np.array(spectrum, copy=True)
        spectrum = spectrum0[0:nlam_t]
        wtemplate = np.array(wtemplate, copy=True)[0:nlam_t]
        nlam = nlam_t
        spectrum_trimmed = True

    # Create a normalized vector of coordinates for computing
    # normalized polynomials:
    dx0 = 1. / (nlam - 1.)
    x0n = dx0 * np.arange(nlam) - 1.
    #x0n = 2*np.arange(nlam, dtype=float) / (nlam - 1.) - 1



    if guess is None:
        # Start with a simple linear wavelength solution.
        guess = [np.diff(wtemplate).mean() * len(template)/2, np.mean(wtemplate), np.median(template)/np.median(spectrum), 1.]
    
    # Define arguments for use by fitting routines:
    fitting_args = (makemodel, x0n, spectrum, wtemplate, template, 1./etemplate**2)

    # Try to find an initial best fit:
    bestparams = an.fmin(pc.errfunc, guess, args=fitting_args, disp=False)
    pdb.set_trace()
    
    # Initial fit is likely a local minimum, so explore parameter
    # space using an MCMC approach.
    ndim = len(guess)
    nwalkers = ndim * 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pc.lnprobfunc, args=fitting_args, threads=nthread)

    # Initialize the sampler with various starting positions:
    e_params1 = np.vstack((np.array(guess)/10., np.zeros(ndim) + .01)).max(0)
    e_params2 = np.vstack((bestparams/10., np.zeros(ndim) + .01)).max(0)
    p0 = np.vstack(([guess, bestparams], \
           [np.random.normal(guess, e_params1) for ii in xrange(nwalkers/2-1)], \
           [np.random.normal(bestparams, e_params2) for ii in xrange(nwalkers/2-1)]))

    # Run the sampler for a while:
    pos, prob, state = sampler.run_mcmc(p0, 300) # Burn-in

    bestparams = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
    
    # Optimize the latest set of best parameters.
    bestparams = an.fmin(pc.errfunc, bestparams, args=fitting_args, disp=False)

    dispersionSolution = bestparams[0:-2]
    if spectrum_trimmed:
        x0n_original = dx0 * np.arange(nlam_s) - 1.
        wavelengths = np.polyval(dispersionSolution, x0n_original)
    else:
        wavelengths = np.polyval(dispersionSolution, x0n)

    return dispersionSolution, wavelengths, bestparams

def makemodel(params, xvec, specvec, wtemplate):
    """Helper function for :func:`wavelengthMatch`: generate a scaled,
    interpolative model of the template."""
    wcoef = params[0:-2]
    scale, offset = params[-2::]
    neww = np.polyval(wcoef, xvec)
    return offset + scale * np.interp(wtemplate, neww, specvec, left=0, right=0)


def normalizeSpecFlat(flatdat, nspec=1, minsep=50, median_width=51, readnoise=40, badpixelmask=None, traces=None):
    """Trace and normalize a spectroscopic flat field frame.

    :INPUTS:
      flatdat : 2D NumPy array
        Master, unnormalized flat frame: assumed to be measured in
        photoelectrons (for computing uncertainties).

      nspec : int
        Number of spectral orders to find and normalize

      minsep : int
        Minimum separation, in pixels, between spectral orders that
        will be found.

      median_width : int
        Width of median-filter kernel used to compute the low-

      readnoise : scalar
        Detector read noise, in electrons.  For computing uncertainties.

      badpixelmask : 2D NumPy array
        bad pixel mask: 1 at bad pixel locations, 0 elsewhere.

      traces : 2D NumPy Array 
        (nord, pord) shaped numpy array representing the polynomial
         coefficients for each order (suitable for use with
         np.polyval), as produced by :func:`traceorders`
    """
    # 2012-04-28 06:22 IJMC: Created
    # 2012-07-24 21:04 IJMC: Now, as a final step, all bad indices are set to unity.
    # 2014-12-17 20:07 IJMC: Added 'traces' option

    import analysis as an
    from scipy import signal

    if badpixelmask is None:
        badpixelmask = np.zeros(flatdat.shape, bool)

    # Enforce positivity and de-weight negative flux values:
    e_flatdat = np.sqrt(flatdat + readnoise**2)
    badindices = ((flatdat<=0) + badpixelmask).nonzero()
    e_flatdat[badindices] = flatdat[badindices] * 1e9
    flatdat[badindices] = 1.

    # Find spectral orders, using derivatives (will probably fail if
    # spec. overlaps the edge!):
    ordvec = an.meanr(flatdat, axis=1, nsigma=3)

    filtvec = signal.medfilt(ordvec, 9)
    dvec1 = np.diff(filtvec)
    dvec2 = -np.diff(filtvec)
    dvec1[dvec1<0] = 0.
    dvec2[dvec2<0] = 0.

    x1 = np.arange(dvec1.size)
    available1 = np.ones(dvec1.size, dtype=bool)
    available2 = np.ones(dvec1.size, dtype=bool)

    pos1 = []
    pos2 = []
    for ii in range(nspec):
        thisx1 = x1[dvec1==dvec1[available1].max()][0]
        available1[np.abs(x1 - thisx1) < minsep] = False
        pos1.append(thisx1)
        thisx2 = x1[dvec2==dvec2[available2].max()][0]
        available2[np.abs(x1 - thisx2) < minsep] = False
        pos2.append(thisx2)

    limits = np.array(zip(np.sort(pos1), np.sort(pos2)))
    # Generate and normalize the spectral traces:
    masterflat = np.ones(flatdat.shape, dtype=float)

    if traces is not None:
        nx = flatdat.shape[1]
        xvec = np.arange(nx)
        ymat = np.tile(np.arange(flatdat.shape[0]), (nx, 1)).T

    for ii in range(nspec):
        if traces is None:
            profvec = np.median(flatdat[limits[ii,0]:limits[ii,1], :], axis=0)
            e_profvec = np.sqrt(an.wmean(flatdat[limits[ii,0]:limits[ii,1], :], 1./e_flatdat[limits[ii,0]:limits[ii,1], :]**2, axis=0) / np.diff(limits[ii]))[0]
            e_profvec[e_profvec <= 0] = profvec.max()*1e9
            smooth_prof = signal.medfilt(profvec, median_width)
            masterflat[limits[ii,0]:limits[ii,1], :] = flatdat[limits[ii,0]:limits[ii,1], :] / smooth_prof
        else:
            traceloc = np.polyval(traces[ii], xvec)
            limind = (limits[:,1] > traceloc.mean()).nonzero()[0][0]
            order_ind_2d = ((ymat - traceloc) > (limits[limind,0] - traceloc.mean())) * ((ymat - traceloc) < (limits[limind,1] - traceloc.mean()))
            profvec = np.array([np.median(flatdat[order_ind_2d[:,jj], jj]) for jj in xrange(nx)])
            smooth_prof = signal.medfilt(profvec, median_width)
            for jj in xrange(nx):
                masterflat[order_ind_2d[:,jj],jj] = flatdat[order_ind_2d[:,jj],jj] / smooth_prof[jj]
                

        # Ideally, we would do some sort of weighted fitting here.  Instead,
        # for now, just take a running median:

    masterflat[badindices] = 1.

    return masterflat


def optspecextr_idl(frame, gain, readnoise, x1, x2, idlexec, clobber=True, tempframefn='tempframe.fits', specfn='tempspec.fits', scriptfn='temp_specextract.pro', IDLoptions="adjfunc='adjgauss', adjoptions={center:1,centerfit:1,centerdeg:3}, bgdeg=3", inmask=None):
    """Run optimal spectral extraction in IDL; pass results to Python.

    :INPUTS:
      frame : str
        filename, or 2D Numpy Array, or list of filenames containing
        frames from which spectra will be extracted.  This should be
        in units of ADU (not electrons) for the noise properties to
        come out properly.  

        Also, the spectral trace must run vertically across the frame.

      gain : scalar
        Detector gain, in electrons / ADU

      readnoise : scalar
        Detector read noise, in electrons

      x1, x2 : ints, or lists of ints
        Start and stop indices of the spectral trace across the frame.
        If multiple frames are input and a single x1/x2 is input, the
        same value will be used for each frame.  Note however that
        multiple x1/x2 can also be input (one for each frame).

      idlexec : str
        Path to the IDL executable.  OPTSPECEXTR.PRO and its
        associated files must be in your IDL path.  If set to None,
        then it will be set to: os.popen('which idl').read().strip()

    :OPTIONS:
      clobber : bool
        Whether to overwrite files when writing input data to TEMPFRAMFN.

      tempframefn : str
        If input 'frame' is an array, it will be written to this
        filename in order to pass it to IDL.

      specfn : str
        IDL will write the spectral data to this filename in order to
        pass it back to Python.

      scriptfn : str
        Filename in which the short IDL script will be written.

      IDLoptions : str
         Options to pass to OPTSPECEXTR.PRO. For example:
         "adjfunc='adjgauss', adjoptions={center:1,centerfit:1,centerdeg:3}, bgdeg=3"

         Note that this Python code will break if you _don't_ trace
         the spectrum (adjoptions, etc.); this is an area for future
         work if I ever use a spectrograph with straight traces.

       inmask : None or str
         Name of the good pixel mask for OPTSPECEXTR.PRO.  Equal to 1
         for good pixels, and 0 for bad pixels.

    :OUTPUTS:
      For each input frame, a list of four items:
        [0] -- Extracted spectrum, ADU per pixel
        [1] -- Uncertainty (1 sigma) of extracted spectrum
        [2] -- Location of trace (in pixels) across the frame
        [3] -- Width of trace across the frame

    :NOTES:
      Note that this more closely follows Horne et al. than does
      :func:`optimalExtract`, and is faster than both that function
      and (especially!) :func:`extractSpectralProfiles`.  The only
      downside (if it is one) is that this function requires IDL.

    :TO-DO:
      Add options for user input of a variance frame, or of sky variance.

      Allow more flexibility (tracing, input/output options, etc.)
      
    :REQUIREMENTS:
       IDL

       `OPTSPECEXTR  <http://physics.ucf.edu/~jh/ast/software.html>`_

    """
    # 2012-08-18 16:36 IJMC: created
    # 2012-08-19 09:39 IJMC: Added 'inmask' option.

    import os

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    

    # Put the input frames in the proper format:
    if isinstance(frame, np.ndarray):
        frameisfilename = False
        if frame.ndim==2:
            frames = [frame]
        elif frame.ndim==1:
            print "Input array should be 2D or 3D -- no telling what will happen next!"
        else:
            frames = frame
    else:
        frameisfilename = True
        if isinstance(frame, str):
            frames = [frame]
        else:
            frames = frame
        
    if not hasattr(x1, '__iter__'):
        x1 = [x1] * len(frames)
    if not hasattr(x2, '__iter__'):
        x2 = [x2] * len(frames)
    
    if idlexec is None:
        idlexec = os.popen('which idl').read().strip()


    # Loop through all files:
    specs = []
    ii = 0
    for frame in frames:
        if frameisfilename:
            tempframefn = frame
        else:
            pyfits.writeto(tempframefn, frame, clobber=clobber)

        # Prepare the temporary IDL script:
        idlcmds = []
        idlcmds.append("frame = readfits('%s')\n" % tempframefn)
        idlcmds.append("gain = %1.3f\n" % gain)
        idlcmds.append("readnoise = %i\n" % readnoise)
        idlcmds.append("varim = abs(frame) / gain + readnoise^2\n")
        idlcmds.append("x1 = %i & x2 = %i\n" % (x1[ii],x2[ii]))
        if inmask is not None:
            idlcmds.append("inmask = readfits('%s')\n" % inmask)
            IDLoptions += ', inmask=inmask'
        idlcmds.append("spec = optspecextr(frame, varim, readnoise, gain, x1, x2, adjparms=adjparm, opvar=opvar, %s)\n" % IDLoptions)
        idlcmds.append("spec_err_loc_width = [[spec], [sqrt(opvar)], [adjparm.traceest], [adjparm.widthest]]\n")
        idlcmds.append("writefits,'%s', spec_err_loc_width\n" % (specfn))
        idlcmds.append("exit\n")

            
        


        # Write it to disk, and execute it.
        f = open(scriptfn, 'w')
        f.writelines(idlcmds)
        f.close()
        os.system('%s %s' % (idlexec, scriptfn))

        # Read the spectrum into Python, and iterate.
        spec = pyfits.getdata(specfn)
        specs.append(spec)
        ii += 1

    # Clean up after ourselves:
    if not frameisfilename and os.path.isfile(tempframefn):
        os.remove(tempframefn)
    if os.path.isfile(specfn):
        os.remove(specfn)
    if os.path.isfile(scriptfn):
        os.remove(scriptfn)

    # If only one file was run, we don't need to return a list.
    if ii==1:  
        specs = specs[0]

    return specs


def optimalExtract(*args, **kw):
    """
    Extract spectrum, following Horne 1986.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       bkg_radii : 2- or 4-sequence
         If length 2: inner and outer radii to use in computing
         background. Note that for this to be effective, the spectral
         trace should be positions in the center of 'data.'
         
         If length 4: start and end indices of both apertures for
         background fitting, of the form [b1_start, b1_end, b2_start,
         b2_end] where b1 and b2 are the two background apertures, and
         the elements are arranged in strictly ascending order.

       extract_radius : int or 2-sequence
         radius to use for both flux normalization and extraction.  If
         a sequence, the first and last indices of the array to use
         for spectral normalization and extraction.


       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       pord : int >= 0
         Degree of polynomial fit to construct profile.

       psigma : int >= 0
         Sigma-clipping threshold for computing profile.

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       nreject : int > 0
         Number of pixels to reject in each iteration.
             
    :RETURNS:
       3-tuple:
          [0] -- spectrum flux (in electrons)

          [1] -- uncertainty on spectrum flux

          [1] -- background flux


    :EXAMPLE:
      ::


    :SEE_ALSO:
      :func:`superExtract`.

    :NOTES:
      Horne's classic optimal extraction algorithm is optimal only so
      long as the spectral traces are very nearly aligned with
      detector rows or columns.  It is *not* well-suited for
      extracting substantially tilted or curved traces, for the
      reasons described by Marsh 1989, Mukai 1990.  For extracting
      such spectra, see :func:`superExtract`.
    """

    # 2012-08-20 08:24 IJMC: Created from previous, low-quality version.
    # 2012-09-03 11:37 IJMC: Renamed to replace previous, low-quality
    #                        version. Now bkg_radii and extract_radius
    #                        can refer to either a trace-centered
    #                        coordinate system, or the specific
    #                        indices of all aperture edges. Added nreject.


    from scipy import signal

    # Parse inputs:
    frame, variance, gain, readnoise = args[0:4]

    # Parse options:
    if kw.has_key('goodpixelmask'):
        goodpixelmask = np.array(kw['goodpixelmask'], copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)

    if kw.has_key('dispaxis'):
        if kw['dispaxis']==1:
            frame = frame.transpose()
            variance = variance.transpose()
            goodpixelmask = goodpixelmask.transpose()

    if kw.has_key('verbose'):
        verbose = kw['verbose']
    else:
        verbose = False

    if kw.has_key('bkg_radii'):
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if kw.has_key('extract_radius'):
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10
        if verbose: message("Setting option 'extract_radius' to: " + str(extract_radius))

    if kw.has_key('bord'):
        bord = kw['bord']
    else:
        bord = 1
        if verbose: message("Setting option 'bord' to: " + str(bord))

    if kw.has_key('bsigma'):
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: message("Setting option 'bsigma' to: " + str(bsigma))

    if kw.has_key('pord'):
        pord = kw['pord']
    else:
        pord = 2
        if verbose: message("Setting option 'pord' to: " + str(pord))

    if kw.has_key('psigma'):
        psigma = kw['psigma']
    else:
        psigma = 4
        if verbose: message("Setting option 'psigma' to: " + str(psigma))

    if kw.has_key('csigma'):
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: message("Setting option 'csigma' to: " + str(csigma))

    if kw.has_key('finite'):
        finite = kw['finite']
    else:
        finite = True
        if verbose: message("Setting option 'finite' to: " + str(finite))

    if kw.has_key('nreject'):
        nreject = kw['nreject']
    else:
        nreject = 100
        if verbose: message("Setting option 'nreject' to: " + str(nreject))

    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    
    variance[True-goodpixelmask] = frame[goodpixelmask].max() * 1e9
    nlam, fitwidth = frame.shape

    xxx = np.arange(-fitwidth/2, fitwidth/2)
    xxx0 = np.arange(fitwidth)
    if len(bkg_radii)==4: # Set all borders of background aperture:
        backgroundAperture = ((xxx0 > bkg_radii[0]) * (xxx0 <= bkg_radii[1])) + \
            ((xxx0 > bkg_radii[2]) * (xxx0 <= bkg_radii[3]))
    else: # Assume trace is centered, and use only radii.
        backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])

    if hasattr(extract_radius, '__iter__'):
        extractionAperture = (xxx0 > extract_radius[0]) * (xxx0 <= extract_radius[1])
    else:
        extractionAperture = np.abs(xxx) < extract_radius

    nextract = extractionAperture.sum()
    xb = xxx[backgroundAperture]

    #Step3: Sky Subtraction
    if bord==0: # faster to take weighted mean:
        background = an.wmean(frame[:, backgroundAperture], (goodpixelmask/variance)[:, backgroundAperture], axis=1)
    else:
        background = 0. * frame
        for ii in range(nlam):
            fit = an.polyfitr(xb, frame[ii, backgroundAperture], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundAperture], verbose=verbose-1)
            background[ii, :] = np.polyval(fit, xxx)

    # (my 3a: mask any bad values)
    badBackground = True - np.isfinite(background)
    background[badBackground] = 0.
    if verbose and badBackground.any():
        print "Found bad background values at: ", badBackground.nonzero()

    skysubFrame = frame - background


    #Step4: Extract 'standard' spectrum and its variance
    standardSpectrum = nextract * an.wmean(skysubFrame[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1) 
    varStandardSpectrum = nextract * an.wmean(variance[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1)

    # (my 4a: mask any bad values)
    badSpectrum = True - (np.isfinite(standardSpectrum))
    standardSpectrum[badSpectrum] = 1.
    varStandardSpectrum[badSpectrum] = varStandardSpectrum[True - badSpectrum].max() * 1e9


    #Step5: Construct spatial profile; enforce positivity & normalization
    normData = skysubFrame / standardSpectrum
    varNormData = variance / standardSpectrum**2


    # Iteratively clip outliers
    newBadPixels = True
    iter = -1
    if verbose: print "Looking for bad pixel outliers in profile construction."
    xl = np.linspace(-1., 1., nlam)

    while newBadPixels:
        iter += 1


        if pord==0: # faster to take weighted mean:
            profile = np.tile(an.wmean(normData, (goodpixelmask/varNormData), axis=0), (nlam,1))
        else:
            profile = 0. * frame
            for ii in range(fitwidth):
                fit = an.polyfitr(xl, normData[:, ii], pord, np.inf, w=(goodpixelmask/varNormData)[:, ii], verbose=verbose-1)
                profile[:, ii] = np.polyval(fit, xl)

        if profile.min() < 0:
            profile[profile < 0] = 0.
        profile /= profile.sum(1).reshape(nlam, 1)

        #Step6: Revise variance estimates 
        modelData = standardSpectrum * profile + background
        variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance

        outlierSigmas = (frame - modelData)**2/variance
        if outlierSigmas.max() > psigma**2:
            maxRejectedValue = max(psigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
            worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        if verbose: print "Rejected %i pixels on this iteration " % numberRejected

        #Step5: Construct spatial profile; enforce positivity & normalization
        varNormData = variance / standardSpectrum**2

    if verbose: print "%i bad pixels found" % iter


    # Iteratively clip Cosmic Rays
    newBadPixels = True
    iter = -1
    if verbose: print "Looking for bad pixel outliers in optimal extraction."
    while newBadPixels:
        iter += 1

        #Step 8: Extract optimal spectrum and its variance
        gp = goodpixelmask * profile
        denom = (gp * profile / variance)[:, extractionAperture].sum(1)
        spectrum = ((gp * skysubFrame  / variance)[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)
        varSpectrum = (gp[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)


        #Step6: Revise variance estimates 
        modelData = spectrum * profile + background
        variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance


        #Iterate until worse outliers are all identified:
        outlierSigmas = (frame - modelData)**2/variance
        if outlierSigmas.max() > csigma**2:
            maxRejectedValue = max(csigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
            worstOutliers = (outlierSigmas>=maxRejectedValues).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        if verbose: print "Rejected %i pixels on this iteration " % numberRejected


    if verbose: print "%i bad pixels found" % iter

    ret = (spectrum, varSpectrum, profile, background, goodpixelmask)

    return  ret



def superExtract(*args, **kw):
    """
    Optimally extract curved spectra, following Marsh 1989.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       trace : 1D numpy array
         location of spectral trace.  If None, :func:`traceorders` is
         invoked.

       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       npoly : int
         Number of profile polynomials to evaluate (Marsh's
         "K"). Ideally you should not need to set this -- instead,
         play with 'polyspacing' and 'extract_radius.' For symmetry,
         this should be odd.

       polyspacing : scalar
         Spacing between profile polynomials, in pixels. (Marsh's
         "S").  A few cursory tests suggests that the extraction
         precision (in the high S/N case) scales as S^-2 -- but the
         code slows down as S^2.

       pord : int
         Order of profile polynomials; 1 = linear, etc.

       bkg_radii : 2-sequence
         inner and outer radii to use in computing background

       extract_radius : int
         radius to use for both flux normalization and extraction

       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       tord : int >= 0
         Degree of spectral-trace polynomial (for trace across frame
         -- not used if 'trace' is input)

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       qmode : str ('fast' or 'slow')
         How to compute Marsh's Q-matrix.  Valid inputs are
         'fast-linear', 'slow-linear', 'fast-nearest,' 'slow-nearest,'
         and 'brute'.  These select between various methods of
         integrating the nearest-neighbor or linear interpolation
         schemes as described by Marsh; the 'linear' methods are
         preferred for accuracy.  Use 'slow' if you are running out of
         memory when using the 'fast' array-based methods.  'Brute' is
         both slow and inaccurate, and should not be used.
         
       nreject : int
         Number of outlier-pixels to reject at each iteration. 

       retall : bool
         If true, also return the 2D profile, background, variance
         map, and bad pixel mask.
             
    :RETURNS:
       object with fields for:
         spectrum

         varSpectrum

         trace


    :EXAMPLE:
      ::

        import spec
        import numpy as np
        import pylab as py

        def gaussian(p, x):
           if len(p)==3:
               p = concatenate((p, [0]))
           return (p[3] + p[0]/(p[1]*sqrt(2*pi)) * exp(-(x-p[2])**2 / (2*p[1]**2)))

        # Model some strongly tilted spectral data:
        nx, nlam = 80, 500
        x0 = np.arange(nx)
        gain, readnoise = 3.0, 30.
        background = np.ones(nlam)*10
        flux =  np.ones(nlam)*1e4
        center = nx/2. + np.linspace(0,10,nlam)
        FWHM = 3.
        model = np.array([gaussian([flux[ii]/gain, FWHM/2.35, center[ii], background[ii]], x0) for ii in range(nlam)])
        varmodel = np.abs(model) / gain + (readnoise/gain)**2
        observation = np.random.normal(model, np.sqrt(varmodel))
        fitwidth = 60
        xr = 15

        output_spec = spec.superExtract(observation, varmodel, gain, readnoise, polyspacing=0.5, pord=1, bkg_radii=[10,30], extract_radius=5, dispaxis=1)

        py.figure()
        py.plot(output_spec.spectrum.squeeze() / flux)
        py.ylabel('(Measured flux) / (True flux)')
        py.xlabel('Photoelectrons')
        


    :TO_DO:
      Iterate background fitting and reject outliers; maybe first time
      would be unweighted for robustness.

      Introduce even more array-based, rather than loop-based,
      calculations.  For large spectra computing the C-matrix takes
      the most time; this should be optimized somehow.

    :SEE_ALSO:

    """

    # 2012-08-25 20:14 IJMC: Created.
    # 2012-09-21 14:32 IJMC: Added error-trapping if no good pixels
    #                      are in a row. Do a better job of extracting
    #                      the initial 'standard' spectrum.

    from scipy import signal
    from pylab import *
    from nsdata import imshow, bfixpix



    # Parse inputs:
    frame, variance, gain, readnoise = args[0:4]

    frame    = gain * np.array(frame, copy=False)
    variance = gain**2 * np.array(variance, copy=False)
    variance[variance<=0.] = readnoise**2

    # Parse options:
    if kw.has_key('verbose'):
        verbose = kw['verbose']
    else:
        verbose = False
    if verbose: from time import time


    if kw.has_key('goodpixelmask'):
        goodpixelmask = kw['goodpixelmask']
        if isinstance(goodpixelmask, str):
            goodpixelmask = pyfits.getdata(goodpixelmask).astype(bool)
        else:
            goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)


    if kw.has_key('dispaxis'):
        dispaxis = kw['dispaxis']
    else:
        dispaxis = 0

    if dispaxis==0:
        frame = frame.transpose()
        variance = variance.transpose()
        goodpixelmask = goodpixelmask.transpose()


    if kw.has_key('pord'):
        pord = kw['pord']
    else:
        pord = 2

    if kw.has_key('polyspacing'):
        polyspacing = kw['polyspacing']
    else:
        polyspacing = 1

    if kw.has_key('bkg_radii'):
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if kw.has_key('extract_radius'):
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10
        if verbose: message("Setting option 'extract_radius' to: " + str(extract_radius))

    if kw.has_key('npoly'):
        npoly = kw['npoly']
    else:
        npoly = 2 * int((2.0 * extract_radius) / polyspacing / 2.) + 1

    if kw.has_key('bord'):
        bord = kw['bord']
    else:
        bord = 1
        if verbose: message("Setting option 'bord' to: " + str(bord))

    if kw.has_key('tord'):
        tord = kw['tord']
    else:
        tord = 3
        if verbose: message("Setting option 'tord' to: " + str(tord))

    if kw.has_key('bsigma'):
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: message("Setting option 'bsigma' to: " + str(bsigma))

    if kw.has_key('csigma'):
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: message("Setting option 'csigma' to: " + str(csigma))

    if kw.has_key('qmode'):
        qmode = kw['qmode']
    else:
        qmode = 'fast'
        if verbose: message("Setting option 'qmode' to: " + str(qmode))

    if kw.has_key('nreject'):
        nreject = kw['nreject']
    else:
        nreject = 100
        if verbose: message("Setting option 'nreject' to: " + str(nreject))

    if kw.has_key('finite'):
        finite = kw['finite']
    else:
        finite = True
        if verbose: message("Setting option 'finite' to: " + str(finite))


    if kw.has_key('retall'):
        retall = kw['retall']
    else:
        retall = False


    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    variance[True-goodpixelmask] = frame[goodpixelmask].max() * 1e9
    nlam, fitwidth = frame.shape

    # Define trace (Marsh's "X_j" in Eq. 9)
    if kw.has_key('trace'):
        trace = kw['trace']
    else:
        trace = None

    if trace is None:
        trace = 5
    if not hasattr(trace, '__iter__'):
        if verbose: print "Tracing not fully tested; dispaxis may need adjustment."
        #pdb.set_trace()
        tracecoef = traceorders(frame, pord=trace, nord=1, verbose=verbose-1, plotalot=verbose-1, g=gain, rn=readnoise, badpixelmask=True-goodpixelmask, dispaxis=dispaxis, fitwidth=min(fitwidth, 80))
        trace = np.polyval(tracecoef.ravel(), np.arange(nlam))

    
    #xxx = np.arange(-fitwidth/2, fitwidth/2)
    #backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) < bkg_radii[1])
    #extractionAperture = np.abs(xxx) < extract_radius
    #nextract = extractionAperture.sum()
    #xb = xxx[backgroundAperture]

    xxx = np.arange(fitwidth) - trace.reshape(nlam,1)
    backgroundApertures = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])
    extractionApertures = np.abs(xxx) <= extract_radius

    nextracts = extractionApertures.sum(1)

    #Step3: Sky Subtraction
    background = 0. * frame
    for ii in range(nlam):
        if goodpixelmask[ii, backgroundApertures[ii]].any():
            fit = an.polyfitr(xxx[ii,backgroundApertures[ii]], frame[ii, backgroundApertures[ii]], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundApertures[ii]], verbose=verbose-1)
            background[ii, :] = np.polyval(fit, xxx[ii])
        else:
            background[ii] = 0.

    background_at_trace = np.array([np.interp(0, xxx[j], background[j]) for j in range(nlam)])

    # (my 3a: mask any bad values)
    badBackground = True - np.isfinite(background)
    background[badBackground] = 0.
    if verbose and badBackground.any():
        print "Found bad background values at: ", badBackground.nonzero()

    skysubFrame = frame - background


    # Interpolate and fix bad pixels for extraction of standard
    # spectrum -- otherwise there can be 'holes' in the spectrum from
    # ill-placed bad pixels.
    fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)

    #Step4: Extract 'standard' spectrum and its variance
    standardSpectrum = np.zeros((nlam, 1), dtype=float)
    varStandardSpectrum = np.zeros((nlam, 1), dtype=float)
    for ii in range(nlam):
        thisrow_good = extractionApertures[ii] #* goodpixelmask[ii] * 
        standardSpectrum[ii] = fixSkysubFrame[ii, thisrow_good].sum()
        varStandardSpectrum[ii] = variance[ii, thisrow_good].sum()


    spectrum = standardSpectrum.copy()
    varSpectrum = varStandardSpectrum

    # Define new indices (in Marsh's appendix):
    N = pord + 1
    mm = np.tile(np.arange(N).reshape(N,1), (npoly)).ravel()
    nn = mm.copy()
    ll = np.tile(np.arange(npoly), N)
    kk = ll.copy()
    pp = N * ll + mm
    qq = N * kk + nn

    jj = np.arange(nlam)  # row (i.e., wavelength direction)
    ii = np.arange(fitwidth) # column (i.e., spatial direction)
    jjnorm = np.linspace(-1, 1, nlam) # normalized X-coordinate
    jjnorm_pow = jjnorm.reshape(1,1,nlam) ** (np.arange(2*N-1).reshape(2*N-1,1,1))

    # Marsh eq. 9, defining centers of each polynomial:
    constant = 0.  # What is it for???
    poly_centers = trace.reshape(nlam, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1) + constant


    # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
    #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
    if verbose: tic = time() 
    if qmode=='fast-nearest': # Array-based nearest-neighbor mode.
        if verbose: tic = time()
        Q = np.array([np.zeros((npoly, fitwidth, nlam)), np.array([polyspacing * np.ones((npoly, fitwidth, nlam)), 0.5 * (polyspacing+1) - np.abs((poly_centers - ii.reshape(fitwidth, 1, 1)).transpose(2, 0, 1))]).min(0)]).max(0)

    elif qmode=='slow-linear': # Code is a mess, but it works.
        invs = 1./polyspacing
        poly_centers_over_s = poly_centers / polyspacing
        xps_mat = poly_centers + polyspacing
        xms_mat = poly_centers - polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for i in range(fitwidth):
            ip05 = i + 0.5
            im05 = i - 0.5
            for j in range(nlam):
                for k in range(npoly):
                    xkj = poly_centers[j,k]
                    xkjs = poly_centers_over_s[j,k]
                    xps = xps_mat[j,k] #xkj + polyspacing
                    xms = xms_mat[j,k] # xkj - polyspacing

                    if (ip05 <= xms) or (im05 >= xps):
                        qval = 0.
                    elif (im05) > xkj:
                        lim1 = im05
                        lim2 = min(ip05, xps)
                        qval = (lim2 - lim1) * \
                            (1. + xkjs - 0.5*invs*(lim1+lim2))
                    elif (ip05) < xkj:
                        lim1 = max(im05, xms)
                        lim2 = ip05
                        qval = (lim2 - lim1) * \
                            (1. - xkjs + 0.5*invs*(lim1+lim2))
                    else:
                        lim1 = max(im05, xms)
                        lim2 = min(ip05, xps)
                        qval = lim2 - lim1 + \
                            invs * (xkj*(-xkj + lim1 + lim2) - \
                                        0.5*(lim1*lim1 + lim2*lim2))
                    Q[k,i,j] = max(0, qval)


    elif qmode=='fast-linear': # Code is a mess, but it's faster than 'slow' mode
        invs = 1./polyspacing
        xps_mat = poly_centers + polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for j in range(nlam):
            xkj_vec = np.tile(poly_centers[j,:].reshape(npoly, 1), (1, fitwidth))
            xps_vec = np.tile(xps_mat[j,:].reshape(npoly, 1), (1, fitwidth))
            xms_vec = xps_vec - 2*polyspacing

            ip05_vec = np.tile(np.arange(fitwidth) + 0.5, (npoly, 1))
            im05_vec = ip05_vec - 1
            ind00 = ((ip05_vec <= xms_vec) + (im05_vec >= xps_vec))
            ind11 = ((im05_vec > xkj_vec) * (True - ind00))
            ind22 = ((ip05_vec < xkj_vec) * (True - ind00))
            ind33 = (True - (ind00 + ind11 + ind22)).nonzero()
            ind11 = ind11.nonzero()
            ind22 = ind22.nonzero()

            n_ind11 = len(ind11[0])
            n_ind22 = len(ind22[0])
            n_ind33 = len(ind33[0])

            if n_ind11 > 0:
                ind11_3d = ind11 + (np.ones(n_ind11, dtype=int)*j,)
                lim2_ind11 = np.array((ip05_vec[ind11], xps_vec[ind11])).min(0)
                Q[ind11_3d] = ((lim2_ind11 - im05_vec[ind11]) * invs * \
                                   (polyspacing + xkj_vec[ind11] - 0.5*(im05_vec[ind11] + lim2_ind11)))
            
            if n_ind22 > 0:
                ind22_3d = ind22 + (np.ones(n_ind22, dtype=int)*j,)
                lim1_ind22 = np.array((im05_vec[ind22], xms_vec[ind22])).max(0)
                Q[ind22_3d] = ((ip05_vec[ind22] - lim1_ind22) * invs * \
                                   (polyspacing - xkj_vec[ind22] + 0.5*(ip05_vec[ind22] + lim1_ind22)))
            
            if n_ind33 > 0:
                ind33_3d = ind33 + (np.ones(n_ind33, dtype=int)*j,)
                lim1_ind33 = np.array((im05_vec[ind33], xms_vec[ind33])).max(0)
                lim2_ind33 = np.array((ip05_vec[ind33], xps_vec[ind33])).min(0)
                Q[ind33_3d] = ((lim2_ind33 - lim1_ind33) + invs * \
                                   (xkj_vec[ind33] * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33) - 0.5*(lim1_ind33*lim1_ind33 + lim2_ind33*lim2_ind33)))
            

    elif qmode=='brute': # Neither accurate, nor memory-frugal.
        oversamp = 4.
        jj2 = np.arange(nlam*oversamp, dtype=float) / oversamp
        trace2 = np.interp(jj2, jj, trace)
        poly_centers2 = trace2.reshape(nlam*oversamp, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1, dtype=float) + constant
        x2 = np.arange(fitwidth*oversamp, dtype=float)/oversamp
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in range(npoly):
            Q[k,:,:] = an.binarray((np.abs(x2.reshape(fitwidth*oversamp,1) - poly_centers2[:,k]) <= polyspacing), oversamp)

        Q /= oversamp*oversamp*2

    else:  # 'slow' Loop-based nearest-neighbor mode: requires less memory
        if verbose: tic = time()
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in range(npoly):
            for i in range(fitwidth):
                for j in range(nlam):
                    Q[k,i,j] = max(0, min(polyspacing, 0.5*(polyspacing+1) - np.abs(poly_centers[j,k] - i)))

    if verbose: print '%1.2f s to compute Q matrix (%s mode)' % (time() - tic, qmode)
        

    # Some quick math to find out which dat columns are important, and
    #   which contain no useful spectral information:
    Qmask = Q.sum(0).transpose() > 0
    Qind = Qmask.transpose().nonzero()
    Q_cols = [Qind[0].min(), Qind[0].max()]
    nQ = len(Qind[0])
    Qsm = Q[:,Q_cols[0]:Q_cols[1]+1,:]

    # Prepar to iteratively clip outliers
    newBadPixels = True
    iter = -1
    if verbose: print "Looking for bad pixel outliers."
    while newBadPixels:
        iter += 1
        if verbose: print "Beginning iteration %i" % iter


        # Compute pixel fractions (Marsh Eq. 5):
        #     (Note that values outside the desired polynomial region
        #     have Q=0, and so do not contribute to the fit)
        #E = (skysubFrame / spectrum).transpose()
        invEvariance = (spectrum**2 / variance).transpose()
        weightedE = (skysubFrame * spectrum / variance).transpose() # E / var_E
        invEvariance_subset = invEvariance[Q_cols[0]:Q_cols[1]+1,:]

        # Define X vector (Marsh Eq. A3):
        if verbose: tic = time()
        X = np.zeros(N * npoly, dtype=float)
        X0 = np.zeros(N * npoly, dtype=float)
        for q in qq:
            X[q] = (weightedE[Q_cols[0]:Q_cols[1]+1,:] * Qsm[kk[q],:,:] * jjnorm_pow[nn[q]]).sum() 
        if verbose: print '%1.2f s to compute X matrix' % (time() - tic)

        # Define C matrix (Marsh Eq. A3)
        if verbose: tic = time()
        C = np.zeros((N * npoly, N*npoly), dtype=float)

        buffer = 1.1 # C-matrix computation buffer (to be sure we don't miss any pixels)
        for p in pp:
            qp = Qsm[ll[p],:,:]
            for q in qq:
                #  Check that we need to compute C:
                if np.abs(kk[q] - ll[p]) <= (1./polyspacing + buffer):
                    if q>=p: 
                        # Only compute over non-zero columns:
                        C[q, p] = (Qsm[kk[q],:,:] * qp * jjnorm_pow[nn[q]+mm[p]] * invEvariance_subset).sum() 
                    if q>p:
                        C[p, q] = C[q, p]


        if verbose: print '%1.2f s to compute C matrix' % (time() - tic)

        ##################################################
        ##################################################
        # Just for reference; the following is easier to read, perhaps, than the optimized code:
        if False: # The SLOW way to compute the X vector:
            X2 = np.zeros(N * npoly, dtype=float)
            for n in nn:
                for k in kk:
                    q = N * k + n
                    xtot = 0.
                    for i in ii:
                        for j in jj:
                            xtot += E[i,j] * Q[k,i,j] * (jjnorm[j]**n) / Evariance[i,j]
                    X2[q] = xtot

            # Compute *every* element of C (though most equal zero!)
            C = np.zeros((N * npoly, N*npoly), dtype=float)
            for p in pp:
                for q in qq:
                    if q>=p:
                        C[q, p] = (Q[kk[q],:,:] * Q[ll[p],:,:] * (jjnorm.reshape(1,1,nlam)**(nn[q]+mm[p])) / Evariance).sum()
                    if q>p:
                        C[p, q] = C[q, p]
        ##################################################
        ##################################################

        # Solve for the profile-polynomial coefficients (Marsh Eq. A)4: 
        if np.abs(np.linalg.det(C)) < 1e-10:
            Bsoln = np.dot(np.linalg.pinv(C), X)
        else:
            Bsoln = np.linalg.solve(C, X)

        Asoln = Bsoln.reshape(N, npoly).transpose()

        # Define G_kj, the profile-defining polynomial profiles (Marsh Eq. 8)
        Gsoln = np.zeros((npoly, nlam), dtype=float)
        for n in range(npoly):
            Gsoln[n] = np.polyval(Asoln[n,::-1], jjnorm) # reorder polynomial coef.


        # Compute the profile (Marsh eq. 6) and normalize it:
        if verbose: tic = time()
        profile = np.zeros((fitwidth, nlam), dtype=float)
        for i in range(fitwidth):
            profile[i,:] = (Q[:,i,:] * Gsoln).sum(0)

        #P = profile.copy() # for debugging 
        if profile.min() < 0:
            profile[profile < 0] = 0. 
        profile /= profile.sum(0).reshape(1, nlam)
        profile[True - np.isfinite(profile)] = 0.
        if verbose: print '%1.2f s to compute profile' % (time() - tic)

        #Step6: Revise variance estimates 
        modelSpectrum = spectrum * profile.transpose()
        modelData = modelSpectrum + background
        variance0 = np.abs(modelData) + readnoise**2
        variance = variance0 / (goodpixelmask + 1e-9) # De-weight bad pixels, avoiding infinite variance

        outlierVariances = (frame - modelData)**2/variance

        if outlierVariances.max() > csigma**2:
            newBadPixels = True
            # Base our nreject-counting only on pixels within the spectral trace:
            maxRejectedValue = max(csigma**2, np.sort(outlierVariances[Qmask])[-nreject])
            worstOutliers = (outlierVariances>=maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            numberRejected = len(worstOutliers[0])
            #pdb.set_trace()
        else:
            newBadPixels = False
            numberRejected = 0
        
        if verbose: print "Rejected %i pixels on this iteration " % numberRejected

            
        # Optimal Spectral Extraction: (Horne, Step 8)
        fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)
        spectrum = np.zeros((nlam, 1), dtype=float)
        #spectrum1 = np.zeros((nlam, 1), dtype=float)
        varSpectrum = np.zeros((nlam, 1), dtype=float)
        goodprof =  profile.transpose() #* goodpixelmask
        for ii in range(nlam):
            thisrow_good = extractionApertures[ii] #* goodpixelmask[ii]
            denom = (goodprof[ii, thisrow_good] * profile.transpose()[ii, thisrow_good] / variance0[ii, thisrow_good]).sum()
            if denom==0:
                spectrum[ii] = 0.
                varSpectrum[ii] = 9e9
            else:
                spectrum[ii] = (goodprof[ii, thisrow_good] * skysubFrame[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
                #spectrum1[ii] = (goodprof[ii, thisrow_good] * modelSpectrum[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
                varSpectrum[ii] = goodprof[ii, thisrow_good].sum() / denom
            #if spectrum.size==1218 and ii>610:
            #    pdb.set_trace()

        #if spectrum.size==1218: pdb.set_trace()

    ret = baseObject()
    ret.spectrum = spectrum
    ret.raw = standardSpectrum
    ret.varSpectrum = varSpectrum
    ret.trace = trace
    ret.units = 'electrons'
    ret.background = background_at_trace

    ret.function_name = 'spec.superExtract'

    if retall:
        ret.profile_map = profile
        ret.extractionApertures = extractionApertures
        ret.background_map = background
        ret.variance_map = variance0
        ret.goodpixelmask = goodpixelmask
        ret.function_args = args
        ret.function_kw = kw

    return  ret


         


def spextractor(frame, gain, readnoise, framevar=None, badpixelmask=None, mode='superExtract', trace=None, options=None, trace_options=None, verbose=False):
    """Extract a spectrum from a frame using one of several methods.

    :INPUTS:
      frame : 2D Numpy array or filename
        Contains a single spectral trace.

      gain : None or scalar
        Gain of data contained in 'frame;' i.e., number of collected
        photoelectrons equals frame * gain.

      readnoise : None or scalar
        Read noise of detector, in electrons.

      framevar : None, 2D Numpy array, or filename
        Variance of values in 'frame.'

        If and only if framevar is None, use gain and readnoise to
        compute variance.

      badpixelmask : None, 2D Numpy array, or filename
        Mask of bad pixels in 'frame.'  Bad pixels are set to 1, good
        pixels are set to 0.

      mode : str
        Which spectral extraction mode to use.  Options are:
        
          superExtract -- see :func:`superExtract`

          optimalExtract -- see :func:`optimalExtract`

          spline -- see :func:`extractSpectralProfiles`
                    Must also input trace_options.

      trace : None, or 1D Numpy Array
        Spectral trace location: fractional pixel index along the
        entire spectral trace.  If None, :func:`traceorders` will be
        called using the options in 'trace_options.'

      options : None or dict
        Keyword options to be passed to the appropriate spectral
        extraction algorithm. Note that you should be able to pass the
        same sets of parameters to :func:`superExtract` and
        :func:`optimalExtract` (the necessary parameter sets overlap,
        but are not identical).

      trace_options : None or dict
        Keyword options to be passed to :func:`traceorders` (if no
        trace is input, or if mode='spline')


    :RETURNS:
      spectrum, error or variance of spectrum, sky background, ...
      
    :NOTES:
      When 'optimalextract' is used: if len(bkg_radii)==2 then the
      background apertures will be reset based on the median location
      of the trace.  If extract_radius is a singleton, it will be
      similarly redefined.

    :EXAMPLE:
     ::

      frame = pyfits.getdata('spectral_filename.fits')
      gain, readnoise = 3.3, 30
      output = spec.spextractor(frame, gain, readnoise, mode='superExtract', \
               options=dict(bord=2, bkg_radii=[20, 30], extract_radius=15, \
               polyspacing=1./3, pord=5, verbose=True, trace=trace, \
               qmode='slow-linear'))

      output2 = spec.spextractor(frame, gain, readnoise, mode='optimalExtract', \
                options=dict(bkg_radii=[20,30], extract_radius=15, bord=2, \
                bsigma=3, pord=3, psigma=8, csigma=5, verbose=1))


        """
    # 2012-09-03 11:12 IJMC: Created
    from tools import array_or_filename

    # Parse inputs:
    frame    = array_or_filename(frame)
    framevar = array_or_filename(framevar, noneoutput=np.abs(frame) / gain + (readnoise / gain)**2)

    if options is None:
        options = dict(dispaxis=0)


    if verbose and not options.has_key('verbose'):
        options['verbose'] = verbose

    if trace is None:
        if trace_options is None:
            trace_options = dict(pord=2)
        if not trace_options.has_key('dispaxis'):
            if options.has_key('dispaxis'): trace_options['dispaxis'] = options['dispaxis']

        trace_coef = traceorders(frame, **trace_options) 
        trace = np.polyval(trace_coef.ravel(), np.arange(frame.shape[1-options['dispaxis']]))

    options['trace'] = trace

    ##################################################
    # Extract spectrum in one of several ways:
    ##################################################
    if mode.lower()=='superextract':
        ret = superExtract(frame, framevar, gain, readnoise, **options)

    elif mode.lower()=='spline':
        # First, set things up:
        try:
            trace_coef += 0.0
        except:
            trace_coef = np.polyfit(np.arange(trace.size), trace, trace_options['pord']).reshape(1, trace_options['pord']+1)
        trace_options['retall'] = True

        if not trace_options.has_key('bkg_radii'):
            if options.has_key('bkg_radii') and len(options['bkg_radii'])==2:
                trace_options['bkg_radii'] = options['bkg_radii']
        if not trace_options.has_key('extract_radius'):
            if options.has_key('extract_radius') and not hasattr(options['extract_radius'], '__iter__'):
                trace_options['extract_radius'] = options['extract_radius']

        prof = makeprofile(frame, trace_coef, **trace_options)

        ret = extractSpectralProfiles(prof, **trace_options)


    elif mode.lower()=='optimalextract':
        # First, re-define bkg_radii and extract_radius (if necessary):
        options = dict().update(options) # prevents alterations of options
        if options.has_key('bkg_radii') and len(options['bkg_radii'])==2:
            t0 = np.median(trace)
            bkg_radii = [t0-options['bkg_radii'][1], t0-options['bkg_radii'][0],
                         t0+options['bkg_radii'][0], t0+options['bkg_radii'][1]]
            options['bkg_radii'] = bkg_radii
            if verbose: print "Re-defining background apertures: ", bkg_radii

        if options.has_key('extract_radius') and \
                (not hasattr(options['extract_radius'], '__iter__') or \
                     (len(options['extract_radius'])==1)):
            extract_radius = [t0 - options['extract_radius'], \
                                  t0 + options['extract_radius']]
            options['extract_radius'] = extract_radius
            if verbose: print "Re-defining extraction aperture: ", extract_radius

        ret = optimalExtract(frame, framevar, gain, readnoise, **options)

    else:
        print "No valid spectral extraction mode specified!"
        ret = -1


    return ret


def scaleSpectralSky_pca(frame, skyframes, variance=None, mask=None, badpixelmask=None, npca=3, gain=3.3, readnoise=30):
    """
    Use PCA and blank sky frames to subtract

    frame : str or NumPy array
      data frame to subtract sky from. Assumed to be in ADU, not
      electrons (see gain and readnoise)
    
    npca : int
      number of PCA components to remove

    f0 = pyfits.getdata(odome.procsci[0])
    mask = pyfits.getdata(odome._proc + 'skyframes_samplemask.fits').astype(bool)
    badpixelmask = pyfits.getdata( odome.badpixelmask).astype(bool)


    The simplest way to fit sky to a 'frame' containing bright
    spectral is to include the spectral-trace regions in 'mask' but
    set the 'variance' of those regions extremely high (to de-weight
    them in the least-squares fit).

    To use for multi-object data, consider running multiple times
    (once per order)

    Returns the best-fit sky frame as determined from the first 'npca'
    PCA components.
    """
    # 2012-09-17 16:41 IJMC: Created

    from scipy import sparse
    from pcsa import pca

    # Parse inputs
    if not isinstance(frame, np.ndarray):
        frame = pyfits.getdata(frame)

    if variance is None:
        variance = np.abs(frame)/gain + (readnoise/gain)**2
    else:
        variance = np.array(variance, copy=False)
    weights = 1./variance

    nx, ny = frame.shape
    n_tot = nx*ny

    if badpixelmask is None:
        badpixelmask = np.zeros((nx, ny), dtype=bool)
    elif isinstance(badpixelmask, str):
        badpixelmask = pyfits.getdata(badpixelmask).astype(bool)
    else:
        badpixelmask = np.array(badpixelmask).astype(bool)
    weights[badpixelmask] = 0.


    if mask is None:
        mask = np.ones(frame.shape, dtype=bool)
        n_elem = n_tot
    else:
        n_elem = mask.astype(bool).sum()
    maskind = mask.nonzero()

    if isinstance(skyframes, np.ndarray) and skyframes.ndim==3:
        pass
    else:
        skyframes = np.array([pyfits.getdata(fn) for fn in skyframes])

    skyframes_ind = np.array([sf[maskind] for sf in skyframes])

    frame_ind = frame[maskind]
    pcaframes = np.zeros((npca, n_elem), dtype=np.float32)
    iter = 0

    pcas = []
    for jj in range(npca):
        pcas.append(pca(skyframes_ind, None, ord=jj+1))
    pcaframes[0] = pcas[0][0][0]
    for jj in range(1, npca):
        pcaframes[jj] = (pcas[jj][0] - pcas[jj-1][0])[0]

#    del pcas

    svecs = sparse.csr_matrix(pcaframes).transpose()
    fitcoef, efitcoef = an.lsqsp(svecs, frame_ind, weights[maskind])
    skyvalues = (fitcoef.reshape(npca, 1) * pcaframes).sum(0)
    #eskyvalues = (np.diag(efitcoef).reshape(npca, 1) * pcaframes).sum(0)

    skyframe = np.zeros((nx, ny), dtype=float)
    skyframe[maskind] = skyvalues
    return skyframe

def scaleSpectralSky_dsa(subframe, variance=None, badpixelmask=None, nk=31, pord=1, nmed=3, gain=3.3, readnoise=30, dispaxis=0, spatial_index=None):
    """
    Use difference-imaging techniques to subtract moderately tilted
    sky background.  Doesn't work so well!

    subframe : NumPy array
      data subframe containing sky data to be subtracted (and,
      perhaps, an object's spectral trace). Assumed to be in ADU, not
      electrons (see gain and readnoise)
    
    variance : None or NumPy array
      variance of each pixel in 'subframe'

    nmed : int
      size of 2D median filter

    pord : int
      degree of spectral tilt.  Keep this number low!

    nk : int
      Number of kernel pixels in :func:`dia.dsa`

    nmed : int
      Size of window for 2D median filter (to reject bad pixels, etc.)

    dispaxis : int
        set dispersion axis: 0 = horizontal and 1 = vertical

    gain, readnoise : ints
       If 'variance' is None, these are used to estimate the uncertainties.

    spatial_index : None, or 1D NumPy array of type bool
       Which spatial rows (if dispaxis=0) to use when fitting the tilt
       of sky lines across the spectrum.  If you want to use all, set
       to None.  If you want to ignore some (e.g., because there's a
       bright object's spectrum there) then set those rows' elements
       of spatial_index to 'False'.

    :NOTES:
       Note that (in my experience!) this approach works better when
       you set all weights to unity, rather than using the suggested
       (photon + read noise) variances.
    

    Returns the best-fit sky frame.
    """
    # 2012-09-17 16:41 IJMC: Created

    from scipy import signal
    import dia

    # Parse inputs
    if not isinstance(subframe, np.ndarray):
        subframe = pyfits.getdata(subframe)

    if variance is None:
        variance = np.abs(subframe)/gain + (readnoise/gain)**2
    else:
        variance = np.array(variance, copy=False)
    weights = 1./variance

    if badpixelmask is None:
        badpixelmask = np.zeros(subframe.shape, dtype=bool)
    elif isinstance(badpixelmask, str):
        badpixelmask = pyfits.getdata(badpixelmask).astype(bool)
    else:
        badpixelmask = np.array(badpixelmask).astype(bool)
    weights[badpixelmask] = 0.

    if dispaxis==1:
        subframe = subframe.transpose()
        variance = variance.transpose()
        weights = weights.transpose()
        badpixelmask = badpixelmask.transpose()

    sub = subframe
    if nmed > 1:
        ss = signal.medfilt2d(sub, nmed)
    else:
        ss = sub.copy()
    ref = np.median(ss, axis=0)

    n, nlam = ss.shape
    if spatial_index is None:
        spatial_index = np.arange(n)
    else:
        spatial_index = np.array(spatial_index, copy=False).nonzero()


    gaussianpar = np.zeros((n, 4))
    for ii in range(n):
        test = dia.dsa(ref, ss[ii], nk, w=weights[ii], noback=True)
        gaussianpar[ii] = fitGaussian(test[1])[0]

    position_fit = an.polyfitr(np.arange(n)[spatial_index], gaussianpar[:,2][spatial_index], pord, 3)
    positions = np.polyval(position_fit, np.arange(n))
    width_fit = np.median(gaussianpar[:,1])

    skyframe = 0*ss
    testDC_spec = np.ones(nlam)
    testx = np.arange(nk, dtype=float)
    lsqfits = np.zeros((n,2))
    for ii in range(n):
        testgaussian = gaussian([1, width_fit, positions[ii], 0.], testx)
        testgaussian /= testgaussian.sum()
        testspectrum = dia.rconvolve1d(ref, testgaussian)
        skyframe[ii] = testspectrum

    if dispaxis==1:
        skyframe = skyframe.transpose()

    return skyframe



def scaleSpectralSky_cor(subframe, badpixelmask=None, maxshift=20, fitwidth=2, pord=1, nmed=3, dispaxis=0, spatial_index=None, refpix=None, tord=2):
    """
    Use cross-correlation to subtract tilted sky backgrounds.

    subframe : NumPy array
      data subframe containing sky data to be subtracted (and,
      perhaps, an object's spectral trace).
    
    badpixelmask : None or NumPy array
      A boolean array, equal to zero for good pixels and unity for bad
      pixels.  If this is set, the first step will be a call to
      :func:`nsdata.bfixpix` to interpolate over these values.
      
    nmed : int
      size of 2D median filter for pre-smoothing.

    pord : int
      degree of spectral tilt.  Keep this number low!

    maxshift : int
      Maximum acceptable shift.  NOT YET IMPLEMENTED!

    fitwidth : int
      Maximum radius (in pixels) for fitting to the peak of the
      cross-correlation.
      
    nmed : int
      Size of window for 2D median filter (to reject bad pixels, etc.)

    dispaxis : int
        set dispersion axis: 0 = horizontal and 1 = vertical

    spatial_index : None, or 1D NumPy array of type *bool*
       Which spatial rows (if dispaxis=0) to use when fitting the tilt
       of sky lines across the spectrum.  If you want to use all, set
       to None.  If you want to ignore some (e.g., because there's a
       bright object's spectrum there) then set those rows' elements
       of spatial_index to 'False'.

    refpix : scalar
       Pixel along spatial axis to which spectral fits should be
       aligned; if a spectral trace is present, one should set
       "refpix" to the location of the trace.

    tord : int
       Order of polynomial fits along spatial direction in aligned
       2D-spectral frame, to account for misalignments or
       irregularities of tilt direction.

    :RETURNS:
      a model of the sky background, of the same shape as 'subframe.'
    """
    # 2012-09-22 17:04 IJMC: Created
    # 2012-12-27 09:53 IJMC: Edited to better account for sharp edges
    #                        in backgrounds.

    from scipy import signal
    from nsdata import bfixpix

    # Parse inputs
    if not isinstance(subframe, np.ndarray):
        subframe = pyfits.getdata(subframe)

    if badpixelmask is None:
        pass
    else:
        badpixelmask = np.array(badpixelmask).astype(bool)
        subframe = bfixpix(subframe, badpixelmask, retdat=True)

    if dispaxis==1:
        subframe = subframe.transpose()

    # Define necessary variables and vectors:
    npix, nlam = subframe.shape
    if spatial_index is None:
        spatial_index = np.ones(npix, dtype=bool)
    else:
        spatial_index = np.array(spatial_index, copy=False)
    if refpix is None:
        refpix = npix/2.

    lampix = np.arange(nlam)
    tpix = np.arange(npix)
    alllags = np.arange(nlam-maxshift*2) - np.floor(nlam/2 - maxshift)

    # Median-filter the input data:
    if nmed > 1:
        ssub = signal.medfilt2d(subframe, nmed)
    else:
        ssub = subframe.copy()
    ref = np.median(ssub, axis=0)


    #allcor = np.zeros((npix, nlam-maxshift*2))
    shift = np.zeros(npix, dtype=float)
    for ii in tpix:
        # Cross-correlate to measure alignment at each row:
        cor = np.correlate(ref[maxshift:-maxshift], signal.medfilt(ssub[ii], nmed)[maxshift:-maxshift], mode='same')
        # Measure offset of each row:
        maxind = alllags[(cor==cor.max())].mean()
        fitind = np.abs(alllags - maxind) <= fitwidth
        quadfit = np.polyfit(alllags[fitind], cor[fitind], 2)
        shift[ii] = -0.5 * quadfit[1] / quadfit[0]

    shift_polyfit = an.polyfitr(tpix[spatial_index], shift[spatial_index], pord, 3) #, w=weights)
    refpos = np.polyval(shift_polyfit, refpix)
    #pdb.set_trace()
    fitshift = np.polyval(shift_polyfit, tpix) - refpos

    # Interpolate each row to a common frame to create an improved reference:
    newssub = np.zeros((npix, nlam))
    for ii in tpix:
        newssub[ii] = np.interp(lampix, lampix+fitshift[ii], ssub[ii])

    #pdb.set_trace()
    newref = np.median(newssub[spatial_index,:], axis=0)

    tfits = np.zeros((nlam, tord+1), dtype=float)
    newssub2 = np.zeros((npix, nlam))
    for jj in range(nlam):
        tfits[jj] = an.polyfitr(tpix, newssub[:,jj], tord, 3)
        newssub2[:, jj] = np.polyval(tfits[jj], tpix)


    # Create the final model of the sky background:
    skymodel = np.zeros((npix, nlam), dtype=float)
    shiftmodel = np.zeros((npix, nlam), dtype=float)
    for ii in tpix:
        #skymodel[ii] = np.interp(lampix, lampix-fitshift[ii], newref)
        skymodel[ii] = np.interp(lampix, lampix-fitshift[ii], newssub2[ii])
        shiftmodel[ii] = np.interp(lampix, lampix+fitshift[ii], ssub[ii])

    #pdb.set_trace()

    if dispaxis==1:
        skymodel = skymodel.transpose()

    return skymodel, shiftmodel, newssub, newssub2





def defringe_sinusoid(subframe, badpixelmask=None, nmed=5, dispaxis=0, spatial_index=None, period_limits=[20, 100], retall=False, gain=None, readnoise=None, bictest=False, sinonly=False):
    """
    Use simple fitting to subtract fringes and sky background.

    subframe : NumPy array
      data subframe containing sky data to be subtracted (and,
      perhaps, an object's spectral trace). Assumed to be in ADU, not
      electrons (see gain and readnoise)
    
    nmed : int
      Size of window for 2D median filter (to reject bad pixels, etc.)

    dispaxis : int
        set dispersion axis: 0 = horizontal and 1 = vertical

    spatial_index : None, or 1D NumPy array of type bool
       Which spatial rows (if dispaxis=0) to use when fitting the tilt
       of sky lines across the spectrum.  If you want to use all, set
       to None.  If you want to ignore some (e.g., because there's a
       bright object's spectrum there) then set those rows' elements
       of spatial_index to 'False'.

     period_limits : 2-sequence
       Minimum and maximum periods (in pixels) of fringe signals to
       accept as 'valid.' Resolution elements with best-fit periods
       outside this range will only be fit by a linear trend.

     gain : scalar
       Gain of detector, in electrons per ADU (where 'subframe' is in
       units of ADUs).

     readnoise : scalar
       Readnoise of detector, in electrons (where 'subframe' is in
       units of ADUs).

     bictest : bool
       If True, use 'gain' and 'readnoise' to compute the Bayesian
       Information Criterion (BIC) for each fit; a sinusoid is only
       removed if BIC(sinusoid fit) is lower than BIC(constant fit).

     sinonly : bool
       If True, the output "model 2D spectrum" will only contain the
       sinusoidal component.  Otherwise, it will contain DC and
       linear-trend terms.

    :NOTES:
       Note that (in my experience!) this approach works better when
       you set all weights to unity, rather than using the suggested
       (photon + read noise) variances.

    :REQUIREMENTS:
       :doc:`analysis` (for :func:`analysis.fmin`)

       :doc:`phasecurves` (for :func:`phasecurves.errfunc`)

       :doc:`lomb` (for Lomb-Scargle periodograms)

       SciPy 'signal' module (for median-filtering)
    

    Returns the best-fit sky frame.
    """
    # 2012-09-19 16:22 IJMC: Created
    # 2012-12-16 15:11 IJMC: Made some edits to fix bugs, based on outlier indexing.

    from scipy import signal
    import lomb
    from analysis import fmin # scipy.optimize.fmin would also be O.K.
    from phasecurves import errfunc

    twopi = np.pi*2

    fudgefactor = 1.5

    def ripple(params, x):  # Sinusoid + constant offset
        if params[1]==0:
            ret = 0.0
        else:
            ret = params[0] * np.cos(twopi*x/params[1] - params[2]) + params[3]
        return ret

    def linripple(params, x): # Sinusoid + linear trend
        if params[1]==0:
            ret = 0.0
        else:
            ret = params[0] * np.cos(twopi*x/params[1] - params[2]) + params[3] + params[4]*x
        return ret

    # Parse inputs
    if not isinstance(subframe, np.ndarray):
        subframe = pyfits.getdata(subframe)

    if gain is None: gain = 1
    if readnoise is None: readnoise = 1

    if badpixelmask is None:
        badpixelmask = np.zeros(subframe.shape, dtype=bool)
    elif isinstance(badpixelmask, str):
        badpixelmask = pyfits.getdata(badpixelmask).astype(bool)
    else:
        badpixelmask = np.array(badpixelmask).astype(bool)

    if dispaxis==1:
        subframe = subframe.transpose()
        badpixelmask = badpixelmask.transpose()

    sub = subframe
    if nmed > 1:
        sdat = signal.medfilt2d(sub, nmed)
    else:
        sdat = sub.copy()

    if bictest:
        var_sdat = np.abs(sdat)/gain + (readnoise/gain)**2

    npix, nlam = sdat.shape
    if spatial_index is None:
        spatial_index = np.ones(npix, dtype=bool)
    else:
        spatial_index = np.array(spatial_index, copy=False).astype(bool)

    ##############################

    periods = np.logspace(0.5, np.log10(npix), npix*2)

    x = np.arange(npix)
    allfits = np.zeros((nlam, 5))

    for jj in range(nlam):
        vec = sdat[:, jj].copy()
        this_index = spatial_index * (True - badpixelmask[:, jj])

        #if jj==402: pdb.set_trace()
        if this_index.sum() > 1:
            # LinFit the data;  exclude values inconsistent with a sinusoid:
            linfit = an.polyfitr(x[this_index], vec[this_index], 1, 3)
            linmodel = (x) * linfit[0] + linfit[1]
            vec2 = vec - (linmodel - linfit[1])
            maxexcursion = an.dumbconf(vec2[this_index], .683)[0] * (fudgefactor / .88)
            index = this_index * (np.abs(vec2 - np.median(vec2[this_index])) <= (6*maxexcursion))

            # Use Lomb-Scargle to find strongest period:
            #lsp = lomb.lomb(vec2[index], x[index], twopi/periods)
            freqs, lsp = lomb.fasper(x[index], vec2[index], 12., 0.5)
            guess_period = (1./freqs[lsp==lsp.max()]).mean()
            # If the best-fit period is within our limits, fit it:
            if (guess_period <= period_limits[1]) and (guess_period >= period_limits[0]):
                #periods2 = np.arange(guess_period-1, guess_period+1, 0.02)
                #lsp2 = lomb.lomb(vec2[index], x[index], twopi/periods2)
                #guess_period = periods2[lsp2==lsp2.max()].mean()
                guess_dc = np.median(vec2[index])
                guess_amp = an.dumbconf(vec2[index], .683, mid=guess_dc)[0] / 0.88
                guess = [guess_amp, guess_period, np.pi, guess_dc]
                if bictest:
                    w = 1./var_sdat[:,jj][index]
                else:
                    w = np.ones(index.sum())

                fit = fmin(errfunc, guess, args=(ripple, x[index], vec2[index], w), full_output=True, disp=False)
                guess2 = np.concatenate((fit[0], [linfit[0]]))
                fit2 = fmin(errfunc, guess2, args=(linripple, x[index], vec[index], w), full_output=True, disp=False)
                if bictest:
                    ripple_model = linripple(fit2[0], x)
                    bic_ripple = (w*(vec - ripple_model)[index]**2).sum() + 6*np.log(index.sum())
                    bic_linfit = (w*(vec - linmodel)[index]**2).sum() + 2*np.log(index.sum())
                    #if jj==546:
                    #    pdb.set_trace()
                    if bic_ripple >= bic_linfit:
                        fit2 = [[0, 1e9, 0, linfit[1], linfit[0]], 0]

                

            else: # Clearly not a valid ripple -- just use the linear fit.
                fit2 = [[0, 1e11, 0, linfit[1], linfit[0]], 0]

        else: # *NO* good pixels!
            #linfit = an.polyfitr(x[spatial_index], vec[spatial_index], 1, 3)
            
            fit2 = [[0, 1e11, 0, np.median(vec[spatial_index]), 0], 0]

        allfits[jj] = fit2[0]


    # Median filter the resulting coefficients
    if nmed > 1: # No median-filtering
        newfits = np.array([signal.medfilt(fits, nmed) for fits in allfits.transpose()]).transpose()
        newfits[:, 2] = allfits[:,2] # Don't smooth phase
    else:
        newfits = allfits

    # Generate the model sky pattern
    skymodel = np.zeros((npix, nlam))
    for jj in range(nlam):
        if sinonly:
            coef = newfits[jj].copy()
            coef[3:] = 0.
        else:
            coef = newfits[jj]
        skymodel[:, jj] = linripple(coef, x)


    if dispaxis==1:
        skymodel = skymodel.transpose()


    if retall:
        ret = skymodel, allfits
    else:
        ret = skymodel

    return ret



#
def makexflat(subreg, xord, nsigma=3, minsnr=10, minfrac=0.5, niter=1):
    """Helper function for XXXX.

    :INPUTS:
      subreg : 2D NumPy array
        spectral subregion, containing spectral background, sky,
        and/or target flux measurements.

      xord : scalar or sequence
        Order of polynomial by which each ROW will be normalized. If
        niter>0, xord can be a sequence of length (niter+1).  A good
        approach for, e.g., spectral dome flats is to set niter=1 and
        xord=[15,2].

      nsigma : scalar
        Sigma-clipping level for calculation of column-by-column S/N

      minsnr : scalar
        Minimum S/N value to use when selecting 'good' columns for
        normalization.

      minfrac : scalar, 0 < minfrac < 1
        Fraction of columns to use, selected by highest S/N, when
        selecting 'good' columns for normalization.

      niter : int
        Number of iterations.  If set to zero, do not iterate (i.e.,
        run precisely once through.)

    :NOTES:
      Helper function for functions XXXX
    """
    # 2013-01-20 14:20 IJMC: Created

    ny,  nx = subreg.shape
    xall = np.arange(2048.)
    subreg_new = subreg.copy()

    iter = 0
    if not hasattr(xord, '__iter__'): xord = [xord]*(niter+1)
    while iter <= niter:
        snr = an.snr(subreg_new, axis=0, nsigma=nsigma)
        ind = (snr > np.sort(snr)[-int(minfrac*snr.size)]) * (snr > minsnr)
        xxx = ind.nonzero()[0]
        norm_subreg = subreg[:,ind] / np.median(subreg[:,ind], 0)
        coefs = np.array([an.polyfitr(xxx, row, xord[iter], 3) for row in norm_subreg])
        xflat = np.array([np.polyval(coef0, xall) for coef0 in coefs])
        iter += 1
        subreg_new = subreg / xflat
    return xflat



def make_spectral_flats(sky, domeflat, subreg_corners, badpixelmask=None, xord_pix=[15,2], xord_sky=[2,1], yord=2, minsnr=5, minfrac_pix=0.7, minfrac_sky=0.5, locs=None, nsigma=3):
    """
    Construct appropriate corrective frames for multi-object
    spectrograph data.  Specifically: corrections for irregular slit
    widths, and pixel-by-pixel detector sensitivity variations.

    sky : 2D NumPy array
       Master spectral sky frame, e.g. from median-stacking many sky
       frames or masking-and-stacking dithered science spectra frames.
       This frame is used to construct a map to correct science frames
       (taken with the identical slit mask!) for irregular sky
       backgrounds resulting from non-uniform slit widths (e.g.,
       Keck/MOSFIRE).

       Note that the dispersion direction should be 'horizontal'
       (i.e., parallel to rows) in this frames.

    domeflat : 2D NumPy array
       Master dome spectral flat, e.g. from median-stacking many dome
       spectra.  This need not be normalized in the dispersion or
       spatial directions. This frame is used to construct a flat map
       of the pixel-to-pixel variations in detector sensitivity.

       Note that the dispersion direction should be 'horizontal'
       (i.e., parallel to rows) in this frames.

    subreg_corners : sequence of 2- or 4-sequences
        Indices (or merely starting- and ending-rows) for each
        subregion of interest in 'sky' and 'domeflat' frames.  Syntax
        should be that of :func:`tools.extractSubregion`, or such that
        subreg=sky[subreg_corners[0]:subreg_corners[1]]

    badpixelmask : 2D NumPy array, or str
        Nonzero for any bad pixels; these will be interpolated over
        using :func:`nsdata.bfixpix`.

    xord_pix : sequence
        Polynomial orders for normalization in dispersion direction of
        pixel-based flat (dome flat) on successive iterations; see
        :func:`makexflat`.

    xord_sky : sequence
        Polynomial orders for normalization in dispersion direction of
        slit-based correction (sky frame) on successive iterations;
        see :func:`makexflat`.

    yord : scalar
        Polynomial order for normalization of pixel-based (dome) flat
        in spatial direction.

    locs : None, or sequence
        Row-index in each subregion of the location of the
        spectral-trace-of-interest.  Only used for rectifying of sky
        frame; leaving this at None will have at most a mild
        effect. If not None, should be the same length as
        subreg_corners.  If subreg_corners[0] = [800, 950] then
        locs[0] might be set to, e.g., 75 if the trace lies in the
        middle of the subregion.


    :RETURNS:
        wideslit_skyflat, narrowslit_domeflat

    :EXAMPLE:
     ::


        try:
            from astropy.io import fits as pyfits
        except:
            import pyfits
        
        import spec
        import ir

        obs = ir.initobs('20121010')
        sky = pyfits.getdata(obs._raw + 'masktersky.fits')
        domeflat = pyfits.getdata(obs._raw + 'mudflat.fits')
        allinds = [[7, 294], [310, 518], [532, 694], [710, 960], [976, 1360], [1378, 1673], [1689, 2022]]
        locs = [221, 77, 53, 88, 201, 96, 194]

        skycorrect, pixcorrect = spec.make_spectral_flats(sky, domeflat, allinds, obs.badpixelmask, locs=locs)
    """
    # 2013-01-20 14:40 IJMC: Created
    from tools import extractSubregion
    from nsdata import bfixpix
    
    # Check inputs:
    if not isinstance(sky, np.ndarray):
        sky = pyfits.getdata(sky)
    if not isinstance(domeflat, np.ndarray):
        domeflat = pyfits.getdata(domeflat)

    ny0, nx0 = sky.shape

    if badpixelmask is None:
        badpixelmask = np.zeros((ny0, nx0), dtype=bool)
    if not isinstance(badpixelmask, np.ndarray):
        badpixelmask = pyfits.getdata(badpixelmask)

    # Correct any bad pixels:
    if badpixelmask.any():
        bfixpix(sky, badpixelmask)
        bfixpix(domeflat, badpixelmask)

    wideslit_skyflat = np.ones((ny0, nx0), dtype=float)
    narrowslit_domeflat = np.ones((ny0, nx0), dtype=float)

    # Loop through all subregions specified:
    for jj in range(len(subreg_corners)):
        # Define extraction & alignment indices:
        specinds = np.array(subreg_corners[jj]).ravel().copy()
        if len(specinds)==2:
            specinds = np.concatenate(([0, nx0], specinds))
        if locs is None:
            loc = np.mean(specinds[2:4])
        else:
            loc = locs[jj]
  
        skysub = extractSubregion(sky, specinds, retall=False)
        domeflatsub = extractSubregion(domeflat, specinds, retall=False)
        badsub = extractSubregion(badpixelmask, specinds, retall=False)
        ny, nx = skysub.shape
        yall = np.arange(ny)
  
        # Normalize Dome Spectral Flat in X-direction (rows):
        xflat_dome = makexflat(domeflatsub, xord_pix, minsnr=minsnr, minfrac=minfrac_pix, niter=len(xord_pix)-1, nsigma=nsigma)
        ymap = domeflatsub*0.0
        xsubflat = domeflatsub / np.median(domeflatsub, 0) / xflat_dome
  
        # Normalize Dome Spectral Flat in Y-direction (columns):
        ycoefs = [an.polyfitr(yall, xsubflat[:,ii], yord, nsigma) for ii in range(nx)]
        yflat = np.array([np.polyval(ycoef0, yall) for ycoef0 in ycoefs]).transpose()
        pixflat = domeflatsub / (xflat_dome * yflat * np.median(domeflatsub, 0))
        bogus = (pixflat< 0.02) + (pixflat > 50)
        pixflat[bogus] = 1.
  
        # Normalize Sky spectral Flat in X-direction (rows):
        askysub = scaleSpectralSky_cor(skysub/pixflat, badsub, pord=2, refpix=loc, nmed=1)
        xflat_sky = makexflat(askysub[1], xord_sky, minsnr=minsnr, minfrac=minfrac_sky, niter=len(xord_sky)-1, nsigma=nsigma)
  
        wideslit_skyflat[specinds[2]:specinds[3], specinds[0]:specinds[1]] = xflat_sky
        narrowslit_domeflat[specinds[2]:specinds[3], specinds[0]:specinds[1]] = pixflat
  
        print "Done with subregion %i" % (jj+1)
    
    return wideslit_skyflat, narrowslit_domeflat


def calibrate_stared_mosfire_spectra(scifn, outfn, skycorrect, pixcorrect, subreg_corners, **kw):
    """Correct non-dithered WIDE-slit MOSFIRE spectral frames for:
          pixel-to-pixel nonuniformities (i.e., traditional flat-fielding)
          
          detector nonlinearities

          tilted spectral lines

          non-uniform slit widths (which cause non-smooth backgrounds) 

       Note that the dispersion direction should be 'horizontal'
       (i.e., parallel to rows) in this frame.

    :INPUTS:
       scifn : str
         Filename of raw, uncalibrated science frame (in ADUs, not electrons)

       outfn : str
         Name into which final, calibrated file  should be written.

       skycorrect : str or 2D NumPy array
         Slitloss correction map (i.e., for slits of nonuniform width),
         such as generated by :func:`make_spectral_flats`.

       pixcorrect : str or 2D NumPy array
         Pixel-by-pixel sensitivity correction map (i.e., flat field),
         such as generated by :func:`make_spectral_flats`.

       subreg_corners : sequence of 2- or 4-sequences
         Indices (or merely starting- and ending-rows) for each
         subregion of interest in 'sky' and 'domeflat' frames.  Syntax
         should be that of :func:`tools.extractSubregion`, or such that
         subreg=sky[subreg_corners[0]:subreg_corners[1]]

       linearize : bool
         Whether to linearity-correct the data.

         If linearizing: linearity correction is computed & applied
         AFTER applying the pixel-by-pixel (flatfield) correction, but
         BEFORE the slitloss (sky) correction.



       bkg_radii : 2-sequence
           Inner and outer radius for background computation and removal;
           measured in pixels from the center of the profile.  The values 
           [11,52] seems to work well for MOSFIRE K-band spectra.

    locs : None, or sequence
        Row-index in each subregion of the location of the
        spectral-trace-of-interest.  Only used for rectifying of sky
        frame; leaving this at None will have at most a mild
        effect. If not None, should be the same length as
        subreg_corners.  If subreg_corners[0] = [800, 950] then
        locs[0] might be set to, e.g., 75 if the trace lies in the
        middle of the subregion.

       gain: scalar
           Detector gain, in electrons per ADU

       readnoise: scalar
           Detector readnoise, in electrons

      polycoef : str, None, or sequence
        Polynomial coefficients for detector linearization: see
        :func:`ir.linearity_correct` and :func:`ir.linearity_mosfire`.

      unnormalized_flat : str or 2D NumPy array
        If not 'None', this is used to define the subregion header
        keywords ('subreg0', 'subreg1', etc.) for each slit's
        two-dimensional spectrum.  These keywords are required by much
        of my extraction machinery!


    :EXAMPLE:
      ::


        try:
            from astropy.io import fits as pyfits
        except:
            import pyfits
        
        import spec
        import ir

        obs = ir.initobs('20121010')
        allinds = [[7, 294], [310, 518], [532, 694], [710, 960], [976, 1360], [1378, 1673], [1689, 2022]]
        locs = [221, 77, 53, 88, 201, 96, 194]
        unnorm_flat_fn = obs._raw + 'mudflat.fits'
        if False:
           sky = pyfits.getdata(obs._raw + 'masktersky.fits')
           unnorm_domeflat = pyfits.getdata(unnorm_flat_fn)
           skycorrect, pixcorrect = spec.make_spectral_flats(sky, unnorm_domeflat, allinds, badpixelmask=obs.badpixelmask, locs=locs)
        else:
           skycorrect = obs._raw + 'skycorrect.fits'
           pixcorrect = obs._raw + 'pixcorrect.fits'

        linearcoef='/Users/ianc/proj/transit/data/mosfire_linearity/linearity/mosfire_linearity_cnl_coefficients_bic-optimized.fits'
        rawfn = obs.rawsci
        outfn = obs.procsci
        spec.calibrate_stared_mosfire_spectra(rawfn, outfn, skycorrect, pixcorrect, allinds, linearize=True, badpixelmask=obs.badpixelmask, locs=locs, polycoef=linearcoef, verbose=True, clobber=True, unnormalized_flat=unnorm_flat_fn)

    :NOTES:
      This routine is *slow*, mostly because of the call to
      :func:`defringe_sinusoid`.  Consider running multiple processes
      in parallel, to speed things up!

    :SEE_ALSO:
      :func:`ir.mosfire_speccal`, :func:`defringe_sinusoid`
    """
    # 2013-01-20 16:17 IJMC: Created.
    # 2013-01-25 15:59 IJMC: Fixed various bugs relating to
    #                        determinations of subregion boundaries.
    

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from nsdata import bfixpix
    import ir
    import os
    from tools import extractSubregion, findRectangles
    import sys

    #pdb.set_trace()

    if not isinstance(scifn, str) and hasattr(scifn, '__iter__'):
        for scifn0, outfn0 in zip(scifn, outfn):
            calibrate_stared_mosfire_spectra(scifn0, outfn0, skycorrect, pixcorrect, subreg_corners, **kw)

    else:
        # Set defaults:
        names = ['linearize', 'bkg_radii', 'gain', 'readnoise', 'polycoef', 'verbose', 'badpixelmask', 'locs', 'clobber', 'unnormalized_flat']
        defaults = [False, [11, 52], 2.15, 21.1, None, False, None, None, False, None]
        for n,d in zip(names, defaults):
            exec('%s = kw["%s"] if kw.has_key("%s") else d' % (n, n, n))


        # Check inputs:
        scihdr = pyfits.getheader(scifn)
        sci = pyfits.getdata(scifn)
        exptime = scihdr['TRUITIME']
        ncoadd  = scihdr['COADDONE'] 
        nread   = scihdr['READDONE']

        if not isinstance(pixcorrect, np.ndarray):
            scihdr.update('SLITFLAT', pixcorrect)
            pixcorrect = pyfits.getdata(pixcorrect)
        else:
            scihdr.update('PIXFLAT', 'User-specified data array')

        if not isinstance(skycorrect, np.ndarray):
            scihdr.update('SLITFLAT', skycorrect)
            skycorrect = pyfits.getdata(skycorrect)
        else:
            scihdr.update('SLITFLAT', 'User-specified data array')

        if unnormalized_flat is not None:
            if not isinstance(unnormalized_flat, np.ndarray):
                scihdr.update('MUDFLAT', unnormalized_flat)
                unnormalized_flat = pyfits.getdata(unnormalized_flat)
            else:
                scihdr.update('MUDFLAT', 'User-specified data array')

        ny0, nx0 = sci.shape

        if badpixelmask is None:
            badpixelmask = np.zeros((ny0, nx0), dtype=bool)
        if not isinstance(badpixelmask, np.ndarray):
            badpixelmask = pyfits.getdata(badpixelmask)

        # Correct any bad pixels:
        if badpixelmask.any():
            bfixpix(sci, badpixelmask)

        if linearize:
            # If you edit this section, be sure to update the same
            # section in ir.mosfire_speccal!
            fmap = ir.makefmap_mosfire() 
            sci_lin, liniter = ir.linearity_correct(sci/pixcorrect, nread-1, ncoadd, 1.45, exptime, fmap, ir.linearity_mosfire, verbose=verbose, linfuncargs=dict(polycoef=polycoef), retall=True)
            ir.hdradd(scihdr, 'COMMENT', 'Linearity-corrected by calibrate_stared_mosfire_spectra.')
            ir.hdradd(scihdr, 'LIN_ITER', liniter)
            if isinstance(polycoef, str):
                ir.hdradd(scihdr, 'LINPOLY', os.path.split(polycoef)[1])
            else:
                if isinstance(polycoef, np.ndarray) and polycoef.ndim==3:
                    ir.hdradd(scihdr, 'LINPOLY', 'user-input array used for linearity correction.')
                else:
                    ir.hdradd(scihdr, 'LINPOLY', str(polycoef))

        else:
            sci_lin = sci/pixcorrect
            ir.hdradd(scihdr, 'COMMENT', 'NOT linearity-corrected.')


        newframe = np.zeros((ny0, nx0), dtype=float)


        # Loop through all subregions specified:
        for jj in range(len(subreg_corners)):
            # Define extraction & alignment indices:
            specinds = np.array(subreg_corners[jj]).ravel().copy()
            if len(specinds)==2:
                specinds = np.concatenate(([0, nx0], specinds))
            if locs is None:
                loc = np.mean(specinds[2:4])
            else:
                loc = locs[jj]

            # Prepare various necessities:
            scisub = extractSubregion(sci, specinds, retall=False)
            badsub = extractSubregion(badpixelmask, specinds, retall=False)
            slitflat = extractSubregion(skycorrect, specinds, retall=False)
            ny, nx = scisub.shape
            yall = np.arange(ny)


            # Define subregion boundaries from the flat-field frame.
            # Ideally, this should go *outside* the loop (no need to
            # re-calculate this for every file...)
            if unnormalized_flat is not None:
                samplemask, boundaries = ir.spectral_subregion_mask(unnormalized_flat)
                samplemask *= (unnormalized_flat > 500)
                samplemask_corners = findRectangles(samplemask, minsepy=42, minsepx=150)
                nspec = samplemask_corners.shape[0]
                for kk in range(nspec):
                    ir.hdradd(scihdr, 'SUBREG%i' % kk, str(samplemask_corners[kk]))
                ir.hdradd(scihdr, 'NSUBREG', nspec)



            # Align the subregion, so that sky lines run along detector columns:
            aligned_scisub = scaleSpectralSky_cor(scisub/slitflat, badsub, pord=2, refpix=loc, nmed=1)

            # Model the sky background, using linear trends plus a sinusoid:
            if locs is None:
                spatind = np.ones(ny, dtype=bool)
            else:
                spatind = (np.abs(np.arange(ny) - loc) > bkg_radii[0]) * (np.abs(np.arange(ny) - loc) < bkg_radii[1])

            sscisub = defringe_sinusoid(aligned_scisub[1], badpixelmask=badsub, period_limits=[9,30], gain=gain, readnoise=readnoise, bictest=False, spatial_index=spatind, nmed=1)

            # Compute the corrected subregion:
            aligned_flattened_skycorrected_subregion = \
                (aligned_scisub[1] - sscisub) * slitflat + aligned_scisub[3]

            #try2 = (aligned_scisub[1] - sscisub) * slitflat + aligned_scisub

            newframe[specinds[2]:specinds[3],specinds[0]:specinds[1]] = aligned_flattened_skycorrected_subregion

            print "Done with subregion %i" % (jj+1)

        scihdr.update('MOSCAL', 'Calibrated by calibrate_stared_mosfire_spectra')
        
        pyfits.writeto(outfn, newframe.astype(float), scihdr, clobber=clobber)

    return 



def reconstitute_gmos_roi(infile, outfile, **kw):
    """Convert GMOS frames taken with custom ROIs into standard FITS frames.

    :INPUTS:
      in : string or sequence of strings
        Input filename or filenames.

      out : string or sequence of strings
        Output filename or filenames.

    :OPTIONS:
      clobber : bool
        Passed to PyFITS; whether to overwrite existing files.
        """
    # 2013-03-28 21:37 IJMC: Created at the summit of Mauna Kea

    import ir

    def trimlims(instr):
        trimchars = '[]:,'
        for trimchar in trimchars:
            instr = instr.replace(trimchar, ' ')
        return map(float, instr.strip().split())
        
    
    # Handle recursive case:
    if not isinstance(infile, str):
        for thisin, thisout in zip(infile, outfile):
            reconstitute_gmos_roi(thisin, thisout, **kw)
        return


    # Handle non-recursive (single-file) case:
    file = pyfits.open(infile)
    hdr = file[0].header
    nroi = hdr['detnroi']
    biny, binx = map(int, file[1].header['ccdsum'].split())

    detsize = trimlims(hdr['detsize'])
    binning = 2
    frame = np.zeros(((detsize[3]-detsize[2]+1)/binx, (detsize[1]-detsize[0]+1)/biny), dtype=int)
    print frame.shape
    
    # Read in the coordinates for each ROI:
    roi_xys = np.zeros((nroi, 4), dtype=int)
    for ii in range(1, nroi+1):
        roi_xys[ii-1, 0] = (hdr['detro%ix' % ii] - 1) 
        roi_xys[ii-1, 1] = (hdr['detro%ix' % ii] + hdr['detro%ixs' % ii] - 1) 
        roi_xys[ii-1, 2] = (hdr['detro%iy' % ii] - 1) /binx
        roi_xys[ii-1, 3] = (hdr['detro%iy' % ii] - 1)/binx + hdr['detro%iys' % ii] - 1
        
    # Read in the C
    nhdus = len(file) - 1
    detsecs = np.zeros((nhdus, 4), dtype=int)
    datasecs = np.zeros((nhdus, 4), dtype=int)
    biassecs = np.zeros((nhdus, 4), dtype=int)
    for ii in range(nhdus):
        detsecs[ii] = trimlims(file[ii+1].header['detsec'])
        datasecs[ii] = trimlims(file[ii+1].header['datasec'])
        biassecs[ii] = trimlims(file[ii+1].header['biassec'])
        di2 = (detsecs[ii,0]-1)/biny, detsecs[ii,1]/biny, (detsecs[ii,2]-1)/binx, detsecs[ii,3]/binx
        thisdat = file[ii+1].data
        #pdb.set_trace()
        if biassecs[ii,0]==1025:
            frame[di2[2]:di2[3], di2[0]:di2[1]] = thisdat[:, :-32]
        elif biassecs[ii,0]==1:
            frame[di2[2]:di2[3], di2[0]:di2[1]] = thisdat[:, 32:]
        else:
            print 'bombed. bummer!  I should have written a better function'        

    file.close()

    hdr = pyfits.getheader(infile)
    for ii in range(nroi):
        ir.hdradd(hdr, 'SUBREG%i' % ii, '[%i %i %i %i]' % tuple(roi_xys[ii]))

    pyfits.writeto(outfile, frame, hdr, **kw)

    return #detsecs, datasecs, biassecs

def rotationalProfile(delta_epsilon, delta_lam):
    """Compute the rotational profile of a star, assuming solid-body
    rotation and linear limb darkening.

    This uses Eq. 18.14 of Gray's Photospheres, 2005, 3rd Edition.

    :INPUTS:

      delta_epsilon : 2-sequence

        [0] : delta_Lambda_L = lambda * V * sin(i)/c; the rotational
              displacement at the stellar limb.

        [1] : epsilon, the linear limb darkening coefficient, used in
              the relation I(theta) = I0 + epsilon * (cos(theta) - 1).

        [2] : OPTIONAL! The central location of the profile (otherwise
              assumed to be located at delta_lam=0).

      delta_lam : scalar or sequence
        Wavelength minus offset: Lambda minus lambda_0.  Grid upon
        which computations will be done.

    :EXAMPLE:
      ::

        import pylab as py
        import spec

        dlam = py.np.linspace(-2, 2, 200) # Create wavelength grid
        profile = spec.rotationalProfile([1, 0.6], dlam)

        py.figure()
        py.plot(dlam, profile)
    """
    # 2013-05-26 10:37 IJMC: Created.

    delta_lambda_L, epsilon = delta_epsilon[0:2]
    if len(delta_epsilon)>2:  # optional lambda_offset
        lamdel2 = 1. - ((delta_lam - delta_epsilon[2])/delta_lambda_L)**2
    else:
        lamdel2 = 1. - (delta_lam/delta_lambda_L)**2
    
    if not hasattr(delta_lam, '__iter__'):
        delta_lam = np.array([delta_lam])

    ret = (4*(1.-epsilon) * np.sqrt(lamdel2) + np.pi*epsilon*lamdel2) / \
        (2*np.pi * delta_lambda_L * (1. - epsilon/3.))    

    ret[lamdel2<0] = 0.

    return ret


def modelline(param, prof2, dv):
  """Generate a rotational profile, convolve it with a second input
  profile, normalize it (simply), and return.
  
  :INPUTS:
    param : 1D sequence
      param[0:3]  -- see :func:`rotationalProfile`
      param[4]  -- multiplicative scaling factor

    dv : velocity grid

    prof2 : second input profile
  """
  # 2013-08-07 09:55 IJMC: Created
  nk = dv.size
  rot_model = rotationalProfile(param[0:3], dv)  
  conv_rot_model = np.convolve(rot_model, prof2, 'same')
  norm_conv_rot_model = conv_rot_model - (conv_rot_model[0] + (conv_rot_model[-1] - conv_rot_model[0])/(nk - 1.) * np.arange(nk))  
  return norm_conv_rot_model * param[3]

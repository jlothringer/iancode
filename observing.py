"""
A main driver function is :func:`makeAnnualChart`

"""
import ephem
import numpy as np
from scipy import ndimage
import pylab as py
import pdb
from tools import isstring

def makeAnnualChart(obs, ra, dec, minElevation=30, twilight=12, oversamp=16, dt=0):
    """
    Make pretty plots of target visibility during the year. E.g., to
    observe the Kepler field from Keck:

    .. plot::

       import observing as obs
       obs.makeAnnualChart('keck', '19:22:40', '+44:30:00', dt=-10)


    :INPUTS:
      obs : str
        'lick' or 'keck' or 'lapalma' or 'mtgraham' or 'mtbigelow' or
         'andersonmesa' or 'kpno' or 'ctio' or 'cerropachon' or
         'palomar' or 'cerroparanal' or 'lasilla' or 'calaralto' or
         'lascampanas' (cf. :func:`setupObservatory`)

      minElevation : float
         Minimum visible elevation angle. '30' implies airmass=2. 

      twilight : float
         Minimum acceptable angular distance of sun below horizon, in
         degrees. 

      dt : scalar
         Timezone offset from UTC. Positive for east, negative for West.
         

    :EXAMPLE:
     ::

      import observing as obs

      # Plot Visibility of the Kepler Field:
      obs.makeAnnualChart('keck', '19:22:40', '+44:30:00', dt=-10)
      obs.makeAnnualChart('mtgraham', '19:22:40', '+44:30:00', dt=-7)


    :NOTES:
      Based on the attractive plots used by the California Planet Search team.
      """
    # 2015-03-19 21:02 IJMC: Created


    observatory = setupObservatory(obs)
    target = setupTarget(ra, dec)

    # Make a grid of annual dates vs. local time
    today = observatory.date.datetime()
    if today.month<=5:
        year0 = today.year
    else:
        year0 = today.year + 1
    dates = year0 + np.round(np.linspace(0, 1, 366.)*365.)/365.
    ndates = dates.size
    hrs = np.linspace(0, 24, 49)
    datetimes = dates + hrs.reshape(hrs.size, 1)/365./24.


    visibility = isObservable(datetimes, observatory, target, minElevation=minElevation, twilight=twilight, oversamp=oversamp)
    title = '%s: RA = %s, DEC = %s\nalt > %1.1f, sun < -%1.1f' % (obs, str(ra), str(dec), minElevation, twilight)
    fig = drawAnnualChart(ndimage.zoom(dates, oversamp), ndimage.zoom(hrs, oversamp), visibility, title=title, dt=dt)

    return fig

def drawAnnualChart(dates, hrs, visibility, title='', fs=16, dt=0):
    """
    Draw charts computed by :func:`makeAnnualChart`.

    :INPUTS:
      dates, hrs, visibility 
        Inputs suitable for pylab.contourf(dates, hrs, visibility)

      fs : scalar
        Font size
    """
    # 2015-03-19 22:38 IJMC: Created
    import matplotlib.dates as mdates

    if True: #dt==0:
        ylabel='UTC Time'
    else:
        ylabel = '(UTC %+1.1f)' % dt
    
    obs = ephem.Observer()
    ddates = []
    for d in dates:
        obs.date = str(d)
        ddates.append(obs.date.datetime())

    fig = py.figure()
    #ax = py.gca()
    axpos = [.1, .12, .77, .76]
    ax = py.subplot(111, position=axpos)
    py.contourf(ddates, hrs, visibility, cmap=py.cm.Greens)
    py.clim(0, 1.5)
    months   = mdates.MonthLocator()  # every month
    ax.xaxis.set_major_locator(months)
    #ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    hfmt = mdates.DateFormatter('%b ')
    ax.xaxis.set_major_formatter(hfmt)
    #yt = (np.array([0, 4, 8, 12, 16, 20, 24]) - 12)  + dt
    yt = np.array([0, 4, 8, 12, 16, 20, 24])

    ax.set_yticks(yt)
    ax.yaxis.set_ticks(range(25), minor=True)
    ax.set_yticklabels(yt % 24 )
    #[tt.set_rotation(30) for tt in ax.get_xticklabels()]
    #ax.set_xlabel('Date', fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_title(title, fontsize=fs*1.2)
    ax.grid(axis='x')

    ax2 = py.twinx()
    ax2.set_position(axpos)
    yt2 = np.arange(-24, 24, 3)
    yt2 = yt2[(yt2>=dt) * (yt2<=(24+dt))]
    ax2.set_ylim(dt, 24+dt)
    ax2.set_yticks(yt2)
    ax2.set_ylabel('Local Time: UTC %+1.1f' % dt, fontsize=fs)
    ax2.grid(axis='y')

    tlabs = yt2 % 12
    tlabs = []
    for yt in yt2:
        if (yt%24)==12:
            lab = 'noon'
        elif (yt%24)==0:
            lab = 'mdnt'
        elif (yt % 24) >= 12:
            lab = '%i pm' % (yt % 12)
        else:
            lab = '%i am' % (yt % 12)
        tlabs.append(lab)
    ax2.set_yticklabels(tlabs)
    ax2.yaxis.set_ticks(range(dt, dt+25), minor=True)
    

    #fig.autofmt_xdate()
    #ax.set_position([.15, .2, .8, .68])
    return fig

def isObservable(dates, obs, target, minElevation=30, twilight=12, oversamp=16):
    """
    True if pyEphem object 'target' is visible to observer 'obs' on
    the input 'dates'; False otherwise.
    """
    # 2015-03-19 22:11 IJMC: Created
    sun = ephem.Sun()


    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
        if dates.size==0:
            dates = np.array([dates])

    dateshape = dates.shape
    dates = dates.ravel()
    ndates = dates.size

    alts = np.zeros(ndates, dtype=float)
    sunalts = np.zeros(ndates, dtype=float)

    for ii in xrange(ndates):
        obs.date = str(dates[ii])
        sun.compute(obs)
        target.compute(obs)
        sunalts[ii] = sun.alt
        alts[ii] = target.alt


    if oversamp<>1:
        alts = ndimage.zoom(alts.reshape(dateshape), oversamp)
        sunalts = ndimage.zoom(sunalts.reshape(dateshape), oversamp)
        dateshape = alts.shape

    vis = ((-sunalts >= (twilight*np.pi/180.)) * (alts >= (minElevation*np.pi/180.))).reshape(dateshape)
    #dates = dates.reshape(dateshape)
    #hivis = ((-hisunalts >= (twilight*np.pi/180.)) * (hialts >= (minElevation*np.pi/180.)))

    #pdb.set_trace()

    return vis
    
#target.compute(obs)
    
def isNumeric(input, cast=None):
    """Simple test for whether input is numeric or not.
    
    cast : None or float
      If float, is also numeric if float(input) is valid.
    """
    # 2015-03-19 21:36 IJMC: Created
    # 2015-04-10 20:15 IJMC: Added 'cast' option.
    try:
        junk = input + 0
        ret = True
    except:
        if cast is not None:
            try:
                junk = cast(input)
                ret = True
            except:
                ret = False
        else:
            ret = False
    return ret

def setupTarget(ra_deg, dec_deg, pmra=0, pmdec=0, name=None, verbose=False):
    """
    ra_deg, dec_deg : strings or scalars
      Right ascenscion and Declination, in *decimal degrees* (scalar)
      or sexagesimal (if strings)

    """
    # 2015-03-19 21:37 IJMC: Created
    target = ephem.star('Rigel')
    if name is not None:
        target.name = name
    if isNumeric(ra_deg):
        ra_deg = hms(ra_deg, output_string=True)
    if isNumeric(dec_deg):
        dec_deg = dms(dec_deg, output_string=True)

    #print ra_deg, dec_deg
    target._ra, target._dec = ra_deg, dec_deg
    #print target._ra, target._dec

    if pmra<>0:
        target._pmra = pmra
    if pmdec<>0:
        target._pmdec = pmdec

    return target

    
def setupObservatory(obs, lat=None, long=None, elevation=None):
    """Set up PyEphem 'observer' object for a given observatory.
    
    :INPUTS:
      obs : str
         Name of an observatory.  'lick' or 'keck' or 'lapalma' or
         'mtgraham' or 'mtbigelow' or 'andersonmesa' or 'kpno' or
         'ctio' or 'cerropachon' or 'palomar' or 'cerroparanal' or
         'lasilla' or 'calaralto' or 'lascampanas' or 'saao' or
         'sidingspring'
    """
    # 2015-03-19 20:59 IJMC: Created
    observer = ephem.Observer()
    if isinstance(obs, ephem.Observer):
        observer = obs
    elif obs=='lick':
        observer.long, observer.lat = '-121:38.2','37:20.6'
        observer.elevation = 1290
    elif obs=='flwo':
        observer.long, observer.lat = '-110:52.7', '31:40.8'
	observer.elevation = 2606
    elif obs=='keck':
        observer.long, observer.lat = '-155:28.7','19:49.7'
	observer.elevation = 4160
    elif obs=='lapalma' or obs=='la palma':
        observer.long, observer.lat = '-17:53.6','28:45.5'
	observer.elevation = 2396
    elif obs=='ctio':
        observer.long, observer.lat = '-70:48:54','-30:9.92'
	observer.elevation = 2215
    elif obs=='dct' or obs=='happy jack' or obs=='happyjack':  # 
        observer.long, observer.lat = '-111:25:20', '34:44:40'
	observer.elevation = 2360
    elif obs=='andersonmesa' or obs=='anderson mesa':  # 
        observer.long, observer.lat = '-111:32:09', '30:05:49'
	observer.elevation = 2163
    elif obs=='mtbigelow' or obs=='mount bigelow' or \
            obs=='mountbigelow' or obs=='catalinastation' or obs=='catalina':
        observer.long, observer.lat = '-110:44:04.3', '32:24:59.3'
        observer.elevation = 2518
    elif obs=='mtgraham' or obs=='mount graham' or obs=='mountgraham':
        observer.long, observer.lat = '-109:53:23', '32:42:05'
        observer.elevation = 3221
    elif obs=='kpno':
        observer.long, observer.lat = '-111:25:48', '31:57:30'
        observer.elevation = 2096
    elif obs=='cerropachon' or obs=='cerro pachon':
        observer.long, observer.lat = '-70:44:11.7', '-30:14:26.6'
        observer.elevation = 2722
    elif obs=='palomar':
        observer.long, observer.lat = '-116:51:50', '33:21:21'
        observer.elevation = 1712
    elif obs=='lasilla' or obs=='la silla':
        observer.long, observer.lat = '-70:43:53', '-29:15:40'
        observer.elevation = 2400
    elif obs=='cerroparanal' or obs=='cerro paranal':
        observer.long, observer.lat = '-70:24:15', '-24:37:38'
        observer.elevation = 2635
    elif obs=='calaralto' or obs=='calar alto':
        observer.long, observer.lat = '-02:32:46', '+37:13:25'
        observer.elevation = 2168
    elif obs=='lascampanas' or obs=='las campanas':
        observer.long, observer.lat = '-70:41:33', '-29:00:53'
        observer.elevation = 2380
    elif obs=='saao' or obs=='sutherland':
        observer.long, observer.lat = '-32:22:42', '+20:48:38'
        observer.elevation = 1798
    elif obs=='sidingspring' or obs=='sidingsprings':
        observer.long, observer.lat = '-31:16:24', '+149:04:16'
        observer.elevation = 1116

    if lat is not None:
        observer.lat = lat
    if long is not None:
        observer.long = long
    if elevation is not None:
        observer.elevation = elevation

    return observer


def hms(d, delim=':', output_string=False):
    """Convert hours, minutes, seconds to decimal degrees, and back.

    EXAMPLES:

    hms('15:15:32.8')
    hms([7, 49])
    hms(18.235097)
    hms(18.235097, output_string=True)
    
    Also works for negative values.

    SEE ALSO:  :func:`dms`
    """
    # 2008-12-22 00:40 IJC: Created
    # 2009-02-16 14:07 IJC: Works with spaced or colon-ed delimiters
    # 2015-03-19 21:29 IJMC: Copied from phot.py. Added output_string.
    # 2015-08-28 03:48 IJMC: Added 'None' check.
    from numpy import sign

    if d is None:
        return np.nan
    elif isstring(d) or hasattr(d, '__iter__'):   # must be HMS
        if isstring(d):
            d = d.split(delim)
            if len(d)==1:
                d = d[0].split(' ')
            if (len(d)==1) and (d.find('h')>-1):
                d.replace('h',delim)
                d.replace('m',delim)
                d.replace('s','')
                d = d.split(delim)
        s = sign(float(d[0]))
        if s==0:  s=1
        degval = float(d[0])*15.0
        if len(d)>=2:
            degval = degval + s*float(d[1])/4.0
        if len(d)==3:
            degval = degval + s*float(d[2])/240.0
        return degval
    else:    # must be decimal degrees
        hour = int(d/15.0)
        d = abs(d)
        min = int((d-hour*15.0)*4.0)
        sec = (d-hour*15.0-min/4.0)*240.0
        ret = (hour, min, sec)
        if output_string:

            ret = ('%02i'+delim+'%02i'+delim+'%05.2f') % ret

        return ret


def dms(d, delim=':', output_string=False):
    """Convert degrees, minutes, seconds to decimal degrees, and back.

    EXAMPLES:

    dms('150:15:32.8')
    dms([7, 49])
    dms(18.235097)
    dms(18.235097, output_string=True)
    
    Also works for negative values.

    SEE ALSO:  :func:`hms`
    """
    # 2008-12-22 00:40 IJC: Created
    # 2009-02-16 14:07 IJC: Works with spaced or colon-ed delimiters
    # 2015-03-19 21:29 IJMC: Copied from phot.py. Added output_string.
    # 2015-08-28 03:48 IJMC: Added 'None' check.
    from numpy import sign

    if d is None:
        return np.nan
    elif isstring(d) or hasattr(d, '__iter__'):   # must be HMS
        if isstring(d): #d.__class__==str or d.__class__==np.string_:
            d = d.split(delim)
            if len(d)==1:
                d = d[0].split(' ')
        s = sign(float(d[0]))
        if s==0:  s=1
        degval = float(d[0])
        if len(d)>=2:
            degval = degval + s*float(d[1])/60.0
        if len(d)==3:
            degval = degval + s*float(d[2])/3600.0
        return degval
    else:    # must be decimal degrees
        if d<0:  
            sgn = -1
        else:
            sgn = +1
        d = abs(d)
        deg = int(d)
        min = int((d-deg)*60.0)
        sec = (d-deg-min/60.0)*3600.0
        ret = (sgn*deg, min, sec)
        if output_string:
            ret = ('%02i'+delim+'%02i'+delim+'%05.2f') % ret
        return ret

def makeStarlistNIRC2(name, ra, dec, epoch=2000, ao=True, rmag=None, bmag=None, vmag=None, verbose=False, lgs_r_cutoff=12):
    """Generate a starlist for Keck/NIRC2. Won't find tip/tilt guidestars, though!

       :INPUT:
        name : str

        ra, dec : strings or scalars
         Right ascenscion and Declination, in *decimal degrees*
         (if scalar) or sexagesimal (if strings)

        lgs_r_cutoff : scalar
         Magnitude fainter than this use LGS mode


       :Example Starlist:
        300 Sag A*      17 42 29.330 -28 59 18.50 1950.0 lgs=1
           0609-0602733 17 45 40.713 -29 00 11.18 2000.0 rmag=14.0 sep=19.3 b-v=0.83 b-r=1.65 S=0.31
           0609-0602749 17 45 42.287 -29 00 36.80 2000.0 rmag=13.5 sep=31.2 b-v=0.68 b-r=1.40 S=0.30
        SKY Sag A*      19 00 00.0   -30 00 00.00 1950.0 lgs=0 comment=no laser
        M5_core         15 18 33.240 +02 05  1.40 2000.0 rmag=11.5 lgs=1 comment=NGS/LGS?
        IRAS 16342      16 37 39.890 -38 20 17.40 2000.0 lgs=1

       :SEE_ALSO:
        http://www2.keck.hawaii.edu/optics/lgsao/lgsstarlists.html
    """
    # 2015-03-23 09:53 IJMC: Created

    def magComment(name, mag, fmt='%3.1f'):
        comment = ''
        if mag is not None:
            comment += ('%s=%s ' % (name, fmt)) % mag
        return comment

    def colorComment(name, mag1, mag2, fmt='%3.1f'):
        comment = ''
        if mag1 is not None and mag2 is not None:
            comment += ('%s=%s ' % (name, fmt)) % (mag1-mag2)
        return comment

    name = np.array(name).ravel()
    ra = np.array(ra).ravel()
    dec = np.array(dec).ravel()
    epoch = np.array(epoch).ravel()
    rmag = np.array(rmag).ravel()
    bmag = np.array(bmag).ravel()
    vmag = np.array(vmag).ravel()

    nstar = max(name.size, ra.size, dec.size, epoch.size, rmag.size, bmag.size, vmag.size)
   
    warnstr = "WARNING: Fewer '%s' input (%i) than number of targets (%i). Check your inputs!"
    if name.size < nstar:
        if verbose:  print warnstr % ('name', name.size, nstar)
        name = np.tile(name, np.ceil(1.0*nstar/name.size))[0:nstar]
    if ra.size < nstar:
        if verbose:  print warnstr % ('ra', ra.size, nstar)
        ra = np.tile(ra, np.ceil(1.0*nstar/ra.size))[0:nstar]
    if dec.size < nstar:
        if verbose:  print warnstr % ('dec', dec.size, nstar)
        dec = np.tile(dec, np.ceil(1.0*nstar/dec.size))[0:nstar]
    if epoch.size < nstar:
        if verbose:  print warnstr % ('epoch', epoch.size, nstar)
        epoch = np.tile(epoch, np.ceil(1.0*nstar/epoch.size))[0:nstar]
    if rmag.size < nstar:
        if verbose:  print warnstr % ('rmag', rmag.size, nstar)
        rmag = np.tile(rmag, np.ceil(1.0*nstar/rmag.size))[0:nstar]
    if bmag.size < nstar:
        if verbose:  print warnstr % ('bmag', bmag.size, nstar)
        bmag = np.tile(bmag, np.ceil(1.0*nstar/bmag.size))[0:nstar]
    if vmag.size < nstar:
        if verbose:  print warnstr % ('vmag', vmag.size, nstar)
        vmag = np.tile(vmag, np.ceil(1.0*nstar/vmag.size))[0:nstar]


    starlist = []
    format1 = '%15s %12s %12s %6.1f %s'
    for istar in xrange(nstar):
        iname = ('%15s' % str(name[istar]))[0:16]
        ira = ra[istar]
        if isNumeric(ira):
            ira = hms(ira, output_string=True, delim=' ')
        idec = dec[istar]
        if isNumeric(idec):
            idec = dms(idec, output_string=True, delim=' ')
        iepoch = float(epoch[istar])

        comment = ''
        if ao:
            if rmag[istar] is None:
                if verbose: print "Warning: AO mode specified, but no rmag entered. Defaulting to lgs=0."
            comment += 'lgs=%i ' % (rmag[istar] is not None and rmag[istar] >= lgs_r_cutoff)
            comment += magComment('rmag', rmag[istar])
            comment += colorComment('b-v', bmag[istar], vmag[istar])
            comment += colorComment('b-r', bmag[istar], rmag[istar])


        iline = format1 % (iname, ira, idec, iepoch, comment)
        if len(iline)>128: iline = iline[0:128]
        starlist.append(iline + '\n')

    return starlist
    
    
        

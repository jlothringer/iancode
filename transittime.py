"""
================================================================================

Routines to make a chart of upcoming transit events.  The main
function to call is :func:`transitChart`, which has an example in its
documentation.  A fully automated chart-maker can be found in :func:`makeCharts`

:REQUIREMENTS:
   PyEphem -- http://rhodesmill.org/pyephem/

   Numpy -- http://numpy.org/

   :doc:`analysis` -- For importing planet-class objects.

   :doc:`observing`

:SEE ALSO: 
   http://var2.astro.cz/EN/tresca/index.php

:TO-DO LIST:
   Print uncertainties in ephemeris when they're sufficiently large


2010-10-11 23:06 IJC: Updated with La Palma location.

2010-03-23 09:37 IJC: Written by IJC to duplicate Josh Winn's page @ MIT.

2011-08-09 13:21 IJMC: Copied over :func:`jd2gd` and :func:`gd2jd`.

2011-09-30 11:01 IJMC: Added moon separation.

================================================================================
"""


try:
    import analysis as an
except:
    pass


import ephem
import observing
#from nsdata import jd2gd, gd2jd
from numpy import cos, pi, array, zeros, ones, sqrt, isfinite, argsort
import pdb

AU = 1.49598e13
msun = 1.98892e33
mjup = 1.8987e30
rjup = 7.1492e9
rsun = 6.955e10
hr = 1./24.
planetkeys = ['tt', 'per', 't14', 'ra_string','dec_string']


# Create a 'star' object, then give it the target's RA and DEC
#target = ephem.star('Rigel')
#target._ra, target._dec = p.ra_string,p.dec_string

def gd2jd(datestr):
    """ Convert a string Gregorian date into a Julian date using Pylab.
        If no time is given (i.e., only a date), then noon is assumed.
        Timezones can be given, but UTC is assumed otherwise.

       :EXAMPLES:
          ::

            print gd2jd('Aug 11 2007')   #---------------> 2454324.5
            print gd2jd('Aug 11 2007, 12:00 PST')  #-----> 2454324.29167
            print gd2jd('12:00 PM, January 1, 2000')  #--> 2451545.0

       :REQUIREMENTS: :doc:`matplotlib`

       :SEE ALSO: :func:`jd2gd`
       """
# 2008-08-26 14:03 IJC: Created        
# 2010-12-08 13:00 IJC: Removed "+ 3442850" from num2julian call
# 2011-05-19 11:37 IJMC: Put the factor back in for error-catching...
    
    import matplotlib.dates as dates
    
    if datestr.__class__==str:
        d = dates.datestr2num(datestr)
        jd = dates.num2julian(d) 
        if jd<0:
            jd = dates.num2julian(d + 3442850)
            print "You are probably using an old version of Matplotlib..."
    else:
        jd = []

    return jd

def jd2gd(juldat):
    """ Convert a numerial Julian date into a Gregorian date using Pylab.
        Timezone returned will be UTC.

       :EXAMPLES:
         ::

          print jd2gd(2454324.5)  #--> 2007-08-12 00:00:00
          print jd2gd(2451545)    #--> 2000-01-01 12:00:00

       :SEE ALSO: :func:`gd2jd`"""
    # 2008-08-26 14:03 IJC: Created    
    # 2011-01-22 16:24 IJC: Removed arbitrary (?) subtraction of 3442850 from 'd'
    import matplotlib.dates as dates
    d = dates.julian2num(juldat)
    gd = dates.num2date(d )

    return gd


def nextTransit(planet, jd=None, dt0=0.0):
    """Return JD of the next transit of a planet after a given JD.  If
    jd==None, the current date is used."""
    # 2010-03-19 11:19 IJC: Created
    import datetime
    if jd==None:
        jd = gd2jd(str(datetime.datetime.today()))
    transitNum = int((jd-(planet.tt + dt0))/planet.per)
    return (planet.tt + dt0) +planet.per*(transitNum+1)

def nextEclipse(planet, jd=None, dt0=0.0):
    """Return JD of the next eclipse of a planet after a given JD.  If
    jd==None, the current date is used.

    Note that this only works for planets on circular orbits --
    eccentric orbits will be given an incorrect Eclipse time!"""
    # 2011-01-22 22:07 IJC: Created
    import datetime
    if jd==None:
        jd = gd2jd(str(datetime.datetime.today()))
    transitNum = int((jd - (planet.tt + dt0))/planet.per)
    return (planet.tt + dt0) + planet.per*(transitNum + 0.5)
    
def getPerimeter(a, ecc):
    """Compute perimeter of an ellipse given its semimajor axis and
    eccentricity.  This is only accurate to a few percent.
    """
    # 2010-03-22 21:37 IJC: Created
    b = a*sqrt(1-ecc**2)

    p = pi*(3*(a+b)-sqrt((3*a+b)*(a+3*b)))
    return p
    
def getTransitDuration(p):
    """Compute duration of transit, in days.  Units of inputs should be:

    Rp -- Jupiter Radii

    depth -- (none)

    a  -- AU

    ecc-- (none)

    P  -- days

    inc-- degrees
    """
    # 2011-01-22 16:42 IJC: Added error-catching for odd, eccentric
    # orbits that aren't properly handled by this algorithm.  I should
    # probably do it correctly!

    peri = getPerimeter(p.a, p.ecc) * AU/rjup
    vorb = (peri/p.per)
    
    term1 = (p.r*(1+1./sqrt(p.depth)))**2
    term2 = (p.a*cos(p.i*pi/180)*AU/rjup)**2

    if term1 > term2:
        ret = 2*sqrt(term1 - term2)/vorb
    else:
        print "%s: CANNOT COMPUTE TRANSIT DURATION!!!" % p.name
        ret = 0.0

    return ret
    
    

def getTransitTimes(planet, *arg, **kw):
    """
    If no arguments passes, get info for the next 10 transits.
    If one (Julian) Date entered, print transits until that date.
    If two (Julian) Dates entered, print transits between those dates.

    If keyword eclipse==True, get the Eclipse times instead.
    """
    # 2010-03-19 11:19 IJC: Created
    # 2011-01-22 22:08 IJC: Added eclipse keyword option

    nextEventFunc = nextTransit
    if kw.has_key('eclipse'):
        if kw['eclipse'] is True:
            nextEventFunc = nextEclipse

    dt0 = 0.0
    if kw.has_key('dt0'):
        dt0 =  kw['dt0']

    if len(arg)==0:
        firstEvent = nextEventFunc(planet, dt0=dt0)
        nEvents = 10

    elif len(arg)==1:
        firstEvent = nextEventFunc(planet,arg[0], dt0=dt0)
        lastEvent = nextEventFunc(planet,arg[0], dt0=dt0)
        nEvents = int((lastEvent-firstEvent)/planet.per+0.001) # roundoff

    else: # len(arg)==2
        firstEvent = nextEventFunc(planet,arg[0], dt0=dt0)
        lastEvent = nextEventFunc(planet,arg[1], dt0=dt0)
        nEvents = int((lastEvent-firstEvent)/planet.per+0.001) # roundoff

    return [firstEvent+ii*planet.per for ii in range(nEvents)]

def getMoonAngle(jd, target, obs):
    """Returns the angle between a target and the moon (in radians).

    :INPUTS:
      jd : float or 1D numpy array
         Julian Dates of times
         
      target : ephem.star
         target  being observed

      obs : ephem.Observer
         observatory with latitude, longitude, and elevation set
         """
    # 2011-09-30 09:56 IJMC: Created from getAirmassAndSunAlt

    if not hasattr(jd, '__iter__'):
        jd = [jd]
        JDwas1D = True
    else:
        JDwas1D = False

    moon = ephem.Moon()

    njd = len(jd)
    moonangle = zeros(njd, float)
    for ii in range(njd):
        t = jd[ii]
        obs.date=jd2gd(t)
        target.compute(obs)
        moon.compute(obs)
        moonangle[ii] = ephem.separation(target, moon)

    if JDwas1D:
        moonangle = moonangle[0]

    return moonangle


def getAirmassAndSunAlt(jd, target, obs) :#, twilight=12, airmax=2.2):
    """ Return airmass and sun altitude at a set of times.
    INPUTS:
       JD -- a sequence of times
       target -- an ephem.star-class object 
       obs -- an ephem.Observer-class object

    Objects w/airmass>9.99 (or below the horizon) are set to 9.99
    """
    #2010-03-19 11:32 IJC: Created

    if not hasattr(jd,'__iter__'):
        jd = [jd]

    sun = ephem.Sun()

    njd = len(jd)
    airmass = zeros(njd,float)
    sunalt = zeros(njd,float)
    for ii in range(njd):
        t = jd[ii]
        obs.date=jd2gd(t)
        target.compute(obs)
        sun.compute(obs)
        airmass[ii] = 1./cos(pi/2-target.alt)
        sunalt[ii] = 180*sun.alt/pi
    airmass[(airmass<0) + (airmass > 9.99)] = 9.99
    return airmass, sunalt

def printflags(observable):
  nobs = len(observable)
  flagstring = ''
  for ii in range(nobs):
    if observable[ii]: 
      flagstring += str(ii)
    else:
      flagstring += ' '
  return flagstring

def printstats(jd, sunalt, airmass, delim=' '):
  day = jd2gd(jd)
  ret = 'UT=%02g%02gZZZsun=%5.1fZZZz=%5.2f' %(day.hour,day.minute,sunalt,airmass)
  #ret = 'UT=%02g%02g sun=%5.1f z=%5.2f' %(day.hour,day.minute,sunalt,airmass)
  ret = ret.replace(' ', '=')
  ret = ret.replace('ZZZ', delim)
  return ret

def printline(observable, jdCen, jd, sunalt, airmass, delim=' '):
  jd = array(jd).ravel()
  sunalt= array(sunalt).ravel()
  airmass = array(airmass).ravel()
  dCen = jd2gd(jdCen)
  njd = len(jd)
  if njd<>len(sunalt) or njd<>len(airmass):
    ret = 'input JD, sunalt, and airmass must have equal lengths\n'
  else:
    ret = printflags(observable)
    ret += '%s%07.6f' % (delim, jdCen)
    ret += '%s%i%s%2i%s%2i' % (delim*2, dCen.year,delim, dCen.month,delim, dCen.day)
    for ii in range(njd):
      ret += '%s%s' % (delim*2, printstats(jd[ii], sunalt[ii], airmass[ii], delim=delim))
  return ret

def isGoodPlanet(planet):
    """Check if planet object has the necessary fields, and that they are
    non-empty."""
    #2010-03-19 13:39 IJC: Created
    # 2011-01-22 16:47 IJC: Updated to check for finite values
    
    isGood = True
    for key in planetkeys:
        if not hasattr(planet,key):
            isGood = False
            break
        else:
            thiskey = eval('planet.%s' % key)
            if thiskey.__class__==str:
                if len(thiskey)==0:
                    isGood = False
                    break
            elif not isfinite(thiskey):
                isGood = False
                #else          Do nothing
                
    return isGood


def transitChart(planets, date1, date2, obs='lick', airmax=2.5, twilight=12, tpad=1, cutoff=3, eclipse=False, delim=' ', chron=False, dt0=0.0):
    """
    :INPUTS:
      planets : objects with the following fields:
          .per (period/days) 

          .tt (transit ephemeris, JD)

          .t14 (transit duration, days)

          .ra_string (string; RA of target in hh:mm:ss)

          .dec_string (string; Dec of target in dd:mm:ss)

      date1, date2 : str
          Date strings (e.g., of the form "YYYY-MM-DD") bracketing the
          timespan for which transits/eclipses will be calculated.

      obs : str
        'lick' or 'keck' or 'lapalma' or 'mtgraham' or 'mtbigelow' or
         'andersonmesa' or 'kpno' or 'ctio' or 'cerropachon' or
         'palomar' or 'cerroparanal' or 'lasilla' or 'calaralto' or
         'lascampanas'

      airmax : float
         Maximum acceptable airmass

      twilight : float
         Minimum acceptable angular distance of sun below horizon, in
         degrees.

      tpad : float
         Number of hours by which ingress and egress should be padded

      cutoff : int
         Minimum number of acceptable "observational checkpoints."
         Valid values are 0-5, inclusive.  A value of "5" will only
         show fully visible transits (pre-ingress, ingress,
         mid-transit, egress, post-egress), "0" will show ALL transits
         (even, e.g., those during daylight), and intermediate values
         will show fully visible transits and some number of partial
         transits.

      eclipse : bool 
         If True, compute times of eclipses, not of transits. Note
         that eclipse times are all computed assuming circular orbits!

      delim : str
         Character(s) used to delimit the output text table

      chron : bool
         If True, sort all computed events (for ALL planets
         considered) by mid-event time. If False, events will be
         listed separately for each planet.

      dt0 : float
        Number of days by which to shift transit center time: a "fudge
        factor."  A positive value here means that the output text
        table will show events occuring later than the '.tt' field of
        the planet object would otherwise indicate.  Leave this set to
        "zero" unless you have a good reason to do otherwise!
                        

    :EXAMPLE:
      ::

        import transittime as tt
        
        class planet:   # Simplest valid planet class
            def __init__(self):
                return

        p = planet()
        p.per = 1.58040482
        p.tt, p.t14 = 2454980.7487955, 0.03661806
        p.ra_string, p.dec_string = '17:15:18.94', '04:57:49.70'
        p.name = 'GJ 1214 b'
        text = tt.transitChart(p, '2013-01-01','2013-12-31')

    :EXAMPLE:
      ::
      
        import analysis as an  # requires :doc:`analysis`
        import transittime as tt
        from numpy import array

        planet_names = an.getobj()
        all_planets = map(an.getobj, planet_names)
        transit_flag = array([p.transit==1 for p in all_planets])
        transiters = array(all_planets)[transit_flag]

        text = tt.transitChart(transiters, '2013-01-01', '2013-08-01')

    :NOTES:
       Always check transit predictions with at least two independent tools!

       Eclipse times are currently computed assuming circular orbits.
       Need to fix this...
    """
    # 2010-03-19 11:39 IJC: Created
    # 2010-10-11 23:08 IJC: Updated with LaPalma location
    # 2011-01-22 22:11 IJC: Added eclipse flag
    # 2011-02-08 09:23 IJC: Added delim flag
    # 2011-08-23 16:43 IJMC: Set target.name = planet.name
    # 2011-08-26 16:41 IJMC: Added 'chron' flag.
    # 2011-09-13 20:59 IJMC: Slightly modified planet header text
    # 2011-09-20 15:06 IJMC: Added 'mtgraham' observatory location
    # 2012-01-25 11:54 IJMC: Added 'cerropachon'
    # 2012-04-06 15:50 IJMC: Added 'palomar'
    # 2012-08-06 22:14 IJMC: Added cerro paranal and la silla and calar alto
    # 2012-09-05 08:52 IJMC: Added las campanas
    # 2013-01-15 10:11 IJMC: Added mtbigelow/catalina option, updated DOC text.

    if not hasattr(planets,'__iter__'):
        planets = [planets]

    nplanets = len(planets)
    jd1 = gd2jd(date1)
    jd2 = gd2jd(date2)
    ind = [1,2,3]  # [01234] = [pre-, ingress, mid-transit, egress, post-]

    obs = str.lower(obs)
    observer = observing.setupObservatory(obs)

    # Start making the text file lines:
    if eclipse:
        eventType = 'ECLIPSE'
    else:
        eventType = 'TRANSIT'
    init_header = ["%s listing between %s and %s, for %s observatory\n" % (eventType, date1,date2, obs)]
    init_header.append("The numerical string specifies which of the following events are visible:\n")
    init_header.append("0: pre-ingress\n1: first contact\n2: mid-%s\n3: fourth contact\n4: post-egress\n" % eventType )
    init_header.append("where pre-ingress and post-egress have been padded by %f hours\n\n" % tpad)
    
    init_header.append('Criteria for observability: z<%f and sun_altitude<-%f deg\n\n' % (airmax,twilight))
    init_header.append("Clock times below are UT, but HAVE NOT been corrected for heliocentricity\n\n")
    init_header.append("Created by Ian Crossfield (but based on Josh Winn of MIT's charts)\n\n")
    planetevents = []
    eventcenters = []
    ret = init_header
    for planet in planets:
        if not hasattr(planet, 'name'):
            planet.name = '(unnamed planet)'
        pheader = []
        pfooter = []
        pheader.append('\n')
        pheader.append('%s      T0: %4.10f     Per: %4.15f\n' %(planet.name,planet.tt + dt0,planet.per))
        if dt0<>0:
            pheader.append('         (all transit center times have been shifted by %4.10f days)\n' % dt0)

        if not isGoodPlanet(planet):
            planet.t14 = getTransitDuration(planet)
            pheader.append ('(transit duration missing; approximately computed: %3.2f min)\n' %(planet.t14*1440) )

        if isGoodPlanet(planet):
            pheader.append('      HJD-midtransit  UTdate-mid            Ingress:                    Middle:                      Egress:\n')
    
            # Setup target object:
            target = ephem.star('Rigel')
            target.name = planet.name
            target._ra, target._dec = planet.ra_string,planet.dec_string
            target._pmra, target._pmdec = 0.,0.
    
            times = getTransitTimes(planet, jd1,jd2, eclipse=eclipse, dt0=dt0)
            ntimes = len(times)
            for ii in range(ntimes):
                tOffsets = array([-planet.t14/2-hr*tpad, -planet.t14/2, 0., \
                                       +planet.t14/2,+planet.t14/2+hr*tpad])+times[ii]
                if not isfinite(tOffsets[0]):
                    print "some of tOffsets were not finite."
                    print "tOffsets>>", tOffsets
                    pdb.set_trace()

                moonangle =  getMoonAngle(tOffsets, target, observer) * 180./pi

                airmass, sunalt = getAirmassAndSunAlt(tOffsets, target, observer)
                observable = (sunalt<-twilight) * (airmass>0) * \
                    (airmass<airmax) #* (moonangle > moonlimit)
                if observable.sum()>=cutoff:
                    thisline = printline(observable, times[ii], tOffsets[ind], sunalt[ind], airmass[ind], delim=delim)
                    moonsep = '%5.1f' % min(moonangle)
  #ret = 'UT=%02g%02g sun=%5.1f z=%5.2f' %(day.hour,day.minute,sunalt,airmass)
                    moonsep = moonsep.replace(' ', '=')
                    thisline = thisline + (delim*2) + 'moonSep=' + moonsep + '\n'
                    pfooter.append(thisline)
                    
            #pdb.set_trace()

        else:
            pheader.append('planet object lacks a necessary field:\n')
            checkstr = ''.join(['%s: %s  ' %(key,eval("planet.%s"%key)) for \
                                    key in planetkeys])
            pfooter.append('Checking for fields:   %s\n' % checkstr)
        
        if len(pfooter)>0:
            ret = ret + pheader + pfooter
            planetevents = planetevents + \
                ([line.replace('\n', delim+planet.name+'\n') \
                      for line in pfooter])


    if chron:
        eventcenters = map(float, [pe[6:20] for pe in planetevents])
        chronindex = argsort(eventcenters)
        chronret = init_header + list(array(planetevents)[chronindex])
        ret = chronret

    return ret
    
    #tOffsets = array([tCen-p.t14/2-hr, tCen-p.t14/2, tCen, tCen+p.t14/2,tCen+p.t14/2+hr])
    #print obs.date, target.alt, 1./cos(pi/2-target.alt), sun.alt
    #

def makeCharts(path='/Users/ianc/temp/', obs=['lick', 'keck', 'cerropachon'], eclipse=[True, False], \
                   months=18, clobber=False, kw=None, planet_names=None):
    """Make transit charts for specified observatories for all planets.

    :OPTIONS:
      path : str
        directory in which to write text output

      obs : list of str
        observatories, acceptable to :func:`transitChart`

      eclipse : list of bool
        whether to compute eclipses (if True) or transits (if False).
        Note that eclipse predictions assume circular orbits!

      months : int
        Number of (31-day) months from present date for computations

      clobber : bool
        If true, overwrite file if it already exists.  Otherwise, don't.

      kw : dict or None
        dict of keywords to pass to :func:`transitChart`

      planet_names : None or list of planet objects
        Planets for which table should be constructed.  If "none", do
        it for all planets in the database.

    :REQUIREMENTS:
      :doc:`analysis`, :doc:`datetime`, :doc:`os`

    :SEE_ALSO:
      :func:`transitChart`
        """
    # 2011-11-01 21:12 IJMC: Created to save me some time.
    # 2013-01-08 23:23 IJMC: Added planet_names option
    # 2015-04-10 20:23 IJMC: Updated checking for transit flag; no longer required.

    import datetime, os
    
    # Check inputs:
    if not hasattr(eclipse, '__iter__'):
        eclipse = [eclipse]

    if not hasattr(obs, '__iter__') or isinstance(obs, str):
        obs = [obs]

    if not kw.__class__==dict:
        print "Input 'kw' must be of type dict; thus 'kw' has not set any keywords."
        kw = dict()
    elif kw.has_key('chron'):
        print "'Chron' key is unnecessary (and ignored).  ",
        print "Both chronological and non-chronological transit charts will be output."
        junk = kw.pop('chron')

    # Prepare dates:
    startdate = datetime.datetime.today() - datetime.timedelta(days=1)
    stopdate  = startdate + datetime.timedelta(days=months*31)
    startstr = ('%s' % startdate)[0:10]
    stopstr  = ('%s' %  stopdate)[0:10]

    # Prepare list of planet objects:
    if planet_names is None:
        planet_names = an.getobj()
        all_planets = map(an.getobj, planet_names)
    else:
        all_planets = planet_names

    transit_flag = ones(len(all_planets), bool)
    for ii,p in enumerate(all_planets):
        if hasattr(p, 'transit'):
            transit_flat[ii] = p.transit==1
    transiters = array(all_planets)[transit_flag]


    # Loop over all options:
    for thisobs in obs:
        format_params = (path, thisobs, startstr, stopstr)
        for doeclipse in eclipse:
            # Set output filenames:
            if doeclipse==True:
                cname = '%s%s_chronological_eclipse_chart_%s--%s.txt' % format_params
                sname = '%s%s_planet-sorted_eclipse_chart_%s--%s.txt' % format_params
            else:
                cname = '%s%s_chronological_transit_chart_%s--%s.txt' % format_params
                sname = '%s%s_planet-sorted_transit_chart_%s--%s.txt' % format_params

            # Generate transit charts:
            ctext = transitChart(transiters, startstr, stopstr, obs=thisobs, \
                                    eclipse=doeclipse, chron=True, **kw)
            stext = transitChart(transiters, startstr, stopstr, obs=thisobs, \
                                    eclipse=doeclipse, chron=False, **kw)

            # Write files to disk:
            if os.path.isfile(cname) and not clobber:
                print "File %s exists and clobber=False; not writing file to disk." % cname
            else:
                f = open(cname, 'w')
                f.writelines(ctext)
                f.close()

            if os.path.isfile(sname) and not clobber:
                print "File %s exists and clobber=False; not writing file to disk." % sname
            else:
                f = open(sname, 'w')
                f.writelines(stext)
                f.close()

    return


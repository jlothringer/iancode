"""
2015-04-23 IJMC: Built this from modified EFOSC reduction script.
"""


import numpy as np
import pylab as py
import tools
import analysis as an
import ir
import spec
from pyraf import iraf 
import os
from astropy.io import fits as pyfits
import nsdata as ns
from scipy import signal
import pdb

_home = os.path.expanduser('~')
dir0 = os.getcwd()

date = '20150823'
date = '20150822'
date = '20150821'
date = '20150824'
date = '20150111'
date = '20150110'
date = '20150825'
date = '20151103'
date = '20151102'
date = '20151101'

_data = _home + '/proj/mdwarfs/data/raw/' + date +'/'
_proc = _home + '/proj/mdwarfs/data/proc/' + date +'/'

doextract=True
writefiles = True
wavecal = True
clobber = True

ordlocs = None
_wldat = 'database'
lamp_list = _home + '/proj/pcsa/data/atmo/efosc_gr16_HeAr.dat'

iraf.load('fitsutil')
iraf.load('noao')
iraf.load('astutil')
iraf.load("imred")
iraf.load('specred')

# Set EFOSC parameters:
#iraf.unlearn('specred')

# Set parameters for aperture tracing, flat-field normalizing, etc.
iraf.specred.dispaxis = 2
iraf.specred.apfind.nfind = 1
iraf.specred.apfind.recenter = "Yes"
iraf.specred.apfind.nsum = -3

iraf.apall.ylevel = "INDEF" #0.05
iraf.apall.bkg = "Yes"
iraf.apall.ulimit = 5
iraf.apall.llimit = -5

t_naverage = -25
iraf.aptrace.order = 3
iraf.aptrace.niterate = 8
iraf.aptrace.step = 60
iraf.aptrace.naverage = t_naverage  
iraf.aptrace.nlost = 9999
iraf.aptrace.recenter = "YES"



# Set detector properties:
gain = 1.35  # photons (i.e., electrons) per data unit
readnoise = 10.7   # photons (i.e., electrons)
iraf.imcombine.gain = gain
iraf.imcombine.rdnoise = readnoise
iraf.apall.gain = gain
iraf.apall.readnoise = readnoise
iraf.apnormalize.gain = gain
iraf.apnormalize.readnoise = readnoise



fns = np.array([line.strip() for line in os.popen('ls %s*.fits' % _data)])
headers = map(pyfits.getheader, fns)


def getkeys(headers, key, default='none'):
    val = []
    for h in headers:
        if key in h.keys():
            val.append(h[key])
        else:
            val.append(default)
    return np.array(val)

objects = getkeys(headers, 'OBJECT')
types = getkeys(headers, 'ESO DPR TECH')
targnames = getkeys(headers, 'ESO OBS TARG NAME')


try:
    masterbias_alt = pyfits.getdata(_proc + 'temp_masterbias.fits')
except:
    pass
if 'BIAS' in objects:
    masterbias = ir.makemasterframe(fns[objects=='BIAS'])
else:
    print "You had no BIAS frames! Seeking masterbias.fits...."
    masterbias = masterbias_alt.copy()


waveCalRef = None
targs = np.unique(targnames)
red_objs = []
#targ = targs[12]
datasets = []
calfns = []
redfns = []
for targ in targs:
    thisind = (targnames == targ) * (types=='SPECTRUM')
    thesefiles = fns[thisind]
    theseflatfns = fns[thisind * (objects=='FLAT')]
    thesearcfns = fns[thisind * (objects=='WAVE')]
    thesearcheaders = map(pyfits.getheader, fns[thisind * (objects=='WAVE')])
    sciind = thisind * (objects<>'WAVE') * (objects<>'FLAT')
    thesescifns = fns[sciind]
    thesesciheaders = map(pyfits.getheader, fns[sciind])
    nthissci = len(thesescifns)
    if len(thesearcfns)>0 and len(theseflatfns)>0 and nthissci>0 and targ<>'none':
        obj = targnames[sciind][0]
        red_objs.append(obj)
        initflat = ir.makemasterframe(theseflatfns) 
        if not initflat.shape==masterbias.shape:  masterbias = masterbias_alt
        thisflat = ir.makemasterframe(theseflatfns) - masterbias
        normflat = spec.normalizeSpecFlat(thisflat.T, nspec=1, minsep=50).T
        if (np.median(thisflat)<1000) or (np.median(initflat)<1000) or (an.dumbconf(normflat.ravel(), .683)[0]<0.01) or (an.dumbconf(normflat.ravel(), .683)[0]>0.1) or \
                (date=='20150821' and (('205924614' in obj) or ('GJ887' in obj))): # Bad flats?!
            print "Bad flat (for some reason). Using default tempflat!"
            normflat = pyfits.getdata(_proc + 'tempflat.fits')
        if writefiles:
            pyfits.writeto(_proc + targ.replace(' ','_')+'normflat.fits', normflat, clobber=clobber)
        for jj in xrange(nthissci):
            calsci = (pyfits.getdata(thesescifns[jj]) - masterbias) / normflat
            calfn = _proc + os.path.split(thesescifns[jj])[1]
            if writefiles:
                pyfits.writeto(calfn, calsci.astype(np.float32), thesesciheaders[jj], clobber=clobber)
            redfn = calfn.replace('.fits', '_spec.fits')
            calfns.append(calfn)
            redfns.append(redfn)

        trace, xy = spec.traceorders(calfns[-1], pord=3, dispaxis=1, retfits=True, ordlocs=[[thesesciheaders[0]['NAXIS2']/2-10, thesesciheaders[0]['NAXIS1']/2-10+20]], fitwidth=50)

        arcspecs = []
        for jj in xrange(len(thesearcfns)):
            calarc = (pyfits.getdata(thesearcfns[jj]) - masterbias) / normflat
            calfn = _proc + os.path.split(thesearcfns[jj])[1]
            if writefiles: 
                pyfits.writeto(calfn, calarc.astype(np.float32), thesearcheaders[jj], clobber=clobber)
            out = spec.makeprofile(calfn, trace, dispaxis=1, nsigma=np.inf, retall=True)[0]
            arcspecs.append(np.array([[(out[1].sum(1))]]))
            redfn = calfn.replace('.fits', '_spec.fits')

        arcspec = np.median(np.array(arcspecs), 0)
        archeader = thesearcheaders[jj]
        keys = 'BANDID1', 'BANDID2', 'BANDID3', 'BANDID4', 'APNUM1', 'WCSDIM'  , 'CTYPE3'  , 'CD3_3'   , 'LTM1_1'  , 'LTM2_2'  , 'LTM3_3'  , 'WAT0_001', 'WAT1_001', 'WAT2_001', 'WAT3_001',  3
        keyvals = 'spectrum: background fit, weights variance, clean yes' , 'background: background fit'  ,'sigma - background fit, weights variance, clean yes', 'wavelength', '1 1 540.99 550.99','1 1 538.02 548.02' ,'LINEAR  ',    1.,    1.,    1.,    1., 'system=equispec'   , 'wtype=linear label=Pixel'    , 'wtype=linear' , 'wtype=linear' 
        for kk, kv in zip(keys, keyvals):
            archeader[kk] = kv

        if writefiles:
            pyfits.writeto(redfn, np.tile(arcspec, (4,1,1)).astype(np.float32), archeader, clobber=clobber)

        if wavecal:
            os.chdir(_proc)
            iraf.chdir(_proc)
            loc_redfn =os.path.split(redfn)[1]
            if waveCalRef is None:
                iraf.identify(loc_redfn, database=_wldat, ftype='emission', fwidth=3, order=2, niterate=3, cradius=3, coordlist=lamp_list, function='spline3')
                waveCalRef = '' + loc_redfn
            else:
                iraf.reidentify(waveCalRef, loc_redfn, interactive='no', override='yes', refit='yes', nlost=1, cradius=10, addfeatures='no', coordlist=lamp_list) #, function='spline3', order=2, niterate=3)
            disp_soln = ns.getdisp(_wldat + os.sep + 'id' + loc_redfn.replace('.fits',''), 'spline3')
            if writefiles:
                ns.wspectext(loc_redfn)
        datasets.append([redfns[-nthissci:], redfn])
    
        os.chdir(dir0)
        iraf.chdir(dir0)





ns.strl2f('tempjunkcal', calfns)
ns.strl2f('tempjunkred', redfns)



if doextract:
    # Construct a MASTER FRAME, and trace it::
    if os.path.isfile(_proc + 'masterframe_override.fits'):
        masterfn = _proc + 'masterframe_override.fits'
    else:
        sci_fns = []
        for d in datasets: sci_fns+= d[0]
        for ii in range(len(sci_fns)): sci_fns[ii] = sci_fns[ii].replace('_spec.fits', '.fits')
        sumframe = np.zeros(pyfits.getdata(sci_fns[0]).shape, dtype=float)
        for fn in sci_fns: sumframe += pyfits.getdata(fn)
        masterfn = _proc + 'masterframe.fits'
        pyfits.writeto(masterfn, sumframe, clobber=clobber)

    horizsamp='%i:%i' % (masterbias.shape[0]*.02, masterbias.shape[0]*.97)
    iraf.apall(masterfn, output=masterfn.replace('.fits', '_spec.fits'), format='onedspec', recenter='YES',resize='YES',extras='yes', nfind=1, nsubaps=1, minsep=100, weights='variance', bkg='yes', b_function='chebyshev', b_order=1,  b_naverage=-3, b_niterate=2, t_order=7, t_niterate=4, t_naverage=t_naverage, background='fit', clean='yes', interactive=True, review=False, b_sample='-30:-10,10:30', trace='YES',edit='YES',fittrace='YES',extract='YES', find='YES', t_sample=horizsamp)

    # Use the master-trace aperture to extract all spectra (even faint ones!)
    if clobber:
        for fn in redfns:
            fn2 = fn.replace('spec.fits', 'spec.0001.fits')
            if os.path.isfile(fn2): os.remove(fn2)
    iraf.apall('@tempjunkcal', output='@tempjunkred', format='onedspec', recenter='NO',resize='NO',extras='yes', nfind=1, nsubaps=1, minsep=100, weights='variance', bkg='yes', b_function='chebyshev', b_order=1,  b_naverage=-3, b_niterate=2, t_order=3, t_niterate=4, t_naverage=t_naverage, background='fit', clean='yes', interactive=True, review=False, b_sample='-30:-10,10:30', trace='NO',edit='NO',fittrace='NO',extract='YES', find='NO', references='last', t_sample=horizsamp)



#Clobber existing output image /Users/ianc/proj/mdwarfs/data/proc/20150821/EFOSC.2015-08-22T03:07:43.730_spec.0001? ('no'): y

os.chdir(dir0)
print "... and we're done!"
print "Generating plots..."







#datasets = tools.loadpickle(_proc + date+'_datasets.pickle')

#w16, t16 = pyfits.getdata(os.path.expanduser('~/proj/mdwarfs/data/proc/20150110/efosc_gr16_counts2flam.fits'))
w16, t16 = pyfits.getdata(os.path.expanduser('~/proj/mdwarfs/data/proc/20150823/efosc_gr16_counts2flam.fits'))
waves, specs, corspecs, especs, corespecs, objs, snrs, exptimes, sampheaders = [], [], [], [], [], [], [], [], []
for ii in xrange(len(datasets)):
  ss_fns = [(sp.replace('.fits', '.0001.fits')) for sp in datasets[ii][0]]
  if np.all(map(os.path.isfile, ss_fns)):
      sampheaders.append(pyfits.getheader(ss_fns[0]))
      ess = np.array([pyfits.getdata(ss_fn) for ss_fn in ss_fns])[:,-1,0]
      ss = np.array([pyfits.getdata(ss_fn) for ss_fn in ss_fns])[:,0,0] 
      bgs = np.array([pyfits.getdata(ss_fn) for ss_fn in ss_fns])[:,-2,0] 
      master = np.median(ss, axis=0)
      scales = np.array([an.lsq(ss[jj], master)[0][0] for jj in range(ss.shape[0])])
      newss = ss * scales.reshape(ss.shape[0], 1)
      spec = np.median(newss, 0) * ss.shape[0]
      espec_empirical   = newss.std(0) * np.sqrt(ss.shape[0])
      espec_statistical = np.sqrt((ess**2).sum(0))

      #espec = np.vstack((espec_empirical, espec_statistical)).max(0)
      espec = espec_statistical
      ww = ns.getdisp(_proc+_wldat + os.sep + 'id' + os.path.split(datasets[ii][1])[1].replace('.fits', ''), 'spline3')
      objs.append(pyfits.getval(datasets[ii][0][0].replace('.fits', '.0001.fits'), 'ESO OBS TARG NAME'))
      exptimes.append(np.sum([pyfits.getval(el.replace('.fits', '.0001.fits'), 'EXPTIME') for el in datasets[ii][0]]))
      waves.append(ww)
      specs.append(spec)
      corspecs.append(spec / np.interp(ww, w16, t16))
      snrs.append(np.median(spec/espec))
      especs.append(espec)
      corespecs.append(espec / np.interp(ww, w16, t16))

py.close('all')
tlines = []
py.ioff()
for ii in range(len(waves)):
    losnr = signal.medfilt(corspecs[ii]/corespecs[ii], 5)
    py.figure(1+ii)
    py.plot(waves[ii], corspecs[ii])
    py.title('%s, <S/N> ~ %i' % (objs[ii], snrs[ii]))
    py.minorticks_on()
    py.xlabel('Wavelength [Ang]')
    py.ylabel('Flux (F_Lambda)')
    ymax = corspecs[ii].max()
    if (losnr>10).any(): 
        ymax = corspecs[ii][losnr>10].max()*1.1
    py.ylim(0, ymax)
    py.figure(1+ii+len(waves))
    py.plot(waves[ii], losnr)
    if py.ylim()[1]>snrs[ii]*3:     py.ylim(0, snrs[ii]*3)
    py.title('%s, <S/N> ~ %i' % (objs[ii], snrs[ii]))
    py.minorticks_on()
    py.xlabel('Wavelength [Ang]')
    py.ylabel('S/N')
    tlines.append("%s\t%1.1f\t%i" % (objs[ii], exptimes[ii], snrs[ii]))

rootfn = _proc + 'spectra_%s_s2n_v' % date
iter = 1
while os.path.isfile(rootfn + '%i.pdf' % iter):
    iter += 1

tools.printfigs(rootfn + '%i.pdf' % iter, range(len(waves)+1, len(waves)*2+1))

rootfn = _proc + 'spectra_%s_flam_v' % date
iter = 1
while os.path.isfile(rootfn + '%i.pdf' % iter):
    iter += 1

tools.printfigs(rootfn + '%i.pdf' % iter, range(0,len(waves)+1))
                      
py.close('all')
py.ion()

for ii in range(len(waves)):
    outdat = np.vstack((waves[ii], corspecs[ii], corespecs[ii]))
    pyfits.writeto(_proc+'%s_flam_ROUGH.fits' % objs[ii], outdat, sampheaders[ii], clobber=clobber)

for line in tlines: print line                            

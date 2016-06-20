# 2015-06-12 IJMC: Copied to AOM's directory and set up for him.

import numpy as np
import pylab as py
import tools
import ir
import spec
from pyraf import iraf 
import os
from astropy.io import fits as pyfits
import nsdata as ns
import analysis as an
from astropy.io import ascii
from scipy import signal

_home = os.path.expanduser('~')
_ianhome = os.path.expanduser('~ianc')
dir0 = os.getcwd()

# SET DATA DIRECTORIES:

#_data = _ianhome + '/proj/mdwarfs/data/raw/20150201/'
#_proc = _home + '/proj/mdwarfs/data/proc/20150201/'
#_data = _ianhome + '/proj/mdwarfs/data/raw/20150129/'
#_proc = _home + '/proj/mdwarfs/data/proc/20150129/'

#_data = _home + '/Astronomy/CAMPARE_2015/raw_data/mdwarfs/20150129/'
#_proc = _home + '/Astronomy/CAMPARE_2015/test_1/'

#_data = _home + '/Astronomy/CAMPARE_2015/raw_data/CHILE/06_28_15_rawdata/'
#_proc = _home + '/CAMPARE_summer_program/data/merged/'

_data = _ianhome + '/proj/mdwarfs/data/raw/20150825/'
_proc = _home + '/proj/mdwarfs/data/proc/20150825/'
_data = _ianhome + '/proj/mdwarfs/data/raw/20150826/'
_proc = _home + '/proj/mdwarfs/data/proc/20150826/'
_data = _ianhome + '/proj/mdwarfs/data/raw/20151026/'
_proc = _home + '/proj/mdwarfs/data/proc/20151026/'


_data = _ianhome + '/proj/mdwarfs/data/raw/20160329/'
_proc = _home + '/proj/mdwarfs/data/proc/20160329/'
_data = _ianhome + '/proj/mdwarfs/data/raw/20160608/'
_proc = _home + '/proj/mdwarfs/data/proc/20160608/'

writefiles = True
wavecal = True
doextract = True 
extractNewOnly = True

ordlocs = None
_wldat = 'database'
lamp_list = _ianhome + '/proj/pcsa/data/atmo/sofi_xenon_r1000_JHK.dat'
_masterflat = _ianhome + '/proj/mdwarfs/data/proc/sofi/sofi_masterflat_2014B.fits'
#lamp_list = _home + '/Astronomy/CAMPARE_2015/raw_data/mdwarfs/20150129/sofi_xenon_r1000_JHK.dat'
#_masterflat = _home + '/Astronomy/CAMPARE_2015/raw_data/mdwarfs/20150129/sofi_masterflat_2014B.fits'
masterflat = pyfits.getdata(_masterflat)
idlexec = os.popen('which idl').read().strip()
#iraf.load('fitsutil')
iraf.load('noao')
iraf.load('astutil')
iraf.load("imred")
iraf.load('specred')

# Set EFOSC parameters:
#iraf.unlearn('specred')

# Set parameters for aperture tracing, flat-field normalizing, etc.
horizsamp = "20:500 550:995"
t_naverage = -6

iraf.specred.dispaxis = 1
iraf.specred.apfind.nfind = 1
iraf.specred.apfind.recenter = "Yes"
iraf.specred.apfind.nsum = -3

iraf.apall.ylevel = "INDEF" #0.05
iraf.apall.bkg = "Yes"
iraf.apall.ulimit = 5
iraf.apall.llimit = -5
iraf.apall.nsum = -50

iraf.aptrace.order = 2
iraf.aptrace.niterate = 3
iraf.aptrace.step = 50
iraf.aptrace.naverage = t_naverage  # take the median
iraf.aptrace.nlost = 999
iraf.aptrace.recenter = "YES"

# Set detector properties:
gain = 5.4  # photons (i.e., electrons) per data unit
readnoise = 2.1   # photons (i.e., electrons)
iraf.imcombine.gain = gain
iraf.imcombine.rdnoise = readnoise
iraf.apall.gain = gain
iraf.apall.readnoise = readnoise
iraf.apnormalize.gain = gain
iraf.apnormalize.readnoise = readnoise



# Try to load filenames ("fns") and headers:
fns = np.array([line.strip() for line in os.popen('ls %s*SOFI.20*.fits' % _data)])
if len(fns)==0:
    fns = np.array([line.strip() for line in os.popen('ls %s*SOFI_*.fits' % _data)])
if len(fns)==0:
    fns = np.array([line.strip() for line in os.popen('ls %s*SOFI.20*.fits' % _proc)])
headers = map(pyfits.getheader, fns)


# Now load header keywords, using this helper function:
def getkeys(headers, key, default='none'):
    val = []
    for h in headers:
        if key in h.keys() or key.replace('HIERARCH ','') in h.keys():
            val.append(h[key])
        else:
            val.append(default)
    return np.array(val)

objects = getkeys(headers, 'OBJECT')
types = getkeys(headers, 'ESO DPR TECH')
targnames = getkeys(headers, 'ESO OBS TARG NAME')
modes = getkeys(headers, 'HIERARCH ESO INS OPTI2 ID')
lamps = getkeys(headers, 'ESO INS LAMP1 NAME')
mjd = getkeys(headers, 'MJD-OBS')
ra0 = getkeys(headers, 'RA')
dec0 = getkeys(headers, 'DEC')
tel_alt0 = (getkeys(headers, 'HIERARCH ESO TEL ALT'))
pixscale = headers[0]['HIERARCH ESO INS PIXSCALE']
nodthrow = getkeys(headers, 'HIERARCH ESO SEQ NODTHROW')
nodoffset = getkeys(headers, 'HIERARCH ESO SEQ CUMOFFSETX')
nod_arcsec = np.zeros(len(nodoffset), dtype=float)



ra  = np.nan + np.zeros(len(ra0 ), dtype=float)
dec = np.nan + np.zeros(len(dec0), dtype=float)
tel_alt = np.nan + np.zeros(len(tel_alt0), dtype=float)
for ii in range(len(tel_alt0)):
    if tools.isnumeric(tel_alt0[ii]): tel_alt[ii] = float(tel_alt0[ii])
    if tools.isnumeric(ra0[ii]):  ra[ii]  = float(ra0[ii])
    if tools.isnumeric(dec0[ii]): dec[ii] = float(dec0[ii])

airmass = 1./np.cos((90-tel_alt)*np.pi/180)



for ii in range(len(nodoffset)):
    try:
        nod_arcsec[ii] = float(nodoffset[ii]) * pixscale
    except:
        nod_arcsec[ii] = np.nan


#masterbias = ir.makemasterframe(fns[objects=='BIAS'])
masterbias = 0.

logsheet = []
timesort = np.argsort(mjd)
fmt = '%20s  %20s  %9.4f  %+9.4f  %4s z=%1.2f  %s\n'
for ii in range(len(fns)):
    logsheet.append(fmt % (objects[timesort][ii], targnames[timesort][ii], ra[timesort][ii], dec[timesort][ii], modes[timesort][ii], airmass[timesort][ii], os.path.split(fns[timesort][ii])[1]))

f = open(_proc + 'logsheet', 'w')
f.writelines(logsheet)
f.close()
arcmodes = modes[(lamps<>'none')]
didwavecal = dict()
for mode in arcmodes:
    didwavecal[mode] = False

#for mode in arcmodes:
#    arcind = (objects=='LAMP') * (modes==mode) * (lamps=='Xenon')
#    thesearcfns = fns[arcind] #fns[thisind * (objects=='LAMP')]
#    thesearcheaders = map(pyfits.getheader, fns[arcind]) #headers[thisind * (objects=='LAMP')]
targs = np.unique(objects)
red_objs = []
datasets = []
calfns = []
redfns = []
for targ in targs:
    thisind0 = (targnames == targ) * (types=='SPECTRUM')
    targmodes = np.unique(modes[thisind0])
    for mode in targmodes:
        thisind = thisind0 * (modes==mode)
        thesefiles = fns[thisind]
        theseflatfns = fns[thisind * (objects=='FLAT')]
        arcind = (objects=='LAMP') * (modes==mode) * (lamps=='Xenon')
        thesearcfns = fns[arcind] #fns[thisind * (objects=='LAMP')]
        thesearcheaders = map(pyfits.getheader, fns[arcind]) #headers[thisind * (objects=='LAMP')]
        sciind = thisind * (objects<>'LAMP') * (objects<>'FLAT')
        thesescifns = fns[sciind]
        thesesciheaders = map(pyfits.getheader, fns[sciind]) #headers[sciind]
        nthissci = len(thesescifns)
        if nthissci>0 and targ<>'none': #and len(theseflatfns)>0
            obj = targnames[sciind][0]
            red_objs.append(obj)
            #thisflat = ir.makemasterframe(theseflatfns) - masterbias
            #normflat = spec.normalizeSpecFlat(thisflat.T, nspec=1, minsep=50).T
            for jj in xrange(nthissci):
                calsci = (pyfits.getdata(thesescifns[jj]) - masterbias) / masterflat
                othernod_ind = (np.abs(nod_arcsec - nod_arcsec[sciind][jj]) > 4 ) * sciind
                if othernod_ind.sum()==0:
                    ref = np.median(calsci, axis=1).reshape(1024, 1)
                elif othernod_ind.sum()==1:
                    ref = (pyfits.getdata(fns[othernod_ind][0]) - masterbias) / masterflat                    
                elif othernod_ind.sum()>=2:
                    dt = np.abs(mjd[sciind][jj] - mjd) #[othernod_ind]
                    ref_inds = [(dt==el).nonzero()[0][0] for el in np.sort(dt[othernod_ind])[0:2]]
                    ref = (0.5*(pyfits.getdata(fns[ref_inds][0]) + pyfits.getdata(fns[ref_inds][1])) - masterbias) / masterflat                    

                procsci = calsci - ref
                calfn = _proc + os.path.split(thesescifns[jj])[1] #.replace('.fits', '_v2.fits')
                if writefiles:
                    thesesciheaders[jj]['DISPAXIS'] = 1
                    pyfits.writeto(calfn, procsci.astype(np.float32).T, thesesciheaders[jj], clobber=True, output_verify='ignore')
                redfn = calfn.replace('.fits', '_spec.fits')
                calfns.append(calfn)
                redfns.append(redfn)

            # Trace out the location of the spectrum in the 2D frame:
            datsub = calsci - np.median(calsci, axis=1).reshape(1024, 1)
            datprofile = np.median(datsub, axis=0)
            datprof_offset = 300
            smdatprofile = datprofile[datprof_offset:-datprof_offset]
            rejcontfit = an.polyfitr(np.arange(smdatprofile.size), smdatprofile, 4, 2)
            cont = np.polyval(rejcontfit, np.arange(smdatprofile.size))
            loc = spec.fitGaussian(smdatprofile-cont, sigmaguess=10)[0][2] + datprof_offset
            fitwidth = 80
            loind, hiind = int(loc-fitwidth/2), int(loc+fitwidth/2)
            smdatsub = datsub[:, loind:hiind]
            smdatsubsub = smdatsub - np.median(smdatsub, axis=1).reshape(1024, 1)
            trace, xy = spec.traceorders(smdatsubsub, pord=3, dispaxis=1, retfits=True, ordlocs=[[thesesciheaders[0]['NAXIS2']/2-10, loc-loind]], medwidth=20)
            trace[:,-1] += loind
            #trace, xy = spec.traceorders(datsub, pord=3, dispaxis=1, retfits=True, ordlocs=[[thesesciheaders[0]['NAXIS2']/2-10, loc]], fitwidth=fitwidth)
            ##trace, xy = spec.traceorders(datsub, pord=3, dispaxis=1, retfits=True, ordlocs=[[thesesciheaders[0]['NAXIS2']/2-10, loc]], fitwidth=50)
            

            arcspecs = []
            for jj in xrange(len(thesearcfns)):
                calarc = (pyfits.getdata(thesearcfns[jj]) - masterbias) / masterflat
                #calfn = _proc + os.path.split(thesescifns[jj])[1].replace('.fits', '_v2.fits')
                calfn = _proc + os.path.split(thesearcfns[jj])[1]
                if writefiles: 
                    pyfits.writeto(calfn, calarc.astype(np.float32), thesearcheaders[jj], clobber=True, output_verify='ignore')
                out = spec.makeprofile(calfn, trace, dispaxis=1, nsigma=np.inf, retall=True)[0]
                arcspecs.append(np.array([[(out[1].sum(1))]]))
                redfn = _proc + ('SOFI_%s_%s_arcs.fits' % (targ, mode)).replace(' ','_') #calfn.replace('.fits', '_spec.fits')

            arcspec = np.median(np.array(arcspecs), 0)
            archeader = thesearcheaders[jj]
            keys = 'BANDID1', 'BANDID2', 'BANDID3', 'BANDID4', 'APNUM1', 'WCSDIM'  , 'CTYPE3'  , 'CD3_3'   , 'LTM1_1'  , 'LTM2_2'  , 'LTM3_3'  , 'WAT0_001', 'WAT1_001', 'WAT2_001', 'WAT3_001',  'FILENAME'
            keyvals = 'spectrum: background fit, weights variance, clean yes' , 'background: background fit'  ,'sigma - background fit, weights variance, clean yes', 'unknown', '1 1 540.99 550.99',3 ,'LINEAR  ',    1.,    1.,    1.,    1., 'system=equispec'   , 'wtype=linear label=Pixel'    , 'wtype=linear' , 'wtype=linear', os.path.split(thesearcfns[-1])[-1]
            for kk, kv in zip(keys, keyvals):
                archeader[kk] = kv

            if (writefiles or wavecal) and not didwavecal[mode]:
                pyfits.writeto(redfn, np.tile(arcspec, (4,1,1)).astype(np.float32), archeader, clobber=True, output_verify='ignore')

            if wavecal and not didwavecal[mode]:
                os.chdir(_proc)
                iraf.chdir(_proc)
                loc_redfn =os.path.split(redfn)[1]
                iraf.identify(loc_redfn, database=_wldat, ftype='emission', fwidth=3, order=2, niterate=3, cradius=3, coordlist=lamp_list, function='spline3')
                disp_soln = ns.getdisp(_wldat + os.sep + 'id' + loc_redfn.replace('.fits',''), 'spline3')
                waveout = pyfits.open(loc_redfn )
                waveout[0].data[-1,0] = disp_soln
                waveout.writeto(loc_redfn, clobber=True)
                if writefiles:
                    ns.wspectext(loc_redfn)
                didwavecal[mode] = True
            datasets.append([redfns[-nthissci:], redfn, mode])

            #os.chdir(dir0)
            #iraf.chdir(dir0)


if extractNewOnly:
    redfns2 = [fn.replace('.fits', '.0001.fits') for fn in redfns]
    ind = np.array(map(os.path.isfile, redfns2))
    calfns = np.array(calfns)[True-ind]
    redfns = np.array(redfns)[True-ind]

nfiles = int(np.ceil(len(calfns)/150.))
for fniter in range(nfiles):
    ns.strl2f('tempjunkcal%i' % fniter, calfns[150*fniter:150*(fniter+1)])
    ns.strl2f('tempjunkred%i' % fniter, redfns[150*fniter:150*(fniter+1)])

if False: # for testing
    input = calfns[0]
    output = redfns[0]
else:
    input = '@tempjunkcal'
    output = '@tempjunkred'
    if len(calfns)==0:  input = None



if doextract and (input is None):
    print "Beginning spectral extraction, but no valid files found for extraction."
elif doextract:
    print "Beginning spectral extraction for %i files" % (len(calfns))
    for fniter in range(nfiles):
        iraf.apall(input+('%i'%fniter), output=output+('%i'%fniter), format='onedspec', recenter='YES',resize='YES',extras='yes', nfind=1, nsubaps=1, minsep=100, weights='variance', bkg='yes', b_function='chebyshev', b_order=1,  b_naverage=-3, b_niterate=2, t_order=4, t_niterate=3, t_naverage=t_naverage, background='fit', clean='yes', interactive=True, review=False, b_sample='-40:-15,15:40', trace='YES',edit='YES',fittrace='YES',extract='YES', find='YES', t_sample=horizsamp)




print "... and we're done!\n"
print "Verify that all datasets were correctly matched up:\n"


for dataset in datasets:
    datfns = [fn.replace('.fits', '.0001.fits') for fn in dataset[0]]
    ra, dec, oo = [], [], []
    for fn in datfns:
        if os.path.isfile(fn):
            ra.append(pyfits.getval(fn, 'RA'))
            dec.append(pyfits.getval(fn, 'DEC'))
            oo.append(pyfits.getval(fn, 'OBJECT'))
        else:
            ra.append(np.nan)
            dec.append(np.nan)
            oo.append(np.nan)
    print "\nNext dataset:"
    for tup in zip(oo, ra, dec, [os.path.split(fn)[1].replace('_spec.0001','') for fn in datfns]):
        print tup

os.chdir(dir0)

#print "Stopping for now. Below are additional spectral calibration steps!"
print "Extracted all spectra. Now beginning spectral calibration steps!"

dataset_modes = np.array([dataset[2] for dataset in datasets])
first_blue = (dataset_modes=='GB').nonzero()[0][0]
first_red = (dataset_modes=='GR').nonzero()[0][0]
wb = pyfits.getdata(datasets[first_blue][1])[-1,0]
wr = pyfits.getdata(datasets[first_red][1])[-1,0]
waves = dict(GB=wb, GR=wr)
w_index = dict(GB=np.argsort(wb), GR=np.argsort(wr))

allspecs = dict(GR=[], GB=[])
#meanspecs = dict(GR=[], GB=[])
#errspecs = dict(GR=[], GB=[])
redobjs = []
redmodes = []
spexkeys = 'DIVISOR', 'XUNITS', 'YUNITS', 'NORDERS', 'ORDERS', 'NAPS', 'START' , 'STOP'
spexvals = 1, 'um', 'DN / s', 1, 1, 1, 1, 1024
for iter,dataset in enumerate(datasets):
    datfns0 = np.array([fn.replace('.fits', '.0001.fits') for fn in dataset[0]])
    datisfile = np.array(map(os.path.isfile, datfns0))
    datfns = datfns0[datisfile]
    if datfns.size==0:
        print "None of these files could be found:", datfns0
    else:
        dats = np.array([pyfits.getdata(fn) for fn in datfns])
        allspecs[mode].append(dats[:,0,0].squeeze())
        #meanspec, errspec = an.wmean(dats[:,0,0], 1./dats[:,-1,0]**2, reterr=True, axis=0)

        for jj,fn in enumerate(datfns):
            if os.path.isfile(fn):
                dathdr = pyfits.getheader(fn)
                mode = dathdr['HIERARCH ESO INS OPTI2 ID']
                obj = dathdr['OBJECT']
                dathdr['ORIGFILE'] = os.path.split(fn)[1]
                dathdr['CO_ADDS'] = dathdr['HIERARCH ESO DET NDIT']
                itime = dathdr['HIERARCH ESO DET DIT'] * dathdr['CO_ADDS']
                dathdr['ITIME'] = itime
                dathdr['AIRMASS'] = 1./np.cos(dathdr['HIERARCH ESO TEL ALT']*np.pi/180.)
                for key, val in zip(spexkeys, spexvals):
                    dathdr[key] = val
                specdat = pyfits.getdata(fn)
                out_spexlike = np.vstack((waves[mode]/1e4, specdat[0,0]/itime, specdat[-1,0]/itime))[:,w_index[mode]].reshape(1,3,1024)
                out_fn = _proc + os.path.split(('sofi_%s_%s_%02i.fits' % (obj, mode, jj)).replace(' ', '_'))[1]
                pyfits.writeto(out_fn, out_spexlike, header=dathdr, clobber=True)
            else:
                print "Could not find file '%s' for converting to SpeXTool format." % fn

        # If only 1 spectrum, can't "xcombspec"; so write out a .DAT file:
        if len(datfns)==1:
            fake_combined_fn = out_fn.replace('00.fits', 'single_combspec')
            pyfits.writeto(fake_combined_fn + '.fits', out_spexlike, dathdr, clobber=True)
            f = open(fake_combined_fn + '.dat', 'w')
            [f.write('%1.7f %1.2f %1.2f\n' % tuple(line)) for line in out_spexlike[0].T];
            f.close()

        redmodes.append(mode)
        if not obj in redobjs:
            redobjs.append(obj)



print "Now run IDL and combine 'red' and 'blue' spectra into one file."
print "   Read the in-function help by clicking 'Help' at bottom."
os.system('cd ' + _proc + '\n' + idlexec + ' -e xcombspec')

comb_fns = np.array([line.strip() for line in os.popen('ls %s*comb*fits' % (_proc))])
for fn in comb_fns:
    comb_dat = pyfits.getdata(fn)
    f = open(fn.replace('.fits', '.dat'), 'w')
    [f.write('%1.7f %1.2f %1.2f\n' % tuple(line)) for line in comb_dat[0].T];
    f.close()

print 'Instructions for IDL XTELLCOR:\n'
print "Select appropriate pairs of A0V/standard and target spectra"
print 'Units need to be set to "um", FWHM to 0.0015.'
print 'Look up B, V, V_rot in Vizier or in literature.'
print 'Make sure to get the velocity shift correction correctly.'
print 'At the end, make sure you write out both Telluric and A0V files.'
#sys.stdout.flush()
os.system('cd ' + _proc + '\n' + idlexec + ' -e xtellcor_general')

  

for comb_fn in comb_fns:
    if 'single_combspec' in comb_fn:
        tell_fns = np.array([line.strip() for line in os.popen('ls %s*A0V.dat' % (comb_fn.replace('_single_combspec.fits', '')))])
    else:
        tell_fns = np.array([line.strip() for line in os.popen('ls %s*A0V.dat' % (comb_fn.replace('_combined.fits', '')))])
        if len(tell_fns)==0:
            tell_fns = np.array([line.strip() for line in os.popen('ls %s*A0V.dat' % (comb_fn.replace('_combspec.fits', '')))])

    hdr = pyfits.getheader(comb_fn)
    comb_dat = pyfits.getdata(comb_fn)
    for tell_fn in tell_fns:
        corr_fn = tell_fn.replace('_A0V','')
        tell_dat = np.loadtxt(corr_fn)
        tell_dat[np.isnan(tell_dat)] = 1e-15
        pyfits.writeto(corr_fn.replace('.dat', '.fits'), tell_dat.T.reshape(1,3,1024), hdr, clobber=True)


os.system('cd ' + _proc + '\n' + idlexec + ' -e xmergexd')

telregs = [1.35, 1.45], [1.8, 2.]

py.ioff()
final_fns = np.array([line.strip() for line in os.popen('ls %s*merged.fits' % (_proc))])
for jj,fn in enumerate(final_fns):
    d = pyfits.getdata(fn)
    h = pyfits.getheader(fn)
    ydat = (d[0]*d[1])/np.median(d[0]*d[1])
    losnr = signal.medfilt(d[1]/d[2], 5)
    medsnr = np.median(losnr[np.isfinite(losnr)])
    py.figure(jj+1); ax1=py.subplot(111)
    py.plot(d[0], ydat, 'k')
    py.ylim(0, np.sort(ydat)[-100]*1.2)
    py.ylabel('$\lambda F_\lambda$', fontsize=18)
    py.figure(jj+1+len(final_fns)); ax2=py.subplot(111)
    py.plot(d[0], losnr, 'k')
    py.ylim(0, medsnr*3)
    py.ylabel('<S/N>', fontsize=18)
    [ax.minorticks_on() for ax in [ax1, ax2]]
    [ax.grid() for ax in [ax1, ax2]]
    [ax.set_xlim(.8, 2.55) for ax in [ax1, ax2]]
    [ax.set_xlabel('Wavelength [micron]') for ax in [ax1, ax2]]
    [ax.set_title('%s, <S/N> ~ %i' %(  h['OBJECT'] , medsnr)) for ax in [ax1, ax2]]
    [[tools.drawRectangle(treg[0], ax.get_ylim()[0], np.diff(treg), np.diff(ax.get_ylim()), color='gray', alpha=0.5, ax=ax) for treg in telregs] for ax in [ax1, ax2]]

rootfn = _proc + 'sofi_spectra_s2n_v' 
iter = 1
while os.path.isfile(rootfn + '%i.pdf' % iter):
    iter += 1

tools.printfigs(rootfn + '%i.pdf' % iter, range(len(final_fns)+1, len(final_fns)*2+1))

rootfn = _proc + 'sofi_spectra_flam_v' 
iter = 1
while os.path.isfile(rootfn + '%i.pdf' % iter):
    iter += 1

tools.printfigs(rootfn + '%i.pdf' % iter, range(0,len(final_fns)+1))
                      
py.close('all')
py.ion()

stop
        
for targ in targs:
    comb_fns = np.array([line.strip() for line in os.popen('ls %s*%s*combin*' % (_proc, targ))])
    if len(comb_fns)>0:
        dats = map(pyfits.getdata, comb_fns)
        py.figure()
        [py.plot(dat[0][0], dat[0][1]) for dat in dats]
        py.title(targ)
        py.xlabel('Wavelength [micron]')
        py.ylabel('Combined Flux')
        py.ylim(0, np.sort(np.concatenate([dat[0][1] for dat in dats]))[-100]*1.1)
        py.minorticks_on()
        py.grid()


for ii,obj in enumerate(redobjs):
    py.figure()
    py.subplot(111, position=[.15, .15, .75, .7])
    py.plot(wr/1e4, allspecs['GR'][ii].T)
    py.plot(wb/1e4, allspecs['GB'][ii].T)
    py.ylim(0, np.sort(np.concatenate((allspecs['GR'][ii].ravel(), allspecs['GB'][ii].ravel())))[-300]*1.1)
    py.title('%s\n%s' % (obj,os.path.split(datfns[-1])[0]) , fontsize=20)
    py.xlabel('Wavelength [micron]')
    py.ylabel('RAW flux')
    py.minorticks_on()
    py.grid()
            
for ii,obj in enumerate(redobjs):
    py.figure()
    py.subplot(111, position=[.15, .15, .75, .7])
    py.plot(wr/1e4, meanspecs['GR'][ii], 'r')
    py.plot(wb/1e4, meanspecs['GB'][ii], 'b')
    py.ylim(0, np.sort(np.concatenate((meanspecs['GR'][ii], meanspecs['GB'][ii])))[-100]*1.1)
    py.title('%s\n%s' % (obj,os.path.split(datfns[-1])[0]) , fontsize=20)
    py.xlabel('Wavelength [micron]')
    py.ylabel('RAW flux')
    py.minorticks_on()
    py.grid()
            


stop

data = tools.loadpickle('20150110_datasets.pickle')
data.pop(2)
w16, t16 = pyfits.getdata('/Users/ianc/proj/mdwarfs/data/proc/20150110/efosc_gr16_counts2flam.fits')
waves, specs, corspecs, especs, corespecs, objs, snrs = [], [], [], [], [], [], []
for ii in xrange(len(data)):
  ss = np.array([pyfits.getdata(sp.replace('.fits', '.0001.fits')) for sp in data[ii][0]])[:,0,0]
  master = np.median(ss, axis=0)
  scales = np.array([an.lsq(ss[jj], master)[0][0] for jj in range(ss.shape[0])])
  newss = ss * scales.reshape(ss.shape[0], 1)
  spec = np.median(newss, 0) * ss.shape[0]
  espec = newss.std(0) * np.sqrt(ss.shape[0])
  ww = ns.getdisp(_wldat + os.sep + 'id' + os.path.split(data[ii][1])[1].replace('.fits', ''), 'spline3')
  objs.append(pyfits.getval(data[ii][0][0].replace('.fits', '.0001.fits'), 'ESO OBS TARG NAME'))
  waves.append(ww)
  specs.append(spec)
  corspecs.append(spec / np.interp(ww, w16, t16))
  snrs.append(np.median(spec/espec))
  especs.append(espec)
  corespecs.append(espec / np.interp(ww, w16, t16))

for ii in range(17):
    figure()
    plot(waves[ii], corspecs[ii])
    title('%s, <S/N> ~ %i' % (objs[ii], snrs[ii]))
    minorticks_on()
    xlabel('Wavelength [Ang]')
    ylabel('Flux (e- ???)')

                                                  


stopstopstop
# Estimate goodness-of-fit

res=1000
maxsnr = 200
#_home = os.path.expanduser('~')
_proc0 = _home + '/proj/mdwarfs/data/proc/'
procs = [_proc0 + el +'/' for el in ['20150128', '20150129', '20150130', '20150131']]
procs = [_proc0 + el +'/' for el in ['20160324', '20160325', '20160326', '20160327']]
_spex = _home + '/proj/mdwarfs/templates/spex/'
spex_fns = [line.strip() for line in os.popen('ls %s*.fits' % _spex)]
sptstr = [os.path.split(el)[1].split('_')[0] for el in spex_fns]
spts = np.array([10*int(el[0]=='M')+int(el[1])-10 for el in sptstr])
spex_dats = []
for fn in spex_fns:
    temp = pyfits.getdata(fn)
    nir = (temp[0] > 0.9) * (temp[0] < 2.4)
    dw = np.diff(temp[0,nir]).mean()
    xkern = np.arange(-40, 40)
    kern = an.gaussian([1., 1./(res*dw), 0, 0], xkern)
    temp[1] = np.convolve(temp[1], kern, 'same')
    spex_dats.append(temp)

# Sort filenames:
final_fns = np.array(tools.flatten([[line.strip() for line in os.popen('ls %s*final.fits' % (proc))] for proc in procs]))
ai = np.argsort([os.path.split(fn)[1] for fn in final_fns])
final_fns = final_fns[ai]

w_ignore = [0, .91], [1.33, 1.45], [1.75, 2.04], [2.4, 999]
nobs = len(final_fns)
nmod = len(spex_fns)
chisq = np.zeros((nobs, nmod), dtype=float) + 9e99

for ii in range(nobs):
    obs0 = pyfits.getdata(final_fns[ii])
    obs0[2] = np.sqrt(obs0[2]**2 + (obs0[1]/maxsnr)**2)
    wbins = 0.5*(obs0[0,1:] + obs0[0,0:-1])
    wbins = np.concatenate((wbins[0]-np.diff(wbins[0:2]), wbins, wbins[-1]+np.diff(wbins[-2:])))
    bestfit, bestmod = 0, 0
    for jj in range(nmod):
        temp = spex_dats[jj].copy()
        #temp[1] *= temp[0] # DIfferent units!!!
        obs = obs0.copy()
        wmin = max(temp[0].min(), obs[0].min())
        wmax = min(temp[0].max(), obs[0].max())
        obs[2, obs[0]<wmin] = 9e99
        obs[2, obs[0]>wmax] = 9e99

        for wing in w_ignore:
          obs[2,(obs[0]>wing[0]) * (obs[0]<=wing[1])] = 9e99
        wtemp, temp2, junk, junk = tools.errxy(temp[0], temp[1], wbins, xerr=None, yerr=None)
        temp
        nanind = np.isnan(temp2) + np.isnan(obs[1]) + np.isnan(obs[2]) + (obs[2]==0)
        temp2[nanind] = np.median(temp2[True-nanind])
        obs[1, nanind] = np.median(obs[1,True-nanind])
        obs[2, nanind] = 9e99
        fit, efit = an.lsq((temp2,), obs[1], 1./obs[2]**2)
        chisq[ii,jj] = (((obs[1]-fit*temp2)/obs[2])**2).sum()
        if chisq[ii,jj] <= chisq[ii].min():
            bestfit, bestmod = fit.copy(), temp2.copy()
    min_ind = (chisq[ii]==chisq[ii].min()).nonzero()[0]

    py.figure()
    py.plot(obs0[0], obs0[0]*obs0[1], '-k', obs0[0], obs0[0]*bestfit*bestmod, '-r')
    py.ylim(0, np.sort(obs[0]*obs0[1])[-100]*1.1)
    py.title('%s ---- %s' % (os.path.split(final_fns[ii])[1], os.path.split(spex_fns[min_ind])[1]))
    py.xlabel('Wavelength [micron]')
    py.ylabel('$\lambda F_\lambda$', fontsize=18)
    py.minorticks_on()
    py.grid()
    py.xlim(.8, 2.55)
    [tools.drawRectangle(wing[0], py.ylim()[0], np.diff(wing), np.diff(py.ylim()), color='k', alpha=0.5) for wing in w_ignore]


badtypes = np.array([spts[el1==el2] for el1, el2 in zip(chisq, chisq.min(1))]).squeeze()
argind = np.argsort(badtypes)
ind2 = np.array([0,1,3,4,6,9] + range(10,14) + range(16,26) + range(27,35) + range(36,40) + [41,42,44])


dy = 0.25
py.figure()
wplots = ((1.45, 1.75), (2.04, 2.4))
ax1=py.subplot(111, position=[.1, .15, .4, .85])
ax2=py.subplot(111, position=[.55, .15, .4, .85])
axs = [ax1, ax2]
for iter, ii in enumerate(argind[ind2]):
    obs0 = pyfits.getdata(final_fns[ii])
    hind = (obs0[0]>1.5) * (obs0[0] < 1.7)
    
    for ax, wplot in zip(axs, wplots):
        pind = ((obs0[0]>wplot[0]) * (obs0[0] < wplot[1]))
        ax.plot(obs0[0,pind], 1+dy*(nobs-iter) + obs0[1,pind]/np.median(obs0[1,hind]), 'k')


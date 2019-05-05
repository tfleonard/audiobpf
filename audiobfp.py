
from scipy import signal
from scipy.io import wavfile
from scipy import fftpack
import time

import numpy as np
from matplotlib import pyplot as plt




def lpf(n, fc, fstp, fs):
    """
        fir low pass filter
        n = number of taps
        fc - cut off freqo
        fstp - stop band freq
        fs - sampling rate
        returns: b the coefficients of x
    """

    s = signal.remez(n, [0, fc, fstp, fs/2], [1., 0.], fs = fs)
    return s




def bpf(n, fstp1, fc1, fc2, fstp2, fs):
    """
        fir band pass filter
        n = number of taps
        fstp1 - first stop band freq
        fc1 - low freq of bpf
        fc2 - hi freq of bpf
        fstp2 - second stop band freq
        fs - sampling rate
        returns: (b,a) the coefficients of x and y.
    """

    s = signal.remez(n, [0, fstp1, fc1, fc2, fstp2, fs/2], [0, 1., 0.], fs = fs)
    return s




def iir_bpf(fstp1, fc1, fc2, fstp2, gpb, gsb, fs):
    """
        iir band pass filter
        fstp1 - first stop band freq, hz
        fc1 - low freq of bpf, hz
        fc2 - hi freq of bpf, hz
        fstp2 - second stop band freq, hz
        gpb - passband ripple, db
        gsb - stop band attenuation, db
        fs - sampling rate, hz
        returns: (b,a), the x and y coefficients of the diff eq
    """
    s = signal.iirdesign([fc1,fc2],[fstp1,fstp2],gpb, gsb, fs=44100. )
    return s



def iir_lpf(fc, fstp, gpb, gsb, fs):
    """
        iir low pass filter
        fstp - stop band freq, hz
        fc - pass band freq of bpf, hz
        gpb - passband ripple, db
        gsb - stop band attenuation, db
        fs - sampling rate, hz
        returns: (b,a), the x and y coefficients of the diff eq
    """
    s = signal.iirdesign(fc,fstp,gpb, gsb, fs=fs )
    return s



def iir_hpf(fc, fstp, gpb, gsb, fs):
    """
        iir low pass filter
        fstp - stop band freq, hz
        fc - pass band freq of bpf, hz
        gpb - passband ripple, db
        gsb - stop band attenuation, db
        fs - sampling rate, hz
        returns: (b,a), the x and y coefficients of the diff eq
    """
    s = signal.iirdesign(fc,fstp,gpb, gsb, fs=fs )
    return s




def filter(b, a, data):

    M = len(b)
    N = len(a)
    n = len(data)

    xcache = np.zeros(M)
    indx = 0

    ycache = np.zeros(N)
    indy = 0

    y = np.zeros(n)

    for j in range(0,n):
        ytemp = 0.
        xcache[indx] = data[j]
        ii = indx
        for i in range(0,M):
            ytemp = ytemp + xcache[ii]*b[i]
            if ii == 0:
                ii = M-1
            else:
                ii = ii-1

        ycache[indy] = 0.
        ii = indy
        for i in range(1,N):
            if ii == 0:
                ii = N-1
            else:
                ii = ii-1
            ytemp = ytemp - ycache[ii]*a[i]

        y[j] = ytemp
        ycache[indy] = ytemp
        indx = (indx + 1) % M
        indy = (indy + 1) % N

    return y


def fir_filter(b, data):

    M = len(b)
    n = len(data)

    xcache = np.zeros(M)
    indx = 0

    y = np.zeros(n)

    for j in range(0,n):
        ytemp = 0.
        xcache[indx] = data[j]
        ii = indx
        for i in range(0,M):
            ytemp = ytemp + xcache[ii]*b[i]
            if ii == 0:
                ii = M-1
            else:
                ii = ii-1

        y[j] = ytemp
        indx = (indx + 1) % M

    return y



# sample rate used with the bpf
fs = 44100.

"""
IIR BPF:
    transition bands 100-300, 800-1000hz
    passband 300-800hz
    passband ripple - 1 db
    stopband attenuation 40 db
    sampling frequency 44100 hz

"""
"""
IF Crystal Filter Specs
-6 db       -60db       Model       CF      ws1     wp1     wp2     ws2
-----       ------      ------      ----
250         687.5       Inrad 97    700hz   356.75  575     825     1043.75
270         1.2 khz     YK88CN      700     100     565     835     1300
500         1.8 khz     YK88C       700     100     450     950     1600
1.8 khz     3.6 khz     YK88SN              0       300     2400    3600
2.4 khz     4.4 knz     YK88S               0       300     2700    4400
6 khz       11 khz      YK88A               0       300     6300    11000
"""



# 2.1 khz bpf
#(b,a) = iir_bpf( 0., 300., 2400., 300., 0.3, 50., fs)
#hpf = iir_hpf(300., 100., 0.3, 50, fs)
bhpf, ahpf = iir_hpf(575., 350., 0.3, 50, fs)

print "hfp:  num b coeff: " + str(len(bhpf)) + "  num a coeff: " + str(len(ahpf)) + "\n"
print "b coeff: " + str(bhpf) + "\n"
print "a coeff: " + str(ahpf) + "\n"
whpf, hhpf = signal.freqz(bhpf,ahpf)
plt.title("IIR High Pass Filter")
plt.xlabel(" Freq(hz)")
plt.ylabel(" Mag (db)")
plt.plot(whpf * fs / (2. * np.pi), 20*np.log10(abs(hhpf)))
plt.show()

#lpf = iir_lpf(2400., 3600., 0.3, 50., fs)
blpf, alpf = iir_lpf(825., 1050., 0.3, 50., fs)

print "lpf: num b coeff: " + str(len(blpf)) + "  num a coeff: " + str(len(alpf)) + "\n"
print "b coeff: " + str(blpf) + "\n"
print "a coeff: " + str(alpf) + "\n"

wlpf, hlpf = signal.freqz(blpf,alpf)
plt.title("IIR Low Pass Filter")
plt.xlabel(" Freq(hz)")
plt.ylabel(" Mag (db)")
plt.plot(wlpf * fs / (2. * np.pi), 20*np.log10(abs(hlpf)))
plt.show()


wbpf = wlpf
hbpf = hlpf * hhpf
plt.title("IIR Band Pass Filter")
plt.xlabel(" Freq(hz)")
plt.ylabel(" Mag (db)")
plt.plot(wbpf * fs / (2. * np.pi), 20*np.log10(abs(hbpf)))
plt.show()


# sampled impulse response of the bpf
hw = fftpack.ifft(hbpf)






#(fs, data) = wavfile.read('ts590_ft8_20m.wav')
(fs, data) = wavfile.read('multicw.wav')
#normalize the data
x = data/float(0x7fff)
X = fftpack.fft(x[0:2048])
N = 2048
fdelt = float(fs) / float(N)
xdata = np.arange(N/2) * fdelt

plt.title("Raw audio signal FFT")
plt.xlabel(" Freq(hz)")
plt.ylabel(" Mag (db)")
plt.plot( xdata, 20*np.log10(abs(X[0:N/2])))
plt.show()

startTime = time.clock()
b1,a1 = lpf
yfilt1 = signal.lfilter(b1,a1,x)
b2,a2 = hpf
y = signal.lfilter(b2,a2,yfilt1)
endTime = time.clock()

fileTime = len(data) / fs

print "file len(sec): " + str(fileTime) + "  filter exec(sec): " + str(endTime-startTime) + "\n"

idx = 0

while idx < len(y):
    X = fftpack.fft(x[idx:idx+N])
    YFILT1 = fftpack.fft(yfilt1[idx:idx+N])
    Y = fftpack.fft(y[idx:idx+N])

    plt.title("Filtered audio signal FFT")
    plt.xlabel(" Freq(hz)")
    plt.ylabel(" Mag (db)")
    plt.plot(xdata, 20*np.log10(abs(X[0:N/2])), 'r', label='X')
    plt.plot(xdata, 20*np.log10(abs(YFILT1[0:N/2])), 'b', label = 'YFILT1')
    plt.plot(xdata, 20*np.log10(abs(Y[0:N/2])), 'g', label = 'Y')
    plt.legend()
    plt.show()
    idx = idx + N


yy = y * float(0x7fff)
newyy = np.int16(yy)
#wavfile.write('ts590_ft8_20m_filt.wav', fs, newyy)
wavfile.write('multicw_filt.wav',fs,newyy)




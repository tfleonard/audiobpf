import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack


def signal (a, f,fs,t):
    """
    generate a single signal of frequence f of length t seconds sampled at n samples per second
    :param a: amplitude
    :param f: signal frequency in hz
    :param fs: sample rate in samples per second
    :param t: signal duration
    :return:
    """
    n = int(fs*t)
    s = np.zeros(n)
    dt = np.arange(0, t, 1.0/fs)
    ss = np.cos(2*np.pi * f * dt)

    return a*ss


def sig(amps, freqs, fs, t):
    """
    :param: amps: list of amplitudes corresponding to frequencies in freqs
    :param freqs: list of frequencies to generate
    :param fs: sample rate
    :param t: signal duration
    :return: signal
    """
    n = int(fs*t)
    s = np.zeros(n)
    for f, a in zip(freqs,amps):
        s += signal(a, f,fs,t)

#    norm = max(s)
#    return s / norm
    return s

def comb(fstart, fend, fstep, fs, t):

    freqs = [f for f in np.arange(fstart,fend,fstep)]
    amps = [1 for f in np.arrange(fstart,fend,fstep)]
    return sig(amps,freqs,fs,t)


if __name__ == '__main__':
    fs = 8000.
#    f1 = 100.
    f1 = 700
    f2 = 1000.
    ft = 100.
    ts = 1.0
#    ss = sig((f1,f2), fs, ts)
    ss = signal(1, f1,fs, ts)
#    ss = comb(f1, f2, ft, fs, ts)
    plt.title("comb generator test")
    plt.xlabel(" Sample")
    plt.ylabel(" Mag ")
    xdata = np.arange(0,fs*ts)
    plt.plot(xdata, ss, 'r', label='ss')
#    plt.plot(sampdata, yfilt1[0:N], 'b', label='yfilt1')
#    plt.plot(sampdata, y[0:N], 'g', label='y')
    plt.legend()
    plt.show()

    n = 1024
    plt.title("Comb sig gen spectrum")
    plt.xlabel(" Freq(hz)")
    plt.ylabel(" Mag (db)")
    X = fftpack.fft(ss[0:n])
    fr = np.zeros(n/2+1)
    for i in range(0, len(fr)):
        fr[i] = (fs/n) * i
    plt.plot(fr, 20 * np.log10(abs(X[0:n/2+1])))
    plt.show()


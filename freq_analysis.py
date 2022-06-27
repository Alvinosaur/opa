import matplotlib.pyplot as plt
import numpy as np
import warnings


def fftPlot(sig, dt=None, ts=None, plot=True, title='Analytic FFT plot'):
    """
    Taken from https://stackoverflow.com/a/53925342
    """
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if ts is None:
        if dt is None:
            dt = 1
            ts = np.arange(0, sig.shape[-1])
            xLabel = 'samples'
        else:
            ts = np.arange(0, sig.shape[-1]) * dt
            xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        ts = ts[0:-1]
        sig = sig[0:-1]

    # Divided by size t for coherent magnitude
    sigFFT = np.fft.fft(sig) / ts.shape[0]

    freq = np.fft.fftfreq(ts.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    # *2 because of magnitude of analytic signal
    sigFFTPos = 2 * sigFFT[0:firstNegInd]

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title(title)
        plt.show()

    return sigFFTPos, freqAxisPos


if __name__ == "__main__":
    dt = 1 / 1000

    # Build a signal within Nyquist - the result will be the positive FFT with actual magnitude
    f0 = 200  # [Hz]
    t = np.arange(0, 1 + dt, dt)
    sig = 1 * np.sin(2 * np.pi * f0 * t) + \
        10 * np.sin(2 * np.pi * f0 / 2 * t) + \
        3 * np.sin(2 * np.pi * f0 / 4 * t) +\
        7.5 * np.sin(2 * np.pi * f0 / 5 * t)

    # Plot the signal
    plt.figure()
    plt.plot(t, sig)
    plt.xlabel('time [s]')
    plt.ylabel('sig')
    plt.title('Signal')
    plt.show()

    # Result in frequencies
    fftPlot(sig, dt=dt)
    # Result in samples (if the frequencies axis is unknown)
    fftPlot(sig)

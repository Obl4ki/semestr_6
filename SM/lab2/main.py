import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.fft import fft


def write_audio_to_file(fname: str, data: np.ndarray, sample_rate: int):
    sf.write(fname, data, sample_rate)


def zad1():
    data, sample_rate = sf.read('sound1.wav', dtype='float32')
    print(data.dtype)
    print(data.shape)

    # sd.play(data, sample_rate)
    # status = sd.wait()
    left_channel, right_channel = data[:, 0], data[:, 1]

    x_seconds = np.arange(0, len(left_channel)/sample_rate, 1/sample_rate)
    # alternatywnie
    # x_seconds = np.linspace(0, len(left_channel) / sample_rate, len(left_channel))
    # plt.subplot(2, 1, 1)
    # plt.plot(x_seconds, left_channel)

    # plt.subplot(2, 1, 2)
    # plt.plot(x_seconds, left_channel)
    # plt.show()
    only_left_channel = np.vstack(
        (left_channel, np.zeros_like(left_channel))).T
    only_right_channel = np.vstack(
        (right_channel, np.zeros_like(right_channel))
    )

    mixed_channels = (left_channel + right_channel) / 2
    mixed_channels = np.vstack(
        (mixed_channels, mixed_channels)).T

    write_audio_to_file('sound_L.wav', only_left_channel, sample_rate)
    write_audio_to_file('sound_R.wav', right_channel, sample_rate)
    write_audio_to_file('sound_mix.wav', mixed_channels, sample_rate)


def zad2():
    data, fs = sf.read('sin_440Hz.wav', dtype=np.int32)

    fsize = 2**12

    plot_audio(data, fs)


def plot_audio(signal: np.ndarray, fs: int, time_margin: list = [0, 0.02]):
    fsize = 2**12

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, signal.shape[0])/fs, signal)
    plt.xlim(time_margin)

    plt.subplot(2, 1, 2)
    yf = fft(signal, fsize)
    plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))

    plt.show()


if __name__ == '__main__':
    # zadania = [zad1, zad2]
    zadania = [zad2]
    [zadanie() for zadanie in zadania]

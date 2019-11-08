### EE524 Pattern Recognition 19 Fall
### PA1 Counting the Total Number of Syllables in a Word
### 20194314 Kim Dayeon
import os, librosa, scipy, shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
# from numpy import *
# from librosa import *

# feature: length, pitch
def main():
    ## train
    DATA_PATH = "./PA1_DB/"
    if os.path.isdir("./img/"):
        shutil.rmtree("./img/")
    os.mkdir("./img/")
    # f_train = open(DATA_PATH+"train.txt", 'r')

    ## train
    f_train = open(DATA_PATH+"train2.txt", 'r')
    lines_train = f_train.readlines()
    traindata = []
    for line_train in lines_train:
        line_train = line_train.rstrip('\n')
        print(line_train)

        # if line_train == "train/5syllables/99_mu-yeok-dae-pyo-bu/0024_099.wav": # last file
        (file_dir, file_id) = os.path.split(line_train)
        syllable = int(file_dir.split('/')[1][0])
        print("file_dir: ", file_dir)
        print("file_id: ", file_id)
        print("label: ", syllable)


        # open(DATA_PATH+line_train, 'r')
        y, sr = librosa.load(DATA_PATH+line_train, sr=16000)
        # sr, audio = wavfile.read(file_path)
        fs, x = wavfile.read(DATA_PATH+line_train)
        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)

        # Find the short time zero crossing rate.
        zc = stzcr(x, scipy.signal.get_window("boxcar", 201))

        # Find the short time energy.
        e = ste(x, scipy.signal.get_window("hamming", 201))


        print(zc)
        print(e)
        # print("length: ", feature1(fs, x))
        # print(y.size)
        # feature1(DATA_PATH+line_train) / y.size : constant -> 둘 중 아무거나 선택
        # zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        # print("zero: ", zero_crossing_rate)


        time = np.linspace(0, len(y)/sr, len(y)) # time axis
        fig, ax1 = plt.subplots() # plot
        ax1.plot(time, y, color = 'b', label='speech waveform')
        ax1.set_ylabel("Amplitude") # y 축
        ax1.set_xlabel("Time [s]") # x 축
        plt.title("["+str(syllable)+"] "+file_id) # 제목

        plt.savefig("./img/"+file_id+'.png')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        # plt.close('all')

    # print(lines)
    f_train.close()

def feature1(fs, x):
    # sr, audio = wavfile.read(file_path)
    length = x.shape[0]/float(fs)
    # getnframes
    return length


def _sgn(x):
  y = np.zeros_like(x)
  y[np.where(x >= 0)] = 1.0
  y[np.where(x < 0)] = -1.0
  return y


def stzcr(x, win):
  """Compute short-time zero crossing rate."""
  if isinstance(win, str):
    win = scipy.signal.get_window(win, max(1, len(x) // 8))
  win = 0.5 * win / len(win)
  x1 = np.roll(x, 1)
  x1[0] = 0.0
  abs_diff = np.abs(_sgn(x) - _sgn(x1))
  return scipy.signal.convolve(abs_diff, win, mode="same")


def ste(x, win):
  """Compute short-time energy."""
  if isinstance(win, str):
    win = scipy.signal.get_window(win, max(1, len(x) // 8))
  win = win / len(win)
  return scipy.signal.convolve(x**2, win**2, mode="same")


if __name__ == '__main__':
    main()
    print("3")

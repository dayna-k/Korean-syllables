### EE524 Pattern Recognition 19 Fall
### PA1 Counting the Total Number of Syllables in a Word
### 20194314 Kim Dayeon
import os, librosa, scipy, shutil, math, statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
# from scipy.stats import norm

# from numpy import *
# from librosa import *

# feature: length, pitch
def feature_extraction():
    ## train
    DATA_PATH = "./PA1_DB/"
    if os.path.isdir("./img/"):
        shutil.rmtree("./img/")
    os.mkdir("./img/")
    # f_train = open(DATA_PATH+"train.txt", 'r')

    ## train
    f_train = open(DATA_PATH+"train.txt", 'r')
    lines_train = f_train.readlines()
    traindata = []
    duration_sum = [0, 0, 0]
    count_3 = 0 # 100
    count_4 = 0 # 100
    count_5 = 0 # 100
    wav_all = [[], [], []]
    duration_all = [[], [], []]
    zc_all = [[], [], []]
    e_all = [[], [], []]
    mfcc_all = [[], [], []]

    length_sum = [0, 0, 0]
    length_all = [[], [], []]

    #### train / learning
    for line_train in lines_train:
        line_train = line_train.rstrip('\n')
        print(line_train)

        # if line_train == "train/5syllables/99_mu-yeok-dae-pyo-bu/0024_099.wav": # last file
        (file_dir, file_id) = os.path.split(line_train)
        #print("file_dir: ", file_dir)
        #print("file_id: ", file_id)
        syllable = label_parse(file_dir)
        #print("label: ", syllable)


        y, sr = librosa.load(DATA_PATH+line_train, sr=16000)
        fs, x = wavfile.read(DATA_PATH+line_train)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)

        l = audio_length(fs, x)

        # Find the short time zero crossing rate.
        zc = stzcr(x, scipy.signal.get_window("boxcar", 500))
        # Find the short time energy.
        e = ste(x, scipy.signal.get_window("hamming", 500))
        # mfcc.shape : ndarray 반환 으로 사용
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho')


        # feature - length, zc, e, mfcc
        if int(syllable) == 3:
            count_3 += 1
            duration_sum[0] += y.size
            duration_all[0].append(y.size)
            zc_all[0].append(y)
            zc_all[0].append(zc)
            e_all[0].append(e)
            mfcc_all[0].append(mfcc)

            length_sum[0] += l
            length_all[0].append(l)


        if int(syllable) == 4:
            count_4 += 1
            duration_sum[1] += y.size
            duration_all[1].append(y.size)
            zc_all[1].append(y)
            zc_all[1].append(zc)
            e_all[1].append(e)
            mfcc_all[1].append(mfcc)

            length_sum[1] += l
            length_all[1].append(l)

        if int(syllable) == 5:
            count_5 += 1
            duration_sum[2] += y.size
            duration_all[2].append(y.size)
            zc_all[2].append(y)
            zc_all[2].append(zc)
            e_all[2].append(e)
            mfcc_all[2].append(mfcc)

            length_sum[2] += l
            length_all[2].append(l)

        # print("length: ", audio_length(fs, x))
        #print("zero crossing rate: ", zc.size, "\n", zc)
        #print("short-time energy: ", e.size, "\n", e)
        #print("length: ", audio_length(fs, x))
        #print("length: ", y.size, "\n")


        # print(y.size)
        # audio_length(DATA_PATH+line_train) / y.size : constant -> 둘 중 아무거나 선택


        show_graph(file_dir, file_id, syllable, sr, y, zc, e)

    # print(lines)
    f_train.close()
    #duration_avg = [duration_sum[0]/count_3, duration_sum[1]/count_4, duration_sum[2]/count_5]
    duration_avg = [np.mean(duration_all[0]), np.mean(duration_all[1]), np.mean(duration_all[2])]
    duration_var = [np.var(duration_all[0]), np.var(duration_all[1]), np.var(duration_all[2])] # 분산

    #length_avg = [length_sum[0]/count_3, length_sum[1]/count_4, length_sum[2]/count_5]
    length_avg = [np.mean(length_all[0]), np.mean(length_all[1]), np.mean(length_all[2])]
    length_var = [np.var(length_all[0]), np.var(length_all[1]), np.var(length_all[2])]

    # normalizing(x, mean, var)

    print("duration_avg: ", duration_avg)
    print("duration_var: ", duration_var)
    print("duration_3: ", min(duration_all[0]), max(duration_all[0]))
    print("duration_4: ", min(duration_all[1]), max(duration_all[1]))
    print("duration_5: ", min(duration_all[2]), max(duration_all[2]), "\n")

    print("length_avg: ", length_avg)
    print("length_var: ", length_var)
    print("length_3: ", min(length_all[0]), max(length_all[0]))
    print("length_4: ", min(length_all[1]), max(length_all[1]))
    print("length_5: ", min(length_all[2]), max(length_all[2]), "\n\n")

    #np.histogram(length_all, bins=10, range=None, normed=None, weights=None, density=None)
    plt.hist(duration_all[0], bins=20)
    sns.distplot(duration_all[0], rug=True, kde=False, fit=sp.stats.norm)
    plt.show()

    return duration_avg, duration_var, length_avg, length_var

    #### test


def classify(duration_avg, duration_var, length_avg, length_var):
    DATA_PATH = "./PA1_DB/"
    f_test = open(DATA_PATH+"test.txt", 'r')
    lines_test = f_test.readlines()
    score = 0
    score2 = 0
    total = 0

    for line_test in lines_test:
        line_test = line_test.rstrip('\n')
        print(line_test)

        (file_dir, file_id) = os.path.split(line_test)
        print("file_dir: ", file_dir)
        print("file_id: ", file_id)
        syllable = label_parse(file_dir)
        print("label: ", syllable)

        y, sr = librosa.load(DATA_PATH+line_test, sr=16000)
        fs, x = wavfile.read(DATA_PATH+line_test)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)
        # Find the short time zero crossing rate.
        zc = stzcr(x, scipy.signal.get_window("boxcar", 500))
        # Find the short time energy.
        e = ste(x, scipy.signal.get_window("hamming", 500))
        l = audio_length(fs, x)

        # length_dif = [abs(duration_avg[0]-y.size), abs(duration_avg[1]-y.size), abs(duration_avg[2]-y.size)]
        # index = length_dif.index(min(length_dif))

        p3 = normalizing(y.size, duration_avg[0], duration_var[0])
        p4 = normalizing(y.size, duration_avg[1], duration_var[1])
        p5 = normalizing(y.size, duration_avg[2], duration_var[2])

        p_length = [p3, p4, p5]
        index = p_length.index(max(p_length))

        if(index+3 == int(syllable)):
            score += 1

        p3_2 = normalizing(l, length_avg[0], length_var[0])
        p4_2 = normalizing(l, length_avg[1], length_var[1])
        p5_2 = normalizing(l, length_avg[2], length_var[2])

        p_length2 = [p3_2, p4_2, p5_2]
        index2 = p_length2.index(max(p_length2))

        if(index2+3 == int(syllable)):
            score2 += 1

        total += 1

        print("y.size: ", y.size, ", p3: ", p3, "p4: ", p4, "p5: ", p5)
        print("l: ", l, ", p3: ", p3_2, "p4: ", p4_2, "p5: ", p5_2)
        print("index: ", index+3, index2+3, "\n")

    print("Accuracy: ", score*100/total, "%, ", score2*100/total, "%")

def audio_length(fs, x):
    # sr, audio = wavfile.read(file_path)
    length = x.shape[0]/float(fs)
    # getnframes
    return length


def normalizing(x, mean, var):
    return (1/math.sqrt(2*math.pi*var))*math.exp(-((x-mean)**2)/(2*var))


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


def label_parse(file_dir):
    """Parse the total number of syllables from file directory."""
    syllable = int(file_dir.split('/')[1][0])
    return syllable


def show_graph(file_dir, file_id, syllable, sr, y, zc, e):
    time = np.linspace(0, len(y)/sr, len(y)) # time axi
    fig = plt.figure(figsize=(18,5))
    # fig, ax1 = plt.subplots() # plot
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(time, y, color = 'b', label='waveform')
    ax1.set_ylabel("Amplitude") # y 축
    ax1.set_xlabel("Time [s]") # x 축
    plt.title("["+str(syllable)+"] "+file_id) # 제목

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(time, zc, color = 'r', label='zero-crossing')
    ax2.set_ylabel("Zero-crossing") # y 축
    ax2.set_xlabel("Time [s]") # x 축

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(time, e, color = 'g', label='energy')
    ax3.set_ylabel("Energy") # y 축
    ax3.set_xlabel("Time [s]") # x 축

    plt.savefig("./img/"+file_dir.split('/')[1]+"_"+file_dir.split('/')[2]+"_"+file_id+'.png')
    #plt.show(block=False)
    #plt.pause(0.1)
    plt.close()



if __name__ == '__main__':
    duration_avg, duration_var, length_avg, length_var = feature_extraction()
    classify(duration_avg, duration_var, length_avg, length_var)
    print("3")

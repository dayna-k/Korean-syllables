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

    # duration_sum = [0, 0, 0]
    # length_sum = [0, 0, 0]
    count_3 = 0 # 100 samples
    count_4 = 0 # 100 samples
    count_5 = 0 # 100 samples
    wave_all = [[], [], []]
    duration_all = [[], [], []]
    duration_parse_all = [[], [], []]
    zc_all = [[], [], []]
    e_all = [[], [], []]
    mfcc_all = [[], [], []]
    length_all = [[], [], []]
    onset_r_all = [[], [], []]
    onset_f_all = [[], [], []]
    onset_all = [[], [], []]

    #### train / learning
    for line_train in lines_train:
        line_train = line_train.rstrip('\n')
        print("Training: ", line_train)

        # if line_train == "train/5syllables/99_mu-yeok-dae-pyo-bu/0024_099.wav": # last file
        (file_dir, file_id) = os.path.split(line_train)
        syllable = label_parse(file_dir)

        y, sr = librosa.load(DATA_PATH+line_train, sr=16000) # sr = 16000
        fs, x = wavfile.read(DATA_PATH+line_train) # fs = 16000
        l = audio_length(fs, x)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)
        #print(y, sr, fs, x)
        new_y, new_x = paddding(y, x)

        ## feature extraction from the test data
        # Find the short time zero crossing rate.
        zc = stzcr(new_x, scipy.signal.get_window("boxcar", 2000))
        # Find the short time energy.
        e = ste(new_x, scipy.signal.get_window("hamming", 2000))
        # mfcc.shape : ndarray 반환 으로 사용
        mfcc = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho')

        parsed_length = speak_length(new_x)
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,backtrack=False)
        # print("onset_raw: ", onset_raw, onset_raw.size)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        # print("onset_frame: ", onset_frames, onset_frames.size)

        # feature - length, zc, e, mfcc
        if int(syllable) == 3:
            count_3 += 1
            duration_all[0].append(y.size)
            duration_parse_all[0].append(parsed_length)
            wave_all[0].append(new_y)
            zc_all[0].append(zc)
            e_all[0].append(e)
            mfcc_all[0].append(mfcc)
            length_all[0].append(l)
            onset_r_all[0].append(onset_raw.size)
            onset_f_all[0].append(onset_frames.size)
            onset_all[0].append(onset_raw.size + onset_frames.size)


        if int(syllable) == 4:
            count_4 += 1
            duration_all[1].append(y.size)
            duration_parse_all[1].append(parsed_length)
            wave_all[1].append(new_y)
            zc_all[1].append(zc)
            e_all[1].append(e)
            mfcc_all[1].append(mfcc)
            length_all[1].append(l)
            onset_r_all[1].append(onset_raw.size)
            onset_f_all[1].append(onset_frames.size)
            onset_all[1].append(onset_raw.size + onset_frames.size)

        if int(syllable) == 5:
            count_5 += 1
            duration_all[2].append(y.size)
            duration_parse_all[2].append(parsed_length)
            wave_all[2].append(new_y)
            zc_all[2].append(zc)
            e_all[2].append(e)
            mfcc_all[2].append(mfcc)
            length_all[2].append(l)
            onset_r_all[2].append(onset_raw.size)
            onset_f_all[2].append(onset_frames.size)
            onset_all[2].append(onset_raw.size + onset_frames.size)

        # print("length: ", audio_length(fs, x))
        # print("zero crossing rate: ", zc.size, "\n", zc)
        # print("short-time energy: ", e.size, "\n", e)
        # print("length: ", audio_length(fs, x), y.size, "\n") # audio_length(fs, x) / y.size : constant -> 둘 중 아무거나 선택
        show_graph(file_dir, file_id, syllable, sr, new_y, zc, e)

    # print(lines)
    f_train.close()
    duration_avg = [np.mean(duration_all[0]), np.mean(duration_all[1]), np.mean(duration_all[2])]
    duration_var = [np.var(duration_all[0]), np.var(duration_all[1]), np.var(duration_all[2])] # 분산
    duration_parse_avg = [np.mean(duration_parse_all[0]), np.mean(duration_parse_all[1]), np.mean(duration_parse_all[2])]
    duration_parse_var = [np.var(duration_parse_all[0]), np.var(duration_parse_all[1]), np.var(duration_parse_all[2])] # 분산
    # length_avg = [np.mean(length_all[0]), np.mean(length_all[1]), np.mean(length_all[2])]
    # length_var = [np.var(length_all[0]), np.var(length_all[1]), np.var(length_all[2])]
    onset_r_avg = [np.mean(onset_r_all[0]), np.mean(onset_r_all[1]), np.mean(onset_r_all[2])]
    onset_r_var = [np.var(onset_r_all[0]), np.var(onset_r_all[1]), np.var(onset_r_all[2])]
    onset_f_avg = [np.mean(onset_f_all[0]), np.mean(onset_f_all[1]), np.mean(onset_f_all[2])]
    onset_f_var = [np.var(onset_f_all[0]), np.var(onset_f_all[1]), np.var(onset_f_all[2])]
    onset_avg = [np.mean(onset_all[0]), np.mean(onset_all[1]), np.mean(onset_all[2])]
    onset_var = [np.var(onset_all[0]), np.var(onset_all[1]), np.var(onset_all[2])]


    print("duration_avg: ", duration_avg)
    print("duration_var: ", duration_var)
    print("duration_3: ", min(duration_all[0]), max(duration_all[0]))
    print("duration_4: ", min(duration_all[1]), max(duration_all[1]))
    print("duration_5: ", min(duration_all[2]), max(duration_all[2]), "\n")

    print("duration_parse_avg: ", duration_parse_avg)
    print("duration_parse_var: ", duration_parse_var)
    print("duration_parse_3: ", min(duration_parse_all[0]), max(duration_parse_all[0]))
    print("duration_parse_4: ", min(duration_parse_all[1]), max(duration_parse_all[1]))
    print("duration_parse_5: ", min(duration_parse_all[2]), max(duration_parse_all[2]), "\n\n")

    print("onset_r_avg: ", onset_r_avg)
    print("onset_r_var: ", onset_r_var)
    print("onset_r_3: ", min(onset_r_all[0]), max(onset_r_all[0]))
    print("onset_r_4: ", min(onset_r_all[1]), max(onset_r_all[1]))
    print("onset_r_5: ", min(onset_r_all[2]), max(onset_r_all[2]), "\n")

    print("onset_f_avg: ", onset_f_avg)
    print("onset_f_var: ", onset_f_var)
    print("onset_f_3: ", min(onset_f_all[0]), max(onset_f_all[0]))
    print("onset_f_4: ", min(onset_f_all[1]), max(onset_f_all[1]))
    print("onset_f_5: ", min(onset_f_all[2]), max(onset_f_all[2]), "\n")

    print("onset_avg: ", onset_avg)
    print("onset_var: ", onset_var)
    print("onset_3: ", min(onset_all[0]), max(onset_all[0]))
    print("onset_4: ", min(onset_all[1]), max(onset_all[1]))
    print("onset_5: ", min(onset_all[2]), max(onset_all[2]), "\n")
    # print("length_avg: ", length_avg)
    # print("length_var: ", length_var)
    # print("length_3: ", min(length_all[0]), max(length_all[0]))
    # print("length_4: ", min(length_all[1]), max(length_all[1]))
    # print("length_5: ", min(length_all[2]), max(length_all[2]), "\n\n")

    # show the distribution histogram of duration
    # plt.hist(duration_all[0], bins=20)
    # sns.distplot(duration_all[0], rug=True, kde=False, fit=sp.stats.norm)
    # plt.show()

    return duration_avg, duration_var, duration_parse_avg, duration_parse_var, onset_r_avg, onset_r_var, onset_f_avg, onset_f_var, onset_avg, onset_var


def classify(duration_avg, duration_var, duration_parse_avg, duration_parse_var, onset_r_avg, onset_r_var, onset_f_avg, onset_f_var, onset_avg, onset_var):
    DATA_PATH = "./PA1_DB/"
    f_test = open(DATA_PATH+"test.txt", 'r')
    lines_test = f_test.readlines()
    score1, score2, score3, score4, score5 = 0, 0, 0, 0, 0
    score12, score14, score15 = 0, 0, 0
    total = 0 # 75 samples
    correct1, correct2, correct3, correct4, correct5 = "", "", "", "", ""
    correct12, correct14, correct15 = "", "", ""


    for line_test in lines_test:
        line_test = line_test.rstrip('\n')
        print("Testing: ", line_test)

        (file_dir, file_id) = os.path.split(line_test)
        # print("file_dir: ", file_dir)
        # print("file_id: ", file_id)
        syllable = label_parse(file_dir)

        y, sr = librosa.load(DATA_PATH+line_test, sr=16000) # sr = 16000
        fs, x = wavfile.read(DATA_PATH+line_test) # fs = 16000
        l = audio_length(fs, x)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)
        new_y, new_x = paddding(y, x)

        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,backtrack=False)
        # print("onset_raw: ", onset_raw, onset_raw.size)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        # print("onset_frame: ", onset_frames, onset_frames.size)

        ## feature extraction from the test data
        # Find the short time zero crossing rate.
        zc = stzcr(x, scipy.signal.get_window("boxcar", 2000))
        # Find the short time energy.
        e = ste(x, scipy.signal.get_window("hamming", 2000))
        # mfcc.shape : ndarray 반환 으로 사용
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho')
        parsed_length = speak_length(new_x)

        # length_dif = [abs(duration_avg[0]-y.size), abs(duration_avg[1]-y.size), abs(duration_avg[2]-y.size)]
        # index = length_dif.index(min(length_dif))
        p3 = normalizing(y.size, duration_avg[0], duration_var[0])
        p4 = normalizing(y.size, duration_avg[1], duration_var[1])
        p5 = normalizing(y.size, duration_avg[2], duration_var[2])
        a, b, c = prob_norm(p3, p4, p5)
        p_length = [a, b, c] # probability
        index1 = p_length.index(max(p_length))
        print("p1: ", index1, max(p_length), p_length)
        if(index1+3 == int(syllable)):
            score1 += 1
            correct1 = "correct"
        else: correct1 = " wrong "

        p3_2 = normalizing(parsed_length, duration_parse_avg[0], duration_parse_var[0])
        p4_2 = normalizing(parsed_length, duration_parse_avg[1], duration_parse_var[1])
        p5_2 = normalizing(parsed_length, duration_parse_avg[2], duration_parse_var[2])
        a, b, c = prob_norm(p3_2, p4_2, p5_2)
        p_length2 = [a, b, c]
        index2 = p_length2.index(max(p_length2))
        print("p2: ", index2, max(p_length2), p_length2)
        if(index2+3 == int(syllable)):
            score2 += 1
            correct2 = "correct"
        else: correct2 = " wrong "

        p3_3 = normalizing(onset_raw.size, onset_r_avg[0], onset_r_var[0])
        p4_3 = normalizing(onset_raw.size, onset_r_avg[1], onset_r_var[1])
        p5_3 = normalizing(onset_raw.size, onset_r_avg[2], onset_r_var[2])
        a, b, c = prob_norm(p3_3, p4_3, p5_3)
        p_3 = [a, b, c] # probability
        index3 = p_3.index(max(p_3))
        print("p3: ", index3, max(p_3), p_3)
        if(index3+3 == int(syllable)):
            score3 += 1
            correct3 = "correct"
        else: correct3 = " wrong "

        p3_4 = normalizing(onset_frames.size, onset_f_avg[0], onset_f_var[0])
        p4_4 = normalizing(onset_frames.size, onset_f_avg[1], onset_f_var[1])
        p5_4 = normalizing(onset_frames.size, onset_f_avg[2], onset_f_var[2])
        a, b, c = prob_norm(p3_4, p4_4, p5_4)
        p_4 = [a, b, c] # probability
        #print("p4: ", p_4)
        index4 = p_4.index(max(p_4))
        if(index4+3 == int(syllable)):
            score4 += 1
            correct4 = "correct"
        else: correct4 = " wrong "

        p3_5 = normalizing(onset_raw.size + onset_frames.size, onset_avg[0], onset_var[0])
        p4_5 = normalizing(onset_raw.size + onset_frames.size, onset_avg[1], onset_var[1])
        p5_5 = normalizing(onset_raw.size + onset_frames.size, onset_avg[2], onset_var[2])
        a, b, c = prob_norm(p3_5, p4_5, p5_5)
        p_5 = [a, b, c] # probability
        #print("p5: ", p_5)
        index5 = p_5.index(max(p_5))
        if(index5+3 == int(syllable)):
            score5 += 1
            correct5 = "correct"
        else: correct5 = " wrong "


        # final control
        # if index1 == index2:
        #     index12 = index1
        #     if index3 == index5 and index4 == index5:
        #         index12 = (index1+index2+index3+index4)//4
        # else:
        #     index12 = (index1+index2+index3)//3
        #
        # # set outliers
        # if ((len(list(onset_raw)) == 1 or len(list(onset_frames)) == 2) or len(list(onset_raw)) + len(list(onset_frames)) <= 4):
        #     index12 = 0
        #
        # if (len(list(onset_frames)) >= 11 or len(list(onset_raw)) + len(list(onset_frames)) >= 16):
        #     index12 = 2

        p_final = [p_length2[0]*p_3[0], p_length2[1]*p_3[1], p_length2[2]*p_3[2]]
        index12 = p_final.index(max(p_final))



        # if (index1 < index3):
        #     index12 = (index1+index3)//2
        # if (index1 != index3):
        #     if max(max(p_length2), max(p_3)) > 0.8:
        #         if max(p_length2) > max(p_3):
        #             index12 = index2
        #         else:
        #             index12 = index1

        if ((len(list(onset_raw)) == 1 or len(list(onset_frames)) == 2) or len(list(onset_raw)) + len(list(onset_frames)) <= 4):
            index12 = 0

        if (len(list(onset_frames)) >= 11 or len(list(onset_raw)) + len(list(onset_frames)) >= 16):
            index12 = 2
        # if (index2 != index3)

        if(index12+3 == int(syllable)):
            score12 += 1
            correct12 = "correct"
        else: correct12 = " wrong "
        # if index1 == index2:
        #     index12 = index1
        # if index3 == index4:
        #     index34 = index2
        #
        # if index12 != 0 :
        #     index_a = index12
        # elif index34 != 0:


        # p_sum = [p3*p3_2, p4*p4_2, p5*p5_2]
        # index3 = p_sum.index(max(p_sum))
        # if(index3+3 == int(syllable)):
        #     score3 += 1
        #     correct3 = "correct"
        # else: correct3 = "wrong"

        total += 1
        print("label:", syllable, "| index1:", index1+3, "| index2:", index2+3,"| index3:", index3+3, "| index12:", index12+3)
        # "| index4:", index4+3, "| index5:", index5+3, "| index12:", index12+3)
        print("label:", syllable, "| ", correct1, " | ", correct2, " | ", correct3, " | ", correct12,"\n")
        # correct4, "| ", correct5, "| ",

    print("Accuracy: ", score1*100/total, score2*100/total, score3*100/total, score4*100/total, score5*100/total, score12*100/total,"\n")

def prob_norm(p1, p2, p3):
    sum = p1 + p2 + p3
    return p1/sum, p2/sum, p3/sum

def paddding(y, x):
    if y.size < 28000:
        n_minus_y = 28000 - y.size
        # print(n_minus_y, type(y), y[0])
        new_y = [0]*(n_minus_y//2) + list(y) + [0]*(28000 - y.size - n_minus_y//2)
        new_y = np.array(new_y)

        new_x = [0]*(n_minus_y//2) + list(x) + [0]*(28000 - y.size - n_minus_y//2)
        new_x = np.array(new_x)
        #print(new_x, type(new_x), new_x.shape)
        #print("zero-padding")
        return new_y, new_x

def speak_length(x):
    start = 0
    end = 28000
    for i in range(x.size):
        if x[i] > 1000:
            if start == 0:
                start = i
            else:
                end = i
    return end - start

def audio_length(fs, x):
    """Length of the audio source (sec)"""
    length = x.shape[0]/float(fs)
    # getnframes
    return length


def normalizing(x, mean, var):
    """Gaussian Normalize Function."""
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
    """Show and Save graphs of Amplitude, Zero-crossing, and Energy"""
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
    duration_avg, duration_var, duration_parse_avg, duration_parse_var, onset_r_avg, onset_r_var, onset_f_avg, onset_f_var, onset_avg, onset_var = feature_extraction()
    classify(duration_avg, duration_var, duration_parse_avg, duration_parse_var, onset_r_avg, onset_r_var, onset_f_avg, onset_f_var, onset_avg, onset_var)
    print("PA1 End")

### EE524 Pattern Recognition 19 Fall
### PA1 Counting the Total Number of Syllables in a Word
### 20194314 Kim Dayeon
import os, librosa, scipy, shutil, math, statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


n_window = 1000


#################################################################
##   Training Fuction; Feature Extraction
##   - Features
##       1) Overall Waveform
##       2) Duration (#Frames)
##       3) Parsed Duration with Energy (#Frames) (*)
##       4) Zero-Crossing Rate: Waveform & Peaks
##       5) Short Time Energy : Waveform & Peaks (*)
##       6) MFCC
##      (*) Are Used for Classification
#################################################################
def feature_extraction():

    DATA_PATH = "./PA1_DB/"
    if os.path.isdir("./img/"):
        shutil.rmtree("./img/")
    os.mkdir("./img/")

    f_train = open(DATA_PATH+"train.txt", 'r')
    lines_train = f_train.readlines()

    # Variables
    count_3, count_4, count_5 = 0, 0, 0 # 100 Samples
    wave_all = [[], [], []]
    duration_all = [[], [], []]
    duration_parse_all = [[], [], []]
    zc_all = [[], [], []]
    zc_peak_dif_all = [[], [], []]
    e_all = [[], [], []]
    e_peak_dif_all = [[], [], []]
    mfcc_all = [[], [], []]

    ## File Iteration
    for line_train in lines_train:
        line_train = line_train.rstrip('\n')
        print("Training: ", line_train)

        ## ex) line_train == "train/5syllables/99_mu-yeok-dae-pyo-bu/0024_099.wav": # last file
        (file_dir, file_id) = os.path.split(line_train)
        syllable = label_parse(file_dir)

        y, sr = librosa.load(DATA_PATH+line_train, sr=16000) # sr = 16000
        fs, x = wavfile.read(DATA_PATH+line_train) # fs = 16000
        l = audio_length(fs, x)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)
        new_y, new_x = paddding(y, x)

        ## Feature Extraction from the Train Data
        zc = stzcr(new_x, scipy.signal.get_window("boxcar", n_window)) # Find the short time zero crossing rate.
        zc_peak_dif = librosa.util.peak_pick(zc, 15, 15, 15, 15, 0.0025, 10)
        e = ste(new_x, scipy.signal.get_window("hamming", n_window)) # Find the short time energy.
        e_peak_dif = librosa.util.peak_pick(e, 15, 15, 15, 15, 0.1, 15)
        mfcc = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho') # mfcc.shape : ndarray 반환 으로 사용 -> since spectrum, didn't used
        parsed_length = speak_length(e)

        ## Save data(features) based on the label(class)
        if int(syllable) == 3:
            count_3 += 1
            duration_all[0].append(y.size)
            duration_parse_all[0].append(parsed_length)
            wave_all[0].append(new_y)
            zc_all[0].append(zc)
            e_all[0].append(e)
            mfcc_all[0].append(mfcc)
            zc_peak_dif_all[0].append(list(zc_peak_dif)[-1]-list(zc_peak_dif)[0])
            e_peak_dif_all[0].append(list(e_peak_dif)[-1]-list(e_peak_dif)[0])

        if int(syllable) == 4:
            count_4 += 1
            duration_all[1].append(y.size)
            duration_parse_all[1].append(parsed_length)
            wave_all[1].append(new_y)
            zc_all[1].append(zc)
            e_all[1].append(e)
            mfcc_all[1].append(mfcc)
            zc_peak_dif_all[1].append(list(e_peak_dif)[-1]-list(e_peak_dif)[0])
            e_peak_dif_all[1].append(list(e_peak_dif)[-1]-list(e_peak_dif)[0])

        if int(syllable) == 5:
            count_5 += 1
            duration_all[2].append(y.size)
            duration_parse_all[2].append(parsed_length)
            wave_all[2].append(new_y)
            zc_all[2].append(zc)
            e_all[2].append(e)
            mfcc_all[2].append(mfcc)
            zc_peak_dif_all[2].append(list(e_peak_dif)[-1]-list(e_peak_dif)[0])
            e_peak_dif_all[2].append(list(e_peak_dif)[-1]-list(e_peak_dif)[0])

        show_graph(file_dir, file_id, syllable, sr, new_y, zc, e)
    f_train.close()

    duration_avg = [np.mean(duration_all[0]), np.mean(duration_all[1]), np.mean(duration_all[2])]
    duration_var = [np.var(duration_all[0]), np.var(duration_all[1]), np.var(duration_all[2])] # 분산
    duration_parse_avg = [np.mean(duration_parse_all[0]), np.mean(duration_parse_all[1]), np.mean(duration_parse_all[2])]
    duration_parse_var = [np.var(duration_parse_all[0]), np.var(duration_parse_all[1]), np.var(duration_parse_all[2])] # 분산
    zc_peak_dif_avg = [np.mean(zc_peak_dif_all[0]), np.mean(zc_peak_dif_all[1]), np.mean(zc_peak_dif_all[2])]
    zc_peak_dif_var = [np.var(zc_peak_dif_all[0]), np.var(zc_peak_dif_all[1]), np.var(zc_peak_dif_all[2])]
    e_peak_dif_avg = [np.mean(e_peak_dif_all[0]), np.mean(e_peak_dif_all[1]), np.mean(e_peak_dif_all[2])]
    e_peak_dif_var = [np.var(e_peak_dif_all[0]), np.var(e_peak_dif_all[1]), np.var(e_peak_dif_all[2])]

    print("\n\n\n#################################################################################")
    print("########################### Feature Extraction Result ###########################\n")
    # print("duration_avg: ", duration_avg)
    # print("duration_var: ", duration_var)
    # print("duration_3: ", min(duration_all[0]), max(duration_all[0]))
    # print("duration_4: ", min(duration_all[1]), max(duration_all[1]))
    # print("duration_5: ", min(duration_all[2]), max(duration_all[2]), "\n")
    print(" Feature 2")
    print(" duration_parse_avg: ", duration_parse_avg)
    print(" duration_parse_var: ", duration_parse_var)
    print(" duration_parse_3: ", min(duration_parse_all[0]), max(duration_parse_all[0]))
    print(" duration_parse_4: ", min(duration_parse_all[1]), max(duration_parse_all[1]))
    print(" duration_parse_5: ", min(duration_parse_all[2]), max(duration_parse_all[2]), "\n\n")

    # print("zc_peak_dif_avg: ", zc_peak_dif_avg)
    # print("zc_peak_dif_var: ", zc_peak_dif_var)
    # print("zc_peak_dif_3: ", min(zc_peak_dif_all[0]), max(zc_peak_dif_all[0]))
    # print("zc_peak_dif_4: ", min(zc_peak_dif_all[1]), max(zc_peak_dif_all[1]))
    # print("zc_peak_dif_5: ", min(zc_peak_dif_all[2]), max(zc_peak_dif_all[2]), "\n")

    print(" Feature 4")
    print(" e_peak_dif_avg: ", e_peak_dif_avg)
    print(" e_peak_dif_var: ", e_peak_dif_var)
    print(" e_peak_dif_3: ", min(e_peak_dif_all[0]), max(e_peak_dif_all[0]))
    print(" e_peak_dif_4: ", min(e_peak_dif_all[1]), max(e_peak_dif_all[1]))
    print(" e_peak_dif_5: ", min(e_peak_dif_all[2]), max(e_peak_dif_all[2]), "\n")

    print("#################################################################################\n\n\n")

    # show the distribution histogram of duration
    # plt.hist(duration_all[0], bins=20)
    # sns.distplot(duration_all[0], rug=True, kde=False, fit=sp.stats.norm)
    # plt.show()

    return duration_avg, duration_var, duration_parse_avg, duration_parse_var, zc_peak_dif_avg, zc_peak_dif_var, e_peak_dif_avg, e_peak_dif_var




#################################################################
##   Testing Fuction; Classification
##   - Extract Features from the Test Data
##   - Using Above Features 3) and 5), Classify the Data
#################################################################
def classify(duration_avg, duration_var, duration_parse_avg, duration_parse_var, zc_peak_dif_avg, zc_peak_dif_var, e_peak_dif_avg, e_peak_dif_var):

    DATA_PATH = "./PA1_DB/"
    f_test = open(DATA_PATH+"test.txt", 'r')
    # modify above line to grade (put the new file name instead of test.txt)
    # ex. f_test = open(DATA_PATH+"new_test.txt", 'r'
    lines_test = f_test.readlines()

    # Variables
    score1, score2, score3, score4, score5 = 0, 0, 0, 0, 0
    score2_3, score2_4, score2_5 = 0, 0, 0
    score4_3, score4_4, score4_5 = 0, 0, 0

    score_a, score_b = 0, 0
    score_b_3, score_b_4, score_b_5 = 0, 0, 0
    total = 0 # 75 Samples
    correct1, correct2, correct3, correct4, correct5 = "", "", "", "", ""
    correct_a, correct_b = "", ""

    ## File Iteration
    for line_test in lines_test:
        line_test = line_test.rstrip('\n')
        print("Testing: ", line_test)

        (file_dir, file_id) = os.path.split(line_test)
        syllable = label_parse(file_dir)

        y, sr = librosa.load(DATA_PATH+line_test, sr=16000) # sr = 16000
        fs, x = wavfile.read(DATA_PATH+line_test) # fs = 16000
        l = audio_length(fs, x)

        x = np.array(x, dtype = float)
        t = np.arange(len(x)) * (1.0 / fs)
        new_y, new_x = paddding(y, x)

        ## Feature Extraction from the Test Data
        zc = stzcr(new_x, scipy.signal.get_window("boxcar", n_window)) # Find the short time zero crossing rate.
        # zc_peak_dif = librosa.util.peak_pick(zc, 15, 15, 15, 15, 0.0025, 10) # removed
        e = ste(new_x, scipy.signal.get_window("hamming", n_window)) # Find the short time energy.
        e_peak_dif = librosa.util.peak_pick(e, 15, 15, 15, 15, 0.1, 15)
        mfcc = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho') # mfcc.shape : ndarray 반환 으로 사용
        parsed_length = speak_length(e)
        # print("zc_peak_dif: ", zc_peak_dif, zc_peak_dif.size, list(zc_peak_dif)[-1]-list(zc_peak_dif)[0])
        # print("e_peak_dif: ", e_peak_dif, e_peak_dif.size, list(e_peak_dif)[-1]-list(e_peak_dif)[0])

        ## Feature Extraction from the Test Data & Classifiaction
        ## - Make Gaussian Distribution with Mean and Variance
        ## - Normalize Probability to Make Sum as 1
        ## - Assign Class with maximum Probability
        ## - Check Correct/Wrong

        ## Feature 2) Duration (#Frames) (Not Used)
        p3 = normalizing(y.size, duration_avg[0], duration_var[0])
        p4 = normalizing(y.size, duration_avg[1], duration_var[1])
        p5 = normalizing(y.size, duration_avg[2], duration_var[2])
        a, b, c = prob_norm(p3, p4, p5)
        p_length = [a, b, c] # probability
        index1 = p_length.index(max(p_length))
        # print("p1: ", index1+3, max(p_length), p_length)
        if(index1+3 == int(syllable)):
            score1 += 1
            correct1 = "correct"
        else: correct1 = " wrong "

        ## Feature 3) Parsed Duration with Energy (#Frames) (*)
        p3_2 = normalizing(parsed_length, duration_parse_avg[0], duration_parse_var[0])
        p4_2 = normalizing(parsed_length, duration_parse_avg[1], duration_parse_var[1])
        p5_2 = normalizing(parsed_length, duration_parse_avg[2], duration_parse_var[2])
        a, b, c = prob_norm(p3_2, p4_2, p5_2)
        p_length2 = [a, b, c]
        index2 = p_length2.index(max(p_length2))
        # print("p2: ", index2+3, max(p_length2), p_length2)
        if(index2+3 == int(syllable)):
            score2 += 1
            correct2 = "correct"
            if index2 == 0: score2_3 += 1
            elif index2 == 1: score2_4 += 1
            else: score2_5 += 1
        else: correct2 = " wrong "

        ## Feature 4) Zero-Crossing Rate: Peaks (Not Used)
        # p3_3 = normalizing(list(zc_peak_dif)[-1]-list(zc_peak_dif)[0], zc_peak_dif_avg[0], zc_peak_dif_var[0])
        # p4_3 = normalizing(list(zc_peak_dif)[-1]-list(zc_peak_dif)[0], zc_peak_dif_avg[1], zc_peak_dif_var[1])
        # p5_3 = normalizing(list(zc_peak_dif)[-1]-list(zc_peak_dif)[0], zc_peak_dif_avg[2], zc_peak_dif_var[2])
        # a, b, c = prob_norm(p3_3, p4_3, p5_3)
        # p_3 = [a, b, c] # probability
        # index3 = p_3.index(max(p_3))
        # # print("p3: ", index3+3, max(p_3), p_3)
        # if(index3+3 == int(syllable)):
        #     score3 += 1
        #     correct3 = "correct"
        # else: correct3 = " wrong "

        ## Feature 4) Short Time Energy: Peaks  (*)
        p3_4 = normalizing(list(e_peak_dif)[-1]-list(e_peak_dif)[0], e_peak_dif_avg[0], e_peak_dif_var[0])
        p4_4 = normalizing(list(e_peak_dif)[-1]-list(e_peak_dif)[0], e_peak_dif_avg[1], e_peak_dif_var[1])
        p5_4 = normalizing(list(e_peak_dif)[-1]-list(e_peak_dif)[0], e_peak_dif_avg[2], e_peak_dif_var[2])
        a, b, c = prob_norm(p3_4, p4_4, p5_4)
        p_4 = [a, b, c] # probability
        index4 = p_4.index(max(p_4))
        # print("p4: ", index4+3, max(p_4), p_4)
        if(index4+3 == int(syllable)):
            score4 += 1
            correct4 = "correct"
            if index4 == 0: score4_3 += 1
            elif index4 == 1: score4_4 += 1
            else: score4_5 += 1
        else: correct4 = " wrong "

        ## Combination of Feature 3) and 4) - Ver.1 (Not Used)
        p_final = [p_length2[0]*p_4[0], p_length2[1]*p_4[1], p_length2[2]*p_4[2]]
        index_a = p_final.index(max(p_final))
        if(index_a+3 == int(syllable)):
            score_a += 1
            correct_a = "correct"
        else: correct_a = " wrong "

        ## Combination of Feature 3) and 4) - Ver.2 (*)
        if index2 == index4:
            index_b = index2
        else:
            if p_length2[index2] < p_4[index4]:
                index_b = index4
            else:
                index_b = index2
        if(index_b+3 == int(syllable)):
            score_b += 1
            correct_b = "correct"
            if index_b == 0: score_b_3 += 1
            elif index_b == 1: score_b_4 += 1
            else: score_b_5 += 1
        else: correct_b = " wrong "

        total += 1
        # print("label:", syllable, "| index1:", index1+3, "| index2:", index2+3, "| index4:", index4+3, "| index_a:", index_a+3, "| index_b:", index_b+3)
        # print("label:", syllable, "| ", correct1, " | ", correct2, " | ", correct4, " |  ", correct_a, " |  ", correct_b,"\n")
        print("label:", syllable, "| index2:", index2+3, "| index4:", index4+3, "| index_b:", index_b+3)
        print("         | ", correct2, " | ", correct4, " |  ", correct_b,"\n")

    print("\n\n\n#################################################################################")
    print("############################# Classification Result #############################\n")

    print("                    | Feature 2 | Feature 4 | Combined Feature")
    print("       -------------------------------------------------------")
    print("         3Syllables | ", "%0.2f" %float(score2_3*100/25),"  |  ","%0.2f" %float(score4_3*100/25),"  |   ","%0.2f" %float(score_b_3*100/25))
    print("         4Syllables |  ", "%0.2f" %float(score2_4*100/25),"  |  ","%0.2f" %float(score4_4*100/25),"  |   ","%0.2f" %float(score_b_4*100/25))
    print("         5Syllables |  ", "%0.2f" %float(score2_5*100/25),"  |  ","%0.2f" %float(score4_5*100/25),"  |   ","%0.2f" %float(score_b_5*100/25))
    print("       -------------------------------------------------------")
    print("           Total    |  ", "%0.2f" %float(score2*100/total), "  |  ", "%0.2f" %float(score4*100/total),"  |   ", "%0.2f" %float(score_b*100/total))
    print("\n         -> Total Accuracy of Combined Feature: ", "%0.2f" %float(score_b*100/total), "%")
    print("\n#################################################################################")


#################################################################
##   Useful Supplementary Fuctions
#################################################################
def prob_norm(p1, p2, p3):
    """Normalize Probability to Make Sum as 1"""
    sum = p1 + p2 + p3
    return p1/sum, p2/sum, p3/sum


def paddding(y, x):
    """Zero-Padding Function to Make Constant Audio Length"""
    if y.size < 28000:
        n_minus_y = 28000 - y.size
        # print(n_minus_y, type(y), y[0])
        new_y = [0]*(n_minus_y//2) + list(y) + [0]*(28000 - y.size - n_minus_y//2)
        new_y = np.array(new_y)

        new_x = [0]*(n_minus_y//2) + list(x) + [0]*(28000 - y.size - n_minus_y//2)
        new_x = np.array(new_x)
        return new_y, new_x


def speak_length(x):
    """Parse the Audio Length with Threshold Short Time Energy"""
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
    """Length of the Audio Source (sec)"""
    length = x.shape[0]/float(fs)
    return length


def normalizing(x, mean, var):
    """Gaussian Normalize Function"""
    return (1/math.sqrt(2*math.pi*var))*math.exp(-((x-mean)**2)/(2*var))


def _sgn(x):
    """Numpy Sign Function"""
    y = np.zeros_like(x)
    y[np.where(x >= 0)] = 1.0
    y[np.where(x < 0)] = -1.0
    return y


def stzcr(x, win):
    """Compute Short Time Zero-Crossing Rate"""
    if isinstance(win, str):
        win = scipy.signal.get_window(win, max(1, len(x) // 8))
    win = 0.5 * win / len(win)
    x1 = np.roll(x, 1)
    x1[0] = 0.0
    abs_diff = np.abs(_sgn(x) - _sgn(x1))
    return scipy.signal.convolve(abs_diff, win, mode="same")


def ste(x, win):
    """Compute Short Time Energy"""
    if isinstance(win, str):
        win = scipy.signal.get_window(win, max(1, len(x) // 8))
    win = win / len(win)
    return scipy.signal.convolve(x**2, win**2, mode="same")


def label_parse(file_dir):
    """Parse the # Syllables from File Directory."""
    syllable = int(file_dir.split('/')[1][0])
    return syllable


def show_graph(file_dir, file_id, syllable, sr, y, zc, e):
    """Show and Save Graphs of Amplitude, Zero-Crossing, and Energy"""
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
    duration_avg, duration_var, duration_parse_avg, duration_parse_var, zc_peak_dif_avg, zc_peak_dif_var, e_peak_dif_avg, e_peak_dif_var = feature_extraction()
    classify(duration_avg, duration_var, duration_parse_avg, duration_parse_var, zc_peak_dif_avg, zc_peak_dif_var, e_peak_dif_avg, e_peak_dif_var)
    print("-END-")

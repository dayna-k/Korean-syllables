### EE524 Pattern Recognition 19 Fall
### PA1 Counting the Total Number of Syllables in a Word
### 20194314 Kim Dayeon
import os, librosa, scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
# from numpy import *
# from librosa import *

# feature: length, pitch
def main():
    ## train
    DATA_PATH = "./PA1_DB/"
    # f_train = open(DATA_PATH+"train.txt", 'r')
    f_train = open(DATA_PATH+"train2.txt", 'r')
    lines_train = f_train.readlines()
    for line_train in lines_train:
        line_train = line_train.rstrip('\n')
        print(line_train)

        # if line_train == "train/5syllables/99_mu-yeok-dae-pyo-bu/0024_099.wav": # last file
        (file_dir, file_id) = os.path.split(line_train)
        print("file_dir: ", file_dir)
        print("file_id: ", file_id)
        print("length: ", feature1(DATA_PATH+line_train))
        # open(DATA_PATH+line_train, 'r')
        y, sr = librosa.load(DATA_PATH+line_train, sr=16000)
        time = np.linspace(0, len(y)/sr, len(y)) # time axis
        fig, ax1 = plt.subplots() # plot
        ax1.plot(time, y, color = 'b', label='speech waveform')
        ax1.set_ylabel("Amplitude") # y 축
        ax1.set_xlabel("Time [s]") # x 축
        plt.title(file_id) # 제목
        plt.savefig("./img/"+file_id+'.png')
        plt.show()
    # print(lines)
    f_train.close()

def feature1(file_path):
    sr, audio = wavfile.read(file_path)
    length = audio.shape[0]/float(sr)
    # getnframes
    return length


if __name__ == '__main__':
    main()
    print("3")

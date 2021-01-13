import wave
from pydub import AudioSegment
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


# The most notable
# downside of using MFCC is its sensitivity to
# noise due to its dependence on the spectral
# form.

import os
print(os.getcwd())


# The voice files for testing
satu = "./data/voicebank/test/satu1.wav"
dua = "./data/voicebank/test/dua1.wav"
get = "./data/voicebank/test/tiga1.wav"
#backward = "../data/voicebank/test/backward01.wav"

word_list = ["satu", "dua", "tiga", "empat",
             "lima", "nol"]  # , "forward" ]


# For each testing file, we are extracting feature vectors...
(rate, sig) = wav.read(satu)
mfcc_feat = mfcc(sig, rate, nfft=1536)
d_mfcc_feat = delta(mfcc_feat, 2)

#print ( "d_mfcc : " , d_mfcc_feat )
#print ( "mfcc : " , mfcc_feat [ : , 1:14   ] )


print("Size mfcc_feat : ", len(mfcc_feat))
print("d_mfcc_feat : ", len(d_mfcc_feat))
satu = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
print("d_mfcc_feat : ", d_mfcc_feat)


(rate, sig) = wav.read(dua)
# print "Read : " , ses_yol + word + "/" + word + str ( i / 10 ) + str ( i %10 ) +".wav"
mfcc_feat = mfcc(sig, rate, nfft=1536)
d_mfcc_feat = delta(mfcc_feat, 2)

dua = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)

(rate, sig) = wav.read(get)
# print "Read : " , ses_yol + word + "/" + word + str ( i / 10 ) + str ( i %10 ) +".wav"
mfcc_feat = mfcc(sig, rate, nfft=1536)
d_mfcc_feat = delta(mfcc_feat, 2)

get = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)

hmm_list = {}
ses_yol = "./data/voicebank/"


scalers = {}


for word in word_list:
    sizes = []
    P1 = []

    cou = 0

    file_list = os.listdir(ses_yol + word)
    for i in file_list:

        if cou == 500:  # 500
            break

        cou += 1

        print("File : " + ses_yol + word + "/" + i)

        audio = AudioSegment.from_file(
            ses_yol + word + "/" + i, format="wav", frame_rate=32000)
        audio = audio.set_frame_rate(16000)
        audio.export("filtered-talk1.wav", format="wav")

        (rate, sig) = wav.read("filtered-talk1.wav")
        mfcc_feat = mfcc(sig, rate, nfft=1536)

        # delta mfcc
        d_mfcc_feat = delta(mfcc_feat, 2)

        concatenated = np.concatenate(
            (mfcc_feat, d_mfcc_feat), axis=1).tolist()

        for sample in concatenated:
            P1.append(sample)

        print("P1 len : ", len(P1))
        print("P2 in len : ", len(P1[0]))
        print("Concat size : ", len(concatenated), ":", len(concatenated[0]))
        # previously mfc__feat
        sizes.append(len(concatenated))

    # standard_scaler = StandardScaler()

    # standard_scaler.fit(P1)
    # scalers[word] = standard_scaler
    # P1 = standard_scaler.transform(P1)

    # previous n_component 4
    kl = hmm.GMMHMM()
    model = hmm.GaussianHMM(n_components=16, n_iter=1000, verbose=True)
    print("Size len : ", len(sizes))
    print("Size : ", sizes)

    # [a1, a2, a3, a4, b1, b2, b3]
    # a = [a1, a2, a3, a4]
    # b = [b1, b2, b3]
    # sizes = [4 , 3]
    print("P 1 2 : ", P1[0: 2])
    print("p1 len : ", len(P1), " sum : ", np.sum(sizes))
    print(len(P1[0: 64]))
    print(len(sizes[0: 64]))
    model.fit(P1, lengths=sizes)
    hmm_list[word] = model
    print("*******Covars*********")


def test(sample, hmm_list):
    global word_list
    result = {}

    for word in word_list:
        result[word] = hmm_list.get(word).score(sample, lengths=[len(sample)])

    label = word_list[0]
    en_buyuk = result[word_list[0]]
    for i in range(1, len(word_list)):
        if result[word_list[i]] > en_buyuk:
            en_buyuk = result[word_list[i]]
            label = word_list[i]

    print("Secilen label : ", label)
    print("Result : ", result)


print("\n\nHMM models will be saved...")

pickle.dump(hmm_list, open("TrainedHmms.hmm", "wb"))
pickle.dump(scalers, open("Scalers.scl", "wb"))
print("HMM models have been saved....\n\n")

print(test(dua, hmm_list))
print(test(satu, hmm_list))

print("Model Bilangan HMM : ")

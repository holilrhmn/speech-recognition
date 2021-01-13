from pydub import AudioSegment
from sklearn import neural_network
import numpy as np
from cv2 import *
import random
import pickle
import math

from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler


def get_score_vector(sample, hmm_list):
    global word_list
    # print("Sample : ", sample.shape)
    # print("Sample : len : ", len(sample))

    result = {}
    smallest = 0
    total = 0

    for word in word_list:
        # print("word : ", word)
        result[word] = hmm_list.get(word).score(sample, lengths=[len(sample)])
        # print("result : ", result[word])
        total += result[word]

    average = total / len(word_list)
    feature_vector = [result[word] for word in word_list]
    # combine each hmm's score for the word, subtract the average score...
    combined = [result[word] - average for word in word_list]
    # print(combined)
    return combined


ses_yol = "./data/voicebank/"
nn_scaler = pickle.load(open("NNTrainer/neurospeechscaler.scl", 'rb'))
hmm_list = pickle.load(open('Trainer/TrainedHmmsPreprocessing.hmm', 'rb'))
scalers = pickle.load(open("Trainer/ScalersPreprocessing.scl", 'rb'))

word_list = ["satu", "dua", "tiga", "empat",
             "lima", "nol"]  # , "forward" ]

trained = pickle.load(open("NNTrainer/neurospeech.nn", "rb"))


audio = AudioSegment.from_file(
    ses_yol + "dua" + "/dua2.wav", format="wav", frame_rate=32000)
audio = audio.set_frame_rate(16000)
audio.export("filtered-talk1.wav", format="wav")

(rate, sig) = wav.read("filtered-talk1.wav")
mfcc_feat = mfcc(sig, rate, nfft=1536)

d_mfcc_feat = delta(mfcc_feat, 2)

data = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1).tolist()

data = scalers["dua"].transform(data)
vector = get_score_vector(data, hmm_list)
print("Vector : ", vector)
vector = nn_scaler.transform([vector])
print("Prediction ", trained.predict(vector))

print("Prediction ", trained.predict_proba(vector))


def test_all_files(path, hmm_list):
    global false_list, word_list, false_number, correct_number
    file_list = []
    files = os.listdir(path)
    for file_name in files:
        for u in word_list:
            if file_name.__contains__(u):
                file_list.append(file_name)

    for file in file_list:

        for labels in word_list:
            if file.__contains__(labels):
                tested_label = labels
                break
        #tested_label = file[ :-6]
        print("Tested label : ", tested_label)

        audio = AudioSegment.from_file(
            path + "/" + file, format="wav", frame_rate=32000)
        audio = audio.set_frame_rate(16000)
        audio.export("filtered-talk1.wav", format="wav")

        (rate, sig) = wav.read("filtered-talk1.wav")
        # print "Read : " , ses_yol + word + "/" + word + str ( i / 10 ) + str ( i %10 ) +".wav"
        mfcc_feat = mfcc(sig, rate, nfft=1536)
        d_mfcc_feat = delta(mfcc_feat, 2)

        data = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1).tolist()

        for label in word_list:
            if label in file:
                break

        data = scalers[label].transform(data)
        vector = get_score_vector(data, hmm_list)
        vector = nn_scaler.transform([vector])
        print("Vector : ", vector.shape)
        predicted = trained.predict(vector)
        print("type : ", type(predicted[0]))
        print("Res : ", str(predicted[0]))
        print("Prediction ", trained.predict_proba(vector))
        if predicted[0] != tested_label:
            false_list.append(file + " predicted answer : " + predicted[0])
            false_number += 1
        else:
            correct_number += 1


correct_number = 0
false_number = 0
false_list = []

test_all_files("./data/voicebank/test", hmm_list)

print("\n\nCorrect numbers : ", correct_number)
print("False numbers : ", false_number)
# print "Score : %" , 100*(correct_number)/(correct_number + false_number)
if (false_number > 0):
    print("False samples : ", false_list)

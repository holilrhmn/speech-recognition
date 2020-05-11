from sklearn import neural_network
import numpy as np
import random
import pickle
import math
import os
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler

print(os.getcwd())
def get_score_vector ( sample , hmm_list ):
    global word_list
    # print("Sample : ", sample.shape)
    # print("Sample : len : ", len(sample))

    result = {}
    smallest = 0
    total = 0

    for word in word_list:
        # print("word : ", word)
        result  [ word ] = hmm_list.get ( word ).score ( sample, lengths=[len(sample)])
        # print("result : ", result[word])
        total += result [ word ]

    average = total / len ( word_list )
    feature_vector = [ result[ word ]   for word in word_list    ]
    # combine each hmm's score for the word, subtract the average score...
    combined = [ result [ word ] - average for word in word_list ]
    # print(combined)
    return combined


word_list = ["shoes" ,"textbook","map" , "cell_phone" , "violin" , "ball" , "computer"] # , "forward" ]
print(os.listdir("../"))
hmm_list = pickle.load ( open ( '../Trainer/TrainedHmmsPreprocessing.hmm' , 'rb'))
scalers = pickle.load(open("../Trainer/ScalersPreprocessing.scl", 'rb'))

network = neural_network.MLPClassifier(tol=1e-7, verbose=1000, learning_rate_init=.040, solver="adam", max_iter = 1500, hidden_layer_sizes=(705, 700))

nfft=1536
X = []
Y = []

from pydub import AudioSegment
ses_yol = "../data/voicebank/"

import os

for word in word_list:

    file_list = os.listdir(ses_yol + word)
    for i in file_list:
        print ("File : " + ses_yol + word + "/" + i)

        # song.export("filtered-talk.wav", format="wav")
        audio = AudioSegment.from_file(ses_yol + word + "/" + i, format="wav", frame_rate=44100)
        audio = audio.set_frame_rate(16000)
        audio.export("filtered-talk1.wav", format="wav")

        (rate, sig) = wav.read("filtered-talk1.wav")
        mfcc_feat = mfcc(sig, rate, nfft=1536)

        d_mfcc_feat = delta(mfcc_feat, 2)

        data = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
        data = scalers[word].transform(data)

        print("mfcc : size : ", mfcc_feat.shape)
        print("d_mfcc : size : ", d_mfcc_feat.shape)

        vector = get_score_vector(data, hmm_list)
        #print ( "vector : " , vector )

        X.append( vector )
        Y.append ( word )

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
pickle.dump(scaler, open("neurospeechscaler.scl", "wb"))

trained = network.fit ( X , Y )
print ("network train edildi...")

pickle.dump ( trained , open ( "neurospeech.nn" , "wb"))

print ("Neural network has been saved....")

print ("Program has finished...")

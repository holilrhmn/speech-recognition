from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pydub import AudioSegment


def test ( sample , hmm_list):
    global word_list
    result = {}

    for word in word_list:
        result  [ word ] = hmm_list.get ( word ).score ( sample,lengths=[len(sample)] )

    label = word_list [ 0 ]
    en_buyuk = result [ word_list [ 0 ] ]
    for i in range ( 1 , len ( word_list ) ):
        if result [ word_list [ i ]  ] > en_buyuk:
            en_buyuk = result [ word_list [ i ] ]
            label = word_list  [ i ]




    print ("Secilen label : " , label)
    print ("Result : " , result)
    return label


test_folder = "data/voicebank/test"

word_list = ["shoes" ,"textbook","map" , "cell_phone" , "violin" , "ball" , "computer"] # , "forward" ]

hmm_list = pickle.load ( open ( 'Trainer/TrainedHmmsPreprocessing.hmm' , 'rb'))

scalers = pickle.load(open("Trainer/ScalersPreprocessing.scl", 'rb'))

import os

correct = 0
fail = 0

for word in os.listdir(test_folder):

    print ( "File : " + test_folder +"/"+ word)

    #song = AudioSegment.from_wav(ses_yol + word +"/" + i)
    #new = song.low_pass_filter(5000)

    #song.export("filtered-talk.wav", format="wav")
    audio = AudioSegment.from_file(test_folder+"/"+word, format="wav", frame_rate=44100)
    audio = audio.set_frame_rate(16000)
    audio.export("filtered-talk1.wav", format="wav")



    (rate,sig) = wav.read( "filtered-talk1.wav" )
    mfcc_feat = mfcc(sig,rate, nfft=1536)

    # delta mfcc
    d_mfcc_feat = delta(mfcc_feat, 2)

    # combine two feature vectors...
    P1 = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1).tolist()
    for label in word_list:
        if label in word:
            break
    P1 = scalers[label].transform(P1)
    # print(P1)
    predicted_label = test(P1, hmm_list)
    print("Predicted : ", predicted_label, " Actual label : ", label)
    if predicted_label == label:
        correct += 1
    else:
        fail += 1

print("Fail : ", fail, " Correct : ", correct)


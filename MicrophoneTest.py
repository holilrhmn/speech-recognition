import pygame
import pickle
import pyaudio
import wave

from pydub import AudioSegment

from pydub.silence import split_on_silence


from hmmlearn import hmm

from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

def get_feature_vector ( sample , hmm_list ):
    global word_list
    result = {}
    smallest = 0
    total = 0
    for word in word_list:
        data_ = scalers[word].transform(sample)
        result  [ word ] = hmm_list.get ( word ).score ( data_ )
        total += result [ word ]
        #we are getting the smallest number, later we will normalize the feature vector...
        if smallest > result[ word ]:
            smallest = result[ word ]

    average = total / len ( word_list )

    feature_vector = [ result[ word ]   for word in word_list    ]
    return [ result [ word ] - average for word in word_list ]

trained = pickle.load ( open ( 'NNTrainer/neurospeech.nn' , 'rb'))
nn_scaler = pickle.load(open("NNTrainer/neurospeechscaler.scl", 'rb'))

def test(sample, hmm_list):
    global word_list, window
    result = {}

    for word in word_list:
        data_ = scalers[word].transform(sample)
        result  [ word ] = hmm_list.get ( word ).score ( data_ )

    label = word_list [ 0 ]
    en_buyuk = result [ word_list [ 0 ] ]
    for i in range ( 1 , len ( word_list ) ):
        if result [ word_list [ i ]  ] > en_buyuk:
            en_buyuk = result [ word_list [ i ] ]
            label = word_list  [ i ]

    print ("Result : ", result)
    return label

max_number = 0
sound_file_index = 0
import os
file_list = os.listdir("recordings")
if len ( file_list ) == 0:
    sound_file_index = 0
else:
    for file in file_list:
        tokens = file.split( "." )
        file_number = int ( tokens[0] [ 5: ] )
        if file_number > max_number:
            max_number  = file_number

sound_file_index = max_number +1

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = None

frames = []

word_list = [ "textbook" , "shoes" , "map" , "cell_phone" ,"ball" , "violin" , "computer"] # , "forward" ]

hmm_list = pickle.load ( open ( 'Trainer/TrainedHmmsPreprocessing.hmm' , 'rb'))
scalers = pickle.load(open("Trainer/ScalersPreprocessing.scl", 'rb'))


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
record = False

import speech_recognition as sr

# Record Audio
r = sr.Recognizer()

print ( "Stream ready..." )
pygame.init()
window = pygame.display.set_mode ( ( 300 , 300 ) )
pygame.display.set_caption('Press a and speak and release the key')
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stream.close()
            pygame.quit();  # sys.exit() if sys is imported

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                print("Hey, you pressed the key, '0'!")
                if record == False:
                    frames = []

                    print ( "Please speak" )
                    stream.start_stream()

                    record = True

            if event.key == pygame.K_ESCAPE:
                if stream is not None:
                    stream.close()
                pygame.quit()
                exit ( 0 )
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                print ( "Key a has been released..." )

                record = False
                stream.stop_stream()

                dat = b''.join(frames)
                audio_data = sr.AudioData(dat , 44100, 2)
                try:
                    google_result = r.recognize_google(audio_data)
                except Exception:
                    google_result = "Error"

                waveFile = wave.open(  "sound.wav" , 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(dat)
                waveFile.close()

                waveFile = wave.open("recordings/sound"+ str ( sound_file_index )  +".wav", 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(dat)
                waveFile.close()
                sound_file_index += 1

                #song = AudioSegment.from_wav("sound.wav")
                #new = song.low_pass_filter(5000)

                #song.export("filtered-talk.wav", format="wav")
                # print "Read : " , ses_yol + word + "/" + word + str ( i / 10 ) + str ( i %10 ) +".wav"

                mic_sig = np.frombuffer(b''.join(frames), dtype=np.int16)

                #print ( "Signal : " , sig )
                #print ( "mic sig : " , mic_sig )

                #print ( "Signal len : " , len ( sig ) , " - frame data len : " , len ( frames ) )

                song = AudioSegment.from_wav("sound.wav")
                new = song.low_pass_filter(5000)

                song.export("filtered-talk.wav", format="wav")
                audio3 = AudioSegment.from_file("filtered-talk.wav", format="wav", frame_rate=44100)
                audio3 = audio3.set_frame_rate(16000)
                audio3.export("filtered-talk1.wav", format="wav")

                sound_file = AudioSegment.from_wav("filtered-talk1.wav")
                audio_chunks = split_on_silence(sound_file,
                                                # must be silent for at least half a second
                                                min_silence_len=200,

                                                # consider it silent if quieter than -16 dBFS
                                                silence_thresh=-16                                                )

                count = 0

                data_ = []

                for i, chunk in enumerate(audio_chunks):
                    count += 1


                    # print "i : " , i , " chunk duration : " , chunk.duration_seconds
                    #if chunk.duration_seconds < 0.200:
                        #continue
                    # print ( "Audio chunk" , chunk.duration_seconds )


                    # chunk.export( out_file , format="wav" )

                    chunk.export("temp" + str(i) + ".wav", format="wav")
                    w = wave.open("temp" + str(i) + ".wav", 'rb')
                    data_.append( w.readframes(w.getnframes()))
                    w.close()
                print ( "Count : " , count )
                if count >  0 :
                    #print ( "ifteyiz")

                    try:
                        #print ( "try catch")

                        output = wave.open("com.wav", 'wb')

                        output.setframerate(16000)
                        output.setsampwidth(audio.get_sample_size(FORMAT))
                        output.setnchannels(CHANNELS)
                        output.writeframes(b''.join(data_))

                        #output.setparams(data[0][0])
                        #output.writeframes(data[0])

                        output.close()
                        print ( "Dosya yazildi..")
                        #(rate, sig) = wav.read("temp" + str( 0 ) + ".wav")
                        (rate, sig) = wav.read( "com.wav" )

                        #(rate, sig) = RATE , mic_sig # wav.read ( "sound.wav" )

                        mfcc_feat = mfcc(sig, rate, nfft=1536)
                        d_mfcc_feat = delta(mfcc_feat, 2)

                        P1 = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
                        # print ( "Predicted Result : " )

                        label = test(P1, hmm_list)

                        vector = get_feature_vector(P1, hmm_list)
                        vector = nn_scaler.transform([vector])
                        predicted = trained.predict(vector)
                        print("predicted = ", predicted)
                        # nn's output is considered, not individual hmm models
                        print("Individual HMM prediction : ", label)
                        label = predicted[0]

                        object_image = pygame.image.load("images/"+label + ".png")
                        window = pygame.display.set_mode((object_image.get_width(), object_image.get_height()), 0, 32)

                        print ( "Our speech recognition (nn) result : " , label )
                        print ( "Google speech recognition result : " , google_result )

                        window.blit(object_image, (0, 0))
                        pygame.display.update()
                        if label =="cell_phone":label="cell phone"
                        print ("type : ", type(predicted[0]))
                        print ("Neural Res : ", str(predicted[0]))

                        print ("Prediction ", trained.predict_proba(vector))

                    except UnicodeEncodeError as err:
                        print ( "Err : " , err  )
                else:
                    print ( "No recognition..." )


    if record == True:
        data = stream.read(CHUNK)
        frames.append(data)
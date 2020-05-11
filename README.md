# HiddenMarkovModelSpeechRecognition
Word recognition with Hidden Markov Models (Python 3.7)

Recognized words = [ "textbook" , "shoes" , "map" , "cell_phone" ,"ball" , "violin" , "computer"]

Each word has been uttered 500 times and data set is created

Trainer/HMMTrainer.py 
 - Extracts MFCC and delta features from sound files, combines these two feature vectors into one and trains one Hidden Markov Model for each word without preprocessing the combined feature vector.

HMMTest.py
 - Loads test sound files and extracts combined feature as HMMTrainer.py does, feeds combined features to every trained Hidden Markov Model and chooses the one that attains the highest score as label for the test sound
 
HMMTrainerwithPreprocessing.py
 - Follows the same steps as Trainer/HMMTrainer.py with only one exception: before training Hidden Markov Models it standardizes the data (subtracts mean and divides with standard deviation) 
 
HMMPreProcessingTest.py
 - Follows the same steps as HMMTest.py for Hidden Markov Models trained via preprocessed data

NeuralSpeech.py
 - For every train data sample, it constructs a feature vector consisting of output scores of each trained HMM; standardizes the whole data set built by feature vector of scores and trains an Multilayer Perceptron

NeuralSpeechTest.py
 - Assesses the performance of trained MLP on the test data
 
MicrophoneTest.py
 - User presses key 'a', utters the word to be recognized and releases the key; program aligns the spoken sound for recognition, extracts features, feeds these features to both individual HMMs and trained MLP. Obtained sound is also sent to Google Speech Recognizer. Program outputs detection results of HMMs, MLP,  Google Speech recognizer. It displays a picture of corresponding word recognized by MLP.

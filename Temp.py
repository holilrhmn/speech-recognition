from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler

shoes_file = "./data/voicebank/test/satu1.wav"
textbox = "./data/voicebank/test/dua1.wav"
get = "./data/voicebank/test/tiga1.wav"

(rate, sig) = wav.read(shoes_file)
print("sig shape : ", sig.shape)
mfcc_feat = mfcc(sig, rate, nfft=1103)
print("mfcc shape : ", mfcc_feat.shape)
d_mfcc_feat = delta(mfcc_feat, 2)
print("delta features shape : ", d_mfcc_feat.shape)

delta_list = d_mfcc_feat.tolist()
print(delta_list)

X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
X2 = [[2.4], [4.2], [0.5], [-0.24]]
X = np.concatenate([X1, X2])
lengths = [len(X1), len(X2)]
print(X)

# hmm = hmm.GaussianHMM(n_components=8, n_iter=100, verbose=True).fit(X, lengths)

kl = hmm.GMMHMM(covariance_type="diag", n_components=8,
                n_iter=100, verbose=True).fit(X, lengths)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

print((X-mean)/std)
scaler = StandardScaler()
scaler.fit(X)
print(scaler.transform(X))

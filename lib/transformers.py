import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import spectrogram
from utils import combiner

class MontageExtractor(BaseEstimator, TransformerMixin):
    """Takes raw data and extracted montages from channels"""
    def __init__(self):
        pass

        self.fs = 0
        self.montages = []
        self.n_montages = 0
        self.montage_type = 1

    def fit(self,X):
        """Reads metadata and set parameters according to the montage type"""
        self.fs = X.info['sfreq']
        self.montage_type = X.info['montage_type']
        if self.montage_type == 3:
            self.n_montages = 20
        else:
            self.n_montages = 22
        return self

    def transform(self,X):
        self.montages, self.montages_names = combiner(X, montage_type=self.montage_type) # Make montages (difference of electrodes)
        self.montages = [montage/montage.max() for montage in self.montages] # Normalization
        self.montages = np.array(self.montages) # Conversion to numpy array
        return self

class SpectrogramTransformer(BaseEstimator, TransformerMixin):
    """Takes extracted montages (time-series data) and returns the associated spectrogram"""
    def __init__(self,
                window=('tukey', .25),
#                 nperseg=None,
                noverlap=None,
                nfft=None,
                detrend='constant',
                return_onesided=True,
                scaling='density',
                axis=- 1,
                mode='psd',
                win_duration=8):
        self.window = window
#        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.mode = mode
        self.win_duration = win_duration

    def fit(self, X):
        self.nperseg = int(X.fs*self.win_duration)
        return self

    def transform(self, X):
        """Performs transformation. Computes for each montage the associated spectrogram. Returns also frequency and time range"""
        spectrograms_list = [spectrogram(X.montages[id_montage],
                                         fs=X.fs,
                                         window=self.window,
                                         nperseg=self.nperseg,
                                         noverlap=self.noverlap,
                                         nfft=self.nfft,
                                         detrend=self.detrend,
                                         return_onesided=self.return_onesided,
                                         scaling=self.scaling,
                                         axis=self.axis,
                                         mode=self.mode)
                             for id_montage in range(X.n_montages)]
        self.f, self.t = spectrograms_list[0][:2] # f,t is the same for every montage
        spectrograms_db_list = [10*np.log10(Sxx) for (_,_,Sxx) in spectrograms_list]
        return self.t, self.f, spectrograms_db_list

class SpectrumXY(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X):
        return self

    def transform(self,X):
        X = X.reshape(len(X), 1)
        return X + X.T



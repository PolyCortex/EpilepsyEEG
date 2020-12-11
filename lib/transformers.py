import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import spectrogram
from utils import combiner


class SpectrogramTransformer(BaseEstimator, TransformerMixin):
    """Takes raw data and returns the associated spectrogram"""
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
        """Reads metadata and set parameters according to the montage type"""
        self.fs = X.info['sfreq']
        self.nperseg = int(self.fs*self.win_duration)
        self.montage_type = X.info['montage_type']
        if self.montage_type == 3:
            self.n_montages = 20
        else:
            self.n_montages = 22
        return self

    def transform(self, X):
        """Performs transformation. Computes for each montage the associated spectrogram. Returns also frequency and time range"""
        montages, self.montages_names = combiner(X, montage_type=self.montage_type) # Make montages (difference of electrodes)
        montages = [montage/montage.max() for montage in montages] # Normalization
        spectrograms_list = [spectrogram(montages[id_montage],
                                         fs=self.fs,
                                         window=self.window,
                                         nperseg=self.nperseg,
                                         noverlap=self.noverlap,
                                         nfft=self.nfft,
                                         detrend=self.detrend,
                                         return_onesided=self.return_onesided,
                                         scaling=self.scaling,
                                         axis=self.axis,
                                         mode=self.mode)
                             for id_montage in range(self.n_montages)]
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



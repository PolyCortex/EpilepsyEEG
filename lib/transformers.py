import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import spectrogram
from utils import combiner

class SpectrogramTransformer(BaseEstimator, TransformerMixin):
    """Takes raw data and returns the associated spectrogram"""
    def __init__(self, window=('tukey', .25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, mode='psd'):
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.mode = mode

    def fit(self, X):
        self.fs = X.info['sfreq']
        self.montage_type = X.info['montage_type']
        return self

    def transform(self, X):
        montages, montages_names = combiner(X, montage_type=self.montage_type)
        montages = [montage/montage.max() for montage in montages]
        return spectrogram( montages[0],
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

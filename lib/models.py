import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

class Baseline(BaseEstimator, ClassifierMixin):
    """Baseline model to predict seizure events. It's predicted as a seizure when it's higher than global spectrum"""
    def __init__(self, width_window):
        self.fs = 200
        self.coef = 0.75
        self.width_window = width_window
#         self.deltaT = self.width_window - self.width_window*200//8/200 # Why 200 and 8 ?
        self.all_montages_predictions = None
        self.combined_predictions = None

        df_model = joblib.load('df_model.dump')

    def fit(self, t, f, spectrograms_db_list):
        """
        Makes the prediction by comparison to the global spectrum.
        : param spectrograms_db_list: list of 3-elements tuples : (frequencies, times, spectrogram in dB). There is one tuple per montage.
        """
        self.all_montages_predictions = []
        self.t = t
        deltaT = t[1] - t[0]
        # Each montage has its own prediction. Seizure sections are predicted as 1
        for id_montage, Sxx_db in enumerate(spectrograms_db_list):
            spectrum_global = Sxx_db.sum(axis=1)/max(t) # Compute global spectrum for each montage
            self.all_montages_predictions.append([sum(sample > spectrum_global) > self.coef*len(sample)
                                                 for sample in Sxx_db.T/deltaT])

        # Combine predictions : Compute for each section on how many montages seizure was seen
        self.combined_predictions = np.array(self.all_montages_predictions).sum(axis=0)

    def predict(self):
        """Read predictions made during *fit* method"""
#         return self.deltaT, self.combined_annotation.astype(bool)
        return self.t, self.combined_predictions.astype(bool)


class PostProcessing(BaseEstimator, ClassifierMixin):
    """Postprocessing eliminates isolated predictions and merges close ones"""
    def __init__(self, minimum_seizure_duration=15, minimum_interseizure_gap=30):
        self.minimum_seizure_duration = minimum_seizure_duration
        self.minimum_interseizure_gap = minimum_interseizure_gap

    def fit(self, t, predictions):
        self.deltaT = t[1] - t[0]
        self.predictions = predictions

    def predict(self):
        """Performs postprocessing"""
        # Remove isolated predictions
        # Labellization of each event
        D = np.diff(np.insert(self.predictions,0,False))
        events_ids = np.cumsum(D)
        events_ids = events_ids - min(events_ids)
        labels = np.unique(events_ids)
        # Compute duration of events
        events_sizes,_ = np.histogram(events_ids,bins=len(labels))
        events_durations = events_sizes*self.deltaT
        # Correct predictions
        new_predictions = self.predictions[:]
        for lab in labels:
            if events_durations[lab] < self.minimum_seizure_duration:
                new_predictions[events_ids==lab] = False

        # Merge predictions close to each other
        # Relabellization of each seizure event
        D = np.diff(np.insert(new_predictions,0,False))
        events_ids = np.cumsum(D)
        events_ids = events_ids - min(events_ids)
        labels = np.unique(events_ids)
        # Recompute duration of events
        events_sizes,_ = np.histogram(events_ids,bins=len(labels))
        events_durations = events_sizes*self.deltaT
        # Correct predictions
        for lab in labels[1:-1]:
            if events_durations[lab] < self.minimum_interseizure_gap:
                new_predictions[events_ids==lab] = True

        return new_predictions

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import edf_loader, find_edf, annotations_reader, section_seiz
from transformers import MontageExtractor

class EEGDataset(Dataset):
    def __init__(self, folder, montage_type=1, use_masks=False):
        self.files = find_edf(folder)
        self.win_duration = 8
        self.M = MontageExtractor()
        self.montage_type = montage_type
        self.use_masks = use_masks

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):
        filepath = self.files[i]
        raw = edf_loader(filepath) # Load raw data
        raw.info['montage_type'] = self.montage_type
        fs = raw.info['sfreq']
        nperseg = int(fs*self.win_duration) # Compute the length of segments
        noverlap = nperseg//8 # Overlapping on 1/8 of the window
        delta = nperseg - noverlap # Compute the distance between two windows
        print(raw.n_times)
        n_segments = len(np.arange(nperseg/2, raw.n_times - nperseg/2 + 1, delta)) # Compute the number of segments
        self.M.fit_transform(raw) # Load montages

        # Extract segments
        segments = np.array([self.M.montages[:, i*delta:(i+1)*delta] for i in range(n_segments-1)]) # Ignore the last segment (truncated)
        segments = np.expand_dims(segments, 0)
        output = (torch.from_numpy(segments), )

        if self.use_masks:
            # Extract labels
            ann = annotations_reader(filepath, self.montage_type)['tse_bi']
            labels = np.array([section_seiz((i*delta/fs, (i+1)*delta/fs), ann) for i in range(n_segments-1)])
            output = output + (torch.from_numpy(labels).long(), )

        return output

import os
import pandas as pd
import numpy as np

FIRST_ELEC = np.array(['FP1', 'F7', 'T3', 'T5', 'FP2', 'F8', 'T4', 'T6', 'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'FP1',
             'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4'])
SECOND_ELEC = np.array(['F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6', 'O2', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2', 'F3', 
              'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2'])

def combiner(raw, montage_type=1):
    """ 
    Combine reference channels to make montages
    ; param raw: mne.Raw
    : param montage_type: int, value in {1,2,3} corresponding to the used montage for the acquisition.

    : returns : tuple (list, list), montages and their names. A montage is a difference of two reference electrodes.
    """
    # Extract temporal data
    data = raw.get_data()

    # Get important channels id (FP1, FP2, ...)
    df_names = pd.read_csv(os.path.join('../dataset/_DOCS/montage_names'), sep=' ', header=None)
    df_names.columns = ['ch_id']
    # Relate each channel id with the corresponding line in data
    channels = {}
    for id in df_names['ch_id']:
        for i, name in enumerate(raw.ch_names):
            if ' '+id in name:
                channels[id] = i
                break
    # Make montages (difference of two reference electrodes)
    if montage_type == 3:
        selector = list(range(8))+list(range(9,12))+list(range(13,22)) # Remove two montages if montage_type==3
    else:
        selector = list(range(22))
    montages = list(map(lambda elec1, elec2: data[channels[elec1],:] - data[channels[elec2],:], 
                    FIRST_ELEC[selector], SECOND_ELEC[selector]))
    montages_names = list(map(lambda elec1, elec2: elec1+'-'+elec2, 
                          FIRST_ELEC[selector], SECOND_ELEC[selector]))

    return montages, montages_names


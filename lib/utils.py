import os
import numpy as np
import pandas as pd
import mne
import sys
sys.path.append('../read_labels/v1.0.0/src')
import sys_tools.nedc_cmdl_parser as ncp
import sys_tools.nedc_file_tools as nft
import sys_tools.nedc_ann_tools as nat

FIRST_ELEC = np.array(['FP1', 'F7', 'T3', 'T5', 'FP2', 'F8', 'T4', 'T6', 'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'FP1',
             'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4'])
SECOND_ELEC = np.array(['F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6', 'O2', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2', 'F3',
              'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2'])


def find_folders(rootpath):
    """ Find all folders with eeg data """
    folders_list = []
    for path, dirs, files in os.walk(rootpath):
        for file in files:
            if file.endswith('.edf'):
                folders_list.append(path)
                break
    return folders_list


def edf_loader(folder):
    """ Read the folder to find a .edf file, and returns the associated mne.Raw object """
    for file in os.listdir(folder):
        if file.endswith('edf'):
            filepath = os.path.join(folder, file)
            break
    raw = mne.io.read_raw_edf(filepath)
    raw.load_data()
    return raw


def combiner(raw, montage_type=1):
    """
    Combine reference channels to make montages
    : param raw: mne.Raw
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

def annotations_reader(folder, montage_type=1):
    """
    Read the folder and look for .tse, .tse_bi, .lbl, .lbl_bi files, abd returns the dictionary of annotations
    : param folder: str, folder where the annotations are located
    : returns annot_dict: dict, keys are the file format, values are the annotations
    """
    n_montages = 22
    if montage_type==3:
        n_montages = 20

    # List annotations files
    filesList = [file for file in os.listdir(folder)
                      if os.path.splitext(file)[-1] in ['.tse', '.tse_bi', '.lbl', '.lbl_bi']]
    # Build dictionary aof annotations
    annot_dict = {}
    for file in filesList:
        ann = nat.Annotations()
        status = ann.load(nft.get_fullpath(os.path.join(folder, file)))
        ext = os.path.splitext(file)[-1][1:]
        if 'lbl' in ext:
            annot_dict[ext] = [ann.get(0,0,i) for i in range(n_montages)]
        else:
            annot_dict[ext] = ann.get(0,0)

    return annot_dict


def section_seiz(section, annotations):
    """
    Tells if the given section is labelled as a seizure

    :param section: tuple, (start, stop)
    :param annotations: list, annotations coming from annotation_reader(_)['tse_bi']
    """
    start, stop = section
    for annot in annotations:
        if start>=annot[0] and stop<=annot[1]:
            return 'seiz' in annot[2].keys()

def find_breakpoints(folder, montage_type=1):
    dict_annot = annotations_reader(folder, montage_type=montage_type)
    # .lbl_bi files
    breakpoints_lbl = set([dict_annot['lbl_bi'][0][0][0]]) # Initialize breakpoints with the first timestamp (0.0)
    for channel_annot in dict_annot['lbl_bi']:
        for section in channel_annot:
            breakpoints_lbl.add(section[1]) # Add breakpoint
    # tse_bi files
    breakpoints_tse = set([dict_annot['tse_bi'][0][0]]) # Initialize breakpoints with the first timestamp (0.0)
    for section in dict_annot['tse_bi']:
        breakpoints_tse.add(section[1]) # Add breakpoint
        
    return breakpoints_lbl, breakpoints_tse



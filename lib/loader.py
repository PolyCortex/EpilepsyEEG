import os
import mne
import sys
sys.path.append('../read_labels/v1.0.0/src')
import sys_tools.nedc_cmdl_parser as ncp
import sys_tools.nedc_file_tools as nft
import sys_tools.nedc_ann_tools as nat

def edf_loader(folder):
    """ Read the folder to find a .edf file, and returns the associated mne.Raw object """
    for file in os.listdir(folder):
        if file.endswith('edf'):
            filepath = os.path.join(folder, file)
            break
    raw = mne.io.read_raw_edf(filepath)
    raw.load_data()
    return raw

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


# there is a 6-8 second lag for BOLD signal

import os
import numpy as np

# data.pymvpa.org/datasets/haxby2001
def load_haxby_data(datapath, sub, mask=None):
    # input arguments:
    # datapath (string): path to the root directory
    # sub (string): subject ID (e.g. subj1, subj2, etc)
    # output:
    # maskeddata (numpy array): samples x voxels data matrix
    # fmrilabel (pandas dataframe): length samples
    import nibabel as nib
    import pandas as pd
    import numpy as np

    fmriobj = nib.load(os.path.join(datapath, sub, 'train.nii.gz'))
    fmridata, fmriheader = fmriobj.get_data(), fmriobj.header
    fmridata = np.rollaxis(fmridata, -1)
    # shift last axis to the first
    fmrilabel = pd.read_table(os.path.join(datapath, sub, 'labels.txt'), delim_whitespace=True)
    if mask is not None:
        maskobj = nib.load(os.path.join(datapath, sub, mask + '.nii.gz'))
        maskdata, maskheader = maskobj.get_data(), maskobj.header
        maskeddata = fmridata[:, maskdata > 0]  # timepoints axis 0, voxels axis 1
        # need to figure out how to mask features back to original geometry
        print maskeddata.shape
    else:
        maskeddata = fmridata

    return maskeddata, fmrilabel[fmrilabel.chunks != 11]  # not loading the testing run that we've set aside

### Setup working directory & load files
# the code below assumes that the folder 'haxby2001-188B' is located in your current directory
cwd = os.getcwd()+'/haxby2001-188B'

# Cruddy way of doing this, should condense this code
# This will load up the data into sub1, sub2, ..., sub5 with labels and features in a matrix
# then it will print out the dimensions to confirm that they are correct
features = np.array([]); labels   = np.array([])
features, labels = load_haxby_data(cwd, 'subj1', 'mask4_vt'); sub1 = np.hstack((labels, features))
features, labels = load_haxby_data(cwd, 'subj2', 'mask4_vt'); sub2 = np.hstack((labels, features))
features, labels = load_haxby_data(cwd, 'subj3', 'mask4_vt'); sub3 = np.hstack((labels, features))
features, labels = load_haxby_data(cwd, 'subj4', 'mask4_vt'); sub4 = np.hstack((labels, features))
features, labels = load_haxby_data(cwd, 'subj5', 'mask4_vt'); sub5 = np.hstack((labels, features))
print "Subject 1:", sub1.shape, "\nSubject 2:", sub2.shape, "\nSubject 3:", sub3.shape, "\nSubject 4:", sub4.shape, "\nSubject 5:", sub5.shape

# this will work on the last data file uploaded (sub5)
# still getting error messages
label_names = labels['labels']
label_chunks= labels['chunks']

for i in range(len(label_names)):
    # Initialize labels
    if i==0: cur_label = label_names[0]
    else   : cur_label = label_names[i]
    
    # Append rows to temporary array
    # Let A be a temporary array with the feature values for that chunk
    # Let B be the final array with the mean values
    A = np.array([])
    B = np.array([])
    if label_names[i+1]!=cur_label:
        A = np.hstack((label_chunks[i], np.mean(A, axis=0)))
        A = np.hstack((cur_label, A))
        B = np.vstack((B, A))
    else: 
        A = np.vstack((A, features[i]))

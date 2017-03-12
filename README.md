### PSYCH 188B

##### Machine Learning for fMRI Data
Get subject neuroimaging data [here](http://data.pymvpa.org/datasets/haxby2001/)!

```python
import os
import numpy as np
import nibabel as nb
import pandas as pd
```

```python
def load_haxby_data(datapath, sub, mask=None):
    # input arguments:
    # datapath (string): path to the root directory
    # sub (string): subject ID (e.g. subj1, subj2, etc)
    # output:
    # maskeddata (numpy array): samples x voxels data matrix
    # fmrilabel (pandas dataframe): length samples
    import os
    import nibabel as nib
    import pandas as pd
    import numpy as np
    
    # generate variable containing path to subject filescwd = os.getcwd()+'/haxby2001-188B'

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
        # print maskeddata.shape
    else:
        maskeddata = fmridata
    
    return maskeddata, fmrilabel[fmrilabel.chunks != 11]
```    

```python
def split2chunk(labels, features, chunk):
    # @param  labels    : n x 2 dataframe
    # @param  features  : n x m array
    # @param  chunk     : integer - chunk of code to be extracted
    # @return dat_chunk : p x m array, where p is the number of samples in the chunk
    #
    # Extracts a specific 'chunk' of data 

    data      = np.hstack((labels, features))  # group labels and features into one matrix, ndarray
    dat_chunk = data[labels['chunks'].values==chunk]  # extract specific 'chunk' of data

    return dat_chunk
```

```python
def split2cat(category, dat_chunk):
    # @param category    : current category to extract
    # @param data_chunks : r x m array
    # @return split_data : data split by chunk and category
    split_data = dat_chunk[dat_chunk[:,0]==category]
    
    return split_data
```

```python
def get_rest(labels, features):
    # @param labels   : n x 2 dataframe
    # @param features : n x m array
    # @return rest    : m x 1 array
    # Takes in labels & features, then returns numpy array with average value per
    # column of rest data.     
    data     = np.hstack((labels, features))  # put labels and features into one matrix
    all_rest = data[labels['labels']=='rest'] # get rest data
    rest     = np.mean(data[:, 2:], axis=0)   # get average feature values and remove rest label
    
    return rest
```

```python
def average_trials(features, labels):
    rest = get_rest(labels, features)                # Get averaged rest data
    labs = labels[labels['labels'].values!='rest']   # Remove rest labels, labs is a pandas dataframe!
    feat = features[labels['labels'].values!='rest'] # remove rest features
    chunks = range(np.max(labels['chunks'])+1)       # List of chunks
    categories = np.unique(labs['labels'])           # Get categories, array of categories
    
    # Initialize final array with dimensions:
    #   Number of rows = number of chunks * number of categories
    #   Number of columns = number of features + 2
    #averaged = np.empty([len(chunks)*len(categories),len(feat.T)+2])
    
    averaged = []
    
    for chunk in chunks:
        # split2chunk returns labels + features in one array for 'chunk'
        # dat_chunk is ndarray
        dat_chunk = split2chunk(labs, feat, chunk) # Split each chunk by category
        
        for cat in categories:
            # split2cat returns labels + features in array for 'chunk' and 'cat'
            split_data = split2cat(cat, dat_chunk)

            new_feats  = split_data[:, 2:]                  # Extract features from labels
            mean_feats = np.mean(new_feats, axis=0)         # Average columns down to 1 row
            feats      = mean_feats-rest                    # Subtract averaged rest values
            new_lab    = np.hstack(np.hstack((cat, chunk))) # Generate new label
            new_data   = np.hstack((new_lab, feats))        # Generate 1-D array containing all data
                                                            #   Labels in indeces 0 and 1
            averaged.append(new_data) # Add current row to final data list
    
    # Convert 'averaged' list into numpy array and return
    return np.array(averaged)
```

```python
cwd = os.getcwd()+'/haxby2001-188B' # set working directory

# MAKE A LIST OF SUBJECTS
# ITERATE THROUGH LIST
# LOAD SUBJECT, APPEND TO LIST
# VERIFY CONTENTS OF LIST

# Initialize subjects list
subjects = []

# Iterate through files in directory
files = os.listdir(cwd)
for f in files:
    features = np.array([]); labels = np.array([])
    features, labels = load_haxby_data(cwd, 'subj1', 'mask4_vt')
    sub = average_trials(features, labels)
    subjects.append(sub)
print subjects, subjects[0][0, 0], subjects[0][0, 1], subjects[0][0, 2], type(subjects[0])
```

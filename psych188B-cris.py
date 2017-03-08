# there is a 6-8 second lag for BOLD signal

import os
import numpy as np

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

def average_trials(features, labels):
    rest = get_rest(labels, features)                # Get averaged rest data
    labs = labels[labels['labels'].values!='rest']   # Remove rest labels, labs is a pandas dataframe!
    feat = features[labels['labels'].values!='rest'] # remove rest features
    chunks = range(np.max(labels['chunks'])+1)       # List of chunks
    categories = np.unique(labs['labels'])           # Get categories, array of categories
    
    averaged = [] # Initialized output matrix
    
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

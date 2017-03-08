def split2chunk(labels, features, chunk):
    # @param  labels   : n x 2 dataframe
    # @param  features : n x m array
    # @param  chunk    : integer - chunk of code to be extracted
    # @return data     : p x m array, where p is the number of samples in the chunk
    #
    # Extract a specific chunk of data 

    data = np.hstack((labels, features))  # group labels and features into one matrix
    data = data[labels['chunks']==chunk]  # extract specific 'chunk' of data

    return data

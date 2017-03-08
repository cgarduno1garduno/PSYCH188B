def get_rest(labels, features):
    # @param labels   : n x 2 dataframe
    # @param features : n x m array
    # @return data    : m x 1 array
    # 
    # This function takes in labels and features and returns a numpy array with the
    # average value of each column of the rest data. To subtract this rest data from
    # the remaining data, use the transpose of data. 
    
    data = np.hstack((labels, features))  # put labels and features into one matrix
    data = data[labels['labels']=='rest'] # get rest data
    data = np.mean(data[:, 2:], axis=0)   # get average feature values and remove rest label
    
    return data

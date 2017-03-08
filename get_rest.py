################################################################################
# @author Cristopher Garduno Luna
# 
# Average the 'rest' values by collapsing along the columns such that the output
# is an m x 1 array where m is the number of features
################################################################################

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

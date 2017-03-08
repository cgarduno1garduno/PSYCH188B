################################################################################
# @author Cristopher Garduno Luna
# 
# Extract the 'chunk' of data specific by the chunk param
################################################################################

def split2chunk(labels, features, chunk):
    # @param  labels    : n x 2 dataframe
    # @param  features  : n x m array
    # @param  chunk     : integer - chunk of code to be extracted
    # @return dat_chunk : p x m array, where p is the number of samples in the chunk
    #
    # Extracts a specific 'chunk' of data 

    data      = np.hstack((labels, features))  # group labels and features into one matrix
    dat_chunk = data[labels['chunks']==chunk]  # extract specific 'chunk' of data

    return dat_chunk

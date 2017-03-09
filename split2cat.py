################################################################################
# @author Cristopher Garduno Luna
# 
# Split chunked data by categories and return a specific category specified by
# the 'category' param
################################################################################

def split2cat(category, dat_chunk):
    # @param category    : current category to extract
    # @param data_chunks : 1 x m array where m is the number of chunk matrices
    # @return split_data : data split by chunk and category
    
    split_data = dat_chunk[dat_chunk==category]
    return split_data

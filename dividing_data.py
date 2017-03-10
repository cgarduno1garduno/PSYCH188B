################################################################################
# @author Bryce Wong
# 
# Convert the data into a usable form for a single subject
################################################################################

def divide_data(labels, features):
	# @ param labels are the labels for one subject
	# @ param features are the features for one subject
	#subject = np.hstack((labels, features)) #combine into one matrix 
	
	# @ output is the cleaned matrix for the subject 
	# This final output is a matrix where the first two columns are the label (ie, "y") and the chunk number 
	# The rest of the columns are feature values (ie, "X")
	# Each row in the matrix corresponds to a single trial in a chunk that has been averaged from all timepoints for that trial
	# A trial is basically one attempt at a categorization
	
	# Intialize a temp variable below
	big_mat = []
	
	import numpy as np

	# Create a list just to iterate through
	chunks = list(range(0,11)) # The number of possible chunks
	
	for chunk in chunks: # Repeat for every chunk
		# Pull out chunk based on the current chunk number being looked at
		current_chunk = split2chunk(labels,features,chunk)
		#current_chunk = subject[subject[:,1] == chunk]
	
		# Create a list of the categories to iterate through 
		categories = ['face','cat','bottle','shoe','scissor','house','chair','scramblepix']	
	
		for category in categories: # Repeat for every category within each chunk
			# For each chunk, pull out subset based on the current category label being looked out
			current_cat = split2cat(category,current_chunk)
			#current_cat = current_chunk[current_chunk[:,0]==category]

			# Transform each category matrix into a numpy array
    			current_cat = np.array(current_cat)
		
    			# Average each category numpy array into a single row
    			current_cat = np.mean(current_cat, axis=0)
			
			# Add to big matrix holding averaged trials for each category from each chunk
			big_mat.append(current_cat)
		
	# Convert final output to numpy array
	big_mat =np.array(big_mat)
	
	# Subtract the averaged rest values from every trial
	avg_rest = get_rest(labels,features) # Get vector of average rest values
	big_mat_features = big_mat[:,2:] # Get only the features from our final cleaned data
	big_mat_labels = big_mat[:,0:2] # Also store only the labels to put together with final features later
	final_features = big_mat_features - avg_rest
	
	# Put final labels and features together
	final_data = np.hstack((big_mat_labels, final_features))
	
	return final_data
	
 

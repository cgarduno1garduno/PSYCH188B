import numpy as np

#create category matrices
faces = []
cats = []
bottles = []
shoes = []
scissors = []
houses = []
chairs = []
controls = []

#divide_data checks the first index of each new array and adds the array to the appropriate category matrix
def divide_data(array):
	if row[0] == 'face'
		faces.append(array)
	if else row[0] == 'cat'
		cats.append(array)
		.
		.
		.
		.
		.
		.
	return

#read through each row in 'sub1' and add to appropro matrix
np.apply_along_axis(divide_data, axis=1, arr=sub1)

#transform each matrix into a numpy array
faces = np.array(faces)
.
.
.
.
.
.
.

#average each numpy matrix into a single row
faces = np.mean(faces, axis=0)
.
.
.
.
.
.
.

#now we have 8 numpy arrays, each representing one trial for the subject. put together into one matrix
np.concatenate((faces,.......), axis=0)

       

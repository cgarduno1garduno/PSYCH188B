### PSYCH 188B

##### Machine Learning for fMRI Data
Get subject neuroimaging data [here](http://data.pymvpa.org/datasets/haxby2001/)!

First, let's set up some modules. 
```python
import os
import numpy as np
import nibabel as nib
import pandas as pd
import sklearn.preprocessing as preproc

from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold

# Modules for plotting
%matplotlib inline
import matplotlib.pyplot as plt
```

Then we define some functions.

`load_haxby_data`: Load fMRI features and labels
```python
def load_haxby_data(datapath, sub, mask=None):
    # input arguments:
    # datapath (string): path to the root directory
    # sub (string): subject ID (e.g. subj1, subj2, etc)
    # output:
    # maskeddata (numpy array): samples x voxels data matrix
    # fmrilabel (pandas dataframe): length samples
    
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

`split2chunk`: Split data into chunks
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

`split2cat`: Split chunked data into categories
```python
def split2cat(category, dat_chunk):
    # @param category    : current category to extract
    # @param data_chunks : r x m array
    # @return split_data : data split by chunk and category
    split_data = dat_chunk[dat_chunk[:,0]==category]
    
    return split_data
```

`get_rest`: Return averaged rest data
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
`average_trials`: For each chunk of data, average the rows for each category and return array of averaged data
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

We've defined our functions and now let's get to work. Please ensure that `/haxby2001-188B` is a subfolder in your directory and that the subject files in are named `subj1`, `subj2`, etc. 
```python
cwd = os.getcwd()+'/haxby2001-188B' # set working directory

# Initialize subjects list
subjects = []

# Iterate through files in directory and average the data across chunks by category for each subject
files = os.listdir(cwd)
for f in files:
    if f.startswith('sub'):
        features = np.array([]); labels = np.array([])
        features, labels = load_haxby_data(cwd, f, 'mask4_vt')
        sub = average_trials(features, labels)
        subjects.append(sub)
```

Now, we will make functions for our models. Here is the function for logistic regression.
```python
def logistic_regression(labels,features):
    #@ param labels are the labels of one subject's cleaned/preprocessed dataset
    # #@ param features are the features of one subject's cleaned/preprocessed dataset
    
    #split into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
    
    # Import our scikit-learn function
    from sklearn.linear_model import LogisticRegression
  
    # Create our logistic regression model
    logreg = LogisticRegression()
  
    # Train our model on our data
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.decision_function(X_test)
  
    # Test our model and score it
    score1 = logreg.score(X_test, y_test)
    print("The logistic regression model has an accuracy score of:", score1)
    
    # Scores the average of ten tests with the model
    score1_avg = cross_val_score(logreg, features, labels, scoring='accuracy', cv=10)
    print("The logistic regression model has an average accuracy score of:", score1_avg.mean())
  
    return
```
Here is the function for SVM with rbf kernel.
```python
def SVM_rbf_kernel(labels, features):
    #@ param labels are the labels for one subject's cleaned/preprocessed dataset
    #@ param features are the features for one subject's cleaned/preprocessed dataset
  
    # Import our scikit-learn stuff
    from sklearn.svm import SVC
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import StratifiedKFold
  
    #split into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
  
    # Set a range of possible values for our C parameter and gamma parameter to iterate through
    possible_C = np.logspace(-3, 9, 13)
    possible_gamma = np.logspace(-7, 4, 12)
  
    # Fill a grid with our possibe combinations of C and gamma values
    param_grid = dict(gamma=possible_gamma, C=possible_C)
  
    # Create our cross-validation function
    cv = StratifiedKFold(y_train, 5) # Uses our labels as our y-vector, makes 5 folds
  
    # Create our svm model
    svc = SVC()
  
    # Cross-validate our parameters in our grid to find best combination of the params
    grid = GridSearchCV(svc, param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)
    #print(grid.best_params_)

    # Create our svm model with rbf kernels using our optimal params and score it
    # Note that the ** syntax may not work in Python <3.5
    svc_rbf = SVC(**grid.best_params_, kernel="rbf", decision_function_shape = 'ovr')
    svc_rbf.fit(X_train, y_train)
    y_pred_svc = svc_rbf.decision_function(X_test)
    score2 = svc_rbf.score(X_test, y_test)
    
    print("The SVC model with RBF kernals has an accuracy score of:", score2)
    
    # Scores the average of ten tests with the model
    score2_avg = cross_val_score(svc_rbf, features, labels, scoring='accuracy', cv=10)
    print("The SVM model with RBF kernals has an average score of:", score2_avg.mean())
    
    # Now we try our svm model with linear kernels using our optimal params and score it
    svc_linear = SVC(**grid.best_params_, kernel="linear", decision_function_shape = 'ovr')
    svc_linear.fit(X_train, y_train)
    y_pred_svc = svc_linear.decision_function(X_test)
    score3 = svc_linear.score(X_test, y_test)
    
    print("The SVM model with linear kernals has an accuracy score of:", score3)

    # Scores the average of ten tests with the model
    score3_avg = cross_val_score(svc_rbf, features, labels, scoring='accuracy', cv=10)
    print("The SVM model with linear kernals has an average score of:", score3_avg.mean())
  
    return
```
Here is the function for keras neural network.
```python
def neural_network(labels,features):
    # @ param labels takes in the labels of one subject's cleaned data set
    # @ param features takes in the features of one subject's cleaned data set
  
    #split into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)

    # specify the hidden layer sizes: --> this is up to us to decide specifications
    layer_sizes = [10, 5]

    # Keras uses the Sequential model for linear stacking of layers.
    # That is, creating a neural network is as easy as (later) defining the layers!
    from keras.models import Sequential
    model = Sequential()
  
    # Use the dropout regularization method
    from keras.layers import Dropout

    # Now that we have the model, let's add some layers:
    from keras.layers.core import Dense, Activation
    # Everything we've talked about in class so far is referred to in 
    # Keras as a "dense" connection between layers, where every input 
    # unit connects to a unit in the next layer

    # First a fully-connected (Dense) hidden layer with appropriate input
    # dimension, 10 outputs, and ReLU activation
    #THIS IS THE INPUT LAYER
    model.add(Dense(
        input_dim=X_train.shape[1], output_dim=layer_sizes[0]
    ))
    model.add(Activation('relu'))

    #ADD DROPOUT --> MUST DECIDE PERCENTAGE OF INPUT UNITS TO DROPOUT
    model.add(Dropout(.2))

    # Now our second hidden layer with 10 inputs (from the first
    # hidden layer) and 5 outputs. Also with ReLU activation
    #THIS IS HIDDEN LAYER
    model.add(Dense(
        input_dim=layer_sizes[0], output_dim=layer_sizes[1]
    ))
    model.add(Activation('relu'))

    #ADD DROPOUT
    model.add(Dropout(.2))

    # Finally, add a readout layer, mapping the 5 hidden units
    # to two output units using the softmax function
    #THIS IS OUR OUTPUT LAYER
    model.add(Dense(output_dim=np.unique(y_train).shape[0], init='uniform'))
    model.add(Activation('softmax'))

    # Next we let the network know how to learn
    from keras.optimizers import SGD
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # Before we can fit the network, we have to one-hot vectorize our response.
    # Fortunately, there is a keras method for that.
    from keras.utils.np_utils import to_categorical
    # for each of our 8 categories, map an output
    # Must first convert each category string to consistent ints    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    y_train_vectorized = to_categorical(encoded_y_train)
    
    #print out shape
    #y_train_vectorized.shape
  
    #remember that the bigger the nb_epoch the better the fit (so go bigger than 50)
    model.fit(X_train, y_train_vectorized, nb_epoch=1000, batch_size=20, verbose = 0)

    #now our neural network works like a scikit-learn classifier
    proba = model.predict_proba(X_test, batch_size=32)

    # Print the accuracy:
    from sklearn.metrics import accuracy_score
    classes = np.argmax(proba, axis=1)
    print("The neural network model has an accuracy score of:", accuracy_score(y_test, classes))
  
    return
```
Running our models.
```python
clf1 = logistic_regression(subjects[0][:,0],subjects[0][:,2:])
clf2 = SVM_rbf_kernel(subjects[0][:,0],subjects[0][:,2:])
clf3 = neural_network(subjects[0][:,0],subjects[0][:,2:])
```
To visualize how well our models did, we plotted ROC curves.
```python
# First, create a set of predicted y-values
pred_y1 = clf1.predict(X_test)
pred_y2 = clf2.predict(X_test)
pred_y3 = clf3.predict(X_test)

# plot ROC for logistic regression
from sklearn import metrics
fpr0, tpr0, thresholds = metrics.roc_curve(y_test, pred_y1)
roc_auc0 = metrics.roc_auc_score(y_test, pred_y1)

plt.figure()
lw = 2
plt.plot(fpr0, tpr0, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc0)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: SVM with RBF Kernel')
plt.legend(loc="lower right")
plt.show()

# plot ROC for SVM with RBF kernel
from sklearn import metrics
fpr0, tpr0, thresholds = metrics.roc_curve(y_test, pred_y2)
roc_auc0 = metrics.roc_auc_score(y_test, pred_y2)

plt.figure()
lw = 2
plt.plot(fpr0, tpr0, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc0)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: SVM with RBF Kernel')
plt.legend(loc="lower right")
plt.show()

# plot ROC for neural network
from sklearn import metrics
fpr0, tpr0, thresholds = metrics.roc_curve(y_test, pred_y3)
roc_auc0 = metrics.roc_auc_score(y_test, pred_y3)

plt.figure()
lw = 2
plt.plot(fpr0, tpr0, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc0)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: SVM with RBF Kernel')
plt.legend(loc="lower right")
plt.show()
```

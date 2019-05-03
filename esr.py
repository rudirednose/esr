'''
Dataset location and description is provided in the link listed https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
# Tasks
1.	Split the data set in train and test
    a.	First 70% as training set
    b.	Rest of 30% as testing set
2.	Use first 178 columns as features (X1 to X178) to predict the 179th column (y)
3.	Develop a classification model to classify seizure (1) vs non-seizure (2, 3, 4, 5)
4.	Develop a classification model to do multiclass classification 
'''

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pcm import plot_confusion_matrix


'''
reading the data from https://archive.ics.uci.edu/ml/machine-learning-databases/00388/
into a pandas data frame
'''

df=pd.read_csv('data.csv')


'''
split data into 70% train and 30% test data without shuffle.
do this for the input matrix the original and binary output vector
'''
Xtrain, Xtest, Ytrain, Ytest, Ybtrain, Ybtest = train_test_split(
    df.drop(['y','Unnamed: 0'], axis=1),          # input matrx
    df['y'],                                      # output vector
    df['y'].apply(lambda x: 1 if x == 1 else 0),  # binary output vector
    test_size=0.3,shuffle=False)

'''
create, train, apply and evaluate the models binary model to classify seizure (1) vs non-seizure (2,3,4,5)
'''
bclasses = ['Seizer', 'No Seizer']
full_classes = ['Seizer', 'Tumor Area', 'Healthy Area','Eyes Closes','Eyes Open']

bmodel = KNeighborsClassifier(n_neighbors=5)
bmodel.fit(Xtrain,Ybtrain)
Ybmodel = bmodel.predict(Xtest)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(Xtrain,Ytrain)
Ymodel = model.predict(Xtest)

# create standard classification report
print(classification_report(Ybtest, Ybmodel, target_names=bclasses))

# create the standard plot
ax1 = plot_confusion_matrix(Ybtest, Ybmodel, bclasses,
                           title='Confusion matrix, without normalization')

print(classification_report(Ytest, Ymodel, target_names=full_classes))

ax2 = plot_confusion_matrix(Ytest, Ymodel, classes=full_classes,
                            title='Confusion matrix, without normalization')


plt.show()

exit(0)
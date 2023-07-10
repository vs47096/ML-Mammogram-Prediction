import numpy
import pandas
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

## Set to ensure same split done by train_test_split as random_state in it internally uses numpy`s random generator
numpy.random.seed(1234)

## Specified array of values which should be treated as missing or n/a values
missing_values = ['?']
## Specified names of columns to be used as features for our model
features = ['Age' , 'Shape',  'Margin',  'Density']

## Loaded data
## Here, na values list is passed to mark specified values as empty cells, so that can be removed by dropna method 
## and, headers which were missing in original file are specified by us in names parameter while loading data
data = pandas.read_csv('mammographic_masses.data', na_values=missing_values, names=['BI-RADS', 'Age' , 'Shape',  'Margin',  'Density', 'Severity'])

## Cleaning data
data.dropna(inplace=True)

## Creating array of feature (input) values & output values
data_features = data[features].values
data_output = data['Severity'].values

## Normalizing data for some of the algo which needs normalized data
## Algos like decision tree, random forest which works fine with un-normalized data as well, will remain unimpacted by normalization
## Creatin instance of pre-processors for normalization
## Pre-processor for KNN & neural network
scaler = preprocessing.StandardScaler()
## Pre-processor for naive bayes, because this algo doesn`t accept negative values
# scaler = preprocessing.MinMaxScaler()
## Normalizing data by feeding to pre-processor
data_features = scaler.fit_transform(data_features)

## Splitting data for trainign & test
(data_features_train,
 data_features_test,
 data_output_train,
 data_output_test) = train_test_split(data_features, data_output, train_size=0.75, random_state=1)


## Creating model instance & training model - START

# # Decision tree - BLOCK START
# # Declare model
# model = DecisionTreeClassifier()
# # Train model with training data
# model = model.fit(data_features_train, data_output_train)
# # Printing decision tree
# tree.plot_tree(model, feature_names=features)

# # Priniting decision tree as  a proper tree image
# from IPython.display import Image  
# from six import StringIO  
# import pydotplus
# dot_data = StringIO()  
# tree.export_graphviz(model, out_file=dot_data,  
#                          feature_names=features)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())
# # Decision tree - BLOCK END


# # Random forests - BLOCK START
# model = RandomForestClassifier(n_estimators=10)
# model = model.fit(data_features_train, data_output_train)
# # Random forests - BLOCK END


# # KNN - BLOCK START
# KNN_k_optimized = 0
# score_optimized = 0
# # Running loop to find optimum value of K for which model is most accurate
# for n in range(1, 50):
#     model = KNeighborsClassifier(n_neighbors=n)
#     scores = cross_val_score(model, data_features_train, data_output_train, cv=10)
#     if scores.mean() > score_optimized:
#         score_optimized = scores.mean()
#         KNN_k_optimized = n
# model = KNeighborsClassifier(n_neighbors=KNN_k_optimized)
# model.fit(data_features_train, data_output_train)
# # KNN - BLOCK END


# # Naive Bayes - BLOCK START
# model = MultinomialNB()
# model.fit(data_features_train, data_output_train)
# # Naive Bayes - BLOCK END


# # SVM - BLOCK START
# C=1.0
# # Possible values for kernel like linear, rbf, sigmoid, poly can be specified to find out which kernel gives max accuracy
# # For this problem, linear is giving maximum accuracy
# # Also, various values for C can be specified but for simplicity for just learning kept as 1.0. 
# # To understand about this hyperparameter, refer notes` SVM section
# model = svm.SVC(kernel='linear', C=C)
# model.fit(data_features_train, data_output_train)
# # SVM - BLOCK END


# # Logistic regression - BLOCK START
# model = LogisticRegression()
# model.fit(data_features_train, data_output_train)
# # Logistic regression - BLOCK START


# Neural network - keras - BLOCK START
def create_model():
    neural_network_model = Sequential()
    neural_network_model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    neural_network_model.add(Dropout(0.5))
    neural_network_model.add(Dense(6, activation='relu'))
    neural_network_model.add(Dropout(0.5))
    neural_network_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    neural_network_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    neural_network_model.fit(data_features_train, data_output_train, batch_size=32, epochs=10, verbose=0)
    
    return neural_network_model

model = KerasClassifier(build_fn=create_model,nb_epoch=100,verbose=0)
# Neural network - keras - BLOCK END

## Print model accuracy by simple score method of model
## !This line won`t work for neural network model, instead use K-fold!
## While running for neural network, comment below line
# model.score(data_features_test, data_output_test)

# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(model, data_features, data_output, cv=10)

# Print the accuracy for each fold:
print(scores)

# And the mean accuracy of all 10 folds:
print(scores.mean())

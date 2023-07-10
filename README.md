
## Problem Statement

* Using "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
* This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

    * BI-RADS assessment: 1 to 5 (ordinal)
    * Age: patient's age in years (integer)
    * Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
    * Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
    * Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
    * Severity: benign=0 or malignant=1 (binominal)

* BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.

* Apply several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=10).

## Model used details

| Model  | Accuracy (%) | Needed normalization |
| ------------- | ------------- |
| Decision Tree  | 73.37  | No  |
| Rainforest  | 75.81  | No  |
| KNN  | 78.31  | Yes  |
| Naive Bayes  | 78.55  | Yes  |
| SVM  | 79.87  | No  |
| Logistic Regression  | 80.77  | No  |
| Neural network - tensorflow - keras  | 80.24  |Yes  |

## Setup environment

- Open jupyter notebook
- Browse to location where file is placed
- Copy paste code from model.py in jupyter notebook
- Click run

## To run

- Install ananconda which will internally install python & jupyter
- Install libraries which are needed for running models in this code
- Download data from "https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass"
- Place "mammographic_masses.data" file from downloaded zip file to a folder where we want to run our code


## Code walkthrough

- Imported modules needed by our models in code to run
- Imported data, cleaned, normalized, split to train/test data
- Created various models using different algorithms
- Various models can be run by turn by turn commenting and running
- Printed accuracy that model gives us by using model`s score method
- Also, printed accuracy of model using K-fold which is more reliable


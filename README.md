# TensorFlow and TensorBoard with Evaluation



## Purpose

The purpose of this lab is twofold.  

1.   to review using `TensorFlow` for modeling and evaluation with neural networks
2.   to learn about [`TensorBoard`](https://www.tensorflow.org/tensorboard)

`TensorBoard` is `TensorFlow`'s visualization toolkit, so it is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. 

We'll be using the canonical [Titanic Data Set](https://www.kaggle.com/competitions/titanic/overview).

## The Titanic

### The Titanic and it's data



RMS Titanic was a British passenger liner built by Harland and Wolf and operated by the White Star Line. It sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton, England to New York City, USA.

Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 

Though there were about 2,224 passengers and crew members, we are given data of about 1,300 passengers. Out of these 1,300 passengers details, about 900 data is used for training purpose and remaining 400 is used for test purpose. The test data has had the survived column removed and we'll use neural networks to predict whether the passengers in the test data survived or not. Both training and test data are not perfectly clean as we'll see.

Below is a picture of the Titanic Museum in Belfast, Northern Ireland.


```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg", width=400, height=400)
```




<img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg" width="400" height="400"/>



### Data Dictionary

*   *Survival* : 0 = No, 1 = Yes
*   *Pclass* : A proxy for socio-economic status (SES)
  *   1st = Upper
  *   2nd = Middle
  *   3rd = Lower
*   *sibsp* : The number of siblings / spouses aboard the Titanic
  *   Sibling = brother, sister, stepbrother, stepsister
  *   Spouse = husband, wife (mistresses and fiancés were ignored)
*   *parch* : The # of parents / children aboard the Titanic
  *   Parent = mother, father
  *   Child = daughter, son, stepdaughter, stepson
  *   Some children travelled only with a nanny, therefore *parch*=0 for them.
*   *Ticket* : Ticket number
*   *Fare* : Passenger fare (British pounds)
*   *Cabin* : Cabin number embarked
*   *Embarked* : Port of Embarkation
  *   C = Cherbourg (now Cherbourg-en-Cotentin), France
  *   Q = Queenstown (now Cobh), Ireland
  *   S = Southampton, England
*   *Name*, *Sex*, *Age* (years) are all self-explanatory

## Libraries and the Data



### Importing libraries


```python
# Load the germane libraries

import pandas as pd
import numpy as np
import seaborn as sns 
from pandas._libs.tslibs import timestamps
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras 
from keras import models
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.wrappers.scikit_learn import KerasClassifier

# Load the TensorBoard notebook extension and related libraries
%load_ext tensorboard
import datetime
```

### Loading the data


```python
# Load the data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# We need to do this for when we mamke our predictions from the test data at the end
ids = test[['PassengerId']]
```

## EDA and Preprocessing

### Exploratory Data Analysis

It is your choice how much or how little EDA that you perform. But you should do enough EDA that you feel comfortable with the data and what you'll need to do to make it so that you can run a neural network on it.

It is prudent to investigate the attributes of the data frames, create visualizations, and perform data analysis.

### Preprocessing

Here are some suggestions:

*   Check to see if you have missing data in the train and test sets.
*   Combine the test and train sets into a dataframe called *combined* since it will make preprocessing more efficient
*   Use the median of the column to replace missing data for numeric values
*   Use the mode of the column to replace missing data for categorical values
*   Change *Pclass* from 1, 2, 3, to 1st, 2nd, 3rd
*   Create a new variable *Child*, where you use the *Age* column to assign those who are 18 years or older a 1, and those younger a 0
*   For *Sex*, let Male = 0 and Female = 1
*   For *Embarked*, Q=0, S=1, and S=2
*   The names of the passengers are not meaningful for the model, but their titles may be.
  * Use the **Code Chunk 1** below to help you with this.
*   Drop the irrelevant columns: *PassengerId*, *Name*, *Ticket*, *Cabin* and use `get_dummies`
  * Use the **Code Chunk 2** below to help you with this.
*  Resplit and scale the data.
  * Use the **Code Chunk 3** below to help you with this.

Of course, for the two code chunks you need to uncomment the germane lines of code.

**Code Chunk 1** Titles


```python
# Break up the string that has the title and names
#combined['Title'] = combined['Name'].str.split('.').str.get(0)  # output : 'Futrelle, Mrs'
#combined['Title'] = combined['Title'].str.split(',').str.get(1) # output : 'Mrs '
#combined['Title'] = combined['Title'].str.strip()               # output : 'Mrs'
#combined.groupby('Title').count()

# Replace the French titles with Enlgish
#french_titles = ['Don', 'Dona', 'Mme', 'Ms', 'Mra','Mlle']
#english_titles = ['Mr', 'Mrs','Mrs','Mrs','Mrs','Miss']
#for i in range(len(french_titles)):
#    for j in range(len(english_titles)):
#        if i == j:
#            combined['Title'] = combined['Title'].str.replace(french_titles[i],english_titles[j])

# Seperate the titles into "major" and "others", the latter would be, e.g., Reverend
#major_titles = ['Mr','Mrs','Miss','Master']
#combined['Title'] = combined['Title'].apply(lambda title: title if title in major_titles else 'Others')
```

**Code Chunk 2** Dropping and Dummies


```python
#Dropping the Irrelevant Columns
#combined.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace=True)

# Getting Dummy Variables and Dropping the Original Categorical Variables
#categorical_vars = combined[['Pclass','Sex','Embarked','Title','Child']] # Get Dummies of Categorical Variables
#dummies = pd.get_dummies(categorical_vars,drop_first=True)
#combined = combined.drop(['Pclass','Sex','Embarked','Title','Child'],axis=1)
#combined = pd.concat([combined, dummies],axis=1)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      


**Code Chunk 3** Resplitting and scalling the data


```python
# Separating the data back into train and test sets
#test = combined[combined['Survived'].isnull()].drop(['Survived'],axis=1)
#train = combined[combined['Survived'].notnull()]

# Training
#X_train = train.drop(['Survived'],1)
#y_train = train['Survived']

# Scaling
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#test = sc.fit_transform(test)
```

## Neural Network Model

### Building the model

#### Define the model as a pipeline

Let's use the data science pipeline for our neural network model.


```python
# It will help to define our model in terms of a pipeline
def build_classifier(optimizer):
    classifier = Sequential()
    # use classifer.add() to add layers
    # 
    # ... 
    #
    # use classifer.compile() as your last line of the definition; use loss='binary_crossentropy',metrics=['accuracy']
    return classifier

# Note: Do not use regularization methods or GridSearch. Those will be for next time!
```

#### Fitting the optimal model and evaluating with `TensorBoaard`

#### `TensorBoard`

`TensorBoard` is `TensorFlow`'s visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.

N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)


```python
# Clear out any prior log data.
!rm -rf logs
# Be careful not to run this command if already have trained your model and you want to use TensorBoard.

# Sets up a timestamped log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(log_dir)


# The callback function, which will be called in the fit()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

#### Fitting the optimal model and evaluating with `TensorBoaard`


```python
# Using KerasClassifier

# Replace the optimizer, batch_size, and epochs appropriately
classifier = KerasClassifier(build_fn = build_classifier,
                             optimizer='none',
                             batch_size='none',
                             epochs='none')

# Fit the model with the tensorboard_callback

# DELETE THESE TWO LINES OF CODE BELOW
X_train = []
y_train = []
# WE DID THIS JUST TO MAKE THE .ipynb HAPPY

classifier.fit(X_train,
               y_train,
               verbose=1,
               callbacks=[tensorboard_callback])


# Warning: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.
# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar).
```


```python
# Call TensorBoard
%tensorboard --logdir logs/fit
```

#### Results and Predictions


```python
# This will export your predictions to a .csv. Uncomment.

#preds = classifier.predict(test)
#results = ids.assign(Survived=preds)
#results['Survived'] = results['Survived'].apply(lambda row: 1 if row > 0.5 else 0)
#results.to_csv('titanic_submission.csv',index=False)
#results.head(20)
```

Continue to tweak your model until you are happy with the results based on model evaluation.

## Conclusion

Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.

We'll continue with this for the next lesson when we learn about model regularization.

TensorFlow and TensorBoard
# TensorFlow and TensorBoard with Evaluation

Purpose
The purpose of this lab is twofold.

to review using TensorFlow for modeling and evaluation with neural networks.
to learn about TensorBoard
TensorBoard is TensorFlow's visualization toolkit, so it is a dashboard that provides visualization and tooling that is needed for machine learning experimentation.

We'll be using the canonical Titanic Data Set.

The Titanic
The Titanic and it's data
The Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. Though there were about 2,224 passengers and crew members, we are given data of about 1,300 passengers. Out of these 1,300 passengers details, about 9000 data is used for training purpose and remaining 400 is used for test purpose. The test data is missing survived column and we'll use neural networks to predict whether the passengers in the test data survived or not. Both training and test data are not perfectly clean as we'll see.

Data Dictionary
Survival : 0 = No, 1 = Yes
Pclass : A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
sibsp : The # of siblings / spouses aboard the Titanic Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)
parch : The # of parents / children aboard the Titanic Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.
Ticket : Ticket number
Fare : Passenger fare
Cabin : Cabin number embarked
Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton
Name, Sex, Age (years) are all self-explanatory
Libraries and the Data
Importing libraries
# Load the germane libraries

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
import keras 
# import the additional libraries here



# Load the TensorBoard notebook extension and related libraries
%load_ext tensorboard
import datetime
Loading the data
# Load the test and train data
EDA and Preprocessing
Exploratory Data Analysis
It is your choice how much or how little EDA that you perform. But you should do enough EDA that you feel comfortable with the data and what you'll need to do to make it so that you can run a neural netowrk on it.

Visualizations
There are a number of plots that we could do.

Make at least four or so plots and interpret each one.

Also consider, what columns are correlated? Does this surprise you? Which columns are not correlated? Does this surprise you?

Data Analysis
Again there are a myriad of data analysis that we can do.

Analyze the data in at least three or so ways and interpret the results.

Preprocessing
Missing Data
In order for our neural network to run properly, we need to see if we have any missing data... and if so, we need to deal with it.

# Before filling the missing values, let's drop Cabin column from both data.
Check to see if you have any missing data. If so, for what variables and data sets are you missing data?

Hint: If you have missing data, use SimpleImputer to deal with the nan's.

Whenever you "clean up" missing data, verify that your method worked.

Converting categorical feasture
We have two categorical features, viz. Sex and Emarked. We need to make these numeric.

For Sex, let

Male = 0
Female = 1
For Embarked, let

Q = 0 (Queenstown)
S = 1 (Southampton)
C = 2 (Cherborg)
Given the above "translation," make the categorical variables numeric.

If you did the heatmap above as part of your data visualizations, you can see that there are some correlations between variables. You can choose to modify the data set if you like, but it is not required to do so.

Drop the PassengerId', 'Ticket', and 'Name' columns.

Neural Network Model
Now you know what to do... create your neural network model using sequential, add, compile, etc. Once you do so, you can experiement with TensorBoard below.

You do not have to run fit yet. There is sample code for that below after the tensorboard_callback code. This is because you need to tun the tensorboard_callback as part of the model fit.

TensorBoard is TensorFlow's visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.

N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)

# Clear out any prior log data.
!rm -rf logs
# Be careful not to run this command if already have trained your model and you want to use TensorBoard.

# Sets up a timestamped log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(log_dir)


# The callback function, which will be called in the fit()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Fit the neural neural network model

# Run this code by removing the #'s and set the values that you would like for epochs and batchsize
#results = model.fit(X_train,
 #         Y_train,
  #        epochs=none,
   #       batch_size=none,
    #      verbose=1, 
     #     callbacks=[tensorboard_callback]
      #    )

# Warning: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.
# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar).
Look at your metrics using the usual visualiation methods. After you do that, then run the chunk below to see how the same look in TensorBoard, and all of the additional information TensorBoard gives you.

# TensorBoard
%tensorboard --logdir logs/fit
Continue to tweak your model until you are happy with the results based on model evaluation.

Conclusion
Now that you have the TensorBoard to help you look at your model, you can better understand how to tweak your model.

We'll continue with this for the next lesson when we learn about model regularization.

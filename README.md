{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# TensorFlow and TensorBoard"
      ],
      "metadata": {
        "id": "jVOtlo1QnMwz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Purpose"
      ],
      "metadata": {
        "id": "J9BDVGNbbEaX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The purpose of this lab is twofold.  \n",
        "\n",
        "1.   to review using `TensorFlow` for modeling and evaluation with neural networks.\n",
        "2.   to learn about [`TensorBoard`](https://www.tensorflow.org/tensorboard)\n",
        "\n",
        "`TensorBoard` is `TensorFlow`'s visualization toolkit, so it is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. \n",
        "\n",
        "We'll be using the canonical [Titanic Data Set](https://www.kaggle.com/competitions/titanic/overview)."
      ],
      "metadata": {
        "id": "_QrFX6SUnUwC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## The Titanic"
      ],
      "metadata": {
        "id": "unefoXqXYRxD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### The Titanic and it's data"
      ],
      "metadata": {
        "id": "GDT8p0nBZjVX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "The Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. Though there were about 2,224 passengers and crew members, we are given data of about 1,300 passengers. Out of these 1,300 passengers details, about 9000 data is used for training purpose and remaining 400 is used for test purpose. The test data is missing survived column and we'll use neural networks to predict whether the passengers in the test data survived or not. Both training and test data are not perfectly clean as we'll see."
      ],
      "metadata": {
        "id": "r2UPXuXmZxj5"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Data Dictionary"
      ],
      "metadata": {
        "id": "UO5Qrri1Zz9b"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "*   Survival : 0 = No, 1 = Yes\n",
        "*   Pclass : A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower\n",
        "*   sibsp : The # of siblings / spouses aboard the Titanic Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)\n",
        "*   parch : The # of parents / children aboard the Titanic Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.\n",
        "*   Ticket : Ticket number\n",
        "*   Fare : Passenger fare\n",
        "*   Cabin : Cabin number embarked\n",
        "*   Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton\n",
        "*   Name, Sex, Age (years) are all self-explanatory"
      ],
      "metadata": {
        "id": "LJTNWdE1ZmI2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Libraries and the Data\n",
        "\n"
      ],
      "metadata": {
        "id": "1adtHjJCE5sd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Importing libraries"
      ],
      "metadata": {
        "id": "Zoz_n8VnFdxB"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "lS0qLxZmnLHw"
      },
      "outputs": [],
      "source": [
        "# Load the germane libraries\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns \n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "\n",
        "import tensorflow as tf\n",
        "import keras \n",
        "# import the additional libraries here\n",
        "\n",
        "\n",
        "\n",
        "# Load the TensorBoard notebook extension and related libraries\n",
        "%load_ext tensorboard\n",
        "import datetime"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Loading the data"
      ],
      "metadata": {
        "id": "Z-ljPxHFaf3_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the test and train data\n",
        "\n"
      ],
      "metadata": {
        "id": "NXuO8yi1EjaX"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## EDA and Preprocessing"
      ],
      "metadata": {
        "id": "kpMM9RGkam8n"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Exploratory Data Analysis\n",
        "\n",
        "It is your choice how much or how little EDA that you perform. But you should do enough EDA that you feel comfortable with the data and what you'll need to do to make it so that you can run a neural netowrk on it."
      ],
      "metadata": {
        "id": "Cth9IzJyFMfB"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### Visualizations\n",
        "\n",
        "There are a number of plots that we could do.\n",
        "\n",
        "Make at least four or so plots and interpret each one."
      ],
      "metadata": {
        "id": "Ih5KU1gIHdAG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Also consider, what columns are correlated? Does this surprise you? Which columns are not correlated? Does this surprise you?"
      ],
      "metadata": {
        "id": "VwkhDeYiKdKA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### Data Analysis\n",
        "\n",
        "Again there are a myriad of data analysis that we can do.\n",
        "\n",
        "Analyze the data in at least three or so ways and interpret the results."
      ],
      "metadata": {
        "id": "8gJ2PaOuKtC2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Preprocessing"
      ],
      "metadata": {
        "id": "RTBTMLo2LmX8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### Missing Data\n",
        "\n",
        "In order for our neural network to run properly, we need to see if we have any missing data... and if so, we need to deal with it."
      ],
      "metadata": {
        "id": "FCmML3GtLtd6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Before filling the missing values, let's drop Cabin column from both data.\n"
      ],
      "metadata": {
        "id": "1jYpz_BQLovQ"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check to see if you have any missing data. If so, for what variables and data sets are you missing data? \n",
        "\n",
        "Hint: If you have missing data, use `SimpleImputer` to deal with the nan's.\n",
        "\n",
        "Whenever you \"clean up\" missing data, verify that your method worked."
      ],
      "metadata": {
        "id": "fJoB_nxHAQr1"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### Converting categorical feasture"
      ],
      "metadata": {
        "id": "Ie7U7C3pO5Bd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "We have two categorical features, viz. Sex and Emarked. We need to make these numeric.\n",
        "\n",
        "For Sex, let\n",
        "*   Male = 0\n",
        "*   Female = 1\n",
        "\n",
        "For Embarked, let\n",
        "*   Q = 0 (Queenstown)\n",
        "*   S = 1 (Southampton)\n",
        "*   C = 2 (Cherborg)\n",
        "\n",
        "Given the above \"translation,\" make the categorical variables numeric."
      ],
      "metadata": {
        "id": "UsQwdUOZPAqX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "If you did the heatmap above as part of your data visualizations, you can see that there are some correlations between variables. You can choose to modify the data set if you like, but it is not required to do so."
      ],
      "metadata": {
        "id": "LCmDdw9LQg9f"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Drop the PassengerId', 'Ticket', and 'Name' columns.  "
      ],
      "metadata": {
        "id": "dYr5XKKeVunL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Neural Network Model"
      ],
      "metadata": {
        "id": "DAC11ZbUU2QP"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now you know what to do... create your neural network model using `sequential`, `add`, `compile`, etc. Once you do so, you can experiement with `TensorBoard` below.\n",
        "\n",
        "You do not have to run `fit` yet. There is sample code for that below after the `tensorboard_callback` code. This is because you need to tun the `tensorboard_callback` as part of the model fit."
      ],
      "metadata": {
        "id": "dxxKkbs7BGnN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "TensorBoard is TensorFlow's visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.\n",
        "\n",
        "N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)\n"
      ],
      "metadata": {
        "id": "gojep38STZCg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Clear out any prior log data.\n",
        "!rm -rf logs\n",
        "# Be careful not to run this command if already have trained your model and you want to use TensorBoard.\n",
        "\n",
        "# Sets up a timestamped log directory\n",
        "log_dir = \"logs/fit/\" + datetime.datetime.now().strftime(\"%Y%m%d-%H%M%S\")\n",
        "\n",
        "# Creates a file writer for the log directory.\n",
        "file_writer = tf.summary.create_file_writer(log_dir)\n",
        "\n",
        "\n",
        "# The callback function, which will be called in the fit()\n",
        "tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)"
      ],
      "metadata": {
        "id": "rpclz6HWTafA"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Fit the neural neural network model\n",
        "\n",
        "# Run this code by removing the #'s and set the values that you would like for epochs and batchsize\n",
        "#results = model.fit(X_train,\n",
        " #         Y_train,\n",
        "  #        epochs=none,\n",
        "   #       batch_size=none,\n",
        "    #      verbose=1, \n",
        "     #     callbacks=[tensorboard_callback]\n",
        "      #    )\n",
        "\n",
        "# Warning: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.\n",
        "# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar)."
      ],
      "metadata": {
        "id": "swULa4QCUC83"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Look at your metrics using the usual visualiation methods. After you do that, then run the chunk below to see how the same look in `TensorBoard`, and all of the additional information `TensorBoard` gives you."
      ],
      "metadata": {
        "id": "EPkxW1lgBsZK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# TensorBoard\n",
        "%tensorboard --logdir logs/fit"
      ],
      "metadata": {
        "id": "UI9lMXCXX5y7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Continue to tweak your model until you are happy with the results based on model evaluation."
      ],
      "metadata": {
        "id": "4xd1lF1f9T4w"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Conclusion"
      ],
      "metadata": {
        "id": "1T1ej4W68j-T"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.\n",
        "\n",
        "We'll continue with this for the next lesson when we learn about model regularization."
      ],
      "metadata": {
        "id": "XHW7vUl19I9R"
      }
    }
  ]
}
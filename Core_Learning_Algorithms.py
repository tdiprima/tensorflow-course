import matplotlib.pyplot as plt
import numpy as np

# These are just lists
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

plt.plot(x, y, 'ro')

# Axis limits
plt.axis([0, 6, 0, 20])

# No axis limits
plt.plot(x, y, 'ro')

plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])

# Calculate the equation of a straight line that best fits the data in x and y.
line_best_fit = np.polyfit(x, y, 1)

# Turn that equation into a function that we can use to calculate the y value for any x value along the line.
my_func = np.poly1d(line_best_fit)

# Make a new list with just the unique values of x
uniq = np.unique(x)

# Calculate the y values for each unique x value, and then plot
plt.plot(uniq, my_func(uniq))

plt.show()

print("\n\"equation\" of best fit:", line_best_fit)  # numpy.ndarray
print("\nnew function:", my_func)  # numpy.poly1d
print("\nunique x:", uniq)  # numpy.ndarray
print("\ncorresponding y:", my_func(uniq), "\n")  # numpy.ndarray

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval  = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')   # testing data

dftrain.head()

# Statistical analysis
dftrain.describe()

# Find one row - df.loc(idx) - locating row 0
dftrain.loc[0]

# df["column"]
dftrain["age"]

# find unique embark town
points = dftrain.embark_town.unique()

# sort values smallest to largest
points.sort()

# display sorted values
points

# How many survived, per gender
series_group_by = dftrain.groupby('sex').survived

pd.DataFrame(series_group_by.mean())  # bc it's 0 and 1

# How many were alone
dftrain.alone.value_counts()

# Plot it, with color!
dftrain.alone.value_counts().plot(kind='barh', stacked=True, color="#00f")

dftrain.groupby('alone').survived.mean()

# Alone survival
dftrain.groupby('alone').survived.mean().plot(kind='barh', color="#7f00ff").set_xlabel('% survive')

# Get our "labels" that we're gonna predict
y_train = dftrain.pop('survived')  # pandas Series
y_eval = dfeval.pop('survived')  # pandas Series

pd.DataFrame(y_train).head()

dftrain.shape  # 627 entries and 9 features

pd.DataFrame(y_train).head()

dftrain.age.hist(bins=20, color="#0ff")

dftrain.sex.value_counts().plot(kind='barh')

dftrain['class'].value_counts().plot(kind='barh', color="#0f0")

# Concat. We took it apart, now we're putting it back together again.
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh', color="#f0f").set_xlabel('% survive')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # Unique features
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    print(feature_name, "\t", vocabulary)

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

pd.DataFrame(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    # Inner function, this will be returned
    def input_function():
        # Create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df)
        )

        # Randomize order of data
        if shuffle:
            ds = ds.shuffle(1000)

        # Split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        
        # ds is of type RepeatDataset

        return ds  # Return a batch of the dataset

    return input_function  # Return a function object for use


# Call the input_function that was returned to us
# to get a dataset object we can feed to the model
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

print("Input functions created successfully.")

# We create a linear estimator by passing the feature columns we created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

clear_output()

type(linear_est)

# Train - with linear classifier
linear_est.train(train_input_fn)

# Get model metrics/stats by testing on testing data
result = linear_est.evaluate(eval_input_fn)

# clear console output
clear_output()

print("\ndict", result)

# a dict of stats about our model
print("\naccuracy:", result['accuracy'])

dftrain

pred_dicts = list(linear_est.predict(eval_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

clear_output()

pd.DataFrame(probs).head()  # 264 rows

# Frequency [vs] Probability of Survival
probs.plot(kind='hist', bins=20, title='predicted probabilities')

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pandas as pd

# Lets define some constants to help us later on

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')

train.head() # the species column is now gone

train.shape  # we have 120 entries with 4 features

def input_fn(features, labels, training=True, batch_size=256):

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat, if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature columns - basically, your Excel table header
my_feature_columns = []

for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

pd.DataFrame(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes, respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

clear_output()

"""
Train the classifier
We include a lambda to avoid creating an inner function, like before.
"""
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

clear_output()

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

predict = {'SepalLength': [0.5], 'SepalWidth': [0.4], 'PetalLength': [0.5], 'PetalWidth': [0.3]}

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

# Here is some example input and expected classes you can try above

expected = ['Setosa', 'Versicolor', 'Virginica']

predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

# Due to a version mismatch with` tensorflow v2` and `tensorflow_probability`, we need to install the most recent version of `tensorflow_probability`:

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on

# Refer to point 2 above
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])

# Refer to points 3 and 4 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])

# Refer to point 5 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# The loc argument represents the mean
# The scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()
print("\nmodel.mean:\n", mean)

# Due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# In the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print("\nmean.numpy:\n", mean.numpy())
# Core Learning Algorithms - Learning Guide

## What You Should Have Learned

### Linear Regression Fundamentals

1. **Line of Best Fit Concepts**
   - Understanding that linear regression finds relationships between variables
   - The equation `y = mx + b` where m is slope and b is y-intercept
   - Using NumPy's `polyfit()` and `poly1d()` for creating predictive functions
   - Visualizing data relationships with matplotlib

2. **TensorFlow Estimators**
   - **LinearClassifier**: For classification problems (predicting categories)
   - **DNNClassifier**: Deep Neural Networks for complex pattern recognition
   - Understanding the difference between regression (predicting numbers) and classification (predicting categories)

### Data Preprocessing and Feature Engineering

3. **Working with Real Data (Titanic Dataset)**
   - Loading data with pandas: `pd.read_csv()`
   - Exploratory data analysis: `.head()`, `.describe()`, `.shape`
   - Data visualization: histograms, bar charts, survival analysis
   - Feature selection and understanding data relationships

4. **Feature Columns**
   - **Categorical features**: Converting text/categories to numbers using `tf.feature_column.categorical_column_with_vocabulary_list()`
   - **Numeric features**: Using `tf.feature_column.numeric_column()`
   - Understanding why machines need numeric data

5. **Data Pipeline Creation**
   - Creating input functions with `tf.data.Dataset.from_tensor_slices()`
   - Batching data for efficient training
   - Shuffling data to prevent overfitting
   - The concept of epochs (how many times the model sees the data)

### Classification Problems

6. **Binary Classification (Titanic Survival)**
   - Predicting yes/no outcomes (survived or not)
   - Training with `.train()` and evaluating with `.evaluate()`
   - Understanding accuracy metrics
   - Making predictions on new data

7. **Multi-class Classification (Iris Flowers)**
   - Predicting among multiple categories (3 flower species)
   - Deep Neural Networks with hidden layers `[30, 10]`
   - Understanding probability distributions in predictions
   - Feature engineering for flower measurements

### Advanced Concepts

8. **Hidden Markov Models**
   - Probabilistic models for sequential data
   - Weather prediction using states and transitions
   - Understanding probability distributions with TensorFlow Probability
   - Initial, transition, and observation distributions

9. **Model Training Concepts**
   - **Epochs**: Number of complete passes through the data
   - **Batch size**: How much data to process at once
   - **Steps**: Number of training iterations
   - **Shuffle**: Randomizing data order to improve learning

### Key Machine Learning Principles

10. **Training vs Testing**
    - Never test on training data (leads to overfitting)
    - Using separate datasets for evaluation
    - Understanding accuracy scores and what they mean

11. **Data Preparation**
    - Handling categorical vs numeric data
    - Creating vocabulary lists for text features
    - Normalizing and preprocessing data
    - Creating input pipelines that TensorFlow can understand

### What's Next

This foundation prepares you for:
- Understanding how neural networks process different types of data
- Building more complex models with multiple layers
- Working with image and text data in later modules
- Understanding the training process and optimization

The algorithms you learned here (linear regression, classification, HMMs) are building blocks for understanding more advanced deep learning concepts.
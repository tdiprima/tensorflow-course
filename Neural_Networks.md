# Neural Networks - Learning Guide

## What You Should Have Learned

### Neural Network Fundamentals

1. **What Neural Networks Are**
   - **Layered representation of data**: Neural networks transform data through multiple layers
   - **Deep vs shallow**: "Deep" refers to having multiple hidden layers
   - **Data transformation**: Each layer transforms data to learn different features
   - **Pattern recognition**: Networks learn to identify patterns in data

2. **How Neural Networks Work**
   - **Weighted sum calculation**: `Y = (Σ wi * xi) + b`
   - **Weights**: Connections between neurons that get adjusted during training
   - **Biases**: Constants that shift the activation function
   - **Activation functions**: Add non-linearity (ReLU, tanh, sigmoid, softmax)
   - **Forward propagation**: Data flows from input to output
   - **Backpropagation**: The algorithm that adjusts weights based on errors

### Neural Network Architecture

3. **Layer Types and Structure**
   - **Input layer**: Where data enters (784 neurons for 28x28 images)
   - **Hidden layers**: Transform data (you choose the number of neurons)
   - **Output layer**: Final predictions (10 neurons for 10 classes)
   - **Dense/Fully connected**: Every neuron connects to every neuron in next layer

4. **Key Components**
   - **Neurons**: Hold and process single numeric values
   - **Connections**: Links between neurons with associated weights
   - **Architecture design**: Choosing layer sizes and activation functions

### Practical Implementation with Keras

5. **Fashion MNIST Dataset**
   - **28x28 grayscale images** of clothing items
   - **10 classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
   - **60,000 training images, 10,000 test images**
   - **Pixel values**: 0-255 (black to white)

6. **Data Preprocessing**
   - **Normalization**: Dividing by 255.0 to scale pixels to 0-1 range
   - **Why normalize**: Smaller values are easier for the model to process
   - **Shape understanding**: (60000, 28, 28) = 60k images of 28x28 pixels

7. **Model Building with Keras Sequential**
   ```python
   model = keras.Sequential([
       keras.layers.Flatten(input_shape=(28, 28)),    # Flatten 2D to 1D
       keras.layers.Dense(128, activation='relu'),     # Hidden layer
       keras.layers.Dense(10, activation='softmax')    # Output layer
   ])
   ```

### Training Process

8. **Model Compilation**
   - **Optimizer**: 'adam' (adaptive learning algorithm)
   - **Loss function**: 'sparse_categorical_crossentropy' (for multi-class)
   - **Metrics**: 'accuracy' (to track performance)

9. **Training and Evaluation**
   - **Epochs**: Number of complete passes through training data
   - **Training vs validation accuracy**: Understanding overfitting
   - **Test evaluation**: Final performance on unseen data
   - **.fit()**: Training the model
   - **.evaluate()**: Testing the model
   - **.predict()**: Making predictions

### Making Predictions

10. **Understanding Model Output**
    - **Probability distributions**: Model outputs probabilities for each class
    - **np.argmax()**: Finding the class with highest probability
    - **Prediction interpretation**: Converting probabilities to class labels

11. **Visualization and Verification**
    - **Plotting images**: Using matplotlib to display fashion items
    - **Prediction visualization**: Showing expected vs predicted labels
    - **Interactive testing**: Building functions to test individual images

### Key Machine Learning Concepts

12. **Loss Functions and Optimization**
    - **Cost/Loss function**: Measures how wrong the model is
    - **Gradient descent**: Algorithm for finding optimal weights
    - **Backpropagation**: How the network learns from mistakes
    - **Training loop**: Forward pass → calculate loss → backward pass → update weights

13. **Activation Functions**
    - **ReLU**: Most common, simple and effective
    - **Softmax**: Converts outputs to probability distribution
    - **Why activation functions**: Add non-linearity for complex patterns

14. **Overfitting Awareness**
    - **Training vs test accuracy**: Test accuracy usually lower
    - **Generalization**: Model's ability to work on new data
    - **Why it happens**: Model memorizes training data

### What's Next

This foundation prepares you for:
- **Convolutional Neural Networks (CNNs)**: Better for image data
- **Recurrent Neural Networks (RNNs)**: For sequential data like text
- **Transfer learning**: Using pre-trained models
- **Hyperparameter tuning**: Optimizing layer sizes, learning rates, etc.
- **More complex architectures**: Multiple hidden layers, different activation functions

The neural network concepts you learned here are the building blocks for all deep learning models. Understanding how data flows through layers, how training works, and how to evaluate models is essential for any AI project.
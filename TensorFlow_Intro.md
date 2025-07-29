# TensorFlow Introduction - Learning Guide

## What You Should Have Learned

### Core Concepts

1. **TensorFlow Basics**
   - How to suppress FutureWarnings for cleaner output
   - Checking your TensorFlow version
   - Understanding that TensorFlow is Google's machine learning framework

2. **Tensors Fundamentals**
   - **What are tensors**: Multi-dimensional arrays that are the basic data structure in TensorFlow
   - **Creating tensors**: Using `tf.Variable()` to create different types of tensors (string, int, float)
   - **Tensor shapes and ranks**: 
     - Rank 0 = scalar (single value)
     - Rank 1 = vector (1D array) 
     - Rank 2 = matrix (2D array)
     - Higher ranks = multi-dimensional arrays

3. **Tensor Operations**
   - **Shape manipulation**: Using `tf.reshape()` to change tensor dimensions
   - **The -1 trick**: Let TensorFlow automatically calculate one dimension
   - **Shape preservation**: Total number of elements must remain constant when reshaping

4. **TensorFlow ↔ NumPy Integration**
   - Converting TensorFlow tensors to NumPy arrays using `.numpy()`
   - Understanding that they share underlying data (changes affect both)
   - Quick shape checking with `.numpy().shape`

5. **Tensor Indexing and Slicing**
   - **Basic indexing**: Accessing specific elements `tensor[row, col]`
   - **Row/column selection**: Getting entire rows `tensor[0]` or columns `tensor[:, 0]`
   - **Advanced slicing**: Using step notation `tensor[1::2]` to select every other element
   - **Range slicing**: Selecting ranges like `tensor[1:3, 0]`

6. **Utility Functions**
   - `tf.ones()`: Creating tensors filled with ones
   - `tf.slice()`: Extracting portions of tensors with begin/size parameters
   - `tf.rank()`: Getting the number of dimensions
   - `.shape`: Getting tensor dimensions

### Key Takeaways

- **Tensors are everywhere**: They're the foundation of all TensorFlow operations
- **Shape matters**: Understanding and manipulating tensor shapes is crucial for deep learning
- **Indexing flexibility**: TensorFlow provides powerful ways to slice and access tensor data
- **NumPy compatibility**: Seamless integration between TensorFlow and NumPy workflows

### What's Next

This introduction prepares you for:
- Building neural network layers (which operate on tensors)
- Data preprocessing and manipulation
- Understanding model inputs and outputs
- Advanced TensorFlow operations and transformations

The tensor manipulation skills you learned here are essential for every TensorFlow project you'll work on.
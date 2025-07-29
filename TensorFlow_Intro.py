from warnings import simplefilter

import tensorflow as tf

# Always a good idea for tensorflow, imo
simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)

# Creating Resource Variables
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# These tensors have a shape of 0 (rank 0), which means they're a "scalar".
print(string)
print(number)
print(floating)

# One list, one array, one dimension; a vector.
rank1_tensor = tf.Variable(["Test"], tf.string)
rank1_tensor

# Lists inside of a list, a matrix:
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
rank2_tensor

# How to get the rank
tf.rank(rank1_tensor)

tf.rank(rank2_tensor)

tf.rank(number)

# TENSOR SHAPE
rank1_tensor.shape # Because it's a rank 1, we only get 1 number

rank2_tensor.shape

tensor1 = tf.ones([1, 2, 3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor1.shape

tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to shape [2,3,1]
tensor2.shape

tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,2]
tensor3.shape

# The number of elements in the reshaped tensor MUST match the number in the original

print("tensor1:", tensor1)
print("\ntensor2:", tensor2)
print("\ntensor3:", tensor3)
# Notice the changes in shape

import tensorflow as tf
import numpy as np

# create a TensorFlow tensor
x = tf.constant([[1, 2], [3, 4]])

# convert the tensor to a NumPy ndarray
x_np = x.numpy()

# print the NumPy ndarray
print("\narray:\n", x_np)
print("\ndtype:", x_np.dtype)
print("\nshape:", x_np.shape)

# So instead of dealing with all this bs, you could just add `.numpy().shape`
tf.constant([[1, 2], [3, 4]]).numpy().shape

tf.ones([3, 4], tf.int32)

t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])

tf.slice(
    t1,
    begin=[1],
    size=[3]
)

# Creating a 2D tensor
matrix = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
]

tensor = tf.Variable(matrix, dtype=tf.int32)
print("\nrank:", tf.rank(tensor))
print("\nshape:", tensor.shape)

# Let's select some different rows and columns from our tensor

three = tensor[0, 2]  # selects the 3rd element from the 1st row
print("\n3rd element:", three)

row1 = tensor[0]  # selects the first row
print("\n1st row:", row1)

column1 = tensor[:, 0]  # selects the first column
print("\n1st column:", column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row (heh?)
print("\nrows 2 and 4:", row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print("\n1st column, rows 2 and 3:", column_1_in_row_2_and_3)
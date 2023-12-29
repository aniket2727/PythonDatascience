import numpy as np

# Create a 1D array
arr1d = np.array([1, 2, 3, 4, 5])

# Create a 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access elements in the arrays
print("1D Array:", arr1d)
print("Element at index 2 in 1D Array:", arr1d[2])

print("\n2D Array:")
print(arr2d)
print("Element at row 1, column 2 in 2D Array:", arr2d[1, 2])

# Perform operations on arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise addition
result_add = arr1 + arr2
print("\nElement-wise Addition:")
print(result_add)

# Element-wise multiplication
result_multiply = arr1 * arr2
print("\nElement-wise Multiplication:")
print(result_multiply)

# Dot product
dot_product = np.dot(arr1, arr2)
print("\nDot Product:")
print(dot_product)


import numpy as np
import matplotlib.pyplot as plt

# generating the data
np.random.seed(0)
num_samples = 50
X = np.linspace(0, 10, num_samples)
true_weights = np.array([2.5])

# Gaussian distribution scaled by a factor of 0.5 (effectively variance = 0.25 and mean 0).
noise = 0.5 * np.random.randn(num_samples)
# ading some noise so the x to y relationship isn't perfectly linear
y = X * true_weights + 1 + noise
# true weight is w = [2.5] and bias term is 1 (effectively a 2-dim w = [1, 2.5])
# Adding a column of ones to X for the bias term
#our inputs for the model is X_bias that is  x+bias and bias is 1.
X_bias = np.c_[np.ones(X.shape[0]), X]

print(X[0:5])
print(noise[0:5])
print(y[0:5])
print(y.shape)

print(X_bias[0:5])
print(X_bias.shape)

print(true_weights)

# Closed form solution using ridge regression for above generated data
# we need to find for w_ridge​=((X_bias^T*​Xbias​+λI)^(−1))*X_bias^T*​y for our closed form solution.
l_value = 0.01

import numpy as np

# here we are creating function for matrix multiplication
def matrix_multiply(A, B):
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix multiplication is not possible")

    matmul = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                matmul[i, j] = matmul[i, j] + (A[i, k] * B[k, j])

    return matmul

# here we are creating function for taking transpose of the matrix
def transpose_matrix(A):
    trans = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            trans[j, i] = A[i, j]
    return trans

# Function for Ridge regression closed-form solution
def ridge_cf_solution(X_bias, y, lambda_val):
    # calculating X_bias_transpose * X_bias
    X_T_X = matrix_multiply(transpose_matrix(X_bias), X_bias)

    # here we are adding the regularization term to X_bias_transpose_X_bias
    X_T_X += lambda_val * np.identity(X_bias.shape[1])

   # here we are solving for (X_bias^T*X_bias+l_value*I)^-1
    X_T_X_inv = np.linalg.inv(X_T_X)

    # Calculate X_bias_transpose * y
    X_T_y = matrix_multiply(transpose_matrix(X_bias), y)

    # Calculate the weights W_ridge_cf for the closed-form solution
    W_ridge_cf = matrix_multiply(X_T_X_inv, X_T_y).flatten()

    return W_ridge_cf

# Ridge regression closed form parameters lambda
lambda_val = 0.01

# Performing Ridge regression using the function
W_ridge_cf_solution = ridge_cf_solution(X_bias, y, lambda_val)

print("Value of W_ridge_cf for closed-form solution is", W_ridge_cf_solution)
# print(W_ridge_cf_solution.shape)

# Gradient descent solution form solution using ridge regression for above generated data
l_value = 0.01
l_rate = 0.001
n = 1000

def ridge_gd_solution(X, y, lambda_val, learning_rate, num_iterations):
    num_features = X.shape[1]
    weights = np.zeros(num_features)

    for i in range(num_iterations):
        # Calculation for gradient=dl/dW=2*X_bias^T(X_bias*w−y)+2*l_value*w
        gradient = np.dot(X.T, np.dot(X, weights) - y) + lambda_val * weights
        weights = weights - learning_rate * gradient

    return weights
# Function for finding number of iterations till convergence
def find_convergence_iteration(weights, X, y, lambda_val, learning_rate=0.01, epsilon=1e-8):
    def iteration(weights):
        gradient = np.dot(X.T, np.dot(X, weights) - y) + lambda_val * weights
        return gradient

    n = 1
    while n < float('inf'):
        gradient = iteration(weights)
        weights = weights - learning_rate * gradient
        n += 1
        if np.linalg.norm(gradient) < epsilon:
            break

    return n

initial_weights = np.zeros(X_bias.shape[1])
# Performing Ridge regression using the gradient descent function
W_ridge_gd_solution = ridge_gd_solution(X_bias, y, l_value, l_rate, n)

# Find the convergence iteration
num_iterations = find_convergence_iteration(W_ridge_gd_solution, X_bias, y, l_value, l_rate)

print("Number of iterations for gradient descent:", num_iterations)

# Re-run Ridge regression with the found number of iterations
W_ridge_gd_solution_new = ridge_gd_solution(X_bias, y, l_value, l_rate, num_iterations)

print("Value of W_ridge_gd for gradient descent solution is", W_ridge_gd_solution)
print(W_ridge_cf_solution)

# Plotting
plt.figure(figsize=(12, 6))

# Plotting the generated data
plt.subplot(1, 2, 1)
plt.scatter(X_bias[:, 1], y, label='Generated Data')

# Plotting the initial line for gradient descent that is zero
initial_line = np.dot(X_bias, initial_weights)
plt.plot(X_bias[:, 1], initial_line, color='gray', linestyle='--', label='Initial Line')

# Plotting the Gradient Descent Solution
plt.plot(X_bias[:, 1], np.dot(X_bias, W_ridge_gd_solution), color='green', label='Gradient Descent Solution')
plt.title('Gradient Descent Solution')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plotting the Closed-form Solution
plt.subplot(1, 2, 2)
plt.scatter(X_bias[:, 1], y, label='Generated Data')
plt.plot(X_bias[:, 1], np.dot(X_bias, W_ridge_cf_solution), color='red', label='Closed-form Solution')
plt.title('Closed-form Solution')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()

# Plot for generated data Gradient descent solution and closed form solution
plt.figure(figsize=(8, 6))
plt.scatter(X_bias[:, 1], y, label='Generated Data')
plt.plot(X_bias[:, 1], np.dot(X_bias, W_ridge_cf_solution), color='red', label='Closed-form Solution')
plt.plot(X_bias[:, 1], np.dot(X_bias, W_ridge_gd_solution), color='green', label='Gradient Descent Solution')
plt.title('Gradient Descent Solution')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

#generating the data for part 2 of question 1
np.random.seed(0)

# Generating 80 values from a uniform distribution scaled by a factor of 5
X = np.random.uniform(-5, 5, 80)

# Applying sin function to the values
y_true = np.sin(X)

# Adding noise sampled from a Gaussian distribution with scaling factor 0.5
noise = 0.5 * np.random.randn(80)
print(noise[0:5])
print(y_true[0:5])
y_noisy = y_true + noise
print(y_noisy[0:5])

#plotting the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X, y_noisy, color='red', label='Noisy Data')
plt.plot(np.linspace(-5,5,80),np.sin(np.linspace(-5,5,80)),c='g',linewidth=2,label='generated input data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Ridge regression closed-form solution for polynomial data points
X_poly = np.c_[X**3, X**2, X, np.ones_like(X)]
#print(X_poly[0:5])


# Ridge regression parameters
lambda_val = 0.001

# Performing Ridge regression using the function that we already defined in question number 1
W_ridge_cf_poly_solution = ridge_cf_solution(X_poly, y_noisy, lambda_val)

print("Value of W_ridge_cf for closed-form solution is", W_ridge_cf_poly_solution)
print(W_ridge_cf_poly_solution.shape)



# Plotting the true result and the result from Ridge regression cf solution
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(-5,5,80),np.sin(np.linspace(-5,5,80)),c='g',linewidth=2,label='generated input data')
plt.scatter(X, y_noisy, label='Data with noise')
plt.plot(X, np.dot(X_poly, W_ridge_cf_poly_solution), label='Ridge Regression cf Result for polynomial', color='blue')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge Regression with Polynomial Features')
plt.show()

l_value = 0.01
l_rate = 0.0001
n = 1000
# Apply Polynomial feature transformation to the new dataset
X_poly = np.c_[X**3, X**2, X, np.ones_like(X)]

# Manually standardize the input data with a small constant added to avoid division by zero
epsilon = 1e-8
X_poly_mean = np.mean(X_poly, axis=0)
X_poly_std = np.std(X_poly, axis=0)

X_poly_std[X_poly_std < epsilon] = epsilon  # Avoid division by very small standard deviations

X_poly_scaled = (X_poly - X_poly_mean) / X_poly_std

# Initialize weights with small random values
# weights = np.zeros(X_poly_scaled.shape[1])

# performing the gradient descent solution with updated parameters with the function that we already defined for part 1
W_ridge_gd_poly_solution = ridge_gd_solution(X_poly_scaled, y_noisy, l_value, l_rate, n)

print("Value of W_ridge_gd for gradient descent solution is", W_ridge_gd_poly_solution)

# Plotting for gd solution of polynomial features
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(-5, 5, 80), np.sin(np.linspace(-5, 5, 80)), c='g', linewidth=2, label='generated input data')
plt.scatter(X, y_noisy, label='Data with noise')
plt.plot(X, np.dot(X_poly_scaled, W_ridge_gd_poly_solution), label='Ridge Regression gd Result for polynomial', color='blue')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge Regression with Polynomial Features')
plt.show()




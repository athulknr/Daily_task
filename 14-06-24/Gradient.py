x = 10
learning_rate = 0.1
num_iteration = 100

# Gradient Descent Loop
for i in range(num_iteration):
    # Compute the gradient of f(x) = x^2, which is f'(x) = 2*x
    gradient = 2 * x
    
    # Update the parameter x
    x = x - learning_rate * gradient
    
    # Print the updated value of x at each iteration
    print(f"Iteration {i+1}: x = {x}")

import numpy

# cost function takes a linear model (y = mx + b) 
# represented by a tuple (m, b) and labeled data set of 
# [x, y] pairs; and outputs the cost 
def calculate_linear_cost(model: tuple, labeled_data: numpy.array):
    (m, b) = model
    total_cost = 0

    for (x, y) in labeled_data:
        total_cost = total_cost + (y - ((m*x) + b))**2
    
    return total_cost/(len(labeled_data *2))

# calculate the derivative of the cost function and adjust the 
# linear constansts by a step in that direction
def gradient_decsent(labeled_data: numpy.array, model: tuple, rate=0.001, iterations=10000):
    (m, b) = model
    for i in range(iterations):

        (derivative_m, derivative_b) = gradient = compute_gradient(labeled_data, (m, b))

        tmp_m = m - (rate * derivative_m)
        tmp_b = b - (rate * derivative_b)

        m = tmp_m
        b = tmp_b
    return m, b

def compute_gradient(labeled_data: numpy.array, model: tuple):
    cost = calculate_linear_cost(model, labeled_data)

    (m, b) = model

    derivative_b = 0
    derivative_m = 0

    for (x, y) in labeled_data:
        derivative_m += ((m*x + b) - y) * x
        derivative_b += ((m*x + b) - y)


    derivative_m = derivative_m / len(labeled_data)
    derivative_b = derivative_b / len(labeled_data)

    return (derivative_m, derivative_b)

def compute_model_output(labeled_data, model):
    input = labeled_data[:,0]

    output = numpy.zeros((len(input), 2))
    (m, b) = model

    for i in range(len(input)):
        prediction = m * input[i] + b
        output[i] = [prediction]

    return output


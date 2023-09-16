import numpy
import matplotlib.pyplot as plt

from linear import (
    calculate_linear_cost, 
    gradient_decsent,
    compute_model_output
    )

model_flat_origin = (0, 0)
training_data_a = numpy.array([
    [1, 2],
    [4, 7],
    [6, 6],
    [5, 8],
    [9, 7]
])

training_data_b = numpy.array([
    [1, 1],
    [0, 0]
])

def test_cost():
    cost =  calculate_linear_cost(model_flat_origin, training_data_a)
    print(cost)
    assert cost

def test_gradient_descent():
    model = gradient_decsent(training_data_a, model_flat_origin, rate=0.01, iterations=10000)
    predictions = compute_model_output(training_data_a, model)
    createplot(training_data_a, predictions)

def createplot(training_data, predictions):
    print(training_data[:,0], predictions)
    plt.scatter(training_data[:,0], training_data[:,1])
    plt.plot(training_data[:,0], predictions, c='b',label='Our Prediction')
    plt.show()





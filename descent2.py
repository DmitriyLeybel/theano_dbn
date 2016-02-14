import numpy as np
import theano.tensor as T
import theano
from sklearn import preprocessing
# Loads the birthrate data in string format
# Dataset from http://www.stat.ufl.edu/~winner/datasets.html
birth = np.loadtxt('birthrate.dat', dtype=np.str)
from matplotlib import pyplot as plt

def str2float(x):
    return float(x)
# Vectorizes the str2float function
str2floatv = np.vectorize(str2float)
# Does an elementwise conversion of the string values to floats of columns 1 and 2, birthrates and per capita income
birth = str2floatv(birth[:,[1,2]]) # the 1 column is birthrates, the second column is income/capita
birth = preprocessing.scale(birth)
x = birth[:,0]
y = birth[:,1]
# GRADIENT DESCENT

X = T.vector()
Y = T.vector()

theta_0 = theano.shared(0.)
theta_1 = theano.shared(0.)


cost = (1/(float(birth.shape[0]))) * T.sum(T.sqr(theta_0 + theta_1 * X - Y))
gradient_t0 = T.grad(cost=cost, wrt=theta_0)     # Optimize with Jacobian later
gradient_t1 = T.grad(cost=cost, wrt=theta_1)

updates_t0 = [(theta_0, theta_0 - gradient_t0 * 0.01)]
updates_t1 = [(theta_1, theta_1 - gradient_t1 * 0.01)]

train_t0 = theano.function(inputs=[X, Y], outputs = [], updates = updates_t0, allow_input_downcast = True)
train_t1 = theano.function(inputs=[X, Y], outputs = [], updates = updates_t1, allow_input_downcast = True)


for i in range(100):
    # tempt0 = theta_0.get_value() # May use with givens to train thetas simultaneously
    # tempt1 = theta_1.get_value()
    train_t0(x,y)
    train_t1(x,y)



y = theta_1.get_value()*birth[:,0] + theta_0.get_value()
plt.scatter(birth[:,0],birth[:,1])
plt.plot(x, y)
plt.show()



# STOCHASTIC GRADIENT DESCENT - TO BE IMPLEMENTED

# for (x_i,y_i) in training_set:
#                             # imagine an infinite generator
#                             # that may repeat examples (if there is only a finite training set)
#     loss = f(params, x_i, y_i)
#     d_loss_wrt_params = ... # compute gradient
#     params -= learning_rate * d_loss_wrt_params
#     if <stopping condition is met>:
#         return params
#
# # Minibatch Stochastic Gradient Descent
#
# # assume loss is a symbolic description of the loss function given
# # the symbolic variables params (shared variable), x_batch, y_batch;
#
# # compute gradient of loss with respect to params
# d_loss_wrt_params = T.grad(loss, params)
#
# # compile the MSGD step into a theano function
# updates = [(params, params - learning_rate * d_loss_wrt_params)]
# MSGD = theano.function([x_batch,y_batch], loss, updates=updates)
#
# for (x_batch, y_batch) in train_batches:
#     # here x_batch and y_batch are elements of train_batches and
#     # therefore numpy arrays; function MSGD also updates the params
#     print('Current loss is ', MSGD(x_batch, y_batch))
#     if stopping_condition_is_met:
#         return params
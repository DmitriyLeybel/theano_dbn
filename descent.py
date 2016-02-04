import numpy as np
import theano.tensor as T
import theano
# Loads the birthrate data in string format
# Dataset from http://www.stat.ufl.edu/~winner/datasets.html
birth = np.loadtxt('birthrate.dat', dtype=np.str)

def str2float(x):
    return float(x)
# Vectorizes the str2float function
str2floatv = np.vectorize(str2float)
# Does an elementwise conversion of the string values to floats of columns 1 and 2, birthrates and per capita income
birth = str2floatv(birth[:,[1,2]]) # the 1 column is birthrates, the second column is income/capita

# GRADIENT DESCENT

# Returns the gradient for A
def derivA(a, B, x, y):
    j = 1/(2*float(birth.shape[0])) * T.sum(T.sqr(a + (B * x) - y))
    gj = T.grad(j, a)
    return gj
# Returns the gradient for B
def derivB(A, b, x, y):
    j = 1/(2*float(birth.shape[0])) * T.sum(T.sqr(A + (b * x) - y))
    gj = T.grad(j, b)
    return gj
# Returns the loss function
def lossfn():
    xT = T.vector('xT')
    yT = T.vector('yT')
    a = T.scalar('a')
    b = T.scalar('b')
    f = 1/(2*float(birth.shape[0])) * T.sum(T.sqr(a + (b * xT) - yT))
    fn = theano.function([a, b,xT, yT], f)
    return fn
# Calculates the loss
def lossX(a, b, x, y):
    return lossfn()(a, b, x, y)
# Returns the update value for the specified variable and gradient d
def update(d,p,var,learning_rate):
    m = T.scalar('m')
    fd = theano.function([var],d)
    f = learning_rate * fd(p)
    return f
# Linear regression through gradient descent to create model y = A + Bx
def gradientDescent():
    x = birth[:,0]
    y = birth[:,1]

    a = T.scalar('a')
    b = T.scalar('b')
    learning_rate = 1
    paramA = 0
    paramB = 0
    tempParamA = 0
    tempParamB = 0
    while True:
        loss = lossX(paramA, paramB, x, y)
        d_loss_wrt_paramsA = derivA(a, paramB, x, y) # The A gradient
        d_loss_wrt_paramsB = derivB(paramA, b, x, y) # The B gradient
        tempParamA -= update(d_loss_wrt_paramsA, paramA,a,learning_rate) # Updates tempParamA
        tempParamB -= update(d_loss_wrt_paramsB, paramB,b, learning_rate) # Updates tempParamB
        paramA = tempParamA
        paramB = tempParamB

        print('Loss:',loss) # Keeps increasing for some reason
        print(tempParamA)
        print(tempParamB)


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
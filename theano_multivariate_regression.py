import numpy as np
import theano.tensor as T
import theano
from sklearn import preprocessing
# Loads the birthrate data in string format
# Dataset from http://www.stat.ufl.edu/~winner/datasets.html
birth = np.loadtxt('birthrate.dat', dtype=np.str)
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

wineArray = np.loadtxt('wine.data',dtype=float,delimiter=',')
wineX = MinMaxScaler().fit_transform(wineArray[:,2:])
wineY = wineArray[:,1]
def str2float(x):
    return float(x)
# # Vectorizes the str2float function
# str2floatv = np.vectorize(str2float)
# # Does an elementwise conversion of the string values to floats of columns 1 and 2, birthrates and per capita income
# birth = str2floatv(birth[:,[1,2]]) # the 1 column is birthrates, the second column is income/capita
# birth = preprocessing.scale(birth)
def multiregression(input_features, output):
    global cost, x_matrix, theta_vec, grad_list
    feature_num = input_features.shape[1]
    x_matrix = T.matrix()
    # x_vec.insert(0, float(1))
    y_vec = T.vector()
    theta_num = feature_num + 1
    theta_v = T.scalar()
    theta_vec = theano.shared(np.zeros(theta_num))

#
# X = T.vector()
# Y = T.vector()
#
# theta_0 = theano.shared(0.)
# theta_1 = theano.shared(0.)
#
#
    cost = (1/(float(input_features.shape[0]))) * T.sum(T.sqr(T.dot(theta_vec, x_matrix) - y_vec))
    ze = theta_vec[0]
    grad_list = T.grad(cost=cost, wrt = theta_vec) # for x in range(theta_num)]

    # [T.grad(cost=cost, wrt=theta) for theta in  ]
# cost = (1/(float(birth.shape[0]))) * T.sum(T.sqr(theta_0 + theta_1 * X - Y))
# gradient_t0 = T.grad(cost=cost, wrt=theta_0)     # Optimize with Jacobian later
# gradient_t1 = T.grad(cost=cost, wrt=theta_1)
#
# updates_t0 = [(theta_0, theta_0 - gradient_t0 * 0.01)]
# updates_t1 = [(theta_1, theta_1 - gradient_t1 * 0.01)]
#
# train_t0 = theano.function(inputs=[X, Y], outputs = [], updates = updates_t0, allow_input_downcast = True)
# train_t1 = theano.function(inputs=[X, Y], outputs = [], updates = updates_t1, allow_input_downcast = True)
#
#
# for i in range(100):
#     tempTheta_0 = theta_0.get_value()
#     train_t0(x,y)
#     theta_0a = theta_0.get_value()
#     theta_0.set_value(tempTheta_0)
#     train_t1(x,y)
#     theta_0.set_value(theta_0a)
#
if __name__ == '__main__':
    multiregression(wineX, wineY)
#     z = theta_1.get_value()*birth[:,0] + theta_0.get_value()
#     plt.scatter(birth[:,0],birth[:,1])
#     plt.plot(x, z)
#     plt.show()
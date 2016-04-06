from theano import tensor as T
from theano import function, shared
import numpy

X = shared(numpy.array([0,1,2,3,4], dtype='int64'))
Y = T.lvector()
X_update = (X, X[2:4]+Y)
f = function(inputs=[Y], updates=[X_update])
f([100,10])
print X.get_value()
# output: [102 13]
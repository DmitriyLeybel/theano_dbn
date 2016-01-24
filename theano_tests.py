import theano.tensor as T
from theano import function
import timeit
import numpy as np

'''Compares the runtimes of various operations accomplished with Theano.
Two different times are measured, the first being slower due to compilation'''
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z)

t1 = timeit.Timer("f(2,4)",setup="from __main__ import f")
print('Runtime for scalar addition:',t1.repeat(2,1))
t1n = timeit.Timer("2+4")
print('Runtime for scalar addition - sans theono: ', t1n.repeat(2,1))


a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.add(a,b)
f1 = function([a,b],c)
i = np.random.random([10000,10000])
j = np.random.random([10000,10000])

t2 = timeit.Timer('f1(i,j)','from __main__ import *')
# t2.repeat(2,4) signifies that the function runs 4 times, for 2 trials
print('Runtime for matrix addition: ',t2.repeat(2,4))
t2n = timeit.Timer('i+j', 'from __main__ import *')
print('Runtime for matrix addition - sans theono: ', t2n.repeat(2,20))

m = T.vector('m')
n = T.vector('n')
o = m ** 2 + n ** 2 + 2 * m * n
f2 = function([m,n],o)
i = np.array([5,4,3])
j = np.array([2,1,10])

t3 = timeit.Timer('f2(i,j)','from __main__ import *')
print('Runtime for vector equation: ', t3.repeat(2,4))
t3n = timeit.Timer('i ** 2 + j ** 2 + 2 * i * j', 'from __main__ import *')
print('Runtime for vector equation - sans theano: ', t3n.repeat(2,4))

''' Logistic Function '''
x = T.matrix('x')
s = 1/ (1 + T.exp(-x))
logisticfn = function([x],s)
z = np.random.random([500,500])
t4 = timeit.Timer('logisticfn(z)', 'from __main__ import *')
print('Runtime for elementwise logistic function on matrix: ', t4.repeat(2,4))
t4n = timeit.Timer('''
1/(1+ np.exp(-z))
''', 'from __main__ import *')
print('Runtime for elementwise logistic function on matrix - sans theono: ', t4n.repeat(2,4))


'''Multiple function capability'''
a, b = T.dmatrices('a','b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
diffc = diff/(diff*-diff)
f = function([a,b],[diff, abs_diff, diff_squared, diffc])

t5 = timeit.Timer('f([[22,5],[5,10]],[[-5,22],[2,1]])', 'from __main__ import *')
print('Runtime for multifunction output', t5.repeat(2,4))

''' Shared Variables'''
from theano import shared
state = shared(1)
mod = T.scalar('mod')
fun = function([], updates=[(state,7+3)])
v = T.scalar(dtype=state.dtype)
funG = state + (4/v)/mod
test = function([v,mod], funG, givens=[(state,v)])    # givens used to replace a variable
print(test)
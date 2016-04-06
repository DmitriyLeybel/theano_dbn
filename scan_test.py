import theano
import theano.tensor as T

import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
def f(prior_result, A):
    return prior_result * A
result, updates = theano.scan(fn=f,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)
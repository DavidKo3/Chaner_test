from chainer import links as L
from chainer import functions as F
import numpy as np
from chainer import Variable


lin = L.Linear(5, 2)
x = Variable(np.ones((3, 5), dtype=np.float32))
y1 = lin(x)

print(x.data)
print(y1.data)

lin2 = L.Linear(4 ,1)
x2=Variable(np.ones((3,4), dtype=np.float32))
y2= lin2(x2)
print (x2.data)
print (y2.data)
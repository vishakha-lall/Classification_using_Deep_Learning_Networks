#XNOR implementation
import numpy as np  
x0=int(1)
x1=int(input("Enter variable 1 "))
x2=int(input("Enter variable 2 "))
Q1=np.array([(-30,20,20),(10,-20,-20)])
Q2=np.array([-10,20,20])
x=np.array([x0,x1,x2])
print(Q1)
print(Q2)
print(x)
a2=np.dot(Q1,x)
print(a2)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
a2=sigmoid(a2)
print(a2)
bias=np.array([1])
a2=np.concatenate([bias,a2])
print(a2)
a3=np.dot(Q2,a2)
a3=sigmoid(a3)
print(a3)

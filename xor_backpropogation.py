#   XOR using backpropogation
import numpy as np
import matplotlib.pyplot as plt

iterations = 60000           # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])


def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
                                                # weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))

errorMatrix = np.empty(iterations,float)
iterationMatrix = np.empty(iterations,int)

weightsHiddenMatrix11 = np.empty(iterations,float)
weightsHiddenMatrix12 = np.empty(iterations,float)
weightsHiddenMatrix13 = np.empty(iterations,float)
weightsHiddenMatrix21 = np.empty(iterations,float)
weightsHiddenMatrix22 = np.empty(iterations,float)
weightsHiddenMatrix23 = np.empty(iterations,float)

weightsOutputMatrix11 = np.empty(iterations,float)
weightsOutputMatrix21 = np.empty(iterations,float)
weightsOutputMatrix31 = np.empty(iterations,float)

for i in range(iterations):

    H = sigmoid(np.dot(X, Wh))                  # hidden layer results
    Z = sigmoid(np.dot(H, Wz))                  # output layer results
    E = Y - Z                                   # how much we missed (error)

    errorMatrix[i] = E[0]
    iterationMatrix[i] = i

    dZ = E * sigmoid_(Z)                        # delta Z
    dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
    Wz +=  H.T.dot(dZ)                          # update output layer weights

    weightsOutputMatrix11[i] = Wz[0][0]
    weightsOutputMatrix21[i] = Wz[1][0]
    weightsOutputMatrix31[i] = Wz[2][0]

    Wh +=  X.T.dot(dH)                          # update hidden layer weights

    weightsHiddenMatrix11[i] = Wh[0][0]
    weightsHiddenMatrix12[i] = Wh[0][1]
    weightsHiddenMatrix13[i] = Wh[0][2]
    weightsHiddenMatrix21[i] = Wh[1][0]
    weightsHiddenMatrix22[i] = Wh[1][1]
    weightsHiddenMatrix23[i] = Wh[1][2]

print(Z)

print("Final values of activation weights")
print("Hidden Layer")
print(Wh)
print("Output Layer")
print(Wz)

plt.figure(1)
plt.plot(iterationMatrix,errorMatrix)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations')
plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(iterationMatrix,weightsHiddenMatrix11,'r--',iterationMatrix,weightsHiddenMatrix12,'b--',iterationMatrix,weightsHiddenMatrix13,'g--',iterationMatrix,weightsHiddenMatrix21,'c--',iterationMatrix,weightsHiddenMatrix22,'m--',iterationMatrix,weightsHiddenMatrix23,'y--')
plt.xlabel('Number of iterations')
plt.ylabel('Weights of Hidden Layer')
plt.subplot(212)
plt.plot(iterationMatrix,weightsOutputMatrix11,'r--',iterationMatrix,weightsOutputMatrix21,'b--',iterationMatrix,weightsOutputMatrix31,'g--')
plt.xlabel('Number of iterations')
plt.ylabel('Weights of Output Layer')
plt.show()

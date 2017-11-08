import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


def NeuralNetwork(layers):
    global activation, activation_prime, weights, r
    activation = tanh
    activation_prime = tanh_prime

    weights = []
    
    
    for i in range(1, len(layers) - 1):
        r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
        weights.append(r)
    r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
    weights.append(r)

def fit(X, y, epochs, learning_rate=0.2):
    global weights, activation, activation_prime


    ones = np.atleast_2d(np.ones(X.shape[0]))
    print ones # for bias units
    X = np.concatenate((ones.T, X), axis=1)
     
    for k in range(epochs):
        if k % 10000 == 0: print 'Iterations:', k
        
        i = np.random.randint(X.shape[0])
        a = [X[i]]

        for l in range(len(weights)):
                dot_value = np.dot(a[l], weights[l])
                _activation = activation(dot_value)
                a.append(_activation)

        error = y[i] - a[-1]
        deltas = [error * activation_prime(a[-1])]

        for l in range(len(a) - 2, 0, -1): 
            deltas.append(deltas[-1].dot(weights[l].T)*activation_prime(a[l]))

        deltas.reverse()

        for i in range(len(weights)):
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            weights[i] += learning_rate * layer.T.dot(delta)

def predict(x): 
    print np.array(x)
    print np.ones(1).T
    a = np.concatenate((np.ones(1).T, np.array(x)))  
    print a
    for l in range(0, len(weights)):
        a = activation(np.dot(a, weights[l]))
    return a

print "ENter number of Iterations : "

epochs = raw_input()

NeuralNetwork([2,2,1])

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])


fit(X, y, int(epochs))

for e in X:
    print(e,predict(e))
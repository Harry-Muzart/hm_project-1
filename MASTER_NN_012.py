##_MASTER_NN_012

print('Deep Machine Learning #1')

##_importing modules and functional datasets

import numpy as np

from sklearn import tree

from sklearn.datasets import load_iris

##_alternatives:
#from sklearn import datasets
#iris = datasets.load_iris()
## alt:
#import sklearn.datasets.load_iris

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

import scipy as sp

#from sklearn.externals.six import StringIO
#import pydot
##_import error: no module named pydot

#import matplotlib.pyplot as plt
##_error: pyparsing

#import tensorflow as tf
##_tensorflow

import random as rd

new_rd = rd.randrange(1,10)

######################

##_Defining nodes
nodes = ['x1', 'x2', 'x3', 'x4', 'm1', 'm2', 'm3', 'm4', 'm5','y1', 'y2', 'y3']

for myNodes in nodes:
    print (myNodes)

print("layers? (2-6)")
layers = int(input("n:"))
##_note: == is equal to for integers/floats/other-strings, and != is not equal to
if(layers <= 6 and layers >= 2):
    		print("ok")
else:
    		print("need integers 2-6")
    		
print("input units? (<=10)")
iunits = int(input("n:"))
if(iunits <= 10):
			print("ok - ANN parameters stated")
else:
			print("reduce i-units number")
    			
def functionANN():
    print('insert function here')
    
functionANN()


def summ(num1, num2, num3):
    return num1 + num2 + num3
	
def div(x1, x2):
	return x1 / x2

x1 = layers
x2 = iunits

##_ratio of inunits to layers
print(div(x2, x1))

##########################

name1 = 'Google'
name2 = 'DeepMind'
name3 = 'Project X'
num1 = 42

print(name1 + name2 + name3)
print(name1, name2, name3)

'Google'.upper()

brand = 'Google DeepMind'
stock_v = 3.45457
##_(here you can insert linked data in real-time)
risk = 0.235

message = 'The stock value of %s is %f, with risk %f' %(brand, stock_v, risk)
print(message)

mess_2 = 'This company is of value {} dollars'.format(stock_v)
print(mess_2)

##_diff str(stock_v)
print(str(stock_v) + str(stock_v))

##_int(stock_v)
print(int(stock_v) + int(stock_v))

list_of_users = ['Harry', 'John', 'Emily']

list_2 = [brand, risk]

print('list of users', list_of_users)
print(list_2)

list_ages = [float(21), float(44), 56.0, 78.0, 32.0]

print(list_ages[1] - list_ages[-1])

print(list_ages[2:4])

new_age = float(input('any new ages?'))

list_ages.append(new_age)

print(list_ages)

print('is the last person older than person 1?', list_ages[0] <= list_ages[5])

if((list_ages[0] <= list_ages[5]) is True):
    print("yes - correct")
else:
    print("no - erroneous info")

if(list_ages[0] <= list_ages[5]):
    print('Y')

if(list_ages[0] >= list_ages[5]):
    print('N')

tuple_list = (2, 5, 6)

print(tuple_list)

qx = float(input('a number?'))
qy = float(input('a 2nd number?'))

if((qx > 5) and (qy > 9) and ((qx + qy) > 20)):
    print('Ok')
else:
    print('need >5 and >9; AND sum > 20')

if(qx > 30 and not(qx == 50)):
    print('Ok')
    
if(qx != 20):
    print('Ok2')

numA = qx if(qx == 69) else qy

print(numA)

if(qx == 72):
    numB = qx
else:
    numB = qy

print(numB)

listX = ['neuron1', 'neuron2', 'neuron3', 'neuron4']

print(listX)

listZ = enumerate(listX[1])

print(listX[1])

for items in listX:
    print(items)

for indx, items in enumerate(listX):
    print(indx, items)

for i in range(0,10,5):
    print(i)

data7 = 14

##_notice 'break', in addition can be followed by: "else 'continue'"

while data7 > 0:
    print ('inf=', data7)
    data7 = data7 - 2
    if data7 == 6:
        break

while data7 > -6:
    print ('infr=', data7)
    data7 = data7 - 2
    if data7 > -4:
        continue
    else:
        print('terminated')
        break
    
num_T = float(input('INSERT num_T'))

num_S = str(input('INSERT num_S'))


message3 = 'x(v) ='
system_spec = '24675'

print(message3, system_spec)

system_spec.replace('24675', 'new_h')

print(message3, system_spec)


import random as rd

new_rd = rd.randrange(1,10)

print('random no')
print(new_rd)
print('random no')


def check(num_tc):
    for x in range(2,num_tc):
        if(num_tc%x == 0):
            return False
    return True

print('is 13 prime?')
print(check(13))

print('end1')


#############################

##_NN set up
training_set = [((1, 0.3, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 0.7, 1), 0)]
weights = [0, 0, 0]

##_dot product()
def dot_prod(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))
##_T
threshold = 1.5
##_LR
learning_rate = 1.0

while True:
    print('-' * 60)
    error_count = 0
    for input_vector, desired_output in training_set:
        print(weights)
        result = dot_prod(input_vector, weights) > threshold
        error = desired_output - result
        if error != 0:
            error_count += 1
            for index, value in enumerate(input_vector):
                weights[index] += learning_rate * error * value
    if error_count == 0:
        break


###############################





import numpy as np
from sklearn import tree

#print(numLayer + "confirmed")
#print('hello, world') 
#numLayer = input("Press Y to initiliase")

print("Initializing...")
print('please input values for A and B')
myA = (input('myA'))
myB = (input('myB'))


##_We can define features and labels
features = [[90, 7], [95, 9], [100, 7], [100, 9], [95, 7], [110, 9],
            [190, 2], [195, 2], [200, 2], [195, 5], [205, 1], [220, 3]]
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

##_Using a classification function f(x) to make decision
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
Rx =(clf.predict([[myA, myB]]))
print(Rx)


#####################

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

#import pylab as pl 
#pl.gray() 
#pl.matshow(digits.images[0]) 
#pl.show() 


##############

##_from datasets > descr
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])


#load_iris
#load_breast_cancer
#load_linnerud



############


import numpy as np



##_dataset array O (output)          
y = np.array([[0,0,1,1]]).T

##_dataset array I (input)
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
 
##_defining a sigmoidal f(x) = y
def nonlinSig(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

##_deterministic property of the calculations 
##_seed random nums
np.random.seed(1)
##_random initialization of weights (note: mean 0)
syn0 = 2*np.random.random((3,1)) - 1


##_note: use range instead of xrange in Python 3.0+
for iter in range(12500):
 	# forward propagation
	V_0 = X
	V_1 = nonlinSig(np.dot(V_0,syn0))

	# error computed:
	V_1_err = y - V_1
 	# take product of discrepency and slope of nonlinSig at V_1 data points
	V_1_delta = V_1_err * nonlinSig(V_1,True)
 	# update w for each node
	syn0 += np.dot(V_0.T,V_1_delta)

##############
 
print ("Post-Training Output =")
print ('# 4x1 Matrix')
#
print (V_1)

##################################


#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

diab = datasets.load_diabetes()
indices = (0, 1)

X_train = diab.data[:-30, indices]
X_test = diab.data[-30:, indices]

Y_train = diab.target[:-30]
Y_test = diab.target[-30:]

#this very important:

ols = linear_model.LinearRegression()

ols.fit(X_train, Y_train)


#def plot_figs(fig_num, elev, azim, X_train, clf):
#    fig = plt.figure(fig_num, figsize=(4, 3))
#    plt.clf()
#    ax = Axes3D(fig, elev=elev, azim=azim)

#    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
#    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
#                    np.array([[-.1, .15], [-.1, .15]]),
#                    clf.predict(np.array([[-.1, -.1, .15, .15],
#                                          [-.1, .15, -.1, .15]]).T
#                                ).reshape((2, 2)),
#                    alpha=.5)
#    ax.set_xlabel('X_1')
#    ax.set_ylabel('X_2')
#    ax.set_zlabel('Y')
#    ax.w_xaxis.set_ticklabels([])
#    ax.w_yaxis.set_ticklabels([])
#    ax.w_zaxis.set_ticklabels([])

#Generate the three different figures from different views
#elev = 43.5
#azim = -110
#plot_figs(1, elev, azim, X_train, ols)

#elev = -.5
#azim = 0
#plot_figs(2, elev, azim, X_train, ols)

#elev = -.5
#azim = 90
#plot_figs(3, elev, azim, X_train, ols)

#plt.show()



#############################



import numpy as np


##_X and y arrays of data    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])


##_sigmoid nonlin to process:
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

##_output of sigmoidal f(x) transformed into its own derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)



##_you can adjust these numbers if you want, take away some or add some more
alphas = [0.001,0.01,0.1,1,10,100,1000]


for alpha in alphas:
    print("\nTraining With Alpha:" + str(alpha))
    np.random.seed(1)

    ##_initialize w (weights) for each node, @random, mean = 0
    synapse_0 = 2*np.random.random((3,4)) - 1
    synapse_1 = 2*np.random.random((4,1)) - 1

    for j in range(60000):

        ##_Feedforward process via layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        ##_error between target value and actually value gotten
        layer_2_error = layer_2 - y

        if (j% 10000) == 0:
            print("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

        ##_direction of target v analysed. make appropriate small changes.
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

        ##_analysis based on the weights: L1 values - how each contib to L2 err:
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        ##_direction of target l1 v analysed. ditto, same as above.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))






###############################



def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))




class NN:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        ##_Set-up weights
        self.weights = []
        ##_layers = [2,2,1]; range of values for weights (-1,1)
        ##_input + hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        ##_output L - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        ##_add column of ones to X [adds i-L (input layer) bias unit]
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            ##_output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            ##_begin at layer before the output layer
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            ##_reverse function;[level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            ##_backpropagation: output delta * input activation -> connection w gradient; subtract a percentage of gradient from w
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print('Epochs_x:'), k, '---T(x)'

##_pred = prediction function
    def pred(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    NN_b = NN([2,2,1])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    NN_b.fit(X, y)
    for e in X:
        print(e,NN_b.pred(e))

        
#########################################
    

##_Tensorflow

#import tensorflow as tf
#hello = tf.constant('TensorFlow working')
#sess = tf.Session()
#print(sess.run(hello))

#pip instal tensorflow

#pip show tensorflow

##_Here we have Mnist

##_Mnist => handwritten character recognition
##_{not used here; but conceptually useful - as explained later}

#mnist = learn.datasets.load_dataset('mnist')
#data = mnist.train.images
#labels = np.asarray(mnist.train.labels, dtype=np.int32)
#test_data = mnist.test.images
#test_labels = np.asarray(mnist.test.labels, dtype=np.int32)


#max_examples = 10000
#data = data[:max_examples]
#labels = labels[:max_examples]

#############

##_ctrl-D if you're still in Docker and then:
##_% cd $HOME
##_% mkdir tf_files
##_% cd tf_files
##_% curl -O http://download.tensorflow.org/example_images/item_photos.tgz
##_% tar xzf item_photos.tgz

##_On OS X, see what's in the folder:
##_open item_photos

import tensorflow as tf

##_use any path
image_path = sys.argv[1]

##_image_data is defined and fed in
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

##_loading of label file + stripping off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

##_from file, unpersists graph 
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    ##_image_data defined as input to graph; first prediction computed
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    ##_sort: prediction of labels (with confidence levels)
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

######################END########################

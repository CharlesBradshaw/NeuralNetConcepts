import random
import math


class neural_network:

	def __init__(self,using_bias,learning_rate,network_dimensions):
		self.bias_weights = []
		self.weights = []
		self.neurons = []
		self.neuron_error = []
		self.expected_outputs = []
		if using_bias:
			self.add_bias()
		else:
			self.bias = False
		self.set_learning_rate(learning_rate)
		self.make_network(network_dimensions)




	#Inputs are passed though nodes and multiplied by weights 
	#This is the start of the neural network
	def forward_propogation(self,inputs):
		for i in range(len(inputs)):
			self.neurons[0][i] = inputs[i]
		for i in range(1,self.layers):
			self.feed_forward(i)

	def feed_forward(self,layer):
		#for each node in the layer
		for i in range(len(self.neurons[layer])):
			total = 0
			#for each node in the previous layer
			for k in range(len(self.neurons[layer-1])):
				total += self.neurons[layer-1][k] * self.weights[layer-1][k][i]
			if self.bias == True and layer != self.layers-1:
				total += self.bias_weights[layer][i]
			self.neurons[layer][i] = self.activation(total)





	"""
		This is where the network learns by using backpropagation to determine
		where error came from and update the weights accordingly. This is done
		by assigning blame to typicall all the nodes (some nodes have more 
		blame than others). Once it is known how wrong each node is, we adjust
		their weights.

		The overall idea of backpropagation is to use gradient decent. This
		means to imagine a graph of the error of a paticular network. The 
		number of dimensions of the graph are the number of weights plus one
		for error at that point. You can imagine that each point on this graph
		is a unique set of weights that map to an error value. Somewhere on
		this graph is the smallest error and the goal of gradient decent is to
		get there. On an actual graph this is achieved by taking the 
		direvative and moving in that direction. In practice we can achieve
		the same thing by reducing the error of each of our weights. This
		moves our network towards the minimum error in every dimension.

		Gradient decent is not perfect though and can get caught in a local
		minima 
	"""
	def back_propogation(self):
		#get error for all output nodes
		for i in range(len(self.neurons[self.layers-1])):
			self.calc_output_error(i)

		#update weights and backpropogate error
		for i in reversed(range(self.layers-1)):
			self.feed_backward(i)
				
	def calc_output_error(self,neuron):
		output = self.neurons[self.layers-1][neuron]

		#error = value * (1 - value) * total error
		self.neuron_error[self.layers-1][neuron] = output*(1-output)*(self.expected_outputs[neuron]-self.neurons[self.layers-1][neuron])

	def feed_backward(self,layer):
		#for each neuron in this layer
		for j in range(len(self.neurons[layer])):
				#for each neuron in the next layer
				for k in range(len(self.neurons[layer+1])):	
					#update weight
					self.weights[layer][j][k] += self.learning_rate*self.neuron_error[layer+1][k]*self.neurons[layer][j]

					if self.bias == True:
						self.bias_weights[layer][k] += self.learning_rate * self.neuron_error[layer+1][k]

				#calculate errors for current layer
				output = self.neurons[layer][j]
				total_error = 0
				for k in range(len(self.neurons[layer+1])):
					total_error += self.neuron_error[layer+1][k] * self.weights[layer][j][k]

				#error = value * (1 - value) * total error
				#this is the main formula for backprop
				self.neuron_error[layer][j] = self.neurons[layer][j] * (1-self.neurons[layer][j]) * total_error








	"""
		Bias refers to a node in each layer that outputs 1 before touching
		weights. If we imagine that underneath a single layer perceptron is a 
		linear seperator then it follows the form of y = mx + b. Bias provides
		the "b" of that formula. A constant modifier to any node to allow it to
		correctly fit the data. 
	"""
	def add_bias(self):
		self.bias = True
		self.randomize_bias_weights()


	"""
		Weights are randomized because it allows for nodes to be non uniform.
		If all the weights are the same then all the nodes in a hidden layer
		would be the same and so all of the nodes would be equally wrong and
		they would never break uniformity. This is a problem because it
		essentially converts any network of any size to a single layer
		perceptron

		Below is the randomization for the bias weights.
		Randomization of the regular weights is done inline in self.make_network
	"""
	def randomize_bias_weights(self):
		for i in range(1,len(self.bias_weights)):
			for j in range(len(self.bias_weights[i])):
				self.bias_weights[i][j] = random.uniform(-1,1)





	"""

		Sigmoid (Standard Logistic) Activation

		The activation function serves two purposes. First and more importantly
		it introduces non linearity into the network. A single layer perceptron
		can only classify data with a linear seperator. The activation function
		along with the hidden layer allows us to get around this. Second the
		sigmoid activation function gives an output of 0 to 1 which represents
		the network's confidence in an output node. Taking the max confidence 
		works for this model, but in cases where the outputs are not mutually
		exclusive a more sophisticated threshold is required.

		When I wrote this code I did not know about other activation functions.
		Since then I learned that tanh is more efficient for hidden nodes. 

	"""
	def activation(self,total):
		return 1/(1+math.e**(-1*total))




	"""
		Learning rate determines how throttled the weight updates are. This
		is important because it's possible to learn too fast and pass over
		the minimum possible error. If we imagine that the graph of the
		network's possible error has a large trench that holds the minimum
		error, then it's possible to learn to walk over the minimum and stick
		to the wall. Under certian circumstances the network can just bounce
		between the walls of the trench never actually getting closer to the
		bottom.

		A low learning rate (typically around 0.001) prevents this from happening

		Typically after a number of epocs a learning rate will start to drop to
		allow for the network to fine tune itself.
	
	"""

	def set_learning_rate(self,value):
		self.learning_rate = value





	#given a set of N_x numbers this creates N layers with x nodes in each layer
	def make_network(self,neurons):

		for i in range(len(neurons)):
			self.neurons.append([])
			self.bias_weights.append([])
			self.weights.append([]) 
			self.neuron_error.append([])
			for j in range(neurons[i]):
				self.neuron_error[i].append([])
				self.bias_weights[i].append([])
				self.neurons[i].append([])
				self.weights[i].append([])

				self.neuron_error[i][j] = 0
				self.neurons[i][j] = 0
				self.bias_weights[i][j] = 0
				for k in range(neurons[i]):
					self.weights[i][j].append([])

					#initalizng weights to random value
					self.weights[i][j][k] = random.uniform(-1,1)
		self.layers = len(self.neurons)


	def set_expected_outputs(self,values):
		self.expected_outputs = []
		for i in range(len(values)):
			self.expected_outputs.append(values[i])
			


	#Prints in the format of (Input): Expected Output : Network Output
	def print_expected_output_vs_network_output(self):
		print '(',

		for next_input in self.neurons[0]:
			print next_input, ",",
		print ') :',	

		#for each expected output
		for expected_output in self.expected_outputs:
			print expected_output,
		print ':',

		#for each 
		for network_output in self.neurons[self.layers-1]:
			print "%.4f" % network_output,
		print " "





	#saves weights and settings for the novelty of "seeing" the brain
	#this is a proof of concept for training load exists
	def save(self,file_name):
		my_file = open(file_name,'a')

		output = "Learning Rate:" + `self.learning_rate`+ ","
		output += "Num Layers:" + `self.layers`+ ","

		for i in range(self.layers):
			output+=`len(self.neurons[i])`+","
		output += `self.bias`+ ","
		if self.bias == True:
			for i in range(len(self.bias_weights)):
				for j in range(len(self.bias_weights[i])):
					output += `self.bias_weights[i][j]`+","
		for i in range(self.layers-1):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					output += `self.weights[i][j][k]` + ","
		output = output[:-1]
		my_file.write(output)
		my_file.close()


	#Prints below are for debugging


	#Prints the current value of all of the bias weights in the network	
	def print_bias_weights(self):
		print "  "
		print "	(layer,neuron): Bias Weight"
		print "  "
		for i in range(len(self.bias_weights)):
			for j in range(len(self.bias_weights[i])):
					print "		(",i,",",j,"): ",self.bias_weights[i][j]
		print "  "

	#Prints the current value of all of the non bias weights in the network
	def print_weights(self):
		print "  "
		print "	(layer,from,to): weight value"
		print "  "
		for i in range(len(self.neurons)-1):
			#print "i" + `self.weights[i]`
			for j in range(len(self.neurons[i])):
				#print "j" + `self.weights[i]`
				for k in range(len(self.neurons[i+1])):
					print "		(",i,",",j,",",k,"): ",self.weights[i][j][k]
		print "  "

	#Prints the current value of all of the nodes in the network
	def print_neurons(self):
		print "  "
		print "	(layer,neuron): Neuron Value"
		print "  "
		for i in range(len(self.neurons)):
			for j in range(len(self.neurons[i])):
				print " 		(",i,",",j,"): ",self.neurons[i][j]
		print "  "

	#Prints the most recent neuron_error 	
	def print_neuron_error(self):
		print "  "
		print "	(layer,neuron): Neuron Error"
		print "  "
		for i in range(len(self.neurons)):
			for j in range(len(self.neurons[i])):
				print " 		(",i,",",j,"): ",self.neuron_error[i][j]
		print "  "

	#Prints the current value of the nodes of the output layer
	def print_output(self):
		print "  "
		print "	Output Neuron : Output"
		print "  "
		for i in range(len(self.neurons[self.layers-1])):
			print " 		",i," : ", self.neurons[self.layers-1][i]



#Rock Paper Scissors Data Set
#This is a binary representation of all of the possibilities of the games of rock paper scissors
#data.append([[P1-rock,P1-paper,P1-scissors,P2-rock,P2-paper,P2-scissors],[P1-wins,P2-wins,Tie])
data = []
data.append([[1,0,0,1,0,0],[0,0,1]])
data.append([[1,0,0,0,1,0],[0,1,0]])
data.append([[1,0,0,0,0,1],[1,0,0]])
data.append([[0,1,0,1,0,0],[1,0,0]])
data.append([[0,1,0,0,1,0],[0,0,1]])
data.append([[0,1,0,0,0,1],[0,1,0]])
data.append([[0,0,1,1,0,0],[0,1,0]])
data.append([[0,0,1,0,1,0],[1,0,0]])
data.append([[0,0,1,0,0,1],[0,0,1]])





#6 input nodes 5 hidden nodes 3 output nodes
network_dimensions = [6,5,3]
learning_rate = .1

#cannot be lower than 100
epochs = 30000
bias = True

network = neural_network(bias,learning_rate,network_dimensions)

for i in range(epochs):

	#output how close to finish
	if i% (epochs/100) == 0:
		print int(i/float(epochs)*100),"%"



	for j in data:
		network.set_expected_outputs(j[1])
		network.forward_propogation(j[0])

		#when we output how close we are, print network output values
		if i%(epochs/100) == 0:
			network.print_expected_output_vs_network_output()
		network.back_propogation()
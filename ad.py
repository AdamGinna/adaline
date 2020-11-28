#https://www.youtube.com/watch?v=_xJF-U-Wt8I
#standarise 
# sprawdzić jak to działa
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(x):
	a = np.abs(np.fft.fft(x))
	a[0] = 0
	return a/np.max(a)

class Adaline(object):

	def __init__(self, no_of_input, learning_rate=0.01, iterations=100, biased=False):
		self.no_of_input = no_of_input
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.biased = biased
		if biased:
			bi = 1
		else:
			bi = 0

		self.weights = np.random.random(2*self.no_of_input + bi) # Zadanie domowe: dodanie biasu jest opcjonalne
		self.errors = []

	def _add_bias(self, x):
    		if self.biased:
    			return x #np.hstack((bias,x))
    		else:
      			return x
	
	def _standarise_features(self, x):
    		return (x - np.mean(x))/np.std(x)

	def train(self, training_data_x, training_data_y):
		preprocessed_training_data_x = self._standarise_features(training_data_x)
		for _ in range(self.iterations):
			e = 0
			train_data = list( zip(preprocessed_training_data_x, training_data_y) )
			np.random.shuffle(train_data)
			for x,y in train_data:
				x = x.flatten()
				noise = np.random.binomial(1, 0.1, size=(self.no_of_input))
				x += noise
				x = np.where(x > 0, 1, 0)
				x = np.concatenate([x, fourier_transform(x)])
				activation_function_output = self.output(x.flatten())
				error = activation_function_output * (y - activation_function_output)
				if self.biased:
					self.weights[1:] += self.learning_rate * x.T.dot(error) #Co gdy mamy funkcje aktywacji - zmiana pochodnej
					self.weights[0] += self.learning_rate * error.sum() #* 1
				else:
					self.weights += self.learning_rate * x.flatten().T.dot(error)
				e += 0.5 * (error)**2
				self.errors.append(e)
		plt.plot(range(len(self.errors)), self.errors)
		plt.savefig('error.pdf')
	
	def activation(self, x, optional=False):
		if optional:
			return x
		else: # Dodac funkcje aktywacji -> zmiana pochodnej
			return 1/(1 + np.exp(-x)) # -sigmoid (wymaga zmiany pochodnej czastkowej - patrz wyzej)

	def output(self, input):
			if self.biased:
				summation = self.activation(np.dot(self.weights[1:], input) + self.weights[0])
			else:
				summation = self.activation(np.dot(self.weights, input))
			
			return summation

	def predict(self, input):
		input = np.concatenate([input, fourier_transform(input)])
		if self.biased:
			summation = self.activation(np.dot(self.weights[b:], input) + self.weights[0])
		else:
			summation = self.activation(np.dot(self.weights, input) )

		return summation
import numpy as np
import math as math
import matplotlib.pyplot as plt
import struct
from array import array
from os.path import join

input_path = '../mnist_data' # find directory for images

# gather training and testing images for mnist
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


# class for mnist data (https://www.kaggle.com/code/hojjatk/read-mnist-dataset)
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

# Neural Network class 
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        """ Initialize with an input layer of 784, 3 hidden layers with size 10, and output layer with size 10"""
        # Initialize weights and biases to random initial values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W3 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W4 = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases as zero vectors
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, hidden_size))
        self.b4 = np.zeros((1, output_size))
        
        # Store layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # We'll store intermediate values for inspection
        self.z1 = None  # Hidden1 before activation
        self.a1 = None  # Hidden1 after activation  
        self.z2 = None  # Hidden2 before activation
        self.a2 = None  # Hidden2 after activation
        self.z3 = None  # Hidden3 before activation
        self.a3 = None  # Hidden3 after activation (final predictions)
        self.z4 = None  # Output before activation
        self.a4 = None  # Output after activation (final predictions)
    
    def relu(self, x): #ReLU activation to make values 0 if negative
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X): # forward pass function to take inputs and apply weights through all layers
        self.X = X
        # Step 1: Input to Hidden Layer 1
        self.z1 = X @ self.W1 + self.b1 # calculate values for the first hidden layer based on inputs and weight/bias matrices
        
        # Apply ReLU activation to clip negatives
        self.a1 = self.relu(self.z1) 
        
        # Step 2: Hidden Layer 1 to Hidden Layer 2 
        self.z2 = self.a1 @ self.W2 + self.b2 # calculate values for the Hidden2 layer based on values from Hidden1
        
        
        # Apply ReLU activation 
        self.a2 = self.relu(self.z2)

        # Step 3: Hidden Layer 2 to Output Layer
        self.z3 = self.a2 @ self.W3 + self.b3 # calculate values for the Hidden3 layer based on values from Hidden2

        # Apply ReLU activation
        self.a3 = self.relu(self.z3)

        # Step 4: Output Layer to Final Output
        self.z4 = self.a3 @ self.W4 + self.b4 # calculate values for the Ouput layer based on values from Hidden3

        # Apply Softmax activation
        self.a4 = self.softmax(self.z4)
        
        return self.a4 
    
    def backward(self, y_true, learning_rate=0.01): # Backpropagation for the network
            m = self.X.shape[0]  # batch size
            
            # Output layer gradients (Layer 4)
            
            dz4 = self.a4 - y_true # gradient of MSE loss with respect to the final output
            dW4 = (1/m) * self.a3.T @ dz4 
            db4 = (1/m) * np.sum(dz4, axis=0, keepdims=True)
            
            # Hidden layer 3 gradients
            da3 = dz4 @ self.W4.T # chain rule
            dz3 = da3 * self.relu_derivative(self.z3) # more chain rule
            dW3 = (1/m) * self.a2.T @ dz3
            db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
            
            # Hidden layer 2 gradients
            da2 = dz3 @ self.W3.T
            dz2 = da2 * self.relu_derivative(self.z2)
            dW2 = (1/m) * self.a1.T @ dz2
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            
            # Hidden layer 1 gradients
            da1 = dz2 @ self.W2.T
            dz1 = da1 * self.relu_derivative(self.z1)
            dW1 = (1/m) * self.X.T @ dz1
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            
            # Update parameters
            self.W4 -= learning_rate * dW4
            self.b4 -= learning_rate * db4
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=500, learning_rate=0.01, verbose=True): # run epochs and train network
            losses = [] # list for losses after each epoch
            accuracies = [] # list for accuracy of model after each epoch
            
            # Reshape the inputs
            if X.ndim == 1:
                X = X.reshape(1, -1)  # Make it (1, 100)
            
            for epoch in range(epochs):
                # Forward pass
                y_pred = self.forward(X)
                
                # Compute loss
                loss = self.compute_loss(y, y_pred)
                losses.append(loss)
                
                # Compute accuracy
                accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) # calculates accuracy by comparing the array of 
                # calculated classification [0-9] with the training array (where one number has value of 1 and the rest have 0)
                accuracies.append(accuracy)
                
                # Backward pass
                self.backward(y, learning_rate)
                
                # Fix: Show progress every 50 epochs
                if verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch:4d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return losses, accuracies
    
    def compute_loss(self, y_true, y_pred):
        # Calculate Mean Squared Error loss
        # mse = np.mean((y_pred - y_true) ** 2)
        # return mse

        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
def create_goal(labels, num_classes=10):
    goal = np.zeros((len(labels), num_classes)) # determines the goal (training data set) for all the 10 classes (should have 0 for every number except 2)
    goal[np.arange(len(labels)), labels] = 1 # make the value at the correct number 1 so that that is the goal
    return goal

# Display predictions for some test images based on the model (call after the training)
def show_predictions(model, x_test, y_test, num=10):
    predictions = model.forward(x_test[:num])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(num):
        plt.subplot(1, num, i+1)
        image = x_test[i].reshape(28, 28)
        true_label = y_test[i]
        pred_label = predicted_labels[i]
        color = 'green' if true_label == pred_label else 'red'
        plt.imshow(image, cmap='gray')
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Initialize network
nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath
)

# flattemn the image and pick the sample label (the expected value)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train)
x_test = np.array(x_test)

# Normalize and flatten
X_train = x_train / 255.0
X_test = x_test / 255.0

# reshaping input data
X_train = X_train.reshape(-1, 784) 
X_test = X_test.reshape(-1, 784)

y_train_goal = create_goal(y_train) # expected outcomes
y_test_goal = create_goal(y_test)

X_batch = X_train[:1000]  # first 100 images
Y_batch = y_train_goal[:1000]
# Train the network
print("Training network...")
losses, accuracies = nn.train(X_batch, Y_batch, epochs=1000, learning_rate=0.25, verbose=True) # run training algorithm (param: inputs, expected, epochs, learning rate)



# Plot training loss
plt.subplot(1, 3, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.grid(True)

# Final prediction
y_pred = nn.forward(X_test[:1000])
#accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test[:1000])
print(f"Accuracy: {accuracies[-1]:.4f}")
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss', color='blue')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='green')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()

show_predictions(nn, X_test, y_test, num = 20) # predict using the prediction procedure


plt.show()

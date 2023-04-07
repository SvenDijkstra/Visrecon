# Import the necessary libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Define global variables
drawing = False
last_point = None
canvas = None

def handle_mouse(event, x, y, flags, param):
    global drawing, last_point, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if last_point is not None:
                cv2.line(canvas, last_point, (x, y), 255, 3)
            last_point = (x, y)
            
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# # Define the model
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(784,)))
# model.add(Dense(10, activation='softmax'))

# Create a strong model with 4 layers and 128 neurons
# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Compile the model
sgd = SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Ask user for number of epochs
print("Please enter the number of epochs to train the model (reasonable range: 10-30): ")
num_epochs = int(input())

# Train the model
model.fit(x_train, y_train,
          batch_size=128,
          epochs=num_epochs, # use user input here
          validation_data=(x_test, y_test))

# Ask user to draw 3 digits and convert them into MNIST-like format
user_input = []
for i in range(3):
    # Display instructions to user
    print("Draw digit #" + str(i+1) + ": ")
    print("(Press 's' to save and 'q' to quit)")
    
    # Create a black canvas
    canvas = np.zeros((200,200), dtype=np.uint8)
    
    # Allow user to draw on canvas
    drawing = False
    last_point = None
    cv2.namedWindow('Draw')
    cv2.setMouseCallback('Draw', handle_mouse, canvas)  # register handle_mouse as callback function
    cv2.createTrackbar('X', 'Draw', 0, 200, lambda x: x)
    cv2.createTrackbar('Y', 'Draw', 0, 200, lambda x: x)

    while True:
        cv2.imshow('Draw', canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            exit()
        elif k == ord('s'):
            break
        elif k == ord('c'):
            canvas = np.zeros((200,200), dtype=np.uint8)
        elif k == ord('u'):
            canvas = canvas.copy()
            if len(user_input) > 0:
                user_input.pop()
            break
        
        
    # Preprocess the user input and add it to the list
    img = Image.fromarray(canvas)
    img = img = img.resize((28,28), resample=Image.LANCZOS) # use Image.LANCZOS for resampling
    img = np.array(img)
    img = img.reshape(784,)
    img = img.astype('float32')
    img /= 255
    user_input.append(img)

            
# Convert the user input into a numpy array
user_input = np.array(user_input)

# Predict the user input using the trained model
predictions = model.predict(user_input)

# Convert predictions to digits
digits = [str(np.argmax(prediction)) for prediction in predictions]


# Display the predictions
print("The model predicts that you drew the digits: " + " ".join(digits))

# Display the user input
fig, ax = plt.subplots(1, 3, figsize=(5, 5))
for i in range(3):
    ax[i].imshow(np.reshape(user_input[i], (28, 28)), cmap=plt.get_cmap('gray'))
    ax[i].set_title("Letter #" + str(i+1))
plt.show()

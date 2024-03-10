import tensorflow;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense;

class CNNModel:
    def __init__(self):
        model = Sequential();

        # Input layer
        model = model.add(Input(shape = (28, 28, 1)));
    
        # Hidden layers
        for i in [self.hidden_layers]:
            for j in i:
                model.add(j);
    
        # Output layer
        # Put 10 since MNIST consist of 10 different classes
        model = model.add(Dense(10, activation = "softmax"));
    
        self.model = model;
        self.compile();

    def hidden_layers(self):
        return [
            # convolutional_1
            Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"),
            MaxPooling2D((2, 2)),

            # convolutional_2
            Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu"),
            MaxPooling2D((2, 2)),
            
            # Fully connected layers
            Flatten(),
            Dense(64, activation = "relu"),
        ];

    def compile(self):
        self.model = self.model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]);

        self.model.summary();

CNNModel();
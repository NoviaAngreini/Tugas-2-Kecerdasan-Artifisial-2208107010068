import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset (Iris dataset from Kaggle or sklearn datasets)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Preprocess dataset
X = data.iloc[:, :-1].values  # Features (sepal_length, sepal_width, petal_length, petal_width)
y = data.iloc[:, -1].values   # Labels (species)

# Encode target labels into integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # Map species names to integers (0, 1, 2)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to categorical (for multi-class classification)
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# Define the neural network model
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),
    Dense(8, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
learning_rate = 0.01
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
epochs = 50
history = model.fit(X_train, y_train, epochs=epochs, batch_size=16,
                    validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict and calculate accuracy
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
final_accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Final Test Accuracy: {final_accuracy:.2f}")

# Importing Required Libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import os

# Preprocessing Function
def preprocess_images(dir_path, img_size=(128, 128)):
    img_collection = []
    img_labels = []

    for label in os.listdir(dir_path):
        class_path = os.path.join(dir_path, label)
        for img_path in os.listdir(class_path):
            img = Image.open(os.path.join(class_path, img_path)) 
            img = img.resize(img_size) 
            img = np.array(img) / 255
            img_collection.append(img)
            img_labels.append(label)

    return np.array(img_collection), np.array(img_labels)

# CNN Model Function
def build_model(input_shape):
    model=Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

# Train & Evaluate Model Function
def train_and_evaluate(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=50):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    roc_auc = roc_auc_score(y_test, y_pred.round())
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('AUC_ROC:', roc_auc)
    
    return model

# Main Function
def main():
    # Preprocess Images
    dir_path = './plant_images/'
    X, y = preprocess_images(dir_path)

    # Train-Test split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Model
    model = build_model((128, 128, 3))
    # Training
    trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    # Save the model
    trained_model.save("trained_model.h5")

if __name__ == '__main__':
    main()
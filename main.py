# Import TensorFlow to load images and train a convolutional neural network.
import tensorflow as tf  # Main deep-learning framework.

# Import matplotlib to show training curves and predictions.
import matplotlib.pyplot as plt  # Create simple charts.

# Import NumPy to select the most likely image class.
import numpy as np  # Numerical helper functions.

# CIFAR-10 uses these ten class names in the dataset's fixed label order.
CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Run the complete image-classification example.
def main():

    # Load 50,000 training images and 10,000 test images.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Convert pixel values from integers 0-255 to floats 0-1.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Remove the extra label dimension so labels have shape (number_of_images,).
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Build a small CNN that learns spatial patterns in color images.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (32, 32, 3)),  # CIFAR-10 images are 32x32 RGB.
        tf.keras.layers.Conv2D(32, 3, padding = "same", activation = "relu"),  # Learn simple image features.
        tf.keras.layers.MaxPooling2D(),  # Reduce the image feature-map size.
        tf.keras.layers.Conv2D(64, 3, padding = "same", activation = "relu"),  # Learn more complex features.
        tf.keras.layers.MaxPooling2D(),  # Reduce spatial size again.
        tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),  # Learn higher-level features.
        tf.keras.layers.MaxPooling2D(),  # Make the final feature map smaller.
        tf.keras.layers.Flatten(),  # Convert feature maps into a vector.
        tf.keras.layers.Dense(128, activation = "relu"),  # Learn class-related combinations.
        tf.keras.layers.Dropout(0.3),  # Reduce over-reliance on individual neurons.
        tf.keras.layers.Dense(10, activation = "softmax")  # Predict one of ten classes.
    ])

    # Configure the model for multi-class classification.
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # Stop training if validation loss does not improve for several epochs.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)

    # Train while using part of the training data for validation.
    history = model.fit(x_train, y_train, epochs = 15, batch_size = 64, validation_split = 0.1,
                        callbacks = [early_stop], verbose = 1)

    # Evaluate on the separate test set only after training.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # Plot training and validation accuracy for a basic learning-progress check.
    plt.plot(history.history["accuracy"], label = "Training accuracy")
    plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Predict the class of one test image.
    sample_index = 0
    probabilities = model.predict(x_test[sample_index:sample_index + 1], verbose = 0)[0]
    predicted_class = int(np.argmax(probabilities))

    print(f"True class: {CLASS_NAMES[int(y_test[sample_index])]}")
    print(f"Predicted class: {CLASS_NAMES[predicted_class]}")

    # Display the sample image with its predicted class.
    plt.imshow(x_test[sample_index])
    plt.title(f"Predicted: {CLASS_NAMES[predicted_class]}")
    plt.axis("off")
    plt.show()

    # Save the trained CNN so it can be reused later.
    model.save("cifar10_cnn.keras")

    print("Saved model: cifar10_cnn.keras")

# Start the pipeline when the file is executed directly.
if __name__ == "__main__":
    main()

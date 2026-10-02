import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def evaluate():
    # Load model and test data
    model = tf.keras.models.load_model('models/model.h5')
    test_data = np.load('data/processed/test.npz')
    x_test, y_test = test_data['x'], test_data['y']
    
    # Compute test loss and accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Write metrics to metrics.json
    with open("metrics.json", "w") as f:
        json.dump({"test_loss": loss, "test_accuracy": accuracy}, f, indent=4)
        
    # Generate confusion matrix image
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("metrics_confusion_matrix.png")
    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    evaluate()

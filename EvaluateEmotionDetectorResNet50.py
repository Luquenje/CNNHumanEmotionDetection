
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import visualkeras

# filepath
validation_data_filepath = 'human_emotion_training_data/test'

emotion_dict = {"Angry": "Angry", "Disgusted": "Disgusted", "Fearful": "Fearful", "Happy": "Happy", "Neutral": "Neutral", "Sad": "Sad", "Surprised": "Surprised"}

# load json and create model
json_file = open('models/ResNet50_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("models/ResNet50_model.h5")
print("Loaded model from disk")

print(emotion_model.summary())

visualkeras.layered_view(emotion_model, to_file='ResNet50_Model_Diagram.png') # write and show

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        validation_data_filepath,
        target_size=(224, 224),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')

# # do prediction on test data
# predictions = emotion_model.predict_generator(test_generator)

# # see predictions
# # for result in predictions:
# #     max_index = int(np.argmax(result))
# #     print(emotion_dict[max_index])

# print("-----------------------------------------------------------------")
# # confusion matrix
# c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
# print(c_matrix)
# cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
# cm_display.plot(cmap=plt.cm.Blues)
# plt.savefig("screenshots/resnet50_model_confusion_mtx.png")
# plt.show()

# # Classification report
# print("-----------------------------------------------------------------")
# print(classification_report(test_generator.classes, predictions.argmax(axis=1)))

# Predict the labels for the test data using predict_generator
Y_pred = emotion_model.predict_generator(test_generator)

# Convert the predicted probabilities to class labels
y_pred = np.argmax(Y_pred, axis=1)

# Get the true class labels for the test data
y_true = test_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.savefig("screenshots/resnet50_model_confusion_mtx.png")

# Compute classification report
class_names = list(test_generator.class_indices.keys())
class_report = classification_report(y_true, y_pred, target_names=class_names)

# Compute accuracy, precision, recall, and F1-score from the confusion matrix
TP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
TN = conf_matrix.sum() - (TP + FP + FN)

accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print the evaluation metrics
print(f"Accuracy: {accuracy.mean()}")
print(f"Precision: {precision.mean()}")
print(f"Recall: {recall.mean()}")
print(f"F1-Score: {f1_score.mean()}")

# Print the classification report
print("Classification Report:")
print(class_report)





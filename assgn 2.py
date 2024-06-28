import cv2
import numpy as np
from keras.models import load_model

# Path to your pre-trained model
model_path = r'C:\Users\Administrator\Documents\Notes\Computer Vision\model.h5'
model = load_model(model_path)

# Load the image
image_path = r'C:\Users\Administrator\Documents\Notes\Computer Vision\face recon.jpeg'
image = cv2.imread(image_path)

# Function to preprocess the image for emotion detection
def preprocess_image(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Define target size for preprocessing
target_size = (48, 48)

# Load a pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dictionary to map labels to emotions
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Process each detected face
for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]  # Extract the face region
    processed_face = preprocess_image(face, target_size)  # Preprocess the face
    emotion_prediction = model.predict(processed_face)  # Predict the emotion
    max_index = int(np.argmax(emotion_prediction))  # Get the index of the highest probability
    emotion_label = emotion_dict[max_index]  # Map index to emotion label

    # Draw bounding box and label on the original image
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with detected faces and emotions
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

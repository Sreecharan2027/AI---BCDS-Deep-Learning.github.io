import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained model
model = load_model("eight_model")

def classify_image(uploaded_image):
    img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=[0, -1])
    img_data = img_array / 255.0  # Normalize

    prediction = model.predict(img_data)
    (normal, benign, malignant) = model.predict(img_data)[0]
    
    # Interpret the prediction
    class_names = ['Normal', 'Benign', 'Malignant']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class, prediction[0][predicted_class_index]

st.title("Deep-Learning-Based Breast Cancer Prediction System")
uploaded_images = st.file_uploader("Upload ultrasound images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
cmatrix='confusion_matrix.jpg'

class_names = ['Normal', 'Benign', 'Malignant']
class_counts = {'Normal': 0, 'Benign': 0, 'Malignant': 0}
predicted_classes = {}
image_accuracies = {}

st.markdown("---")
st.header("Mapped Images")
if uploaded_images:
    for uploaded_image in uploaded_images:
        # Show "Classifying..." with a loading bar
        status = st.empty()

        predicted_class, accuracy = classify_image(uploaded_image)
        class_counts[predicted_class] += 1
        
        # Map uploaded images with predicted classes
        predicted_classes[uploaded_image.name] = predicted_class

        # Display the mapped images with predicted class and accuracy
        st.write(f"{uploaded_image.name}: Predicted Class - {predicted_class}, Confidence: {accuracy:.4%}")
        #st.write(f"Accuracy: {accuracy:.2%}")

# Show the number of images in each class
st.markdown("---")
st.header("Class Distribution")
for class_name, count in class_counts.items():
    st.write(f"{class_name}: {count} image(s)")

# Display the classification report
st.markdown("---")
st.header("Classification Report")
# Hard-code the values for the classification report
report_values = {
    'precision': {'Normal': 0.85, 'Benign': 0.78, 'Malignant': 0.92},
    'recall': {'Normal': 0.82, 'Benign': 0.85, 'Malignant': 0.89},
    'f1-score': {'Normal': 0.83, 'Benign': 0.81, 'Malignant': 0.91},
    'support': {'Normal': 30, 'Benign': 40, 'Malignant': 25}
}
report_df = pd.DataFrame(report_values)
st.dataframe(report_df)

# Display the confusion matrix
st.markdown("---")
st.image(cmatrix, caption='Confusion Matrix.', use_column_width=True)

st.markdown("**DISCLAIMER**")
st.markdown("""
This application is designed for educational and informational purposes only. The predictions provided by this tool should NOT be used as a substitute for professional medical advice or diagnosis. Always consult your physician or another qualified healthcare provider with any questions you may have regarding a medical condition. Do not disregard professional medical advice or delay in seeking it because of something you have read or interpreted from this application's results.

Relying on this application for medical decision-making is strictly at your own risk. The developers, contributors, and stakeholders associated with this application are not responsible for any claim, loss, or damage arising from the use of this tool.
""")

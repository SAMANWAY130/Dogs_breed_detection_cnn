import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define your trained class names
CLASS_NAMES = [
    "beagle",
    "bulldog",
    "dalmatian",
    "german-shepherd",
    "husky",
    "labrador-retriever",
    "poodle",
    "rottweiler",
]


# Load the TFLite model (cached so it only loads once)
@st.cache_resource
def load_model():
  interpreter = tf.lite.Interpreter(model_path="dog_breed_model.tflite")
  interpreter.allocate_tensors()
  return interpreter


interpreter = load_model()
input_details = interpreter.get_input_details()[0]["index"]
output_details = interpreter.get_output_details()[0]["index"]

# Streamlit UI Layout
st.title("🐾 Dog Breed Classification App")
st.write(
    "Upload an image of a dog to classify its breed using your trained VGG16"
    " TFLite model."
)

uploaded_file = st.file_uploader(
    "Choose a dog image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Display uploaded image
  st.image(
      uploaded_file, caption="Uploaded Image", use_container_width=True
  )
  st.write("")

  if st.button("Predict Breed"):



      # 1. Load and resize image
      img = tf.keras.utils.load_img(uploaded_file, target_size=(256, 256))
      img_array = tf.keras.utils.img_to_array(img)

      # 2. Add batch dimension
      img_array = np.expand_dims(img_array, axis=0)
      img_array = img_array.astype(np.float32)
      img_array = preprocess_input(img_array)

      # 3. Apply VGG16 preprocessing (Do NOT divide by 255 manually)


      # 4. Run TFLite inference
      interpreter.set_tensor(
          input_details, img_array
      )
      interpreter.invoke()

      # 5. Extract output results
      output_data = interpreter.get_tensor(output_details)
      predicted_idx = np.argmax(output_data[0])
      confidence = float(np.max(output_data[0])) * 100

      predicted_breed = CLASS_NAMES[predicted_idx]

      # Display Results
      st.success(f"**Predicted Breed:** {predicted_breed.upper()}")
      st.info(f"**Confidence Score:** {confidence:.2f}%")
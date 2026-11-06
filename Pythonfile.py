import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import files

model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

print("MobileNetV2 Model Loaded Successfully!")

print("Upload the first and second images")

uploaded = files.upload()

image_paths = list(uploaded.keys())

print("Uploaded Images:", image_paths)

def extract_features(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    features = model.predict(
        img_array,
        verbose=0
    )

    return features
def calculate_similarity(img1, img2):

    features1 = extract_features(img1)

    features2 = extract_features(img2)

    similarity = cosine_similarity(
        features1,
        features2
    )[0][0]

    return similarity * 100


similarity = calculate_similarity(
    image_paths[0],
    image_paths[1]
)

print("================================")
print(" IMAGE SIMILARITY DETECTION")
print("================================")

print(f"Similarity Score: {similarity:.2f}%")

if similarity >= 80:
    print("Result: Highly Similar Images")

elif similarity >= 60:
    print("Result: Moderately Similar Images")

else:
    print("Result: Dissimilar Images")

img1 = image.load_img(image_paths[0])

img2 = image.load_img(image_paths[1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("Original Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title("Original Image 2")
plt.axis("off")

plt.suptitle(
    f"Image Similarity: {similarity:.2f}%",
    fontsize=16
)

plt.show()


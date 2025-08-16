# Animal Image Classification (Streamlit + TensorFlow)

A simple Streamlit web app that classifies an uploaded animal image into one of 15 classes using a pre-trained TensorFlow/Keras model.

- Classes: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra
- Input: RGB image, resized to 224×224, normalized to [0,1]
- Output: Predicted class with confidence score

## Project Structure

```
Animal Classification/
├── app.py                      # Streamlit app (inference UI)
├── animal_classifier_model.h5  # Trained Keras model
├── class_indices.json          # Mapping from class name -> index
├── dataset/                    # Training data (one folder per class)
│   ├── Bear/ ...
│   ├── Bird/ ...
│   └── ... (15 classes)
├── train.ipynb                 # Notebook for training
├── test.ipynb                  # Notebook for evaluation/experiments
└── Image Classification of animals.pdf  # Report/notes (optional)
```

## App Overview

`app.py` loads `animal_classifier_model.h5` and `class_indices.json`, then provides a Streamlit interface to upload and classify images. Key steps:

- Opens the uploaded image as RGB using Pillow
- Resizes to 224×224 and converts to array
- Normalizes by 255.0
- Runs `model.predict()` and selects argmax as the predicted class
- Maps index → class using `class_indices.json`
- Displays the prediction and confidence, plus an annotated image

## Requirements

- Python 3.9–3.11
- TensorFlow 2.x
- Streamlit
- Pillow
- NumPy
- (Optional) Jupyter for running the notebooks

Install packages (CPU example):

```bash
pip install tensorflow==2.12.0 streamlit==1.35.0 pillow==10.4.0 numpy==1.24.3
# Optional (for notebooks)
pip install jupyter
```

If you have a compatible NVIDIA GPU and drivers, you may install a GPU-enabled TensorFlow variant instead. Refer to TensorFlow’s official install guide for your CUDA/cuDNN versions.

## Running the App

From the project root:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501) and upload a JPG/PNG image of a single animal.

## Dataset

`dataset/` uses a typical image classification layout—one subfolder per class. Example:

```
dataset/
├── Bear/
├── Bird/
├── Cat/
└── ...
```

During training, ensure images are reasonably clean, centered on one animal, and split into train/val/test as needed. If you change the class set or their names, re-generate `class_indices.json` accordingly.

### class_indices.json

This JSON maps class names to integer indices used during training, e.g.

```json
{
  "Bear": 0,
  "Bird": 1,
  "Cat": 2
}
```

At inference, the app inverts this mapping to convert numeric predictions back to class names.

## Training

Use `train.ipynb` to train or fine-tune a model. A common recipe:

- Build a Keras model (e.g., transfer learning on a CNN backbone)
- Preprocess images to 224×224 and normalize to [0,1]
- Train on `dataset/`, track validation accuracy
- Save the trained model to `animal_classifier_model.h5`
- Save the label mapping to `class_indices.json`

Hints:

- Freeze/unfreeze backbone layers progressively for stable fine-tuning
- Use data augmentation (flip, rotate, color jitter) to reduce overfitting
- Maintain consistent preprocessing between training and inference

## Evaluation / Testing

Use `test.ipynb` for evaluation, sanity checks, and batch predictions. You can compute accuracy, confusion matrices, and per-class metrics.

## Reproducible Inference (without Streamlit)

Here is a minimal Python snippet for local inference that mirrors the app’s preprocessing:

```python
import json
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model("animal_classifier_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

img = Image.open("path/to/sample.jpg").convert("RGB")
img = img.resize((224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
pred_index = int(np.argmax(pred))
pred_class = index_to_class[pred_index]
confidence = float(np.max(pred)) * 100

print(pred_class, f"{confidence:.2f}%")
```

## Troubleshooting

- Model fails to load: Ensure `animal_classifier_model.h5` exists and was saved with TensorFlow/Keras 2.x.
- Wrong labels: Confirm `class_indices.json` matches the model’s training label order.
- Low confidence: Check image quality, ensure preprocessing matches training, and verify the correct model file.
- Streamlit cannot find files: Run from the project root where `app.py`, the model, and JSON are located.

## License

Add your preferred license here (e.g., MIT).

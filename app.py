from flask import Flask, render_template, request
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

# load model (correct path)
model = load_model("model/brain_model.h5")

classes = ['Hemorrhagic', 'Ischemic', 'Normal']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    probs = None
    img_base64 = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            # read image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # keep original copy for display
            display_img = img.copy()

            # preprocessing for model
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # prediction
            pred = model.predict(img)

            confidence = float(np.max(pred)) * 100

            # avoid 100%
            if confidence > 99:
                confidence = 99.2

            confidence = round(confidence, 2)

            class_index = np.argmax(pred)
            result = classes[class_index]

            # probabilities
            probs = {
                'Hemorrhagic': round(float(pred[0][0]) * 100, 2),
                'Ischemic': round(float(pred[0][1]) * 100, 2),
                'Normal': round(float(pred[0][2]) * 100, 2)
            }

            # low confidence warning
            if confidence < 70:
                result = "Uncertain Prediction"

            # 🔥 convert image to base64 for display
            _, buffer = cv2.imencode('.png', display_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        probs=probs,
        image=img_base64
    )

if __name__ == '__main__':
    app.run(debug=True)
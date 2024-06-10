from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# URL del stream
video_stream_url = 'http://192.168.137.46:81/stream'

# Cargar el modelo de clasificación
model = load_model('modelo.h5')

classes = {
    0: 'Amblypygi',
    1: 'Araneae',
    2: 'Pseudoscorpiones',
    3: 'Scorpion',
    4: 'Solifugae',
}

def generate_stream():
    cap = cv2.VideoCapture(video_stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Realizar predicción en el frame
        prediction = predict(frame)
        
        # Mostrar el resultado en el frame
        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def predict(frame):
    # Procesar el frame para la predicción
    resized_frame = cv2.resize(frame, (224, 224))
    image = np.array(resized_frame).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Realizar la predicción
    predict = model.predict(image)
    class_predicted = np.argmax(predict)
    
    return classes[class_predicted]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


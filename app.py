from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.feature import hog
import joblib
import base64

app = Flask(__name__)

# Load the character recognition model
char_model = load_model('HCR_English.h5')

# Load the digit recognition model and preprocessor
digit_model = tensorflow.keras.models.load_model('mnist_mlp_model.h5')

char_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n',
    50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x',
    60: 'y', 61: 'z'
}

digit_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}


# Load the traffic sign prediction model
traffic_sign_model = load_model('model.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Define the class names based on class numbers
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles',
        'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Uturn',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons'
    ]
    return class_names[classNo]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    return getClassName(classIndex[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_char', methods=['POST'])
def predict_char():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Character Recognition
        char_prediction, char_uploaded_image = recognize_characters(im)

        return render_template(
            'char_prediction_result.html',
            char_prediction=char_prediction,
            char_uploaded_image=char_uploaded_image,
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Digit Recognition
        digit_predictions, uploaded_digits = recognize_digits(im)
        digit_count = len(digit_predictions)

        return render_template(
            'digit_prediction_result.html',
            digit_prediction=digit_predictions,
            uploaded_digits=uploaded_digits,
            digit_count=digit_count,
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_group', methods=['POST'])
def predict_group():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Character Recognition for a group of characters
        group_predictions, group_uploaded_images = recognize_group_characters(im)

        return render_template(
            'group_prediction_result.html',
            group_predictions=group_predictions,
            group_uploaded_images=group_uploaded_images,
            group_count=len(group_predictions),
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_traffic_sign', methods=['POST'])
def predict_traffic_sign():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Traffic Sign Prediction
        traffic_sign_prediction, traffic_sign_uploaded_image = recognize_traffic_sign(im)

        return render_template(
            'traffic_sign_prediction_result.html',
            traffic_sign_prediction=traffic_sign_prediction,
            traffic_sign_uploaded_image=traffic_sign_uploaded_image,
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

def recognize_characters(im):
    # Convert the image to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Resize the image
    im_gray = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Threshold the image
    _, im_thresh = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Ensure the image has a single channel
    if len(im_thresh.shape) > 2:
        im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)

    # Use the character model for prediction
    char_prediction = predict_character(im_thresh)

    # Convert the character image to base64 for display in HTML
    char_uploaded_image = base64.b64encode(cv2.imencode('.png', im_thresh)[1].tobytes()).decode()

    return char_prediction, char_uploaded_image

def predict_character(im):
    char_probabilities = char_model.predict(np.expand_dims(im, axis=0))
    char_label = np.argmax(char_probabilities, axis=1)[0]
    char_prediction = char_dict.get(char_label, 'Unknown Character')
    return char_prediction

def recognize_digits(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    digit_predictions = []
    uploaded_digits = []

    for rect in rects:
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = roi.astype('float32') / 255.0  # Normalize the image
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
        roi = np.expand_dims(roi, axis=0)   # Add batch dimension

        nbr = digit_model.predict(roi)
        digit_predictions.append(digit_dict.get(int(np.argmax(nbr)), 'Unknown Digit'))

        # Convert the image to base64 for HTML display
        roi_base64 = base64.b64encode(cv2.imencode('.png', roi[0, :, :, 0] * 255)[1].tobytes()).decode()
        uploaded_digits.append(roi_base64)

    return digit_predictions, uploaded_digits

def recognize_group_characters(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im_thresh = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(im_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    group_predictions = []
    group_uploaded_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_region = im_thresh[y:y + h, x:x + w]
        char_region = cv2.resize(char_region, (28, 28), interpolation=cv2.INTER_AREA)
        char_prediction = predict_character(char_region)
        char_uploaded_image = base64.b64encode(cv2.imencode('.png', char_region)[1].tobytes()).decode()
        group_predictions.append(char_prediction)
        group_uploaded_images.append(char_uploaded_image)

    return group_predictions, group_uploaded_images

def recognize_traffic_sign(im):
    img = cv2.resize(im, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = traffic_sign_model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    traffic_sign_prediction = getClassName(classIndex[0])

    traffic_sign_uploaded_image = base64.b64encode(cv2.imencode('.png', im)[1].tobytes()).decode()

    return traffic_sign_prediction, traffic_sign_uploaded_image

if __name__ == '__main__':
    app.run(debug=True)

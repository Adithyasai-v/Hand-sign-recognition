import cv2
import numpy as np
from keras.models import load_model


model = load_model("D:/archive/inceptionv3/sign.h5")


# Mapping the classes with gestures
gestures = {
    0: '1', 1: '3', 2: '4', 3: '5', 4: '7', 5: '8', 6: '9', 7: 'A', 8: 'B', 9: 'Baby', 10: 'Brother', 11: 'C', 12: 'D',
    13: "Don't like", 14: 'E', 15: 'F', 16: 'Friend', 17: 'G', 18: 'H', 19: 'Help', 20: 'House', 21: 'I', 22: 'J',
    23: 'K', 24: 'L', 25: 'Like', 26: 'Love', 27: 'M', 28: 'Make', 29: 'More', 30: 'N', 31: 'Name', 32: 'No',
    33: 'O_OR_0', 34: 'P', 35: 'Pay', 36: 'Play', 37: 'Q', 38: 'R', 39: 'S', 40: 'Stop', 41: 'T', 42: 'U', 43: 'V_OR_2',
    44: 'W_OR_6', 45: 'With', 46: 'X', 47: 'Y', 48: 'Yes', 49: 'Z', 50: 'nothing'
}

def predict(gesture):  # Method for predicting the gesture
    img = cv2.resize(gesture, (200, 200))
    img = img.reshape(-1, 200, 200, 3)
    img = img / 255.0
    prd = model.predict(img)
    index = prd.argmax()  # Selecting Best Estimate
    return gestures[index]


capture = cv2.VideoCapture(0)
rval, frame = capture.read()
pred_text = ''
count_frames = 0
flag = False

while True:

    if frame is not None:

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (400, 400))
        cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 2)  # Defining ROI
        crop_img = frame[100:300, 100:300]
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        if flag == True:
            wait = 0
            pred_text = predict(crop_img)
            count_frames = 0
            cv2.putText(blackboard, pred_text, (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
            wait += 1
        result = np.hstack((frame, blackboard))  # Concatening both frames
        cv2.imshow("Frame", result)
    rval, frame = capture.read()
    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('c'):  # Press C/c for enabling translation mode
        flag = True

    if keypress & 0xFF == ord('q'):  # Press q to exit
        break

capture.release()
cv2.destroyAllWindows()
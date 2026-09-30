import time
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from warning import give_voice_warning
from notification import notification_send

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Load models
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector_trained_model.h5")

# Start video
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Variables
last_prediction_time = 0
prediction_interval = 1.0
mask_start_time = None
warning_stage = 0
notification_sent = False

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    current_time = time.time()

    if (current_time - last_prediction_time) > prediction_interval:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        last_prediction_time = current_time

        mask_detected = False

        for pred in preds:
            (mask, withoutMask) = pred
            if mask > withoutMask:
                mask_detected = True
                break

        if not mask_detected:
            # Reset everything if no mask is detected
            mask_start_time = None
            warning_stage = 0
            notification_sent = False
            print("[INFO] No mask detected. System reset.")
        else:
            if mask_start_time is None:
                mask_start_time = current_time
                warning_stage = 0
                notification_sent = False
                print("[INFO] Mask detected. Timer started.")
            else:
                elapsed = current_time - mask_start_time

                if warning_stage == 0 and elapsed > 5:
                    give_voice_warning(1)
                    print("[WARNING] First warning given.")
                    warning_stage = 1

                elif warning_stage == 1 and elapsed > 20:
                    give_voice_warning(2)
                    print("[WARNING] Second warning given.")
                    warning_stage = 2

                elif warning_stage == 2 and elapsed > 35 and not notification_sent:
                    notification_send()
                    print("[ALERT] Notification sent.")
                    notification_sent = True
                    warning_stage = 3

    # Display result
    if 'locs' in locals() and 'preds' in locals():
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 0, 255) if mask > withoutMask else (0, 255, 0)  # Red for mask, Green for no mask

            label_text = f"{label}: {max(mask, withoutMask)*100:.2f}%"

            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)

    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

import cv2
import numpy as np
from PIL import Image
import os
import time
import codecs


def new_user(message):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    name = message

    with codecs.open('faces/profiles.txt', 'r+', 'utf-8') as f:
        profiles = f.read()
        if len(profiles) == 0:
            face_id = 1
        else:
            face_id = int([str(s) for s in profiles.split()][-2]) + 1
        f.write(str(face_id) + ' ' + name + '\n')

    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            cv2.imwrite("faces/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)
            print(count)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()


def train():
    path = 'faces/dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml");

    def get_images(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        id_array = []
        for imagePath in image_paths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                id_array.append(id)
        return face_samples, id_array

    print("\n Идет обучение...")
    faces, id_array = get_images(path)
    recognizer.train(faces, np.array(id_array))

    recognizer.write('faces/trainer/trainer.yml')

    print("\n {} пользователей обучено.".format(len(np.unique(id_array))))


def test():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('faces/trainer/trainer.yml')
    cascadePath = "cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    names = ['None']

    with codecs.open('faces/profiles.txt', 'r', 'utf-8') as f:
        for line in f:
            new_line = line.split()
            names.append(new_line[1])

    cam = cv2.VideoCapture(0)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    end_time = time.time() + 15
    while time.time() < end_time:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if (confidence < 100):
                id = names[id]
                if (100 - confidence) >= 70:  # Процент достоверности
                    cam.release()
                    cv2.destroyAllWindows()
                    return True
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

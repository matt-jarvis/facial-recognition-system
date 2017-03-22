'''
Created on Sep 11, 2015

@author: mattjarvis
'''
# Standard imports
import os
import csv
import pickle
from collections import Counter

# Third-party imports
import cv2
import numpy as np

# Local imports
import app
import image
import camera


# Face detection constants
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 8
MIN_SIZE = (250, 250)
CASCADE = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


def detect_face(np_img):
    TAG = 'FaceDetector'
    # Assume there is no face in the image to begin with
    has_face = False
    face_area = None

    # Detect faces in the image
    faces = CASCADE.detectMultiScale(np_img,
                                     scaleFactor=SCALE_FACTOR,
                                     minNeighbors=MIN_NEIGHBORS,
                                     minSize=MIN_SIZE)

    # Exit, if more than one face is detected
    if len(faces) == 0:
        app.log(TAG, 'No face detected')
    elif len(faces) > 1:
        app.log(TAG, 'Contains more than one face')
    elif len(faces) == 1:
        has_face = True
        face_area = faces[0]
        app.log(TAG, 'Face detected')

    return has_face, face_area


def scan_webcam_for_face():
    TAG = 'FaceScanner'

    face_img_list = []
    while True:  # Inner loop
        success, img = camera.get_raw_image()
        if success:  # Try to detect a face
            gray_img = image.to_grayscale(img)
            has_face, face_area = detect_face(gray_img)
            if has_face:  # Crop the image around face area
                face_img = image.crop_image(gray_img, face_area)
                face_img_list.append(face_img)
                app.log(TAG, 'Face added', len(face_img_list))
                if len(face_img_list) == 10:  # Scan complete
                    break  # Exit the outer loop and return

    return face_img_list


class Recognizer():

    TAG = 'Recognizer'
    THRESHOLD = 50
    REC_FILE = 'recognizer.yml'
    IMG_FILE = 'images.p'
    LBL_FILE = 'labels.p'

    def __init__(self):
        self.recognizer = self._new_recognizer()
        self.images = []
        self.labels = []

    def _new_recognizer(self):
        return cv2.createLBPHFaceRecognizer(threshold=self.THRESHOLD)

    def save(self, path):
        rec_file = os.path.join(path, self.REC_FILE)
        img_file = os.path.join(path, self.IMG_FILE)
        lbl_file = os.path.join(path, self.LBL_FILE)
        self.recognizer.save(rec_file)
        pickle.dump(self.images, open(img_file, 'wb'))
        pickle.dump(self.labels, open(lbl_file, 'wb'))

    def load(self, path):
        rec_file = os.path.join(path, self.REC_FILE)
        img_file = os.path.join(path, self.IMG_FILE)
        lbl_file = os.path.join(path, self.LBL_FILE)
        self.recognizer.load(rec_file)
        self.images = pickle.load(open(img_file, 'rb'))
        self.labels = pickle.load(open(lbl_file, 'rb'))

    def train(self, imgs, labels):
        np_images = image.image_list_to_numpy_list(imgs)
        np_labels = np.array(labels)
        try:  # Try to update the recognizer first
            self.recognizer.update(np_images, np_labels)
            app.log(self.TAG, 'Update completed')
        except cv2.error as e:
            app.log(self.TAG, 'Error updating -', e)
            app.log(self.TAG, 'Attempting to train instead')
            self.recognizer.train(np_images, np_labels)
            app.log(self.TAG, 'Training completed')
        self.images.extend(imgs)
        self.labels.extend(labels)

    def remove_label(self, label):
        new_imgs = []
        new_labels = []
        for i, l in zip(self.images, self.labels):
            if l != label:
                new_imgs.append(i)
                new_labels.append(l)
        self.images = new_imgs
        self.labels = new_labels
        if not self.images and not self.labels:  # Reset the recogniser
            self.recognizer = self._new_recognizer()
        else:  # Train the recogniser again
            np_images = image.image_list_to_numpy_list(self.images)
            np_labels = np.array(self.labels)
            self.recognizer.train(np_images, np_labels)
        app.log(self.TAG, 'Removed label -', label)

    def predict_face(self, img):
        np_img = image.image_to_numpy(img)
        return self.recognizer.predict(np_img)

    def predict_face_from_list(self, imgs):
        pred_labels = []
        for img in imgs:
            label, conf = self.predict_face(img)
            app.log(self.TAG, 'Label: {} Conf: {}'.format(label, conf))
            pred_labels.append(label)
        predicted_label = Counter(pred_labels).most_common(1)[0][0]
        if predicted_label == -1:  # Face wasn't recognised
            return None
        return predicted_label


class FaceDatabase():

    TAG = 'FaceDatabase'
    ROOT = 'data'
    CSV_FILE = 'labels&names.csv'
    FIELD = ['label', 'name']

    def __init__(self, path):
        self.recognizer = Recognizer()
        self.person_list = []
        self.label_counter = 0
        self.subdir = os.path.join(self.ROOT, path)
        self.csv_file = os.path.join(self.subdir, self.CSV_FILE)

        if os.path.isdir(self.subdir):  # If exists and is directory
            self.load()
        else:  # Create the database files
            os.makedirs(self.subdir)
            self.recognizer.save(self.subdir)
            self.create_csv()
            app.log(self.TAG, 'New database created')

    def load(self):
        try:
            self.recognizer.load(self.subdir)
            with open(self.csv_file) as c:
                data = csv.DictReader(c)
                for row in data:
                    label = int(row[self.FIELD[0]])
                    name = row[self.FIELD[1]]
                    p = Person(label, name)
                    self.person_list.append(p)
        except IOError as e:
            app.log(self.TAG, 'Error loading data -', e)
        if self.person_list:  # Set label counter
            self.label_counter = self.person_list[-1].get_label()
        app.log(self.TAG, 'Existing data loaded')

    def save(self):
        self.recognizer.save(self.subdir)

    def get_person_list(self):
        return self.person_list

    def get_recognizer(self):
        return self.recognizer

    def add_person(self, name, imgs):
        self.label_counter += 1
        label = self.label_counter
        p = Person(label, name)  # Create the person

        # Add the person to the database
        self.person_list.append(p)
        with open(self.csv_file, 'a') as c:
            w = csv.DictWriter(c, fieldnames=self.FIELD)
            w.writerow({'label': p.get_label(), 'name': p.get_name()})
        app.log(self.TAG, 'Added -', p.get_name())

        # Lastly, train the recognizer with the new person
        labels = [p.get_label()] * len(imgs)
        self.recognizer.train(imgs, labels)

        return p

    def remove_person(self, person):
        self.recognizer.remove_label(person.get_label())
        self.person_list.remove(person)
        self.create_csv()

        with open(self.csv_file, 'a') as c:
            for p in self.person_list:
                w = csv.DictWriter(c, fieldnames=self.FIELD)
                w.writerow({'label': p.get_label(), 'name': p.get_name()})
        app.log(self.TAG, 'Removed -', person.get_name())

    def has_person(self, person):
        if person in self.person_list:
            return True
        return False

    def get_person_from_label(self, label):
        for p in self.person_list:
            if p.has_label(label):
                return p
        return None

    def create_csv(self):
        with open(self.csv_file, 'w') as c:  # Persistant storage
            w = csv.DictWriter(c, fieldnames=self.FIELD)
            w.writeheader()


class Person():

    def __init__(self, label, name):
        self.label = label
        self.name = name

    def get_label(self):
        return self.label

    def get_name(self):
        return self.name

    def has_label(self, label):
        if self.label == label:
            return True
        return False

    def tostring(self):
        return str(self.label) + ' ' + self.name

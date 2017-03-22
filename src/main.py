'''
Created on Aug 21, 2015
@author: mattjarvis
'''
import app
import face
import sys


TAG = 'FaceRecDemo'
app.log(TAG, 'Loading data...')
data = face.FaceDatabase('database1')

welcome = "\n************************************************************\n"
welcome += "*************  Face Recognition Demonstration  *************\n"
welcome += "************************************************************"

instructions = '\nUsage: a  -  add new person to the dataset\n'
instructions += '       l  -  list all existing people in the dataset\n'
instructions += '       r  -  remove an existing person from the dataset\n'
instructions += '       p  -  predict a face by scanning the webcam\n'
instructions += '       e  -  exit.\n'


def run_program():
    print welcome
    print instructions
    while True:
        i = raw_input("-> ")
        if i == 'a':
            add_person()
        elif i == 'l':
            list_all_persons()
        elif i == 'r':
            remove_person()
        elif i == 'p':
            predict_face_from_webcam()
        elif i == 'h':
            print instructions
        elif i == 'e':
            exit_program()
        else:
            app.log(TAG, 'Invalid option, try again or type h for help.')


def exit_program():
    try:
        data.save()
        app.log(TAG, 'Data saved')
    except Exception as e:
        app.log(TAG, 'Error saving data - ', e)
    app.log(TAG, 'Goodbye')
    sys.exit(0)


def add_person():
    imgs = face.scan_webcam_for_face()
    name = raw_input('\nPlease enter your name: ')
    data.add_person(name, imgs)


def list_all_persons():
    app.log(TAG, 'Existing people:', len(data.get_person_list()))
    for p in data.get_person_list():
        app.log(TAG, '>', p.get_label(), '-', p.get_name())


def remove_person():
    while True:
        i = raw_input('Enter label to remove (c to cancel):')
        if i == 'c':
            return
        p = data.get_person_from_label(int(i))
        if not p:
            continue
        data.remove_person(p)
        app.log(TAG, 'Person removed -', p.get_name())
        break


def predict_face_from_webcam():
    if not data.get_person_list():
        app.log(TAG, 'Please add a person first')
        return
    imgs = face.scan_webcam_for_face()
    predicted_label = data.get_recognizer().predict_face_from_list(imgs)
    if predicted_label is None:
        app.log(TAG, 'Access denied')
    else:
        p = data.get_person_from_label(predicted_label)
        app.log(TAG, 'Access granted. Recognised as', p.get_name())


if __name__ == '__main__':
    run_program()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:47:31 2021

@author: hassen
"""
# from https://github.com/ageitgey/face_recognition
import face_recognition
import numpy as np
import cv2, queue, threading, time
import requests, os, re

from PIL import Image, ImageDraw
from IPython.display import display

# Chargement du vidéo
video_capture = cv2.VideoCapture('videos/vehicule3.avi')
video_capture.set(5, 2)



# tableau photos chaffauers autorisés
known_faces_filenames = []
#tableau noms chauffeurs autorisés
known_face_names = []
# tableau photos chaffauers autorisés encodés
known_face_encodings = []

# récupérer les photos des chauffeurs autorisés dans le dossier 'img/known'
# ces photos doivent contenir un visage 
for (dirpath, dirnames, filenames) in os.walk('img/known/'):
    known_faces_filenames.extend(filenames)
    break

# Learning from photos
for filename in known_faces_filenames:
    # si la photo ne contient pas un visage humain on va avoir une erreur
    face = face_recognition.load_image_file('img/known/' + filename)
    # on rempli le tableau des noms depuis les noms des photos sans l'extension
    known_face_names.append(re.sub("[0-9]",'', filename[:-4]))
    """
    L'encodage est simplement une représentation de faible dimension d'une face 
    qui peut être facilement comparée à d'autres faces que la bibliothèque 
    reconnaîtra à l'avenir.
    """
    known_face_encodings.append(face_recognition.face_encodings(face)[0])

print('Learned encoding for', len(known_face_encodings), 'images.')

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# fonction qui permet de dessiner un cadre autour du visage
def drawRectangleOnFace(face_locations, face_names, similarity_text, autorized=True ):
    if autorized:
        color = [0,255,0]
        name = face_names
    else:
        color = [0,0,255]
        name = "Unknown ("+similarity_text+")"
    for (top, right, bottom, left), name in zip(face_locations, name):
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            #text_width, text_height = draw.textsize(name)
            cv2.putText(frame, name+"("+similarity_text+")", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) 
            
    
while True:
    ret, frame = video_capture.read()
    
    # checher tout les visages dans 'frame'
    face_locations = face_recognition.face_locations(frame)
    # encoder 'frame'
    face_encodings = face_recognition.face_encodings(frame, face_locations)
   
    
    face_names = []
    # Boucle sur chaque visage trouvé dans l'image frame
    for face_encoding in face_encodings: 
        # Voir si le visage correspond au (x) visage (s) connu (s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # Ou à la place, utilisez les visages connu avec la plus petite distance par rapport au nouveau visage
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        similarity = (1 - face_distances[0])*100
        similarity_text = str(np.round(similarity))+"%"
        print("Similarity=>"+similarity_text)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_names.append(name)    
            autorized = True
        else:
            name = "Unknown"
            face_names.append(name)      
            autorized = False
            
        drawRectangleOnFace(face_locations, face_names, similarity_text, autorized)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break        
                    
video_capture.release()
cv2.destroyAllWindows()
import os,face_recognition,cv2,re
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import easygui
import numpy as np
from datetime import datetime
import shutil
from PIL import Image
import sys
import time

# Chargement du vidéo
video_capture = cv2.VideoCapture('videos/vehicule.avi')
video_capture.set(5, 2)

# tableau photos chaffauers autorisés
known_faces_filenames = []
#tableau noms chauffeurs autorisés
known_face_names = []
# tableau photos chaffauers autorisés encodés
known_face_encodings = []

is_processing_fail = True

# fonction encodage visages autorisés
def loadKnownDrivers():
    # récupérer les photos des chauffeurs autorisés dans le dossier 'img/known'
    # ces photos doivent contenir un visage 
    for (dirpath, dirnames, filenames) in os.walk('img/known/'):
        known_faces_filenames.extend(filenames)
        break
    
    # Learning from photos
    if len(known_faces_filenames) == 0:
        is_processing_fail = True
    else:
        is_processing_fail = False
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
            faces = face_recognition.face_encodings(face)
            if len(faces) < 1:
                is_processing_fail = True
            else:
                is_processing_fail = False
                known_face_encodings.append(faces[0])

    print('Learned encoding for', len(known_face_encodings), 'images.')
    return is_processing_fail

# fonction qui permet de dessiner un cadre autour du visage
def drawRectangleOnFace(face_locations, face_names, similarity_text, autorized=True):
    if autorized:
        color = [0,255,0]
        name = face_names
    else:
        color = [0,0,255]
        name = "Unknown ("+similarity_text+")"
    for (top, right, bottom, left), name in zip(face_locations, name):
        if not autorized:
           captureFace(frame, top, right, bottom, left)
        
        # draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name+"("+similarity_text+")", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) 

# fonction accept the new driver to drive the car and save image
def captureFace(frame, top, right, bottom, left):
    crop_img = frame[top: top+bottom+30, left: left+bottom] 
    date_now = datetime.today().strftime('%Y%m%d%H%M%S')
    img_path = 'img/unknown/unknown_'+date_now+'.jpg'
    if cv2.imwrite(img_path, crop_img):
        image = Image.open(img_path)
        image.show()
        autoriseUnknownDriver(img_path)
        
# fonction accept the new driver to drive the car and save image
def autoriseUnknownDriver(img_path):
    top = Tk()
    def quitTk(top):
        top.destroy()
        sys.exit()
    
    response = messagebox.askquestion("autorise unknown driver", "Do you want to authorize this person to drive your car ?", icon='warning')
    print(response)
    if response == 'yes':
        canvas1 = tk.Canvas(top, width = 400, height = 300,  relief = 'raised')
        canvas1.pack()
        
        label1 = tk.Label(top, text='Authorization of a new driver')
        label1.config(font=('helvetica', 14))
        canvas1.create_window(200, 25, window=label1)
        
        label2 = tk.Label(top, text='please enter the first name of the new driver:')
        label2.config(font=('helvetica', 10))
        canvas1.create_window(200, 100, window=label2)
        
        entry1 = tk.Entry(top) 
        canvas1.create_window(200, 140, window=entry1)
        
        def getName():
            name = entry1.get()
            shutil.move(img_path, 'img/known/'+name+'.jpg')
            loadKnownDrivers()
            top.destroy()
            
        button1 = tk.Button(text='Valider', command=getName, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
        canvas1.create_window(200, 180, window=button1)
        top.mainloop()
    else:
        print('not autorized')
        # create button to implement destroy()
        Button(top, text="Quit", command=quitTk(top)).pack()
       
        

face_locations = []
face_encodings = []
face_names = []
known_drivers_loaded = loadKnownDrivers()

while True:
    if known_drivers_loaded:
        break
        
    ret, frame = video_capture.read()
    
    # Detecter tout les visages dans un 'frame'
    start_time = time.time()
    face_locations = face_recognition.face_locations(frame,1,"hog")
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    #print(time_elapsed)
    
    # Extraction des caractéristiques ou encoder d'un 'frame'
    face_encodings = face_recognition.face_encodings(frame, face_locations)
       
    face_names = []
    
    # Boucle sur chaque visage extrait depuis le frame dans "face_encodings"
    for face_encoding in face_encodings: 
        # Voir si le visage correspond au (x) visage (s) connu (s)
        # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # print(matches)
        # Ou à la place, on utilise les visages connu avec la plus petite 
        # distance par rapport au nouveau visage
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        print('best_match_index=>'+str(best_match_index))
        
        print('face_distances=>'+str(face_distances))
        similarity = (1 - face_distances[best_match_index])*100
        similarity_text = str(np.round(similarity))+"%"

        if face_distances[best_match_index] <= 0.6:
            name = known_face_names[best_match_index]
            face_names.append(name)    
            autorized = True
        else:
            name = "Unknown"
            face_names.append(name)
            autorized = False   
            cv2.waitKey(-1) #wait until any key is pressed
                   
        drawRectangleOnFace(face_locations, face_names, similarity_text, autorized)
        cv2.imshow('Video', frame)
        
          
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break        
                    
video_capture.release()
cv2.destroyAllWindows()
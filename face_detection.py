#importing libraries
import cv2
import numpy as np
import face_recognition as face_rec

#image detection
pari=face_rec.load_image_file('SmartAttendence\pari.jpg')
pari=cv2.cvtColor(pari,cv2.COLOR_BGR2RGB)
pari=cv2.resize(pari,(300,300))

pari_tilted=face_rec.load_image_file('SmartAttendence\pari_tilted.jpg')
pari_tilted=cv2.cvtColor(pari_tilted,cv2.COLOR_BGR2RGB)
pari_tilted=cv2.resize(pari_tilted,(300,300))

#finding face location
faceLocation_pari= face_rec.face_locations(pari)[0]
encode_pari=face_rec.face_encodings(pari)[0]
cv2.rectangle(pari,(faceLocation_pari[3],faceLocation_pari[0]),(faceLocation_pari[1],faceLocation_pari[2]),(255,0,255),3)



faceLocation_pari_tilted= face_rec.face_locations(pari_tilted)[0]
encode_pari_tilted=face_rec.face_encodings(pari_tilted)[0]
cv2.rectangle(pari_tilted,(faceLocation_pari[3],faceLocation_pari[0]),(faceLocation_pari[1],faceLocation_pari[2]),(255,0,255),3)

results=face_rec.compare_faces([encode_pari],encode_pari_tilted)
print (results)
cv2.putText(pari_tilted,f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)
cv2.imshow('main_image',pari)
cv2.imshow('tilted_image',pari_tilted)
cv2.waitKey(0)
cv2.destroyAllWindows()
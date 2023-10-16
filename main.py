import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os
now = datetime.now()

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('trim.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

count=0
tracker=Tracker()   

area1=[(772,305),(752,310),(780,422),(802,421)]
area2=[(50,399),(27,405),(306,482),(310,460)]

area1_c=set()
area2_c=set()

def imgwrite(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.png' % current_time
    cv2.imwrite(os.path.join(r"D:\yolov8peoplecount-main1\yolov8peoplecount-main\img",filename), img)
    

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))
        
    
    results=model.predict(frame)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")

    list=[]
    for index,row in px.iterrows():

 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])
            
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
         
                
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if results>=0:
           crop=frame[y3:y4,x3:x4]
           imgwrite(crop)
           cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           cv2.circle(frame,(cx,cy),5,(255,0,255),-1)
           cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
           area1_c.add(id)
           

        results1=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if results1>=0:
           crop=frame[y3:y4,x3:x4]
           imgwrite(crop)
           cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           cv2.circle(frame,(cx,cy),5,(255,0,255),-1)
           cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)   
           area2_c.add(id)
    first=(len(area1_c))
    sec=(len(area2_c))
    
    cv2.putText(frame,f'People Entry:{first}',(10,80),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2) 
    cv2.putText(frame,f'People Exit:{sec}',(10,100),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2) 
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


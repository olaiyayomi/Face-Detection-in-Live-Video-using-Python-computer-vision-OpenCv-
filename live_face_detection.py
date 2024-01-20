import cv2 as cv
import numpy as np
import sys

video = cv.VideoCapture(0)

if not video.isOpened():
    sys.exit("unable to detect video frames")
    
while True:
    __, frame = video.read()
    
    path = "D:/YOMTECH PROJECTS/my python/OpenCV/cature.png"
    
    template = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    cvt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    match = cv.matchTemplate(cvt, template, cv.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    
    top_left = max_loc
    
    w, h = template.shape[::-1]
    
    bottom_right = (top_left[0]+w, top_left[1]+h)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    degree = max_val*100
    cv.putText(frame, "scanning for a face.............", (100,100), font, 1, 255, 2)
    
    text = "Rate: %s"%int(degree)+"%"
    if degree >= 50:
        cv.rectangle(frame, (top_left),(bottom_right), 255, 3)
        cv.putText(frame, text, (bottom_right), font, 1, 255, 2)
        
    
    cv.imshow("my live video", frame)
    
    
    key = cv.waitKey(1)
    
    if key == ord("c"):
        
        cv.imwrite("cature.png",frame)
        
    
    if key == ord("q"):
        break
cv.destroyAllWindows()

video.release()

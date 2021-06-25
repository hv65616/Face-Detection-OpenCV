# Starting of file will be importing of cv2 module which allows to do certain functions with our image
import cv2

#Then importing of OS module
import os

#Here,face_classifier stores the data set of frontal face
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Here,eye_classifier stores the data set of detecting of eyes
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

#Here,smile_classifier stores the data of detecting of smile on face
smile_classifier = cv2.CascadeClassifier("haarcascade_smile.xml")

#Activating the camera and storing the video capture in cap
cap = cv2.VideoCapture(0)

#Initializing of a variable with initial value of 0
count = 0

while True:

    #At this line we are doing 2 things
    #In ret we will be storing either true or false of image reading
    #In frame we will be storing the frames that our camera produces
    ret, frame = cap.read()

    #putText() used to put the text with certain compulsory parameters
    cv2.putText(frame,"Face Detection",(200,450),cv2.FONT_HERSHEY_PLAIN,2.3,(255,255,0),2)

    #Here, we will convert the imgae into the gray basically the pixls become either 0 or 225
    convert_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #After converting into gray we will pass the converted image into detectMultiScale so that we can find face more appropiately
    faces = face_classifier.detectMultiScale(convert_gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    #Making the suitable size frame of our face now we are assigning its height width value to variables so that we can make rectangle box
    for (x, y, w, h) in faces:

        #Here,we create rectangle with dimensions specified and color of box and thickness
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        #This will crop the gray_image according to dimensions passed
        roi_gray = convert_gray[y:y+h, x:x+w]

        #This will crop the original frame according to the dimensions passed
        roi_color = frame[y:y+h, x:x+w]

        #Providing the path value where we will store the captured frames
        path = r'C:\Users\hpw\OneDrive\Documents\ImageProcessing\Output'

        #Assigning the frames a name from which they will save
        frame_number = "Frame_{}.jpg".format(count)

        os.chdir(path)

        #This will create and save the images
        cv2.imwrite(frame_number, roi_color)

        #Updating the value of count so that name of frame get continuously updated
        count += 1

        #here we are taking out only the part of our smile
        smiles = smile_classifier.detectMultiScale(roi_gray)

        for (sx, sy, sw, sh) in smiles:

            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        
        #Here we are taking out part for our eye
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 225), 2)

    #This is responsible for showing our video on screen
    cv2.imshow("My video", frame)

    #Waitkey() allows us to set the time interval in between reading and showing
    key = cv2.waitKey(1)

    #For exiting the window we press q key
    if key == ord("q"):
        break

#At last release the cap
cap.release()

#And release all the windows
cap.deleteAllWindows()

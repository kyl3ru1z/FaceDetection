import cv2

# accessing the haarcascades xml files
# haarcascades are pre-trained classifiers that already knows what it is looking for, haarcascades will look at an image
# and try to find certain features in that image.
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
catCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

# second property fixes a bug that shows as an error after running the program
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Frame Width
cap.set(3, 640)
# Frame Height
cap.set(4, 480)
# adjusting brightness
cap.set(10, 140)

while True:
    # cap.read return two values: a boolean that sees if the image reading was successful and the image itself.
    sucess, video = cap.read()

    # mirroring the video from the webcam and converting it into gray scale (gray-scale is needed for the face detection algorithm)
    videoFlip = cv2.flip(video, 1)
    # converting to gray scale because it is what is needed for the algorithm
    videoGray = cv2.cvtColor(videoFlip, cv2.COLOR_RGB2GRAY)

    # 1. second property is Scale Factor - your model has a fixed size defined during training, which is visible in the xml.
    # This means that this size of face is detected in the image if present. However, by rescaling the input image, you can
    # resize a larger face to a smaller one, making it detectable by the algorithm.
    # 2. third property is Minimum Neighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.

    faces = faceCascade.detectMultiScale(videoGray, 1.3, 5)
    eyes = eyeCascade.detectMultiScale(videoGray, 1.3, 5)
    cats = catCascade.detectMultiScale(videoGray, 1.3, 5)

    # creates a rectangle around any face, eye, or cat.
    # everytime a haarcascade has detected something it gives it values for a rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(videoFlip, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(videoFlip, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(videoFlip, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    for (x, y, w, h) in cats:
        cv2.rectangle(videoFlip, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(videoFlip, 'Cat', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

    # creates a window
    cv2.imshow("Face Detection Demo", videoFlip)

    # press q to break out of the loop and break out of the program
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

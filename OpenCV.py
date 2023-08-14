import cv2


def photoFaceDetect(img):
    import cv2
    cascpath = r"haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascpath)
    faces = faceCascade.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        # 画出人脸框，蓝色，画笔宽度微
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)


def videoFaceDetect():
    import cv2
    cascpath = r"haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascpath)
    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()

        faces = faceCascade.detectMultiScale(
            img, 1.1, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow("DetectFaces", img)
        if not ret:
            print("Erro:Video capture failed.")
            break
        else:
            cv2.imshow("DetectFaces", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


# img = cv2.imread("Ruiimg.jpg")

# photoFaceDetect(img)

videoFaceDetect()

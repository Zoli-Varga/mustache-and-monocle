import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('Resources/cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('Resources/cascades/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Resources/cascades/third-party/Nose18x15.xml')

mustache = cv2.imread('Resources/mustache.png', -1)
monocle = cv2.imread("Resources/monocle.png", -1)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.25, minNeighbors=6)
        for (ex, ey, ew, eh) in eyes:
            roi_eyes = roi_gray[ey:ey + eh, ex:ex + ew]
            monocle2 = image_resize(monocle.copy(), width=int(ew * 0.5))

            gw, gh, gc = monocle2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if monocle2[i, j][3] != 0:
                        roi_color[ey + int(eh / 3.5) + i, ex + j] = monocle2[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.25, minNeighbors=6)
        for (nx, ny, nw, nh) in nose:
            roi_nose = roi_gray[ny:ny + nh, nx:nx + nw]
            mustache2 = image_resize(mustache.copy(), width=int(nw * 1.2))

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    if mustache2[i, j][3] != 0:
                        roi_color[ny + int(nh / 2.0) + i, nx + j] = mustache2[i, j]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('Mustache & Monocle', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

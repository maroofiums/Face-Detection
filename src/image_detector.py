import cv2


def main():
    image = cv2.imread("data/images/person.jpg")

    if image is None:
        raise FileNotFoundError("Image not found.")

    face_detector = cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml"
    )

    if face_detector.empty():
        raise FileNotFoundError(
            "Could not load Haar Cascade XML."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
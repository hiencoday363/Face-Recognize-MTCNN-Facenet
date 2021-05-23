import sys
import time
import cv2
from facenet.face_contrib import *


def add_overlays(frame, faces, colors, confidence=0.5):
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)

            # face_bb : là 2 array [x1,y1, x2,y2]

            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            if face.name and face.prob:
                if face.prob > confidence:
                    class_name = face.name
                else:
                    class_name = 'Unknow'

                cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[idx], thickness=2, lineType=2)
                cv2.putText(frame, f'{face.prob * 100:.02f}', (face_bb[0], face_bb[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=1, lineType=2)


def run(model_checkpoint, classifier, img_path=None, output_file=None):
    if img_path is None or output_file is None:
        return

    face_recognition = Recognition(model_checkpoint, classifier)
    colors = np.random.uniform(0, 255, size=(1, 3))

    img = cv2.imread(img_path)

    # resize image
    width, height = 800, 530
    dim = (width, height)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Image Original', image)

    faces = face_recognition.identify(image)
    for i in range(len(colors), len(faces)):
        colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)

    add_overlays(image, faces, colors)

    cv2.imshow('Image Predict', image)

    # write image
    cv2.imwrite(output_file, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    run('models', 'models/your_model.pkl', img_path='input_image/madona-check.jpg', output_file='output_image/result2.jpg')

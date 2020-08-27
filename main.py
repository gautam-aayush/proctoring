import cv2
import time

from face_detector import FaceDetector

PROTOTXT_PATH = 'models/face_detector/deploy.prototxt'
MODEL_PATH = 'models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'


def run_face_detection_on_webcam(src=0):
    video_stream = cv2.VideoCapture(src)
    print("Starting video stream...")
    time.sleep(2.0)
    detector = FaceDetector(PROTOTXT_PATH, MODEL_PATH)
    detector.detect_faces_from_video(video_stream)


def crop_faces(img_path):
    img = cv2.imread(img_path)
    face_detector = FaceDetector(PROTOTXT_PATH, MODEL_PATH)
    cropped_faces = face_detector.get_cropped_faces(img)
    import os
    output_path = 'output/' + img_path.split('/')[1] + '/' +img_path.split('/')[2][:-4]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, face in enumerate(cropped_faces):
        cv2.imwrite(output_path+f'/face_{i}.jpg', face)


def run_face_detection_image(img_path):
    img = cv2.imread(img_path)
    face_detector = FaceDetector(PROTOTXT_PATH, MODEL_PATH)
    face_detector.detect_faces_from_image(img)


if __name__ == '__main__':
    # run_face_detection_on_webcam(3)
    # run_face_detection_on_webcam('data/video/video_1.mp4')
    run_face_detection_image('data/multiple_faces/img_3.webp')
    # crop_faces('data/multiple_faces/img_3.webp')
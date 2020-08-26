import cv2
import time

from FaceDetector import FaceDetector

PROTOTXT_PATH = 'models/face_detector/deploy.prototxt'
MODEL_PATH = 'models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'


def run_face_detection():
    video_stream = cv2.VideoCapture(0)
    print("Starting video stream...")
    time.sleep(2.0)
    detector = FaceDetector(PROTOTXT_PATH, MODEL_PATH)
    detector.detect_faces_from_video(video_stream)


if __name__ == '__main__':
    run_face_detection()
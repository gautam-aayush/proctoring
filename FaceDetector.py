import cv2
import numpy as np


class BoundingBox(object):
    """
    Class for storing coordinates of bounding box and its confidence
    """
    def __init__(self, x_left_bottom, y_left_bottom, x_right_top, y_right_top, confidence):
        self.x_left_bottom = x_left_bottom
        self.y_left_bottom = y_left_bottom
        self.x_right_top = x_right_top
        self.y_right_top = y_right_top
        self.confidence = confidence

    def get_box(self):
        return self.x_left_bottom, self.y_left_bottom, self.x_right_top, self.y_right_top


class FaceDetector(object):
    """
    Wrapper for detecting a faces in an image or video stream using
    a pretrained SSD Caffe model available in:
    https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
    Documented in:
    https://github.com/opencv/opencv/tree/master/samples/dnn#face-detection
    """
    def __init__(self, prototext: str, model_path: str, threshold=0.5,
                 image_height=300, image_width=300, channel_means=(104.0, 177.0, 123.0)):
        """
        Initialize the face detector model
        :param prototext: path to the prototext file available in the above link
        :param model_path: path to saved caffe model
        :param threshold: threshold probability below which detected faces are filtered
        :param image_height: model input image height
        :param image_width: model input image width
        :param channel_means: mean values of (B, G, R) channels of data on which the model was trained
        """
        self.prototext = prototext
        self.model_path = model_path
        self.threshold = threshold
        self.image_height = image_height
        self.image_width = image_width
        self.channel_means = channel_means
        self.model = cv2.dnn.readNetFromCaffe(self.prototext, self.model_path)

    def get_bounding_boxes(self, image: np.ndarray):
        """
        Generate bounding boxes for detected faces in the given image
        :param image: (np.ndarray) having shape (height, width, 3) loaded in (B, G, R) format
        :return:
        """
        blob = self.get_processed_blob(image)
        self.model.setInput(blob)
        detections = self.model.forward()
        img_height, img_width = image.shape[:2]
        # detections is a numpy array having shape (1, 1, num_detections, 7)
        # detections[0,0,i,2] = confidence
        # detections[0,0,i,3:7] = bounding box having values in range(0,1)
        bounding_boxes = []
        for i in range(detections.shape[2]): # detected faces
            confidence = detections[0,0,i,2]
            if confidence > self.threshold:
                box = (detections[0,0,i,3:7] * np.array([img_width, img_height, img_width, img_height])).astype(int)
                bounding_boxes.append(BoundingBox(*box, confidence))
        return bounding_boxes

    def detect_faces_from_image(self, image, wait=True):
        """
        Detect faces from an image
        :param image: (np.ndarray) having shape (height, width, 3) loaded in (B, G, R) format
        :return:
        """
        bounding_boxes = self.get_bounding_boxes(image)
        for box in bounding_boxes:
            # green face rectangle
            cv2.rectangle(image, (box.x_left_bottom, box.y_left_bottom),
                          (box.x_right_top, box.y_right_top), (0, 255, 0))
            label = 'face: %.2f' % box.confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # while rectangle for printing label
            cv2.rectangle(image, (box.x_left_bottom, box.y_left_bottom - label_size[1]),
                          (box.x_left_bottom + label_size[0], box.y_left_bottom + base_line),
                          (255,255,255), cv2.FILLED)
            # write text in black
            cv2.putText(image, label, (box.x_left_bottom, box.y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.imshow("face_detection", image)
        if wait:
            cv2.waitKey(0)

    def detect_faces_from_video(self, video_stream: cv2.VideoCapture):
        """
        Detect faces in a video stream
        :param video_stream: video stream and instance of cv2.VideoCapture
        :return:
        """
        while True:
            ret, frame = video_stream.read()
            self.detect_faces_from_image(frame, wait=False)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    def get_processed_blob(self, image: np.ndarray):
        """
        Process image to make it model ready
        resize the image, subtract mean and reshape
        :param image:
        :return: (np.ndarray) having shape (1, 3, self.image_width, self.image_height)
        """
        resized_image = cv2.resize(image, (self.image_height, self.image_width))
        blob = cv2.dnn.blobFromImage(resized_image, scalefactor=1, size=(self.image_height, self.image_width),
                                     mean=self.channel_means)
        return blob


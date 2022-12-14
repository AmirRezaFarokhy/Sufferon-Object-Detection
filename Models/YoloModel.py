import numpy as np
import cv2 
import time 
import torch

class RealTimeObjectDetection:

    def __init__(self, capture_index, model_path):
        self.capture_index = capture_index
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        if torch.cuda.is_available():
            self.device = 'cuda'  
        else:
            self.device = 'cpu'
        
        print(f"Using Device: {self.device}")

    
    def VideoCapture(self):
        return cv2.VideoCapture(self.capture_index)

    def LoadModel(self, model_path):
        if model_path:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        return model

    def ScoreFrames(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results..xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def ClassToLable(self, x):
        return self.classes[int(x)]

    def PlotBoxes(self, results, frame):
        labels, cord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(len(labels)):
            row = cord[i]
            if row[4]>=0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)      
        
        return frame

    def __call__(self):
        cap = self.get_video_capture()
        done = False
        while not done:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (416, 416))
            start_time = time.time()
            results = self.ScoreFrames(frame)
            frame = self.PlotBoxes(results, frame)
            fps = 1 / np.round(endtime-start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('Flower Detection', frame)
            if cv2.waitKey(0) & 0xFF=='q':
                done = True
        
        cap.release()


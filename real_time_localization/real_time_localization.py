import cv2
import json
import numpy as np
import os
from datetime import datetime

class Capture:
    def __init__(self):
        try:
            with open(settings, 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from {settings}")
        
        self.cameras = [cv2.VideoCapture(1), cv2.VideoCapture(3), cv2.VideoCapture(2)]
        self.reference_shape = self.cameras[0].read()[1].shape[:2]
            
    def __str__(self):
        return json.dumps(self.settings, indent=4)
    
    def frame_concatent(self, frames, reference_shape):
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], (reference_shape[1], reference_shape[0]))
        return np.concatenate(frames, axis=1)
            
    def default_capture(self, frame):
        return frame
        
    def save_video(self, capture_function = self.default_capture):
        for i in range(len(self.cameras)):
            self.cameras[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cameras[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            if not self.cameras[i].isOpened():
                print(f"Error: Could not open camera {i}")
                return
            ret, frame = self.cameras[i].read()
            frames.append(frame)
        
        frame = self.frame_concatent(frames, self.reference_shape)
        self.height, self.width = frame.shape[:2]
        
        if not os.path.exists('output'):
            os.makedirs('output')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/{timestamp}.mp4'
        
        if len(self.settings['cameras']) == 1:
            filename = f'output/{self.settings['cameras'][0]}_{timestamp}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, 24, (self.width, self.height))


        while True:
        frames = []
        
        show_frames = []
        for i in range(len(self.cameras)):
            ret, frame = self.cameras[i].read()
            frames.append(frame)
            shown_frame = frames[i].copy()
            shown_frame = capture_function(shown_frame)
            show_frames.append(shown_frame)
            
        self.out.write(self.frame_concatent(frames, self.reference_shape))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        self.out.release()
        for i in range(len(self.cameras)):
            self.cameras[i].release()
        cv2.destroyAllWindows()

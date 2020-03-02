import threading
import time
import zmq
import cv2
import numpy as np
from PIL import Image, ImageTk
import cv2
from tkinter import *

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import *
from utils.utils import *
from utils.datasets import *

classes = None
class_colors = None
model = None
device = None
zmq_thread = None
main_wnd = None


class ZMQThread(threading.Thread):
    def __init__(self):
        super(ZMQThread, self).__init__()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR) # TODO: Use POLL instead of PAIR
        self._stop = False

    def terminate(self):
        print('Stopping ZMQ...')
        self._socket.close()
        self._context.destroy()       
        self._stop = True

    def run(self):        
        self._socket.bind('tcp://*:5555')

        frames_counter = 0
        start_time = time.time()    
        while not self._stop:
            msg = []
            try:
                msg = self._socket.recv()
            except Exception: # Less clean. should use POLL instead
                pass
            if len(msg) != 0:
                height = int.from_bytes(msg[0:2], byteorder='little')
                width = int.from_bytes(msg[2:4], byteorder='little')
                channel = int.from_bytes(msg[4:6], byteorder='little')
                img = np.frombuffer(msg[6:], dtype=np.uint8) # Make sure dtype is uint8 
                img = img.reshape((height, width, channel))            

                img_tensor = transforms.ToTensor()(Image.fromarray(img))
                img_tensor, _ = pad_to_square(img_tensor, 0) # Doesn't seem to impact results much

                img_tensor = resize(img_tensor, 416)
                img_tensor = torch.unsqueeze(img_tensor, 0) # Insert minibatch axis
                img_tensor = img_tensor.to(device)

                detections = model(img_tensor)
                detections = non_max_suppression(detections, 0.8, 0.4) # Merge bounding boxes using NMS

                if len(detections) > 0 and detections[0] is not None:                
                    detections = rescale_boxes(detections[0], 416, np.asarray(img).shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:                    
                        cv2.rectangle(img, (x1, y1), (x2, y2), class_colors[int(cls_pred)], 2)
                        cv2.putText(img, str(classes[int(cls_pred)]), (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[int(cls_pred)], 2, 2)


                frames_counter += 1.0
                end_time = time.time()
                fps = 0.0
                if end_time - start_time != 0:
                    fps = frames_counter / (end_time - start_time)
                if frames_counter == 10000:
                    start_time = time.time()
                    frames_counter = 0

                cv2.putText(img, f'Avg FPS: {int(fps)} sec: {end_time-start_time}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                img = Image.fromarray(img, mode='RGB')
                img = ImageTk.PhotoImage(img)
                img_box.configure(image=img)
                img_box.image = img

def close_window():
    zmq_thread.terminate()
    main_wnd.quit()

if __name__ == '__main__':
    
    classes = load_classes('data/coco.names')  # Extracts class labels from file
    class_colors = np.random.uniform(0, 255, size=(len(classes), 3))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # model = Darknet('config/yolov3-tiny.cfg', img_size=416).to(device)
    # model.load_darknet_weights('weights/yolov3-tiny.weights')
    model = Darknet('config/yolov3.cfg', img_size=416).to(device)
    model.load_darknet_weights('weights/yolov3.weights')

    model.eval()

    main_wnd = Tk()    
    # Cannot use Label(main_wnd, text='', image=ImageTk.PhotoImage(img)) directly
    # due to http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
    img_box = Label(main_wnd, text='Image', image=None) 
    img_box.pack()    
    
    btn = Button(main_wnd, text='Exit', command=close_window)
    btn.pack()

    zmq_thread = ZMQThread()
    zmq_thread.start()

    main_wnd.mainloop()
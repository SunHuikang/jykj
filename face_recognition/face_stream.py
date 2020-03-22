#!/usr/bin/env python3

import logging
import socketserver
from threading import Condition, Thread
from PIL import Image
import cv2
import traceback
import io
import time
import sys
import json

from http.server import BaseHTTPRequestHandler, HTTPServer

from recognition_process import face_encode, face_recognize, display_face, generate_json_frame

known_face_encodings = []
known_face_names = []
interval = 25
#json_frame=json.dumps({'frame_id':'', 'time':'', 'objects':''})

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                traceback.print_exc()
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # self.wfile.write(json.dumps(data).encode())

            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(json_frame.encode('UTF-8'))
                    self.wfile.write(b'\r\n')
            except Exception as e:
                pass
        else:
            self.send_error(404)
            self.end_headers()



class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class Camera:
    def __init__(self, video_url, output, show_face):
        self.video_url = video_url
        self.output = output
        self.show_face = show_face

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.video_url))
        self.stop_capture = False
        self.thread = Thread(target=self.capture)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_capture = True
        self.thread.join()
        self.cap.release()

    def capture(self):
        print("Width: %d, Height: %d, FPS: %d" % (self.cap.get(3), self.cap.get(4), self.cap.get(5)))
        framerate = self.cap.get(5)
        frame_duration = 1. / framerate
        frame_id = 1
        frame_id_tmp = 1
        while not self.stop_capture:
            if frame_id_tmp != frame_id:
                frame_id = frame_id_tmp
            start = time.time()
            ret, frame = self.cap.read()
            if ret:
                ########################人脸识别#######################
                #调用人脸识别模块函数
                if frame_id%interval == 1:   #隔帧识别
                    #获取识别结果
                    face_names, face_locations, frame = face_recognize(frame, known_face_names, known_face_encodings)
                    print("recognize frame, Frame_ID=", frame_id, face_names)
                
                global json_frame
                json_frame = generate_json_frame(frame_id, face_names)
                print(type(json_frame),json_frame)
                frame_id = frame_id + 1
                
                #将矩形框识别结果展示在图片上
                if(int(self.show_face) == 1):
                    frame = display_face(face_names, face_locations, frame)
                
                #if frame_id == 20000:
                #    frame_id = 0
                #######################################################
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.save(self.output, format='JPEG', quality=20)
            elapsed = time.time() - start
            logging.debug("Frame acquisition time: %.2f" % elapsed)
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            frame_id_tmp = frame_id_tmp + 1

try:
    print("face encoding")
    known_face_names, known_face_encodings = face_encode(str(sys.argv[2]))   #编码人脸库
    print("face encoding finish, start recognition")


    output = StreamingOutput()
    with Camera(sys.argv[1], output, sys.argv[4]) as camera:
        address = ('', int(sys.argv[3]))
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()

except KeyboardInterrupt:
    pass

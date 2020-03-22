#coding=utf-8
import face_recognition
import cv2
import numpy as np
import os
from os import listdir, getcwd
from os.path import join
import json
import time
from PIL import Image, ImageFont, ImageDraw

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

def face_encode(face_dir):
    known_face_encodings = []
    known_face_names = []
    try:
        for each_face in os.listdir(face_dir):
            face_image = face_recognition.load_image_file(face_dir+'/'+each_face)
            face_codes = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_codes)
            known_face_names.append(each_face.split('.')[0].split('_')[0]+'_'+each_face.split('.')[0].split('_')[1])
        return known_face_names, known_face_encodings
    except:
        print("Faces Encoding Error")


def face_recognize(frame, known_face_names, known_face_encodings):
    # 初始化变量
    face_locations = []
    face_encodings = []
    face_names = []
    #process_this_frame = True

    # 将图像缩放为原来的1/4加快识别速度
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

    # 将BGR通道（OpenCV中使用）转换为RGB通道（face_recognition中使用）
    rgb_small_frame = small_frame[:, :, ::-1]

    # 找到每一帧中的人脸并对其进行编码
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 比较当前人脸是否存在于人脸库中
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #best_match_index = np.argmin(face_distances)
        #if matches[best_match_index]:
        #    name = known_face_names[best_match_index]

        face_names.append(name)

    return face_names, face_locations, frame

def display_face(face_names, face_locations, frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != 'Unknown':
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 10
            right *= 10
            bottom *= 10
            left *= 10

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            #中文字体
            fontpath = "./font/simhei.ttf"  # 宋体字体文件
            font_1 = ImageFont.truetype(fontpath, 30)  # 加载字体, 字体大小
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((left + 10, bottom - 32), name.split('_')[1], font=font_1, fill=(255, 255, 255))  # xy坐标, 内容, 字体, 颜色
            frame = np.array(img_pil)

    return frame

def generate_json_frame(frame_id, face_names):
    
    #获取系统时间
    time_stamp = time.time()  # 当前时间的时间戳
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time) #格式化时间输出

    #构造Json
    face_names_known = []
    for elem_known in face_names:
        if elem_known!='Unknown':
            face_names_known.append(elem_known.split('_')[0])

    json_dict = {'frame_id':frame_id, 'time':str_time, 'objects':face_names_known}
    json_frame = json.dumps(json_dict, ensure_ascii=False)

    return json_frame


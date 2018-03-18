from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys, csv
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy import misc
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--parent_folder", required=True, help="Name of parent folder")
args = vars(ap.parse_args())
parent_folder = args['parent_folder']

home_dir = '/data0/krohitm/posture_dataset/scott_vid/facenet_dataset/{0}/testing'.format(parent_folder)
results_dir = '/data0/krohitm/posture_dataset/scott_vid/facenet_dataset/{0}/results'.format(parent_folder)

ovthresh = 0.5

bbox = pd.DataFrame(columns=['image_name', 'xmin', 'ymin', 'xmax', 'ymax'])

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        #pnet, rnet, onet = detect_face.create_mtcnn(sess, './Path to det1.npy,..')
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '/home/krohitm/code/facenet/src/align')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.9]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        #HumanNames = ['Human_a','Human_b','Human_c','...','Human_h']    #train human name
        HumanNames = ['control', 'patient']

        print('Loading feature extraction model')
        #modeldir = '/..Path to pre-trained model../20170512-110547/20170512-110547.pb'
        modeldir = '/home/krohitm/code/facenet/src/models/20170512-110547/20170512-110547.pb'
        
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        #classifier_filename = '/..Path to classifier model../my_classifier.pkl'
        classifier_filename = '/home/krohitm/code/facenet/src/models/lfw_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        #video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        prevTime = 0
        
        _,_,files = os.walk(os.path.join(home_dir,parent_folder)).next()
        files.sort()
        counter = 0
        for frame in files:
            counter+=1
            if counter >= 10:
                break
            img_name = frame
            if not (frame.endswith(".jpg") or frame.endswith(".png")):
                continue
            #ret, frame = video_capture.read()

            img_full = os.path.join(home_dir, parent_folder, frame)
            print (img_full)
            frame = misc.imread(img_full)
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[0] = facenet.flip(cropped[0], False)
                        scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                        scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[0] = facenet.prewhiten(scaled[0])
                        scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                        boundaries = {'image_name':[img_full], 'xmin':[bb[i][0]], 
                                      'ymin':[bb[i][1]], 'xmax':[bb[i][2]], 
                                      'ymax':[bb[i][3]]}
                        bbox_temp = pd.DataFrame(data=boundaries)
                        #bbox_temp = pd.DataFrame([img_full, bb[i][0], bb[i][1], 
                         #                      bb[i][2], bb[i][3]])
                        print (bbox_temp)
                        bbox = bbox.append(bbox_temp, ignore_index=True)

                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        # print('result: ', best_class_indices[0])
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')
                    bbox_temp = pd.concat([img_full, -1, -1, -1, -1], axis = 1)
                    bbox = bbox.append(bbox_temp, ignore_index=True)
                    

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            plt.imshow(frame[:,:,[0,1,2]])
            #plt.axis('off')
            #fig = plt.gcf()
            #fig.set_size_inches(12, 12)
            #fig.savefig(results_dir+'/'+img_name)

        #video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()
        bbox.to_csv(os.path.join('/data0/krohitm/posture_dataset/scott_vid/facenet_dataset', 
                                 parent_folder, 'face_boundaries.csv'), index=False)

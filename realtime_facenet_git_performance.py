from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
#from os.path import join as pjoin
#import sys
import  csv
import time
#import copy
#import math
import pickle
#from sklearn.svm import SVC
#from sklearn.externals import joblib

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--parent_folder", required=True, help="Name of parent folder")
args = vars(ap.parse_args())
parent_folder = args['parent_folder']

#home_dir = '/data0/krohitm/posture_dataset/scott_vid/realtime_deep_face/testing'
home_dir = '/data0/krohitm/posture_dataset/scott_vid/facenet_dataset/{0}/testing'.format(parent_folder)
with open (os.path.join(home_dir, 'bbox_final.csv'), 'r') as f:
    print ("Reading ground truth bbox file")
    reader = csv.reader(f, delimiter = ',')
    coords = []
    for row in reader:
        coords.append(row)
f.close()

BBGT = np.array(coords)
BBGT = BBGT[1:]
#npos = len(BBGT)
npos = 0
TP = np.zeros(len(BBGT))
FP = np.zeros(len(BBGT))
FN = np.zeros(len(BBGT))
ovthresh = 0.5


print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        #pnet, rnet, onet = detect_face.create_mtcnn(sess, './Path to det1.npy,..')
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '/home/krohitm/code/facenet/src/align')

        minsize = 20  # minimum size of face
        #threshold = [0.6, 0.7, 0.7]  # original three steps's threshold
        threshold = [0.6, 0.7, 0.9]  # three steps's threshold for iHealth
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 1
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        #HumanNames = ['Human_a','Human_b','Human_c','...','Human_h']    #train human name
        #HumanNames = ['2017-07-12-1127-30']
        HumanNames = ['control',parent_folder]

        print('Loading feature extraction model')
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

        
        image_dir = '/data0/krohitm/posture_dataset/scott_vid/facenet_dataset/{0}/testing/{1}'.format(parent_folder,parent_folder)
        #image_dir = '/data0/krohitm/posture_dataset/scott_vid/facenet_dataset_jpg/2017-06-19-1435-27'
        #store_folder = '/data0/krohitm/face_recognition_data/face_rec/017-07-12-1127-30'
        #try:
        #    os.mkdir(store_folder)
        #except OSError:
        #    pass

        _,_,image_names = os.walk(image_dir).next()
        image_names.sort()
        #video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        prevTime = 0
        #while True:
        #    ret, frame = video_capture.read()
        for j in range(len(BBGT)):
        #for j in range(len(image_names)):
            img_full = BBGT[j][0]
            #img_full = os.path.join(image_dir, image_names[j])
            #print (img_full)
            frame = misc.imread(img_full)
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            
            #get the gound truth bbox for the image
            #BBGT = np.array(coords[j][1:5])#.astype(float)
            #print (BBGT)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            #if c <=200:
            #    c += 1
            #    continue
            if (c % timeF == 0):
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                #print(('Detected_FaceNum: {0} in frame {1}.').format(nrof_faces, c))

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
                        #print (best_class_probabilities)
                        #print (best_class_indices)
                       
                        #setting threshold on confidence level
                        if best_class_probabilities[0] >=0.90 and HumanNames[best_class_indices[0]] == parent_folder:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            
                            
                            #check for accuracy
                            #annotation saves twice the size, so have to divide ground truth by 2
                            ixmin = np.maximum(float(BBGT[j,1])/2, bb[i,0])
                            iymin = np.maximum(float(BBGT[j,2])/2, bb[i,1])
                            ixmax = np.minimum(float(BBGT[j,3])/2, bb[i,2])
                            iymax = np.minimum(float(BBGT[j,4])/2, bb[i,3])
                            iw = np.maximum(ixmax - ixmin + 1., 0.)
                            ih = np.maximum(iymax - iymin + 1., 0.)
                            inters = iw * ih
                            
                             #union
                            uni = ((bb[i,2] - bb[i,0] + 1.) * (bb[i,3] - bb[i,1] + 1.) +
                                    (float(BBGT[j,3])/2 - float(BBGT[j,1])/2 + 1.) *
                                    (float(BBGT[j,4])/2 - float(BBGT[j,2])/2 + 1.) - inters)
                                        
                            
                            overlaps = inters / uni
                            ovmax = np.max(overlaps)
                            jmax = np.argmax(overlaps)
                             
                            if ovmax > ovthresh:
                                TP[j] = 1.
                                npos += 1
                            else:
                                FP[j] = 1.
                                if BBGT[j][1] != BBGT[j][2] != BBGT[j][3] != BBGT[j][4] != -1:
                                    npos += 1
                                
                                
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(best_class_probabilities[0]), (text_x, bb[i][1] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')
                    #if face was present but didn't detect
                    if BBGT[j][1] != BBGT[j][2] != BBGT[j][3] != BBGT[j][4] != -1:
                        FN[j] = 1
                        npos += 1
            

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            fpsStr = 'FPS: %2.3f' % fps
            print (fpsStr)
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, fpsStr, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # compute precision recall
        FP = np.cumsum(FP)
        TP = np.cumsum(TP)
        #print (FP[-1])
        #print (TP[-1])
        #print (npos)
        #print (TP)
        rec = TP / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        accuracy = TP / np.maximum(npos + FP, np.finfo(np.float64).eps)
        #video_capture.release()
        # #video writer
        # out.release()
        print ("Recall: {0}, Precision: {1}, accuracy: {2}, Average precision: {3}".format(
                rec[-1], prec[-1], accuracy[-1], ap))
        pickle.dump( "Recall: {0}, Precision: {1}, accuracy: {2}, Average precision: {3}".format(
                rec[-1], prec[-1], accuracy[-1], ap), open( os.path.join(home_dir, "results.p"), "wb" ) )
        #with open (os.path.join(home_dir, 'results.csv'), 'w') as f:
        #    writer = csv.writer(f, delimiter = ',')
        #    writer.writerows(coords)
        #f.close()
        cv2.destroyAllWindows()

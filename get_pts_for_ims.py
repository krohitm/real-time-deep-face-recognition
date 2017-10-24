# Import the required modules
import cv2
#import argparse
import numpy as np

def run(im,coords,frame_num, multi=True):
    im_disp = im.copy()
    im_draw = im.copy()
    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1768, 992)
    if np.max(coords) > 0:
       cv2.rectangle(im_draw, (coords[0]-4,coords[1]-5),(coords[2],coords[3]), (255, 255, 255), 3)
    cv2.putText(im_draw, frame_num, (0,30) , cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    cv2.imshow(window_name, im_draw)

    # List containing top-left and bottom-right to crop the image.
    pts_1 = []
    pts_2 = []

    rects = []
    run.mouse_down = False

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if multi == False and len(pts_2) == 1:
                print "WARN: Cannot select another object in SINGLE OBJECT TRACKING MODE."
                print "Delete the previously selected object using key `d` to mark a new location."
                return
            run.mouse_down = True
            pts_1.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down == True:
            run.mouse_down = False
            pts_2.append((x, y))
            print "Object selected at [{}, {}]".format(pts_1[-1], pts_2[-1])
        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down == True:
            im_draw = im.copy()
            cv2.rectangle(im_draw, (pts_1[-1][0]-4,pts_1[-1][1]-5), (x, y), (255,255,255), 3)
            cv2.imshow(window_name, im_draw)

    print "Press and release mouse around the object to be tracked. \n You can also select multiple objects."
    cv2.setMouseCallback(window_name, callback)
    
    print "Press key `n` to continue with the previous points."
    print "Press key `d` to continue with the selected points."
    print "Press key 'w' to skip the frames, in case no identifiable object exists"
    print "Press key `f` to discard the last object selected."
    print "Press key `s` to quit the program."

    while True:
        # Draw the rectangular boxes on the image
        for pt1, pt2 in zip(pts_1, pts_2):
            rects.append([pt1[0],pt2[0], pt1[1], pt2[1]])
            cv2.rectangle(im_disp, (pt1[0]-4,pt1[1]-5), pt2, (255, 255, 255), 3)
        # Display the cropped images
        #cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
        #cv2.imshow(window_name_2, im_disp)
        key = cv2.waitKey(30)
        # Press key 'w' to skip the frames, in case no identifiable object exists
        if key == ord('w'):
            return [-1,-1,-1,-1]
        elif key == ord('d'):
            # Press key `d` to return the selected points
            cv2.destroyAllWindows()
            point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
            #print point
            corrected_point=check_point(point)
            if len(corrected_point) == 1:
                out_area=(corrected_point[2]-corrected_point[0])*(corrected_point[3]-corrected_point[1])
                if out_area < 50:
                    return_prev=coords
                    return return_prev
            #print corrected_point
            return corrected_point
        elif key == ord('s'):
            # Press key `s` to quit the program
            print "Quitting with saving."
            cv2.destroyAllWindows()
            return_noth=[0,0,0,0]
            return return_noth
        elif key == ord('f'):
            # Press key `f` to delete the last rectangular region
            if run.mouse_down == False and pts_1:
                print "Object deleted at  [{}, {}]".format(pts_1[-1], pts_2[-1])
                pts_1.pop()
                pts_2.pop()
                im_disp = im.copy()
            else:
                print "No object to delete."
        elif key == ord('n'):
            #press key 'n' to use the bboxes from the previous frames
            return [-5,-5,-5,-5]
    cv2.destroyAllWindows()
    point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
    corrected_point=check_point(point)
    print corrected_point.shape
    return corrected_point

def check_point(points):
    out=np.zeros((1,4),dtype=np.int)
    for point in points:
        #to find min and max x coordinates
        if point[0]<point[2]:
            minx=point[0]
            maxx=point[2]
        else:
            minx=point[2]
            maxx=point[0]
        #to find min and max y coordinates
        if point[1]<point[3]:
            miny=point[1]
            maxy=point[3]
        else:
            miny=point[3]
            maxy=point[1]
        out=[minx,miny,maxx,maxy]

    return out

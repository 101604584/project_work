'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import cv2.aruco as aruco
import sys
from utils import ARUCO_DICT
import argparse
import time
import math


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 
            
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec_list, tvec_list, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 20, matrix_coefficients, distortion_coefficients)
            
            rvec = rvec_list[0][0]
            tvec = tvec_list[0][0]
            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 10) 
            
            rvec_flip = rvec * -1
            tvec_flip = tvec * -1
            rotation_matrix, jacobian = cv2.Rodrigues(rvec_flip)
            realworld_tvec = np.dot(rotation_matrix, tvec_flip)
        
            pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)
            x = "{:.2f}".format(realworld_tvec[0])
            y = "{:.2f}".format(realworld_tvec[1])
            z = "{:.2f}".format(realworld_tvec[2])
            pitch = "{:.2%}".format(pitch)
            roll = "{:.2%}".format(roll)
            yaw = "{:.2%}".format(yaw)
            print("X = ",x, "Y = ", y, "Z = ", z, "Pitch = ", pitch, "Roll = ", roll, "Yaw = ", yaw)  

    return frame
    
    
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
    
    
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z])

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    marker_size = 100
    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)
        
        cv2.imshow('Estimated Pose', output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
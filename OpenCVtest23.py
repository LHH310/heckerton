import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def draw_3d_box(image, landmarks, connections):
    # 기존 함수와 동일

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

image_path = "img3.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"이미지를 읽을 수 없습니다: {image_path}")
    exit()

image = cv2.resize(image, (500, 500))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 양쪽 어깨와 골반의 중간점 계산
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        
        # 척추의 방향 벡터 계산
        spine_vector = mid_shoulder - mid_hip
        
        # 수직 벡터 (y축)
        vertical_vector = np.array([0, -1])
        
        # 척추와 수직 벡터 사이의 각도 계산
        angle = np.degrees(np.arccos(np.dot(spine_vector, vertical_vector) / 
                                     (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))))
        
        # 측면 이미지 여부 확인
        is_side_view = abs(left_shoulder[0] - right_shoulder[0]) < 0.1
        
        if is_side_view:
            # 측면 이미지일 경우 허리 각도 계산
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            angle = calculate_angle(shoulder, hip, knee)
        
        # 자세 판단 및 표시
        if is_side_view:
            if angle > 170:
                posture = "Straight"
                color = (0, 255, 0)  # 녹색
            elif angle > 150:
                posture = "Slightly bent"
                color = (0, 255, 255)  # 노란색
            else:
                posture = "Bent"
                color = (0, 0, 255)  # 빨간색
        else:
            if angle < 10:
                posture = "Straight"
                color = (0, 255, 0)  # 녹색
            elif angle < 20:
                posture = "Slightly bent"
                color = (0, 255, 255)  # 노란색
            else:
                posture = "Bent"
                color = (0, 0, 255)  # 빨간색
        
        # 결과 표시
        cv2.putText(image, f'Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Posture: {posture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 결과 이미지 표시
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Pose', 500, 500)
    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

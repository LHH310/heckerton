import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

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
        
        # 어깨, 골반, 무릎의 좌표 추출
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        
        # 측면 이미지 여부 확인 (어깨 너비와 골반 너비 비교)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_width = np.linalg.norm(right_hip - left_hip)
        is_side_view = shoulder_width < hip_width * 0.7
        
        if is_side_view:
            # 측면 이미지일 경우 허리 각도 계산
            if left_shoulder[0] < right_shoulder[0]:  # 왼쪽을 바라보고 있는 경우
                angle = calculate_angle(left_shoulder, left_hip, left_knee)
            else:  # 오른쪽을 바라보고 있는 경우
                angle = calculate_angle(right_shoulder, right_hip, right_knee)
        else:
            # 정면 이미지일 경우 척추 기울기 계산
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            mid_hip = (left_hip + right_hip) / 2
            spine_vector = mid_shoulder - mid_hip
            vertical_vector = np.array([0, -1])
            angle = np.degrees(np.arccos(np.dot(spine_vector, vertical_vector) / 
                                         (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))))
        
        # 자세 판단 및 표시
        if is_side_view:
            if angle > 170:
                posture = "Straight"
                color = (0, 255, 0)  # 녹색
            elif angle > 140:
                posture = "Slightly bent"
                color = (0, 255, 255)  # 노란색
            else:
                posture = "Bent"
                color = (0, 0, 255)  # 빨간색
        else:
            if angle < 5:
                posture = "Straight"
                color = (0, 255, 0)  # 녹색
            elif angle < 15:
                posture = "Slightly bent"
                color = (0, 255, 255)  # 노란색
            else:
                posture = "Bent"
                color = (0, 0, 255)  # 빨간색
        
        # 결과 표시
        cv2.putText(image, f'Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Posture: {posture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(image, f'View: {"Side" if is_side_view else "Front"}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 결과 이미지 표시
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Pose', 500, 500)
    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# landmarkの繋がり表示用
landmark_line_ids = [ 
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指
    (17, 18), (18, 19), (19, 20),   # 小指
]


def main():

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )    
    
    
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        success,image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = image.shape
        if results.multi_hand_landmarks:
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for line_id in landmark_line_ids:
                    # 1点目座標取得
                    lm = hand_landmarks.landmark[line_id[0]]
                    lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
                    # 2点目座標取得
                    lm = hand_landmarks.landmark[line_id[1]]
                    lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                    # line描画
                    cv2.line(image, lm_pos1, lm_pos2, (128, 0, 0), 1)

        cv2.imshow('MediaPipe Hands Detection', cv2.flip(image,1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()

if __name__== "__main__":
    main()

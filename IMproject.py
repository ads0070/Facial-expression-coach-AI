import tkinter as tk   # Tk GUI 툴킷에 대한 표준 파이썬 인터페이스
from PIL import ImageTk, Image # 이미지 처리를 위한 라이브러리
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib import font_manager, rc

""" 데이터 모델 다운로드 코드
!if not exist "./files" mkdir files
# Download Face detection XML
!curl -L -o ./files/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# Download emotion trained data
!curl -L -o ./files/emotion_model.hdf5 https://mechasolution.vn/source/blog/AI-tutorial/Emotion_Recognition/emotion_model.hdf5
"""

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 캡쳐 이미지 저장 리스트
capture_expression = [""]*7

# 각 감정 퍼센트 저장 리스트
value_expression = [0] * 7

# 7가지 감정 리스트
EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]
EMOTIONS_KR = ["분노", "혐오", "공포", "행복", "슬픔", "놀람", "중립"]

count = {} # 각 감정의 측정 횟수를 기록할 dict형 변수

# 전역 변수 초기화 함수
def init_value():
    global capture_expression, value_expression, count
    capture_expression = [""] * 7
    value_expression = [0] * 7
    count = {}

# 루키AI 실행 함수
def start_lookieAI():
    global capture_expression, value_expression, EMOTIONS, count
    
    init_value() # 초기화 함수 호출
    
    # Face detection XML load and trained model loading -얼굴 감지 XML로드 및 훈련 된 모델로드
    face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
    
    E_values = []
    for i in EMOTIONS:
        count[i] = 0    # "Angry":0 ,"Disgusting":0 ,"Fearful":0, "Happy":0, "Sad":0, "Surpring":0, "Neutral":0
    
    # Video capture using webcam 웹캠을 사용한 비디오 캡처
    camera = cv2.VideoCapture(0)
    
    while True:
        # Capture image from camera 카메라에서 이미지 캡처
        ret, frame = camera.read()
        
        # Convert color to gray scale 색상을 회색조로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection in frame -프레임 내 얼굴 감지
        faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30))
        
        # Perform emotion recognition only when face is detected - 얼굴이 감지 된 경우에만 감정 인식 수행
        if len(faces) > 0:
            # For the largest image 가장 큰 이미지
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            
            # Resize the image to 48x48 for neural network 신경망 용 이미지 크기를 48x48로 조정
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Emotion predict 감정 예측
            preds = emotion_classifier.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            # 가장 높은 감정의 count를 1만큼 증가
            count[label] += 1
            
            # Assign labeling 라벨 지정
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 3)
            
            # 웹캠 캡쳐본 저장
            if preds[preds.argmax()]*100 > value_expression[preds.argmax()]:
                value_expression[preds.argmax()] = preds[preds.argmax()]*100
                capture_expression[preds.argmax()] = frame
            
        ## Display image ("Emotion Recognition") 디스플레이 이미지 ( "감정 인식")
        cv2.imshow('Expression CAM', frame)
        
        # q to quit 종료하려면 q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            for i in count.values():
                E_values.append(i)
            
            # 배경 하얀색으로 변경
            white_bg = tk.Label(window, bg = "white", width = 500, height = 32)
            white_bg.place(x=0,y=113)
            
            # 그래프 출력
            result_graph = plt.Figure(figsize=(6,6))
            ax1 = result_graph.add_subplot(111)
            
            barChart = FigureCanvasTkAgg(result_graph, window)
            barChart.get_tk_widget().place(x=0, y=135)
            
            ax1.set_title("표정 통계", fontsize = 15)
            ax1.bar(EMOTIONS_KR, E_values, color = '#2f5597')
            ax1.set_xticklabels(EMOTIONS_KR, fontsize=12)
            
            # 리스트 컴프리헨션 이용
            # 최대값이 2개 이상인 경우 감정 %가 더 높게 나온 것을 최대로 설정
            most_value = [k for k,v in count.items() if max(count.values()) == v]
            most_index = -1
            if len(most_value) > 1:
                check_num = 0
                for i in range(len(most_value)):
                    index_num = EMOTIONS.index(most_value[i])
                    if value_expression[index_num] > check_num:
                        most_index = index_num
            elif len(most_value) == 1:
                most_index = EMOTIONS.index(max(count,key=count.get))
            
            # 돌발 표정 인덱스
            min_value = max(count.values())
            check_index = 0
            min_index = 0
            isface = False
            
            for value in count.values():
                if value != 0:
                    if value < min_value:
                        min_value = value
                        min_index = check_index
                        isface = True
                check_index += 1
            
            # 프레임에 측정된 결과값과 가장 많이 나온 감정 출력
            result_label = tk.Label(window, text = "당신의 표정을 분석하여 측정한 결과,", font = ("맑은 고딕",12), bg="white")
            result_label.place(x=500,y=165)
            
            text1 = ""
            text2 = ""
            for i in range(len(EMOTIONS_KR)):
                text1 += "{} 표정은\n".format(EMOTIONS_KR[i])
                text2 += "{} 회\n".format(str(count[EMOTIONS[i]]))
            
            
            # 측정 결과 출력
            text_label1 = tk.Label(window, text = text1, font = ("맑은 고딕",12), bg="white")
            text_label1.place(x=500,y=225)
            text_label1.config(justify = "left")
            text_label2 = tk.Label(window, text = text2, font = ("맑은 고딕",12), bg="white")
            text_label2.place(x=610,y=225)
            text_label2.config(justify = "right")
            
            if most_index == -1:
                noData_label = tk.Label(window, text = "측정 값이 없습니다.", font = ("맑은 고딕",12), fg = "red", bg="white")
                noData_label.place(x=500,y=400)
            else:
                text_label3 = tk.Label(window, text = "측정되었습니다.", font = ("맑은 고딕",12), bg="white")
                text_label3.place(x=680,y=351)
                
                most_label = tk.Label(window, text = "대체적으로 측정된 표정은 '{}' 이고,".format(EMOTIONS_KR[most_index]), font = ("맑은 고딕",12), bg="white")
                most_label.place(x=500,y=400)
                if not isface:
                    noData_label2 = tk.Label(window, text = "돌발적으로 측정된 표정은 없습니다.", font = ("맑은 고딕",12), bg="white")
                    noData_label2.place(x=500,y=425)
                else:
                    least_label = tk.Label(window, text = "돌발적으로 측정된 표정은 '{}' 입니다.".format(EMOTIONS_KR[min_index]), font = ("맑은 고딕",12), bg="white")
                    least_label.place(x=500,y=425)
            
            text_label4 = tk.Label(window, text = "아래의 버튼을 눌러 각각의 순간 표정을 확인하세요.", font = ("맑은 고딕",12), bg="white")
            text_label4.place(x=500,y=450)
            text_label5 = tk.Label(window, text = "", font = ("맑은 고딕",12), bg="white")
            text_label5.place(x=500,y=475)
            
            # 가장 많이 나온 표정 확인 버튼 생성
            checkMax_btn = tk.Button(window, text="표정확인", width=8, height=2, command = checkExpression)
            checkMax_btn.place(x=570,y=500)
            
            # 돌발 표정 확인 버튼 생성
            checkMin_btn = tk.Button(window, text="돌발표정", width=8, height=2, command = checkOutbreakExpression)
            checkMin_btn.place(x=670,y=500)
            
            break
    
    # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()

# 루키AI 종료
def close_lookieAI():
    window.destroy()

# 루키AI 매뉴얼
def manual_lookieAI():
    # 라벨 초기화 (공백으로 덮어씌움)
    error_text = tk.Label(window, text = "            ", bg="white")
    error_text.place(x=575,y=545)
    error_text2 = tk.Label(window, text = "            ", bg="white")
    error_text2.place(x=675,y=545)
    
    manual = cv2.imread("manual.png", cv2.IMREAD_ANYCOLOR)
    cv2.imshow('Manual', manual)
    if cv2.waitKey():
        cv2.destroyAllWindows()
        start_lookieAI()

# 가장 많이 나온 표정 확인
def checkExpression():
    global capture_expression, value_expression, EMOTIONS, count
    
    if sum(value_expression) == 0:
        error_text = tk.Label(window, text = "없습니다.", fg = '#FA412C', bg="white")
        error_text.place(x=575,y=545)
    else:
        ex_index = EMOTIONS.index(max(count,key=count.get))
            
        frequent_face = capture_expression[ex_index]
        font=cv2.FONT_HERSHEY_SIMPLEX
        
        value_text = "{:.2f}%".format(value_expression[ex_index])
        
        expression_text = EMOTIONS[ex_index]
        
        cv2.putText(frequent_face, expression_text+value_text ,(50,50),font,1,(255,0,0),2)
        cv2.imshow('Frequent Expression', frequent_face)
        
        if cv2.waitKey():
            cv2.destroyAllWindows()

# 돌발 표정 확인
def checkOutbreakExpression():
    global capture_expression, value_expression, EMOTIONS, count
    
    min_value = max(count.values())
    check_index = 0
    min_index = 0
    isface = False
    
    # 돌발 표정 인덱스 추출
    for value in count.values():
        if value != 0:
            if value < min_value:
                min_value = value
                min_index = check_index
                isface = True
        check_index += 1
    
    if not isface:
        error_text = tk.Label(window, text = "없습니다.", fg = '#FA412C', bg="white")
        error_text.place(x=675,y=545)
    else:
        outbreak_face = capture_expression[min_index]
        font=cv2.FONT_HERSHEY_SIMPLEX
        
        value_text = "{:.2f}%".format(value_expression[min_index])
        expression_text = EMOTIONS[min_index]
        
        cv2.putText(outbreak_face, expression_text+value_text ,(50,50),font,1,(255,0,0),2)
        cv2.imshow('Outbreak Expression', outbreak_face)
    
        if cv2.waitKey():
            cv2.destroyAllWindows()

# GUI 구현
window = tk.Tk() # TK 객체 생성

window.title("LOOKie AI")
window.geometry("900x700+50+50")
window.resizable(False, False) # 창 사이즈를 변환할 수 없도록 설정

background_img = Image.open("./start.png")  # 임시로 넣어둔 이미지 start.png

# 배경 이미지 사이즈 및 위치 설정
bg_img = background_img.resize((900,700))
background_tkimg = ImageTk.PhotoImage(bg_img)
background_label = tk.Label(window, image=background_tkimg)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# 시작 버튼 (클릭하면 manual_lookieAI 함수 호출)
start_btn = tk.Button(window, text="시작하기", width=8, height=2, command = manual_lookieAI)
start_btn.place(x=50,y=630)

# 종료 버튼 (클릭하면 close_lookieAI 함수 호출)
close_btn = tk.Button(window, text="끝내기", width=8, height=2, command = close_lookieAI)
close_btn.place(x=150,y=630)

window.mainloop()
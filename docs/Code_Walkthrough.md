# Pupil Labs U-Mamba: 실무자를 위한 코드 딥다이브 가이드 (Code Walkthrough)

이 문서는 **실제 코드를 유지보수하고 수정해야 하는 개발자를 위한 코드 레벨 해설서** 입니다.

동공을 추적하는 과정에서 프레임 1장이 시스템 내부에서 어떻게 흘러가고, 우리가 짠 U-Mamba 코드가 언제 어떻게 호출되는지 아주 구체적으로 파헤칩니다.

---

## 목차
1. [전체 파이프라인: 카메라 프레임의 일생](#1-전체-파이프라인-카메라-프레임의-일생)
2. [이벤트 트리거: `detector_base_plugin.py`](#2-이벤트-트리거-detector_base_pluginpy)
3. [U-Mamba 핵심 로직: `detector_2d_plugin.py`](#3-u-mamba-핵심-로직-detector_2d_pluginpy)
4. [데이터의 표준 규격: `datum` 이란?](#4-데이터의-표준-규격-datum-이란)
5. [미래를 위한 수정 가이드 (How to modify)](#5-미래를-위한-수정-가이드-how-to-modify)

---

## 1. 전체 파이프라인: 카메라 프레임의 일생

Pupil Capture 프로그램을 켜면 내부적으로 여러 개의 **프로세스(Process)** 가 돌아갑니다.

1. **`main.py`**: 제일 먼저 실행되는 대장입니다. 여기서 `weights_only=False` 몽키패치를 적용한 뒤, ** World Process **(풍경 카메라)와 ** Eye Process**(눈 카메라)를 각각 독립된 프로세스로 띄웁니다.
2. **카메라 캡처**: Eye Process가 카메라 하드웨어로부터 1초에 수십 장씩 눈 사진(`frame`)을 찍어냅니다.
3. **이벤트 브로드캐스팅**: 사진이 찍힐 때마다 Eye Process 내부의 모든 플러그인들에게 "새 프레임 도착했어!"라고 알림(`recent_events`)을 보냅니다.
4. **2D 동공 검출 (U-Mamba)**: 알림을 받은 `detector_2d_plugin.py`가 사진을 가로채서 U-Mamba 모델에 집어넣고, 타원을 그려서 2D 좌표(`datum`)를 만들어냅니다.
5. **3D 안구 추적 (Pye3D)**: 알림을 받은 `pye3d_plugin.py`가 조금 늦게 일어나서, 아까 만들어둔 2D 좌표를 가져다가 3D 안구 모델을 업데이트합니다.

---

## 2. 이벤트 트리거: `detector_base_plugin.py`

이 파일은 모든 검출기 플러그인의 **'부모(Base)'** 역할을 합니다. 
가장 중요한 메서드는 바로 **`recent_events(self, event)`** 입니다.

```python
# pupil_src/shared_modules/pupil_detector_plugins/detector_base_plugin.py

def recent_events(self, event):
    frame = event.get("frame", None) # 1. 새 프레임 꺼내기

    if not frame or not self.enabled:
        return

    # 2. 이전에 실행된 플러그인들의 결과물 가져오기
    previous_detection_results = event.get(EVENT_KEY, [])

    # 3. ★ 핵심 분기점 (U-Mamba 호출 로직)
    if hasattr(self, 'detect_umamba'):
        # 내 자신(self)이 detector_2d_plugin 이라면 이 경로를 탑니다.
        detection_result = self.detect_umamba(frame=frame)
    else:
        # 내 자신(self)이 pye3d_plugin 이라면 원래 있던 detect()를 탑니다.
        detection_result = self.detect(
            frame=frame,
            previous_detection_results=previous_detection_results,
        )

    # 4. 방금 찾은 동공 결과를 버스(EVENT_KEY)에 태워 보냅니다.
    event[EVENT_KEY] = previous_detection_results + [detection_result]
```

> **왜 이렇게 분기했나요?**
> 모든 플러그인(`Pye3D` 포함)은 이 부모 클래스를 상속받습니다. 만약 무식하게 `self.detect_umamba()`를 강제로 호출해버리면, `detect_umamba` 함수가 없는 `Pye3D` 플러그인 차례에서 `AttributeError`가 나며 프로그램이 뻗어버립니다. 그래서 `hasattr`(함수가 존재하는가?)로 안전하게 분기한 것입니다.

---

## 3. U-Mamba 핵심 로직: `detector_2d_plugin.py`

프레임이 `detect_umamba(self, frame)` 함수로 넘어오면, 비로소 진짜 딥러닝 연산이 시작됩니다. 실무에서 가장 많이 들여다봐야 할 곳입니다.

### 1단계: 해상도 맞추기 (가장 흔한 에러 발생 지점)
카메라에서 들어오는 원본 사진(`frame.gray`)은 192x192 같은 작은 정사각형일 수 있습니다. 하지만 **U-Mamba는 무조건 400x640 크기의 사진만 먹도록 학습되었습니다.**
```python
# 1. 400x640으로 강제 리사이즈 (찌그러지더라도 해야 함)
resized_img = cv2.resize(frame.gray, (640, 400), interpolation=cv2.INTER_LINEAR)

# 2. PyTorch 텐서로 변환 (1, 1, 400, 640)
tensor_img = torch.from_numpy(resized_img).float().unsqueeze(0).unsqueeze(0)
```

### 2단계: U-Mamba 추론
```python
# nnUNet 예측기에 텐서를 찔러넣음
prediction = self.predictor.predict_single_npy_array(
    tensor_img.numpy(), 
    ...
)
```
이 `prediction` 안에는 0(배경), 1(공막), 2(홍채), 3(동공) 등의 숫자로 이루어진 지도가 들어있습니다.

### 3단계: 동공 마스크 추출 및 복원
```python
# 1. 값이 3(동공)인 곳만 하얗게(255) 칠해서 이진 마스크 생성
pupil_mask = np.zeros_like(prediction_2d, dtype=np.uint8)
pupil_mask[prediction_2d == 3] = 255

# 2. ★ 매우 중요: 마스크를 다시 원래 카메라 해상도(예: 192x192)로 되돌림!
orig_h, orig_w = frame.gray.shape
restored_mask = cv2.resize(pupil_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
```
> **왜 되돌려야 하나요?**
> 나중에 화면에 초록색 타원을 그릴 때, 원본 카메라 화면 크기에 맞춰서 그려야 하기 때문입니다.

### 4단계: 타원 피팅 (OpenCV)
픽셀 형태의 마스크를 수학적인 '타원' 도형으로 변환합니다.
```python
# 하얀색 덩어리의 윤곽선을 땀
contours, _ = cv2.findContours(restored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선에 딱 맞는 타원을 찾음
(cx, cy), (MA, ma), angle_deg = cv2.fitEllipse(best_contour)
```

### 5단계: Datum 포장
찾아낸 타원 정보를 표준 규격인 `datum` 딕셔너리로 묶어서 반환합니다.

---

## 4. 데이터의 표준 규격: `datum` 이란?

Pupil Labs 시스템 전체에서 공용으로 쓰는 **"동공 발견 보고서"** 양식입니다. 파이썬 딕셔너리(`dict`) 형태이며, 다음과 같이 생겼습니다.

```json
{
    "id": "1",                    // 오른쪽 눈인지 왼쪽 눈인지 식별
    "topic": "pupil.1.2d",        // 데이터의 주제 (이벤트 버스용)
    "method": "2d c++",           // 중요! 3D 모델은 이 method가 "2d c++"인 데이터만 주워먹습니다.
    "timestamp": 1234567.89,      // 프레임이 찍힌 시간
    "confidence": 1.0,            // 딥러닝이므로 무조건 1.0(100%)을 줍니다.
    "norm_pos": [0.4, 0.6],       // 전체 화면 대비 동공의 X,Y 비율 좌표
    "diameter": 25.3,             // 동공의 가장 긴 지름
    "ellipse": {
        "center": [80.5, 120.3],  // 픽셀 좌표계 기준 동공 중심
        "axes": [25.3, 20.1],     // 타원의 장축, 단축 길이
        "angle": 45.0             // 타원의 기울기
    }
}
```
**주의사항**: U-Mamba를 썼더라도 `method` 값을 `"umamba"`로 바꾸면 안 됩니다. 하류 파이프라인(Pye3D)이 `"2d c++"`라는 글자만 찾도록 하드코딩되어 있기 때문입니다.

---

## 5. 미래를 위한 수정 가이드 (How to modify)

실무를 하다 보면 코드를 바꿔야 할 일이 반드시 생깁니다.

### Q1. U-Mamba 말고 다른 가벼운 YOLO 모델로 바꾸고 싶다면?
1. `detector_2d_plugin.py` 의 `__init__` 부분에서 `self.predictor` 대신 새로운 YOLO 모델을 로드하세요.
2. `detect_umamba` 함수 내부의 "1단계(해상도 맞추기)"와 "2단계(추론)" 코드를 YOLO용으로 갈아끼우면 끝입니다. 뒤쪽의 타원 피팅(4단계)과 Datum 포장(5단계)은 건드릴 필요가 없습니다.

### Q2. 400x640 해상도 리사이즈를 없애버리고 싶다면?
U-Mamba 모델 자체를 192x192 이미지로 새로 재학습(Fine-tuning)시키지 않는 이상 해상도 리사이즈를 빼면 안 됩니다. 만약 모델을 새로 학습했다면, `detector_2d_plugin.py`에서 `cv2.resize` 로직을 지우고 `frame.gray`를 그대로 텐서로 만들면 됩니다.

### Q3. U-Mamba가 너무 느려서 화면이 버벅인다면?
현재 GPU(CUDA)를 쓰고 있는지 환경 점검이 1순위입니다. 만약 GPU 리소스가 꽉 찼다면, `detector_2d_plugin.py`의 UI 토글 스위치 설정에서 **"Show RITnet vs U-Mamba"** 기능이 켜져있는지 확인하세요. 이 기능이 켜져 있으면 구형 RITnet과 U-Mamba를 동시에 돌리느라 속도가 반토막이 납니다.

---
**[작성자 요약]**
가장 핵심은 **"원본 프레임 -> 400x640 리사이즈 -> 마스크 생성 -> 다시 원본 크기로 복원 -> 타원 추출"** 이라는 모래시계 ⏳ 형태의 흐름을 이해하는 것입니다. 이 흐름만 꿰고 있으면 어디서 에러가 나더라도 `cv2.imshow`를 띄워가며 쉽게 디버깅할 수 있습니다.

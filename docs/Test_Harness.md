# Pupil Labs 딥러닝 동공 디텍터 사용자 가이드 및 실행 매뉴얼 (User Guide & Harness Manual)

(pupil-umamba) byeongjun@server3:~/PycharmProjects/pupil/pupil_src$ cd ~/PycharmProjects/pupil/pupil_src && python main.py
[20:34:10] WARNING  eye0 - uvc: Could not set Value. 'Backlight Compensation'.                                                                                                                           uvc_backend.py:395
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Initializing TemporalUNet and 2D nnUNet models...                                                                           detector_2d_plugin.py:130
           WARNING  eye1 - uvc: Could not set Value. 'Backlight Compensation'.                                                                                                                           uvc_backend.py:395
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Initializing TemporalUNet and 2D nnUNet models...                                                                           detector_2d_plugin.py:130
TemporalUNet created from /home/byeongjun/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/fold_0/checkpoint_best.pth
  Encoder params (frozen): 14,158,944
  Decoder params (trainable): 52,351,256
  Total trainable: 38,192,312
[20:34:11] INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: ✅ TemporalUNet & 2D nnUNet initialized successfully.                                                                       detector_2d_plugin.py:170
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: RITnet model initialized successfully as fallback.                                                                          detector_2d_plugin.py:194
TemporalUNet created from /home/byeongjun/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/fold_0/checkpoint_best.pth
  Encoder params (frozen): 14,158,944
  Decoder params (trainable): 52,351,256
  Total trainable: 38,192,312
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: ✅ TemporalUNet & 2D nnUNet initialized successfully.                                                                       detector_2d_plugin.py:170
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: RITnet model initialized successfully as fallback.                                                                          detector_2d_plugin.py:194
[20:34:26] INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
open(): 그런 파일이나 디렉터리가 없습니다
[20:34:43] INFO     world - calibration_choreography.base_plugin: Stopping  Calibration                                                                                                                  base_plugin.py:574
open(): 그런 파일이나 디렉터리가 없습니다
The maximum number of function evaluations is exceeded.
Function evaluations 100, initial cost 1.2221e+03, final cost 9.9826e+00, first-order optimality 6.70e-03.
`ftol` termination condition is satisfied.
Function evaluations 20, initial cost 3.2332e+00, final cost 1.0544e+00, first-order optimality 6.16e-05.
`ftol` termination condition is satisfied.
Function evaluations 24, initial cost 1.2624e+03, final cost 1.0168e+01, first-order optimality 3.85e-04.
[20:34:45] INFO     world - accuracy_visualizer: Angular accuracy: 15.241 degrees                                                                                                                accuracy_visualizer.py:402
           INFO     world - accuracy_visualizer: Angular precision: 0.095 degrees                                                                                                                accuracy_visualizer.py:411
^C[20:34:56] WARNING  MainProcess - root: Launcher shutting down with active children: [<Process name='eye1' pid=214611 parent=214507 started>, <Process name='eye0' pid=214607 parent=214507 started>, <Process  main.py:351
                    name='world' pid=214522 parent=214507 started>]                                                                                                                                                        
           WARNING  eye0 - uvc: Turbojpeg jpeg2yuv: b'Corrupt JPEG data: premature end of data segment'    

 문서는 **Pupil Labs 딥러닝 동공 디텍터 모듈(`Detector2DPlugin`)의 실행 환경, GUI 조작법, 오프라인 검증, 트러블슈팅 및 캘리브레이션/검증 타겟 좌표 설정법을 정리한 유저 매뉴얼**입니다.

---

## 1. 빠른 실행 요약 (Quick Reference)

| 작업 목적 | 실행 명령어 / 조작 방법 | 비고 |
|:---|:---|:---|
| **가상환경 활성화** | `conda activate pupil-umamba` | Python 3.10 + PyTorch CUDA 환경 |
| **Pupil Capture GUI 실행** | `cd ~/PycharmProjects/pupil/pupil_src && python main.py` | 실시간 AR 글래스 및 카메라 수신 |
| **오프라인 더미 검증 실행** | `cd ~/PycharmProjects/pupil && python tests/test_dummy_harness.py` | 하드웨어 없이 200 FPS 오프라인 검증 |
| **디텍터 모델 전환 (UI)** | `Eye -> 2D Detector -> Active Model` 드롭다운 | `TemporalUNet` (메인) / `nnUNet 2D` / `RITnet` / `2D C++` |
| **Left Eye (Eye 0) 반전** | `Eye -> 2D Detector -> Flip Vertically` 체크 | 광학 거울 180도 뒤집힘 보정 |

---

## 2. 필수 실행 환경 (Environment Setup)

### 2.1 Conda 가상환경 활성화
본 시스템은 **`pupil-umamba`** Conda 가상환경(Python 3.10, PyTorch CUDA 12, NumPy 1.26.4)에서 구동됩니다.

```bash
# 1. Conda 가상환경 활성화
conda activate pupil-umamba

# 2. 필수 환경 변수 확인 (자동 설정)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 2.2 디렉토리 및 모델 경로 구조

- **Pupil 소스 루트**: `~/PycharmProjects/pupil/pupil_src`
- **nnUNet 모델 및 가중치**: `~/PycharmProjects/nnUNet/nnUNet_results/`
  - `TemporalUNet` (메인): `TemporalUNet_v1/checkpoint_best.pth`
  - `nnUNet 2D` (Vanilla): `Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/fold_0/checkpoint_best.pth`
- **nnUNet 모듈 코드**: `~/PycharmProjects/nnUNet` 및 `~/PycharmProjects/nnUNet_legacy`

---

## 3. 프로그램 실행 방법 (Execution Guide)

### 3.1 실제 AR 글래스 착용 및 수신 시 (Pupil Capture GUI)

AR 글래스를 시연 컴퓨터 USB 포트에 연결한 후 다음 명령어로 실시간 GUI를 실행합니다:

```bash
cd ~/PycharmProjects/pupil/pupil_src
python main.py
```

- 실행 시 **Pupil Capture 메인 창**과 **안구 카메라 패널(Eye 0 / Eye 1)**이 표시됩니다.

### 3.2 하드웨어 미연결 시 사전 검증 (오프라인 더미 테스트)

AR 글래스 없이 디텍터 연산, 모델 로딩, 200 FPS 추론 지연시간을 사전 검증하려면 하네스 스크립트를 실행합니다:

```bash
cd ~/PycharmProjects/pupil
python tests/test_dummy_harness.py
```

---

## 4. UI 인터페이스 조작 및 설정 가이드 (UI User Manual)

Pupil Capture GUI 좌측/상단 메뉴를 통한 디텍터 설정 방법입니다.

### 4.1 디텍터 선택 및 Active Model 스위칭
1. Eye 창(안구 카메라 창)의 메뉴 아이콘 클릭
2. **`Pupil Detector 2D`** 설정 메뉴 이동
3. **`Active Model`** 드롭다운 항목 선택:
   - **`TemporalUNet` (기본값 / 추천)**: 시계열 ConvLSTM 디코더 기반. 프레임 간 연속성 및 노이즈 억제 우수 (~200 FPS).
   - **`nnUNet 2D`**: 정적 데이터 바닐라 nnUNet 2D 모델.
   - **`RITnet`**: DenseNet2D 기반 레거시 베이스라인.
   - **`2D C++`**: Pupil Labs 기존 C++ 기하학 디텍터.

### 4.2 Eye 0 (왼쪽 안구) 방향 보정
Pupil Core 스마트 글래스의 광학 거울 반사 구조상 Eye 0 카메라는 180도 뒤집힌 상태로 수신됩니다:
- **`Flip Vertically (Eye 0)`**: 체크박스를 켜면 모델 추론 전 상하 반전 후 바르게 세워 추론하며, 결과를 원래 좌표로 원복합니다.
- **`Flip Horizontally (Eye 0)`**: 좌우 반전 보정이 필요한 경우 체크합니다.

### 4.3 반눈(Half-blink) 및 이동평균(EMA) 필터링
- **반눈 찌그러짐 자동 차단**: 눈꺼풀에 동공이 눌려 종횡비 `aspect_ratio < 0.65` 이하로 찌그러진 경우 자동으로 빈 데이터를 반환하여 오류 유입 차단.
- **시계열 이동평균(EMA)**: `alpha=0.4` 수준의 지수이동평균을 적용하여 동공 중심점 미세 떨림 보정.

---

## 5. 입력 규격 및 자동 변환 파이프라인 (Data Pipeline)

| 항목 | Pupil Core 카메라 (실제 입력) | OpenEDS 학습 모델 (내부 연산) | 자동 처리 방식 |
|:---|:---|:---|:---|
| **해상도** | $192 \times 192$ (정사각형) | $640 \times 400$ (직사각형) | **Letterboxing**: 비율 유지 $400\times400$ 확대 $\rightarrow$ 좌우 120px 패딩 후 추론 $\rightarrow$ 120px 슬라이싱 복원 |
| **밝기 정규화** | 동적 IR 조명 변화 | Z-Score Normalization | **Dynamic Normalization**: 프레임별 `(frame - mean) / std` 적용 |

---

## 6. 트러블슈팅 및 문제 해결 (Troubleshooting)

1. **`UnpicklingError` 발생 시**:
   - `main.py` 최상단에 `weights_only=False` 몽키패치가 적용되어 있는지 확인.
2. **`ModuleNotFoundError: No module named 'nnunetv2'`**:
   - `conda activate pupil-umamba` 가상환경이 켜져 있는지 확인.
3. **카메라 입력이 찌그러지거나 동공이 튀는 현상**:
   - UI에서 `Flip Vertically`가 올바르게 켜져 있는지, `Active Model`이 `TemporalUNet`으로 설정되어 있는지 확인.

---

## 7. 🎯 캘리브레이션 및 검증(Test Validation) 타겟 좌표 설정 매뉴얼

사용자 시선 교정(Calibration) 및 검증(Validation/Accuracy Test) 시 표시되는 타겟 포인트 좌표는 코드 상에 아래 위치에 정의되어 있습니다.

### 7.1 화면 마커 캘리브레이션 및 검증 타겟 좌표 (Screen Marker)
- **파일 경로**: [pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py#L71-L82)
- **메서드 위치**: `ScreenMarkerChoreographyPlugin.get_list_of_markers_to_show(mode)` (Lines 71~82)

```python
@staticmethod
def get_list_of_markers_to_show(mode: ChoreographyMode) -> list:
    if ChoreographyMode.CALIBRATION == mode:
        # 캘리브레이션 3x3 Grid (1..9 순서: 좌상단 -> 우하단)
        return [
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),  # 1, 2, 3 (상단 행)
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),  # 4, 5, 6 (중단 행)
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),  # 7, 8, 9 (하단 행)
        ]
    if ChoreographyMode.VALIDATION == mode:
        # 검증(Validation / Accuracy Test) 타겟: (0.5, 1.0) 4회 연속 표시
        return [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]
```
- **수정 가이드**:
  - 화면 정중앙 및 4개 모서리(5-point) 외에 9-point 패턴이나 커스텀 좌표로 수정하려면 해당 리스트 `[(x1, y1), (x2, y2), ...]`의 튜플 항목을 직접 편집합니다.

### 7.2 단일 마커 캘리브레이션 고정 좌표 (Single Marker)
- **파일 경로**: [pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py#L100)
- **상수 위치**: `SingleMarkerChoreographyPlugin._FIXED_MARKER_POSITION` (Line 100)
```python
_FIXED_MARKER_POSITION = (0.5, 0.5)  # 화면 중앙 고정 마커 좌표
```

### 7.3 오프라인 테스트 하네스 안구 타겟 좌표 (Synthetic Dummy Frame)
- **파일 경로**: [tests/test_dummy_harness.py](file:///home/byeongjun/PycharmProjects/pupil/tests/test_dummy_harness.py#L32-L35)
- **함수 위치**: `make_synthetic_eye(height, width)` (Lines 32~35)
```python
cx, cy = width // 2, height // 2  # 가상 생성 동공 타겟 좌표
```

### 7.4 Accuracy Visualizer Outlier Threshold 기본값 수정 매뉴얼
Accuracy Visualizer(시선 측정 정확도 시각화) 플러그인이 실행될 때 적용되는 이상치 각도 차단 기준값(Outlier Threshold [degrees])의 기본값 설정 방법입니다.

- **파일 경로**: [pupil_src/shared_modules/accuracy_visualizer.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py#L178-L185)
- **클래스 및 함수 위치**: `Accuracy_Visualizer.__init__` (Line 181)

```python
class Accuracy_Visualizer(Plugin):
    def __init__(
        self,
        g_pool,
        outlier_threshold=1.3,  # Outlier Threshold 디폴트값 (기본 1.3도)
        vis_mapping_error=True,
        vis_calibration_area=True,
    ):
```

- **수정 가이드**:
  - 테스트(Validation) 실행 시 자동으로 초기화되는 Outlier Threshold 기본값을 변경하려면 `outlier_threshold=1.3` 기본 매개변수의 숫자값을 원하는 값(예: `1.0`, `2.0` 등)으로 직접 변경하시면 됩니다.

### 7.5 Sample Duration (샘플 수집 프레임 수) 기본값 수정 매뉴얼
각 캘리브레이션/검증 마커 지점에서 멈춰서 안구 데이터를 샘플링하는 duration(프레임 수 단위)의 기본값 설정 방법입니다.

- **파일 경로 1**: [pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py#L85-L95)
- **파일 경로 2**: [pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py#L102-L110)
- **클래스 및 함수 위치**: `ScreenMarkerChoreographyPlugin.__init__` / `SingleMarkerChoreographyPlugin.__init__`

```python
    def __init__(
        self,
        g_pool,
        fullscreen=True,
        marker_scale=1.0,
        sample_duration=60,  # Sample Duration 디폴트값 (기본 60프레임)
        monitor_name=None,
        **kwargs,
    ):
```

- **수정 가이드**:
  - 마커 1개당 샘플링할 데이터 프레임 수 기본값을 변경하려면 `sample_duration=60` 인자값을 원하는 수치(예: `40`, `80` 등)로 변경하시면 됩니다. (UI 내 Slider를 통해서도 10~100 범위 조작 가능)

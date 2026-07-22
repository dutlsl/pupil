# Pupil Labs 2D Detector Plugin - Test Harness Guide

본 문서는 AR 글래스 및 하드웨어(USB 입력) 연결 없이, **더미 합성 입력 프레임을 활용해 딥러닝 2D 동공 디텍터 플러그인(`Detector2DPlugin`)의 동작 무결성을 오프라인에서 점검하기 위한 테스트 하네스 명세서**입니다.

개발자 및 AI Agent는 새로운 모델을 통합하거나 리팩토링 후 시연 장소로 이동하기 전에 이 가이드와 하네스 스크립트를 사용하여 디텍터 무결성을 사전 점검할 수 있습니다.

---

## 1. 개요 및 목적

- **목적**: AR 글래스가 없는 환경에서 동공 세그멘테이션 모델(TemporalUNet, nnUNet 2D, RITnet, 2D C++)의 로딩, GPU 추론, 데이터 변환, 시계열 필터링 및 Pupil Labs `datum` 규격 반환 무결성을 사전 검증.
- **핵심 목표**:
  - 모델 체크포인트 로딩 및 CUDA 메모리 할당 검증.
  - 전처리(Z-score 정규화 및 64픽셀 패딩) 및 후처리(Gaussian Blur, Contour extraction, fitEllipse) 흐름 검증.
  - 반눈(Half-blink) 찌그러짐 차단 필터(`aspect_ratio < 0.65`) 및 시계열 EMA 이동평균/점프 차단 연산 검증.
  - 실시간 추론 지연시간(Latency) 측정 (~200 FPS 목표).

---

## 2. 하네스 모킹 구조 (Mock Architecture)

하네스 스크립트(`tests/test_dummy_harness.py`)는 Pupil Capture의 필수 런타임 객체를 다음과 같이 경량화하여 모킹합니다:

```python
class DummyRoi:
    bounds = (0, 0, 640, 400)

class DummyGPool:
    eye_id = 0
    roi = DummyRoi()
    display_mode = "video"

class DummyFrame:
    def __init__(self, img_np):
        self.gray = img_np
        self.height, self.width = img_np.shape[:2]
        self.timestamp = time.time()
```

### 합성 프레임 (Synthetic Frame) 생성 기준
- **해상도**: $400 \times 640$ (Grayscale)
- **배경 (공막/피부)**: 밝기값 ~180
- **합성 동공**: 중심점 $(320, 200)$, 장축 40px, 단축 35px, 기울기 $15^\circ$의 어두운 타원(밝기 20)
- **노이즈**: $[-10, 10]$ 범위의 가우시안/무작위 노이즈 추가로 실제 카메라 감도 모사

---

## 3. 테스트 하네스 실행 방법

### 3.1 Conda 환경 설정
`pupil-umamba` (Python 3.10, PyTorch CUDA 12, NumPy 1.26.4) 환경 활성화:

```bash
conda activate pupil-umamba
```

### 3.2 테스트 하네스 실행
프로젝트 루트 디렉토리(`/home/byeongjun/PycharmProjects/pupil`)에서 아래 명령어 실행:

```bash
python tests/test_dummy_harness.py
```

### 3.3 정상 검증 출력 예시

```
======================================================================
🤖 PUPIL LABS DETECTOR 2D PLUGIN - DUMMY TEST HARNESS
======================================================================
[1/3] Instantiating Detector2DPlugin & Loading PyTorch Models...
TemporalUNet created from .../fold_0/checkpoint_best.pth
  Encoder params (frozen): 14,158,944
  Decoder params (trainable): 52,351,256
✅ Plugin Instantiated Successfully.

[2/3] Executing Multi-Model Streaming Inference Tests...

▶ Model Mode: TemporalUNet
   Frame 1: Latency = 364.92 ms | Conf = 0.863 | NormPos = (0.5136, 0.4936) | Center = (328.7, 202.5)
   Frame 2: Latency =   5.24 ms | Conf = 0.863 | NormPos = (0.5136, 0.4936) | Center = (328.7, 202.6)
   Frame 3: Latency =   5.04 ms | Conf = 0.864 | NormPos = (0.5136, 0.4936) | Center = (328.7, 202.5)
   Frame 4: Latency =   4.97 ms | Conf = 0.864 | NormPos = (0.5136, 0.4936) | Center = (328.7, 202.5)
   Frame 5: Latency =   5.05 ms | Conf = 0.864 | NormPos = (0.5136, 0.4936) | Center = (328.7, 202.5)
   ✓ Average Streaming Latency (FPS): 5.08 ms (197.0 FPS)

...

[3/3] Integrity Verification Completed.
======================================================================
🎉 HARNESS RESULT: PASS (All Model Modes Verified)
======================================================================
```

---

## 4. Agent 및 개발자를 위한 체크리스트

코드를 수정하거나 새로운 모델을 추가한 경우, 다음 항목이 모두 통과하는지 확인해야 합니다:

- [ ] **Import & Path Check**: `sys.path`에 `~/PycharmProjects/nnUNet_legacy`, `~/PycharmProjects/nnUNet`, `~/PycharmProjects/nnUNet/agent`가 올바르게 주입되어 `temporal_unet` 및 `nnunetv2`를 에러 없이 임포트할 수 있는가?
- [ ] **NumPy Compatibility**: NumPy 2.x와 NumPy 1.x 간 unpickling 호환을 위한 `sys.modules['numpy._core']` 에일리어싱 및 `numpy<2` (1.26.4) 환경이 유지되는가?
- [ ] **Model Switching**: UI 드롭다운 선택값(`active_model`)에 따라 `"TemporalUNet"`, `"nnUNet 2D"`, `"RITnet"`, `"2D C++"`로 에러 없이 분기 및 동적 전환되는가?
- [ ] **Datum Structure Check**: 반환된 딕셔너리에 `norm_pos`, `diameter`, `confidence`, `timestamp`, `ellipse` (axes, angle, center) 키가 누락 없이 포함되어 있는가?
- [ ] **Half-Blink & EMA Check**: 찌그러짐(`aspect_ratio < 0.65`) 발생 시 신뢰도가 필터링되고, 급격한 이동(> 40px) 시 순간이동 차단 및 EMA 이동평균이 적절히 적용되는가?

---

## 5. 트러블슈팅 & 복구 팁

1. **`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`**:
   - Python 3.9 환경에서 3.10 문법(`str | None`)이 포함된 모듈을 부를 때 발생합니다. 실행 가상환경을 `pupil-umamba` (Python 3.10)로 지정했는지 확인하세요.
2. **`ImportError: numpy.core.multiarray failed to import`**:
   - Cython 모듈(`pyglui`)이 NumPy 1.x로 컴파일되어 NumPy 2.x와 충돌할 때 발생합니다. `pip install "numpy<2"`로 NumPy 1.26.4 버전으로 맞추어 해결합니다.
3. **`KeyError: 'conv_kernel_sizes'`**:
   - `nnunetv2` 버전 간 `plans.json` 스키마 차이로 발생합니다. `temporal_unet.py` 내 dynamic signature inspection 코드가 정상 적용되어 있는지 점검하세요.

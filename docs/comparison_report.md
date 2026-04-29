# 비교 보고서: 로컬 프로젝트 vs. Upstream Pupil Labs

날짜: 2026-02-13
Upstream 저장소: [pupil-labs/pupil](https://github.com/pupil-labs/pupil)
로컬 프로젝트: 커스텀 포크 (PyTorch & DVS 통합)

## 1. 요약 표

| 파일 / 모듈 | 기존 기능 (Original) | 수정된 기능 (Modified) | 영향도 (Impact) |
| :--- | :--- | :--- | :--- |
| `detector_2d_plugin.py` | C++ 기반 기하학적 2D 동공 검출 (윤곽선, 에지 피팅) 래퍼. | PyTorch 기반 딥러닝 모델(`DenseNet`)로 교체/증강됨 . |  높음: 핵심 검출 알고리즘이 컴퓨터 비전(CV)에서 딥러닝(DL)으로 변경됨. GPU/CUDA 필요. |
| `dvs_detector_plugin.py` | *존재하지 않음* (표준 Pupil은 프레임 기반 카메라만 지원). | 신규 플러그인 : DVS (Dynamic Vision Sensors) 지원 구현. |  높음: `dv_processing` 및 `tonic`을 사용하여 뉴로모픽 이벤트 기반 시선 추적 활성화. |
| `requirements.txt` | 표준 과학 계산 스택 (`numpy`, `opencv`, `pyglui`, `zeromq`). | 의존성 누락 : 코드 실행에 `torch`, `torchvision`, `tonic`, `dv_processing`이 필요하지만 목록에 없음. |  치명적: `pip install -r requirements.txt` 실행 시 환경이 깨짐. |
| `main.py` | 표준 실행 진입점(entry point). | 환경 변수 추가 (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`). | 낮음: 멀티 코어 프로세싱 최적화 (주로 PyTorch용). |
| `eye.py` | eye 프로세스를 관리하고 기본 플러그인을 로드. | 커스텀 검출기 초기화를 수용하도록 수정됨. | 보통: eye 프로세스의 시작 시퀀스 변경. |

## 2. 코드 분석 및 핵심 로직 변경점

### A. 2D 동공 검출 (`pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py`)

기존 Upstream:
기존 플러그인은 C++ 객체인 `Detector2D`를 초기화합니다. UI를 통해 노출된 파라미터 튜닝(임계값, 최소/최대 반경)에 의존합니다.

수정 사항:
`__init__` 메서드를 오버라이드하여 PyTorch 모델을 로드하도록 변경했습니다. 이는 학습된 모델로 C++ 검출기를 우회하거나 증강시키고 있음을 시사합니다.

```python
# 수정된 코드
def __init__(self, g_pool=None, properties=None, detector_2d: Detector2D = None):
    super().__init__(g_pool=g_pool)
    self.detector_2d = detector_2d or Detector2D(properties or {})
    
    # 커스텀 수정: 딥러닝 초기화
    model_name = "densenet"
    model_path = "./best_model.pkl"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = torch.device(device_str)
    # ... 모델 로드 로직 ...
```

### B. DVS (이벤트 카메라) 지원 (`pupil_src/shared_modules/pupil_detector_plugins/dvs_detector_plugin.py`)

기존 Upstream:
이벤트 카메라(Event Cameras)를 기본적으로 지원하지 않습니다.

수정 사항:
완전히 새로운 플러그인 클래스인 `DVSDetectorPlugin`을 생성했습니다. 이는 특정 하드웨어 SDK(`dv_processing`) 및 뉴로모픽 학습 라이브러리(`tonic`)를 통합합니다.

```python
# 새 파일
class DVSDetectorPlugin(PupilDetectorPlugin):
    def __init__(self, g_pool, config):
        # ...
        # 하드웨어 통합
        self.capture = CameraCapture() 
        self.slicer  = EventStreamSlicer()
        
        # 이벤트 처리를 위한 딥러닝
        self.net = Model(args).cuda().eval()
```

## 3. 의존성 및 환경 분석

발견된 치명적인 격차:
코드에 `requirements.txt`에 반영되지 않은 중요한 새 의존성들이 추가되었습니다.

*   코드에서 사용됨: `torch`, `torchvision`, `tonic`, `dv_processing`, `PIL` (Pillow).
*   `requirements.txt`에 존재: 위 항목 중 아무것도 없음.

권장 사항:
재현성을 보장하기 위해 `requirements.txt`를 업데이트하거나 `requirements-custom.txt`를 새로 생성해야 합니다.

## 4. 아키텍처 변경 사항

*   하이브리드 아키텍처 : 순수 CPU 기반의 가벼운 C++ 파이프라인에서  GPU 집약적인 Python 기반 딥러닝 파이프라인으로 전환하고 있습니다. 이로 인해 하드웨어 요구 사항이 크게 증가합니다(CUDA 지원 GPU 권장).
*   플러그인 시스템 활용: Pupil Labs의 플러그인 시스템(`PupilDetectorPlugin`)을 올바르게 활용하여 기능을 확장했으며, 이는 아키텍처 관점에서 "정상적인 경로(happy path)"입니다. 하지만 기본 `2d` 검출기의 내부를 교체하는 것은 병렬 검출기를 등록하는 것에 비해 침습적(invasive)입니다.
*   스레딩: `main.py`에 `OMP_NUM_THREADS=4`를 추가한 것은 병렬 처리를 수동으로 튜닝했음을 나타냅니다. 이는 PyTorch나 NumPy가 CPU 코어를 독점하여 실시간 UI/Capture 스레드가 굶주리는(starving) 현상을 방지하기 위함일 가능성이 높습니다.

## 5. 비교 요약

이 로컬 프로젝트는 차세대 시선 추적 연구  를 위해 설계된 Pupil Capture의 특화된 포크(Fork) 입니다.

1.  AI 우선 접근 (AI-First Approach): 동공 검출을 위해 기존의 컴퓨터 비전을 딥러닝(DenseNet)으로 대체합니다.
2.  뉴로모픽 하드웨어: 표준 상용 버전에는 없는 기능인 DVS 카메라 지원을 도입했습니다.
3.  프로토타입 상태: `requirements.txt`의 불일치와 "테스트" 스크립트(`pupil_detectors_test.py`, `detector_2d_plugin_cpu.py`)의 존재는 이 프로젝트가 완성된 배포판이라기보다는 활발하게 연구 중인 프로토타입임을 시사합니다.

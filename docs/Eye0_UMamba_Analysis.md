# Eye 0에서 U-Mamba 성능이 무너지는 원인 분석

## 결론 (요약)

Eye 0 성능 저하의 **핵심 원인은 3가지가 복합적으로 작용**합니다:

1. **종횡비 파괴** (192×192 → 640×400): 가장 큰 원인
2. **정규화 누락** (ZScoreNormalization 미적용): 두 번째 원인
3. **Mamba 아키텍처의 방향 민감성** + Eye 0 상하 반전: 세 번째 원인

RITnet(DenseNet 기반)은 완전 합성곱(FC)이라 입력 크기에 유연하고 방향에 덜 민감하지만, U-Mamba는 이 세 가지 모두에 취약합니다.

---

## 원인 1: 종횡비 파괴 (Critical)

### 사실 관계
| 항목 | 값 |
|---|---|
| Pupil Core 아이캠 해상도 | **192 × 192** (정사각형) |
| OpenEDS 학습 데이터 해상도 | **640 × 400** (가로로 긴 직사각형) |
| nnUNet patch_size | **640 × 384** |
| 현재 `_detect_umamba` 코드 | `cv2.resize(gray, (640, 400))` |

### 문제
192×192 정사각형 이미지를 640×400으로 리사이즈하면:
- **가로 3.33배 확대**, 세로 2.08배 확대
- 눈동자가 **가로로 60% 더 늘어난** 타원 형태로 왜곡됨
- U-Mamba가 학습한 동공 형태와 완전히 다른 입력이 들어감

### 왜 RITnet은 괜찮은가?
`ComparisonVisualizer._run_ritnet()`을 보면 **리사이즈를 전혀 하지 않습니다**. 192×192를 그대로 받습니다.
RITnet(DenseNet)은 Fully Convolutional이라 어떤 해상도든 그대로 처리합니다.

### 해결 방안
정사각형 비율을 유지하면서 640×400 캔버스에 맞추는 **letterboxing(패딩)** 방식을 써야 합니다:
```
192×192 → 400×400으로 비율 유지 리사이즈 → 좌우 120px씩 패딩 → 640×400
```

---

## 원인 2: 정규화 누락 (High)

### 사실 관계
nnUNet plans.json에 명시된 정규화 방식:
```json
"normalization_schemes": ["ZScoreNormalization"]
```

dataset_fingerprint.json의 통계:
```json
{
    "mean": 66.78,
    "std": 24.67,
    "median": 63.0,
    "min": 13.0,
    "max": 255.0
}
```

### 문제
현재 `_detect_umamba` 코드:
```python
input_npy = gray_resized[np.newaxis, np.newaxis].astype(np.float32)
```
**raw 픽셀값(0~255)을 그대로 넣고 있습니다.**

nnUNet의 `predict_single_npy_array`가 내부적으로 정규화를 수행할 수도 있지만, 이것은 **학습 시 사용된 데이터셋의 통계**에 기반합니다. 문제는 Pupil Core의 192×192 IR 카메라 이미지와 OpenEDS VR HMD 카메라 이미지의 **밝기 분포가 완전히 다르다**는 것입니다.

> [!IMPORTANT]
> nnUNetPredictor는 내부적으로 ZScore 정규화를 자동 적용합니다. 그러나 이 정규화는 **OpenEDS 학습 데이터의 mean/std**를 사용합니다. Pupil Core 카메라의 IR 이미지는 밝기 분포가 다르므로, 정규화 후에도 학습 분포와 불일치합니다.

### 해결 방안
RITnet의 전처리처럼 **Gamma Correction + CLAHE**를 U-Mamba 전처리에도 적용하면 도메인 간 밝기 차이를 줄일 수 있습니다.

---

## 원인 3: Mamba의 방향 민감성 + Eye 0 반전 (Medium)

### 사실 관계
- Pupil Core의 Eye 0(오른쪽 눈) 카메라 센서가 물리적으로 뒤집혀 장착됨 → 이미지가 상하 반전
- Pupil Capture 소프트웨어가 UI 표시용으로 자동 보정하지만, **`frame.gray`에는 원본(뒤집힌) 이미지가 들어옴**
- OpenEDS 2020에서는 오른쪽 눈을 **좌우 반전**시켜 일관성을 맞춤 (상하 아님)

### Mamba vs CNN의 방향 민감성
| 아키텍처 | 방향 민감성 | 이유 |
|---|---|---|
| **RITnet (DenseNet/CNN)** | 낮음 | 합성곱(Convolution)은 위치 불변(translation invariant). 눈의 상하가 뒤집혀도 필터가 동일하게 반응 |
| **U-Mamba (SSM)** | 높음 | State Space Model은 이미지를 **1D 시퀀스로 래스터 스캔**하여 처리. 스캔 방향이 바뀌면 시퀀스 패턴 자체가 변함 |

U-Mamba의 Mamba 블록은 이미지를 좌→우, 위→아래 순서로 직렬화합니다. 뒤집힌 이미지에서는:
- "눈썹→눈꺼풀→홍채→동공" 순서가 "동공→홍채→눈꺼풀→눈썹"으로 역전
- 학습된 시퀀스 패턴과 불일치 발생

### 왜 Flip 토글이 효과가 없었나?
뒤집기 자체는 작동하지만, **원인 1(종횡비 파괴)과 원인 2(정규화 누락)가 너무 치명적**이라 뒤집기만으로는 성능 개선이 눈에 띄지 않습니다.

---

## 양 눈 모두 성능이 떨어지는 이유

**Eye 1(뒤집히지 않은 눈)에서도 U-Mamba가 RITnet보다 못하는 이유:**

이것은 원인 1 + 원인 2의 결과입니다:
- 192×192 → 640×400 종횡비 파괴는 양 눈 모두 동일하게 적용
- 도메인 갭(VR HMD 카메라 vs IR 아이캠)은 양 눈 모두 동일

**Eye 0이 특히 더 나쁜 이유:**
- 위의 공통 문제에 + 원인 3(상하 반전)이 추가로 겹침

---

## 수정 우선순위

### 1순위: Letterbox 리사이즈 (종횡비 보존)
```python
# 현재 (잘못됨)
gray_resized = cv2.resize(gray, (640, 400))

# 수정 (비율 유지 + 패딩)
scale = min(640 / orig_w, 400 / orig_h)
new_w, new_h = int(orig_w * scale), int(orig_h * scale)
resized = cv2.resize(gray, (new_w, new_h))
canvas = np.zeros((400, 640), dtype=np.uint8)
y_off = (400 - new_h) // 2
x_off = (640 - new_w) // 2
canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
```

### 2순위: 도메인 적응 전처리
```python
# Gamma + CLAHE (RITnet과 동일)
table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
gray = cv2.LUT(gray, table.astype(np.uint8))
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
gray = clahe.apply(gray)
```

### 3순위: Flip은 1+2 수정 후 재평가
종횡비와 정규화를 고친 후, Eye 0에서 flip을 켜고/끄며 비교해야 진짜 효과를 알 수 있습니다.

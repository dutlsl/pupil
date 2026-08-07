# Server 3 TDTracker benchmark

측정 시각: 2026-07-29 12:49 (Asia/Seoul)

## 결과

모든 시간은 각 케이스 2,000회의 wall-clock latency 통계다. 매 반복에서
TDTracker 실행 직전에 CUDA를 동기화하고, 실행 후 다시 동기화한 뒤 타이머를
종료했다. Hz는 `1000 / 평균 ms`로 계산했다.

### TDTracker-only

| Case | 방식 | 평균 (ms) | p50 (ms) | p95 (ms) | Hz | Eager 대비 개선율 | 최고 |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | Eager | 0.7768 | 0.7423 | 1.0006 | 1,287.37 | 0.00% | |
| 2 | `torch.compile` | 0.4756 | 0.3903 | 0.7407 | 2,102.60 | 38.77% | |
| 3 | Eager + 외부 CUDA Graph | 0.4555 | 0.4003 | 0.6835 | 2,195.25 | 41.36% | |
| 4 | `torch.compile` + 외부 CUDA Graph | **0.3549** | **0.3128** | **0.5565** | **2,817.69** | **54.31%** | **최고** |

### RGB 실행 직후 TDTracker

RGB와 TDTracker는 직렬로 실행했다. 각 반복은 `RGB 실행 → synchronize →
타이머 시작 → TDTracker 실행 → synchronize → 타이머 종료` 순서이며, 표의
시간에 RGB 실행시간은 포함되지 않는다.

| Case | 방식 | 평균 (ms) | p50 (ms) | p95 (ms) | Hz | Eager 대비 개선율 | 최고 |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 5 | Eager RGB → Eager TD | 0.8400 | 0.7883 | 0.9997 | 1,190.51 | 0.00% | |
| 6 | Compile RGB → Compile TD | 0.5137 | 0.4962 | 0.8193 | 1,946.56 | 38.84% | |
| 7 | 외부 Graph RGB → 외부 Graph TD | 0.5845 | 0.5518 | 0.8989 | 1,711.01 | 30.42% | |
| 8 | Compile + 외부 Graph RGB → Compile + 외부 Graph TD | **0.4505** | **0.3303** | **0.8079** | **2,219.63** | **46.36%** | **최고** |

## 환경과 입력

- Host: `server3`
- CPU: 13th Gen Intel Core i9-13900K
- GPU: NVIDIA GeForce RTX 4090 1개, `cuda:0`
- Python: 3.9.21
- PyTorch: 2.6.0+cu124
- PyTorch CUDA runtime: 12.4
- cuDNN: 9.1.0 (`90100`)
- NVIDIA driver: 535.309.01
- `nvidia-smi`가 표시한 최대 지원 CUDA: 12.2
- TDTracker 입력: `(1, 8, 2, 60, 80)`, GPU float32, 실측 범위
  `0.0001373..14.9998093`, 고정 `data_ptr=128550564790272`
- RITnet 입력: `(1, 1, 400, 400)`, GPU float32, 실측 범위
  `0.00000095..0.99998844`, 고정 `data_ptr=128550565097472`
- random seed: `20260729`
- `cudnn.benchmark=True`, matmul TF32 활성, cuDNN TF32 활성
- compile/capture 준비 1,000회, 측정 전 warm-up 1,000회, 실측 2,000회

시작 시 `nvidia-smi`는 GPU 온도 39°C, P8, 29 W, 948 MiB 사용,
GPU-Util 27%를 표시했다. Xorg, GNOME Shell, PyCharm, Chrome, Firefox 등
그래픽 프로세스가 GPU를 사용 중이었고 별도의 CUDA compute 프로세스는
표시되지 않았다. 전체 원문은 raw JSON과 실행 로그에 보존했다.

## 실행 구성

- TDTracker 타이머 범위는 모델 forward와 GPU SimDR decode를 모두 포함한다.
  decode에는 좌표 argmax, avg-pool, softmax confidence 계산이 포함된다.
- RGB 실행은 RITnet forward와 GPU argmax를 포함한다.
- Compile-only는
  `torch.compile(runner, mode="reduce-overhead")`를 사용했으며 별도 외부
  CUDA Graph를 적용하지 않았다.
- Compile + 외부 Graph는
  `torch.compile(runner, options={"triton.cudagraphs": False})`로 compile
  내부 CUDA Graph를 끄고, 1,000회 준비 후 compiled runner 전체를 외부
  CUDA Graph로 capture했다.
- Eager + 외부 Graph는 eager runner 전체를 capture했다.
- RGB와 TDTracker 외부 CUDA Graph는 서로 별도의 `CUDAGraph` 객체다.
- 최초 생성한 두 GPU 입력 tensor를 전체 측정 동안 유지했으며, 모든
  반복과 모든 케이스에서 동일 tensor 및 memory address를 사용했다.

## TDTracker Dynamo 진단

- Dynamo graph count: **3**
- Graph break count: **2**
- 주요 break 원인: `step_unsupported`
- 위치: `dvs_models/TDTracker.py:73`, `x, _ = self.gru(x)`

즉, 현재 PyTorch/Dynamo 조합에서는 TDTracker의 GRU 호출 지점이 지원되지
않아 runner가 3개 graph로 나뉜다.

## 측정 범위 주의사항

- 실제 카메라와 Pupil runtime을 실행한 측정이 아니다.
- RGB와 TDTracker는 동시에 실행하지 않고 직렬 실행했다.
- RGB + Event 결과에 RGB 실행시간은 포함되지 않는다.
- H2D, D2H, FIFO, DAVIS, BinaRep, IPC, pye3d, UI는 제외했다.
- 환산 Hz는 실제 카메라 출력률이 아니라 TDTracker 구간의 이론적 직렬
  처리율이다.

원시 결과: `server3_tdtracker_results.json`

전체 실행 로그: `server3_tdtracker_run.log`

재현 스크립트: `server3_tdtracker_benchmark.py`

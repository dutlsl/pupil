# Server 3 real-camera runtime benchmark

측정 시각: 2026-07-29 13:00–13:04 (Asia/Seoul)

## 측정 구성

- Pupil Capture 3.6.0 실제 실행
- world: Pupil Cam1 ID2, 1280×720 @ 60 FPS
- eye0: Pupil Cam2 ID0, 400×400 @ 60 FPS
- eye1: Pupil Cam2 ID1, 400×400 @ 60 FPS
- event: INI DAViS FX3, 346×260, 1 ms `EventStreamSlicer`
- eye0에서 RITnet, DAVIS/BinaRep/TDTracker, pye3d를 실행
- eye1에서 RITnet과 pye3d를 실행
- world/UI/IPC/publisher와 `Accuracy_Visualizer`도 로드
- TDTracker device: `cuda:0`
- C++ BinaRep, 비동기 result slots 8개, publish target 1,000 Hz
- 각 모드의 종료 직전 정상 동작 구간 중 동일한 30개 1초 표본 사용
- 종료 신호가 들어간 표본과 모델 compile/capture 준비 구간은 제외

`TD sequence`는 metrics의 `graph_replay_hz` 필드다. 이름은 graph replay지만
eager/compile에서는 각각 해당 runner가 처리한 TD sequence 증가율을 뜻한다.
`fresh`는 eye0 parent가 실제로 새 TD 결과를 받은 비율이며, `publish`는
latest state를 IPC로 발행한 비율이다.

## 30초 steady-state 결과

| TDTracker runtime mode | TD sequence 평균 Hz | p50 Hz | p95 Hz | 유효 처리 간격 ms | Eager 대비 | Fresh 평균 Hz | Fresh/sequence | Publish 평균 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Eager | 693.65 | 718.64 | 780.73 | 1.442 | 기준 | 658.73 | 94.97% | 987.90 |
| `torch.compile(mode="reduce-overhead")` | **969.07** | **979.29** | **1,003.28** | **1.032** | **+39.71%** | **762.46** | 78.68% | **989.92** |
| Eager + 외부 CUDA Graph | 959.74 | 954.91 | 999.99 | 1.042 | +38.36% | 148.02 | 15.42% | 987.59 |

평균 TD sequence 기준으로는 compile mode가 가장 높았다. 외부 CUDA Graph도
약 960 Hz로 TD submit 자체는 빠르지만, 현재 비동기 result ring과
latest-only parent queue 조합에서는 새 결과가 parent까지 전달되는 비율이
크게 낮았다. Publish 약 988–990 Hz만 보면 이 병목이 가려진다.

## 누락과 worker 관찰

| mode | 30초 parent missing sequence | 초당 평균 missing |
|---|---:|---:|
| Eager | 1,048 | 34.93 |
| Compile | 6,202 | 206.73 |
| 외부 CUDA Graph | 24,364 | 812.13 |

Compile mode는 현재 남아 있는 상세 worker 로그의 같은 30초 구간에서 다음과
같았다.

| worker 지표 | 평균 Hz | p50 Hz | p95 Hz | 최소–최대 Hz |
|---|---:|---:|---:|---:|
| DAVIS slice | 1,002.33 | 1,000.10 | 1,062.54 | 885.20–1,077.30 |
| TD submit | 972.35 | 974.90 | 1,018.31 | 848.30–1,020.30 |
| TD infer 완료 | 972.35 | 974.90 | 1,018.31 | 848.30–1,020.30 |

- Compile worker drop: 0
- Compile state age: 평균 0.428 ms, p50 0.005 ms, p95 1.987 ms,
  최대 1.996 ms
- Eager는 동기식 forward/readback 때문에 event slice 처리율 자체가 약
  516–793 Hz 범위로 제한됐다.
- 외부 Graph는 event slice와 TD sequence를 약 1,000 Hz에 가깝게 유지했지만
  비동기 완료 slot/queue 전달 단계에서 결과가 대량으로 병합 또는 누락됐다.

## 해석

실카메라 전체 runtime 기준 현재 최고 선택은 compile mode다. 외부 CUDA
Graph의 synthetic TD 구간 자체는 가장 빨랐지만, 실제 runtime에서는 GPU
kernel 시간보다 비동기 result 회수와 parent queue 전달이 병목이다.
따라서 실제 fresh TD 갱신률을 높이려면 CUDA Graph를 더 빠르게 만드는 것보다
completion polling, result ring, parent queue drain 경로를 먼저 수정해야 한다.

이번 결과는 앞선 고정 GPU tensor microbenchmark와 범위가 다르다. 여기에는
실제 UVC 카메라, DAVIS event read, BinaRep, H2D/D2H, RITnet, IPC, pye3d,
world/UI 부하가 포함된다. 환산 처리율은 카메라의 물리적 FPS와 동일한 의미가
아니다.

Production runtime은 현재 `eager`, compile-only, eager + 외부 CUDA Graph
세 모드만 제공한다. 앞선 synthetic case의 compile + 외부 CUDA Graph 전체
capture는 production `create_tdtracker_runner`에 구현되어 있지 않아 이번
실카메라 표에는 포함하지 않았다.

## 원시 데이터

- `server3_camera_eager_metrics.jsonl`
- `server3_camera_compile_metrics.jsonl`
- `server3_camera_graph_metrics.jsonl`
- `capture_settings/capture.log` — 마지막 compile 실행 상세 로그

종료 시 SIGINT가 worker에도 전달돼 `KeyboardInterrupt` traceback이 기록됐지만,
측정 구간 이후의 종료 과정이며 모든 Pupil 자식 프로세스와 카메라 handle은
정상 해제된 것을 확인했다.

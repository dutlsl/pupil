# U-Mamba Integration & Troubleshooting Documentation

이 문서는 Pupil Labs 환경에 U-Mamba(nnUNet) 동공 분할 모델을 성공적으로 이식하기 위해 수행된 전체 변경 사항과 해결된 이슈들을 기록합니다.

## 1. PyTorch 2.6 모델 로딩 호환성 패치
> [!WARNING]
> **원인**: PyTorch 2.6 버전부터 `torch.load`의 `weights_only` 기본값이 `True`로 변경되어, nnUNet 및 기존 모델(pkl, pth) 로딩 시 `UnpicklingError` 발생.
> **해결**: 모든 Child Process가 파생되는 `pupil_src/main.py` 최상단에 `torch.load`를 전역적으로 몽키 패치(Monkey-patch)하여 `weights_only=False`를 강제하도록 수정했습니다.

## 2. 플러그인 아키텍처 의존성 분리
> [!IMPORTANT]
> **원인**: `detector_base_plugin.py`에서 무조건적으로 `detect_umamba` 메서드를 호출하도록 하드코딩되어 있어, 해당 메서드가 없는 `pye3d_plugin` 등의 서드파티/코어 플러그인 초기화 시 `AttributeError`가 발생하며 Eye 프로세스가 충돌했습니다.
> **해결**: `hasattr(self, 'detect_umamba')` 체크 로직을 추가하여, U-Mamba 지원 플러그인인 경우에만 해당 메서드를 호출하고 그 외의 플러그인은 기존 레거시 `detect` 파이프라인을 타도록 분기 처리했습니다.

## 3. U-Mamba 해상도(Resolution) 불일치 및 성능 개선
> [!TIP]
> **원인**: U-Mamba는 OpenEDS 데이터셋 기반인 `384x640` (또는 `400x640`) 해상도로 학습되었습니다. 하지만 카메라의 Raw Image(`192x192` 등)를 그대로 넣었을 때, nnUNet 내부 전처리기가 이를 384x640 타일에 맞추기 위해 극단적인 패딩(Padding)을 적용하여 분할 퀄리티가 현저히 떨어졌습니다.
> **해결**: `detector_2d_plugin.py`의 `detect_umamba` 메서드 진입 시 입력 프레임을 강제로 `400x640`으로 리사이즈(`cv2.INTER_LINEAR`)하여 추론한 뒤, 얻은 세그멘테이션 마스크를 다시 원본 카메라 해상도로 복원(`cv2.INTER_NEAREST`)하는 파이프라인을 구축했습니다. 이를 통해 원래 논문/실험 수준의 분할 정확도를 되찾았습니다.

## 4. 실시간 RITnet vs U-Mamba 비교 모듈 도입
추론 정확도를 정성적으로 비교하기 위해, 메인 추론 루프의 속도 저하를 분리하고 안전하게 끌 수 있는 독립 비교 모듈을 설계했습니다.

- **`comparison_visualizer.py` 생성**: 기존 RITnet 모델(`best_model.pkl`)을 백그라운드로 로드하고, 동일 프레임에 대해 RITnet과 U-Mamba의 세그멘테이션 마스크/타원 피팅 결과를 좌우로 나란히 시각화하는 독립 클래스를 구현했습니다.
- **UI 토글 스위치 추가**: `detector_2d_plugin.py`의 `init_ui` 메서드에 `ui.Switch("show_comparison", ...)`을 추가했습니다.
- Pupil Capture의 '2d c++' 설정 창에서 **Show RITnet vs U-Mamba** 버튼을 토글하여, 필요할 때만 비교 창을 띄우고 끌 수 있습니다. (꺼져 있을 때는 RITnet이 실행되지 않아 속도 저하가 없습니다).

## 5. 기타 수정 사항
- 깨진(Corrupted) `best_model.pkl` 파일을 복구하여 RITnet 및 Pye3D 플러그인의 폴백(Fallback) 동작을 정상화했습니다.
- 파일 수정 시 소유자(`byeongjun`)와 실행 환경(`iulab`) 간의 퍼포미션 이슈를 겪어, `pupil_src` 디렉토리에 전역 쓰기 권한(`chmod -R 777` 등)이 부여되어 이후 유지보수가 원활해졌습니다.

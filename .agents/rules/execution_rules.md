# Execution & Responsive Session Rules

## 1. No Lingering Background Tasks (Zero Message Queue Lag)
- **절대 비동기 백그라운드 프로세스를 방치하지 말 것**: `run_command` 실행 시 항상 충분한 동기 대기 시간을 두거나 즉시 완료되도록 처리할 것.
- **매 턴 종료 전 프로세스 정리 확인**: 백그라운드 태스크가 남아있으면 IDE가 세션을 잠그고 사용자 메시지를 대기열(Queued message)로 넘기므로, 절대로 백그라운드에 프로세스를 남겨두지 않는다.
- **즉시 응답 원칙**: 불필요한 반복 툴 호출을 지양하고 사용자 입력에 즉각 응답한다.

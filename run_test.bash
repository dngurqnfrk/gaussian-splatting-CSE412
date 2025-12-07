#!/bin/bash

# 중요: 스크립트 실행 중 에러가 발생하면 즉시 중단 (set -e)
# 컴파일(pip install)이 실패했는데 렌더링으로 넘어가는 것을 방지합니다.
set -e

echo "==========================================="
echo ">> [1/3] Reinstalling Rasterizer..."
echo "==========================================="

# 서브모듈 디렉토리로 이동
cd submodules/diff-gaussian-rasterization

# 기존 패키지 제거 (설치 안 되어 있을 경우 에러 무시를 위해 || true 추가)
pip uninstall diff-gaussian-rasterization -y || true

# 빌드 캐시 삭제 (-r 대신 -rf를 써서 폴더가 없어도 에러 안 나게 함)
rm -rf build

# 패키지 재설치
pip install .

# 다시 프로젝트 루트로 복귀
cd ../..
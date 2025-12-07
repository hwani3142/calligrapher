# TF 1.15.0 + Python 3 이미 들어있는 공식 이미지 (CPU용)
FROM tensorflow/tensorflow:1.15.0-py3

# 터미널 출력 버퍼링 끄기 (로그 바로바로 보기 위함)
ENV PYTHONUNBUFFERED=1

# 작업 디렉터리
WORKDIR /workspace

# 현재 프로젝트 전체를 컨테이너 안 /workspace 로 복사
# (calligrapher/, checkpoints/, data/, styles/ 전부 포함)
COPY . /workspace

# 필요한 파이썬 패키지 설치
# - numpy 1.19.5: TF1.15와 가장 안정적인 조합
# - tensorflow-probability 0.7.0: TF1.15 시대 버전
# - matplotlib, pillow: demo에서 그림 저장/표시용
RUN pip install --upgrade pip && \
    pip install \
        numpy==1.19.5 \
        tensorflow-probability==0.7.0 \
        matplotlib \
        pillow
RUN pip install -r requirements.txt

# (선택) 기본 실행 명령: 바로 demo 돌리고 싶을 때 사용
# 나중에 docker run 할 때 다른 명령을 주면 이건 무시됨.
CMD ["python", "calligrapher/demo.py"]


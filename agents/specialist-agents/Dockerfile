# 전문 에이전트 4종 공통 컨테이너 이미지(GEOAI_AGENT_ID 로 역할을 정한다). 빌드 문맥은 저장소 최상위다.
#   docker build -f agents/specialist-agents/Dockerfile -t <레지스트리>/geoai/specialist-agent:0.1.0 .
#
# 실행 중 인터넷 의존 금지 규칙: 모든 의존 패키지는 빌드 단계에서 해시 고정 잠금 파일로
# 설치하고, 실행 단계는 외부 접속 없이 동작한다. 사내 패키지 저장소는 PIP_INDEX_URL 로 지정한다.
# 사내 프록시 인증서가 필요하면 --secret id=pip_ca,src=<인증서 묶음> 으로 넘긴다(이미지에 남지 않는다).

ARG PYTHON_IMAGE=python:3.12.8-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY agents/specialist-agents/requirements.lock /src/requirements.lock
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --index-url "${PIP_INDEX_URL}" --require-hashes -r /src/requirements.lock
COPY libs/geoai-common /src/libs/geoai-common
COPY agents/specialist-agents /src/agents/specialist-agents
# 저장소 내부 패키지는 별도 빌드 환경에서 바퀴(wheel)로 만든 뒤 의존성 없이 설치한다.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
 && python -m venv /opt/build \
 && /opt/build/bin/pip install --index-url "${PIP_INDEX_URL}" "hatchling==1.27.0" \
 && /opt/build/bin/pip wheel --no-deps --no-build-isolation -w /wheels /src/libs/geoai-common /src/agents/specialist-agents \
 && /opt/venv/bin/pip install --no-deps --no-index /wheels/*.whl

FROM ${PYTHON_IMAGE}
RUN useradd --uid 10001 --create-home geoai
COPY --from=build /opt/venv /opt/venv
COPY ontology /app/ontology
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GEOAI_ONTOLOGY_DIR=/app/ontology \
    PORT=8080
WORKDIR /app
USER 10001
EXPOSE 8080
ENTRYPOINT ["specialist-agent"]

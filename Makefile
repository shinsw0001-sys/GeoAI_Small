# 자주 쓰는 명령. 가상 환경을 활성화한 상태에서 실행한다.
REGISTRY ?= geoai
VERSION  ?= 0.1.0

.PHONY: install dev test test-e2e test-integration lint schemas image manifests deps-up deps-down

PKGS = libs/geoai-common agents/studio-agent agents/specialist-agents console/ops-console tools/geoai-tools
LOCKS = agents/studio-agent/requirements.lock agents/specialist-agents/requirements.lock \
        console/ops-console/requirements.lock tools/geoai-tools/requirements.lock

install:
	pip install --require-hashes $(addprefix -r ,$(LOCKS))
	pip install --no-deps $(addprefix -e ,$(PKGS))
	pip install pytest==9.2.0 pytest-asyncio==1.4.1.post1 ruff==0.17.4

# 개발용 일체형 실행: 운영 콘솔 http://127.0.0.1:8000 (합성 표본 자동 생성)
dev:
	GEOAI_ONTOLOGY_DIR=ontology geoai-dev --port 8000 --data .geoai-data

# 건물 추출 종단 시험 (처리 도구까지 실제 실행, 약 1~2분)
test-e2e:
	pytest -q -m e2e tests/e2e

test:
	pytest -q -m "not e2e"

deps-up:
	docker compose -f deploy/compose/test-deps.yaml up -d --wait

deps-down:
	docker compose -f deploy/compose/test-deps.yaml down -v

test-integration: deps-up
	GEOAI_TEST_DATABASE_URL=postgresql://geoai:geoai@localhost:5432/geoai \
	GEOAI_TEST_NATS_URL=nats://localhost:4222 pytest -q -m integration

lint:
	ruff check . && ruff format --check .

schemas:
	python -m geoai_common.messages schemas

image:
	docker build -f agents/studio-agent/Dockerfile -t $(REGISTRY)/studio-agent:$(VERSION) .
	docker build -f agents/specialist-agents/Dockerfile -t $(REGISTRY)/specialist-agent:$(VERSION) .
	docker build -f console/ops-console/Dockerfile -t $(REGISTRY)/ops-console:$(VERSION) .
	docker build -f tools/geoai-tools/Dockerfile -t $(REGISTRY)/geoai-tools:$(VERSION) .

manifests:
	kustomize build deploy/k8s/overlays/dev

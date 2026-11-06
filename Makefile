# 자주 쓰는 명령. 가상 환경을 활성화한 상태에서 실행한다.
REGISTRY ?= geoai
VERSION  ?= 0.1.0

.PHONY: install test test-integration lint schemas image manifests deps-up deps-down

install:
	pip install --require-hashes -r agents/studio-agent/requirements.lock
	pip install --no-deps -e libs/geoai-common -e agents/studio-agent
	pip install pytest==9.2.0 pytest-asyncio==1.4.1.post1 httpx==0.28.1 ruff==0.17.4

test:
	pytest -q

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

manifests:
	kustomize build deploy/k8s/overlays/dev

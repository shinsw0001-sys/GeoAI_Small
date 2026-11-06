"""언어 모델 호출부 추상화.

에이전트는 이 프로토콜만 의존한다. 구현은 OpenAI 호환 채팅 완성 API 하나로 두어,
외부 상용 API와 자체 서버의 vLLM 추론 서버를 설정(LLM_BASE_URL, LLM_MODEL)만으로
바꿔 쓴다. 1단계 Studio Agent의 계획기는 온톨로지 규칙만 쓰므로 아직 호출하지 않는다.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


class LanguageModel(Protocol):
    async def complete(self, messages: list[ChatMessage], *, max_tokens: int = 1024) -> str: ...


class OpenAICompatibleModel:
    """OpenAI 호환 /v1/chat/completions 엔드포인트 (vLLM 포함)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("언어 모델 주소는 http(s) 여야 합니다.")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    async def complete(self, messages: list[ChatMessage], *, max_tokens: int = 1024) -> str:
        import asyncio

        body = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")  # noqa: S310

        def _send() -> str:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310 - 생성자에서 http(s)만 허용
                data = json.load(response)
            return data["choices"][0]["message"]["content"]

        return await asyncio.to_thread(_send)

import base64
import os
from typing import Optional

import httpx
from openai import OpenAI, AsyncOpenAI, APIConnectionError
from pydantic import BaseModel


class CategoryResponse(BaseModel):
    presence: int
    confidence: int
    description: str


class Middle:
    def __init__(self):
        self.base_url = os.environ["BASE_URL"]
        self.model = os.environ["MODEL"]

        # Optional generation controls (only sent if set)
        self.max_completion_tokens: Optional[int] = None
        self.temperature: Optional[float] = None
        self.seed: Optional[int] = None

        _mct = os.getenv("MAX_COMPLETION_TOKENS")
        if _mct is not None and _mct.strip() != "":
            try:
                self.max_completion_tokens = int(_mct)
            except ValueError:
                print(f"WARNING: Invalid MAX_COMPLETION_TOKENS={_mct!r}; ignoring.")

        _temp = os.getenv("TEMPERATURE")
        if _temp is not None and _temp.strip() != "":
            try:
                self.temperature = float(_temp)
            except ValueError:
                print(f"WARNING: Invalid TEMPERATURE={_temp!r}; ignoring.")

        _seed = os.getenv("SEED")
        if _seed is not None and _seed.strip() != "":
            try:
                self.seed = int(_seed)
            except ValueError:
                print(f"WARNING: Invalid SEED={_seed!r}; ignoring.")

        def _env_true(v: str | None) -> bool:
            return (v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

        self.disable_thinking: bool = _env_true(os.getenv("DISABLE_THINKING"))

        self.client = OpenAI(base_url=self.base_url)
        self.async_client = AsyncOpenAI(base_url=self.base_url)

        print(
            "Middle point has been initialized.\n"
            f"Your URL is {self.base_url}.\n"
            f"The chosen model is {self.model}"
        )

    def _common_chat_kwargs(self):
        kwargs = {}
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed

        if self.disable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return kwargs

    def test_connection(self):
        print(f"Attempting to connect to {self.base_url} and get a response from {self.model}...")
        try:
            kwargs = self._common_chat_kwargs()
            kwargs["timeout"] = 15

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": "Test. Respond with OK."}]
                }],
                **kwargs,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                print("Connection and LLM response successful!")
                return True
            raise Exception("LLM connection test returned an empty response.")
        except (APIConnectionError, httpx.ConnectError) as e:
            print(f"--- FATAL CONNECTION ERROR ---\nCould not connect to the server: {e}")
            raise
        except Exception as e:
            print(f"--- FATAL LLM RESPONSE ERROR ---\nServer connected, but the LLM failed to respond: {e}")
            raise

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    async def sendImageRequestAsync(self, image_bytes: bytes, prompt: str):
        """
        Returns (parsed: CategoryResponse, usage_dict or None).
        Raises on any failure (fail-hard).
        """
        b64 = self._encode_bytes(image_bytes)

        kwargs = self._common_chat_kwargs()
        kwargs["timeout"] = 90

        # Hard requirement: structured parsing available and successful
        try:
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }
                ],
                response_format=CategoryResponse,
                **kwargs,
            )
        except AttributeError as e:
            raise RuntimeError(
                "Your OpenAI SDK does not support chat.completions.parse(). "
                "Upgrade the openai package or implement a supported fallback."
            ) from e

        parsed = response.choices[0].message.parsed

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        return parsed, usage
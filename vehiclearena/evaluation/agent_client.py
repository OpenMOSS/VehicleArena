import logging
import time
import httpx
from openai import OpenAI
import os
from simulation.model_concurrency import model_io
from typing import List, Dict, Any, Optional

# Models that do not support custom temperature (must use default)
_NO_CUSTOM_TEMPERATURE_MODELS = {"gpt-5", "o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o4-mini"}
_REASONING_EFFORTS = {
    "max", "xhigh", "high", "medium", "low", "minimal", "none"}
logger = logging.getLogger(__name__)


def _is_non_retryable_api_error(exc: Exception) -> bool:
    """Return true for request failures another retry cannot repair.

    Authentication, permission/quota, invalid-request and missing-model
    responses are stable for the lifetime of one experiment run. Retrying
    those failures on every wake only burns requests and wall time.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code in {400, 401, 403, 404}


class AgentClientError(RuntimeError):
    """Raised when an LLM request fails after the configured retries."""

class AgentClient:
    """
    Client for interacting with LLM API using OpenAI-like interface.
    """

    def __init__(self,
                 api_base: str = "",
                 api_key: Optional[str] = "",
                 model: str = "",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 thinking_mode: str = "default",
                 reasoning_effort: Optional[str] = None,
                 chat_template_enable_thinking: Optional[bool] = None):
        """
        Initialize the AgentClient.

        Args:
            api_base: Base URL for the API
            api_key: API key (reads from environment if None)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in the response
        """
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = str(thinking_mode or "default").lower()
        if self.thinking_mode not in {"default", "enabled", "disabled"}:
            raise ValueError(
                "thinking_mode must be default, enabled, or disabled")
        self.reasoning_effort = (
            str(reasoning_effort).lower()
            if reasoning_effort is not None else None)
        if (self.reasoning_effort is not None
                and self.reasoning_effort not in _REASONING_EFFORTS):
            raise ValueError(
                "reasoning_effort must be one of: "
                + ", ".join(sorted(_REASONING_EFFORTS)))
        self.chat_template_enable_thinking = (
            bool(chat_template_enable_thinking)
            if chat_template_enable_thinking is not None else None)
        self.last_call_metadata: Dict[str, Any] = {}

        if not self.api_key:
            raise ValueError("API key must be provided either in constructor or as OPENAI_API_KEY environment variable")
        # Model endpoints are explicitly selected by the experiment.  Do not
        # let notebook-wide proxy variables silently redirect or block the
        # request; direct connectivity is required for the configured API.
        client_kwargs = {
            "api_key": self.api_key,
            "http_client": httpx.Client(trust_env=False),
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        self.client = OpenAI(**client_kwargs)

    def _effective_thinking_enabled(self) -> bool:
        """Resolve all supported thinking controls to one boolean value.

        Explicit experiment configuration takes precedence over the legacy
        ``-thinking`` model-name convention.  Keeping this decision in one
        place prevents a request from carrying contradictory Qwen flags such
        as ``enable_thinking=false`` and
        ``chat_template_kwargs.enable_thinking=true``.
        """
        chat_template_flag = getattr(
            self, "chat_template_enable_thinking", None)
        if chat_template_flag is not None:
            return bool(chat_template_flag)
        thinking_mode = getattr(self, "thinking_mode", "default")
        if thinking_mode != "default":
            return thinking_mode == "enabled"
        return "-thinking" in self.model.lower()
    

    @model_io
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a non-tool request with bounded retries and provenance."""
        started = time.perf_counter()
        is_qwen3 = "qwen3" in self.model.lower()
        legacy_thinking_model = (
            is_qwen3 and "-thinking" in self.model.lower())
        enable_thinking = self._effective_thinking_enabled()
        for attempt in range(1, 4):
            try:
                kwargs = {
                    "model": (self.model.removesuffix("-thinking")
                              if legacy_thinking_model else self.model),
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "timeout": 300,
                }
                model_base = self.model.lower().split("/")[-1]
                if model_base not in _NO_CUSTOM_TEMPERATURE_MODELS:
                    kwargs["temperature"] = self.temperature
                if is_qwen3:
                    kwargs["extra_body"] = {
                        "enable_thinking": enable_thinking}
                if self.thinking_mode != "default":
                    kwargs.setdefault("extra_body", {})["thinking"] = {
                        "type": self.thinking_mode}
                if getattr(self, "reasoning_effort", None) is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                if getattr(
                        self, "chat_template_enable_thinking", None
                ) is not None:
                    kwargs.setdefault("extra_body", {})[
                        "chat_template_kwargs"] = {
                            "enable_thinking":
                            self.chat_template_enable_thinking}
                if legacy_thinking_model:
                    kwargs["stream"] = True
                    stream = self.client.chat.completions.create(**kwargs)
                    chunks = []
                    usage = None
                    for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage = chunk.usage
                        if (chunk.choices
                                and chunk.choices[0].delta.content):
                            chunks.append(chunk.choices[0].delta.content)
                    response = "".join(chunks).strip()
                    finish_reason = None
                    response_id = None
                    fingerprint = None
                else:
                    chat_response = self.client.chat.completions.create(
                        **kwargs)
                    choice = chat_response.choices[0]
                    response = (choice.message.content or "").strip()
                    usage = getattr(chat_response, "usage", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                    response_id = getattr(chat_response, "id", None)
                    fingerprint = getattr(
                        chat_response, "system_fingerprint", None)
                prompt_tokens = int(
                    getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(
                    getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                self.last_call_metadata = {
                    "ok": True, "model": self.model,
                    "api_base": self.api_base, "attempts": attempt,
                    "temperature_requested": self.temperature,
                    "temperature_sent": kwargs.get("temperature"),
                    "thinking_mode": self.thinking_mode,
                    "reasoning_effort": getattr(
                        self, "reasoning_effort", None),
                    "chat_template_enable_thinking": getattr(
                        self, "chat_template_enable_thinking", None),
                    "max_tokens": self.max_tokens,
                    "finish_reason": finish_reason,
                    "response_id": response_id,
                    "system_fingerprint": fingerprint,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_s": round(time.perf_counter() - started, 6),
                }
                return (response, total_tokens, prompt_tokens,
                        completion_tokens)
            except Exception as exc:
                terminal = _is_non_retryable_api_error(exc)
                if attempt < 3 and not terminal:
                    logger.warning("chat retry %d/3: %s: %s", attempt,
                                   type(exc).__name__, exc)
                    continue
                self.last_call_metadata = {
                    "ok": False, "model": self.model,
                    "api_base": self.api_base, "attempts": attempt,
                    "temperature_requested": self.temperature,
                    "thinking_mode": self.thinking_mode,
                    "reasoning_effort": getattr(
                        self, "reasoning_effort", None),
                    "chat_template_enable_thinking": getattr(
                        self, "chat_template_enable_thinking", None),
                    "max_tokens": self.max_tokens,
                    "latency_s": round(time.perf_counter() - started, 6),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raise AgentClientError(
                    f"chat failed after {attempt} attempts: "
                    f"{type(exc).__name__}: {exc}") from exc

    def build_tool_chat_request(
        self, messages: List[Dict], tools: List[Dict],
    ) -> Dict[str, Any]:
        """Build the exact SDK kwargs used by ``chat_with_tools``.

        Keeping this in one public, side-effect-free method lets evaluators
        persist a replayable request before network I/O without duplicating
        provider option logic.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "timeout": 300,
        }
        model_base = self.model.lower().split("/")[-1]
        if model_base not in _NO_CUSTOM_TEMPERATURE_MODELS:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if "qwen3" in self.model.lower():
            kwargs["extra_body"] = {
                "enable_thinking": self._effective_thinking_enabled()}
        thinking_mode = getattr(self, "thinking_mode", "default")
        if thinking_mode != "default":
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": thinking_mode}
        reasoning_effort = getattr(self, "reasoning_effort", None)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        chat_template_enable_thinking = getattr(
            self, "chat_template_enable_thinking", None)
        if chat_template_enable_thinking is not None:
            kwargs.setdefault("extra_body", {})[
                "chat_template_kwargs"] = {
                    "enable_thinking": chat_template_enable_thinking}
        return kwargs

    @model_io
    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> tuple:
        """
        Send a chat completion request with function-calling tools.

        Unlike chat(), this method:
        - Passes a tools list to the API
        - Returns the full message object (with tool_calls) instead of just content string

        Args:
            messages: List of message dicts (role/content, may include tool role messages)
            tools: List of tool dicts in OpenAI format

        Returns:
            Tuple of (response_message, total_tokens, prompt_tokens, completion_tokens)
            where response_message is the full ChatCompletionMessage object
            (has .content, .tool_calls, .role attributes)
        """
        prompt_token_length = 0
        completion_token_length = 0
        total_token_length = 0
        retry_num = 3
        attempt = 0
        started = time.perf_counter()

        while True:
            attempt += 1
            try:
                kwargs = self.build_tool_chat_request(messages, tools)
                thinking_mode = getattr(self, "thinking_mode", "default")
                reasoning_effort = getattr(self, "reasoning_effort", None)
                chat_template_enable_thinking = getattr(
                    self, "chat_template_enable_thinking", None)

                chat_response = self.client.chat.completions.create(**kwargs)

                response_message = chat_response.choices[0].message
                usage = getattr(chat_response, "usage", None)
                total_token_length += int(
                    getattr(usage, "total_tokens", 0) or 0)
                prompt_token_length += int(
                    getattr(usage, "prompt_tokens", 0) or 0)
                completion_token_length += int(
                    getattr(usage, "completion_tokens", 0) or 0)
                choice = chat_response.choices[0]
                self.last_call_metadata = {
                    "ok": True,
                    "model": self.model,
                    "api_base": self.api_base,
                    "attempts": attempt,
                    "temperature_requested": self.temperature,
                    "temperature_sent": kwargs.get("temperature"),
                    "thinking_mode": thinking_mode,
                    "reasoning_effort": reasoning_effort,
                    "chat_template_enable_thinking": (
                        chat_template_enable_thinking),
                    "max_tokens": self.max_tokens,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "response_id": getattr(chat_response, "id", None),
                    "system_fingerprint": getattr(
                        chat_response, "system_fingerprint", None),
                    "prompt_tokens": prompt_token_length,
                    "completion_tokens": completion_token_length,
                    "total_tokens": total_token_length,
                    "latency_s": round(time.perf_counter() - started, 6),
                }

                break

            except Exception as e:
                retry_num -= 1
                if retry_num == 0 or _is_non_retryable_api_error(e):
                    self.last_call_metadata = {
                        "ok": False,
                        "model": self.model,
                        "api_base": self.api_base,
                        "attempts": attempt,
                        "temperature_requested": self.temperature,
                        "thinking_mode": getattr(
                            self, "thinking_mode", "default"),
                        "reasoning_effort": getattr(
                            self, "reasoning_effort", None),
                        "chat_template_enable_thinking": getattr(
                            self, "chat_template_enable_thinking", None),
                        "max_tokens": self.max_tokens,
                        "latency_s": round(
                            time.perf_counter() - started, 6),
                        "error": f"{type(e).__name__}: {e}",
                    }
                    raise AgentClientError(
                        f"tool chat failed after {attempt} attempts: "
                        f"{type(e).__name__}: {e}") from e
                logger.warning("tool chat retry %d/3: %s: %s",
                               attempt, type(e).__name__, e)

        return response_message, total_token_length, prompt_token_length, completion_token_length

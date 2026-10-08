import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import List, Optional

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.services import gemini_auth
from app.utils import utils

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# Claude Code CLI 默认使用编码 agent 的系统提示词，其中大量约束与文案写作
# 无关，会让脚本和关键词生成偏离要求，因此调用时整体替换掉。
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` 关闭全部内置工具，`--safe-mode` 关闭 CLAUDE.md、skills、hooks、
# plugins、MCP 等所有用户级定制，同时保持鉴权、模型选择和权限正常工作。
# 二者需要较新的 CLI；低版本会以 "unknown option" 退出，由调用处转成明确提示。
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# 这些环境变量会让 CLI 改用 API Key 或第三方供应商（Bedrock、Vertex、Foundry、
# Mantle、Gateway 等），从而绕过订阅登录并产生额外计费。逐个列举容易漏项，
# 而且 CLI 后续还会新增供应商，因此按前缀整类剔除：
#   ANTHROPIC_*           API Key、Auth Token、Base URL、各家供应商端点和 Profile
#   CLAUDE_CODE_USE_*     供应商开关
#   CLAUDE_CODE_SKIP_*_AUTH  跳过供应商鉴权的开关
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# 这两类变量不能剔除：
#   CLAUDE_CODE_OAUTH_TOKEN 是容器内唯一的订阅鉴权方式（不匹配上面的前缀）；
#   *_CONFIG_DIR 只是指出凭证存放位置，剔除后反而会让已登录的订阅失效。
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """判断某个环境变量是否会把 CLI 从订阅登录切换到别的鉴权方式。"""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    把配置里的超时值解析成正的有限秒数。

    TOML 既可能写成 `claude_code_timeout = 300`（int/float），也可能写成
    `"300"`（字符串），因此不能直接调用 `strip()`。nan / inf 会让
    `subprocess.run(timeout=...)` 永久阻塞，这里一并拒绝。
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool 是 int 的子类，但 True 秒显然不是用户想要的超时配置。
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    只有「未配置」时才回退到 Registry 默认值。

    之前用 `raw or default_value`，会把 0 和 false 这类合法取值也当成未配置
    替换掉：`claude_code_timeout = 0` 被静默改成 300，而 `"0"` 却报错。默认值
    只在 None 或空白字符串时生效，配置校验才能对所有写法保持一致。
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    构造只依赖订阅登录的子进程环境。

    返回 (环境变量字典, 被剔除的变量名列表)。剔除的是会切换鉴权方式或供应商
    的变量，`CLAUDE_CODE_OAUTH_TOKEN` 必须保留：容器内没有 keychain，CLI 只能
    靠它完成订阅鉴权。
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # 前面的 ``strip()`` 已经清理首尾空白。这里必须保留正文中的单换行和
    # 双换行：脚本生成依赖双换行区分段落，字幕处理也会按行读取用户文案。
    return content


def _sanitize_error_message(error: object) -> str:
    """
    清理返回给 WebUI/API 的错误信息，避免自定义 base_url 中的凭据泄露。

    一些 OpenAI-compatible SDK 会把请求 URL 原样拼进异常信息。如果用户为了
    代理网关配置了 `https://user:pass@example.com/v1`，直接返回 `str(e)`
    就会把密码暴露给页面、API 调用方或后续日志。这里仅处理错误文案，不改变
    实际请求地址，避免影响正常调用链路。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(prompt: str, app_config=None) -> str:
    try:
        # WebUI 在视频生成期间允许用户准备下一条文案。调用方可以传入提交瞬间
        # 的配置快照，确保模型请求重试期间不会因为后台任务结束并应用新配置，
        # 而切换到另一个 Provider、Base URL 或模型。
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama 的默认地址依赖当前是否运行在容器中，无法作为静态 Registry
        # 值保存；Registry 仍负责模型和必填规则，运行环境差异在这里解析。
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        vertex_gemini = adapter == "gemini" and gemini_auth.use_vertexai(runtime_app_config)
        if provider.requires_api_key and not api_key and not vertex_gemini:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                # List scripts and edit plans are long JSON documents, and
                # thinking models spend part of this budget before answering.
                max_output_tokens=32768,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            # Resolved before the request so a missing key or Vertex project is
            # reported as such, not as an invalid model response.
            client_kwargs = gemini_auth.client_kwargs(runtime_app_config, api_key)
            try:
                # 新版 google-genai 通过统一 Client 暴露模型服务。上下文管理器
                # 会在请求结束后关闭底层 HTTP 连接，避免频繁生成时积累连接资源。
                with genai.Client(
                    **client_kwargs,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare 当前推荐的 AI Gateway REST API 兼容 OpenAI SDK。
            # Account ID 用于构造统一端点，Gateway ID 通过请求头选择；这里
            # 不再调用 Workers AI 的 /ai/run/{model} 专用接口。
            client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
            # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
            # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
            # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "claude_code":
            # Claude 订阅（Pro / Max / Team）不签发 API Key，其凭证只能由
            # Claude Code 官方客户端自己使用。这里不直接请求 Anthropic API，
            # 而是以 headless 模式调用本机已登录的 claude CLI（`claude -p`），
            # 由 CLI 完成鉴权，脚本生成只消费它返回的文本。
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            command = [
                cli_path,
                "-p",
                prompt,
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # 关闭全部内置工具，保证只做文本生成。
                "--tools",
                "",
                # 关闭 CLAUDE.md、skills、hooks、plugins、MCP 等用户级定制；
                # 鉴权与模型选择不受影响（不能用 --bare，它会禁用 OAuth）。
                "--safe-mode",
            ]
            # 模型名留空时沿用 CLI 自己的默认模型，避免这里硬编码的模型 ID
            # 随订阅可用模型变化而失效。
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # 只记录变量名，不记录取值，避免把密钥写进日志。
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # CLI 会读取工作目录下的 CLAUDE.md 和项目设置，这些内容会污染
            # 文案结果，因此固定在一个临时空目录中执行。
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        # The CLI always emits UTF-8. Without an explicit encoding,
                        # text=True decodes with the system locale (e.g. cp1252 on
                        # non-English Windows), so every non-ASCII character reaches
                        # the script as mojibake.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # 未登录、用量耗尽这类失败同样会返回 JSON（`is_error` 为真，
            # `result` 是可读原因），只是退出码非 0。因此先解析 stdout，
            # 只有在拿不到 JSON 时才回退到退出码和 stderr。
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # 容器里无法执行交互式 /login，这里直接给出可用的鉴权方式。
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "modelscope":
            content = ""
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"


def test_connection() -> tuple[bool, str, float]:
    """
    使用当前 Provider 配置发起一次最小请求，验证实际生成链路是否可用。

    连接测试直接复用 `_generate_response()`，因此会覆盖 API Key、Base URL、
    模型名称和 Provider 专用字段，但不会进入脚本生成的重试逻辑，也不会发送
    用户的视频主题或文案。返回值依次为成功状态、错误信息和请求耗时。
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # Some upstream providers may return quota errors as plain text.
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # 有序关键词模式下，示例数量要和 amount 保持一致，避免模型被固定
        # 的 4 个示例误导，导致长文案只返回少量关键词，影响素材覆盖度。
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # generate_terms 的公开返回类型是 List[str]。如果把 Provider 的
                # 错误文案原样返回，下游只做空值判断时会把非空字符串误认为成功，
                # 素材下载循环还会按字符遍历错误文案，产生无意义的外部请求。
                # 这里统一返回空列表，让任务编排层在真实故障位置立即结束任务。
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# List-format scripts ("every X explained")
#
# 长视频清单格式：开场 + N 个条目 + 结尾。每个条目单独配一张画面，
# 渲染层据此逐段合成配音并与画面精确对齐，因此这里要求模型返回结构化 JSON，
# 而不是普通脚本的纯文本段落。
# =============================================================================

MIN_LIST_ITEM_COUNT = 3
DEFAULT_LIST_WORDS_PER_ITEM = 110
MIN_LIST_WORDS_PER_ITEM = 30
MAX_LIST_WORDS_PER_ITEM = 400


PERSONAS = {
    "otter": (
        "The narrator is the channel's mascot, a curious, friendly otter with round glasses; it speaks in the first "
        "person and may make ONE light, self-aware joke about being an otter (never more)."
    ),
}


def _intro_rule(words: int, persona: str = "") -> str:
    """The opening: a hook, then the topic said out loud, so the viewer knows at once what the video is about."""
    joke = (
        ' (for example: "That is why today we are going to talk about the greatest scientists in human history... '
        'well, even though I am an otter.")'
        if persona == "otter" else
        ' (for example: "That is why today we are going to talk about the greatest scientists in human history.")'
    )
    return (
        f'"intro": at most {words} words, in two moves. First a hook: open INSIDE a concrete, gripping moment (a precise '
        "time and place, a person living it, one or two sensory details) or with a surprising, verifiable fact or a question "
        "the viewer cannot ignore. Then, in one natural sentence, say clearly what the video is about using the idea of the "
        f"title{joke}, and promise what the viewer will understand by the end. Never greet, never say \"welcome\" or "
        '"in this video", never start with a definition.'
    )


def build_list_script_prompt(
    video_subject: str,
    item_count: int,
    language: str = "",
    words_per_item: int = DEFAULT_LIST_WORDS_PER_ITEM,
    persona: str = "",
) -> str:
    language_rule = (
        f"write title, intro, names, texts and outro in {language}"
        if language
        else "write title, intro, names, texts and outro in the same language as the video subject"
    )
    example = {
        "title": "Every Planet Explained",
        "intro": "Hook sentence that makes the viewer stay.",
        "intro_image_term": "solar system overview illustration",
        "items": [
            {
                "name": "Mercury",
                "text": "Mercury is ...",
                "image_term": "small grey planet mercury close to the sun",
            }
        ],
        "outro": "Closing sentence and a question for the comments.",
        "outro_image_term": "planets lined up in space",
    }
    return f"""
# Role: Script writer for long-form "every X explained" list videos

## Goal:
Write the narration for a video about the subject below. The video walks through
exactly {item_count} items, one at a time, each shown with its own picture.

## Constrains:
1. return only a JSON object with the keys shown in the output example; no markdown, no code fences.
2. {language_rule}; every image_term must be in English.
3. "title": a catchy video title, at most 70 characters.
4. {_intro_rule(80, persona)}{(" " + PERSONAS[persona]) if persona in PERSONAS else ""}
5. "items": exactly {item_count} objects, ordered to keep curiosity high, with the most surprising item last.
6. each item "name": at most 5 words.
7. each item "text": about {words_per_item} words of natural spoken narration, like a friendly science YouTuber who is also a careful teacher talking to a curious friend: scientific but easy to follow.
   - start by saying the item name;
   - define each technical term in one plain sentence the first time it appears ("voltage is the push that moves the electrons");
   - explain the mechanism: HOW it works and WHY it happens, step by step, with cause and effect ("because...", "that is why...");
   - when a quantity appears, say its unit and what it measures ("voltage is measured in volts, current in amperes");
   - when a simple law or formula applies, say it in words with its name ("Ohm's law: voltage equals current times resistance") and a tiny worked example with everyday numbers;
   - name the real structures, organs, devices and processes precisely (e.g. "the sinoatrial node, the heart's natural pacemaker"), so they can be shown on screen;
   - add one everyday comparison and one surprising, verifiable fact;
   - short sentences, no lists, no markdown, no emojis, no textbook tone.
8. each "image_term" (including intro_image_term and outro_image_term): 3 to 8 English words describing one concrete, drawable picture, with no text in the picture.
9. "outro": at most 40 words, closing the video with a question that invites comments.
10. use only accurate, verifiable facts with correct scientific terminology and real numbers with their units; never invent figures; for health topics never give treatment instructions or dosages.

## Output Example:
{json.dumps(example, ensure_ascii=False)}

## Video Subject:
{video_subject}
""".strip()


SCRIPT_FORMATS = ("list", "story")


def build_story_script_prompt(
    video_subject: str,
    item_count: int,
    language: str = "",
    words_per_item: int = DEFAULT_LIST_WORDS_PER_ITEM,
    persona: str = "",
) -> str:
    """A narrative script: one continuous story told in chapters, opened by a gripping situation."""
    language_rule = (
        f"write title, intro, names, texts and outro in {language}"
        if language
        else "write title, intro, names, texts and outro in the same language as the video subject"
    )
    example = {
        "title": "What It's Really Like to Work in a Call Centre",
        "intro": "It's 5:30 in the morning. Ana is already wearing her headset...",
        "intro_image_term": "woman with headset in a dark office at dawn",
        "items": [
            {"name": "The call centre", "text": "Ana is not alone. ...",
             "image_term": "call centre office photo"},
        ],
        "outro": "So the next time a friendly voice answers your call... Would you notice? Tell me in the comments.",
        "outro_image_term": "phone on a desk at night",
    }
    return f"""
# Role: Head writer of a calm, hand-drawn educational YouTube channel

## Goal:
Write the narration of ONE continuous story about the subject below, told in exactly {item_count} chapters,
as gripping as the best "what it's really like to be..." and animated explainer channels: the viewer
should feel they are living it, understand every idea completely and not be able to stop watching.

## Constrains:
1. return only a JSON object with the keys shown in the output example; no markdown, no code fences.
2. {language_rule}; every image_term must be in English.
3. "title": a curiosity-driven title of at most 70 characters (for example "What it's really like to...", "Why ... is disappearing", "What would happen if...").
4. {_intro_rule(100, persona)} The hook opens a question the video will answer.{(" " + PERSONAS[persona]) if persona in PERSONAS else ""}
5. "items": exactly {item_count} chapters that continue the same story in order (cause and consequence, rising stakes), each with:
   - "name": the subject of the chapter in 1 to 4 words, usually the person, place, thing or discovery it is about (for a video about scientists: "Isaac Newton"); it is shown on the chapter's title card with a photo and SAID ALOUD right before the chapter, so it must sound natural on its own;
   - "text": about {words_per_item} words that flow from the previous chapter (do not repeat the name as the first words: it was just said), develop ONE main idea completely (what happens, how it works and why, with one concrete example, number or character), and end with a line that pulls into the next chapter.
6. pacing: calm and clear, like a thoughtful narrator. Short sentences, one idea per sentence, many full stops so the voice can pause; finish every idea before starting the next; no lists, no markdown, no emojis, no hype words.
7. explain like a great teacher: when a technical term appears, say what it means in plain words; when a number appears, make it tangible with a comparison; always say why things happen.
8. "outro": at most 50 words: answer the question opened in the intro, leave a final thought, and ask one question for the comments.
9. "intro_image_term" and "outro_image_term": 3 to 8 English words describing one concrete, drawable picture of that moment. Each item's "image_term": an English search for a real photo of the chapter's subject for its title card (a portrait for a person: "Isaac Newton portrait"; a photo for a place or a thing).
10. use only accurate, verifiable facts and real numbers; invented characters must be presented as typical, not as real people, while real historical people are named and described accurately; for health topics never give treatment instructions or dosages.

## Output Example:
{json.dumps(example, ensure_ascii=False)}

## Video Subject:
{video_subject}
""".strip()


def _parse_list_script_response(response: str) -> dict:
    text = _strip_code_fence(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 部分模型会在 JSON 前后附带解释文字，退回提取最外层对象。
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("list script response is not a JSON object")
    return data


def generate_list_script(
    video_subject: str,
    item_count: int,
    language: str = "",
    words_per_item: int = DEFAULT_LIST_WORDS_PER_ITEM,
    app_config=None,
    script_format: str = "list",
    persona: str = "",
):
    """Generate an editable ListVideoScript for a list-format video.

    ``script_format`` "story" writes one continuous narrative in chapters
    (opened by a gripping situation) instead of an "every X explained" list.
    ``persona`` "otter" lets the narrator be the channel's otter mascot.

    Returns None when the provider fails or keeps returning invalid JSON, so the
    caller can report the failure instead of rendering an empty video.
    """
    # 延迟导入，避免 schema 在 llm 模块导入阶段读取配置造成循环依赖。
    from app.models.schema import MAX_LIST_VIDEO_ITEMS, ListVideoScript

    video_subject = _limit_script_text(video_subject, 500, "video_subject")
    if not video_subject:
        raise ValueError("video_subject is required for a list script")
    item_count = max(MIN_LIST_ITEM_COUNT, min(int(item_count), MAX_LIST_VIDEO_ITEMS))
    words_per_item = max(
        MIN_LIST_WORDS_PER_ITEM, min(int(words_per_item), MAX_LIST_WORDS_PER_ITEM)
    )
    build = build_story_script_prompt if script_format == "story" else build_list_script_prompt
    prompt = build(
        video_subject=video_subject,
        item_count=item_count,
        language=language,
        words_per_item=words_per_item,
        persona=persona,
    )
    logger.info(
        f"generating list script: subject={video_subject}, items={item_count}, "
        f"words_per_item={words_per_item}"
    )

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate list script: {response}")
                return None
            data = _parse_list_script_response(response)
            # 模型偶尔会额外返回说明性字段，只保留脚本模型认识的键。
            allowed_keys = set(ListVideoScript.model_fields)
            data = {key: value for key, value in data.items() if key in allowed_keys}
            if isinstance(data.get("items"), list):
                allowed_item_keys = {"name", "text", "image_term"}
                data["items"] = [
                    {k: v for k, v in item.items() if k in allowed_item_keys}
                    for item in data["items"]
                    if isinstance(item, dict)
                ]
            script = ListVideoScript.model_validate(data)
            if len(script.items) != item_count:
                logger.warning(
                    f"list script has {len(script.items)} items, "
                    f"requested {item_count}"
                )
            logger.success(
                f"list script generated: title={script.title!r}, "
                f"items={len(script.items)}"
            )
            return script
        except Exception as e:
            logger.warning(f"failed to parse list script: {type(e).__name__}: {e}")
        if i < _max_retries - 1:
            logger.warning(f"failed to generate list script, trying again... {i + 1}")

    return None


def build_translate_list_script_prompt(script_data: dict, language: str) -> str:
    return f"""
# Role: Translator and adapter for an educational YouTube channel

## Goal:
Rewrite the list-video script below in {language} for native speakers, so it
sounds as if it had been written in {language} from the start.

## Constrains:
1. return only a JSON object with exactly the same keys, and the same number of items in the same order; no markdown, no code fences.
2. translate "title", "intro", "outro" and every item's "name" and "text" into natural spoken {language}; adapt idioms, jokes, puns and recurring title formulas naturally (for example "La electricidad explicada para nutrias" becomes "Electricity Explained for Otters" in English).
3. keep the tone, the length and every fact; do not add or remove information.
4. copy every "image_term", "intro_image_term" and "outro_image_term" unchanged.

## Script:
{json.dumps(script_data, ensure_ascii=False)}
""".strip()


def translate_list_script(script, language: str, app_config=None):
    """Adapt a ListVideoScript to ``language``; None when the model fails.

    Pictures are language-independent, so image terms and files are always
    copied from the original rather than trusted from the model.
    """
    from app.models.schema import ListVideoScript

    language = (language or "").strip()
    if not language:
        raise ValueError("a target language is required to translate a list script")
    original = script.model_dump()
    source = {
        key: original[key]
        for key in ("title", "intro", "intro_image_term", "outro", "outro_image_term")
    }
    source["items"] = [
        {"name": item["name"], "text": item["text"], "image_term": item["image_term"]}
        for item in original["items"]
    ]
    prompt = build_translate_list_script_prompt(source, language)
    logger.info(f"translating list script to {language}: items={len(source['items'])}")

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to translate list script: {response}")
                return None
            data = _parse_list_script_response(response)
            items = data.get("items")
            if not isinstance(items, list) or len(items) != len(original["items"]):
                raise ValueError("translated script has a different number of items")
            translated = dict(original)
            for key in ("title", "intro", "outro"):
                value = data.get(key)
                if isinstance(value, str) and (value.strip() or not original[key]):
                    translated[key] = value.strip()
            translated["items"] = []
            for source_item, item in zip(original["items"], items):
                if not isinstance(item, dict):
                    raise ValueError("translated item is not an object")
                translated["items"].append(
                    {
                        **source_item,
                        "name": str(item.get("name") or "").strip(),
                        "text": str(item.get("text") or "").strip(),
                    }
                )
            result = ListVideoScript.model_validate(translated)
            logger.success(f"list script translated to {language}: {result.title!r}")
            return result
        except Exception as e:
            logger.warning(f"failed to parse translated list script: {type(e).__name__}: {e}")
        if i < _max_retries - 1:
            logger.warning(f"failed to translate list script, trying again... {i + 1}")
    return None


# =============================================================================
# Edit plan for list videos
#
# 让模型像剪辑师一样为每个片段安排画面节奏：角色表情、在说到某几个词时弹出
# 的配图，以及关键数字/术语的文字标注。锚点必须是旁白原文，渲染层据此换算
# 出精确的出现时间。
# =============================================================================

EDIT_BEAT_TYPES = ("image", "images", "text", "react")
EDIT_IMAGE_LOOKS = ("diagram", "photo")
EDIT_HOST_MODES = ("full", "lead", "react", "lead+react", "off")
EDIT_SCENE_TYPES = (
    "statement", "stat", "sequence", "compare", "diagram", "figure", "zoom", "story",
    "steps", "bars", "grid", "formula", "timeline", "gauge", "question",
    "definition", "equation", "annotate", "chain", "branch",
)
# Shots of the doodle look: everything above plus drawn compositions, real
# video clips in a frame and comic reaction cut-ins.
SHOT_TYPES = EDIT_SCENE_TYPES + ("single", "speech", "illustration", "clip", "meme", "animation", "archive")
MAX_SHOT_CHARACTERS = 4
MAX_ANIMATION_FRAMES = 6
MAX_SHOTS_PER_SEGMENT = 80
CAMERA_MOVES = ("in", "out", "left", "right")
CLIP_FRAMES = ("card", "full")
CLIP_AMOUNTS = ("none", "some", "more")
# Reactions of a meme cut-in (the folder of memes uses the same names).
MEME_MOODS = ("shock", "mindblown", "laugh", "facepalm", "confused", "scared", "sad", "proud", "suspicious", "panic")
EDIT_SCENE_MARKS = ("cross", "check")
MAX_EDIT_BEATS_PER_SEGMENT = 8
MAX_EDIT_REACTIONS_PER_SEGMENT = 2
MAX_EDIT_BACKGROUNDS_PER_SEGMENT = 8
MAX_EDIT_SCENES_PER_SEGMENT = 4
MAX_EDIT_TEXT_LENGTH = 40
MAX_SCENE_LABEL_LENGTH = 32
MAX_STATEMENT_LENGTH = 48


def _edit_plan_example(expressions: list) -> dict:
    expression = expressions[0] if expressions else ""
    return {
        "segments": [
            {
                "index": 1,
                "expression": expression,
                "backgrounds": [
                    {"at": "while you sleep", "query": "person sleeping in bed"},
                    {"at": "the fluid around your brain", "query": "water flowing slow motion"},
                    {"at": "one night without sleep", "query": "tired man at desk night"},
                ],
                "opener": {
                    "query": "glymphatic system brain illustration", "query_local": "sistema glinfático cerebro",
                    "look": "diagram", "icon": "🧠", "draw": "a brain being washed by a gentle stream of water",
                },
                "scenes": [
                    {
                        "type": "definition", "at": "the glymphatic system", "term": "glymphatic system",
                        "text": "the brain's night-time cleaning network", "symbol": "", "unit": "",
                        "icon": "🧠", "query": "glymphatic system illustration", "draw": "a brain with tiny water channels",
                    },
                    {
                        "type": "sequence",
                        "items": [
                            {"at": "not a muscle", "label": "a muscle", "icon": "💪", "query": "flexed arm muscle", "draw": "a flexed arm muscle", "mark": "cross"},
                            {"at": "but a cleaning crew", "label": "a cleaning crew", "icon": "🧹", "query": "", "draw": "a tiny broom sweeping", "mark": "check"},
                        ],
                    },
                    {"type": "figure", "at": "this is how it flows", "query": "glymphatic system diagram", "query_local": "", "look": "diagram", "seconds": 7, "label": ""},
                    {
                        "type": "chain",
                        "items": [
                            {"at": "the fluid enters", "label": "fluid", "icon": "💧", "query": "cerebrospinal fluid illustration", "draw": "a drop of clear fluid", "link": "washes"},
                            {"at": "around the neurons", "label": "neurons", "icon": "🧠", "query": "neurons illustration", "draw": "a few neurons", "link": "drains"},
                            {"at": "into the veins", "label": "veins", "icon": "🩸", "query": "veins illustration", "draw": "a blue vein"},
                        ],
                    },
                ],
                "beats": [
                    {"type": "image", "at": "you wake up tired", "query": "tired man yawning in bed", "look": "photo", "icon": "🥱"},
                    {
                        "type": "images",
                        "items": [
                            {"at": "coffee", "query": "cup of coffee", "look": "photo", "icon": "☕"},
                            {"at": "or an energy drink", "query": "energy drink can", "look": "photo", "icon": "🥤"},
                        ],
                    },
                    {"type": "text", "at": "one night without sleep", "text": "24 h awake"},
                ]
                + ([{"type": "react", "at": "it starts to fail", "expression": expressions[-1]}] if expressions else []),
            }
        ]
    }


OPENER_RULE = (
    '6b. "opener" (items only): the picture shown for 3 seconds beside the item\'s number and title when the item starts: {{"query": ..., "query_local": ..., "look": ..., "icon": ..., "draw": ...}} strictly about the item TITLE (not about a detail of its text): "query" an English search for the single most representative picture of that title (e.g. title "Voltaje y corriente" -> "voltage and current circuit diagram"), "query_local" the same in {language_name}, "draw" an English description of a simple illustration of the title. The first sentence of every item is said during the opener, so put no beat or scene on it.'
)


def build_edit_plan_prompt(
    segments: list, expressions: list, language: str = "", reference: Optional[list] = None, openers: bool = True
) -> str:
    language_name = language or "the language of the narration"
    if expressions:
        expression_rule = (
            '"expression": one of '
            + json.dumps(expressions, ensure_ascii=False)
            + " that matches the emotion of what is being said"
        )
        react_rule = (
            '- react beat: {"type": "react", "at": ..., "expression": ...} with an "expression" from the same list, '
            "only for a line with real emotional punch (a shocking number, something gross, funny or sad); "
            f"at most one per segment ({MAX_EDIT_REACTIONS_PER_SEGMENT} for very long ones)."
        )
        statement_rule = (
            '- {"type": "statement", "at": ..., "text": ..., "expression": ...}: the host alone on the channel colour '
            f"with a punchline of at most 6 words in {language_name}; for the single most important or surprising line, "
            "at most one per item and not in every item."
        )
    else:
        expression_rule = '"expression": always "" (no character is available)'
        react_rule = "- never use react beats (no character is available)."
        statement_rule = (
            '- {"type": "statement", "at": ..., "text": ...}: a punchline of at most 6 words in '
            f"{language_name} on the channel colour; at most one per item."
        )
    opener_rule = OPENER_RULE.format(language_name=language_name) if openers else (
        "6b. the segments flow into each other like one story: do not plan an \"opener\"."
    )
    reference_rule = ""
    if reference:
        reference_rule = f"""
## Reference plan:
The same video was already edited in another language. Reuse its visual plan so both versions
look the same: keep, segment by segment and in the same order, the same backgrounds queries,
scene types, openers, icons, "draw" descriptions, formulas, symbols, values, units, marks,
picture queries and expressions.
Only translate the labels and texts into {language_name} and choose new "at" anchors copied
from this version's text. Drop an element only when its idea is missing from this version.
{json.dumps(reference, ensure_ascii=False)}
"""
    return f"""
# Role: Video editor for an educational YouTube channel

## Goal:
Plan what appears on screen while each segment below is narrated, so the video feels
hand-edited, clear and never monotonous: footage that follows the narration, minimalist
explainer scenes that make an idea obvious at a glance (definitions, formulas, labelled
real pictures, processes), a host character that reacts, real and explanatory pictures of
the things being explained and key facts as short text. The channel is scientific but easy
to follow: show the real thing, its parts, how it works and why.

## Constrains:
1. return only a JSON object {{"segments": [...]}} with one entry per input segment, in the same order, each with "index", "expression", "backgrounds", "scenes" and "beats" (and "opener" for items); no markdown, no code fences.
2. {expression_rule}.
3. every "at" is 2 to 6 consecutive words copied exactly from that segment's text; the element appears when those words are spoken.
4. "backgrounds": stock footage behind the narration, one entry about every 15 to 20 words (a new shot every 6 to 8 seconds), the first one on the segment's first words; each is {{"at": ..., "query": ...}} where "query" is an English stock video search of 2 to 4 words describing a concrete scene a camera can film (people, places, objects, animals, nature, close-ups), never text, logos, charts or abstract ideas, and different from the other queries of the video.
5. "beats" (pictures that pop in over the footage; the main resource: 3 to 6 per item, 0 to 2 for intro and outro, never during a scene). Footage alone is the last resort: every sentence that names a concrete thing, a structure, a quantity, a unit or a mechanism gets a beat or a scene, so something explanatory is on screen at least every 6 to 8 seconds:
- image beat: {{"type": "image", "at": ..., "query": ..., "look": ..., "icon": ...}}: ONE picture of exactly what is being said, shown big in the centre. When the narration explains HOW something works or WHAT it is made of, search for an explanatory picture of exactly that (look "diagram"): "electrons flowing through a wire diagram", "like charges repel diagram", "heart electrical conduction system illustration"; for everyday things the SIMPLEST picture a 12-year-old gets at once (look "photo"): "tired man after workout", "car braking", never a dense textbook figure. "query" is an English search of 2 to 6 words; "icon" is an emoji used if no good picture exists.
- images beat: {{"type": "images", "items": [2 to 4 image items, each with its own "at", "query", "look", "icon"]}}: several things named one after the other, shown side by side (left/right, or left/centre/right) as each is named.
- text beat: {{"type": "text", "at": ..., "text": ...}} where "text" has at most 5 words in {language_name}: a number with its unit, a key term or a surprising fact stated in the narration; at most one per segment.
{react_rule}
6. "scenes": full-screen explainer moments that replace the footage to make one idea crystal clear; one or two per item (0 or 1 for intro and outro), never two in a row on consecutive sentences, and use MANY different types across the video (never the same type twice in a row; a statement at most every third item). Prefer the scientific ones whenever the narration allows: "definition" when a technical term is defined, "equation" when a formula or law is stated, "annotate" when the parts of a structure are named, "chain" or "steps" when a process is explained. Types:
- {{"type": "definition", "at": ..., "term": ..., "text": ..., "symbol": ..., "unit": ..., "icon": ..., "query": ..., "draw": ...}}: a glossary card when the narration defines a technical term: "term" at most 3 words; "text" the definition in at most 12 words in {language_name}, simplified from the narration; "symbol" its letter if it has one ("V", "I", "R", else ""); "unit" how it is measured when the narration says so ("se mide en voltios (V)", else ""); "icon", "query" and "draw" for a picture of it.
- {{"type": "equation", "at": ..., "name": ..., "formula": ..., "terms": [{{"symbol": ..., "label": ..., "unit": ..., "at": ...}}], "example": ...}}: a formula or law the narration states, written big with each symbol explained underneath: "name" e.g. "Ley de Ohm"; "formula" with single-letter symbols and spaces, e.g. "V = I × R" (operators =, +, −, ×, /); "terms" one per symbol with "label" (at most 2 words), "unit" (e.g. "voltios (V)") and an optional "at" where the narration explains it; "example" an optional worked example with numbers stated in the narration, e.g. "12 V = 2 A × 6 Ω".
- {{"type": "annotate", "at": ..., "query": ..., "query_local": ..., "look": "diagram" or "photo", "labels": [{{"label": ..., "at": ...}}]}}: a real picture of a structure (an organ, a machine, a cell, a circuit) with 2 to 5 labels pointing at the parts the narration names, e.g. the heart's conduction system with "nodo sinusal", "nodo AV", "haz de His"; "query" in English for that structure ("heart electrical conduction system diagram"), "query_local" the same in {language_name}; each label at most 3 words in {language_name} with an optional "at" where it is named.
- {{"type": "chain", "items": [2 to 4 items, each with "link"]}}: a process told as real things left to right joined by arrows, each arrow labelled with what that thing does to the next ("link": a verb of 1 or 2 words in {language_name}, e.g. "la turbina" -gira-> "los imanes" -empujan-> "los electrones" -llegan-> "tu casa"); give each item a "query" for a real picture.
- {{"type": "branch", "center": item, "items": [2 to 4 items]}}: one cause on the left and the effects it leads to fanning out on the right (the current -> light, heat, motion).
{statement_rule}
- {{"type": "question", "at": ..., "text": ..., "expression": ...}}: a big question the narration asks, with the host thinking.
- {{"type": "figure", "at": ..., "query": ..., "query_local": ..., "look": ..., "seconds": ..., "label": ...}}: a real picture filling the screen: an explanatory diagram, chart or infographic (look "diagram", e.g. "voltage current resistance diagram") or a striking photo (look "photo"); "query" in English, "query_local" the same search in {language_name} (it finds diagrams labelled in that language); "seconds" 3 to 4 for a photo, 6 to 8 for a diagram with text to read; "label" an optional caption of at most 6 words.
- {{"type": "zoom", "at": ..., "label": ..., "icon": ..., "query": ..., "draw": ..., "direction": "in" or "out"}}: one key thing shown big while the camera slowly pushes in or pulls out.
- {{"type": "stat", "at": ..., "value": 70, "unit": "%", "label": ..., "chart": "pie" or "number", "icon": ...}}: a number stated in the narration; "pie" only for a percentage of a whole.
- {{"type": "grid", "at": ..., "value": 7, "total": 10, "label": ..., "icon": ...}}: "7 out of 10" as a grid of icons with 7 coloured in.
- {{"type": "gauge", "at": ..., "value": 0 to 100, "label": ..., "low": ..., "high": ...}}: a meter whose needle swings to a level (risk, temperature, strength).
- {{"type": "bars", "unit": ..., "items": [2 to 5 items with "value"]}}: quantities compared as growing bars.
- {{"type": "sequence", "items": [2 to 4 items]}}: things the narration lists, popping in left to right as each one is named; "mark" is "cross" for something the narration denies ("it is not X, nor Y"), "check" for the right answer, otherwise "".
- {{"type": "compare", "items": [left, right]}}: two contrasting situations side by side (before/after, with/without, see it/lose it).
- {{"type": "steps", "cycle": false, "items": [2 to 5 items]}}: a process in order with numbered arrows; "cycle": true when it loops back (like the water cycle).
- {{"type": "timeline", "items": [2 to 5 items with "date"]}}: events in time on a line.
- {{"type": "formula", "operator": "+", "items": [2 to 3 items], "result": item}}: ingredients that combine into a result ("heat + oxygen + fuel = fire"); operator "+", "×", "−" or "→". For a real physics or chemistry formula use "equation" instead.
- {{"type": "diagram", "center": item, "items": [3 to 5 items]}}: several factors or parts that lead to one central idea; an arrow is drawn from each item to the centre as it is named.
- {{"type": "story", "items": [2 to 4 frames, each an item with "expression"]}}: a tiny flipbook where the host does something and something happens (plug in -> current flows -> the bulb lights up); each frame's "label" is a short caption.
   An item is {{"at": ..., "label": ..., "icon": ..., "query": ..., "draw": ...}} (plus "mark", "value", "date", "link" or "expression" where a type asks for them): "label" has at most 3 words in {language_name}; "icon" is ONE emoji that depicts the thing literally (when no emoji fits, 1 or 2 English words such as "kidney" or "stomach"); "query" is an English search of 2 to 4 words for a real picture of a concrete thing (real pictures are preferred: give one for every concrete thing, organ, device or place; leave it empty only for abstract ideas, where the emoji is clearer); "draw" is an English description of 5 to 12 words of a simple illustration of that thing.
{opener_rule}
7. never add facts that the narration does not state, and never show the same thing twice: every picture query, "draw" description and icon must be different across the whole video (pick another angle of the idea instead of repeating one).
{reference_rule}
## Output Example:
{json.dumps(_edit_plan_example(expressions), ensure_ascii=False)}

## Segments:
{json.dumps(segments, ensure_ascii=False)}
""".strip()


def _normalize_backgrounds(entries) -> list:
    backgrounds = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        anchor = str(entry.get("at") or "").strip()
        query = str(entry.get("query") or "").strip()[:100]
        if anchor and query:
            backgrounds.append({"at": anchor, "query": query})
    return backgrounds[:MAX_EDIT_BACKGROUNDS_PER_SEGMENT]


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain_number(number: float):
    return int(number) if float(number).is_integer() else round(number, 2)


def _scene_item(data, needs_anchor: bool = True, lookup: Optional[dict] = None) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    item = {
        "at": str(data.get("at") or "").strip(),
        "label": str(data.get("label") or "").strip()[:MAX_SCENE_LABEL_LENGTH],
        "icon": str(data.get("icon") or data.get("emoji") or "").strip()[:40],
        "draw": " ".join(str(data.get("draw") or "").split())[:300],
    }
    if (needs_anchor and not item["at"]) or not (item["label"] or item["icon"] or item["draw"] or data.get("pose")):
        return None
    query = str(data.get("query") or "").strip()[:80]
    if query:
        item["query"] = query
    if data.get("mark") in EDIT_SCENE_MARKS:
        item["mark"] = data["mark"]
    value = _number(data.get("value"))
    if value is not None:
        item["value"] = _plain_number(value)
    date = str(data.get("date") or "").strip()[:14]
    if date:
        item["date"] = date
    link = str(data.get("link") or "").strip()[:24]
    if link:
        item["link"] = link
    if data.get("otter"):
        item["otter"] = True
    _picture_extras(data, item)
    if lookup is not None:
        pose = lookup.get(str(data.get("pose") or "").strip().lower(), "")
        if pose:
            item["pose"] = pose
        expression = lookup.get(str(data.get("expression") or "").strip().lower(), "")
        if expression:
            item["expression"] = expression
    return item


def _slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:40]


def _picture_extras(data: dict, out: dict) -> dict:
    """Keep what a drawn or real picture may carry: the recurring characters it shows ("characters"),
    a gentle movement for an AI video ("motion") and a picture file pinned in a review ("image")."""
    if not isinstance(data, dict):
        return out
    characters = data.get("characters")
    if isinstance(characters, str):
        characters = [characters]
    if isinstance(characters, list):
        ids = [c for c in dict.fromkeys(_slug(c) for c in characters if c) if c][:MAX_SHOT_CHARACTERS]
        if ids:
            out["characters"] = ids
    motion = " ".join(str(data.get("motion") or "").split())[:200]
    if motion:
        out["motion"] = motion
    image = str(data.get("image") or "").strip()[:500]
    if image:
        out["image"] = image
    video = str(data.get("video") or "").strip()[:500]
    if video:
        out["video"] = video
    avoid = data.get("avoid")
    if isinstance(avoid, list):
        urls = [str(u).strip()[:500] for u in avoid if str(u or "").strip()][:20]
        if urls:
            out["avoid"] = urls
    return out


_ITEM_LIMITS = {
    "sequence": (2, 4), "compare": (2, 2), "diagram": (2, 5), "story": (2, 4),
    "steps": (2, 5), "bars": (2, 5), "timeline": (2, 5), "formula": (2, 3),
    "chain": (2, 4), "branch": (2, 4),
}
_SHOT_ITEM_LIMITS = {"sequence": (1, 4), "compare": (2, 2), "steps": (2, 5), "chain": (2, 4)}
MAX_FORMULA_LENGTH = 40


def _text(data: dict, key: str, limit: int) -> str:
    return " ".join(str(data.get(key) or "").split())[:limit]


def _normalize_scene(entry: dict, lookup: dict, shot: bool = False) -> Optional[dict]:
    kind = entry["type"]
    anchor = str(entry.get("at") or "").strip()
    label = str(entry.get("label") or "").strip()[:MAX_SCENE_LABEL_LENGTH]
    icon = str(entry.get("icon") or "").strip()[:40]
    expression = lookup.get(str(entry.get("expression") or "").strip().lower(), "")
    if kind in ("statement", "question"):
        text = str(entry.get("text") or "").strip()[:MAX_STATEMENT_LENGTH if kind == "statement" else 80]
        if anchor and text:
            return {"type": kind, "at": anchor, "text": text, "expression": expression}
        return None
    if kind == "stat":
        value = _number(entry.get("value"))
        if not anchor or value is None:
            return None
        unit = str(entry.get("unit") or "").strip()[:6]
        chart = "pie" if entry.get("chart") == "pie" and unit == "%" and 0 < value <= 100 else "number"
        stat = {"type": kind, "at": anchor, "value": _plain_number(value), "unit": unit, "label": label, "chart": chart, "icon": icon}
        for key in ("draw", "query"):
            if _text(entry, key, 160):
                stat[key] = _text(entry, key, 160)
        return _picture_extras(entry, stat)
    if kind == "grid":
        value, total = _number(entry.get("value")), _number(entry.get("total") or 10)
        if not anchor or value is None or total is None or not 2 <= total <= 100 or not 0 <= value <= total:
            return None
        return {"type": kind, "at": anchor, "value": int(round(value)), "total": int(round(total)), "label": label, "icon": icon}
    if kind == "gauge":
        value = _number(entry.get("value"))
        if not anchor or value is None:
            return None
        return {
            "type": kind, "at": anchor, "value": _plain_number(min(100.0, max(0.0, value))), "label": label,
            "low": str(entry.get("low") or "").strip()[:MAX_SCENE_LABEL_LENGTH],
            "high": str(entry.get("high") or "").strip()[:MAX_SCENE_LABEL_LENGTH],
        }
    if kind == "zoom":
        item = _scene_item(dict(entry, at=anchor), needs_anchor=True)
        if item is None:
            return None
        item.update(type=kind, direction="out" if entry.get("direction") == "out" else "in")
        return item
    if kind == "animation":
        frames = []
        for data in entry.get("frames") or entry.get("items") or []:
            draw = _text(data, "draw", 300) if isinstance(data, dict) else " ".join(str(data or "").split())[:300]
            if draw:
                frame = {"draw": draw}
                said = _text(data, "at", 80) if isinstance(data, dict) else ""
                if said:
                    frame["at"] = said
                if isinstance(data, dict) and str(data.get("image") or "").strip():
                    frame["image"] = str(data["image"]).strip()[:500]
                frames.append(frame)
        if not anchor or not frames:
            return None
        animation = {"type": kind if len(frames) > 1 else "illustration", "at": anchor, "text": _text(entry, "text", 16)}
        if len(frames) > 1:
            animation["frames"] = frames[:MAX_ANIMATION_FRAMES]
        else:
            animation["draw"] = frames[0]["draw"]
            if frames[0].get("image"):
                animation["image"] = frames[0]["image"]
        _picture_extras({k: v for k, v in entry.items() if k != "image"}, animation)
        if entry.get("otter"):
            animation["otter"] = True
        if entry.get("continue") is True or str(entry.get("continue")).lower() == "true":
            animation["continue"] = True
        camera = str(entry.get("camera") or "").strip().lower()
        if camera in CAMERA_MOVES:
            animation["camera"] = camera
        query = _text(entry, "query", 80)
        if query:
            animation["query"] = query
        return animation
    if kind in ("single", "illustration"):
        item = _scene_item(dict(entry, at=anchor), needs_anchor=True, lookup=lookup)
        if item is None or (kind == "illustration" and not item.get("draw")):
            return None
        item.update(type=kind, text=_text(entry, "text", 24 if kind == "illustration" else MAX_STATEMENT_LENGTH))
        if kind == "illustration":
            if entry.get("continue") is True or str(entry.get("continue")).lower() == "true":
                item["continue"] = True
            camera = str(entry.get("camera") or "").strip().lower()
            if camera in CAMERA_MOVES:
                item["camera"] = camera
        return item
    if kind == "archive":
        query = _text(entry, "query", 120)
        if not anchor or not query:
            return None
        archive = {
            "type": kind, "at": anchor, "query": query, "query_local": _text(entry, "query_local", 120),
            "text": _text(entry, "caption", 60) or _text(entry, "text", 60),
        }
        draw = _text(entry, "draw", 300)
        if draw:
            archive["draw"] = draw
        camera = str(entry.get("camera") or "").strip().lower()
        if camera in CAMERA_MOVES:
            archive["camera"] = camera
        return _picture_extras(entry, archive)
    if kind == "clip":
        query = _text(entry, "query", 80)
        if not anchor or not query:
            return None
        clip = {
            "type": kind, "at": anchor, "query": query, "label": label,
            "frame": entry.get("frame") if entry.get("frame") in CLIP_FRAMES else "card",
        }
        draw = _text(entry, "draw", 160)
        if draw:
            clip["draw"] = draw
        return _picture_extras(entry, clip)
    if kind == "meme":
        if not anchor:
            return None
        mood = str(entry.get("mood") or "").strip().lower()
        meme = {"type": kind, "at": anchor, "mood": mood if mood in MEME_MOODS else "shock", "text": _text(entry, "text", 40)}
        draw = _text(entry, "draw", 160)
        if draw:
            meme["draw"] = draw
        return meme
    if kind == "speech":
        people = [p for p in (_scene_item(d, needs_anchor=False, lookup=lookup) for d in entry.get("items") or []) if p][:2]
        if not anchor or not people:
            return None
        return {"type": kind, "at": anchor, "text": _text(entry, "text", 70), "items": people}
    if kind == "definition":
        term = _text(entry, "term", MAX_SCENE_LABEL_LENGTH) or label
        text = _text(entry, "text", 90)
        if not anchor or not term or not text:
            return None
        return {
            "type": kind, "at": anchor, "term": term, "text": text, "symbol": _text(entry, "symbol", 4),
            "unit": _text(entry, "unit", 40), "icon": icon, "query": _text(entry, "query", 80), "draw": _text(entry, "draw", 160),
        }
    if kind == "equation":
        formula = _text(entry, "formula", MAX_FORMULA_LENGTH)
        if not anchor or not formula:
            return None
        terms = []
        for data in entry.get("terms") or []:
            if not isinstance(data, dict):
                continue
            symbol = _text(data, "symbol", 6)
            if symbol and (data.get("label") or data.get("unit")):
                terms.append({
                    "symbol": symbol, "label": _text(data, "label", MAX_SCENE_LABEL_LENGTH),
                    "unit": _text(data, "unit", 24), "at": _text(data, "at", 80),
                })
        return {
            "type": kind, "at": anchor, "name": _text(entry, "name", 40), "formula": formula,
            "terms": terms[:4], "example": _text(entry, "example", 48),
        }
    if kind == "annotate":
        query = _text(entry, "query", 100)
        labels = [
            {"label": _text(data, "label", MAX_SCENE_LABEL_LENGTH), "at": _text(data, "at", 80)}
            for data in entry.get("labels") or entry.get("items") or [] if isinstance(data, dict) and _text(data, "label", 1)
        ]
        if not anchor or not query or len(labels) < 2:
            return None
        return {
            "type": kind, "at": anchor, "query": query, "query_local": _text(entry, "query_local", 100),
            "look": "photo" if entry.get("look") == "photo" else "diagram", "labels": labels[:5],
        }
    if kind == "figure":
        query = str(entry.get("query") or "").strip()[:100]
        if not anchor or not query:
            return None
        seconds = _number(entry.get("seconds")) or 5
        return {
            "type": kind, "at": anchor, "query": query,
            "query_local": str(entry.get("query_local") or "").strip()[:100],
            "look": "photo" if entry.get("look") == "photo" else "diagram",
            "seconds": _plain_number(min(9.0, max(3.0, seconds))), "label": str(entry.get("label") or "").strip()[:60],
        }
    items = [
        item
        for item in (_scene_item(data, lookup=lookup) for data in entry.get("items") or entry.get("frames") or [])
        if item
    ]
    if kind == "bars":
        items = [item for item in items if "value" in item]
    if kind == "timeline":
        items = [item for item in items if item.get("date")] or items
    low, high = (_SHOT_ITEM_LIMITS.get(kind) if shot else None) or _ITEM_LIMITS[kind]
    if len(items) < low:
        return None
    scene = {"type": kind, "items": items[:high]}
    if kind in ("diagram", "branch"):
        center = _scene_item(entry.get("center"), needs_anchor=False)
        scene["center"] = center or {"at": "", "label": "", "icon": "", "draw": ""}
    elif kind == "steps":
        scene["cycle"] = bool(entry.get("cycle"))
    elif kind == "bars":
        scene["unit"] = str(entry.get("unit") or "").strip()[:6]
    elif kind == "formula":
        result = _scene_item(entry.get("result"))
        if result is None:
            return None
        operator = str(entry.get("operator") or "+").strip()
        scene["operator"] = operator if operator in ("+", "×", "x", "−", "-", "→", "÷") else "+"
        scene["result"] = result
    elif kind == "story":
        scene["expression"] = expression
    return scene


def _normalize_scenes(entries, lookup: dict) -> list:
    scenes = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("type") not in EDIT_SCENE_TYPES:
            continue
        scene = _normalize_scene(entry, lookup)
        if scene is not None:
            scenes.append(scene)
    return scenes[:MAX_EDIT_SCENES_PER_SEGMENT]


def _normalize_opener(data) -> Optional[dict]:
    """The picture that opens an item: what to search and what to draw if nothing fits."""
    if not isinstance(data, dict):
        return None
    query = str(data.get("query") or "").strip()[:100]
    if not query:
        return None
    opener = {
        "query": query,
        "query_local": str(data.get("query_local") or "").strip()[:100],
        "look": "photo" if data.get("look") == "photo" else "diagram",
    }
    icon = str(data.get("icon") or "").strip()[:40]
    draw = str(data.get("draw") or "").strip()[:160]
    if icon:
        opener["icon"] = icon
    if draw:
        opener["draw"] = draw
    return _picture_extras({k: v for k, v in data.items() if k in ("image", "avoid")}, opener)


def _image_beat(beat: dict, anchor: str) -> Optional[dict]:
    query = str(beat.get("query") or "").strip()[:100]
    if not query or not anchor:
        return None
    image = {
        "type": "image", "at": anchor, "query": query,
        "look": beat.get("look") if beat.get("look") in EDIT_IMAGE_LOOKS else "diagram",
    }
    icon = str(beat.get("icon") or "").strip()[:40]
    if icon:
        image["icon"] = icon
    return image


def normalize_edit_plan(data, segment_count: int, expressions: list) -> list:
    """Validate a model edit plan; unknown or malformed parts are dropped."""
    if isinstance(data, dict):
        data = data.get("segments")
    if not isinstance(data, list):
        raise ValueError("edit plan has no segments list")
    lookup = {expression.lower(): expression for expression in expressions}
    by_index = {}
    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        index = entry.get("index", position)
        if not isinstance(index, int) or not 0 <= index < segment_count:
            index = position
        if index >= segment_count or index in by_index:
            continue
        expression = lookup.get(str(entry.get("expression") or "").strip().lower(), "")
        beats = []
        reactions = []
        for beat in entry.get("beats") or []:
            if not isinstance(beat, dict) or beat.get("type") not in EDIT_BEAT_TYPES:
                continue
            if beat["type"] == "images":
                members = [
                    image for image in (
                        _image_beat(item, str(item.get("at") or "").strip())
                        for item in beat.get("items") or [] if isinstance(item, dict)
                    ) if image
                ][:4]
                if len(members) >= 2:
                    beats.append({"type": "images", "items": [{k: v for k, v in m.items() if k != "type"} for m in members]})
                elif members:
                    beats.append(members[0])
                continue
            anchor = str(beat.get("at") or "").strip()
            if not anchor:
                continue
            if beat["type"] == "image":
                image = _image_beat(beat, anchor)
                if image:
                    beats.append(image)
            elif beat["type"] == "text":
                text = str(beat.get("text") or "").strip()[:MAX_EDIT_TEXT_LENGTH]
                if text:
                    beats.append({"type": "text", "at": anchor, "text": text})
            else:
                reaction = lookup.get(str(beat.get("expression") or "").strip().lower(), "")
                if reaction:
                    reactions.append({"type": "react", "at": anchor, "expression": reaction})
        normalized = {
            "index": index,
            "expression": expression,
            "backgrounds": _normalize_backgrounds(entry.get("backgrounds")),
            "scenes": _normalize_scenes(entry.get("scenes"), lookup),
            "beats": beats[:MAX_EDIT_BEATS_PER_SEGMENT] + reactions[:MAX_EDIT_REACTIONS_PER_SEGMENT],
        }
        host = str(entry.get("host") or "").strip().lower()
        if host in EDIT_HOST_MODES:
            normalized["host"] = host
        opener = _normalize_opener(entry.get("opener"))
        if opener:
            normalized["opener"] = opener
        by_index[index] = normalized
    return [
        by_index.get(index, {"index": index, "expression": "", "backgrounds": [], "scenes": [], "beats": []})
        for index in range(segment_count)
    ]


# ---------------------------------------------------------------------------
# The director: a visual bible for the whole video, then the shots, then a review
# ---------------------------------------------------------------------------

MAX_BIBLE_CHARACTERS = 10
MAX_ARCHIVE_IDEAS = 6


def build_visual_bible_prompt(segments: list, language: str = "", persona: str = "") -> str:
    """One look at the whole script before any shot is planned: the art direction, the recurring characters
    (how each looks, and a real portrait to draw them from) and the real historical pictures worth showing."""
    language_name = language or "the language of the narration"
    mascot = (
        "The channel's mascot, an otter with round glasses, a teal sweater and a pencil behind its ear, narrates and "
        "appears in the human situations; do not list it as a character."
        if persona == "otter" else ""
    )
    example = {
        "style": "warm candle-lit 17th and 19th century interiors and night skies; muted ochre, teal and deep blue",
        "characters": [
            {"id": "newton", "name": "Isaac Newton",
             "look": "English scholar in his twenties, long wavy brown hair to the shoulders, pale thin face, white linen "
                     "shirt with a loose collar under a long dark brown coat",
             "portrait": "Isaac Newton portrait Godfrey Kneller 1689"},
        ],
        "sections": [
            {"index": 2, "archive": [
                {"query": "Great Plague of London 1665 engraving", "query_local": "gran peste de Londres 1665 grabado",
                 "caption": "La gran peste de Londres, 1665", "about": "the plague that closed Cambridge"},
                {"query": "Philosophiae Naturalis Principia Mathematica 1687 first edition title page",
                 "query_local": "Principia Mathematica 1687 portada", "caption": "Principia, 1687",
                 "about": "the book where he published the law of gravitation"},
            ]},
        ],
    }
    return f"""
# Role: Art director of an animated educational YouTube channel

## Goal:
Before the shots are planned, read the whole narration below and write the visual bible of this video, so every
picture is consistent and the real history is shown with real pictures.

## Constrains:
1. return only a JSON object like the example; no markdown.
2. "style": one sentence of art direction for this video (moods, light, era details, two or three accent colours);
   it is added to every drawing of the channel's polished 2D cartoon style.
3. "characters": every real or invented person who appears in more than one moment (at most {MAX_BIBLE_CHARACTERS}):
   "id" (lowercase, no spaces), "name", "look" (an English description of 20 to 40 words: age, face, hair, clothes and
   colours of the period, so an illustrator draws them the same way every time) and, for a real historical person,
   "portrait": an English search for a real, well-known portrait or photograph of them. {mascot}
4. "sections": for each segment index, "archive": up to {MAX_ARCHIVE_IDEAS} REAL pictures worth showing full screen when
   that segment is narrated: famous paintings, engravings, historical photographs, manuscripts, title pages, maps,
   museum objects and real places of exactly what the narration tells (a portrait of the person, a painting of the
   event, a photograph of the real instrument). Each with "query" (a precise English search for Wikimedia Commons:
   who or what, the kind of picture, the year or the artist), "query_local" (the same in {language_name}), "caption"
   (at most 6 words in {language_name}, e.g. a place and a year) and "about" (what it shows, in English).
   Only things that really were painted, engraved or photographed; never invent a painting.
5. never add facts that the narration does not state.

## Output Example:
{json.dumps(example, ensure_ascii=False)}

## Segments:
{json.dumps(segments, ensure_ascii=False)}
""".strip()


def normalize_visual_bible(data) -> dict:
    """A valid visual bible: style, characters by id and archive ideas by segment index."""
    if not isinstance(data, dict):
        raise ValueError("visual bible is not an object")
    bible = {"style": " ".join(str(data.get("style") or "").split())[:300], "characters": [], "sections": {}}
    seen = set()
    for entry in data.get("characters") or []:
        if not isinstance(entry, dict):
            continue
        key = _slug(entry.get("id") or entry.get("name"))
        look = " ".join(str(entry.get("look") or "").split())[:400]
        if not key or not look or key in seen:
            continue
        seen.add(key)
        bible["characters"].append({
            "id": key, "name": " ".join(str(entry.get("name") or key).split())[:60], "look": look,
            "portrait": " ".join(str(entry.get("portrait") or "").split())[:120],
        })
        if len(bible["characters"]) >= MAX_BIBLE_CHARACTERS:
            break
    for entry in data.get("sections") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), int):
            continue
        ideas = []
        for idea in entry.get("archive") or []:
            if isinstance(idea, dict) and _text(idea, "query", 120):
                ideas.append({
                    "query": _text(idea, "query", 120), "query_local": _text(idea, "query_local", 120),
                    "caption": _text(idea, "caption", 60), "about": _text(idea, "about", 160),
                })
        if ideas:
            bible["sections"][entry["index"]] = ideas[:MAX_ARCHIVE_IDEAS]
    return bible


def generate_visual_bible(segments: list, language: str = "", persona: str = "", app_config=None) -> Optional[dict]:
    """The visual bible of a video, or None when the model keeps failing."""
    prompt = build_visual_bible_prompt(segments, language, persona)
    for i in range(min(_max_retries, 3)):
        try:
            response = _generate_response(prompt) if app_config is None else _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to write the visual bible: {response}")
                return None
            bible = normalize_visual_bible(_parse_list_script_response(response))
            logger.success(
                f"visual bible written: {len(bible['characters'])} characters, "
                f"{sum(len(v) for v in bible['sections'].values())} archive pictures"
            )
            return bible
        except Exception as e:
            logger.warning(f"failed to parse the visual bible: {type(e).__name__}: {e}")
    return None


def bible_for(bible: Optional[dict], indexes) -> dict:
    """The part of the bible a chunk of segments needs: all characters, its own archive ideas."""
    if not bible:
        return {}
    wanted = set(indexes)
    return {
        "style": bible.get("style", ""),
        "characters": bible.get("characters", []),
        "archive": {str(k): v for k, v in (bible.get("sections") or {}).items() if k in wanted},
    }


def _storyboard_example(expressions: list, clips: bool = True, memes: bool = False) -> dict:
    shots = [
        {"type": "archive", "at": "in the summer of 1665", "query": "Great Plague of London 1665 engraving",
         "query_local": "gran peste de Londres 1665", "caption": "Londres, 1665", "camera": "in",
         "draw": "a deserted 17th-century London street at dusk, doors marked with red crosses, carts of the sick"},
        {"type": "animation", "at": "Newton went back to his farm", "characters": ["newton"], "camera": "right",
         "query": "Woolsthorpe Manor", "frames": [
             {"draw": "wide shot: young Newton walks along a country lane towards a small stone farmhouse with an apple "
                      "orchard, late afternoon light"},
             {"draw": "medium shot: Newton sits under an apple tree reading a book, the farmhouse behind him",
              "at": "under an apple tree"},
             {"draw": "close-up: an apple falls from the branch past Newton's face, he looks up surprised",
              "at": "an apple fell"},
         ]},
        {"type": "illustration", "at": "the same force that pulls the apple", "characters": ["newton"], "camera": "out",
         "text": "", "query": "moon orbit earth illustration", "motion": "the moon slowly circles the earth",
         "draw": "Newton on a hilltop at night points from a falling apple to the full moon, a faint dotted curve joins "
                 "them, starry sky"},
        {"type": "equation", "at": "the force grows with the masses", "name": "Ley de gravitación",
         "formula": "F = G × m1 × m2 / r²", "terms": [{"symbol": "F", "label": "fuerza", "unit": "newtons (N)"},
                                                     {"symbol": "r", "label": "distancia", "unit": "metros (m)"}]},
        {"type": "archive", "at": "he published it in 1687", "query": "Principia Mathematica 1687 first edition title page",
         "query_local": "Principia Mathematica 1687", "caption": "Principia, 1687", "camera": "left",
         "draw": "an old leather-bound book opened on a desk by candlelight, a quill beside it"},
    ]
    if clips:
        shots.append({"type": "clip", "at": "a storm over the sea", "query": "lightning storm over sea", "label": "",
                      "frame": "full", "draw": "lightning over a dark sea at night"})
    if memes:
        shots.append({"type": "meme", "at": "and nobody believed him", "mood": "facepalm", "text": "",
                      "draw": "the otter with a paw on its face, sighing"})
    return {"segments": [{"index": 2, "shots": shots}]}


def storyboard_shot_target(seconds: float, shot_seconds: float = 5.0) -> int:
    """How many shots a segment of ``seconds`` needs to change picture every ``shot_seconds``."""
    return max(1, int(round(max(0.0, seconds) / max(1.5, shot_seconds))))


COMPOSITION_TYPES = ("stat", "equation", "timeline", "compare", "single", "question", "statement")


def build_storyboard_prompt(
    segments: list, expressions: list, language: str = "", reference: Optional[list] = None, openers: bool = True,
    clips: str = "some", memes: bool = False, bible: Optional[dict] = None,
) -> str:
    """The director of the doodle look: a calm animated film of full-screen pictures that follow the narration.

    Segments may carry their spoken length ("seconds") and how many shots they
    need ("shots"). ``bible`` is the part of the visual bible these segments
    use (style, characters and real pictures worth showing). ``clips`` is how
    often a real video clip appears ("none", "some", "more"); ``memes``
    allows comic reaction cut-ins.
    """
    language_name = language or "the language of the narration"
    pose_rule = (
        '"pose": one of ' + json.dumps(expressions, ensure_ascii=False) + " to show the channel's otter in that mood "
        "exactly as it is drawn"
        if expressions else '"pose": always ""'
    )
    opener_rule = (
        "12. each item already opens with a 3-second title card (its number, its name and a photo) while its first "
        "sentence is said: start its first shot after that sentence, and give items an \"opener\": {\"query\": an English "
        "search for the most representative real photo or portrait of the item's subject, \"draw\": an illustration "
        "of it to use if no photo fits}."
        if openers else "12. the segments flow into each other like one continuous story: keep the visual thread going across them."
    )
    clips = clips if clips in CLIP_AMOUNTS else "some"
    clip_type = ""
    clip_rule = 'never use "clip" (no real video clips in this video).'
    if clips != "none":
        clip_rule = (
            f'a "clip" (a real stock video) about {3 if clips == "some" else 7}% of the shots, only for real places, '
            "nature, machines and everyday actions that a camera can film today."
        )
        clip_type = (
            '\n- {"type": "clip", "at": ..., "query": an English stock-video search of 2 to 4 words, "label": "", '
            '"frame": "full" or "card", "draw": an illustration to use instead if no video is found}.'
        )
    meme_rule = meme_type = ""
    if memes:
        meme_rule = (
            '13. "meme": a two-second comic reaction, only on a real punchline or an absurd fact, at most one every '
            "60 seconds."
        )
        meme_type = (
            '\n- {"type": "meme", "at": ..., "mood": one of ' + json.dumps(list(MEME_MOODS)) + ', "text": "" or a caption '
            f'of at most 4 words in {language_name}, "draw": the otter\'s exaggerated reaction}}.'
        )
    bible_rule = ""
    if bible:
        bible_rule = f"""
## Visual bible of this video (follow it):
{json.dumps(bible, ensure_ascii=False)}
- every drawing that shows one of these characters lists its id in "characters" and does not describe its looks
  again (the illustrator draws it from its character sheet);
- the "archive" ideas of a segment are real pictures that exist: use them as archive shots when their moment is said.
"""
    reference_rule = ""
    if reference:
        reference_rule = f"""
## Reference storyboard:
The same video was already planned in another language. Keep, shot by shot, the same types, drawings, queries,
characters and numbers, so the same pictures are reused; only translate texts and labels into {language_name} and
pick new "at" anchors from this version's text.
{json.dumps(reference, ensure_ascii=False)}
"""
    return f"""
# Role: Director of a calm, professional animated documentary for YouTube

## Goal:
Plan what is on screen while the narration below is said, like the director of a good documentary: full-screen
pictures that show EXACTLY what the narrator is saying at that moment and put it in context, one after another at a
calm pace, so the viewer follows the story without effort. Three kinds of pictures fill the screen:
- REAL pictures ("archive") for real history and the real world: the portrait of the person who is named, a painting or
  engraving of the event, a historical photograph, the real instrument in a museum, a manuscript, a map. When the
  narration tells real history, the real picture is always the first choice;
- short ANIMATIONS ("animation"): 2 to 4 cartoon drawings of one action in the same place, one after another, about
  1.5 to 2 seconds each (the moment an atom's core lights up, an apple falling, a scientist discovering something);
- cartoon ILLUSTRATIONS ("illustration"): one drawn moment of the story with a slow camera move, for what has no real
  picture (an idea, a mechanism, an invisible process, a scene with the channel's otter).
Explainer compositions on the coloured background are the exception: only a number, a formula or a date line that is
much clearer written.

## Constrains:
1. return only a JSON object {{"segments": [...]}} with one entry per input segment, in the same order, each with "index" and "shots"; no markdown.
2. pace: calm. Each segment gives its spoken length in "seconds" and how many shots it needs in "shots": plan about
   that many (an animation counts as one shot), in narration order, so the picture changes every 4 to 7 seconds; never
   two shots less than 4 seconds apart. Cover EVERY sentence with something that shows what it says.
3. every "at" is 2 to 6 consecutive words copied exactly from that segment's text (same spelling and accents); the shot
   (or the frame) appears when they are spoken. The anchors of a segment are all different and follow the text's order.
4. precision first: each picture shows exactly the person, object, place, event or idea being said at that moment, in its
   real historical context (period, place, clothes, instruments). Never a generic or decorative picture (a character
   walking on a road while the narration talks about a plague is wrong: show the plague). Never show the same object,
   person in the same pose or place twice unless the action continues.
5. mix for history, biography and science: about 30% "archive" when the narration tells real history (less when it
   does not), about 35% "animation", about 25% "illustration", and at most 10% compositions ("stat", "equation",
   "timeline", "single", "question", "statement"); {clip_rule} At most ONE "compare" (two things side by side) in the
   whole video, and only when the narration explicitly contrasts two things; never two compositions in a row.
6. animations are the jewel of the video: in each frame after the first describe ONLY what changes (the core starts to
   glow -> it shines brightly -> rays burst out); keep the same framing and place; give a frame its own "at" when it
   must appear on precise words.
7. "continue": true on an illustration or an animation draws its first picture from the last drawn picture before it
   (same place and characters), to keep a scene going.
8. drawings: "draw" is an English description of 15 to 50 words written for an illustrator: the framing (wide shot,
   medium shot, close-up, cutaway, top view), the subject, the action, the place and period, the light and mood. Keep
   it simple and readable: one clear focal point, few objects, no complex machinery in tiny detail, never text,
   letters, numbers, labels or formulas inside a drawing. "camera": "in", "out", "left" or "right"; vary it.
9. the channel's mascot (an otter with round glasses and a teal sweater) appears only in a few human situations or
   reactions ("otter": true, or a {pose_rule}); real people are drawn as themselves.
10. every picture shot also has "query": a short English search for a real photo of the same moment, used if the
   drawing fails; an illustration may have "motion": one short English sentence describing a gentle movement that would
   bring it to life as a 4-second video (the apple slowly falls, the candle flame flickers), or "" when nothing moves.
11. labels and captions have at most 6 words in {language_name}; most pictures need no caption; never add facts that the
   narration does not state.
{opener_rule}
{meme_rule}

## Shot types:
- {{"type": "archive", "at": ..., "query": a precise English search for Wikimedia Commons (who or what, the kind of picture, the year or the artist), "query_local": the same in {language_name}, "caption": a place and year or a name ("Londres, 1665"), "camera": ..., "draw": an illustration of the same moment to use if no real picture fits}}: a real picture filling the screen.
- {{"type": "animation", "at": ..., "characters": [ids], "otter": false, "continue": false, "camera": ..., "query": ..., "frames": [{{"draw": ..., "at": optional}}, 2 to 4 frames]}}: a short animation of one action.
- {{"type": "illustration", "at": ..., "draw": ..., "characters": [ids], "otter": false, "continue": false, "camera": ..., "text": "" or a caption of at most 3 words such as "PADUA, 1609", "query": ..., "motion": ...}}: one drawn moment.
- {{"type": "stat", "at": ..., "value": 6, "unit": "", "label": ..., "chart": "number" or "pie", "draw": a drawing of what is counted, "query": ...}}: a number that counts up next to its drawing.
- {{"type": "equation", "at": ..., "name": ..., "formula": with ×, /, ² and Greek letters as symbols (never *, ^ or sqrt), "terms": [{{"symbol": ..., "label": ..., "unit": ...}}]}}: only when the narration states a formula.
- {{"type": "timeline", "items": [2 to 4 items with "date", each with "label", "draw", "query" and "at"]}}: dates in order.
- {{"type": "single", "at": ..., "text": "" or a title, "label": ..., "draw": a detailed drawing of one object, "query": ...}}: one object studied up close (a sample of polonium, a voltaic pile).
- {{"type": "compare", "items": [left, right, each {{"at": ..., "label": ..., "draw": ..., "query": ...}}]}}: two things side by side (at most once).
- {{"type": "question", "at": ..., "text": ..., "expression": ...}}, {{"type": "statement", "at": ..., "text": ..., "expression": ...}}: a big question or punchline with the otter (at most one per segment).{clip_type}{meme_type}
{bible_rule}{reference_rule}
## Output Example:
{json.dumps(_storyboard_example(expressions, clips != "none", memes), ensure_ascii=False)}

## Segments:
{json.dumps(segments, ensure_ascii=False)}
""".strip()


def normalize_storyboard(data, segment_count: int, expressions: list) -> list:
    """Validate a storyboard; unknown or malformed shots are dropped."""
    if isinstance(data, dict):
        data = data.get("segments")
    if not isinstance(data, list):
        raise ValueError("storyboard has no segments list")
    lookup = {expression.lower(): expression for expression in expressions}
    by_index = {}
    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        index = entry.get("index", position)
        if not isinstance(index, int) or not 0 <= index < segment_count:
            index = position
        if index >= segment_count or index in by_index:
            continue
        shots = []
        for shot in entry.get("shots") or entry.get("scenes") or []:
            if isinstance(shot, dict) and shot.get("type") in SHOT_TYPES:
                normalized = _normalize_scene(shot, lookup, shot=True)
                if normalized is not None:
                    shots.append(normalized)
        planned = {"index": index, "expression": lookup.get(str(entry.get("expression") or "").strip().lower(), ""),
                   "shots": shots[:MAX_SHOTS_PER_SEGMENT], "backgrounds": [], "scenes": [], "beats": []}
        opener = _normalize_opener(entry.get("opener"))
        if opener:
            planned["opener"] = opener
        by_index[index] = planned
    return [
        by_index.get(index, {"index": index, "expression": "", "shots": [], "backgrounds": [], "scenes": [], "beats": []})
        for index in range(segment_count)
    ]


STORYBOARD_CHUNK_SHOTS = 45  # shots asked in one request at most (long answers get cut or rushed)


def storyboard_chunks(segments: list, most: int = STORYBOARD_CHUNK_SHOTS) -> List[list]:
    """Consecutive groups of segments planned in one request each."""
    chunks: List[list] = []
    current: list = []
    planned = 0
    for segment in segments:
        shots = segment.get("shots", 0) if isinstance(segment, dict) else 0
        shots = shots if isinstance(shots, int) else 0
        if current and planned + shots > most:
            chunks.append(current)
            current, planned = [], 0
        current.append(segment)
        planned += shots
    if current:
        chunks.append(current)
    return chunks


def _storyboard_request(prompt: str, count: int, expressions: list, app_config=None) -> Optional[list]:
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt) if app_config is None else _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate the storyboard: {response}")
                return None
            return normalize_storyboard(_parse_list_script_response(response), count, expressions)
        except Exception as e:
            logger.warning(f"failed to parse the storyboard: {type(e).__name__}: {e}")
        if i < _max_retries - 1:
            logger.warning(f"failed to generate the storyboard, trying again... {i + 1}")
    return None


def build_storyboard_review_prompt(
    segments: list, board: list, language: str = "", bible: Optional[dict] = None, clips: str = "some",
) -> str:
    """The film editor's pass over a planned storyboard: it fixes what a viewer would notice before anything is drawn."""
    language_name = language or "the language of the narration"
    planned = [
        {"index": entry.get("index"), "shots": entry.get("shots") or [], **({"opener": entry["opener"]} if entry.get("opener") else {})}
        for entry in board
    ]
    bible_rule = f"\n## Visual bible of this video:\n{json.dumps(bible, ensure_ascii=False)}\n" if bible else ""
    return f"""
# Role: Film editor of a calm, professional animated documentary for YouTube

## Goal:
A director planned the pictures of the narration below. Watch it in your head, sentence by sentence, as a viewer
would, and return the storyboard corrected, so that every picture shows exactly what is being said, in context, at a
calm pace that is easy to follow.

## Fix:
1. a picture that does not show exactly what is said at that moment (a generic, decorative or off-topic picture), or
   whose real-world context is wrong (period, place, person, instrument): change it so it does;
2. real history told with a drawing when a real picture exists (a portrait of the person named, a painting or engraving
   of the event, a historical photograph, the real instrument, the manuscript): turn it into an "archive" shot with a
   precise "query" (and keep its "draw" as the fallback);
3. compositions on the coloured background that are not essential: turn them into full-screen "illustration",
   "animation" or "archive" shots. Keep at most ONE "compare" in the whole video and never two compositions in a row;
4. repetition: the same object, person pose or place shown twice without the action continuing: replace one;
5. pace: shots less than 4 seconds apart, or one picture held while the narration moves to something new: merge or add
   so the picture changes every 4 to 7 seconds;
6. drawings that ask for text, labels, numbers, formulas or intricate machinery inside the picture: simplify them;
7. recurring characters of the bible that are described again instead of listed in "characters": list their ids;
8. anchors ("at") that are not words copied exactly from the segment's text, in order: fix them.
Keep everything that is already good. Labels and captions stay in {language_name}.{" No clips." if clips == "none" else ""}

## Constrains:
return only the corrected JSON object {{"segments": [...]}} with every segment and ALL its shots (not only the changed
ones), in exactly the same format; no markdown, no comments.
{bible_rule}
## Narration:
{json.dumps(segments, ensure_ascii=False)}

## Storyboard to review:
{json.dumps({"segments": planned}, ensure_ascii=False)}
""".strip()


def generate_storyboard(
    segments: list, expressions: list, language: str = "", app_config=None, reference: Optional[list] = None,
    openers: bool = True, clips: str = "some", memes: bool = False, bible: Optional[dict] = None,
    review: bool = True,
):
    """Ask the model for the doodle storyboard; None when it keeps failing.

    Long videos are planned a few segments at a time, in parallel, so every
    answer stays short enough to be complete. With ``review`` a second pass,
    the film editor, corrects each part against the narration (relevance,
    real pictures, repetition, pace) before anything is drawn. Segments whose
    request failed come back without "shots" (the caller draws them its own
    simple way).
    """
    count = len(segments)
    chunks = storyboard_chunks(segments) if any(isinstance(s, dict) and s.get("shots") for s in segments) else [segments]

    def plan(chunk: list) -> Optional[list]:
        indexes = [s.get("index", n) if isinstance(s, dict) else n for n, s in enumerate(chunk)]
        wanted = set(indexes) if len(chunks) > 1 else None
        part = [e for e in reference or [] if not wanted or (isinstance(e, dict) and e.get("index") in wanted)]
        guide = bible_for(bible, indexes) or None
        prompt = build_storyboard_prompt(chunk, expressions, language, part or None, openers, clips, memes, guide)
        board = _storyboard_request(prompt, count, expressions, app_config)
        if board is None or not review or reference:
            return board
        mine = [board[i] for i in indexes if isinstance(i, int) and 0 <= i < count]
        reviewed = _storyboard_request(
            build_storyboard_review_prompt(chunk, mine, language, guide, clips), count, expressions, app_config
        )
        if reviewed is None:
            return board
        for i in indexes:
            if isinstance(i, int) and 0 <= i < count and reviewed[i].get("shots"):
                if not reviewed[i].get("opener") and board[i].get("opener"):
                    reviewed[i]["opener"] = board[i]["opener"]
                board[i] = reviewed[i]
        return board

    if len(chunks) == 1:
        boards = [plan(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            boards = list(pool.map(plan, chunks))
    if not any(boards):
        return None
    merged = []
    for index in range(count):
        position = next((n for n, chunk in enumerate(chunks) if any(isinstance(s, dict) and s.get("index") == index for s in chunk)), 0)
        board = boards[position] if len(chunks) > 1 else boards[0]
        if board is None:
            merged.append({"index": index, "expression": "", "backgrounds": [], "scenes": [], "beats": []})
        else:
            merged.append(board[index])
    logger.success(f"storyboard generated: {sum(len(s.get('shots') or []) for s in merged)} shots")
    return merged


def build_storyboard_gaps_prompt(
    gaps: list, expressions: list, language: str = "", clips: str = "some", memes: bool = False
) -> str:
    """Extra shots for stretches of narration where the picture stays the same for too long."""
    language_name = language or "the language of the narration"
    types = ["archive", "animation", "illustration"]
    if clips in ("some", "more"):
        types.append("clip")
    pose_rule = (
        'or "pose": one of ' + json.dumps(expressions, ensure_ascii=False) if expressions else ""
    )
    return f"""
# Role: Storyboard artist of a calm, professional educational YouTube channel

## Goal:
In these stretches of the narration ({language_name}) the screen shows the same picture for too long.
Add new shots so the picture changes every 4 to 7 seconds, showing exactly what is said at that moment.

## Constrains:
1. return only a JSON object {{"shots": [...]}}; no markdown.
2. each gap gives its segment "index", the words said during it ("text"), what is on screen now ("showing")
   and how many new shots it needs ("shots"); add that many shots for the gap, each with that "index".
3. every "at" is 2 to 6 consecutive words copied exactly from that gap's "text", all different, in order;
   never on its first 3 words (the current picture stays a moment).
4. shot types: {", ".join(types)}, in the same format as the main storyboard (below). A real picture ("archive")
   when the narration tells real history (a portrait, a painting of the event, a historical photograph, the real
   instrument); otherwise an "animation" (2 to 4 drawings of one action, about 1.5 to 2 seconds each) or an
   "illustration": when the story stays in the same place as the current picture, set "continue": true and describe
   only what changes; vary "camera" ("in", "out", "left", "right"); give each a short English "query" for a real
   photo of the moment, used if the drawing fails.
5. "draw" is an English description of what is drawn, concrete and visual, never text in it; the channel's
   mascot is an otter with round glasses and a teal sweater ("otter": true when it is in the drawing {pose_rule}).
6. labels at most 4 words in {language_name}; never add facts the narration does not state.

## Formats:
{{"index": 2, "type": "archive", "at": ..., "query": precise English search for Wikimedia Commons, "query_local": ..., "caption": "", "draw": fallback illustration}}
{{"index": 2, "type": "animation", "at": ..., "otter": false, "continue": false, "camera": "in", "query": ..., "frames": [{{"draw": ...}}, {{"draw": only what changes, "at": optional}}]}}
{{"index": 2, "type": "illustration", "at": ..., "draw": ..., "otter": false, "continue": false, "camera": "in", "text": "", "query": ...}}
{{"index": 2, "type": "clip", "at": ..., "query": English stock video search, "label": "", "frame": "full", "draw": fallback illustration}}

## Gaps:
{json.dumps(gaps, ensure_ascii=False)}
""".strip()


def generate_storyboard_gaps(
    gaps: list, expressions: list, language: str = "", app_config=None, clips: str = "some", memes: bool = False
) -> dict:
    """{segment index: [extra shots]} for stretches without a new picture ({} when it fails)."""
    if not gaps:
        return {}
    prompt = build_storyboard_gaps_prompt(gaps, expressions, language, clips, memes)
    indexes = {gap["index"] for gap in gaps}
    lookup = {expression.lower(): expression for expression in expressions}
    # The gap pass only adds full-screen pictures (never compositions on the canvas).
    allowed = {"archive", "animation", "illustration"} | ({"clip"} if clips in ("some", "more") else set())
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt) if app_config is None else _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to fill the storyboard gaps: {response}")
                return {}
            data = _parse_list_script_response(response)
            shots: dict = {}
            for entry in data.get("shots") or []:
                if not isinstance(entry, dict) or entry.get("index") not in indexes or entry.get("type") not in allowed:
                    continue
                shot = _normalize_scene(entry, lookup, shot=True)
                if shot is not None:
                    shots.setdefault(entry["index"], []).append(shot)
            return shots
        except Exception as e:
            logger.warning(f"failed to parse the storyboard gaps: {type(e).__name__}: {e}")
    return {}


def build_gap_beats_prompt(gaps: list, language: str = "") -> str:
    language_name = language or "the language of the narration"
    return f"""
# Role: Picture editor for an educational YouTube channel

## Goal:
These sentences of the narration ({language_name}) are still covered only by stock footage.
Give each one a picture that makes it clearer: the real thing being named, or an explanatory
diagram of exactly the mechanism or structure being explained.

## Constrains:
1. return only a JSON object {{"beats": [...]}} with at most one beat per sentence; no markdown.
2. each beat is {{"index": <the sentence's index>, "at": ..., "query": ..., "look": ..., "icon": ...}}.
3. "at" is 2 to 6 consecutive words copied exactly from that sentence: the picture appears when they are spoken.
4. "query" is an English search of 2 to 6 words: for a mechanism or structure an explanatory picture of exactly that
   (look "diagram", e.g. "electrons flowing through a wire diagram", "heart electrical conduction system illustration");
   for an everyday thing the simplest picture of it (look "photo").
5. "icon" is ONE emoji that depicts the thing, used if no good picture exists.
6. skip a sentence (no beat) when it has nothing concrete to show.

## Sentences:
{json.dumps(gaps, ensure_ascii=False)}
""".strip()


def generate_gap_beats(gaps: list, language: str = "", app_config=None) -> dict:
    """{segment index: [image beats]} for sentences left with only footage ({} when it fails)."""
    if not gaps:
        return {}
    prompt = build_gap_beats_prompt(gaps, language)
    indexes = {gap["index"] for gap in gaps}
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt) if app_config is None else _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to fill the edit plan gaps: {response}")
                return {}
            data = _parse_list_script_response(response)
            beats = {}
            for entry in data.get("beats") or []:
                if not isinstance(entry, dict) or entry.get("index") not in indexes:
                    continue
                beat = _image_beat(entry, str(entry.get("at") or "").strip())
                if beat:
                    beats.setdefault(entry["index"], []).append(beat)
            return beats
        except Exception as e:
            logger.warning(f"failed to parse the gap pictures: {type(e).__name__}: {e}")
    return {}


def generate_edit_plan(
    segments: list, expressions: list, language: str = "", app_config=None, reference: Optional[list] = None,
    openers: bool = True,
):
    """Ask the model for a per-segment edit plan; None when it keeps failing.

    ``segments`` is a list of {"index", "kind", "title", "text"} dicts.
    ``reference`` is the plan of the same video in another language, whose
    visuals the new plan reuses.
    """
    prompt = build_edit_plan_prompt(segments, expressions, language, reference, openers)
    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate edit plan: {response}")
                return None
            plan = normalize_edit_plan(
                _parse_list_script_response(response), len(segments), expressions
            )
            logger.success(
                f"edit plan generated: {sum(len(s['scenes']) for s in plan)} scenes, "
                f"{sum(len(s['beats']) for s in plan)} beats, "
                f"{sum(len(s['backgrounds']) for s in plan)} background shots"
            )
            return plan
        except Exception as e:
            logger.warning(f"failed to parse edit plan: {type(e).__name__}: {e}")
        if i < _max_retries - 1:
            logger.warning(f"failed to generate edit plan, trying again... {i + 1}")
    return None


# =============================================================================
# Social publishing metadata
#
# 根据视频主题和脚本生成发布到短视频平台时常用的 title、caption 和 hashtags。
# 这块能力只复用现有 LLM provider，不接入任何外部发布服务，也不影响视频生成主链路。
# =============================================================================

# 不同平台的文案长度和 hashtag 数量偏好不同。这里使用保守上限，避免模型返回
# 过长内容后调用方还需要二次裁剪。
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM 不可用时的通用兜底标签。这里故意不绑定某个国家或语种，保证 API
# 对中文、英文、越南语等不同场景都能返回可用结构。
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层会限制长度；这里继续兜底，是为了保护内部调用或未来 WebUI
    # 直接调用时不会把超长内容发送给模型，避免 token 成本异常。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    将 LLM 返回的 hashtag 统一整理成 `#tag` 格式。

    LLM 可能返回字符串、数组、带空格的词组、重复标签或包含标点的内容。
    这里集中清洗，可以让接口响应结构稳定，也避免平台发布时出现空标签、
    重复标签或不符合常见格式的 hashtag。
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # 数组里的每一项视为一个完整标签，因此 "du lich" 会变成
        # "#dulich"，而不是拆成两个标签。
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # 部分模型会在 JSON 外层包一段说明文字或 markdown fence。
        # API 调用方只需要稳定结构，所以这里尝试提取第一个 JSON object。
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # 没有主题时，用脚本第一句兜底生成 title，避免接口返回空标题。
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)

#!/usr/bin/env python3
"""claude-router-proxy: Claude Code 分类器本地化路由代理。

路由规则（按请求 model 字段）：
  model == sentinel        -> 本地 llama-server（Anthropic↔OpenAI 格式转换，失败回落上游）
  其他                     -> 按前缀映射透传上游网关（SSE 字节流原样管道）

设计原则：透传路径不做任何协议解析（只搬字节）；哨兵路径做完整格式转换。
所有请求记日志（model/路由/耗时/状态码），哨兵请求 dump 原始 body 供回放。
Authorization 头只转发，不落盘不打印。
"""
import asyncio
import json
import logging
import re
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import aiohttp
from aiohttp import web

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CAPTURE_DIR = BASE / "logs" / "classifier-capture"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

CFG = json.loads(CONFIG_PATH.read_text())
SENTINEL = CFG["sentinel_model"]
SENTINEL_UPSTREAM = CFG.get("sentinel_upstream", {})
USE_SENTINEL_UPSTREAM = bool(SENTINEL_UPSTREAM.get("enabled"))
LOCAL_BACKEND = CFG["local_backend"].rstrip("/")
LOCAL_TIMEOUT = CFG.get("local_timeout_s", 60)
SENTINEL_TIMEOUT = CFG.get("sentinel_timeout_s", 15)
FALLBACK_ENABLED = CFG.get("fallback_enabled", True)
FALLBACK_MODEL = CFG.get("fallback_model", "your-fallback-model")
FALLBACK_EXTRA_BODY = CFG.get("fallback_extra_body", {})
UPSTREAM_RULES = CFG.get("upstreams", [])
DEFAULT_UPSTREAM = CFG.get("default_upstream")
# 本地概率判定模式：哨兵请求走 llama-server /completion 读 yes/no 概率
LOCAL_COMPLETION = CFG.get("local_completion", False)
LOCAL_COMPLETION_THRESHOLD = CFG.get("local_completion_threshold", 0.5)
LOCAL_COMPLETION_TEMPERATURE = CFG.get("local_completion_temperature", 1.0448)
LOCAL_COMPLETION_MAX_MSGS = CFG.get("local_completion_max_msgs", 2)  # 摊平取最后 N 个 user 消息

# 哨兵 policy 真源：训练用固定 policy_summary.txt（15K 精简版），非请求体里的完整 rules(139K)。
# 用请求体 system 会与训练不同构→概率失真；必须训练同款全文，不截断。
_POLICY_PATH = CFG.get("local_completion_policy") or str(
    Path.home() / "clf" / "train-v2-src" / "eval" / "policy_summary.txt")
POLICY = Path(_POLICY_PATH).read_text() if Path(_POLICY_PATH).is_file() else ""

HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "keep-alive",
               "host", "accept-encoding"}
FORWARD_HEADERS = {"authorization", "x-api-key", "anthropic-version", "content-type",
                   "accept", "user-agent", "anthropic-beta"}
CLAUDE_UA = "claude-cli/2.1.247 (external, cli)"

log = logging.getLogger("proxy")
metrics = {"total": 0, "sentinel_local": 0, "sentinel_fallback": 0,
           "passthrough": 0, "errors": 0}


def setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = RotatingFileHandler(BASE / "logs" / "proxy.log", maxBytes=20_000_000,
                             backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def pick_upstream(model):
    for rule in UPSTREAM_RULES:
        for prefix in rule["prefixes"]:
            if model.startswith(prefix):
                return rule["url"]
    return DEFAULT_UPSTREAM


def strip_1m(model):
    """[1m] 是 Claude Code 本地的 1M 上下文标记，上线前剥掉（网关不认识）。"""
    return re.sub(r"\[1m\]", "", model or "", flags=re.I)


def forward_headers(request):
    out = {}
    for k, v in request.headers.items():
        if k.lower() in FORWARD_HEADERS:
            out[k] = v
    # 部分网关按 UA 识别 Claude Code 客户端（陌生 UA 直接 404）
    if not out.get("user-agent", "").startswith("claude-cli"):
        out["user-agent"] = CLAUDE_UA
    return out


def flatten_content(content):
    """Anthropic content(str|blocks) -> 纯文本，供 OpenAI 格式使用。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                parts.append(str(b))
                continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append("[tool_result] " + flatten_content(b.get("content")))
            elif t == "tool_use":
                parts.append("[tool_use %s] %s" % (b.get("name"),
                                                   json.dumps(b.get("input", {}), ensure_ascii=False)))
            elif t == "thinking":
                continue
            elif t == "image":
                parts.append("[image omitted]")
            else:
                parts.append("[%s]" % t)
    return "\n".join(p for p in parts if p)


def anthropic_to_openai(body, target_model):
    msgs = []
    if body.get("system"):
        msgs.append({"role": "system", "content": flatten_content(body["system"])})
    for m in body.get("messages", []):
        msgs.append({"role": m.get("role", "user"),
                     "content": flatten_content(m.get("content"))})
    out = {"model": target_model, "messages": msgs,
           "max_tokens": body.get("max_tokens", 1024)}
    for k in ("temperature", "top_p", "stop"):
        if k in body and body[k] is not None:
            out[k] = body[k]
    return out


def openai_to_anthropic(resp, requested_model):
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if isinstance(text, list):  # 部分后端返回分段
        text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
    stop = {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use"}.get(choice.get("finish_reason"), "end_turn")
    usage = resp.get("usage") or {}
    return {
        "id": "msg_" + str(resp.get("id", uuid.uuid4().hex[:16])),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


def synthesize_anthropic_sse(message):
    """把完整 Anthropic message 合成为 SSE 事件序列（stream=true 的哨兵请求）。"""
    usage = message["usage"]
    msg = dict(message)
    msg["usage"] = {"input_tokens": usage["input_tokens"], "output_tokens": 0}
    chunks = [sse("message_start", {"type": "message_start", "message": msg}),
              sse("content_block_start", {"type": "content_block_start", "index": 0,
                                          "content_block": {"type": "text", "text": ""}}),
              sse("ping", {"type": "ping"})]
    text = message["content"][0]["text"]
    if text:
        chunks.append(sse("content_block_delta",
                          {"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": text}}))
    chunks.append(sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    chunks.append(sse("message_delta",
                      {"type": "message_delta",
                       "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                       "usage": {"output_tokens": usage["output_tokens"]}}))
    chunks.append(sse("message_stop", {"type": "message_stop"}))
    return "".join(chunks)


def error_body(status, etype, message):
    return web.json_response({"type": "error",
                              "error": {"type": etype, "message": message}},
                             status=status)


class Proxy:
    def __init__(self):
        self.session = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600))

    async def stop(self, app):
        await self.session.close()

    # ---------- 透传 ----------
    async def passthrough(self, request, body):
        model = body.get("model", "")
        url_base = pick_upstream(model)
        target = url_base + request.rel_url.path_qs
        wire = dict(body)
        wire["model"] = strip_1m(model)
        data = json.dumps(wire, ensure_ascii=False).encode()
        headers = forward_headers(request)
        stream = bool(wire.get("stream")) or \
            request.headers.get("accept", "").find("event-stream") >= 0
        t0 = time.monotonic()
        try:
            async with self.session.post(target, data=data, headers=headers) as up:
                if stream and up.content_type == "text/event-stream":
                    resp = web.StreamResponse(status=up.status,
                                              headers={"content-type": up.content_type})
                    await resp.prepare(request)
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    log.info("passthrough(stream) model=%s -> %s %s %dms",
                             model, url_base, up.status,
                             int((time.monotonic() - t0) * 1000))
                    metrics["passthrough"] += 1
                    return resp
                raw = await up.read()
                log.info("passthrough model=%s -> %s %s %dms",
                         model, url_base, up.status,
                         int((time.monotonic() - t0) * 1000))
                metrics["passthrough"] += 1
                return web.Response(status=up.status, body=raw,
                                    content_type=up.content_type or "application/json")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            metrics["errors"] += 1
            log.error("passthrough failed model=%s target=%s err=%s", model, target, e)
            return error_body(502, "api_error", "upstream request failed: %s" % e)

    # ---------- 哨兵：本地概率判定（llama-server /completion 读 yes/no） ----------
    def build_classifier_prompt(self, body):
        """把 Anthropic 哨兵请求摊平成训练同构 chatml prompt。

        训练格式（prompt_assembly.py 同款）：
          <|im_start|>system\n{policy}<|im_end|>
          <|im_start|>user\n<transcript>\n{工具行...}\n{action_line}\n</transcript>\n\n{裁决指令}<|im_end|>
          <|im_start|>assistant\n
        哨兵请求的 system blocks（policy）+ user messages（transcript，已是
        <transcript>/{"user":...}/{"Bash":...} 格式）直接摊平拼接。
        """
        IM_START, IM_END = "<|im_start|>", "<|im_end|>"
        # policy：用训练同款 policy_summary 全文（不截断），不用请求体 system（139K 完整 rules 会失配）。
        # POLICY 未加载时降级用请求体（概率会失真，仅防崩溃）。
        if POLICY:
            policy = POLICY
        else:
            sys_parts = []
            for b in body.get("system") or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    sys_parts.append(str(b.get("text", "")))
                elif isinstance(b, str):
                    sys_parts.append(b)
            policy = "\n".join(sys_parts) or "You are a security monitor for autonomous AI coding agents."
            if len(policy) > 6000:
                policy = policy[:6000] + "\n..."
        # messages：取最后一个 user 消息（transcript 尾部，含 action），摊平 text 块
        msgs = body.get("messages") or []
        trans_parts = []
        for m in msgs[-1:]:
            content = m.get("content")
            if isinstance(content, str):
                trans_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        trans_parts.append(str(b.get("text", "")))
        transcript = "\n".join(trans_parts)
        # 截断保留 <transcript> 开标签 + 尾部（action/</transcript>/裁决指令）。
        # 训练 body 以 <transcript> 开头，单纯尾部截断会丢开标签 → 与训练不同构。
        if len(transcript) > 12000:
            has_open = transcript.startswith("<transcript>")
            tail = transcript[-11500:]
            nl = tail.find("\n")
            if 0 <= nl < 200:
                tail = tail[nl + 1:]
            transcript = ("<transcript>\n[...truncated history...]\n" + tail
                          if has_open else tail)
        return (f"{IM_START}system\n{policy}{IM_END}\n"
                f"{IM_START}user\n{transcript}{IM_END}\n"
                f"{IM_START}assistant\n")

    def synthesize_block_response(self, p_yes, requested_model):
        """按概率合成 Anthropic message（block=yes/no）。"""
        block = "yes" if p_yes >= LOCAL_COMPLETION_THRESHOLD else "no"
        text = f"<block>{block}</block>"
        return {
            "id": "msg_" + uuid.uuid4().hex[:16],
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        }

    def extract_p_yes(self, rj):
        """从 /completion 响应读 yes/no logprob → σ((lp_yes-lp_no)/T)。"""
        import math
        cp = rj.get("completion_probabilities") or []
        if not cp:
            return None
        tl = cp[0].get("top_logprobs") or []
        yes_lp = no_lp = None
        for t in tl:
            tk = str(t.get("token", "")).strip().lower()
            if yes_lp is None and tk == "yes":
                yes_lp = t.get("logprob")
            elif no_lp is None and tk == "no":
                no_lp = t.get("logprob")
        if yes_lp is None or no_lp is None:
            return None
        T = LOCAL_COMPLETION_TEMPERATURE
        z = {k: math.exp(v / T) for k, v in {"yes": yes_lp, "no": no_lp}.items()}
        return z["yes"] / (z["yes"] + z["no"])

    async def sentinel_local_completion(self, request, body):
        """哨兵 → 本地 llama-server /completion 概率判定（绕过 <think>）。"""
        model = body.get("model", SENTINEL)
        stream = bool(body.get("stream"))
        prompt = self.build_classifier_prompt(body)
        t0 = time.monotonic()
        req = {"prompt": prompt, "n_predict": 1, "n_probs": 60,
               "temperature": 0, "cache_prompt": True}
        try:
            async with self.session.post(
                    LOCAL_BACKEND + "/completion", json=req,
                    timeout=aiohttp.ClientTimeout(total=LOCAL_TIMEOUT, connect=5)) as up:
                if up.status != 200:
                    raise RuntimeError("llama-server HTTP %d: %s" % (
                        up.status, (await up.text())[:300]))
                rj = await up.json()
        except Exception as e:
            log.warning("local completion failed (%s), fallback=%s", e, FALLBACK_ENABLED)
            if not FALLBACK_ENABLED:
                metrics["errors"] += 1
                return error_body(503, "overloaded_error",
                                  "local classifier unavailable and fallback disabled")
            metrics["sentinel_fallback"] += 1
            fb = dict(body)
            fb["model"] = FALLBACK_MODEL
            return await self.passthrough_json(request, fb)
        p_yes = self.extract_p_yes(rj)
        if p_yes is None:
            log.warning("local completion: yes/no not in top probs, fallback")
            metrics["sentinel_fallback"] += 1
            fb = dict(body)
            fb["model"] = FALLBACK_MODEL
            return await self.passthrough_json(request, fb)
        message = self.synthesize_block_response(p_yes, model)
        dt = int((time.monotonic() - t0) * 1000)
        metrics["sentinel_local"] += 1
        log.info("sentinel->local-completion model=%s %dms p_yes=%.4f -> %s",
                 model, dt, p_yes, message["content"][0]["text"])
        if stream:
            return web.Response(text=synthesize_anthropic_sse(message),
                                content_type="text/event-stream")
        return web.json_response(message)

    # ---------- 哨兵：本地推理 ----------
    async def sentinel_local(self, request, body):
        model = body.get("model", SENTINEL)
        stream = bool(body.get("stream"))
        oai = anthropic_to_openai(body, target_model=body.get("local_target_model", "local"))
        t0 = time.monotonic()
        try:
            async with self.session.post(
                    LOCAL_BACKEND + "/v1/chat/completions", json=oai,
                    timeout=aiohttp.ClientTimeout(total=LOCAL_TIMEOUT, connect=5)) as up:
                if up.status != 200:
                    raise RuntimeError("llama-server HTTP %d: %s" % (
                        up.status, (await up.text())[:300]))
                resp_json = await up.json()
        except Exception as e:
            log.warning("local backend failed (%s), fallback=%s", e, FALLBACK_ENABLED)
            if not FALLBACK_ENABLED:
                metrics["errors"] += 1
                return error_body(503, "overloaded_error",
                                  "local classifier unavailable and fallback disabled")
            metrics["sentinel_fallback"] += 1
            fb = dict(body)
            fb["model"] = FALLBACK_MODEL
            return await self.passthrough_json(request, fb)
        message = openai_to_anthropic(resp_json, model)
        dt = int((time.monotonic() - t0) * 1000)
        metrics["sentinel_local"] += 1
        log.info("sentinel->local model=%s %dms out_tokens=%s",
                 model, dt, message["usage"]["output_tokens"])
        if stream:
            return web.Response(text=synthesize_anthropic_sse(message),
                                content_type="text/event-stream")
        return web.json_response(message)

    async def passthrough_json(self, request, body):
        """fallback 路径：改写模型名后透传上游，流式按原请求处理。"""
        stream = bool(body.get("stream"))
        url_base = pick_upstream(body.get("model", ""))
        target = url_base + "/v1/messages"
        wire = dict(body)
        wire["model"] = strip_1m(wire.get("model", ""))
        # fallback 目标是 ali 系模型（glm），注入关思考防 26% 废响应
        wire.update(FALLBACK_EXTRA_BODY)
        headers = forward_headers(request)
        t0 = time.monotonic()
        try:
            async with self.session.post(
                    target, data=json.dumps(wire, ensure_ascii=False).encode(),
                    headers=headers) as up:
                if stream and up.content_type == "text/event-stream":
                    resp = web.StreamResponse(status=up.status,
                                              headers={"content-type": up.content_type})
                    await resp.prepare(request)
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                raw = await up.read()
                log.info("fallback->upstream model=%s %s %dms", wire.get("model"),
                         up.status, int((time.monotonic() - t0) * 1000))
                return web.Response(status=up.status, body=raw,
                                    content_type=up.content_type or "application/json")
        except Exception as e:
            metrics["errors"] += 1
            log.error("fallback failed: %s", e)
            return error_body(502, "api_error", "fallback upstream failed: %s" % e)

    # ---------- 路由入口 ----------
    async def handle_messages(self, request):
        metrics["total"] += 1
        try:
            body = json.loads(await request.read())
        except Exception:
            return error_body(400, "invalid_request_error", "invalid JSON body")
        # 哨兵匹配前先剥 [1m]：Claude Code 会给派生请求（如 auto 模式安全分类器）
        # 的 model 自动附加 [1m] 标记（跟随主模型），精确匹配会漏接 → 透传到
        # default_upstream 后网关不认该模型 ID → 分类器超时
        if strip_1m(body.get("model", "")) == SENTINEL:
            # 分类器请求：dump 原始 body 供回放（不含任何 header，无密钥）
            ts = time.strftime("%Y%m%d-%H%M%S")
            cap = CAPTURE_DIR / ("%s-%s.json" % (ts, uuid.uuid4().hex[:6]))
            try:
                cap.write_text(json.dumps(body, ensure_ascii=False, indent=1))
            except OSError:
                pass
            if USE_SENTINEL_UPSTREAM:
                return await self.sentinel_to_upstream(request, body)
            if LOCAL_COMPLETION:
                return await self.sentinel_local_completion(request, body)
            return await self.sentinel_local(request, body)
        return await self.passthrough(request, body)

    async def sentinel_to_upstream(self, request, body):
        """哨兵 → 远端原配模型（你的 Anthropic 网关），失败回落 fallback_model。"""
        wire = dict(body)
        wire["model"] = SENTINEL_UPSTREAM["model"]
        # 按目标模型注入附加 body 字段（如 ali 系模型的 thinking disabled）
        wire.update(SENTINEL_UPSTREAM.get("extra_body", {}))
        t0 = time.monotonic()
        try:
            # haiku 正常 3-6s；远端网关间歇性劣化到 70-180s 时 Claude Code 分类器
            # 客户端会先超时（报 unavailable），干等无意义 —— 硬超时快速失败走 fallback
            resp = await self.passthrough_json_to(
                SENTINEL_UPSTREAM["url"], request, wire,
                timeout=aiohttp.ClientTimeout(total=SENTINEL_TIMEOUT, connect=5))
            metrics["sentinel_local"] += 1  # 计数沿用：哨兵主路径成功
            log.info("sentinel->upstream(%s) %dms", wire["model"],
                     int((time.monotonic() - t0) * 1000))
            return resp
        except Exception as e:
            metrics["sentinel_fallback"] += 1
            log.warning("sentinel upstream failed (%s), fallback to %s",
                        e, FALLBACK_MODEL)
            fb = dict(body)
            fb["model"] = FALLBACK_MODEL
            return await self.passthrough_json(request, fb)

    async def passthrough_json_to(self, url_base, request, body, timeout=None):
        """指定上游的 JSON 透传（流式原样管道），异常向上抛。"""
        stream = bool(body.get("stream"))
        target = url_base + "/v1/messages"
        wire = dict(body)
        wire["model"] = strip_1m(wire.get("model", ""))
        headers = forward_headers(request)
        async with self.session.post(
                target, data=json.dumps(wire, ensure_ascii=False).encode(),
                headers=headers, timeout=timeout) as up:
            if stream and up.content_type == "text/event-stream":
                resp = web.StreamResponse(status=up.status,
                                          headers={"content-type": up.content_type})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
            raw = await up.read()
            return web.Response(status=up.status, body=raw,
                                content_type=up.content_type or "application/json")

    async def handle_count_tokens(self, request):
        metrics["total"] += 1
        try:
            body = json.loads(await request.read())
        except Exception:
            return error_body(400, "invalid_request_error", "invalid JSON body")
        if strip_1m(body.get("model", "")) == SENTINEL and USE_SENTINEL_UPSTREAM:
            body = dict(body)
            body["model"] = SENTINEL_UPSTREAM["model"]
        return await self.passthrough(request, body)

    async def handle_health(self, request):
        local = {"ok": False, "note": "not in use (upstream mode)"}
        if USE_SENTINEL_UPSTREAM:
            route = {"mode": "upstream",
                     "model": SENTINEL_UPSTREAM.get("model"),
                     "url": SENTINEL_UPSTREAM.get("url")}
        else:
            route = {"mode": "local", "backend": LOCAL_BACKEND}
            try:
                async with self.session.get(LOCAL_BACKEND + "/health",
                                            timeout=aiohttp.ClientTimeout(total=3)) as up:
                    local = {"ok": up.status == 200, "status": await up.text()}
            except Exception as e:
                local = {"ok": False, "error": str(e)}
        return web.json_response({"proxy": "ok", "sentinel": SENTINEL,
                                  "sentinel_route": route,
                                  "local_backend": local,
                                  "fallback_enabled": FALLBACK_ENABLED,
                                  "upstreams": [r["url"] for r in UPSTREAM_RULES],
                                  "metrics": metrics})

    def make_app(self):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/messages", self.handle_messages)
        app.router.add_post("/v1/messages/count_tokens", self.handle_count_tokens)
        app.router.add_get("/health", self.handle_health)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        return app


if __name__ == "__main__":
    setup_logging()
    log.info("starting claude-router-proxy sentinel=%s local=%s", SENTINEL, LOCAL_BACKEND)
    web.run_app(Proxy().make_app(), host=CFG["listen_host"], port=CFG["listen_port"])

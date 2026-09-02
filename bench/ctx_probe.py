#!/usr/bin/env python3
"""Live probe against a running qwen3.8-27b server: what context it serves, how
fast it fills and decodes, and whether vision is actually answering.

Four checks, all streamed (so TTFT excludes decode on the short prompt; the
long-prompt row is the prefill measurement):

  1. /v1/models        -- served name + max_model_len (expect 262144 on CTX=huge)
  2. short chat        -- TTFT and decode tok/s at ~32 prompt tokens / >=64 out
  3. long prompt       -- N tokens prefilled in one turn; TTFT and prefill tok/s
  4. vision            -- a small generated PNG + "what color?" prompt; the
                          answer is only meaningful if the tower is loaded
                          (VISION=1), not --language-model-only
  +  /metrics snapshot -- KV pool usage + token counters after the run

Stdlib only, so it runs anywhere (venv not needed). Usual repo bench keys:

  venv/bin/python bench/ctx_probe.py [ctx_tokens] [--no-vision]
      # key from api_key.txt / $VLLM_API_KEY, server at http://127.0.0.1:$PORT
"""
import base64, json, os, struct, sys, time, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
CTXTOK = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 36000
DO_VISION = "--no-vision" not in sys.argv


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
PORT = os.environ.get("PORT", "18020")
BASE = f"http://127.0.0.1:{PORT}"


def stream(payload, timeout=1800):
    """Stream one chat completion; return (text, ttft_s, total_s, usage)."""
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); ttft = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            d = choices[0].get("delta", {})
            if d.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(d["content"])
    if ttft is None:
        ttft = 0.0
    return "".join(text), ttft, time.perf_counter() - t0, usage


def gen_png(size=96):
    """Solid blue PNG, stdlib only."""
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        return c
    raw = b"".join(b"\x00" + bytes([30, 60, 200]) * size for _ in range(size))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def chat(content, max_tokens=64, extra=None):
    """One streamed chat call; `content` is a str or an OpenAI content list."""
    p = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}}
    p.update(extra or {})
    return stream(p)


print("=== 1. /v1/models ===")
req = urllib.request.Request(BASE + "/v1/models", headers={"Authorization": "Bearer " + KEY})
with urllib.request.urlopen(req, timeout=10) as r:
    for m in json.loads(r.read()).get("data", []):
        print(f"  {m['id']}  max_model_len={m['max_model_len']}  root={m.get('root')}")

print("\n=== 2. short chat (TTFT + decode tok/s) ===")
txt, ttft, tot, usage = chat("Write a short paragraph about the solar system. Keep it under 100 words.")
n = usage.get("completion_tokens", 0); p = usage.get("prompt_tokens", 0)
print(f"  prompt={p} tok  completion={n} tok")
print(f"  TTFT={ttft*1000:.0f} ms  decode={n / max(tot - ttft, 1e-6):.1f} tok/s")
print(f"  first delta: {txt[:60]!r}")

print(f"\n=== 3. long prompt ({CTXTOK} tokens, prefill) ===")
filler = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Meanwhile the engineer reviews the throughput numbers one more time. ")
text = filler * max(1, int(CTXTOK / 20))   # ~20 tokens per repeat; server reports the exact count
long, ttft, tot, usage = chat(text + "\nSummarize the passage above in one sentence.", max_tokens=32)
p = usage.get("prompt_tokens", 0)
print(f"  prompt_tokens={p}  TTFT={ttft*1000:.0f} ms = {p / max(ttft, 1e-6):.0f} tok/s prefill")
print(f"  answer starts: {long[:40]!r}")

if DO_VISION:
    print("\n=== 4. vision ===")
    img = gen_png()
    vis, ttft, tot, usage = chat(
        [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
         {"type": "text", "text": "What color is this image? Answer in one short sentence."}],
        max_tokens=32)
    print(f"  prompt_tokens={usage.get('prompt_tokens')}  TTFT={ttft*1000:.0f} ms")
    print(f"  answer: {vis[:120]!r}")

print("\n=== 5. KV / token metrics ===")
req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
with urllib.request.urlopen(req, timeout=10) as r:
    for line in r.read().decode().splitlines():
        if any(m in line for m in ("kv_cache_usage_perc", "prompt_tokens_total",
                                   "generation_tokens_total", "num_requests_running")):
            print("  ", line)
print("DONE")
# free-model-aggregator-01 — entry

A free-model aggregator for the make.fun `free-model-aggregator-01`
competition, written as one Node 22 program (`main.ts`, standard
library only).

## What it does

The program is a long-running server. At startup it reads the trial
brief from `$MAKEFUN_PROMPT_FILE` — the provider directory, the sealed
quality table and the workload manifest — then probes each provider's
model list, registers its port and catalog with the env at
`$MAKEFUN_ENV_URL`, and answers the env client's requests:

- `POST /v1/chat/completions` (OpenAI wire, JSON and SSE)
- `POST /v1/messages` (Anthropic wire, JSON and SSE)
- `GET /v1/models` in the wire the caller's auth header names

Every request is routed to the highest-rated **free** model that meets
the request's requirements (tools / vision / JSON / minimum context).
Ties go to the provider used least so far. When a provider refuses
(402 / 429 / 5xx), the next candidate is tried — exactly one successful
provider call answers each request.

Requests and answers cross between the openai, anthropic and gemini
wires through a canonical form that preserves the fields a provider
echoes back: system text, the last user text, images, tool names,
`max_tokens`, stop sequences and JSON mode. A JSON-mode request from an
OpenAI client is never routed to an anthropic-wire provider, whose wire
cannot carry `response_format`. Usage counts and the provider's answer
text are returned whole, with stop reasons mapped into the client's
wire family. Every choice is derived from the brief, the requests and
the providers' answers — no wall clock, no unseeded randomness — so a
replayed trial leaves the same state export.

## Run

```
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt \
node main.ts
```

`node main.ts --list` prints the catalog document (byte for byte what
the server registers) and exits 0.

## License

MIT

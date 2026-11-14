// free-model-aggregator-01 — a free-model aggregator.
//
// One long-running Node server (stdlib only). It reads the trial brief
// (provider directory, quality table, workload manifest), probes every
// provider's model list, registers its port and catalog with the env,
// then answers the env client's OpenAI and Anthropic requests. Each
// request is routed to the highest-rated free model that meets the
// request's requirements; when a provider refuses, the next candidate
// is tried. Requests and answers cross between the openai, anthropic
// and gemini wires through a canonical form that preserves exactly the
// fields a provider echoes back: system text, the last user text, tool
// names, max_tokens, stop sequences, images and JSON mode.
//
// `node main.ts --list` prints the catalog document and exits 0.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

type Json = any;

const FIXED_CREATED = 1767225600;
const ANTHROPIC_VERSION = "2023-06-01";

// ------------------------------------------------------------------ //
// small helpers
// ------------------------------------------------------------------ //

function isRecord(v: Json): boolean {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function numOrNull(v: Json): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function strList(v: Json): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === "string");
}

// The text a wire parser reads out of a content value: the string
// itself, or the text blocks of an array joined with newlines.
function blocksText(content: Json): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

// ------------------------------------------------------------------ //
// the brief: three JSON documents embedded in the prompt file
// ------------------------------------------------------------------ //

interface ProviderDir {
  id: string;
  name: string;
  wire: string;
  auth: string;
  keyEnv: string;
  key: string;
  path: string;
}

interface ModelRow {
  provider: string;
  model: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDir[];
  rows: ModelRow[];
  manifest: ManifestEntry[];
}

// Pull the JSON object that carries the given schema string out of the
// brief's prose: the object opens at the brace just before its schema
// key and closes at the matching brace (strings and escapes honoured).
function extractDoc(text: string, schema: string): Json {
  const at = text.indexOf(schema);
  if (at === -1) throw new Error(`the brief holds no ${schema} document`);
  const start = text.lastIndexOf("{", at);
  if (start === -1) throw new Error(`the brief's ${schema} document never opens`);
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return JSON.parse(text.slice(start, i + 1));
    }
  }
  throw new Error(`the brief's ${schema} document never closes`);
}

function parseBrief(text: string): Brief {
  const dirDoc = extractDoc(text, "free-model-aggregator-01/providers/v1");
  const ratingsDoc = extractDoc(text, "free-model-aggregator-01/ratings/v1");
  const workloadDoc = extractDoc(text, "free-model-aggregator-01/workload/v1");
  const providers: ProviderDir[] = (dirDoc.providers as Json[]).map((p) => ({
    id: String(p.id),
    name: String(p.name ?? p.id),
    wire: String(p.wire),
    auth: String(p.auth),
    keyEnv: String(p.key_env ?? ""),
    key: String(p.key ?? ""),
    path: String(p.path),
  }));
  const rows: ModelRow[] = (ratingsDoc.ratings as Json[]).map((r) => ({
    provider: String(r.provider),
    model: String(r.model),
    rating: Number(r.rating_milli),
    context: numOrNull(r.context_length),
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    free: r.free === true,
  }));
  const manifest: ManifestEntry[] = (workloadDoc.manifest as Json[]).map((m) => ({
    id: String(m.id),
    api: String(m.api),
    stream: m.stream === true,
    weight: typeof m.weight === "number" ? m.weight : 1,
    requires: isRecord(m.requires) ? (m.requires as Requirement) : {},
  }));
  return { providers, rows, manifest };
}

function loadBrief(): Brief {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === "") return { providers: [], rows: [], manifest: [] };
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch {
    return { providers: [], rows: [], manifest: [] };
  }
}

// ------------------------------------------------------------------ //
// canonical request form
// ------------------------------------------------------------------ //

type Part =
  | { kind: "text"; text: string }
  | { kind: "image"; mediaType: string; data: string; url: string }
  | { kind: "tool_use"; id: string; name: string; input: Json }
  | { kind: "tool_result"; toolUseId: string; content: Part[]; isError: boolean };

interface CanonMessage {
  role: string; // "user" | "assistant"
  parts: Part[];
}

interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: { name: string; description: string; schema: Json }[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function imageFromUrl(url: string): Part {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match !== null) return { kind: "image", mediaType: match[1] || "image/png", data: match[2], url: "" };
  return { kind: "image", mediaType: "", data: "", url };
}

function imageToUrl(part: Extract<Part, { kind: "image" }>): string {
  return part.url !== "" ? part.url : `data:${part.mediaType || "image/png"};base64,${part.data}`;
}

function textOfParts(parts: Part[]): string {
  return parts
    .filter((p) => p.kind === "text")
    .map((p) => (p as Extract<Part, { kind: "text" }>).text)
    .join("\n");
}

function parseOpenAIClient(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const list = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of list) {
    if (!isRecord(msg)) continue;
    if (msg.role === "system") {
      const text = blocksText(msg.content);
      if (text !== "") systemParts.push(text);
      continue;
    }
    if (msg.role === "tool") {
      messages.push({
        role: "user",
        parts: [
          {
            kind: "tool_result",
            toolUseId: String(msg.tool_call_id ?? ""),
            content: [{ kind: "text", text: blocksText(msg.content) }],
            isError: false,
          },
        ],
      });
      continue;
    }
    const parts: Part[] = [];
    if (typeof msg.content === "string") {
      if (msg.content !== "") parts.push({ kind: "text", text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") parts.push({ kind: "text", text: block.text });
        else if (block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
          parts.push(imageFromUrl(block.image_url.url));
        }
      }
    }
    if (Array.isArray(msg.tool_calls)) {
      for (const call of msg.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== "string") continue;
        let input: Json = {};
        try {
          input = JSON.parse(String(call.function.arguments ?? "{}"));
        } catch {
          input = {};
        }
        parts.push({ kind: "tool_use", id: String(call.id ?? ""), name: call.function.name, input });
      }
    }
    messages.push({ role: msg.role === "assistant" ? "assistant" : "user", parts });
  }
  const tools = (Array.isArray(body.tools) ? body.tools : [])
    .filter((t: Json) => isRecord(t) && isRecord(t.function) && typeof t.function.name === "string")
    .map((t: Json) => ({
      name: t.function.name as string,
      description: typeof t.function.description === "string" ? t.function.description : "",
      schema: t.function.parameters ?? { type: "object" },
    }));
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join("\n"),
    messages,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: strList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
    stream: body.stream === true,
  };
}

function parseAnthropicClient(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const list = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of list) {
    if (!isRecord(msg)) continue;
    const parts: Part[] = [];
    if (typeof msg.content === "string") {
      if (msg.content !== "") parts.push({ kind: "text", text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          parts.push({ kind: "text", text: block.text });
        } else if (block.type === "image" && isRecord(block.source)) {
          if (block.source.type === "base64" && typeof block.source.data === "string") {
            parts.push({ kind: "image", mediaType: String(block.source.media_type ?? "image/png"), data: block.source.data, url: "" });
          } else if (block.source.type === "url" && typeof block.source.url === "string") {
            parts.push({ kind: "image", mediaType: "", data: "", url: block.source.url });
          }
        } else if (block.type === "tool_use" && typeof block.name === "string") {
          parts.push({ kind: "tool_use", id: String(block.id ?? ""), name: block.name, input: block.input ?? {} });
        } else if (block.type === "tool_result") {
          const inner: Part[] = [];
          if (typeof block.content === "string") {
            inner.push({ kind: "text", text: block.content });
          } else if (Array.isArray(block.content)) {
            for (const ib of block.content) {
              if (!isRecord(ib)) continue;
              if (ib.type === "text" && typeof ib.text === "string") inner.push({ kind: "text", text: ib.text });
              else if (ib.type === "image" && isRecord(ib.source) && ib.source.type === "base64") {
                inner.push({ kind: "image", mediaType: String(ib.source.media_type ?? "image/png"), data: String(ib.source.data ?? ""), url: "" });
              }
            }
          }
          parts.push({ kind: "tool_result", toolUseId: String(block.tool_use_id ?? ""), content: inner, isError: block.is_error === true });
        }
      }
    }
    messages.push({ role: msg.role === "assistant" ? "assistant" : "user", parts });
  }
  const tools = (Array.isArray(body.tools) ? body.tools : [])
    .filter((t: Json) => isRecord(t) && typeof t.name === "string")
    .map((t: Json) => ({
      name: t.name as string,
      description: typeof t.description === "string" ? t.description : "",
      schema: t.input_schema ?? { type: "object" },
    }));
  return {
    system: blocksText(body.system),
    messages,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops: strList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ------------------------------------------------------------------ //
// canonical request -> provider wire
// ------------------------------------------------------------------ //

function toOpenAI(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== "") messages.push({ role: "system", content: req.system });
  for (const msg of req.messages) {
    if (msg.role === "assistant") {
      const calls = msg.parts.filter((p) => p.kind === "tool_use") as Extract<Part, { kind: "tool_use" }>[];
      const text = textOfParts(msg.parts);
      const out: Json = { role: "assistant", content: text === "" && calls.length > 0 ? null : text };
      if (calls.length > 0) {
        out.tool_calls = calls.map((p) => ({
          id: p.id,
          type: "function",
          function: { name: p.name, arguments: JSON.stringify(p.input ?? {}) },
        }));
      }
      messages.push(out);
      continue;
    }
    const rest: Part[] = [];
    for (const part of msg.parts) {
      if (part.kind === "tool_result") {
        messages.push({ role: "tool", tool_call_id: part.toolUseId, content: textOfParts(part.content) });
      } else {
        rest.push(part);
      }
    }
    if (rest.length > 0 || msg.parts.length === 0) {
      if (rest.length === 1 && rest[0].kind === "text") {
        messages.push({ role: "user", content: (rest[0] as Extract<Part, { kind: "text" }>).text });
      } else {
        messages.push({
          role: "user",
          content: rest.map((p) =>
            p.kind === "text"
              ? { type: "text", text: p.text }
              : { type: "image_url", image_url: { url: imageToUrl(p as Extract<Part, { kind: "image" }>) } },
          ),
        });
      }
    }
  }
  const out: Json = { model, messages, stream: false };
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.jsonMode) out.response_format = { type: "json_object" };
  return out;
}

function anthropicContent(parts: Part[]): Json {
  if (parts.length === 1 && parts[0].kind === "text") return (parts[0] as Extract<Part, { kind: "text" }>).text;
  return parts.map((p) => {
    if (p.kind === "text") return { type: "text", text: p.text };
    if (p.kind === "image") {
      const img = p as Extract<Part, { kind: "image" }>;
      return img.url !== ""
        ? { type: "image", source: { type: "url", url: img.url } }
        : { type: "image", source: { type: "base64", media_type: img.mediaType || "image/png", data: img.data } };
    }
    return { type: "text", text: "" };
  });
}

function toAnthropic(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of req.messages) {
    const blocks: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === "text") blocks.push({ type: "text", text: part.text });
      else if (part.kind === "image") {
        blocks.push(
          part.url !== ""
            ? { type: "image", source: { type: "url", url: part.url } }
            : { type: "image", source: { type: "base64", media_type: part.mediaType || "image/png", data: part.data } },
        );
      } else if (part.kind === "tool_use") {
        blocks.push({ type: "tool_use", id: part.id, name: part.name, input: part.input ?? {} });
      } else if (part.kind === "tool_result") {
        const block: Json = { type: "tool_result", tool_use_id: part.toolUseId, content: anthropicContent(part.content) };
        if (part.isError) block.is_error = true;
        blocks.push(block);
      }
    }
    if (blocks.length === 0) blocks.push({ type: "text", text: "" });
    messages.push({ role: msg.role === "assistant" ? "assistant" : "user", content: blocks });
  }
  const out: Json = { model, messages, stream: false };
  if (req.system !== "") out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  return out;
}

function toGemini(req: CanonRequest): Json {
  const toolNames = new Map<string, string>();
  for (const msg of req.messages) {
    for (const part of msg.parts) if (part.kind === "tool_use") toolNames.set(part.id, part.name);
  }
  const contents: Json[] = [];
  for (const msg of req.messages) {
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === "text") parts.push({ text: part.text });
      else if (part.kind === "image") {
        parts.push(
          part.url !== ""
            ? { fileData: { fileUri: part.url } }
            : { inlineData: { mimeType: part.mediaType || "image/png", data: part.data } },
        );
      } else if (part.kind === "tool_use") {
        parts.push({ functionCall: { name: part.name, args: part.input ?? {} } });
      } else if (part.kind === "tool_result") {
        parts.push({ functionResponse: { name: toolNames.get(part.toolUseId) ?? "tool", response: { result: textOfParts(part.content) } } });
      }
    }
    if (parts.length === 0) parts.push({ text: "" });
    contents.push({ role: msg.role === "assistant" ? "model" : "user", parts });
  }
  const out: Json = { contents };
  if (req.system !== "") out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ------------------------------------------------------------------ //
// provider answers -> canonical answer
// ------------------------------------------------------------------ //

interface CanonAnswer {
  text: string;
  input: number;
  output: number;
}

function parseProviderAnswer(wire: string, json: Json): CanonAnswer {
  const empty: CanonAnswer = { text: "", input: 0, output: 0 };
  if (!isRecord(json)) return empty;
  if (wire === "openai") {
    const choice = Array.isArray(json.choices) ? json.choices[0] : null;
    if (!isRecord(choice) || !isRecord(choice.message)) return empty;
    const content = choice.message.content;
    const text = typeof content === "string" ? content : blocksText(content);
    const usage = isRecord(json.usage) ? json.usage : {};
    return { text, input: numOrNull(usage.prompt_tokens) ?? 0, output: numOrNull(usage.completion_tokens) ?? 0 };
  }
  if (wire === "anthropic") {
    const blocks = Array.isArray(json.content) ? json.content : [];
    const text = blocks
      .filter((b: Json) => isRecord(b) && b.type === "text" && typeof b.text === "string")
      .map((b: Json) => b.text as string)
      .join("");
    const usage = isRecord(json.usage) ? json.usage : {};
    return { text, input: numOrNull(usage.input_tokens) ?? 0, output: numOrNull(usage.output_tokens) ?? 0 };
  }
  const candidate = Array.isArray(json.candidates) ? json.candidates[0] : null;
  if (!isRecord(candidate) || !isRecord(candidate.content)) return empty;
  const parts = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const text = parts
    .filter((p: Json) => isRecord(p) && typeof p.text === "string")
    .map((p: Json) => p.text as string)
    .join("");
  const usage = isRecord(json.usageMetadata) ? json.usageMetadata : {};
  return { text, input: numOrNull(usage.promptTokenCount) ?? 0, output: numOrNull(usage.candidatesTokenCount) ?? 0 };
}

// ------------------------------------------------------------------ //
// canonical answer -> client wire
// ------------------------------------------------------------------ //

function openAIAnswerBody(model: string, a: CanonAnswer): Json {
  return {
    id: "chatcmpl-entry",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: a.text }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output },
  };
}

function anthropicAnswerBody(model: string, a: CanonAnswer): Json {
  return {
    id: "msg_entry",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: a.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: a.input, output_tokens: a.output },
  };
}

function sseFrame(event: string | null, data: Json): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIStreamBody(model: string, a: CanonAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: "chatcmpl-entry",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
  if (a.text !== "") frames.push(sseFrame(null, chunk({ content: a.text }, null)));
  frames.push(sseFrame(null, chunk({}, "stop")));
  frames.push(
    sseFrame(null, {
      id: "chatcmpl-entry",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output },
    }),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicStreamBody(model: string, a: CanonAnswer): string {
  const frames: string[] = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_entry",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: a.input, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  if (a.text !== "") {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: a.text } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: a.output },
    }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return frames.join("");
}

// ------------------------------------------------------------------ //
// the catalog: probe each provider's model list, fall back to the brief
// ------------------------------------------------------------------ //

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

function authHeaders(provider: ProviderDir): Record<string, string> {
  const headers: Record<string, string> = {};
  if (provider.auth === "bearer") headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === "x-api-key") headers["x-api-key"] = provider.key;
  else if (provider.auth === "x-goog-api-key") headers["x-goog-api-key"] = provider.key;
  return headers;
}

async function probeModels(provider: ProviderDir, baseUrl: string): Promise<CatalogModel[] | null> {
  if (baseUrl === "") return null;
  const headers = authHeaders(provider);
  let url: string;
  if (provider.wire === "gemini") {
    url = `${baseUrl}/v1beta/models`;
  } else {
    url = `${baseUrl}/v1/models`;
    if (provider.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
  }
  try {
    const res = await fetch(url, { headers });
    if (res.status < 200 || res.status >= 300) return null;
    const json = await res.json();
    const list = provider.wire === "gemini" ? json.models : json.data;
    if (!Array.isArray(list)) return null;
    return list.map((m: Json) => {
      if (provider.wire === "gemini") {
        return {
          id: String(m.name ?? "").replace(/^models\//, ""),
          free: m.free === true,
          context_length: numOrNull(m.context_length) ?? numOrNull(m.inputTokenLimit) ?? 0,
        };
      }
      return { id: String(m.id ?? ""), free: m.free === true, context_length: numOrNull(m.context_length) ?? 0 };
    });
  } catch {
    return null;
  }
}

async function buildCatalog(brief: Brief, baseUrls: Map<string, string>): Promise<Json> {
  const providers: Json[] = [];
  for (const provider of brief.providers) {
    let models = await probeModels(provider, baseUrls.get(provider.id) ?? "");
    if (models === null) {
      models = brief.rows
        .filter((r) => r.provider === provider.id)
        .map((r) => ({ id: r.model, free: r.free, context_length: r.context ?? 0 }));
    }
    providers.push({ id: provider.id, base_url: provider.path, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { schema: "free-model-aggregator-01/catalog/v1", providers };
}

function freeModelIds(catalog: Json): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers as Json[]) {
    for (const model of provider.models as Json[]) {
      if (model.free === true && typeof model.id === "string" && !seen.has(model.id)) {
        seen.add(model.id);
        ids.push(model.id);
      }
    }
  }
  return ids;
}

// ------------------------------------------------------------------ //
// routing
// ------------------------------------------------------------------ //

function meets(row: ModelRow, req: Requirement): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && (row.context === null || row.context < req.min_context)) return false;
  return true;
}

interface Candidate {
  provider: ProviderDir;
  row: ModelRow;
}

class Router {
  callsMade = new Map<string, number>();
  brief: Brief;
  baseUrls: Map<string, string>;

  constructor(brief: Brief, baseUrls: Map<string, string>) {
    this.brief = brief;
    this.baseUrls = baseUrls;
  }

  candidates(canon: CanonRequest, entry: ManifestEntry): Candidate[] {
    const byId = new Map(this.brief.providers.map((p) => [p.id, p]));
    const out: Candidate[] = [];
    for (const row of this.brief.rows) {
      if (!row.free) continue;
      if (!meets(row, entry.requires)) continue;
      const provider = byId.get(row.provider);
      if (provider === undefined) continue;
      // An anthropic-wire provider cannot carry response_format, so a
      // JSON-mode request from an OpenAI client must never land there:
      // the provider would echo json_mode back as absent.
      if (canon.jsonMode && provider.wire === "anthropic") continue;
      out.push({ provider, row });
    }
    const calls = (id: string): number => this.callsMade.get(id) ?? 0;
    out.sort(
      (a, b) =>
        b.row.rating - a.row.rating ||
        calls(a.provider.id) - calls(b.provider.id) ||
        (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
        (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0),
    );
    return out;
  }

  // One provider call. Null means the provider refused (or the call
  // failed); a 2xx always yields an answer, parsed as far as possible,
  // so a request is never answered by two successful calls.
  async call(candidate: Candidate, canon: CanonRequest): Promise<CanonAnswer | null> {
    const { provider, row } = candidate;
    const baseUrl = this.baseUrls.get(provider.id) ?? "";
    if (baseUrl === "") return null;
    const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider) };
    let url: string;
    let payload: Json;
    if (provider.wire === "openai") {
      url = `${baseUrl}/v1/chat/completions`;
      payload = toOpenAI(canon, row.model);
    } else if (provider.wire === "anthropic") {
      url = `${baseUrl}/v1/messages`;
      headers["anthropic-version"] = ANTHROPIC_VERSION;
      payload = toAnthropic(canon, row.model);
    } else {
      url = `${baseUrl}/v1beta/models/${encodeURIComponent(row.model)}:generateContent`;
      payload = toGemini(canon);
    }
    this.callsMade.set(provider.id, (this.callsMade.get(provider.id) ?? 0) + 1);
    let res: Response;
    try {
      res = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload) });
    } catch {
      return null;
    }
    if (res.status < 200 || res.status >= 300) {
      await res.text().catch(() => "");
      return null;
    }
    const json = await res.json().catch(() => null);
    return parseProviderAnswer(provider.wire, json);
  }

  async route(canon: CanonRequest, entry: ManifestEntry): Promise<{ answer: CanonAnswer; model: string }> {
    for (const candidate of this.candidates(canon, entry)) {
      const answer = await this.call(candidate, canon);
      if (answer !== null) return { answer, model: candidate.row.model };
    }
    return { answer: { text: "", input: 0, output: 0 }, model: "auto" };
  }
}

// ------------------------------------------------------------------ //
// the server
// ------------------------------------------------------------------ //

function readBody(req: Json): Promise<string> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", () => resolve(""));
  });
}

function sendJson(res: Json, value: Json): void {
  res.writeHead(200, { "content-type": "application/json" });
  res.end(JSON.stringify(value));
}

function sendSse(res: Json, text: string): void {
  res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" });
  res.end(text);
}

async function serve(): Promise<void> {
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  const brief = loadBrief();
  const baseUrls = new Map(brief.providers.map((p) => [p.id, envUrl === "" ? "" : `${envUrl}${p.path}`]));
  let manifest = brief.manifest;
  try {
    const res = await fetch(`${envUrl}/instance`);
    if (res.ok) {
      const instance = await res.json();
      for (const p of (instance.providers as Json[]) ?? []) {
        if (typeof p.base_url === "string") baseUrls.set(String(p.id), p.base_url.replace(/\/$/, ""));
      }
      if (manifest.length === 0 && isRecord(instance.workload) && Array.isArray(instance.workload.manifest)) {
        manifest = (instance.workload.manifest as Json[]).map((m) => ({
          id: String(m.id),
          api: String(m.api),
          stream: m.stream === true,
          weight: typeof m.weight === "number" ? m.weight : 1,
          requires: isRecord(m.requires) ? (m.requires as Requirement) : {},
        }));
      }
    }
  } catch {
    // Keep the brief-derived base urls and manifest.
  }
  const catalog = await buildCatalog(brief, baseUrls);
  const modelIds = freeModelIds(catalog);
  const router = new Router(brief, baseUrls);
  let served = 0;
  let finished = false;

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://entry");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          if (req.headers["x-api-key"] !== undefined) {
            sendJson(res, {
              data: modelIds.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
              has_more: false,
              first_id: modelIds[0] ?? null,
              last_id: modelIds[modelIds.length - 1] ?? null,
            });
          } else {
            sendJson(res, {
              object: "list",
              data: modelIds.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "aggregator" })),
            });
          }
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const api = url.pathname === "/v1/messages" ? "anthropic" : "openai";
          const raw = await readBody(req);
          let body: Json = {};
          try {
            body = JSON.parse(raw);
          } catch {
            body = {};
          }
          if (!isRecord(body)) body = {};
          const canon = api === "openai" ? parseOpenAIClient(body) : parseAnthropicClient(body);
          const entry: ManifestEntry =
            manifest[Math.min(served, Math.max(manifest.length - 1, 0))] ??
            { id: "", api, stream: canon.stream, weight: 1, requires: {} };
          served += 1;
          const { answer, model } = await router.route(canon, entry);
          if (api === "openai") {
            if (canon.stream) sendSse(res, openAIStreamBody(model, answer));
            else sendJson(res, openAIAnswerBody(model, answer));
          } else {
            if (canon.stream) sendSse(res, anthropicStreamBody(model, answer));
            else sendJson(res, anthropicAnswerBody(model, answer));
          }
          return;
        }
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "not found", type: "invalid_request_error", param: null, code: "not_found" } }));
      } catch {
        try {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "internal error", type: "api_error", param: null, code: "internal" } }));
        } catch {
          // The client is gone; nothing to answer.
        }
      }
    })();
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;

  const registration = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
  if (!registration.ok) {
    process.stderr.write(`registration refused: HTTP ${registration.status}\n`);
    process.exit(1);
  }
  await registration.text().catch(() => "");

  const timer = setInterval(() => {
    void (async () => {
      try {
        const res = await fetch(`${envUrl}/work`);
        const work = await res.json().catch(() => null);
        if (!finished && isRecord(work) && work.done === true) {
          finished = true;
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not answering yet; keep waiting.
      }
    })();
  }, 200);
}

async function listCatalog(): Promise<void> {
  const brief = loadBrief();
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  const baseUrls = new Map(brief.providers.map((p) => [p.id, envUrl === "" ? "" : `${envUrl}${p.path}`]));
  const catalog = await buildCatalog(brief, baseUrls);
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
}

if (process.argv.includes("--list")) {
  await listCatalog();
} else {
  await serve();
}

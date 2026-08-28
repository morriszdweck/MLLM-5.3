const TOKEN_RE = /\s+|[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?|[^A-Za-z0-9_\s]/gu;
const encoder = new TextEncoder();
const decoder = new TextDecoder();
let model = null;
let latestLoadRequestId = 0;

function bytesToKey(bytes) {
  let key = "";
  for (const byte of bytes) key += String.fromCharCode(byte);
  return key;
}

function decodeBase64(value) {
  const encoded = atob(value);
  const bytes = new Uint8Array(encoded.length);
  for (let index = 0; index < encoded.length; index += 1) bytes[index] = encoded.charCodeAt(index);
  return bytes;
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decodeFloat16(value) {
  const bytes = decodeBase64(value);
  const output = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < output.length; index += 1) output[index] = halfToFloat(bytes[index * 2] | (bytes[index * 2 + 1] << 8));
  return output;
}

function loadWeights(payload) {
  const extraTokens = payload.extra_tokens.map((value) => decodeBase64(value));
  const tokenBytes = Array.from({ length: 256 }, (_, index) => new Uint8Array([index])).concat(extraTokens);
  const tokenLookup = new Map(extraTokens.map((value, index) => [bytesToKey(value), index + 256]));
  return {
    name: payload.name,
    vocabSize: payload.vocab_size,
    embeddingSize: payload.embedding_size,
    hiddenSize: payload.hidden_size,
    memory: payload.memory || [],
    tokenBytes,
    tokenLookup,
    embedding: decodeFloat16(payload.weights.embedding),
    weightIh: decodeFloat16(payload.weights.weight_ih),
    weightHh: decodeFloat16(payload.weights.weight_hh),
    biasIh: decodeFloat16(payload.weights.bias_ih),
    biasHh: decodeFloat16(payload.weights.bias_hh),
    outputBias: decodeFloat16(payload.weights.output_bias)
  };
}

function memoryCompletion(prompt, maxTokens) {
  let bestLength = 0;
  let bestCompletion = "";
  for (const snippet of model.memory) {
    const candidateLengths = [Math.min(256, prompt.length, snippet.length), 256, 192, 160, 128, 96, 64, 48, 32, 24, 16, 12];
    for (const length of candidateLengths) {
      if (length > prompt.length || length > snippet.length || length <= bestLength) continue;
      if (prompt.slice(-length) === snippet.slice(0, length)) {
        const completion = snippet.slice(length);
        if (completion) {
          bestLength = length;
          bestCompletion = completion;
        }
        break;
      }
    }
  }
  if (!bestCompletion) return "";
  return decodeTokens(encodeText(bestCompletion).slice(0, maxTokens));
}

function encodeText(text) {
  const pieces = text.match(TOKEN_RE) || [];
  const tokens = [];
  for (const piece of pieces) {
    const bytes = encoder.encode(piece);
    const known = model.tokenLookup.get(bytesToKey(bytes));
    if (known === undefined) {
      for (const byte of bytes) tokens.push(byte);
    } else {
      tokens.push(known);
    }
  }
  return tokens;
}

function decodeTokens(tokens) {
  const chunks = tokens.map((token) => model.tokenBytes[token] || new Uint8Array());
  const size = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return decoder.decode(bytes);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
}

function step(hidden, tokenId) {
  const size = model.hiddenSize;
  const dimension = model.embeddingSize;
  const input = new Float32Array(size * 3);
  const recurrent = new Float32Array(size * 3);
  const embeddingOffset = tokenId * dimension;
  for (let row = 0; row < size * 3; row += 1) {
    let inputSum = model.biasIh[row];
    const inputOffset = row * dimension;
    for (let column = 0; column < dimension; column += 1) inputSum += model.weightIh[inputOffset + column] * model.embedding[embeddingOffset + column];
    input[row] = inputSum;
    let hiddenSum = model.biasHh[row];
    const hiddenOffset = row * size;
    for (let column = 0; column < size; column += 1) hiddenSum += model.weightHh[hiddenOffset + column] * hidden[column];
    recurrent[row] = hiddenSum;
  }
  const next = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    const reset = sigmoid(input[index] + recurrent[index]);
    const update = sigmoid(input[size + index] + recurrent[size + index]);
    const candidate = Math.tanh(input[size * 2 + index] + reset * recurrent[size * 2 + index]);
    next[index] = (1 - update) * candidate + update * hidden[index];
  }
  return next;
}

function logits(hidden) {
  const output = new Float32Array(model.vocabSize);
  for (let token = 0; token < model.vocabSize; token += 1) {
    let value = model.outputBias[token];
    const offset = token * model.embeddingSize;
    for (let index = 0; index < model.embeddingSize; index += 1) value += model.embedding[offset + index] * hidden[index];
    output[token] = value;
  }
  return output;
}

function nextRandom(state) {
  state.value ^= state.value << 13;
  state.value ^= state.value >>> 17;
  state.value ^= state.value << 5;
  return ((state.value >>> 0) % 1000000) / 1000000;
}

function pickToken(values, temperature, topK, generated, randomState) {
  const adjusted = Array.from(values);
  for (const token of new Set(generated)) adjusted[token] = adjusted[token] >= 0 ? adjusted[token] / 1.05 : adjusted[token] * 1.05;
  if (temperature === 0) {
    let best = 0;
    for (let index = 1; index < adjusted.length; index += 1) if (adjusted[index] > adjusted[best]) best = index;
    return best;
  }
  const candidates = Array.from({ length: adjusted.length }, (_, index) => index).sort((left, right) => adjusted[right] - adjusted[left]).slice(0, Math.min(topK, adjusted.length));
  const scores = candidates.map((token) => Math.exp((adjusted[token] - adjusted[candidates[0]]) / temperature));
  const total = scores.reduce((sum, value) => sum + value, 0);
  let target = nextRandom(randomState) * total;
  for (let index = 0; index < candidates.length; index += 1) {
    target -= scores[index];
    if (target <= 0) return candidates[index];
  }
  return candidates[candidates.length - 1];
}

function generate(prompt, options) {
  const remembered = memoryCompletion(prompt, options.maxTokens);
  if (remembered) return remembered;
  let hidden = new Float32Array(model.hiddenSize);
  for (const token of encodeText(prompt)) if (token < model.vocabSize) hidden = step(hidden, token);
  let values = logits(hidden);
  const generated = [];
  const randomState = { value: (options.seed || 7) | 0 };
  for (let index = 0; index < options.maxTokens; index += 1) {
    const token = pickToken(values, options.temperature, options.topK, generated, randomState);
    generated.push(token);
    hidden = step(hidden, token);
    values = logits(hidden);
    if (generated.length > 3 && decodeTokens(generated).includes("\n")) break;
  }
  return decodeTokens(generated);
}

self.addEventListener("message", async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      latestLoadRequestId = message.loadRequestId;
      const response = await fetch(message.url, { cache: "force-cache" });
      if (!response.ok) throw new Error(`Could not fetch ${message.url} (${response.status})`);
      const loadedModel = loadWeights(await response.json());
      if (message.loadRequestId !== latestLoadRequestId) return;
      model = loadedModel;
      self.postMessage({ type: "model-ready", model: model.name, loadRequestId: message.loadRequestId });
      return;
    }
    if (message.type === "generate") {
      if (!model) throw new Error("Load a model before generating");
      const started = performance.now();
      const completion = generate(message.prompt, message);
      self.postMessage({ type: "completion", requestId: message.requestId, loadRequestId: message.loadRequestId, model: model.name, completion, elapsedMs: performance.now() - started });
    }
  } catch (error) {
    if (message.loadRequestId !== undefined && message.loadRequestId !== latestLoadRequestId) return;
    self.postMessage({ type: "error", requestId: message.requestId, loadRequestId: message.loadRequestId, message: error instanceof Error ? error.message : "Local model error" });
  }
});

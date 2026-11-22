// Talks to a running Rosetta-Jane gateway with the same client libraries Project NOMAD uses
// (the `ollama` and `openai` npm packages), making the same calls NOMAD's OllamaService makes.
// Usage: node nomad_clients.mjs <gateway url> <folder holding node_modules>
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import path from 'node:path'

const [base, modules] = process.argv.slice(2)
const require = createRequire(pathToFileURL(path.join(modules, 'package.json')))
const { Ollama } = require('ollama')
const OpenAI = require('openai').default ?? require('openai')

const messages = [{ role: 'user', content: 'w3 w9 w20 w41' }]
const options = { temperature: 0, num_predict: 10, num_ctx: 8192 }
const out = {}

// NOMAD's connection test and reachability probe.
out.v1_models_ok = (await fetch(`${base}/v1/models`)).ok

// What NOMAD checks to decide the server speaks Ollama's own API.
const tags = await (await fetch(`${base}/api/tags`)).json()
out.native = Array.isArray(tags.models)
out.model = tags.models[0].name

const show = await (await fetch(`${base}/api/show`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: out.model }),
})).json()
out.context_length = show.model_info[`${show.model_info['general.architecture']}.context_length`]
out.thinking = show.capabilities.includes('thinking')

const ollama = new Ollama({ host: base })
const whole = await ollama.chat({ model: out.model, messages, stream: false, think: false, options, keep_alive: '5m' })
out.native_text = whole.message.content
out.native_count = whole.eval_count

let streamed = ''
let pieces = 0
for await (const part of await ollama.chat({ model: out.model, messages, stream: true, options })) {
  streamed += part.message.content
  pieces += 1
}
out.native_stream_text = streamed
out.native_stream_pieces = pieces

const openai = new OpenAI({ apiKey: 'nomad', baseURL: `${base}/v1` })
out.openai_models = (await openai.models.list()).data.map((m) => m.id)
const reply = await openai.chat.completions.create({ model: out.model, messages, temperature: 0, max_tokens: 10 })
out.openai_text = reply.choices[0].message.content
out.openai_tokens = reply.usage.completion_tokens

let sse = ''
const stream = await openai.chat.completions.create({
  model: out.model, temperature: 0, max_tokens: 10, stream: true,
  messages: [{ role: 'user', content: [{ type: 'text', text: 'w3 w9 w20 w41' }] }],
})
for await (const chunk of stream) sse += chunk.choices[0]?.delta?.content ?? ''
out.openai_stream_text = sse

// The unload hint NOMAD sends when switching models must not start an answer.
const hint = await (await fetch(`${base}/api/generate`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: out.model, keep_alive: 0 }),
})).json()
out.hint_done = hint.done === true && hint.response === ''

const embed = await fetch(`${base}/api/embed`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: 'nomic-embed-text', input: ['x'] }),
})
out.embed_status = embed.status

const status = await (await fetch(`${base}/api/rosetta/status`)).json()
out.status = status

console.log(JSON.stringify(out))

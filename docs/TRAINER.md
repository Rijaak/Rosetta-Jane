# Training one model across several PCs

This is the `rosetta_jane lan` part of Rosetta-Jane.

Fine-tune one large language model across several PCs by pooling their graphics memory over a network.

Each PC holds a run of the model's layers. A training example passes through every PC in turn and the corrections pass back the other way. The result is a standard LoRA adapter, the same one a single machine with enough memory would have produced.

It works with one Ethernet cable between two PCs, with a room full of PCs on a switch, or across the internet.

## Status: early, and not yet run on real graphics cards

Read this before relying on it.

**Tested** (29 automated tests, all on a tiny random model, CPU only, over real network sockets on one machine):

- A run split across 1, 2, 3 and 4 machines produces the same adapter weights as ordinary single-machine training, to within rounding.
- Killing a guest mid-run and restarting it ends at the same weights as an uninterrupted run.
- Stopping and restarting the host resumes from the last checkpoint to the same weights.
- The merged adapter loads on the full model with the standard `peft` call.
- A wrong secret or group code is refused. Tampered and replayed messages are rejected. Traffic is unreadable to an eavesdropper.
- Nine model families (list below).

**Not tested:**

- Any real graphics card, 4-bit loading (`--load-4bit`), half precision, or `--fp16-scaling`. The code paths exist and are unproven.
- Any real model. Only tiny random ones.
- Two physical machines, a real switch, Wi-Fi, or an internet link.
- Speed. No timings exist yet.

Treat your first real run as a test: do a short one and compare its loss curve against expectations before committing days to it.

## How it works

```
host:   embeddings + first layers                         final layer + loss
guest1:                         middle layers
guest2:                                        more layers
```

- The **host** owns the training text. It holds the first layers and also the very last step, where the answer is graded.
- **Guests** hold middle layers. They receive a block of numbers, run their layers, and send a block back. They never receive the text or the answers.
- The host relays between guests, so guests never talk to each other.
- After each batch the host collects every machine's gradient size, computes one shared clipping factor, and tells everyone to step together. That is what makes the result match single-machine training.
- Every machine saves a checkpoint at the same step number. After any failure the host reconnects, finds the newest step everyone saved, rolls everyone back to it and continues.

Model weights never travel over the network. Every machine downloads the same base model and cuts out its own layers locally.

## Requirements

- Python 3.13 or newer on every machine (needed for the built-in encryption).
- Rosetta-Jane installed (`pip install .` in the project folder), plus `bitsandbytes` for `--load-4bit`.
- The same base model, in safetensors format, downloaded on every machine.
- Wired Ethernet. Wi-Fi will work but a multi-day run over it is asking for dropouts.

## Two PCs, one cable

Example: a host with a 16 GB card and a guest called `tower` with an 8 GB card, joined by an Ethernet cable. Give the two network ports fixed addresses, for example `10.0.0.1` (host) and `10.0.0.2` (tower).

**1. On the host, make a plan and pick a group code.**

```bash
rosetta_jane lan plan --model /models/base --host-mem 16 \
    --guest tower=10.0.0.2:7000=8 --out plan.json
```

Copy `plan.json` to the guest. Agree on a group code, at least 8 characters, longer is better.

**2. On the guest, start the worker.**

```bash
export ROSETTA_JANE_GROUP_CODE="blue falcon 42 harbor"
rosetta_jane lan worker --model /models/base --plan plan.json --name tower \
    --listen 10.0.0.2:7000 --load-4bit --dtype float16
```

**3. On the host, start training.**

```bash
export ROSETTA_JANE_GROUP_CODE="blue falcon 42 harbor"
rosetta_jane lan train --model /models/base --plan plan.json --pairs pairs.jsonl \
    --holdout holdout.jsonl --load-4bit --dtype float16 --fp16-scaling \
    --grad-checkpointing --accum 16 --epochs 3 --max-tokens 2048 --save-every 200
```

`pairs.jsonl` has one JSON object per line with `question` and `answer`. Extra fields are ignored.

When it finishes, the adapter for the full model is in `jane-train-run/adapter-run/`. Load it with `peft` on the full base model, or merge and convert it for the program you run models in.

## More machines (a LAN party)

Add one `--guest` per machine, in the order the layers should run. Each gets layers in proportion to its memory.

```bash
rosetta_jane lan plan --model /models/base --host-mem 16 \
    --guest ana=192.168.1.21:7000=12 \
    --guest raj=192.168.1.22:7000=8 \
    --guest kim=192.168.1.23:7000=24 --out plan.json
```

Everyone needs the same base model, the same `plan.json`, and the same group code. Each guest runs `worker --name <their name>`. The run goes at the pace of the slowest machine, and it pauses if any machine drops out until that machine returns.

## Group codes

A group code works like a co-op game lobby: everyone who types the same code can connect to each other, and nobody else can.

- Spaces and capital letters are ignored, so a code read aloud still matches.
- Set it with `--group-code` or the `ROSETTA_JANE_GROUP_CODE` environment variable. The variable keeps it out of your command history.
- The code is also the encryption key. A short or guessable code is a weak lock. Four or five random words is a good code.
- For the strongest option, run `rosetta_jane lan secret` and copy the file it makes to each machine by hand, then use `--secret-file`.
- The special code `public` means anyone who can reach your machine may connect. See the warnings below before using it.

## Across the internet

A guest has to be reachable from the host. Home routers block incoming connections, so pick one of these.

| Method | How | Notes |
| --- | --- | --- |
| Tailscale or another mesh VPN | Put every machine on the same private network and use its addresses in the plan | Easiest. Works through home routers. Adds its own encryption. |
| WireGuard | Set up a tunnel between the machines | Same idea, more manual. |
| Port forwarding | Forward a port on the guest's router to the worker | Exposes the worker to the whole internet. Use a long group code. |
| A reverse TCP tunnel service | The guest opens an outbound tunnel and gets a public address | The tunnel company can see connection times and volumes, not the contents. |

Expect it to be slower than a local network. Every training example sends a block of numbers (tens of megabytes for a large model at long lengths) to each guest and back, so home upload speed becomes the limit.

## Security and privacy: what you are exposed to

**Protected:**

- Only machines with the secret or group code can connect.
- Traffic is encrypted, and each message is checked against tampering and replay.
- Nothing received from the network is ever executed. Messages are JSON and raw numbers, and adapters are safetensors. No pickle crosses the network.
- A worker only works while its program is running, only for a host with the code, and stops when you close it.

**Not protected. Know these before you use it:**

1. **Guests can learn about your text.** They never receive it, but the numbers passing through their layers can be turned back into an approximation of it. Only train private material with people you trust. A group code keeps strangers out; it does not make your friends unable to look.
2. **A guest can sabotage the run.** It can return wrong numbers and quietly ruin or bias the result. Nothing here detects that. Train only with machines you trust.
3. **A guest holds part of the result.** Each machine keeps the adapter weights for its own layers until the end. A guest that leaves for good takes its part with it, and the run has to restart without it.
4. **Weak codes can be guessed.** Anyone who can reach a worker can try codes against it. There is no lockout. Use a long code.
5. **`public` is open to everyone.** With the code `public`, encryption still hides traffic from bystanders, but any stranger can connect to a worker and use its hardware, and a host has no idea who its guests are. Points 1 and 2 then apply to strangers.
6. **Your IP address is visible** to every machine you connect to, and your internet provider can see that you are exchanging large amounts of data with them.
7. **`--no-encrypt` sends everything readable.** Use it only inside a VPN, or on a cable between two machines you own.
8. **A worker uses real power and heat** for as long as a run lasts, at whatever intensity the host drives it. Cap its memory with `--max-memory` and stop it with Ctrl-C at any time.

This software has had no outside security review.

## If something goes wrong

- **A guest crashes or loses power:** restart its worker with the same command. The host notices, waits, rolls everyone back to the last shared checkpoint and continues.
- **The host crashes:** run the same `train` command again. It resumes from the last shared checkpoint.
- **Wrong code, wrong model or wrong plan on a guest:** the host stops at once and says which.
- **The same step fails three times in a row** (for example a guest running out of memory on a long example): the host stops with the last error. Lower `--max-tokens` or change the plan.
- **Bad gradients** (possible in half precision): every machine skips that step together.

Work since the last checkpoint is repeated after a failure, so `--save-every` is the most you can lose.

## Model families

Split runs were checked against single-machine training for: `llama`, `qwen2`, `qwen3`, `mistral`, `mixtral`, `phi3`, `gemma`, `gemma3_text`, `starcoder2`.

Any other family is refused unless you pass `--allow-untested-model` to `plan`. Gemma 2 is left out on purpose: its split run drifted slightly from the reference. Checks used short sequences, so behaviour of sliding-window attention at long lengths is not covered.

## Limits

- One example at a time (batch size 1), with `--accum` examples per optimizer step.
- Text-only decoder models in safetensors format.
- Guests must be reachable from the host; there is no matchmaking server yet.
- No detection of dishonest guests.

## Running the tests

```bash
pip install pytest
python -m pytest tests/lan -q
```

They take about a minute on a laptop and need no graphics card and no downloads.

## License

GNU Affero General Public License, version 3. See `LICENSE`.

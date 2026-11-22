# Rosetta-Jane

Community AI. Run one large AI model across many volunteer devices, peer to peer, with no central server.

Named for Jane, the mind that lives across a network in Orson Scott Card's *Speaker for the Dead*.

Rosetta-Jane is a companion to [Project NOMAD](https://github.com/Crosstalk-Solutions/project-nomad). NOMAD is an offline box that keeps everything on one machine. Rosetta-Jane is the opposite trade: it reaches out over a network so that devices too small to run a model alone can run it together. NOMAD connects to it through the same "remote AI server" setting it already has.

## Status: early, and not yet run on real hardware

**Tested** (160 automated tests, tiny random models, CPU only, real network connections on one machine):

- Text produced through a chain of peers matches the whole model on one machine, token for token, and the scores behind every token match too.
- That holds when the route uses only part of a peer's layers, and when a peer dies mid-answer and others take over.
- Nodes that each know one other node all find each other, with no server. Nodes that leave are forgotten.
- Groups with different codes never see each other.
- Peers that return wrong results are caught and blocked, including when they are the majority and when they only lie after first behaving (see "Catching peers that lie").
- Training through peers gives the same trained weights as ordinary LoRA training on one machine. That holds with lying peers in the group, with a peer leaving in the middle of an example, and after stopping and starting again. The result loads as a standard adapter.
- Training one model across several PCs on a local network matches single-machine training (see `docs/TRAINER.md`).
- The Ollama-style interface answers `/api/chat` and `/api/generate`, streaming or not.
- The client libraries Project NOMAD uses (the `ollama` and `openai` npm packages) can list the model, read its details, and chat through the gateway, streaming or not. This test only runs where NOMAD's code is present.
- Nine model families (list below).

**Not tested:**

- Any real graphics card, 4-bit loading, or half precision.
- Any real model, or a real tokenizer's chat format.
- More than one physical machine, a home router, or the internet.
- A real NOMAD install talking to it.
- Speed, or more than a handful of peers.
- Local-network discovery by broadcast (only tested with direct packets).

There has been no outside security review.

## How it works

A model is a stack of layers. Rosetta-Jane splits the stack between devices.

```
you:      words -> first layers                               last layer -> words
peer A:                      layers 2-11
peer B:                                   layers 12-23
peer C:                                                layers 24-31
```

- **The p2p service** (`serve`): a device loads a run of middle layers and applies them for anyone in its network who asks.
- **Asking** (`gateway` or `ask`): your device keeps the first few layers and the final output layer. It sends the in-between numbers through a chain of peers that together cover the rest.
- After the first pass, each new word is one small message to each peer.
- If a peer drops out mid-answer, your device finds others for those layers and replays what the lost peer had seen.

Every device needs its own copy of the same base model, in safetensors format. Model files are never sent over the network.

## No server: how devices find each other

Every copy of Rosetta-Jane is the same program. Nobody runs anything special.

- To join a network, give your node the address of **any one node already in it** with `--peer`. That is the only introduction it needs.
- Nodes regularly swap the addresses of peers they have connected to. Within a few rounds everyone knows everyone.
- A node only passes on addresses it has reached itself, so one node cannot fill the network with invented peers.
- On a local network, add `--lan` and nodes announce themselves to each other.
- The first node of a new network starts with no `--peer` at all.

This is how LimeWire and BitTorrent peer exchange worked. Like them, it still needs that first address from somewhere: a friend, a forum post, a list in a README.

## Group codes

A group code works like a co-op game lobby. Everyone who enters the same code is in the same private network. Nobody else can connect, and nobody else can read the traffic.

- Set it with `--group-code` or the `ROSETTA_JANE_GROUP_CODE` environment variable.
- Spaces and capital letters are ignored.
- The code is the lock. Use four or five random words. A short code can be guessed.
- The code `public` is the open network. Anyone running Rosetta-Jane can join it.

## Install

Python 3.13 or newer. From the project folder:

```bash
pip install .
```

Before loading a model, read "Check the terms of the model you load" below.

Add `bitsandbytes` for `--load-4bit`.

## Before you run the p2p service

The p2p service is safest inside a group code, where every peer is someone you chose. On the `public` network, use is open to anyone.

- **Open P2P runs strangers' prompts.** Your device computes layers for whatever anyone on the network asks. It cannot see or filter what that is.
- **Your device never sees the text.** It receives processed numbers from partway through the model, not words. They are held in memory while an answer is being made and are not saved; the handoff log keeps only a small fingerprint of each exchange. These numbers are not a strong privacy guarantee for the person asking, and they do not tell you what your device helped answer.
- **Your device is reachable from the internet.** The p2p service needs an open port (7700 by default) or a tunnel. Anyone can reach that port and try to talk to the program. Only devices with your group code get past the first exchange, and traffic is encrypted, but on `public` everyone has the code, and a bug in this program could expose the device either way. Use a mesh VPN such as Tailscale together with a group code to avoid an open port.
- **It uses your device's hardware, power and bandwidth** whenever the program runs. `--max-sessions` limits how much. Ctrl-C stops it.
- **No warranty.** This software is provided as is, under the GNU Affero General Public License, version 3.

## Run the p2p service

```bash
export ROSETTA_JANE_GROUP_CODE="blue falcon harbor ninety"
rosetta_jane serve --model /models/base --span 12 --peer 203.0.113.5:7700 \
    --load-4bit --dtype float16
```

Left to itself, the p2p service looks at the swarm when it starts and takes the layers nobody is serving, or failing that the ones with the fewest serving peers. `--span 12` says how many layers your memory can hold. Peers that join one after another fill the model in between them, so a group does not have to coordinate who takes what. To choose yourself, pass `--layers 8:20` (layers 8 through 19). Layer 0 can never be served; it always stays with the person asking. The choice is made once at start; it does not move later if others leave.

Add `--training` to also help other people train. It is off unless you say so, because training is heavier work than answering.

A serving device prints how much work it has done once a minute. It works only while this program runs and only for its network. `--max-sessions` caps how many answers it helps with at once. Ctrl-C stops it.

## Ask through the swarm

```bash
export ROSETTA_JANE_GROUP_CODE="blue falcon harbor ninety"
rosetta_jane ask "What is a heat pump?" --model /models/base --own-layers 2 --peer 203.0.113.5:7700
```

`--own-layers` is how many of the first layers stay on your device. More layers at home means peers see a more processed form of your prompt, at the cost of more of your own memory.

## Use it from NOMAD

```bash
rosetta_jane gateway --model /models/base --own-layers 2 --peer 203.0.113.5:7700
```

This opens an Ollama-style address on your own device, `http://127.0.0.1:11435`. Set NOMAD's remote AI server address to it, and choose the model named `rosetta`.

- Only chat and text generation go through the swarm. Embeddings and model downloads are refused, so NOMAD's document indexing needs to stay on a local AI server.
- This has not been tried against a real NOMAD install yet.

## See who is out there

```bash
rosetta_jane peers --peer 203.0.113.5:7700
```

## Reaching each other across the internet

Only devices that **serve** need to be reachable. Devices that only ask connect outward and work behind any home router.

| Method | How | Notes |
| --- | --- | --- |
| Port forwarding | Forward the node's port (7700 by default) on the home router | Works with the open network. Exposes that port to the internet. |
| Mesh VPN such as Tailscale | Put the group's devices on one private network and use those addresses | Easiest for a group of friends. No router changes. |
| WireGuard tunnel | Set up tunnels between members | Same idea, more manual. |
| Reverse TCP tunnel service | The serving peer opens an outbound tunnel and shares the public address it is given | The tunnel company sees connection times and volumes, not contents. |

## Risks: read before using

**What is protected**

- Only devices with the group code can connect, and traffic is encrypted.
- Messages are checked against tampering and replay.
- Nothing received from the network is executed. Messages are plain data.
- Your words going in and the words coming out are produced on your own device.

**What is not**

1. **Peers can learn roughly what you asked.** Each peer in the chain receives an encoded form of your *entire* prompt, not a fragment. That encoding can be turned back into an approximation of the text. Splitting the model spreads the work; it does not split the secret. Keeping more layers at home makes this harder, not impossible. Do not send private material through peers you do not trust. NOMAD's privacy setting for keeping chosen files and chats off a remote server exists for this.
2. **Peers can lie.** A peer can return wrong numbers and steer or spoil an answer. The checks under "Catching peers that lie" guard against this. Known-answer tests are prepared automatically the first time you ask or train; know the limits listed there.
3. **You are pseudonymous, not hidden.** There are no accounts or names, and each run uses a fresh random ID. But every peer you connect to sees your IP address, peers share lists of addresses with each other, and your internet provider can see that you are exchanging data with them.
4. **Serving peers compute things they cannot see.** A serving device has no idea what questions it is helping answer. On the open network that includes questions from strangers.
5. **Serving peers run an open port** and spend real electricity for other people. Requests are size-limited and sessions are capped, but a busy or hostile network can still keep a serving peer's hardware occupied.
6. **The `public` code has no lock at all.** Its key is known to everyone, so its encryption only stops bystanders who are not running Rosetta-Jane. Points 1, 2 and 4 then apply to anyone on the internet.
7. **Weak codes can be guessed,** and there is no lockout after wrong guesses.
8. **Answers can be slow or fail.** Every word makes a round trip through every peer in the chain. If no reachable peers cover some layers, you get an error saying which.

## Catching peers that lie

A peer can return wrong numbers. Four checks guard against that. The first two are on by default.

| Check | How to get it | What it does | What it costs |
| --- | --- | --- | --- |
| Known answers | Automatic: prepared on your first run with a model (or run `rosetta_jane calibrate` yourself) | Tests every new peer with work whose correct result your device already knows, before trusting it, and again at random | A slow one-time preparation on your device |
| Second opinions | Automatic when at least 3 peers are available (`--redundancy auto`) | Sends each stretch of layers to 2 independent peers and compares them at every step | Twice the work for serving peers. More peers see your prompt. |
| Rechecking | `--audit` | Your own device recomputes peers' work from its copy of the model, a few layers at a time | Time on your device |
| Refusing unchecked answers | `--require-checked` | Refuses to answer if any stretch lacks a second opinion | Fewer answers when the network is thin |

### Known answers

A peer never sees a question. It gets a block of numbers for one layer and returns a block for a later layer. So a test with a known answer is a pair of blocks: what goes in, and what must come out.

```bash
rosetta_jane calibrate --model /models/base --own-layers 2
```

- Your device takes private test prompts (arithmetic, logic, instructions, with numbers randomised per device) and works out the correct numbers at every layer, loading a few layers at a time. This is slow and done ahead of time.
- When asking, each new peer is given one of these as if it were real work. A wrong result gets it blocked before it touches a real question.
- A peer cannot tell a test from a real question, so it cannot behave only when watched.
- When two peers disagree, each is given a fresh test to help decide who is lying.
- Each test is shown to a given address only once, since a repeat would give it away. When they run out, the program says so; run `calibrate` again.
- Only the question halves are ever sent to peers.

**Known answers cannot be passed around.** Anyone holding a device's answers could pass that device's tests, so:

- Every device invents its own test prompts at random. Someone else's answers do not fit your tests.
- The answers are stored encrypted, locked to the device that made them. Copied to another machine, with or without its key file, they cannot be read and are refused.
- A device only trusts answers it worked out itself. There is no way to load a set made elsewhere.
- What this cannot do: stop the owner of a device from digging out their own data if determined. That weakens only their own protection, never anyone else's.
- If the machine's identity changes (a reinstall, a new container), its old answers stop opening. Run `calibrate` again.

### How disagreements are settled

- **Peers agree:** accepted.
- **Peers disagree, with known answers prepared:** both are retested. One that fails is blocked.
- **Still unsettled, with `--audit`:** your device recomputes the disputed step. Whoever does not match is blocked. This is right however many peers are lying.
- **Still unsettled, without `--audit`:** more peers are asked until a strict majority agrees. If nobody is left to ask, the answer is refused.
- **Peers agree but might all be lying together:** with `--audit`, the first step of every answer and a random share after it (`--audit-rate`, 10% by default) are rechecked anyway.

### Limits to know

1. **A vote can be rigged.** Identities are free, so one person can run many peers and be the majority. With nothing but second opinions, a lying majority wins and the honest peer is the one blocked. Known answers and `--audit` are the checks that do not depend on who has more voices.
2. **A known-answer test covers the first several steps of a session.** A peer that lies only deep into long answers outlasts it. Second opinions and `--audit` cover that.
3. **Checks are per precision.** A peer running a compressed 4-bit copy gives different numbers from an uncompressed one, by about as much as a deliberate lie could. So results of different precisions are not compared with each other by default. To check peers of several precisions, run `calibrate` once for each (for example once plain and once with `--load-4bit`); the right set is picked for each peer automatically. A peer whose precision you have no answers for is used untested.

   Two uncompressed precisions may be close enough to share one set. Measure it on your own model first:

   ```bash
   rosetta_jane measure-precision --model /models/base --a float32 --b float16
   ```

   It prints the gap layer by layer and says whether sharing is safe. If it is, add `--treat-as-same float32,float16` when asking. On a deep random test model, float16 stayed within 0.2% of float32, while bfloat16 drifted to 1.7%, too close to the 2% tolerance to share. A trained model may behave differently; that is why the measurement is yours to run.
4. **Random checks are a sample.** A peer that lies only occasionally can slip past until a checked step lands on a lie.
5. **Small nudges pass.** Results are compared with a tolerance, because different hardware gives slightly different numbers. A change smaller than the tolerance is not detected.
6. **Blocks are local and temporary.** A caught peer is blocked on your device until you restart. It can return under a new identity, which is why new peers are always tested first.
7. **If every available peer for some layers is lying,** a checked answer is refused with an error. It is never replaced by a wrong one.

## Train through the swarm

**Build a better offline model, together.** Don't be limited by personal hardware. Pool your machines as a community to teach a model what we all need it to know. When training is done, download the result, check it, and use it offline. The network is only needed while training.

You teach the model from question-and-answer pairs:

```bash
rosetta_jane train --model /models/base --own-layers 2 --peer 203.0.113.5:7700 \
    --data pairs.jsonl --epochs 3 --out my-adapter
```

`pairs.jsonl` has one `{"question": ..., "answer": ...}` per line. The result in `my-adapter` is a standard LoRA adapter for the whole model, usable with the PEFT library.

**How the work is divided.** Your device keeps the training text, the first layers, the output layer, every trained weight, and the optimizer. Peers started with `serve --training` hold the middle layers and act as calculators. For each pair, a peer is sent the small trained matrices for its layers, runs the pair forward, runs the correction backward, and hands back everything it computed. It keeps nothing.

That design is what makes training workable among strangers:

- **A peer that leaves costs one pair.** Nothing trained lives on it, so the pair is redone with another peer and the result is unchanged.
- **Work can be checked.** Two peers given the same pair must return the same numbers. All the checks from "Catching peers that lie" apply at each of the three stages (forward, backward, and the weight corrections): comparison between peers (automatic with three or more), recomputing on your device with `--audit`, and known-answer tests before a peer is given any of your text.
- **Stopping is safe.** Progress is saved on your device every `--save-every` steps. Run the same command again to carry on.

**What it costs.** Every pair sends its hidden states out and back through each peer, plus that peer's weight corrections, so training this way is much slower than on hardware you own. With checking on, each stretch of layers is computed twice.

**What peers can see.** The hidden states for your training text, which can be turned back into an approximation of it, and the trained matrices for their own layers at that moment. No single peer sees the whole adapter unless it serves every layer that leaves your device, but **do not train on private material with peers you do not trust.** Use a group code shared with people you know, or train on your own machines with the LAN trainer below.

**Limits specific to training.** Known-answer tests exercise a peer's answering path. A peer could pass them and still lie during training, which is why comparison between peers or `--audit` should be on for the open network; with one unchecked peer the summary line reports the work as unchecked. LoRA dropout is not supported. One pair is processed at a time.

## Trained adapters are never applied for you (on purpose)

When a training run finishes, the result is saved as a file and nothing else happens. Rosetta-Jane does not load it, the gateway keeps answering with the unchanged model, and NOMAD's page offers only a download. To use an adapter, a person has to download that one adapter, look it over, and apply it themselves. They can try it and score it first (see below); a trial shows what the adapter would do without switching it on.

This is a deliberate speed bump, not a missing feature. The reasons:

- **Other people's devices did part of the work.** The checks catch peers that return wrong numbers. They cannot tell you the finished adapter is good, or that the pairs it learned from were.
- **An adapter changes how the model answers everything afterwards.** A wrong answer is one bad reply. A bad adapter applied automatically would shape every reply, quietly, until someone noticed.
- **The purpose of swarm training is to make new models possible, not to switch them on instantly.** The swarm supplies the hardware. Whether a result deserves to be used is a judgement, and it should be a person's.
- **One at a time.** There is no "apply all" and no automatic update to a newer run. Each adapter is its own decision.

To make that judgement easier, every download includes a `READ-ME-FIRST.txt` and a `rosetta-jane-run.json` stating how many pairs it learned from, how much of the work was double-checked and how much rested on one peer's word, how many peers were blocked during the run, and the exact base model it fits. Try an adapter on questions you already know the answers to before relying on it.

If you build on Rosetta-Jane, keep this step. Removing it trades a minute of someone's attention for the one safeguard that does not depend on the network behaving.

## Scoring, trying and sharing adapters

Whether an adapter deserves to be used is a person's decision. These are the tools for making it, and for helping others make it.

**Scoring by comparison.** The useful question about an adapter is whether the model is better with it than without it, on questions it never saw.

- Every training run keeps one pair in ten back, never trains on it, and afterwards puts those pairs to the unchanged model and to the model with the adapter. The result goes into the adapter's `READ-ME-FIRST.txt` and facts file.
- Anyone who has an adapter can score it again on their own questions:

  ```bash
  rosetta_jane score --model /models/base --peer 203.0.113.5:7700 --adapter ./their-adapter --data my-questions.jsonl
  ```

- A score reports the share of answer words the model got right and the loss, before and after, with a cautious one-line result. Few pairs are called "too few to rely on". A lying peer cannot bend a score any more than it can bend training: scoring goes through the same checks.

**Trying it.** A trial answers a question twice, with the unchanged model and with the adapter, so the two can be read side by side. You can give it a list of questions, or hold a trial chat in which each side carries its own earlier replies forward.

```bash
rosetta_jane try --model /models/base --peer 203.0.113.5:7700 --adapter ./their-adapter "How do I purify water?"
```

Trying is not applying. A trial always shows both answers, always says which is which, and changes nothing: ordinary answers through the gateway are unaffected, and a test confirms it. Trials are slow, because they run on the training route and work the whole reply out again for every word, so replies are short by default. They need peers that have chosen to help with training.

**Each other's adapters.** Send someone the zip you downloaded. They add it in the phone app or on NOMAD's page ("Add an adapter someone shared"), or unzip it for the commands above. A shared zip is treated as untrusted: only the adapter's own three files are taken, by fixed name, whatever paths the zip claims; its settings are rewritten from checked values; and the weights are a safetensors file, which holds numbers and cannot carry code. An adapter made for a different model is refused before anything is sent to the group. Adding one applies nothing.

**Score cards.** A score card belongs to whoever made it. It holds numbers only, never questions or answers, and it stays on their device. The phone app has a "Share score card" button that opens the phone's share menu, and NOMAD's page has "Copy to Share". Nothing is sent until the person chooses where it goes. Cards are compared by people, in the open, because a score reported by a stranger on questions you cannot see cannot be checked. That is also why Rosetta-Jane does not count downloads or uses: there is no server to count them, and devices do not report what their owners do.

**Model Adapter Score.** Scores are not kept in this README or in any shared list. Each one lives with its adapter, on the device of whoever made it. To see them together, make the document called Model Adapter Score: one landscape page that sets each adapter beside the unchanged model, on the same questions. It is arranged by model, then by adapter. The model whose best adapter improved it most comes first, and within a model the adapters that improved it most come first.

```bash
rosetta_jane scores            # writes "Model Adapter Score.html" from the adapters and score cards in the work folder
```

NOMAD's page and the phone app each have a button that makes the same page. It is one file with nothing loaded from the network, so it opens offline in any browser and prints in landscape. Every row says who scored it, on how many pairs, and whether the adapter had seen them, because scores made on different questions are not strictly comparable. The result shown for a row is worked out from its numbers, not copied from what a shared card says about itself, and anything written by someone else is shown as plain text. The document is made on your device and goes nowhere unless you send it.

Limits: scores come from loss and next-word accuracy, which measure whether the adapter learned the answers, not whether the answers are good. Nothing here has been run on a real model.

## A front end: controls for NOMAD's settings page

`rosetta_jane gateway --controls --serve 8:16` runs asking, serving and training in one program and lets a front end manage it. Project NOMAD's Rosetta-Jane settings page uses this. With controls on, a front end can read the swarm's status, switch serving and training help on and off, start, watch and stop a training run, and download finished adapters.

Changing anything, and downloading adapters, requires the control token: a random secret the gateway saves in `control-token` in its work folder. Reading status does not. Training and answering take turns, so chat keeps working during a run.

## Train one model across several PCs you own

`rosetta_jane lan` pools graphics memory across PCs on a local network to fine-tune a model too big for any one of them. Each PC keeps its own trained weights, so it is faster than swarm training but needs every PC to stay up, and is meant for machines you trust. See `docs/TRAINER.md`.

## Fingerprints of every handoff

Each time numbers pass from one device to another, both devices add one line to `handoffs.jsonl` in their work folder: the time, the kind of work, which layers, and a fingerprint of the numbers. Nothing is sent anywhere. It is on unless you pass `--no-handoff-log`, and it covers answering, swarm training and the LAN trainer.

When a run across several machines goes wrong, collect the file from each one and run:

```bash
rosetta_jane compare-handoffs pc1/handoffs.jsonl pc2/handoffs.jsonl pc3/handoffs.jsonl
```

It reports how many handoffs arrived intact (the same fingerprint on both ends), which ones have no matching arrival, and where two peers given the same work returned different results, split into "rounding" and "more than rounding" with the address of each peer.

A fingerprint is a one-way hash, one overall size, and the eight largest values at one position. It cannot be turned back into text. The file does list the addresses of peers this device talked to, so keep it private. Each file is capped at 64 MB with one older file kept.

This is a tool for finding causes afterwards. It does not check anything while an answer is being made; redundancy, rechecking and known answers do that.

## Phones and other devices without PyTorch

`rosetta_jane/mobile` is Rosetta-Jane written with Python and NumPy only, for devices that cannot run PyTorch, such as Android phones. It reads the same model files and speaks the same protocol, so a phone is an ordinary peer: it can run the p2p service, ask through the group, and train through the group. The Android app in `android/` is this code with a screen on top (see `android/README.md`).

```bash
python -m rosetta_jane.mobile serve --model /models/base --span 2 --peer 203.0.113.5:7700 --training
```

What is tested, always in a process where PyTorch cannot be imported at all:

- **Same numbers.** Its results match PyTorch's to about one part in ten million, for answering and for training (forward, backward and the weight corrections), on the `llama`, `mistral` and `qwen2` families. Other families are refused.
- **The p2p service.** Text answered through a phone-style peer matches the whole model; training through it matches one machine; it and a desktop peer given the same work agree, so each can check the other.
- **Asking.** A phone-style device asking through desktop peers, or through another phone-style peer, gets the whole model's answer. It uses the same checking rules as the desktop program: a lying peer among three was caught and blocked.
- **Training.** A run started from a phone-style device gives the same trained weights as one machine, outvotes a lying peer, and carries on after being stopped. The result is a standard adapter with the same facts file, and nothing applies it.
- **Text.** The tokenizer and chat formatting are written without the usual libraries and agree with them on every sample, for the tokenizer designs the supported families use. Real models' tokenizer files could not be downloaded where the tests run, so the first real model is the real test.
- **On Android.** The app was run on a cloud Pixel 9 (Android 14): the p2p service, asking, and a training run all gave the expected results with a tiny test model.

Limits: it computes in 32-bit floats, so it is a `float32` peer and is compared only with other `float32` peers (start a desktop peer with `--dtype float32` to pair with it). It needs four bytes of memory per weight, so a phone holds a few layers of a small model. When asking or training from a phone, known-answer tests and rechecking on the device are not available yet; comparison between peers is. Nothing has run on a handheld phone or with a real model.

## Check the terms of the model you load

Rosetta-Jane does not come with a model, and its license (AGPL-3.0) covers this program only. Every model has terms of its own, set by whoever published it, and you are responsible for following them. Read them before you load a model, because Rosetta-Jane does things with a model that ordinary private use does not:

- **The p2p service runs the model for other people.** Some terms allow personal or research use only, or restrict offering the model to others or using it commercially.
- **A group needs identical copies.** Everyone must obtain the model themselves from a source that is allowed to provide it. Rosetta-Jane never sends model files between devices.
- **A trained adapter is built on the model.** Terms often say what you may do with fine-tuned versions, including whether you may share them and what notices must go with them.
- **Some terms restrict what the model may be used for,** whoever runs it. On the open network you cannot see what others ask of the layers you serve.

Where to look: the model's page where you downloaded it, and any `LICENSE`, `NOTICE` or `USE_POLICY` file that came with it. Terms differ widely, from fully open licenses to custom ones with conditions, and they can change between versions of the same model. If the terms do not clearly allow what you plan to do, pick a model whose terms do. This is not legal advice.

## Model families

Checked against whole-model results: `llama`, `qwen2`, `qwen3`, `mistral`, `mixtral`, `phi3`, `gemma`, `gemma3_text`, `starcoder2`. Others are refused unless you pass `--allow-untested-model`.

Peers running different precisions (say 4-bit and 16-bit) can be mixed in one chain, but answers then vary with the route. Use `--variant` to stick to one.

## Limits

- One answer at a time per gateway.
- Swarm training has only been run on tiny models; expect it to be slow on real ones.
- No rewards for serving.
- Known-answer tests are prepared automatically on the first run, which makes that first start slow. Peers of another precision still need a `calibrate` run with that precision's options.
- No automatic port opening on routers.
- Text only.

## Tests

```bash
pip install pytest tokenizers
python -m pytest tests -q
```

## Credits

Rosetta-Jane is the work of its author, Rijaak, who conceived and directed it. Code was written with the assistance of Claude AI. It was inspired, in both concept and interface, by [Project N.O.M.A.D.](https://github.com/Crosstalk-Solutions/project-nomad) and the people who made it. It is an independent project and is not affiliated with or endorsed by them. See `CONTRIBUTORS.md`.

## The name

Only versions that keep the safeguards may call themselves Rosetta-Jane: trained adapters are never applied automatically, the warning comes before the p2p service starts, checked and unchecked work are reported honestly, a device only takes part when its owner starts it, and the source is available. A version that removes any of these must use a different name, though it may say what it was based on. See `NAME-POLICY.md`.

## Contributing

See `CONTRIBUTING.md`. First-time contributors agree to the short `CONTRIBUTOR-AGREEMENT.md`, which lets the project keep a single licence and change it later if it ever needs to.

## License

GNU Affero General Public License, version 3 (AGPL-3.0-only). See `LICENSE`.

In plain terms: anyone may use, study, change and share Rosetta-Jane, including for money. Anyone who passes on a changed version, or lets other people use a changed version over a network, must make the source code of their version available under this same license. Improvements stay open to everyone; nobody can take it private.

This is deliberately different from Project NOMAD, which uses the more permissive Apache-2.0. Rosetta-Jane contains none of NOMAD's code, so the two do not conflict. The Rosetta-Jane settings page for NOMAD is a change to NOMAD itself and stays under NOMAD's Apache-2.0 license; it talks to Rosetta-Jane as a separate program.

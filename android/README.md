# Rosetta-Jane for Android

Licensed under the GNU Affero General Public License, version 3, like the rest of Rosetta-Jane. If you give someone a build of this app, changed or not, you must also offer them its source code.

A full peer on a phone: run the p2p service, ask through the group, and train through the group. It runs the
same Python code as the desktop program (everything under `rosetta_jane/mobile`, plus the shared networking and
checking code) on a Python interpreter inside the app, with NumPy doing the arithmetic.

## Build

Needs the Android SDK (platform 34, build-tools 34), Java 17 or newer, Gradle 8.11 or newer, and Python 3.13 on
the build machine.

```bash
echo "sdk.dir=/path/to/android-sdk" > local.properties
gradle assembleDebug          # app/build/outputs/apk/debug/app-debug.apk
```

A debug build can be driven from a computer for testing (see `automate` in `MainActivity.java`). A release
build cannot.

## Release build

A release must be signed with a key that only you hold. Make one once and keep it safe: anyone with the file
and its password can publish updates that phones will accept as yours, and if you lose it you cannot update
the app for people who already installed it.

```bash
keytool -genkeypair -v -keystore rosetta-jane.jks -alias rosetta-jane -keyalg RSA -keysize 4096 -validity 10000
gradle assembleRelease -PRJ_KEYSTORE=/absolute/path/rosetta-jane.jks -PRJ_KEYSTORE_PASSWORD=...
```

Never commit the key file or its password.

## What works, and how it was checked

On a cloud Pixel 9 (Android 14, arm64), with a tiny random test model:

- the p2p service: answers through the phone matched the whole model, a phone and a desktop peer checked each
  other, and training through the phone matched one machine;
- asking from the phone through two desktop peers gave exactly the whole model's reply;
- a training run started from the phone finished, and the adapter it saved loads with the standard PEFT
  library.

Not checked: a real model, a handheld phone, battery and heat, hours in the background, mobile networks, and
most of the on-screen buttons (the tests started things by command; only the Ask form was typed and tapped).

## Adapters in the app

The app keeps one pair in ten back from every training run and scores the result on them. Under "Trained adapters" each adapter can be tried (a list of questions or a trial chat, each answered without the adapter and with it), scored on your own pairs, saved as a zip, and its score card shared through the phone's share menu. "Save Model Adapter Score" writes one landscape page with every score on the phone, each adapter beside the unchanged model, grouped by model with the best first. "Add an adapter someone shared" takes a zip from someone else. Nothing in the app applies an adapter.

These screens were built after the last run on a test device. The Python underneath them is tested; the buttons are not.

## Limits on a phone

- Model families: `llama`, `mistral`, `qwen2`. The tokenizer must be a `tokenizer.json` of the BPE kind.
- 32-bit arithmetic only, four bytes of memory per weight. A phone holds a few layers of a small model.
- Checking when asking or training from a phone: comparison between peers, majority, a third opinion, and
  replacing peers that leave. Known-answer tests and rechecking on the device are desktop-only so far.
- Most mobile carriers block incoming connections, so the p2p service usually needs Wi-Fi or a mesh VPN.

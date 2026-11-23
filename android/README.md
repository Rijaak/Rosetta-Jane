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

A test build of 0.2.0, on the same kind of cloud phone, against PCs standing in for two leaders:

- the phone made its own key and signed with it in plain Python, and the leaders' screens showed the same tag
  as the phone's own;
- it signed on to a teaching team and did training work, fetched the receipt for its points, spent some with
  the same leader, and some in a second leader's team (refused by default; taken once that leader chose to,
  with the first leader's device confirming);
- answers through a team matched the whole model with the adapter, and a team led from the phone answered a
  PC's question through the phone;
- on the screen, the group code and a name were typed in, and these buttons were pressed and did what they
  say: whose points a team takes, End the team, Find teams in my group, Look at a team, Sign on, See my
  teaching points, Spend 5 points.

Not checked: a real model, a handheld phone, battery and heat, hours in the background, mobile networks, the
release build of 0.2.0 (a test build of the same code was used), and these parts of the screen: rating and
blind-comparison buttons, requests, invitations and removing a device. Points were earned at a speeded-up rate
for the test, and the link between the cloud phone and the test PCs dropped twice during it.

## Adapters in the app

The app keeps one pair in ten back from every training run and scores the result on them. Under "Trained adapters" each adapter can be tried (a list of questions or a trial chat, each answered without the adapter and with it), scored on your own pairs, saved as a zip, and its score card shared through the phone's share menu. "Save Model Adapter Score" writes one landscape page with every score on the phone, each adapter beside the unchanged model, grouped by model with the best first. "Add an adapter someone shared" takes a zip from someone else. Nothing in the app applies an adapter by itself.

Since 0.2 each adapter also has "Lead a team with it", "Train it further", and, when other devices' work went into it, a button to offer it to those devices. The "Teams: use an adapter together" section is where you look at a team by its code, find the teams listed in your group, use quick match, accept an invitation, sign on, and leave; ask to join a team that takes requests, or lead a team with no adapter; a leader begins the team there, chooses who can sign on (anyone, people who ask and are approved, or code and invitation only), approves requests, invites devices, removes a device and ends it. A button says whether leaders can see this phone in their invite lists, and switches it. Session teams and teaching teams are listed apart; in a teaching team the phone asks no questions, and a training run started on it uses only the team's devices. Each adapter has "Check it for flattery", which measures whether it goes along with false statements more than the unchanged model does; the same check runs at the end of every training run. In a team that uses an adapter you can make blind comparisons (two unlabelled answers to a trick question or to the question in the Ask box; you say which was right), and answer three questions about the adapter with marks from 1 to 5, towards its community rating. In a session team you can see your teaching points with that leader and spend them on a larger share of the team's computing power. A box takes the name you go by on team screens. The phone makes a secret key of its own the first time it is needed and keeps it with the app's files; your name is shown with a short tag made from that key, so people you team with can tell it is this phone again. The phone has no signing library, so it signs in plain Python, which the tests check against the standard's own examples. Teaching points this phone earns arrive as receipts signed by the leader, and can be spent in any team whose leader takes them. When you lead a team, buttons choose whose points it takes. Removing the app, or clearing its data, removes the key: the phone is then a stranger again and its points cannot be spent. Leading a team, being invited and offering an adapter need the p2p service on, because other devices have to reach the phone. Signing on with a code does not. The buttons' logic was tested against desktop peers (`tests/test_app.py`); the screen itself has not been tapped through on a phone yet.

These screens were built after the last run on a test device. The Python underneath them is tested; the buttons are not.

## Limits on a phone

- Model families: `llama`, `mistral`, `qwen2`. The tokenizer must be a `tokenizer.json` of the BPE kind.
- 32-bit arithmetic only, four bytes of memory per weight. A phone holds a few layers of a small model.
- Checking when asking or training from a phone: comparison between peers, majority, a third opinion, and
  replacing peers that leave. Known-answer tests and rechecking on the device are desktop-only so far.
- Most mobile carriers block incoming connections, so the p2p service usually needs Wi-Fi or a mesh VPN.

# Contributing to Rosetta-Jane

Contributions are welcome: fixes, tests on real hardware, support for more model families, translations of the
warnings, and reports of anything that does not behave as the README says.

## Before your first contribution

1. Read `CONTRIBUTOR-AGREEMENT.md`. It is short. It lets the project keep one licence for everything, and change
   that licence later if it ever needs to, without having to track down every past contributor.
2. Agree to it by adding your name to the list at the bottom of that file, in the same pull request as your
   first contribution. That is all the signing there is.

## What a contribution must keep

Rosetta-Jane has a few safeguards that are the point of the project, not details of it. A change that removes
or weakens one of them will not be accepted, and a version without them may not use the Rosetta-Jane name
(see `NAME-POLICY.md`):

- **Trained adapters are never applied automatically.** A person downloads one adapter, looks it over, and
  applies it themselves.
- **The warning comes before the p2p service starts**, and the person has to acknowledge it.
- **Checked and unchecked work are reported honestly.** Nothing may present unchecked work as checked.
- **A device only joins when its owner starts it**, and only for the group whose code its owner entered.

## Practical points

- Run the tests before sending a change: `python -m pytest tests -q`. New behaviour needs a test.
- Everything under `rosetta_jane/mobile`, and the files it shares with the desktop program, must work without
  PyTorch. The tests run that code in a process where PyTorch cannot be imported.
- Say plainly in the README what has and has not been tested. Do not claim more than was checked.
- Say how each contribution was made. If an AI system wrote some or all of it, say so in the pull request, and say
  who directed it. `CONTRIBUTORS.md` shows how this project records that.

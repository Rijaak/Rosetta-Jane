"""Drive the real command line the way a person would: secret, plan, worker, train."""
import json
import os
import subprocess
import sys

import torch

from common import ROOT, free_port, make_model, wait_port


def _add_tokenizer(model_dir: str) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = ["[UNK]", "<s>", "</s>", "<|", "|>", "user", "assistant", "what", "is", "the", "color", "of",
             "sky", "grass", "blue", "green", "it", "?", ".", "sun", "yellow", "snow", "white"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="<s>",
                                   eos_token="</s>", pad_token="[UNK]")
    fast.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} </s> {% endfor %}"
                          "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    fast.save_pretrained(model_dir)


def _run(args, **kw):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-m", "rosetta_jane.lan"] + args, env=env, capture_output=True,
                          text=True, timeout=120, **kw)


def test_full_command_line_flow(tmp_path):
    model = make_model(str(tmp_path / "model"))
    _add_tokenizer(model)
    pairs = tmp_path / "pairs.jsonl"
    rows = [("what is the color of the sky ?", "it is blue ."), ("what is the color of grass ?", "it is green ."),
            ("what is the color of the sun ?", "it is yellow ."), ("what is the color of snow ?", "it is white .")]
    pairs.write_text("\n".join(json.dumps({"id": i, "question": q, "answer": a}) for i, (q, a) in enumerate(rows)))
    secret, plan = str(tmp_path / "secret.txt"), str(tmp_path / "plan.json")
    port = free_port()

    r = _run(["secret", "--out", secret])
    assert r.returncode == 0 and oct(os.stat(secret).st_mode & 0o777) == "0o600"
    assert _run(["secret", "--out", secret]).returncode != 0           # never overwrites a secret
    r = _run(["plan", "--model", model, "--host-mem", "3", "--host-reserve", "1",
              "--guest", f"tower=127.0.0.1:{port}=2", "--out", plan])
    assert r.returncode == 0, r.stderr
    assert "tower" in r.stdout and json.load(open(plan))["stages"][1]["name"] == "tower"

    env = dict(os.environ, PYTHONPATH=ROOT)
    worker = subprocess.Popen([sys.executable, "-m", "rosetta_jane.lan", "worker", "--model", model, "--plan", plan,
                               "--name", "tower", "--listen", f"127.0.0.1:{port}", "--secret-file", secret,
                               "--work-dir", str(tmp_path / "w")], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(port, worker)
        train = ["train", "--model", model, "--plan", plan, "--pairs", str(pairs), "--holdout", str(pairs),
                 "--secret-file", secret, "--work-dir", str(tmp_path / "run"), "--steps", "6", "--accum", "2",
                 "--save-every", "2", "--warmup", "1", "--lr", "0.02", "--rank", "4", "--alpha", "8",
                 "--targets", "q_proj,v_proj", "--max-tokens", "64", "--eval-every", "3"]
        r = _run(train)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        assert "4 training pairs" in r.stdout and "finished" in r.stdout
        adapter = str(tmp_path / "run" / "adapter-run")
        metrics = [json.loads(line) for line in open(tmp_path / "run" / "metrics-run.jsonl")]
        assert [m["step"] for m in metrics] == [1, 2, 3, 4, 5, 6]
        assert metrics[-1]["loss"] < metrics[0]["loss"]                 # it actually learned something
        assert "holdout_loss" in metrics[2] and metrics[0]["net_mb"] > 0

        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(model)
        tuned = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model), adapter)
        ids = torch.tensor([[3, 5, 4, 7, 8, 9, 10, 11, 9, 12, 17]])
        with torch.no_grad():
            assert float((tuned(input_ids=ids).logits - base(input_ids=ids).logits).abs().max()) > 1e-3

        # The worker refuses a host that types a different group code or secret.
        other = str(tmp_path / "other.txt")
        assert _run(["secret", "--out", other]).returncode == 0
        bad = _run([a if a != secret else other for a in train] + ["--run-id", "x"])
        assert bad.returncode != 0 and "different secret or group code" in bad.stderr   # fails fast, says why

        # Running the same command again finds the finished checkpoint and does not retrain.
        r2 = _run(train)
        assert r2.returncode == 0 and "restored to step 6" in r2.stdout and "step 6/6" not in r2.stdout
    finally:
        worker.kill()


def test_bad_pairs_file_gives_a_clear_error(tmp_path):
    from rosetta_jane.lan.data import read_pairs
    import pytest
    p = tmp_path / "bad.jsonl"
    p.write_text('{"question": "q", "answer": "a"}\n{"question": "only a question"}\n')
    with pytest.raises(ValueError, match="line 2"):
        read_pairs(str(p))

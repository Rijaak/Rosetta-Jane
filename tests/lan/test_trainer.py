"""End-to-end tests over real TCP sockets on a tiny random model (CPU, float32).

The key claim under test: a run split across several machines produces the same
LoRA weights as ordinary training on one machine, including after a guest is
killed and restarted partway through.
"""
import json
import os
import socket

import pytest
import torch

from rosetta_jane.lan.host import Host
from rosetta_jane.lan.protocol import Channel, ProtocolError, tensor_from_bytes, tensor_to_bytes
from rosetta_jane.lan.slicing import SliceError, load_plan, make_slice, model_fingerprint, split_layers

from common import (SECRET, free_port, make_examples, make_model, max_diff, plan_for, reference_train,
                    settings, start_process_worker, start_thread_worker, train_cfg)

TOL = 2e-5


def _setup(tmp_path, guest_mems, layers=6):
    model_dir = make_model(str(tmp_path / "model"), layers=layers)
    ports = [free_port() for _ in guest_mems]
    plan = plan_for(model_dir, guest_mems, ports)
    plan_path = str(tmp_path / "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    return model_dir, plan, plan_path, ports


def _initial_adapter(host: Host, tmp_path) -> dict:
    """The adapter exactly as every machine holds it at step 0 of a fresh run."""
    from safetensors.torch import load_file
    host.connect_all()
    host._sync()
    out = host.export(str(tmp_path / "init-adapter"))
    return load_file(os.path.join(out, "adapter_model.safetensors"))


@pytest.mark.parametrize("guest_mems,checkpointing", [([], False), ([2.0], False), ([2.0, 2.0], False),
                                                      ([1.0, 1.0, 1.0], False), ([2.0, 2.0], True)])
def test_matches_single_machine(tmp_path, guest_mems, checkpointing):
    """0, 1, 2 and 3 break points all reproduce single-machine training."""
    from safetensors.torch import load_file
    model_dir, plan, plan_path, ports = _setup(tmp_path, guest_mems)
    s, cfg = settings(grad_checkpointing=checkpointing), train_cfg()
    for i, port in enumerate(ports):
        start_thread_worker(model_dir, plan_path, f"g{i + 1}", str(tmp_path / f"w{i + 1}"), port)
    examples = make_examples(10)
    host = Host(model_dir, plan, str(tmp_path / "host"), SECRET, s, cfg)
    try:
        init = _initial_adapter(host, tmp_path)
        result = host.train(examples, holdout=examples[:3])
        out = host.export(str(tmp_path / "adapter"))
    finally:
        host.close()
    assert result["steps"] == cfg.total_steps and result["recoveries"] == 0
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    want, ref_model = reference_train(model_dir, init, examples, s, cfg, host.lr_at)
    assert max_diff(init, want) > 1e-3, "reference training did not move the weights; test is vacuous"
    assert max_diff(got, want) < TOL

    # The exported adapter must load on the full model with the standard library call.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    full = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), out)
    ids = examples[0][0]
    with torch.no_grad():
        a = full(input_ids=ids).logits
        b = ref_model(input_ids=ids).logits
    assert float((a - b).abs().max()) < 1e-4
    # And the holdout loss computed across machines must match the full model's.
    with torch.no_grad():
        ref_losses = []
        for hid, hlab in examples[:3]:
            lg = ref_model(input_ids=hid).logits
            ref_losses.append(float(torch.nn.functional.cross_entropy(lg[0, :-1], hlab[0, 1:], ignore_index=-100)))
    assert abs(result["holdout_loss"] - sum(ref_losses) / 3) < 1e-4


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "mistral", "phi3", "gemma", "gemma3", "mixtral", "starcoder2"])
def test_other_model_families_match_single_machine(tmp_path, family):
    from safetensors.torch import load_file
    model_dir = make_model(str(tmp_path / "model"), family=family, layers=7)   # uneven cuts on purpose
    ports = [free_port(), free_port()]
    plan = plan_for(model_dir, [2.0, 2.0], ports)
    plan_path = str(tmp_path / "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    s, cfg = settings(target_modules=["o_proj"]), train_cfg(total_steps=4)
    for i, port in enumerate(ports):
        start_thread_worker(model_dir, plan_path, f"g{i + 1}", str(tmp_path / f"w{i + 1}"), port)
    examples = make_examples(8)
    host = Host(model_dir, plan, str(tmp_path / "host"), SECRET, s, cfg)
    try:
        init = _initial_adapter(host, tmp_path)
        host.train(examples)
        out = host.export(str(tmp_path / "adapter"))
    finally:
        host.close()
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    want, _ = reference_train(model_dir, init, examples, s, cfg, host.lr_at)
    assert max_diff(init, want) > 1e-3 and max_diff(got, want) < TOL


def test_unverified_model_family_is_refused(tmp_path):
    model_dir = make_model(str(tmp_path / "model"), family="gemma2")
    with pytest.raises(SliceError, match="not been verified"):
        plan_for(model_dir, [2.0], [7000])
    from rosetta_jane.lan.slicing import make_plan
    plan = make_plan(model_dir, 3.0, [{"name": "g1", "addr": "127.0.0.1:7000", "mem_gb": 2.0}],
                     host_reserve_gb=1.0, allow_untested=True)       # the owner can still opt in
    assert plan["allow_untested"] and plan["model_type"] == "gemma2"


def test_guest_killed_midrun_resumes_to_same_result(tmp_path):
    """Kill a guest process partway, restart it, and the run must end exactly where an
    uninterrupted single-machine run ends."""
    from safetensors.torch import load_file
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0, 2.0])
    secret_file = str(tmp_path / "secret.txt")
    with open(secret_file, "wb") as f:
        f.write(SECRET)
    s, cfg = settings(), train_cfg(total_steps=8, save_every=2)
    procs = {}

    def launch(i):
        procs[i] = start_process_worker(model_dir, plan_path, f"g{i + 1}", str(tmp_path / f"w{i + 1}"),
                                        ports[i], secret_file)

    launch(0)
    launch(1)
    examples = make_examples(10)
    host = Host(model_dir, plan, str(tmp_path / "host"), SECRET, s, cfg)
    killed = []

    def on_step(step):
        if step == 5 and not killed:      # between checkpoints (saved at 4, next at 6)
            procs[1].kill()
            procs[1].wait()
            killed.append(step)
            launch(1)                     # the owner restarts the guest program

    try:
        init = _initial_adapter(host, tmp_path)
        result = host.train(examples, on_step=on_step)
        out = host.export(str(tmp_path / "adapter"))
    finally:
        host.close()
        for p in procs.values():
            p.kill()
    assert killed == [5]
    assert result["recoveries"] >= 1 and result["steps"] == cfg.total_steps
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    want, _ = reference_train(model_dir, init, examples, s, cfg, host.lr_at)
    assert max_diff(got, want) < TOL


def test_host_restart_resumes_from_checkpoint(tmp_path):
    """Stop the host partway, start a new host process state, and it continues from the last
    checkpoint to the same result."""
    from safetensors.torch import load_file
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0])
    start_thread_worker(model_dir, plan_path, "g1", str(tmp_path / "w1"), ports[0])
    s = settings()
    examples = make_examples(10)

    class Stop(Exception):
        pass

    def bail(step):
        if step == 5:
            raise Stop()

    host = Host(model_dir, plan, str(tmp_path / "host"), SECRET, s, train_cfg(total_steps=8))
    init = _initial_adapter(host, tmp_path)
    with pytest.raises(Stop):
        host.train(examples, on_step=bail)
    host.close()

    host2 = Host(model_dir, plan, str(tmp_path / "host"), SECRET, s, train_cfg(total_steps=8))
    try:
        result = host2.train(examples)
        out = host2.export(str(tmp_path / "adapter"))
    finally:
        host2.close()
    assert result["steps"] == 8
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    want, _ = reference_train(model_dir, init, examples, s, train_cfg(total_steps=8), host2.lr_at)
    assert max_diff(got, want) < TOL


def test_nonfinite_gradients_skip_the_step_everywhere(tmp_path):
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0])
    worker = start_thread_worker(model_dir, plan_path, "g1", str(tmp_path / "w1"), ports[0])
    host = Host(model_dir, plan, str(tmp_path / "host"), SECRET, settings(),
                train_cfg(total_steps=3, fp16_scaling=True, init_scale=8.0, save_every=100))
    examples = make_examples(6)
    real_prepare = worker.__class__.handle
    poisoned = []

    def handle(self, ch, header, tensor, blob):
        if header["op"] == "prepare" and not poisoned:
            poisoned.append(True)
            for p in self.stage._params:
                if p.grad is not None:
                    p.grad.fill_(float("inf"))
                    break
        return real_prepare(self, ch, header, tensor, blob)

    worker.__class__.handle = handle
    try:
        host.connect_all()
        host._sync()
        before_host = {k: v.clone() for k, v in host.stage.adapter_state().items()}
        result = host.train(examples)
    finally:
        worker.__class__.handle = real_prepare
        host.close()
    assert result["skipped_steps"] == 1
    assert host.scale == 4.0                      # halved once after the bad step
    assert max_diff(before_host, host.stage.adapter_state()) > 0   # later steps still trained


def test_setup_mistakes_stop_the_host_at_once(tmp_path):
    """A wrong secret or a guest on a different plan must fail fast with a reason, not retry for hours."""
    import time
    from rosetta_jane.lan.protocol import AuthError, MismatchError
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0])
    start_thread_worker(model_dir, plan_path, "g1", str(tmp_path / "w1"), ports[0])
    t0 = time.time()
    host = Host(model_dir, plan, str(tmp_path / "h1"), b"another-secret-0123456789", settings(), train_cfg())
    with pytest.raises(AuthError):
        host.train(make_examples(4))
    host.close()
    other = json.loads(json.dumps(plan))
    other["stages"][0]["layers"][1] -= 1
    other["stages"][1]["layers"][0] -= 1            # host believes the guest holds one more layer
    host = Host(model_dir, other, str(tmp_path / "h2"), SECRET, settings(), train_cfg())
    with pytest.raises(MismatchError):
        host.train(make_examples(4))
    host.close()
    assert time.time() - t0 < 30


def test_wrong_secret_is_refused(tmp_path):
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0])
    start_thread_worker(model_dir, plan_path, "g1", str(tmp_path / "w1"), ports[0])
    with pytest.raises(ProtocolError):
        Channel.connect("127.0.0.1", ports[0], b"not-the-secret-not-the-secret")
    ch = Channel.connect("127.0.0.1", ports[0], SECRET)   # the right secret still works afterwards
    assert ch.call({"op": "hello"})[0]["name"] == "g1"
    ch.close()


def test_group_code_lets_matching_peers_in_and_keeps_others_out(tmp_path):
    from rosetta_jane.lan.protocol import secret_from_group_code
    assert secret_from_group_code("Blue Falcon 42") == secret_from_group_code("bluefalcon42")   # as read aloud
    assert secret_from_group_code("bluefalcon42") != secret_from_group_code("bluefalcon43")
    with pytest.raises(ValueError):
        secret_from_group_code("short")
    secret_from_group_code("public")                    # the open lobby is allowed, by name only
    model_dir, plan, plan_path, ports = _setup(tmp_path, [2.0])
    start_thread_worker(model_dir, plan_path, "g1", str(tmp_path / "w1"), ports[0],
                        secret=secret_from_group_code("blue falcon 42"))
    with pytest.raises(ProtocolError):
        Channel.connect("127.0.0.1", ports[0], secret_from_group_code("red falcon 42"))
    with pytest.raises(ProtocolError):
        Channel.connect("127.0.0.1", ports[0], secret_from_group_code("public"))
    ch = Channel.connect("127.0.0.1", ports[0], secret_from_group_code("BLUE FALCON 42"))
    assert ch.call({"op": "hello"})[0]["name"] == "g1"
    ch.close()


def test_traffic_is_encrypted_on_the_wire():
    """Relay a connection through a tap and check the tensor never appears in the clear."""
    import threading
    key = b"s" * 32
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    tap_listener = socket.socket()
    tap_listener.bind(("127.0.0.1", 0))
    tap_listener.listen(1)
    seen = bytearray()

    def pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                seen.extend(data)
                dst.sendall(data)
        except OSError:
            pass

    def tap():
        client, _ = tap_listener.accept()
        upstream = socket.create_connection(listener.getsockname())
        threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
        pump(upstream, client)

    got = {}

    def server():
        conn, _ = listener.accept()
        ch = Channel.accept(conn, key)
        got["msg"] = ch.recv()
        ch.send({"op": "x_ok"})

    threading.Thread(target=tap, daemon=True).start()
    t = threading.Thread(target=server, daemon=True)
    t.start()
    marker = torch.full((64,), 1234.5, dtype=torch.float32)
    ch = Channel.connect("127.0.0.1", tap_listener.getsockname()[1], key)
    ch.call({"op": "x", "note": "plaintext-marker-zzz"}, tensor=marker)
    t.join(10)
    assert torch.equal(got["msg"][1], marker)
    plain = tensor_to_bytes(marker)[1]
    assert len(seen) > len(plain)                       # the tap really saw the traffic
    assert plain[:32] not in bytes(seen) and b"plaintext-marker-zzz" not in bytes(seen)
    assert b"LNLR" not in bytes(seen)                   # not even the protocol's own header is visible

    # With encryption off the same data is readable, which is what --no-encrypt warns about.
    a, b = socket.socketpair()
    Channel(a, key).send({"op": "x", "note": "plaintext-marker-zzz"})
    assert b"plaintext-marker-zzz" in b.recv(4096)
    # An encrypted worker refuses a client that is not encrypting, and the reverse.
    def refuse_server():
        conn, _ = listener.accept()
        try:
            Channel.accept(conn, key)
            got["plain_client"] = "accepted"
        except ProtocolError:
            got["plain_client"] = "refused"
    t2 = threading.Thread(target=refuse_server, daemon=True)
    t2.start()
    with pytest.raises((ProtocolError, ConnectionError, OSError)):
        Channel.connect("127.0.0.1", listener.getsockname()[1], key, encrypt=False, timeout=5)
    t2.join(10)
    assert got["plain_client"] == "refused"
    for s in (a, b, listener, tap_listener):
        s.close()


def test_tampered_frame_is_rejected():
    a, b = socket.socketpair()
    key = b"k" * 32
    sender, receiver = Channel(a, key), Channel(b, key)
    sender.send({"op": "x"}, tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3))
    raw = bytearray(b.recv(1 << 16, socket.MSG_PEEK))
    b.recv(len(raw))
    raw[-40] ^= 0x01                                  # flip one bit inside the tensor payload
    c, d = socket.socketpair()
    c.sendall(bytes(raw))
    with pytest.raises(ProtocolError):
        Channel(d, key).recv()
    # replaying an old, valid frame is rejected too (sequence number moved on)
    e, f = socket.socketpair()
    s2, r2 = Channel(e, key), Channel(f, key)
    s2.send({"op": "one"})
    first = f.recv(1 << 16, socket.MSG_PEEK)
    r2.recv()
    e.sendall(first)
    with pytest.raises(ProtocolError):
        r2.recv()
    for s in (a, b, c, d, e, f):
        s.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_tensor_bits_survive_the_wire(dtype):
    t = (torch.randn(3, 5) * 1e4).to(dtype)
    meta, payload = tensor_to_bytes(t)
    back = tensor_from_bytes(meta, payload)
    assert back.dtype == dtype and torch.equal(back, t)
    with pytest.raises(ProtocolError):
        tensor_from_bytes({"dtype": meta["dtype"], "shape": [3, 6]}, payload)
    with pytest.raises(ProtocolError):
        tensor_from_bytes({"dtype": "complex64", "shape": [3, 5]}, payload)


def test_plan_and_slice_checks(tmp_path):
    assert split_layers(10, [1, 1]) == [(0, 5), (5, 10)]
    assert split_layers(7, [12.0, 6.5, 6.5])[-1][1] == 7
    assert all(b > a for a, b in split_layers(3, [100.0, 1.0, 1.0]))
    with pytest.raises(SliceError):
        split_layers(2, [1, 1, 1])
    model_dir = make_model(str(tmp_path / "model"))
    other_dir = make_model(str(tmp_path / "other"), layers=4)
    assert model_fingerprint(model_dir) != model_fingerprint(other_dir)
    plan = plan_for(model_dir, [2.0], [7000])
    with pytest.raises(SliceError):           # a guest with a different base model must be stopped
        make_slice(other_dir, str(tmp_path / "s"), plan, "g1")
    bad = json.loads(json.dumps(plan))
    bad["stages"][1]["layers"][0] += 1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SliceError):
        load_plan(str(p))
    # a guest slice holds only its own layers and a token table of negligible size
    from safetensors import safe_open
    out = make_slice(model_dir, str(tmp_path / "g1"), plan, "g1")
    with safe_open(os.path.join(out, "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert keys and all(k.startswith("model.layers.") for k in keys)

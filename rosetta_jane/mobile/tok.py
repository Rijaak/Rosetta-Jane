"""Turning text into token numbers and back, without the tokenizers library.

Reads the ordinary `tokenizer.json` that ships with a model and follows its own
description of itself: how text is cleaned up, cut into words, and merged into
tokens. Only the building blocks used by the supported model families are
implemented. Anything else is refused by name rather than tokenized wrongly,
because a wrong token is a wrong question.

Chat formatting uses the model's own template (`chat_template`) through Jinja,
the same way the desktop program does.
"""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Dict, List, Optional, Tuple

try:
    import regex as re                             # needed for patterns such as \p{L}
except ImportError:                                # pragma: no cover
    import re


class Unsupported(ValueError):
    pass


def _byte_table() -> Dict[int, str]:
    """The fixed map from bytes to printable characters that byte-level tokenizers use."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = keep[:]
    n = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


_BYTE_CHAR = _byte_table()
_CHAR_BYTE = {c: b for b, c in _BYTE_CHAR.items()}
_GPT2_SPLIT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def _pattern(spec) -> "re.Pattern":
    if isinstance(spec, dict) and "Regex" in spec:
        return re.compile(spec["Regex"])
    if isinstance(spec, dict) and "String" in spec:
        return re.compile(re.escape(spec["String"]))
    raise Unsupported(f"tokenizer pattern {spec!r} is not supported")


# -- cleaning up text -----------------------------------------------------------------------
def _normalizer(spec):
    if spec is None:
        return lambda s: s
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [_normalizer(x) for x in spec["normalizers"]]

        def run(s):
            for step in steps:
                s = step(s)
            return s
        return run
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        return lambda s: unicodedata.normalize(kind, s)
    if kind == "Prepend":
        return lambda s: spec["prepend"] + s if s else s
    if kind == "Replace":
        pattern, content = _pattern(spec["pattern"]), spec["content"]
        return lambda s: pattern.sub(lambda m: content, s)
    if kind == "Lowercase":
        return lambda s: s.lower()
    if kind == "Strip":
        left, right = spec.get("strip_left", True), spec.get("strip_right", True)
        return lambda s: (s.lstrip() if left else s) if not right else (s.strip() if left else s.rstrip())
    raise Unsupported(f"tokenizer cleanup step {kind!r} is not supported on this device")


# -- cutting text into words ----------------------------------------------------------------
def _pre_tokenizer(spec):
    """Returns f(pieces, first_section) -> pieces, where each piece is a word to tokenize on its own."""
    if spec is None:
        return lambda pieces, first: pieces
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [_pre_tokenizer(x) for x in spec["pretokenizers"]]

        def run(pieces, first):
            for step in steps:
                pieces = step(pieces, first)
            return pieces
        return run
    if kind == "ByteLevel":
        prefix, use_regex = spec.get("add_prefix_space", True), spec.get("use_regex", True)
        split = re.compile(_GPT2_SPLIT)

        def run(pieces, first):
            out = []
            for piece in pieces:
                if prefix and not piece.startswith(" "):
                    piece = " " + piece
                words = split.findall(piece) if use_regex else [piece]
                out.extend("".join(_BYTE_CHAR[b] for b in w.encode("utf-8")) for w in words if w)
            return out
        return run
    if kind == "Metaspace":
        mark = spec.get("replacement", "▁")
        scheme = spec.get("prepend_scheme") or ("always" if spec.get("add_prefix_space", True) else "never")
        do_split = spec.get("split", True)

        def run(pieces, first):
            out = []
            for n, piece in enumerate(pieces):
                piece = piece.replace(" ", mark)
                if piece and not piece.startswith(mark) and (scheme == "always" or (scheme == "first" and first and n == 0)):
                    piece = mark + piece
                if do_split:
                    out.extend(w for w in re.findall(f"{re.escape(mark)}?[^{re.escape(mark)}]*", piece) if w)
                elif piece:
                    out.append(piece)
            return out
        return run
    if kind == "Split":
        pattern, behavior, invert = _pattern(spec["pattern"]), spec.get("behavior", "Isolated"), spec.get("invert", False)
        if behavior not in ("Isolated", "Removed"):
            raise Unsupported(f"tokenizer split behaviour {behavior!r} is not supported on this device")

        def run(pieces, first):
            out = []
            for piece in pieces:
                at = 0
                for m in pattern.finditer(piece):
                    if m.end() == m.start():
                        continue
                    gap, hit = piece[at:m.start()], m.group(0)
                    if invert:                     # the matches are the words; what lies between is the separator
                        if gap and behavior == "Isolated":
                            out.append(gap)
                        out.append(hit)
                    else:
                        if gap:
                            out.append(gap)
                        if behavior == "Isolated":
                            out.append(hit)
                    at = m.end()
                tail = piece[at:]
                if tail and not (invert and behavior == "Removed"):
                    out.append(tail)
            return out
        return run
    if kind in ("Whitespace", "WhitespaceSplit"):
        pattern = re.compile(r"\w+|[^\w\s]+" if kind == "Whitespace" else r"\S+")
        return lambda pieces, first: [w for piece in pieces for w in pattern.findall(piece)]
    raise Unsupported(f"tokenizer word-splitting step {kind!r} is not supported on this device")


# -- turning tokens back into text ----------------------------------------------------------------
def _decoder(spec):
    """Returns f(tokens) -> tokens (a pipeline; the pieces are joined at the end)."""
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [_decoder(x) for x in spec["decoders"]]

        def run(tokens):
            for step in steps:
                tokens = step(tokens)
            return tokens
        return run
    if kind == "ByteLevel":
        def run(tokens):
            raw = bytearray()
            for t in tokens:
                for c in t:
                    if c in _CHAR_BYTE:
                        raw.append(_CHAR_BYTE[c])
                    else:
                        raw.extend(c.encode("utf-8"))
            return [raw.decode("utf-8", errors="replace")]
        return run
    if kind == "Metaspace":
        mark = spec.get("replacement", "▁")
        scheme = spec.get("prepend_scheme") or ("always" if spec.get("add_prefix_space", True) else "never")

        def run(tokens):
            out = []
            for i, t in enumerate(tokens):
                t = t.replace(mark, " ")
                if i == 0 and scheme != "never" and t.startswith(" "):
                    t = t[1:]
                out.append(t)
            return out
        return run
    if kind == "Replace":
        pattern, content = _pattern(spec["pattern"]), spec["content"]
        return lambda tokens: [pattern.sub(lambda m: content, t) for t in tokens]
    if kind == "ByteFallback":
        def run(tokens):
            out, pending = [], bytearray()

            def flush():
                if pending:
                    text = bytes(pending).decode("utf-8", errors="ignore")
                    out.append(text if text else "�" * len(pending))
                    pending.clear()
            for t in tokens:
                if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                    try:
                        pending.append(int(t[3:5], 16))
                        continue
                    except ValueError:
                        pass
                flush()
                out.append(t)
            flush()
            return out
        return run
    if kind == "Fuse":
        return lambda tokens: ["".join(tokens)]
    if kind == "Strip":
        content, start, stop = spec.get("content", " "), spec.get("start", 0), spec.get("stop", 0)

        def run(tokens):
            out = []
            for t in tokens:
                a = 0
                while a < start and t[a:a + len(content)] == content:
                    a += len(content)
                b = len(t)
                n = 0
                while n < stop and b - len(content) >= a and t[b - len(content):b] == content:
                    b -= len(content)
                    n += 1
                out.append(t[a:b])
            return out
        return run
    raise Unsupported(f"tokenizer decoding step {kind!r} is not supported on this device")


class Tokenizer:
    def __init__(self, model_dir: str):
        path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise Unsupported("this model folder has no tokenizer.json, which asking and training need")
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        model = spec.get("model") or {}
        self.kind = model.get("type") or ("BPE" if "merges" in model else None)
        if self.kind not in ("BPE", "WordLevel"):
            raise Unsupported(f"tokenizer type {self.kind!r} is not supported on this device")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix") or model.get("dropout"):
            raise Unsupported("this tokenizer uses word-piece markers that are not supported on this device")
        self.vocab: Dict[str, int] = dict(model["vocab"])
        self.unk = model.get("unk_token")
        self.byte_fallback = bool(model.get("byte_fallback"))
        self.ignore_merges = bool(model.get("ignore_merges"))
        self.ranks: Dict[Tuple[str, str], int] = {}
        for i, merge in enumerate(model.get("merges", [])):
            a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
            self.ranks[(a, b)] = i
        self.normalize = _normalizer(spec.get("normalizer"))
        self.pre = _pre_tokenizer(spec.get("pre_tokenizer"))
        self.decode_steps = _decoder(spec.get("decoder"))
        self.tokens: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.special_ids = set()
        added = []
        for entry in spec.get("added_tokens", []):
            self.tokens[entry["id"]] = entry["content"]
            self.vocab.setdefault(entry["content"], entry["id"])
            if entry.get("special"):
                self.special_ids.add(entry["id"])
            added.append(entry)
        self._added = {e["content"]: e for e in added}
        ordered = sorted(self._added, key=len, reverse=True)
        self._added_re = re.compile("|".join(re.escape(t) for t in ordered)) if ordered else None
        self._cache: Dict[str, List[int]] = {}
        self._load_chat(model_dir)

    # -- text -> ids ---------------------------------------------------------------------------
    def _word(self, word: str) -> List[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        if self.kind == "WordLevel":
            ids = [self.vocab[word] if word in self.vocab else self._unknown(word)]
        elif self.ignore_merges and word in self.vocab:
            ids = [self.vocab[word]]
        else:
            parts = list(word)
            while len(parts) > 1:
                best, where = None, -1
                for i in range(len(parts) - 1):
                    rank = self.ranks.get((parts[i], parts[i + 1]))
                    if rank is not None and (best is None or rank < best):
                        best, where = rank, i
                if best is None:
                    break
                a, b = parts[where], parts[where + 1]
                merged, i = [], 0
                while i < len(parts):
                    if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(parts[i])
                        i += 1
                parts = merged
            ids = []
            for part in parts:
                if part in self.vocab:
                    ids.append(self.vocab[part])
                elif self.byte_fallback and all(f"<0x{b:02X}>" in self.vocab for b in part.encode("utf-8")):
                    ids.extend(self.vocab[f"<0x{b:02X}>"] for b in part.encode("utf-8"))
                else:
                    ids.append(self._unknown(part))
        if len(self._cache) < 50000:
            self._cache[word] = ids
        return ids

    def _unknown(self, piece: str) -> int:
        if self.unk is None or self.unk not in self.vocab:
            raise Unsupported(f"the tokenizer has no token for {piece!r}")
        return self.vocab[self.unk]

    def _plain(self, text: str, first: bool) -> List[int]:
        text = self.normalize(text)
        if not text:
            return []
        out: List[int] = []
        for word in self.pre([text], first):
            out.extend(self._word(word))
        return out

    def encode(self, text: str) -> List[int]:
        """Token numbers for text, with no start or end markers added (chat templates write their own)."""
        if self._added_re is None:
            return self._plain(text, True)
        out: List[int] = []
        at, first = 0, True
        for m in self._added_re.finditer(text):
            entry = self._added[m.group(0)]
            start, end = m.start(), m.end()
            if entry.get("single_word"):
                before = text[start - 1] if start > 0 else " "
                after = text[end] if end < len(text) else " "
                if before.isalnum() or after.isalnum():
                    continue
            if entry.get("lstrip"):
                while start > at and text[start - 1].isspace():
                    start -= 1
            if entry.get("rstrip"):
                while end < len(text) and text[end].isspace():
                    end += 1
            if start < at:
                continue
            if start > at:
                out.extend(self._plain(text[at:start], first))
            out.append(entry["id"])
            at, first = end, False
        if at < len(text):
            out.extend(self._plain(text[at:], first))
        return out

    # -- ids -> text ---------------------------------------------------------------------------
    def decode(self, ids, skip_special: bool = True) -> str:
        tokens: List[str] = []
        for i in ids:
            i = int(i)
            token = self.tokens.get(i)
            if token is None or (skip_special and i in self.special_ids):
                continue
            tokens.append(token)
        if self.decode_steps is None:
            return " ".join(tokens)
        return "".join(self.decode_steps(tokens))

    # -- chat ------------------------------------------------------------------------------------
    def _load_chat(self, model_dir: str) -> None:
        self.chat_template: Optional[str] = None
        self.bos_token = self.eos_token = ""
        self.eos_token_id: Optional[int] = None
        cfg = {}
        path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)

        def text(value) -> str:
            return value.get("content", "") if isinstance(value, dict) else (value or "")
        self.bos_token, self.eos_token = text(cfg.get("bos_token")), text(cfg.get("eos_token"))
        self.eos_token_id = self.vocab.get(self.eos_token) if self.eos_token else None
        template = cfg.get("chat_template")
        jinja_file = os.path.join(model_dir, "chat_template.jinja")
        if os.path.exists(jinja_file):
            with open(jinja_file, encoding="utf-8") as f:
                template = f.read()
        if isinstance(template, list):             # several named templates: take the default one
            template = next((t.get("template") for t in template if t.get("name") == "default"), None)
        self.chat_template = template if isinstance(template, str) else None

    def chat_text(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        if not self.chat_template:
            raise Unsupported("this model has no chat template, so its chat format is unknown")
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise TemplateError(message)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **{k: v for k, v in kw.items() if k == "indent"})
        return env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, bos_token=self.bos_token,
            eos_token=self.eos_token, tools=None)

    def chat_ids(self, messages: List[dict], add_generation_prompt: bool = True) -> List[int]:
        return self.encode(self.chat_text(messages, add_generation_prompt))

    def pair_example(self, question: str, answer: str, max_tokens: int):
        """One training example: (ids, labels) with the question masked out of the grading, or None if too long."""
        user = [{"role": "user", "content": question}]
        prompt = self.chat_ids(user, True)
        full = self.chat_ids(user + [{"role": "assistant", "content": answer}], False)
        if full[:len(prompt)] != prompt:
            raise Unsupported("this model's chat template does not put the question first; cannot train with it")
        full = full[:max_tokens]
        if len(full) - len(prompt) < 1:
            return None
        return full, [-100] * len(prompt) + full[len(prompt):]

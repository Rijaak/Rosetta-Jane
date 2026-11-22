"""The phone's own tokenizer and chat formatting, checked against the standard libraries.

Real model tokenizers cannot be downloaded where these tests run, so tokenizers of each
supported kind are built here with the standard library, saved in the standard file
format, and the phone's reader must agree with the standard one on every sample.
"""
import json
import os
import random
import tempfile

import pytest
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from rosetta_jane.mobile.tok import Tokenizer as Mine
from rosetta_jane.mobile.tok import Unsupported

corpus = ["The quick brown fox jumps over the lazy dog.", "Rosetta-Jane splits one model across many devices!",
          "Ünïcödé façade naïve café — “quotes” and ‘marks’…", "数学は楽しい。你好，世界！", "def f(x):\n    return x**2 + 1  # code",
          "emoji 🙂🚀 and tabs\tand   spaces", "It's what we've said: don't, won't, I'll, you're, I'd, I'm.", "1234567890 3.14159 2e-5 $9.99 50%"] * 6
random.seed(1)
def samples():
    out = list(corpus[:8]) + ["", " ", "  leading and trailing  ", "a", "\n\n", "x<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "<s>hello</s>", "tab\there", "ß ẞ ﬁ ½", "‍zero‍width", " nbsp em"]
    words = " ".join(corpus).split()
    for _ in range(60):
        out.append(" ".join(random.choice(words) for _ in range(random.randint(1, 12))) + random.choice(["", " ", "!", "\n"]))
    return out
SPECIAL = [AddedToken("<|im_start|>", special=True), AddedToken("<|im_end|>", special=True), AddedToken("<s>", special=True), AddedToken("</s>", special=True)]
def build(kind):
    if kind == "bytelevel":      # GPT-2 / Qwen2 style
        t = Tokenizer(models.BPE()); t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); t.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    elif kind == "llama3":       # Split by regex, then bytes; whole known words taken directly
        t = Tokenizer(models.BPE(ignore_merges=True))
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        t.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    elif kind == "sentencepiece":  # Llama-2 / Mistral style
        t = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
        t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
        t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
        tr = trainers.BpeTrainer(vocab_size=700, special_tokens=["<unk>"] + [f"<0x{b:02X}>" for b in range(256)])
    elif kind == "metaspace":
        t = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
        t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first"); t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first")
        tr = trainers.BpeTrainer(vocab_size=700, special_tokens=["<unk>"] + [f"<0x{b:02X}>" for b in range(256)])
    else:                        # words, as the project's own test tokenizer
        t = Tokenizer(models.WordLevel(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace()
        tr = trainers.WordLevelTrainer(vocab_size=300, special_tokens=["[UNK]"])
    if kind == "sentencepiece":
        t.train_from_iterator([c.replace(" ", "▁") for c in corpus], tr)
    else:
        t.train_from_iterator(corpus, tr)
    t.add_tokens(SPECIAL)
    return t


KINDS = ("bytelevel", "llama3", "sentencepiece", "metaspace", "wordlevel")
CHATML = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
INST = ("{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% else %}"
        "{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}")


def saved(kind, template=None):
    folder = tempfile.mkdtemp()
    standard = build(kind)
    standard.save(os.path.join(folder, "tokenizer.json"))
    if template:
        with open(os.path.join(folder, "tokenizer_config.json"), "w") as f:
            json.dump({"chat_template": template, "bos_token": "<s>", "tokenizer_class": "PreTrainedTokenizerFast",
                       "eos_token": {"content": "</s>", "__type": "AddedToken"}}, f)
    return folder, standard


@pytest.mark.parametrize("kind", KINDS)
def test_text_becomes_the_same_tokens_and_back(kind):
    folder, standard = saved(kind)
    mine = Mine(folder)
    for text in samples():
        want = standard.encode(text, add_special_tokens=False).ids
        assert mine.encode(text) == want, repr(text)
        for skip in (True, False):
            assert mine.decode(want, skip) == standard.decode(want, skip_special_tokens=skip), repr(text)
        # A reply arrives one token at a time; every prefix must decode the same way too.
        for n in range(1, min(len(want), 12) + 1):
            assert mine.decode(want[:n]) == standard.decode(want[:n]), (repr(text), n)


@pytest.mark.parametrize("kind,template", [("bytelevel", CHATML), ("llama3", CHATML), ("sentencepiece", INST),
                                           ("metaspace", INST), ("wordlevel", CHATML)])
def test_chats_are_laid_out_the_way_the_model_expects(kind, template):
    from transformers import AutoTokenizer
    folder, _ = saved(kind, template)
    mine, standard = Mine(folder), AutoTokenizer.from_pretrained(folder)
    chat = [{"role": "user", "content": "The quick brown fox, it's 3.14!"},
            {"role": "assistant", "content": "A lazy dog jumps over many devices."},
            {"role": "user", "content": "and tabs\tand   spaces?"}]
    for messages, prompt in ((chat[:1], True), (chat[:2], False), (chat, True)):
        want_text = standard.apply_chat_template(messages, tokenize=False, add_generation_prompt=prompt)
        assert mine.chat_text(messages, prompt) == want_text
        want = standard.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt)
        want = list(getattr(want, "input_ids", want))
        assert mine.chat_ids(messages, prompt) == want
    assert mine.eos_token_id == standard.eos_token_id
    if template is INST:
        with pytest.raises(Exception, match="Only user and assistant"):
            mine.chat_text([{"role": "system", "content": "x"}])


def test_training_pairs_are_prepared_the_way_the_desktop_prepares_them():
    from transformers import AutoTokenizer

    from rosetta_jane.lan.data import tokenize_pairs
    folder, _ = saved("bytelevel", CHATML)
    mine, standard = Mine(folder), AutoTokenizer.from_pretrained(folder)
    pairs = [{"question": "The quick brown fox?", "answer": "jumps over the lazy dog."},
             {"question": "Rosetta-Jane splits what?", "answer": "one model across many devices!"}]
    want, _ = tokenize_pairs(pairs, standard, 64)
    for row, (ids, labels) in zip(pairs, want):
        got = mine.pair_example(row["question"], row["answer"], 64)
        assert got == (ids[0].tolist(), labels[0].tolist())
    assert mine.pair_example("a question that leaves no room", "x", 3) is None


def test_unknown_tokenizer_designs_are_refused_by_name(tmp_path):
    folder, _ = saved("bytelevel")
    path = os.path.join(folder, "tokenizer.json")
    spec = json.load(open(path))
    for change, word in (({"model": {**spec["model"], "type": "Unigram"}}, "Unigram"),
                         ({"normalizer": {"type": "BertNormalizer"}}, "BertNormalizer"),
                         ({"pre_tokenizer": {"type": "Punctuation"}}, "Punctuation"),
                         ({"decoder": {"type": "WordPiece"}}, "WordPiece")):
        json.dump({**spec, **change}, open(path, "w"))
        with pytest.raises(Unsupported, match=word):
            Mine(folder)
    with pytest.raises(Unsupported, match="no tokenizer.json"):
        Mine(str(tmp_path))
    json.dump(spec, open(path, "w"))
    with pytest.raises(Unsupported, match="no chat template"):
        Mine(folder).chat_text([{"role": "user", "content": "hi"}])

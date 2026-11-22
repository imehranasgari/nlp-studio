# -*- coding: utf-8 -*-
"""
File: tokenizer.py
Tozihat: Marhale chaharom pipeline (Tokenization - Jalase 2).
"""

import re


class TextTokenizer:
    def __init__(self, config: dict):
        self.cfg = config.get("tokenization", {})
        self.abbreviations = r"(?:dr|mr|mrs|ms|prof|sr|jr|vs|etc|e\.g|i\.e|u\.s\.a|u\.k)"

    def split_sentences(self, text: str) -> list[str]:
        protected = re.sub(
            rf"\b({self.abbreviations})\.",
            lambda m: m.group(1) + "<DOT>",
            text,
            flags=re.IGNORECASE
        )
        raw_sents = re.split(r"(?<!\d)[.!?؟]+(?!\d)\s*", protected)
        return [s.replace("<DOT>", ".").strip() for s in raw_sents if s.strip()]

    def chunk_tokens(self, tokens: list[str], chunk_size: int, overlap_ratio: float = 0.15) -> list[list[str]]:
        if chunk_size <= 0 or len(tokens) <= chunk_size:
            return [tokens]
        step = max(1, int(chunk_size * (1 - overlap_ratio)))
        chunks = []
        for i in range(0, len(tokens), step):
            chunks.append(tokens[i:i + chunk_size])
            if i + chunk_size >= len(tokens):
                break
        return chunks

    def _bpe_subword_demo(self, tokens: list[str]) -> list[str]:
        subwords = []
        prefixes = ("un", "re", "pre", "dis", "غیر", "بی")
        suffixes = ("ing", "ed", "ly", "ness", "های", "ترین", "انه")
        for tok in tokens:
            if tok.startswith("<") and tok.endswith(">"):
                subwords.append(tok)
                continue
            matched = False
            for pref in prefixes:
                if tok.startswith(pref) and len(tok) > len(pref) + 2:
                    subwords.extend([pref, "##" + tok[len(pref):]])
                    matched = True
                    break
            if not matched:
                for suff in suffixes:
                    if tok.endswith(suff) and len(tok) > len(suff) + 2:
                        subwords.extend([tok[:-len(suff)], "##" + suff])
                        matched = True
                        break
            if not matched:
                subwords.append(tok)
        return subwords

    def tokenize(self, text: str) -> tuple[list[str], list[str], dict]:
        method = self.cfg.get("method", "regex_smart")
        sentences = self.split_sentences(text) if self.cfg.get("sentence_split", True) else [text]
        smart_pattern = r"<[a-zA-Z]+>|[\u0600-\u06FF]+(?:\u200c[\u0600-\u06FF]+)*|[a-zA-Z0-9_]+"

        if method == "whitespace":
            tokens = text.split()
        elif method == "regex_smart":
            tokens = re.findall(smart_pattern, text)
        elif method == "nltk":
            try:
                from nltk.tokenize import wordpunct_tokenize
                tokens = wordpunct_tokenize(text)
            except Exception:
                tokens = re.findall(smart_pattern, text)
        elif method == "hazm":
            try:
                from hazm import word_tokenize
                tokens = word_tokenize(text)
            except ImportError:
                tokens = re.findall(smart_pattern, text)
                method = "regex_smart (hazm_not_installed)"
        elif method == "subword_bpe_demo":
            base_tokens = re.findall(smart_pattern, text)
            tokens = self._bpe_subword_demo(base_tokens)
        else:
            tokens = text.split()

        chunk_size = self.cfg.get("chunk_size", 0)
        chunks = self.chunk_tokens(tokens, chunk_size, self.cfg.get("chunk_overlap_ratio", 0.15))

        # Sakht Token IDs, One-Hot Vectors, va Padded Batch Tensor (Slide 1 & 12 PDF 04)
        special_vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        vocab_map = dict(special_vocab)
        for tok in tokens:
            if tok not in vocab_map:
                vocab_map[tok] = len(vocab_map)

        token_ids = [vocab_map[tok] for tok in tokens]

        # Padded / Truncated fixed-length sequence (max_len = 16 baraye namayesh dar UI)
        max_seq_len = 16
        padded_ids = token_ids[:max_seq_len]
        if len(padded_ids) < max_seq_len:
            padded_ids = padded_ids + [0] * (max_seq_len - len(padded_ids))
        attention_mask = [1 if tid != 0 else 0 for tid in padded_ids]

        # Bardar vizhegi (Feature Vector) baraye har token:
        # 1) One-Hot vector rooye vazhegan yekta
        # 2) Bardar vizhegi adadi har token: [token_id, char_len, has_zwnj, is_ascii]
        unique_list = list(dict.fromkeys(tokens))[:10]
        one_hot_preview = {
            "columns": unique_list,
            "rows": [
                {
                    "token": tok,
                    "token_id": vocab_map[tok],
                    "token_features": {
                        "token_id": vocab_map[tok],
                        "char_len": len(tok),
                        "has_zwnj": 1 if "\u200c" in tok else 0,
                        "is_ascii": 1 if tok.isascii() else 0
                    },
                    "one_hot": [1 if tok == col else 0 for col in unique_list]
                }
                for tok in tokens[:8]
            ]
        }

        audit = {
            "tokenizer_used": method,
            "sentence_count": len(sentences),
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "chunk_count": len(chunks),
            "token_ids_sample": token_ids[:20],
            "padded_sequence_len_16": {
                "input_ids": padded_ids,
                "attention_mask": attention_mask
            },
            "one_hot_preview": one_hot_preview
        }
        return sentences, tokens, audit

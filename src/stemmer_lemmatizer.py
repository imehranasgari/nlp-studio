# -*- coding: utf-8 -*-
"""
File: stemmer_lemmatizer.py
Tozihat: Marhale sheshom pipeline (Stemming & Lemmatization - Jalase 3).
"""

import re


class MorphologyProcessor:
    def __init__(self, config: dict):
        self.cfg = config.get("morphology", {})
        self.protected_terms = {
            "nlp", "llm", "llms", "rag", "python", "pytorch", "tensorflow", "bert", "gpt",
            "bpe", "tfidf", "tf", "idf", "pos", "ner", "ocr", "api", "html", "json", "cpu", "gpu"
        }
        self.fa_verb_lemmas = {
            "می‌روم": "رفت", "میروم": "رفت", "رفتم": "رفت", "رفته": "رفت", "می‌رود": "رفت", "می‌روند": "رفت",
            "می‌خواهم": "خواست", "میخواهم": "خواست", "خواستم": "خواست", "می‌خواهد": "خواست",
            "می‌کنم": "کرد", "میکنم": "کرد", "کردم": "کرد", "می‌کند": "کرد", "می‌کنند": "کرد", "کرده": "کرد",
            "می‌خوانم": "خواند", "می‌خوانید": "خواند", "می‌خواند": "خواند", "خواندم": "خواند",
            "نوشتم": "نوشت", "می‌نویسم": "نوشت", "نوشته": "نوشت",
            "دیده‌ام": "دید", "دیدم": "دید", "می‌بیند": "دید",
            "گفتم": "گفت", "می‌گویم": "گفت", "گفته": "گفت",
            "می‌دهد": "داد", "می‌دهند": "داد", "دادم": "داد",
            "می‌گیرند": "گرفت", "می‌گیرد": "گرفت", "گرفتم": "گرفت",
        }
        self.en_irregular_lemmas = {
            "was": "be", "were": "be", "is": "be", "are": "be", "am": "be", "been": "be",
            "studies": "study", "studying": "study", "studied": "study",
            "running": "run", "ran": "run", "runs": "run",
            "better": "good", "best": "good", "worse": "bad", "worst": "bad",
            "mice": "mouse", "children": "child", "men": "man", "women": "woman",
            "corpora": "corpus", "analyses": "analysis", "criteria": "criterion",
            "processing": "process", "analyzing": "analyze", "producing": "produce",
            "pipelines": "pipeline", "experiments": "experiment", "results": "result",
            "datasets": "dataset", "texts": "text", "books": "book", "cars": "car",
            "happens": "happen", "splits": "split", "removes": "remove", "negations": "negation",
            "works": "work", "turning": "turn", "reviews": "review", "positives": "positive",
            "unhappiness": "unhappiness", "preprocessing": "preprocess"
        }

    def _fa_stem_rule(self, token: str) -> str:
        if token in {"تهران", "ایران", "زمان", "زبان", "فراوان", "تومان", "واژگان"}:
            return token
        if token == "دانشجویان":
            return "دانشجو"
        t = re.sub(r"\u200c(های|هایی|ها|ترین|تر|ام|ات|اش|مان|تان|شان)$", "", token)
        if t != token and len(t) >= 2:
            return t
        t2 = re.sub(r"(های|ها|ترین|تر)$", "", t)
        return t2 if len(t2) >= 3 else t

    def _fa_lemma_rule(self, token: str) -> str:
        if token in self.fa_verb_lemmas:
            return self.fa_verb_lemmas[token]
        if token == "دانشجویان":
            return "دانشجو"
        if token == "مقالات":
            return "مقاله"
        return self._fa_stem_rule(token)

    def _en_stem_rule(self, token: str, kind: str) -> str:
        try:
            if kind == "porter":
                from nltk.stem import PorterStemmer
                return PorterStemmer().stem(token)
            elif kind == "lancaster":
                from nltk.stem import LancasterStemmer
                return LancasterStemmer().stem(token)
            else:
                from nltk.stem import SnowballStemmer
                return SnowballStemmer("english").stem(token)
        except Exception:
            if token.endswith("ies") and len(token) > 4:
                return token[:-3] + "i"
            if token.endswith("ing") and len(token) > 5:
                return token[:-3]
            if token.endswith("ed") and len(token) > 4:
                return token[:-2]
            if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                return token[:-1]
            return token

    def _en_lemma_rule(self, token: str, pos: str) -> str:
        if token in self.en_irregular_lemmas:
            return self.en_irregular_lemmas[token]
        try:
            from nltk.stem import WordNetLemmatizer
            return WordNetLemmatizer().lemmatize(token, pos=pos)
        except Exception:
            if token.endswith("ies") and len(token) > 4:
                return token[:-3] + "y"
            if token.endswith("ing") and len(token) > 5:
                return token[:-3]
            if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                return token[:-1]
            return token

    def _is_persian(self, token: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", token))

    def process(self, tokens: list[str]) -> tuple[list[str], dict]:
        mode = self.cfg.get("mode", "lemma")
        if mode == "none":
            return tokens, {"mode": "none", "changed_count": 0, "mappings": []}

        en_stemmer = self.cfg.get("en_stemmer", "snowball")
        en_pos = self.cfg.get("en_lemma_pos", "v")
        fa_method = self.cfg.get("fa_method", "rule_based")

        processed, mappings = [], []
        for tok in tokens:
            if (tok.startswith("<") and tok.endswith(">")) or tok.lower() in self.protected_terms:
                processed.append(tok)
                continue

            if self._is_persian(tok):
                if fa_method == "hazm":
                    try:
                        from hazm import Stemmer, Lemmatizer
                        res = Lemmatizer().lemmatize(tok) if mode == "lemma" else Stemmer().stem(tok)
                    except ImportError:
                        res = self._fa_lemma_rule(tok) if mode == "lemma" else self._fa_stem_rule(tok)
                else:
                    res = self._fa_lemma_rule(tok) if mode == "lemma" else self._fa_stem_rule(tok)
            else:
                res = self._en_stem_rule(tok, en_stemmer) if mode == "stem" else self._en_lemma_rule(tok, en_pos)

            processed.append(res)
            if res != tok:
                mappings.append({"original": tok, "root": res})

        return processed, {"mode": mode, "changed_count": len(mappings), "mappings": mappings}

# -*- coding: utf-8 -*-
"""
File: normalizer.py
Tozihat: Marhale sevom pipeline (4-Level Normalization - Jalase 2).
"""

import re
import unicodedata


class TextNormalizer:
    def __init__(self, config: dict):
        self.cfg = config.get("normalization", {})
        self.arabic_to_persian_map = {
            "\u064a": "\u06cc",  # ي -> ی
            "\u0649": "\u06cc",  # ى -> ی
            "\u0643": "\u06a9",  # ك -> ک
            "\u0629": "\u0647",  # ة -> ه
            "\u0623": "\u0627",  # أ -> ا
            "\u0625": "\u0627",  # إ -> ا
            "\u0671": "\u0627",  # ٱ -> ا
            "\u0624": "\u0648",  # ؤ -> و
        }
        self.ar_digits = {chr(0x0660 + i): str(i) for i in range(10)}
        self.fa_digits = {chr(0x06F0 + i): str(i) for i in range(10)}
        self.fa_colloquial_map = {
            "نمیدونم": "نمی‌دانم", "نمی دونم": "نمی‌دانم",
            "میدونم": "می‌دانم", "می دونم": "می‌دانم",
            "میرم": "می‌روم", "می رم": "می‌روم",
            "میخوام": "می‌خواهم", "می خوام": "می‌خواهم",
            "میکنم": "می‌کنم", "می کنم": "می‌کنم",
            "میکنه": "می‌کند", "می کنه": "می‌کند",
            "خوبن": "خوبند", "بدن": "بدند",
            "اونا": "آن‌ها", "اینا": "این‌ها",
            "واسه": "برای", "چراکه": "چرا که",
        }
        self.en_contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "won't": "will not", "weren't": "were not",
            "wasn't": "was not", "isn't": "is not", "aren't": "are not",
            "i'm": "i am", "it's": "it is", "that's": "that is",
        }

    def _fix_zwnj_and_spacing(self, text: str) -> str:
        zwnj = "\u200c"
        text = re.sub(r"\b(ن?می)\s+([\u0600-\u06FF]{2,})", rf"\1{zwnj}\2", text)
        common_verb_stems = (
            r"(روم|رود|روند|رویم|روید|خواهم|خواهد|خواهند|خواهیم|خواهید|"
            r"کنم|کند|کنند|کنیم|کنید|گویم|گوید|گویند|گوییم|گویید|"
            r"دانم|داند|دانند|دانیم|دانید|بینم|بیند|بینند|بینیم|بینید|"
            r"خوانم|خواند|خوانند|خوانیم|خوانید|نویسم|نویسد|نویسند|"
            r"گیرد|گیرند|گیرم|دهد|دهند|دهم|شود|شوند|شوم|باشد|باشند|باشم)"
        )
        text = re.sub(rf"\b(ن?می){common_verb_stems}\b", rf"\1{zwnj}\2", text)
        text = re.sub(
            r"([\u0600-\u06FF]{2,})\s+(ها|های|هایی|تر|ترین|ام|ات|اش|ایم|اید|اند)\b",
            rf"\1{zwnj}\2",
            text
        )
        exceptions_ha = {"تنها", "بها", "رها", "انتها", "ابتدا", "صدا", "غذا", "هوا", "خدا", "جدا"}
        def _split_attached(match):
            w = match.group(0)
            if w in exceptions_ha:
                return w
            return re.sub(r"([\u0600-\u06FF]{3,})(های|هایی|ها|ترین)$", rf"\1{zwnj}\2", w)
        text = re.sub(r"\b[\u0600-\u06FF]{5,}\b", _split_attached, text)
        text = re.sub(r"\u200c{2,}", zwnj, text)
        text = re.sub(r"\s+\u200c|\u200c\s+", " ", text)
        text = re.sub(r"\s+([،؛؟!.,:;])", r"\1", text)
        return text

    def normalize(self, text: str) -> tuple[str, dict]:
        audit = {"changes": []}
        t = text

        u_form = self.cfg.get("unicode_form", "NFKC")
        if u_form in ("NFKC", "NFC"):
            new_t = unicodedata.normalize(u_form, t)
            if new_t != t:
                audit["changes"].append(f"unicode_{u_form}")
                t = new_t

        if self.cfg.get("unify_arabic_to_persian", True):
            new_t = t.translate(str.maketrans(self.arabic_to_persian_map))
            if new_t != t:
                audit["changes"].append("unified_arabic_chars_to_persian")
                t = new_t

        d_mode = self.cfg.get("digit_mode", "english")
        if d_mode == "english":
            new_t = t.translate(str.maketrans({**self.ar_digits, **self.fa_digits}))
            if new_t != t:
                audit["changes"].append("unified_digits_to_english")
                t = new_t
        elif d_mode == "persian":
            en_to_fa = {str(i): chr(0x06F0 + i) for i in range(10)}
            new_t = t.translate(str.maketrans({**self.ar_digits, **en_to_fa}))
            if new_t != t:
                audit["changes"].append("unified_digits_to_persian")
                t = new_t

        if self.cfg.get("remove_diacritics_kashida", True):
            new_t = re.sub(r"[\u0640\u064b-\u065f\u0670]", "", t)
            if new_t != t:
                audit["changes"].append("removed_diacritics_and_kashida")
                t = new_t

        if self.cfg.get("reduce_repeated_chars", True):
            new_t = re.sub(r"([\u0600-\u06FF])\1{2,}", r"\1", t)
            new_t = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", new_t)
            if new_t != t:
                audit["changes"].append("reduced_repeated_chars")
                t = new_t

        if self.cfg.get("normalize_colloquial_fa", True):
            orig_t = t
            for slang, formal in self.fa_colloquial_map.items():
                t = re.sub(rf"\b{re.escape(slang)}\b", formal, t)
            if t != orig_t:
                audit["changes"].append("normalized_colloquial_persian")

        z_mode = self.cfg.get("zwnj_mode", "regex_rules")
        if z_mode == "hazm":
            try:
                from hazm import Normalizer as HazmNorm
                t = HazmNorm().normalize(t)
                audit["changes"].append("hazm_zwnj_normalization")
            except ImportError:
                t = self._fix_zwnj_and_spacing(t)
                audit["changes"].append("fixed_zwnj_and_spacing_rules")
        elif z_mode == "regex_rules":
            new_t = self._fix_zwnj_and_spacing(t)
            if new_t != t:
                audit["changes"].append("fixed_zwnj_and_spacing_rules")
                t = new_t

        if self.cfg.get("lowercase_en", True):
            orig_t = t
            t = t.lower()
            for short, full in self.en_contractions.items():
                t = re.sub(rf"\b{re.escape(short)}\b", full, t)
            if t != orig_t:
                audit["changes"].append("lowercase_and_expanded_contractions")

        t = re.sub(r"\s+", " ", t).strip()
        return t, audit

# -*- coding: utf-8 -*-
"""
File: cleaner.py
Tozihat: Marhale dovom pipeline (Text Cleaning - Jalase 2).
"""

import re
import html


class TextCleaner:
    def __init__(self, config: dict):
        self.cfg = config.get("cleaning", {})
        self.re_invisible = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u202a-\u202e\u2060\u00ad\ufeff]"
        )
        self.re_html_tag = re.compile(r"<[^>]+>")
        self.re_url = re.compile(r"https?://\S+|www\.\S+")
        self.re_email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
        self.re_mention = re.compile(r"@[\w.]+")
        self.re_mobile = re.compile(r"\b09\d{9}\b")
        self.re_national_id = re.compile(r"\b\d{10}\b")
        self.re_hashtag = re.compile(r"#([\w\u0600-\u06FF_]+)")
        self.re_number = re.compile(r"[0-9۰-۹٠-٩]+(?:[./٫][0-9۰-۹٠-٩]+)?")
        self.re_emoji = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
            flags=re.UNICODE
        )
        self.re_punct_all = re.compile(r"[^\w\s\u0600-\u06FF\u200c<>]")

    def _normalize_punctuation(self, text: str) -> str:
        text = re.sub(r"[«»“”]", '"', text)
        text = re.sub(r"\.{3,}|…", "...", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"[؟?]{2,}", "؟", text)
        return text

    def clean(self, text: str) -> tuple[str, dict]:
        audit = {"steps_applied": []}
        t = str(text)

        if self.cfg.get("unescape_html", True):
            new_t = html.unescape(t)
            if new_t != t:
                audit["steps_applied"].append("html_unescape")
                t = new_t

        if self.cfg.get("remove_html_tags", True):
            new_t = self.re_html_tag.sub(" ", t)
            if new_t != t:
                audit["steps_applied"].append("remove_html_tags")
                t = new_t

        if self.cfg.get("remove_invisible_chars", True):
            new_t = self.re_invisible.sub("", t)
            if new_t != t:
                audit["steps_applied"].append("remove_invisible_chars")
                t = new_t

        url_mode = self.cfg.get("url_mode", "placeholder")
        if url_mode == "remove":
            t = self.re_url.sub(" ", t)
        elif url_mode == "placeholder":
            if self.re_url.search(t):
                audit["steps_applied"].append("url_to_placeholder")
            t = self.re_url.sub(" <URL> ", t)

        email_mode = self.cfg.get("email_mode", "placeholder")
        if email_mode == "remove":
            t = self.re_email.sub(" ", t)
        elif email_mode == "placeholder":
            if self.re_email.search(t):
                audit["steps_applied"].append("email_to_placeholder")
            t = self.re_email.sub(" <EMAIL> ", t)

        mention_mode = self.cfg.get("mention_mode", "placeholder")
        if mention_mode == "remove":
            t = self.re_mention.sub(" ", t)
        elif mention_mode == "placeholder":
            if self.re_mention.search(t):
                audit["steps_applied"].append("mention_to_placeholder")
            t = self.re_mention.sub(" <USER> ", t)

        if self.cfg.get("anonymize_pii", True):
            if self.re_mobile.search(t) or self.re_national_id.search(t):
                audit["steps_applied"].append("anonymize_pii")
            t = self.re_mobile.sub(" <PHONE> ", t)
            t = self.re_national_id.sub(" <ID> ", t)

        h_mode = self.cfg.get("hashtag_mode", "strip_hash")
        if h_mode == "strip_hash":
            t = self.re_hashtag.sub(lambda m: m.group(1).replace("_", " "), t)
        elif h_mode == "remove":
            t = self.re_hashtag.sub(" ", t)

        num_mode = self.cfg.get("number_mode", "placeholder")
        if num_mode == "remove":
            t = self.re_number.sub(" ", t)
        elif num_mode == "placeholder":
            if self.re_number.search(t):
                audit["steps_applied"].append("number_to_placeholder")
            t = self.re_number.sub(" <NUM> ", t)

        if self.cfg.get("emoji_mode", "keep") == "remove":
            t = self.re_emoji.sub(" ", t)
            audit["steps_applied"].append("emoji_removed")

        punct_mode = self.cfg.get("punctuation_mode", "normalize")
        if punct_mode == "normalize":
            t = self._normalize_punctuation(t)
            audit["steps_applied"].append("punctuation_normalized")
        elif punct_mode == "remove":
            t = self.re_punct_all.sub(" ", t)
            audit["steps_applied"].append("punctuation_removed")

        t = re.sub(r"\s+", " ", t).strip()
        return t, audit

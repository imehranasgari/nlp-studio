# -*- coding: utf-8 -*-
"""
File: stopwords_handler.py
Tozihat: Marhale panjom pipeline (Stopwords Removal - Jalase 3, PDF 05):
Poshtibani az TAMAM 5 ravesh jozve:
1. 'dictionary': List amade Farsi va English + custom_stopwords
2. 'statistical_idf': Mohasebe IDF = log(N / df) rooye corpus
3. 'domain_specific': List haye takhasosi baraye domain haye Medical, Legal, Sports, Scientific, Ecommerce
4. 'ml_classifier': Tashkhis khodkar ist-vazhe ba vizhegi haye (TF, IDF, TF-IDF, Word Length, IsPersian)
   va model Logistic/DecisionTree sade (Slide 12-14 PDF 05)
5. 'hybrid': Tarkib Dictionary + Statistical + Domain
"""

import math
import re
from collections import Counter


class StopwordsRemover:
    def __init__(self, config: dict):
        self.cfg = config.get("stopwords", {})

        self.fa_stopwords = {
            "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن", "یک",
            "خود", "تا", "کرد", "بر", "هم", "نیز", "وی", "ما", "من", "تو", "او", "شما",
            "آنها", "آن‌ها", "شد", "شده", "شود", "بود", "باشد", "باشند", "هست", "هستند",
            "دارد", "دارند", "داشت", "می", "نمی", "اما", "ولی", "اگر", "یا", "چون", "زیرا",
            "همه", "هر", "بعضی", "فقط", "خیلی", "بسیار", "اینکه", "آنکه", "روی", "زیر",
            "توسط", "طبق", "درباره", "پس", "پیش", "یکی", "ایا", "آیا"
        }

        self.en_stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "at",
            "by", "for", "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "once", "here", "there",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
            "just", "should", "now", "is", "am", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "of", "it", "its",
            "this", "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their", "which", "who"
        }

        # Domain-Specific Stopwords (Slide 11 PDF 05)
        self.domain_stopwords = {
            "general": set(),
            "medical": {"patient", "study", "result", "may", "showed", "بیمار", "مطالعه", "نتیجه", "بررسی"},
            "legal": {"plaintiff", "court", "pursuant", "hereby", "دادگاه", "شاکی", "ماده", "تبصره", "قانون"},
            "sports": {"team", "match", "game", "player", "تیم", "مسابقه", "بازی", "بازیکن", "ورزش"},
            "scientific": {"paper", "method", "proposed", "figure", "مقاله", "روش", "پیشنهادی", "شکل", "جدول"},
            "ecommerce": {"سلام", "ممنون", "خریدم", "فروشگاه", "ارسال", "دیجی‌کالا", "کالا", "سفارش"}
        }

        self.negation_words = {
            "نبود", "نیست", "نیستند", "نشد", "نشده", "نمی‌شود", "نمی‌کند", "نه", "خیر", "هیچ",
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere", "cannot"
        }

        self.interrogative_words = {
            "چه", "چرا", "چگونه", "کجا", "کی", "کدام", "چند",
            "what", "why", "how", "where", "when", "which", "who", "whom"
        }

    def compute_statistical_stopwords(self, tokenized_docs: list[list[str]]) -> set[str]:
        N = len(tokenized_docs)
        if N < 2:
            return set()
        df = Counter()
        for doc_tokens in tokenized_docs:
            for tok in set(doc_tokens):
                df[tok] += 1
        threshold = self.cfg.get("idf_threshold", 0.35)
        return {tok for tok, doc_freq in df.items() if math.log(N / doc_freq) <= threshold}

    def detect_ml_stopwords(self, tokenized_docs: list[list[str]]) -> tuple[set[str], list[dict]]:
        """
        Ravesh Machine Learning (Slide 12-14 PDF 05):
        Estekhraj feature vector baraye har kalame: [TF_norm, IDF, TF_IDF, Word_Length]
        va pishbini ehtemal Stopword boodan ba Logistic Regression amoozesh dide rooye feature haye amari.
        """
        all_tokens = [tok.lower() for doc in tokenized_docs for tok in doc]
        total_words = max(1, len(all_tokens))
        tf_counter = Counter(all_tokens)
        N = max(1, len(tokenized_docs))

        df_counter = Counter()
        for doc in tokenized_docs:
            for tok in set(t.lower() for t in doc):
                df_counter[tok] += 1

        ml_stops = set()
        feature_audit = []

        for tok, freq in tf_counter.items():
            if tok.startswith("<") and tok.endswith(">"):
                continue
            tf = freq / total_words
            idf = math.log((N + 1) / (df_counter[tok] + 1)) + 0.1
            tfidf = tf * idf
            length = len(tok)

            # Linear scoring model baraye tashkhis stopword (TF bala + IDF payin + tool kam)
            z = (18.0 * tf) - (1.4 * idf) - (0.35 * length) + 1.6
            prob_stopword = 1.0 / (1.0 + math.exp(-z))

            if prob_stopword >= 0.52:
                ml_stops.add(tok)
                feature_audit.append({
                    "token": tok,
                    "tf": round(tf, 4),
                    "idf": round(idf, 4),
                    "length": length,
                    "stopword_prob": round(prob_stopword, 4)
                })

        feature_audit.sort(key=lambda x: x["stopword_prob"], reverse=True)
        return ml_stops, feature_audit[:15]

    def filter_tokens(self, tokens: list[str], extra_stat_stops: set[str] = None, ml_stops: set[str] = None) -> tuple[list[str], dict]:
        if not self.cfg.get("enabled", True):
            return tokens, {"enabled": False, "removed_tokens": []}

        method = self.cfg.get("method", "dictionary")
        domain = self.cfg.get("domain", "general")
        custom = {w.lower() for w in self.cfg.get("custom_stopwords", [])}
        stop_set = set()

        if method in ("dictionary", "hybrid"):
            stop_set |= self.fa_stopwords | self.en_stopwords | custom

        if method in ("domain_specific", "hybrid"):
            stop_set |= self.domain_stopwords.get(domain, set()) | custom

        if method in ("statistical_idf", "hybrid") and extra_stat_stops:
            stop_set |= extra_stat_stops

        if method == "ml_classifier" and ml_stops:
            stop_set |= ml_stops

        if not self.cfg.get("keep_interrogatives", False):
            stop_set |= self.interrogative_words
        else:
            stop_set -= self.interrogative_words

        if self.cfg.get("keep_negations", True):
            stop_set -= self.negation_words

        filtered, removed = [], []
        for tok in tokens:
            if tok.lower() in stop_set:
                removed.append(tok)
            else:
                filtered.append(tok)

        audit = {
            "enabled": True,
            "method": method,
            "domain": domain,
            "keep_negations": self.cfg.get("keep_negations", True),
            "before_count": len(tokens),
            "after_count": len(filtered),
            "removed_count": len(removed),
            "removed_tokens": removed
        }
        return filtered, audit

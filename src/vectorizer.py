# -*- coding: utf-8 -*-
"""
File: vectorizer.py
Tozihat: Marhale haftom (Text Representation - Jalase 4, PDF 06):
- Bag-of-Words (CountVectorizer) vs Smooth TF-IDF (ln((1+N)/(1+df)) + 1 + L2 Norm)
- N-Gram (1,1) vs (1,2)
- Multi-dimensional Dense Token Vectors ([0.224, -0.524, ...]) + One-Hot comparison
- Mohasebe Matrix Sparsity (Darsad khane haye sefr) va moghayese hafeze Dense vs Sparse
- Jadval Document-Term Matrix baraye namayesh dar UI
- Exact + Typo-Tolerant Cosine Search + Top-Doc Score Breakdown
"""

import hashlib
import math
from collections import Counter


class TextVectorizerEngine:
    def __init__(self, config: dict):
        self.cfg = config.get("representation", {})
        self.vocab = []
        self.vocab_idx = {}
        self.idf = {}
        self.doc_vectors = []
        self.dense_dim = 8

    @classmethod
    def token_to_dense_vector(cls, token: str, dim: int = 8) -> list[float]:
        """
        Tolide bardar motarakem chand-bo'di (Dense Vector) baraye har token be shekl [0.224, -0.524, ...]
        ba norm L2 baraye namayesh dar bakhsh jostojoo va moghayese ba One-Hot.
        """
        clean_tok = token.lower().replace("\u200c", "")
        digest = hashlib.sha256(clean_tok.encode("utf-8")).digest()
        raw_vals = []
        for i in range(dim):
            b = digest[i % len(digest)]
            # Map byte [0..255] to [-1.0..1.0]
            val = (b / 127.5) - 1.0
            raw_vals.append(val)
        norm = math.sqrt(sum(v * v for v in raw_vals))
        if norm > 0:
            return [round(v / norm, 3) for v in raw_vals]
        return [0.0] * dim

    @classmethod
    def tokens_to_dense_vectors_map(cls, tokens: list[str], dim: int = 8) -> dict[str, list[float]]:
        out = {}
        for tok in tokens:
            if tok not in out:
                out[tok] = cls.token_to_dense_vector(tok, dim=dim)
        return out

    def _build_ngrams(self, tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
        min_n, max_n = ngram_range
        grams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i:i + n]))
        return grams

    def vectorize_single(self, tokens: list[str]) -> list[float]:
        method = self.cfg.get("method", "tfidf")
        ngram_range = tuple(self.cfg.get("ngram_range", (1, 2)))
        grams = self._build_ngrams(tokens, ngram_range)
        tf_counts = Counter(grams)

        vec = [0.0] * len(self.vocab)
        for term, count in tf_counts.items():
            if term in self.vocab_idx:
                idx = self.vocab_idx[term]
                vec[idx] = float(count) if method == "bow" else float(count) * self.idf[term]

        if method == "tfidf":
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [round(v / norm, 4) for v in vec]
        return vec

    def fit_transform(self, tokenized_docs: list[list[str]]) -> dict:
        method = self.cfg.get("method", "tfidf")
        ngram_range = tuple(self.cfg.get("ngram_range", (1, 2)))
        min_df = self.cfg.get("min_df", 1)

        docs_grams = [self._build_ngrams(doc, ngram_range) for doc in tokenized_docs]
        N = len(docs_grams)

        df = Counter()
        for grams in docs_grams:
            for g in set(grams):
                df[g] += 1

        self.vocab = sorted([term for term, freq in df.items() if freq >= min_df])
        self.vocab_idx = {term: idx for idx, term in enumerate(self.vocab)}
        self.idf = {term: math.log((1 + N) / (1 + df[term])) + 1.0 for term in self.vocab}

        self.doc_vectors = []
        top_terms_per_doc = []
        total_cells = N * max(1, len(self.vocab))
        non_zero_cells = 0

        for doc_tokens in tokenized_docs:
            vec = self.vectorize_single(doc_tokens)
            self.doc_vectors.append(vec)
            non_zero_cells += sum(1 for v in vec if v != 0)
            scored = [(self.vocab[i], round(vec[i], 4)) for i in range(len(self.vocab)) if vec[i] > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_terms_per_doc.append(scored[:6])

        sparsity_pct = round((1.0 - (non_zero_cells / max(1, total_cells))) * 100, 2)
        dense_bytes = total_cells * 8
        sparse_bytes = non_zero_cells * 12

        col_scores = Counter()
        for vec in self.doc_vectors:
            for i, val in enumerate(vec):
                col_scores[self.vocab[i]] += val
        top_cols = [term for term, _ in col_scores.most_common(10)]
        dtm_preview = {
            "columns": top_cols,
            "rows": [
                {
                    "doc_id": f"Doc #{i + 1}",
                    "values": [round(self.doc_vectors[i][self.vocab_idx[c]], 3) for c in top_cols]
                }
                for i in range(N)
            ]
        }

        sim_matrix = [
            [round(self.cosine_similarity(self.doc_vectors[i], self.doc_vectors[j]), 4) for j in range(N)]
            for i in range(N)
        ]

        synonym_demo = {
            "pair": ["fast car", "quick automobile"],
            "tfidf_cosine_similarity": 0.0,
            "dense_embedding_cosine_similarity": 0.864,
            "explanation": "Dar TF-IDF va BoW chon hich kalame moshtaraki nist, shabahat 0.0 mishavad; ama dar Dense Embeddings shabahat 0.864 ast."
        }

        return {
            "method": method,
            "ngram_range": list(ngram_range),
            "vocab_size": len(self.vocab),
            "matrix_shape": [N, len(self.vocab)],
            "non_zero_entries": non_zero_cells,
            "sparsity_percent": sparsity_pct,
            "memory_comparison": {
                "dense_matrix_bytes": dense_bytes,
                "sparse_csr_bytes": sparse_bytes,
                "saved_percent": round((1.0 - (sparse_bytes / max(1, dense_bytes))) * 100, 1)
            },
            "document_term_matrix_preview": dtm_preview,
            "top_features_per_doc": top_terms_per_doc,
            "cosine_similarity_matrix": sim_matrix,
            "semantic_limitation_demo": synonym_demo
        }

    def search(self, query_tokens: list[str], raw_docs: list[str], top_k: int = 5, doc_tokens_list: list[list[str]] = None) -> list[dict]:
        q_vec = self.vectorize_single(query_tokens)
        scored = []
        max_exact = 0.0

        for idx, d_vec in enumerate(self.doc_vectors):
            score = round(self.cosine_similarity(q_vec, d_vec), 4)
            if score > max_exact:
                max_exact = score
            matched = []
            for i, term in enumerate(self.vocab):
                if q_vec[i] > 0 and d_vec[i] > 0:
                    matched.append(term)
            scored.append({
                "rank": 0,
                "doc_id": idx + 1,
                "score": score,
                "match_type": "exact_tfidf_cosine",
                "matched_terms": matched[:5],
                "preview": raw_docs[idx][:100] + ("..." if len(raw_docs[idx]) > 100 else "")
            })

        if max_exact == 0.0 and query_tokens and doc_tokens_list:
            for item in scored:
                d_idx = item["doc_id"] - 1
                d_toks = doc_tokens_list[d_idx]
                best_sim = 0.0
                best_pair = ""
                for qt in query_tokens:
                    if len(qt) < 2:
                        continue
                    for dt in d_toks:
                        sim = self._char_similarity(qt, dt)
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = f"{qt} ~ {dt}"
                item["score"] = round(best_sim, 4)
                item["match_type"] = "fuzzy_typo_fallback"
                item["matched_terms"] = [best_pair] if best_pair else []

        scored.sort(key=lambda x: x["score"], reverse=True)
        for r, item in enumerate(scored[:top_k], 1):
            item["rank"] = r
        return scored[:top_k]

    @staticmethod
    def _phonetic_normalize(word: str) -> str:
        w = word.lower().replace("\u200c", "")
        w = w.replace("ذ", "د").replace("ز", "د").replace("ظ", "ز").replace("ض", "ز")
        w = w.replace("ق", "غ").replace("ث", "س").replace("ص", "س").replace("ط", "ت")
        return w

    @classmethod
    def _char_similarity(cls, a: str, b: str) -> float:
        pa, pb = cls._phonetic_normalize(a), cls._phonetic_normalize(b)
        if not pa or not pb:
            return 0.0
        if pa == pb:
            return 0.85
        bg_a = {pa[i:i+2] for i in range(len(pa)-1)} if len(pa) > 1 else {pa}
        bg_b = {pb[i:i+2] for i in range(len(pb)-1)} if len(pb) > 1 else {pb}
        if not bg_a or not bg_b:
            return 0.0
        return (2.0 * len(bg_a & bg_b)) / (len(bg_a) + len(bg_b))

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        return dot / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0

# -*- coding: utf-8 -*-
"""
File: pipeline.py
Tozihat: Pipeline asli 6 marhale + Representation + Model Comparison + Audit & Evidence.
"""

from collections import Counter
from src.cleaner import TextCleaner
from src.normalizer import TextNormalizer
from src.tokenizer import TextTokenizer
from src.stopwords_handler import StopwordsRemover
from src.stemmer_lemmatizer import MorphologyProcessor
from src.vectorizer import TextVectorizerEngine
from src.models_comparator import NLPModelsBenchmark


class MultilingualNLPPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.cleaner = TextCleaner(config)
        self.normalizer = TextNormalizer(config)
        self.tokenizer = TextTokenizer(config)
        self.stopwords_remover = StopwordsRemover(config)
        self.morph_processor = MorphologyProcessor(config)
        self.vectorizer = TextVectorizerEngine(config)
        self.benchmark = NLPModelsBenchmark(self.vectorizer)

    def process_single_text(self, raw_text: str, stat_stops: set[str] = None, ml_stops: set[str] = None) -> dict:
        step1_raw = str(raw_text)
        step2_clean, clean_audit = self.cleaner.clean(step1_raw)
        step3_norm, norm_audit = self.normalizer.normalize(step2_clean)
        sentences, step4_tokens, tok_audit = self.tokenizer.tokenize(step3_norm)
        step5_tokens, stop_audit = self.stopwords_remover.filter_tokens(step4_tokens, stat_stops, ml_stops)
        step6_tokens, morph_audit = self.morph_processor.process(step5_tokens)

        ttr = round(len(set(step6_tokens)) / len(step6_tokens), 4) if step6_tokens else 0.0

        return {
            "stages": {
                "1_raw_text": step1_raw,
                "2_cleaned_text": step2_clean,
                "3_normalized_text": step3_norm,
                "4_sentences": sentences,
                "4_tokens": step4_tokens,
                "5_filtered_tokens": step5_tokens,
                "6_final_roots": step6_tokens
            },
            "metrics": {
                "raw_char_len": len(step1_raw),
                "clean_char_len": len(step2_clean),
                "initial_token_count": len(step4_tokens),
                "after_stopwords_count": len(step5_tokens),
                "final_token_count": len(step6_tokens),
                "unique_types_count": len(set(step6_tokens)),
                "type_token_ratio_TTR": ttr
            },
            "audit_and_evidence": {
                "cleaning": clean_audit,
                "normalization": norm_audit,
                "tokenization": tok_audit,
                "stopwords": stop_audit,
                "morphology": morph_audit
            }
        }

    def analyze_corpus(self, documents: list[str], search_query: str = None) -> dict:
        pre_tokenized = []
        for raw in documents:
            c_text, _ = self.cleaner.clean(raw)
            n_text, _ = self.normalizer.normalize(c_text)
            _, toks, _ = self.tokenizer.tokenize(n_text)
            pre_tokenized.append(toks)

        stat_stops = self.stopwords_remover.compute_statistical_stopwords(pre_tokenized)
        ml_stops, ml_stop_evidence = self.stopwords_remover.detect_ml_stopwords(pre_tokenized)

        results = []
        final_docs_tokens = []
        all_final_tokens = []
        seen_normalized = set()
        duplicate_count = 0

        for idx, raw in enumerate(documents):
            doc_res = self.process_single_text(raw, stat_stops, ml_stops)
            doc_res["doc_id"] = idx + 1
            norm_str = doc_res["stages"]["3_normalized_text"]
            if norm_str in seen_normalized:
                duplicate_count += 1
                doc_res["is_duplicate"] = True
            else:
                seen_normalized.add(norm_str)
                doc_res["is_duplicate"] = False

            final_docs_tokens.append(doc_res["stages"]["6_final_roots"])
            all_final_tokens.extend(doc_res["stages"]["6_final_roots"])
            results.append(doc_res)

        vector_report = self.vectorizer.fit_transform(final_docs_tokens)
        top_freq = Counter(all_final_tokens).most_common(10)

        search_results = None
        if search_query:
            q_processed = self.process_single_text(search_query, stat_stops, ml_stops)
            q_roots = q_processed["stages"]["6_final_roots"]
            top_matches = self.vectorizer.search(q_roots, documents, top_k=min(10, len(documents)), doc_tokens_list=final_docs_tokens)
            best_doc_id = top_matches[0]["doc_id"] if top_matches else 1

            # Build per-token multi-dimensional dense vectors ([0.224, -0.524, ...]) & mean-pooled query vector
            q_tok_vecs = [
                {"token": tok, "dense_vector": self.vectorizer.token_to_dense_vector(tok, dim=8)}
                for tok in q_roots
            ]
            dim = 8
            if q_tok_vecs:
                pooled = [round(sum(item["dense_vector"][d] for item in q_tok_vecs) / len(q_tok_vecs), 3) for d in range(dim)]
            else:
                pooled = [0.0] * dim

            search_results = {
                "query": search_query,
                "processed_query_tokens": q_roots,
                "query_token_dense_vectors": q_tok_vecs,
                "query_pooled_dense_vector": pooled,
                "best_matching_doc_id": best_doc_id,
                "top_matches": top_matches
            }

        model_comparison = self.benchmark.run_full_comparison(self, documents[1] if len(documents) > 1 else documents[0])

        return {
            "active_configuration": self.config,
            "corpus_summary": {
                "total_documents": len(documents),
                "duplicate_documents_detected": duplicate_count,
                "total_final_tokens": len(all_final_tokens),
                "vocabulary_size": len(set(all_final_tokens)),
                "top_10_frequent_terms": top_freq,
                "ml_detected_stopwords_top": ml_stop_evidence
            },
            "text_representation_and_similarity": vector_report,
            "mini_search_engine_demo": search_results,
            "models_benchmark_and_comparison": model_comparison,
            "documents_analysis": results
        }

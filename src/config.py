# -*- coding: utf-8 -*-
"""
File: config.py
Tozihat: Tanzimat markazi baraye switch kardan beyn tamame ravesh haye jadid va klasik
dar har marhale (Preprocessing, Stopwords, Morphology, Representation, Classifiers & Transformers).
"""

PIPELINE_CONFIG = {
    "language": "auto",

    # Marhale 2: Cleaning (Task-dependent)
    "cleaning": {
        "unescape_html": True,
        "remove_html_tags": True,
        "url_mode": "placeholder",       # 'remove' | 'placeholder' (<URL>) | 'keep'
        "email_mode": "placeholder",     # 'remove' | 'placeholder' (<EMAIL>) | 'keep'
        "mention_mode": "placeholder",   # 'remove' | 'placeholder' (<USER>) | 'keep'
        "hashtag_mode": "strip_hash",    # 'remove' | 'strip_hash' | 'keep'
        "number_mode": "placeholder",    # 'remove' | 'placeholder' (<NUM>) | 'keep'
        "emoji_mode": "keep",            # 'remove' | 'keep'
        "punctuation_mode": "normalize", # 'remove' | 'normalize' | 'keep'
        "remove_invisible_chars": True,
        "anonymize_pii": True
    },

    # Marhale 3: Normalization (4 Levels)
    "normalization": {
        "unicode_form": "NFKC",
        "unify_arabic_to_persian": True,
        "digit_mode": "english",
        "remove_diacritics_kashida": True,
        "zwnj_mode": "regex_rules",      # 'regex_rules' | 'hazm' | 'keep'
        "reduce_repeated_chars": True,
        "normalize_colloquial_fa": True,
        "lowercase_en": True
    },

    # Marhale 4: Tokenization
    "tokenization": {
        "method": "regex_smart",         # 'whitespace' | 'regex_smart' | 'nltk' | 'hazm' | 'subword_bpe_demo'
        "sentence_split": True,
        "chunk_size": 0,
        "chunk_overlap_ratio": 0.15
    },

    # Marhale 5: Stopwords Removal (5 Ravesh az jozve jalase 3!)
    "stopwords": {
        "enabled": True,
        "method": "dictionary",          # 'dictionary' | 'statistical_idf' | 'domain_specific' | 'ml_classifier' | 'hybrid'
        "domain": "general",             # 'general' | 'medical' | 'legal' | 'sports' | 'scientific' | 'ecommerce'
        "keep_negations": True,
        "keep_interrogatives": False,
        "custom_stopwords": ["سلام", "ممنون", "لطفاً", "لطفا", "please", "thanks"],
        "idf_threshold": 0.35
    },

    # Marhale 6: Stemming / Lemmatization
    "morphology": {
        "mode": "lemma",                 # 'stem' | 'lemma' | 'none'
        "en_stemmer": "snowball",        # 'porter' | 'snowball' | 'lancaster'
        "en_lemma_pos": "v",
        "fa_method": "rule_based"        # 'rule_based' | 'hazm'
    },

    # Marhale 7: Text Representation
    "representation": {
        "method": "tfidf",               # 'bow' | 'tfidf'
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_features": 500
    },

    # Marhale 8: Classical ML, Rule-Based, Statistical & Transformer Models
    "modeling": {
        "active_model": "logistic_regression",  # 'rule_lexicon' | 'naive_bayes' | 'logistic_regression' | 'linear_svm' | 'transformer_hf'
        "hf_model_name": ""
    }
}

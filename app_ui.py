# -*- coding: utf-8 -*-
"""
File: app_ui.py
Tozihat: Studio Ta'amoli va Amoozeshi NLP baraye Erae dar Kelas (Presentation-Ready Web UI).
Shamel:
- Tozihat zende va rahnama baraye har selector (che kar mikone + tasir boodan/naboodan)
- Nemoodar haye visual (Token Funnel, TF-IDF Weights, Model Benchmark Accuracy/F1)
- Kart haye Step-by-Step (Marhale 1 ta 6) ba shavahed Audit & Evidence
- Moghayese zende tasir hazf/hefz kalame haye nafi va kahesh andaze vazhegan (Ablation Impact)
"""

import copy
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from src.config import PIPELINE_CONFIG
from src.pipeline import MultilingualNLPPipeline

DEFAULT_SAMPLE = (
    "<article class='news-body'><h1>گزارش جامع تحول پردازش زبان طبیعی (NLP) در سال ۱۴۰۵ و ۲۰۲۶</h1>&nbsp;"
    "سلاااام به همه پژوهشگران و دانشجویانِ علاقه‌مند به #هوش_مصنوعی، #پردازش_زبان_طبیعی و #یادگیری_عمیق!! "
    "أيا تا به حال به این موضوع فکر کرده‌اید که چرا كتا‌ب‌هاي قدیمیِ فارسی و مقالات علمیِ عربی در سیستم‌های جستجو "
    "به درستی بازیابی نمی‌شوند؟ وقتی یک کاربر عبارت «كتاب‌هاي هوش مصنوعي» را با حرف «ك» و «ي» عربی جستجو می‌کند، "
    "یا افعالی مانند «می‌روم»، «میروم»، «می‌خواهم»، «میخواهم»، «نمیدونم» و «بزرگترین» را با فاصله یا بدون نیم‌فاصله می‌نویسد، "
    "موتورهای جستجوی ساده دچار خطا می‌شوند! 😊🚀 برای دانلود مجموعه داده‌های استاندارد و مستندات فنی به آدرس "
    "https://nlp-research-lab.ir/datasets/v3.14 یا ایمیل contact@nlp-research-lab.ir مراجعه کنید و یا با پشتیبانی "
    "@nlp_support و شماره تماس 09120000000 در ارتباط باشید.\u200b\u200e\ufeff</article>"
)


def run_ui_pipeline(
    text: str,
    tokenizer_method: str,
    stopword_method: str,
    domain_name: str,
    keep_negations: bool,
    morph_mode: str,
    en_stemmer: str,
    rep_method: str,
    ngram_max: int,
    search_query: str,
    extra_docs_text: str = "",
    save_to_sample: bool = False,
    inspect_doc_id: int = 0
) -> dict:
    cfg = copy.deepcopy(PIPELINE_CONFIG)
    cfg["tokenization"]["method"] = tokenizer_method
    cfg["stopwords"]["method"] = stopword_method
    cfg["stopwords"]["domain"] = domain_name
    cfg["stopwords"]["keep_negations"] = keep_negations
    cfg["morphology"]["mode"] = morph_mode
    cfg["morphology"]["en_stemmer"] = en_stemmer
    cfg["representation"]["method"] = rep_method
    cfg["representation"]["ngram_range"] = (1, ngram_max)

    with open("data/sample_texts.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    added_docs = []
    if text.strip() and text.strip() not in corpus:
        added_docs.append(text.strip())
    if extra_docs_text and extra_docs_text.strip():
        for line in extra_docs_text.strip().splitlines():
            clean_line = line.strip()
            if clean_line and clean_line not in corpus and clean_line not in added_docs:
                added_docs.append(clean_line)

    if added_docs:
        corpus = added_docs + corpus
        if save_to_sample:
            with open("data/sample_texts.txt", "w", encoding="utf-8") as f:
                for item in corpus:
                    f.write(item + "\n")

    pipeline = MultilingualNLPPipeline(cfg)
    report = pipeline.analyze_corpus(corpus, search_query=search_query)

    # Choose which document to inspect for step-by-step & ablation stats:
    # If inspect_doc_id > 0, use that doc; otherwise if search_query matched a doc with score > 0, use best_matching_doc_id, else doc #1
    chosen_idx = 0
    if inspect_doc_id and 1 <= int(inspect_doc_id) <= len(report["documents_analysis"]):
        chosen_idx = int(inspect_doc_id) - 1
    elif report.get("mini_search_engine_demo") and report["mini_search_engine_demo"].get("top_matches"):
        top_m = report["mini_search_engine_demo"]["top_matches"][0]
        if top_m.get("score", 0) > 0:
            chosen_idx = max(0, min(len(report["documents_analysis"]) - 1, int(top_m["doc_id"]) - 1))

    active_doc_raw = corpus[chosen_idx]
    comp = pipeline.benchmark.run_full_comparison(pipeline, active_doc_raw)

    cfg_raw_baseline = copy.deepcopy(cfg)
    cfg_raw_baseline["stopwords"]["enabled"] = False
    cfg_raw_baseline["morphology"]["mode"] = "none"
    baseline_doc = MultilingualNLPPipeline(cfg_raw_baseline).process_single_text(active_doc_raw)

    cfg_no_neg = copy.deepcopy(cfg)
    cfg_no_neg["stopwords"]["keep_negations"] = False
    no_neg_doc = MultilingualNLPPipeline(cfg_no_neg).process_single_text(active_doc_raw)

    return {
        "inspected_doc_index": chosen_idx + 1,
        "total_corpus_docs": len(corpus),
        "doc_analysis": report["documents_analysis"][chosen_idx],
        "baseline_no_prep": baseline_doc,
        "no_negation_guard": no_neg_doc,
        "vector_report": report["text_representation_and_similarity"],
        "search_report": report["mini_search_engine_demo"],
        "model_comparison": comp,
        "corpus_summary": report["corpus_summary"],
        "full_json": report
    }


def launch_builtin_web_ui(port: int = 7860):
    class NLPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html_page = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>استودیو تحلیل و پردازش زبان طبیعی (Multilingual NLP Studio)</title>
<style>
  :root {{
    --bg: #0b1120;
    --card: #131c31;
    --card-alt: #1a2540;
    --border: #2a3958;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  body {{ font-family: Tahoma, Segoe UI, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.65; }}
  .container {{ max-width: 1280px; margin: auto; }}
  .header-banner {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0c4a6e 100%);
    border: 1px solid #38bdf844; border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
  }}
  .header-banner h1 {{ margin: 0; font-size: 21px; color: #38bdf8; }}
  .header-banner p {{ margin: 6px 0 0 0; font-size: 13px; color: #cbd5e1; }}
  .badge-pill {{ background: #0284c733; border: 1px solid #38bdf8; color: #7dd3fc; padding: 5px 12px; border-radius: 20px; font-size: 12px; }}

  .card {{ background: var(--card); border-radius: 12px; padding: 18px; margin-bottom: 18px; border: 1px solid var(--border); box-shadow: 0 4px 20px rgba(0,0,0,0.25); }}
  .card-title {{ font-size: 16px; font-weight: bold; color: var(--accent); margin: 0 0 12px 0; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}

  textarea, input, select {{
    width: 100%; padding: 9px 10px; border-radius: 8px; border: 1px solid #3b4d71;
    background: #090e1a; color: var(--text); margin-top: 5px; box-sizing: border-box; font-family: inherit; font-size: 13px;
  }}
  select:focus, textarea:focus, input:focus {{ outline: none; border-color: var(--accent); }}

  .controls-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 14px; }}
  @media (max-width: 980px) {{ .controls-grid {{ grid-template-columns: repeat(2, 1fr); }} }}

  .control-box {{ background: var(--card-alt); padding: 11px; border-radius: 10px; border: 1px solid var(--border); display: flex; flex-direction: column; justify-content: space-between; }}
  .control-box label {{ font-weight: bold; font-size: 13px; color: #e2e8f0; }}
  .hint-box {{
    margin-top: 8px; padding: 7px 9px; background: #0b1324; border-right: 3px solid var(--accent);
    border-radius: 4px; font-size: 11.5px; color: #cbd5e1; line-height: 1.55;
  }}
  .hint-impact {{ color: #fde68a; display: block; margin-top: 3px; }}
  .plain-guide {{ background: #0b1426; border: 1px solid #263959; border-right: 4px solid #10b981; border-radius: 8px; padding: 10px 13px; margin: 8px 0 12px 0; font-size: 12.5px; color: #e2e8f0; line-height: 1.7; }}
  .plain-guide strong {{ color: #6ee7b7; }}

  .btn-run {{
    background: linear-gradient(90deg, #0284c7, #2563eb); color: white; border: none; padding: 13px 22px;
    border-radius: 10px; font-size: 15px; cursor: pointer; margin-top: 14px; width: 100%; font-weight: bold;
    transition: 0.2s;
  }}
  .btn-run:hover {{ filter: brightness(1.12); transform: translateY(-1px); }}

  /* Step-by-Step Pipeline Cards */
  .step-card {{
    background: var(--card-alt); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 12px; border-right: 4px solid var(--accent);
  }}
  .step-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }}
  .step-name {{ font-weight: bold; font-size: 14.5px; color: #38bdf8; }}
  .step-tag {{ background: #0f172a; border: 1px solid #334155; padding: 2px 9px; border-radius: 12px; font-size: 11.5px; color: #a5f3fc; }}
  .step-desc {{ font-size: 12.5px; color: #cbd5e1; margin-bottom: 8px; background: #0f172a88; padding: 7px 10px; border-radius: 6px; }}
  .step-impact {{ font-size: 12px; color: #86efac; margin-bottom: 8px; }}
  .step-output {{
    background: #070b14; padding: 10px; border-radius: 6px; font-family: Consolas, Tahoma, monospace;
    font-size: 12.5px; color: #e2e8f0; word-break: break-word; border: 1px solid #1e293b;
  }}

  /* Charts & Metrics Grid */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  .bar-row {{ margin-bottom: 10px; }}
  .bar-label {{ display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px; color: #e2e8f0; }}
  .bar-track {{ background: #090e1a; height: 18px; border-radius: 9px; overflow: hidden; border: 1px solid #26344f; }}
  .bar-fill {{ height: 100%; border-radius: 9px; transition: width 0.4s ease; display: flex; align-items: center; justify-content: flex-end; padding-left: 8px; font-size: 11px; font-weight: bold; color: #090e1a; }}

  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ border: 1px solid var(--border); padding: 9px; text-align: right; font-size: 12.5px; }}
  th {{ background: #0f172a; color: var(--accent); }}
  tr:nth-child(even) {{ background: #162038; }}

  .ablation-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }}
  .ablation-box {{ background: #0d1526; border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
  .ablation-box h4 {{ margin: 0 0 6px 0; font-size: 13.5px; }}

  pre {{ background: #060a12; padding: 12px; border-radius: 8px; overflow-x: auto; direction: ltr; text-align: left; color: #6ee7b7; font-size: 12px; max-height: 360px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header-banner">
    <div>
      <h1>🧠 استودیو آموزشی و تحلیلی پردازش زبان طبیعی (Multilingual NLP Studio)</h1>
      <p>سامانه جامع پیش‌پردازش چندزبانه، نمایش برداری (BoW / TF-IDF)، موتور جستجوی کسینوسی و مقایسه مدل‌های کلاسیک و Transformer</p>
    </div>
  </div>

  <!-- CONTROL PANEL WITH LIVE EXPLANATIONS -->
  <div class="card">
    <div class="card-title">🎛️ پیکربندی پایپ‌لاین و پارامترهای پردازش متن</div>

    <label style="font-weight:bold; font-size:13px;">۱. متن خام ورودی (ترکیب فارسی، حروف عربی ي/ك، نیم‌فاصله اشتباه، HTML، لینک، ایمیل، موبایل و جمله منفی):</label>
    <textarea id="txt" rows="3">{DEFAULT_SAMPLE}</textarea>

    <div class="controls-grid">
      <!-- 4. Tokenization -->
      <div class="control-box">
        <div>
          <label>۴. روش توکنایزیشن (Tokenization):</label>
          <select id="tok" onchange="updateHints()">
            <option value="regex_smart">Regex Smart (حفظ نیم‌فاصله ZWNJ)</option>
            <option value="whitespace">Whitespace (Split ساده با فاصله)</option>
            <option value="subword_bpe_demo">Subword BPE Demo (زیرواژه ترنسفورمر)</option>
            <option value="nltk">NLTK WordPunct Tokenizer</option>
            <option value="hazm">Hazm Word Tokenizer</option>
          </select>
        </div>
        <div class="hint-box" id="hint-tok"></div>
      </div>

      <!-- 5. Stopwords -->
      <div class="control-box">
        <div>
          <label>۵. روش حذف ایست‌واژه‌ها (Stopwords):</label>
          <select id="stop" onchange="updateHints()">
            <option value="dictionary">Dictionary (لیست واژگان دستوری)</option>
            <option value="statistical_idf">Statistical IDF (آماری log(N/df))</option>
            <option value="domain_specific">Domain-Specific (مبتنی بر دامنه)</option>
            <option value="ml_classifier">ML Classifier (تشخیص با یادگیری ماشین)</option>
            <option value="hybrid">Hybrid (ترکیب دیکشنری + آماری + دامنه)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-stop"></div>
      </div>

      <!-- Domain -->
      <div class="control-box">
        <div>
          <label>دامنه تخصصی (Domain Context):</label>
          <select id="dom" onchange="updateHints()">
            <option value="general">General (عمومی)</option>
            <option value="legal">Legal (حقوقی: ماده، قانون، دادگاه)</option>
            <option value="medical">Medical (پزشکی: بیمار، مطالعه، نتیجه)</option>
            <option value="scientific">Scientific (علمی: مقاله، روش، شکل)</option>
            <option value="sports">Sports (ورزشی: تیم، مسابقه، بازی)</option>
            <option value="ecommerce">E-Commerce (فروشگاهی: سلام، ارسال، کالا)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-dom"></div>
      </div>

      <!-- Negation Guard -->
      <div class="control-box">
        <div>
          <label>محافظ کلمات نفی (Negation Guard):</label>
          <select id="neg" onchange="updateHints()">
            <option value="true">فعال (حفظ نبود / نیست / not / never)</option>
            <option value="false">غیرفعال (حذف کلمات نفی همراه ایست‌واژه‌ها)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-neg"></div>
      </div>

      <!-- 6. Morphology -->
      <div class="control-box">
        <div>
          <label>۶. ریشه‌یابی (Morphology):</label>
          <select id="morph" onchange="updateHints()">
            <option value="lemma">Lemmatization (ریشه‌یابی دقیق فرهنگ لغت)</option>
            <option value="stem">Stemming (برش سریع پسوندها)</option>
            <option value="none">None (بدون ریشه‌یابی — خام)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-morph"></div>
      </div>

      <!-- English Stemmer -->
      <div class="control-box">
        <div>
          <label>الگوریتم Stemmer انگلیسی:</label>
          <select id="stemmer" onchange="updateHints()">
            <option value="snowball">Snowball / Porter2 (متعادل و استاندارد)</option>
            <option value="porter">Porter Stemmer (کلاسیک و آموزشی)</option>
            <option value="lancaster">Lancaster Stemmer (بسیار تهاجمی)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-stemmer"></div>
      </div>

      <!-- 7. Vectorization -->
      <div class="control-box">
        <div>
          <label>۷. نمایش برداری (Representation):</label>
          <select id="rep" onchange="updateHints()">
            <option value="tfidf">TF-IDF (وزن‌دهی واژه‌های ممیز + نرم L2)</option>
            <option value="bow">Bag-of-Words (شمارش ساده تکرار)</option>
          </select>
        </div>
        <div class="hint-box" id="hint-rep"></div>
      </div>

      <!-- N-Gram -->
      <div class="control-box">
        <div>
          <label>بازه ترکیب واژگان (N-Gram Range):</label>
          <select id="ngram" onchange="updateHints()">
            <option value="2">Unigram + Bigram (1,2) — حفظ ترکیب‌های دوتایی</option>
            <option value="1">Unigram (1,1) — فقط تک‌واژه‌ها</option>
          </select>
        </div>
        <div class="hint-box" id="hint-ngram"></div>
      </div>
    </div>

    <div class="two-col" style="margin-top:12px;">
      <div>
        <label style="font-weight:bold; font-size:13px;">افزودن اسناد جدید به پیکره (هر خط یک سند جدید — همه آمارها و ماتریس‌ها با آن به‌روز می‌شوند):</label>
        <textarea id="extraDocs" rows="2" placeholder="مثال: سیستم‌های هوشمند تشخیص تقلب بانکی با یادگیری ماشین و تحلیل سری‌های زمانی..."></textarea>
        <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:#cbd5e1; margin-top:6px; cursor:pointer;">
          <input type="checkbox" id="saveSample" style="width:auto; margin:0;">
          <span>ذخیره دائمی اسناد جدید در فایل <code>data/sample_texts.txt</code></span>
        </label>
      </div>
      <div>
        <label style="font-weight:bold; font-size:13px;">پرسش موتور جستجو (Cosine Similarity Query):</label>
        <input id="qry" value="پردازش زبان طبیعی و مدل‌های زبانی و بازیابی دانش">
        <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;">
          <button type="button" onclick="setPresetQuery('پردازش زبان طبیعی و مدل‌های زبانی و بازیابی دانش')" style="background:#1e293b; color:#7dd3fc; border:1px solid #334155; padding:4px 9px; border-radius:6px; font-size:11.5px; cursor:pointer;">نمونه ۱: پرسش دقیق فارسی</button>
          <button type="button" onclick="setPresetQuery('نوروپاطی')" style="background:#1e293b; color:#fde68a; border:1px solid #334155; padding:4px 9px; border-radius:6px; font-size:11.5px; cursor:pointer;">نمونه ۲: غلط املایی («نوروپاطی»)</button>
          <button type="button" onclick="setPresetQuery('multilingual corpora pipelines not good')" style="background:#1e293b; color:#a7f3d0; border:1px solid #334155; padding:4px 9px; border-radius:6px; font-size:11.5px; cursor:pointer;">نمونه ۳: پرسش انگلیسی</button>
        </div>
        <label style="display:block; font-weight:bold; font-size:12.5px; margin-top:8px;">انتخاب سند برای بررسی گام‌به‌گام و نمودارها (خودکار یا شماره سند):</label>
        <select id="inspectDoc">
          <option value="0">خودکار (برترین نتیجه جستجو یا سند #۱)</option>
          <option value="1">سند #۱</option>
          <option value="2">سند #۲</option>
          <option value="3">سند #۳</option>
          <option value="4">سند #۴</option>
          <option value="5">سند #۵</option>
          <option value="6">سند #۶</option>
          <option value="7">سند #۷</option>
          <option value="8">سند #۸ (در صورت افزودن سند جدید)</option>
          <option value="9">سند #۹ (در صورت افزودن سند جدید)</option>
        </select>
      </div>
    </div>

    <button class="btn-run" onclick="runAnalysis()">🚀 اجرای کامل پایپ‌لاین، رسم نمودارهای تحلیلی و مقایسه مدل‌ها</button>
  </div>

  <!-- RESULTS AREA -->
  <div id="results" style="display:none;">
    <div id="liveCorpusBanner" class="plain-guide" style="border-right-color:#38bdf8; margin-bottom:14px;"></div>

    <!-- VISUAL CHARTS SECTION -->
    <div class="two-col">
      <div class="card">
        <div class="card-title">📊 نمودار ۱: قیف کاهش حجم متن و واژگان (Token & Vocabulary Funnel)</div>
        <div class="plain-guide">
          <strong>📊 تحلیل کاهش ابعاد واژگان:</strong><br>
          متن خام پر از کلمات تکراری و حروف اضافه است. هر چقدر میله‌های پایین‌تر کوتاه‌تر شوند، یعنی مدل با <strong>داده‌های کمتر اما مفیدتر</strong> کار می‌کند:<br>
          • <strong>میله ۱:</strong> تعداد کل کلمات (توکن‌ها) پس از شکستن جمله.<br>
          • <strong>میله ۲:</strong> کلمات باقی‌مانده پس از حذف حروف اضافه (مثل «و، در، به»).<br>
          • <strong>میله ۴ (واژگان یکتا):</strong> اگر دو شکل «کتاب» و «کتاب‌ها» هر دو به «کتاب» تبدیل شوند، به جای ۲ ستون فقط ۱ ستون در جدول مدل ساخته می‌شود.
        </div>
        <div id="funnelChart"></div>
      </div>

      <div class="card">
        <div class="card-title">📈 نمودار ۲: مقایسه دقت ۶ خانواده مدل روی دیتاست آزمون چالش‌برانگیز</div>
        <div class="plain-guide">
          <strong>📈 ارزیابی عملکرد مدل‌ها روی داده آزمون:</strong><br>
          این نمودار مدل‌ها را روی ۴۸ جمله واقعی (شامل جملات منفی و تله‌های نفی) آزمایش می‌کند و <strong>بدون Overfitting</strong> نشان می‌دهد:<br>
          • <strong>مدل‌های خطی (Logistic Regression / Naive Bayes):</strong> روی داده‌های متنی TF-IDF سریع و دقیق‌اند، اما اگر <em>محافظ نفی</em> یا <em>Bigram</em> را خاموش کنید، دقتشان افت می‌کند!<br>
          • <strong>Random Forest:</strong> چون درخت تصمیم روی جدول‌های پرصفر (Sparse) فقط چند کلمه تصادفی می‌بیند، دقت پایین‌تری می‌گیرد (طبق جزوه، Random Forest برای تشخیص ایست‌واژه مناسب‌تر است تا طبقه‌بندی کل متن).
        </div>
        <div id="modelChart"></div>
      </div>
    </div>

    <!-- ABLATION & IMPACT COMPARISON -->
    <div class="card">
      <div class="card-title">🔬 آزمایش تأثیر (Ablation Study): مقایسه «بودن» در برابر «نبودن» پیش‌پردازش و محافظ نفی</div>
      <div class="ablation-grid" id="ablationArea"></div>
    </div>

    <!-- STEP BY STEP CARDS -->
    <div class="card">
      <div class="card-title">🪜 کالبدشکافی گام‌به‌گام (Step-by-Step) مراحل ۱ تا ۶ با توضیح عملکرد، تأثیر و شواهد تغییر</div>
      <div id="stepByStepArea"></div>
    </div>

    <!-- DEDICATED VECTOR REPRESENTATION STUDIO (CHARTS, HEATMAPS & SEMANTIC SIMULATION) -->
    <div class="card">
      <div class="card-title">🧮 آزمایشگاه نمایش‌های برداری (تبدیل متن به عدد برای مدل‌های یادگیری ماشین)</div>
      <div class="plain-guide">
        <strong>💡 چرا این بخش وجود دارد؟</strong> مدل‌های یادگیری ماشین نمی‌توانند حروف و کلمات را مستقیماً بخوانند؛ بنابراین متن تمیز را به یک <strong>جدول اعداد (ماتریس)</strong> تبدیل می‌کنیم. اگر <em>Bag-of-Words</em> را انتخاب کنید فقط تعداد تکرار شمرده می‌شود، اما در <em>TF-IDF</em> به کلمات کلیدی و تخصصی وزن بالاتری داده می‌شود.
      </div>

      <div class="two-col">
        <div class="ablation-box">
          <h4 style="color:#818cf8;">⚖️ نمودار ۳: مهم‌ترین کلمات کلیدی سند شما (Top Weighted Features)</h4>
          <div class="plain-guide" style="font-size:12px;">
            <strong>راهنمای خواندن:</strong> هر میله نشان می‌دهد کدام کلمه یا ترکیب دوتایی (Bigram) بیشترین امتیاز را در این سند گرفته است. در TF-IDF کلماتی که فقط در همین سند پرتکرارند (نه در همه اسناد) صدرنشین می‌شوند.
          </div>
          <div id="tfidfChart"></div>
        </div>

        <div class="ablation-box">
          <h4 style="color:#38bdf8;">💾 نمودار ۴: صرفه‌جویی در حافظه با ماتریس خلوت (Dense vs Sparse)</h4>
          <div class="plain-guide" style="font-size:12px;">
            <strong>راهنمای خواندن:</strong> چون هر سند فقط شامل بخش کوچکی از کل کلمات زبان است، بیش از ۸۰٪ خانه‌های جدول عدد <strong>صفر</strong> هستند. به جای ذخیره همه صفرها (Dense)، فقط خانه‌های غیرصفر را ذخیره می‌کنیم (Sparse) تا رم مصرفی به شدت کم شود.
          </div>
          <p id="sparsitySummary" style="font-size:12px; color:#cbd5e1; margin:2px 0 8px 0;"></p>
          <div id="memoryChart"></div>
          <div id="synonymSimBox" style="margin-top:10px; padding:8px; background:#090e1a; border-right:3px solid #f59e0b; border-radius:5px; font-size:12px; line-height:1.6;"></div>
        </div>
      </div>

      <div class="two-col" style="margin-top:14px;">
        <div class="ablation-box">
          <h4 style="color:#10b981;">📋 جدول ماتریس سند-واژه </h4>
          <div class="plain-guide" style="font-size:12px;">
            <strong>راهنمای ماتریس سند-واژه:</strong><br>
            این جدول دقیقاً شبیه یک فایل اکسل است: <strong>هر سطر یک سند</strong> (Doc #1 تا Doc #7) و <strong>هر ستون یک کلمه کلیدی</strong> است. عدد <code>0</code> یعنی آن کلمه در آن سند وجود ندارد، و هر عدد بزرگ‌تر از صفر نشان‌دهنده اهمیت (وزن) آن کلمه در آن سند است.
          </div>
          <div id="dtmTableArea" style="overflow-x:auto;"></div>
        </div>

        <div class="ablation-box">
          <h4 style="color:#f472b6;">🔥 جدول شباهت اسناد به یکدیگر (Cosine Similarity Heatmap)</h4>
          <div class="plain-guide" style="font-size:12px;">
            <strong>راهنمای ماتریس سند-واژه:</strong><br>
            این جدول میزان شباهت موضوعی هر دو سند را با عددی بین <strong>۰ (کاملاً بی‌ربط)</strong> تا <strong>۱ (کاملاً یکسان)</strong> نشان می‌دهد:<br>
            • قطر اصلی جدول همیشه <code>1.00</code> است چون هر سند با خودش ۱۰۰٪ شبیه است.<br>
            • خانه‌های سبزتر (مثل تقاطع <strong>Doc #1</strong> و <strong>Doc #5</strong> که هر دو درباره چالش‌های فارسی و نیم‌فاصله هستند) شباهت بیشتری دارند.
          </div>
          <div id="cosineHeatmapArea" style="overflow-x:auto;"></div>
        </div>
      </div>
    </div>

    <!-- MODEL PREDICTIONS & SEARCH -->
    <div class="two-col">
      <div class="card">
        <div class="card-title">🤖 نتیجه تشخیص لحن متن شما توسط ۶ مدل مختلف</div>
        <div class="plain-guide" style="font-size:12px;">
          <strong>راهنمای خواندن:</strong> متن بالایی شما پس از پاک‌سازی، هم‌زمان به ۶ مدل مختلف داده شده است تا ببینیم هر مدل چه تشخیصی (مثبت یا منفی) می‌دهد و چقدر مطمئن است.
        </div>
        <div id="predictionsTableArea"></div>
      </div>

      <div class="card">
        <div class="card-title">🔍 شبیه‌ساز موتور جستجو (پیدا کردن مرتبط‌ترین سند به سؤال شما)</div>
        <div class="plain-guide" style="font-size:12px;">
          <strong>راهنمای خواندن:</strong> پرسش جستجوی شما به بردار عددی تبدیل شده و با تمام ۷ سند مقایسه می‌شود. سندی که بیشترین امتیاز شباهت کسینوسی را بگیرد در <strong>رتبه #۱</strong> قرار می‌گیرد.
        </div>
        <div id="searchTableArea"></div>
      </div>
    </div>

    <!-- RAW JSON AUDIT -->
    <div class="card">
      <div class="card-title">🧾 گزارش کامل Audit & Evidence در قالب JSON (قابل استناد و ذخیره)</div>
      <pre id="jsonOut"></pre>
    </div>

  </div>
</div>

<script>
const HINTS = {{
  tok: {{
    regex_smart: ["واژه‌های فارسی دارای نیم‌فاصله (مثل «کتاب‌ها») و نشانه‌های ثابت (<URL>) را یکپارچه حفظ می‌کند.", "تأثیر: از دو تکه شدن کلمات فارسی و افت دقت مدل جلوگیری می‌کند."],
    whitespace: ["متن را فقط بر اساس فاصله (Space) می‌شکند.", "تأثیر: علائم نگارشی به کلمات می‌چسبند (مثلاً «خوبن،») و واژگان تکراری کاذب می‌سازند."],
    subword_bpe_demo: ["کلمات طولانی یا ناشناخته (OOV) را به پیشوند، ریشه و پسوند می‌شکند (پایه BERT و GPT).", "تأثیر: مشکل کلمات خارج از واژگان (UNK) را حل می‌کند اما تعداد توکن‌ها را افزایش می‌دهد."],
    nltk: ["توکنایزر استاندارد زبان انگلیسی مبتنی بر علائم نگارشی.", "تأثیر: برای انگلیسی دقیق است اما در فارسی نیم‌فاصله را به درستی تشخیص نمی‌دهد."],
    hazm: ["توکنایزر تخصصی زبان فارسی مبتنی بر قواعد صرفی.", "تأثیر: بهترین عملکرد برای متون فارسی خالص (در صورت نصب کتابخانه Hazm)."]
  }},
  stop: {{
    dictionary: ["حروف اضافه، ربط و افعال کمکی پرتکرار را از لیست استاندارد فارسی و انگلیسی حذف می‌کند.", "تأثیر: حدود ۳۰ تا ۴۵ درصد حجم متن کم می‌شود و تمرکز روی کلمات کلیدی می‌رود."],
    statistical_idf: ["کلماتی که در اکثر اسناد تکرار شده‌اند و IDF=log(N/df) پایینی دارند را خودکار حذف می‌کند.", "تأثیر: مستقل از زبان است و کلمات عمومی کل پیکره را بدون نیاز به دیکشنری می‌شناسد."],
    domain_specific: ["واژگان بسیار رایج در یک حوزه خاص (مثل «بیمار» در پزشکی یا «دادگاه» در حقوقی) را حذف می‌کند.", "تأثیر: دقت طبقه‌بندی موضوعی در متون تخصصی را به شدت بالا می‌برد."],
    ml_classifier: ["با محاسبه بردار ویژگی (TF، IDF، طول کلمه) احتمال ایست‌واژه بودن هر توکن را با مدل خطی می‌سنجد.", "تأثیر: شناسایی هوشمند ایست‌واژه‌ها بر اساس رفتار آماری پیکره داده."],
    hybrid: ["ترکیب هم‌زمان لیست دیکشنری، فیلتر آماری IDF و واژگان تخصصی دامنه.", "تأثیر: بیشترین پاک‌سازی و فشرده‌سازی واژگان برای موتورهای جستجو."]
  }},
  dom: {{
    general: ["بدون اعمال فیلتر دامنه تخصصی اضافی.", "تأثیر: مناسب برای متون عمومی و ترکیبی."],
    legal: ["واژگان پرتکرار حقوقی (ماده، قانون، شاکی، دادگاه) را به لیست حذف اضافه می‌کند.", "تأثیر: تمرکز مدل روی موضوع اصلی پرونده حقوقی."],
    medical: ["واژگان پرتکرار پزشکی (بیمار، مطالعه، نتیجه، بررسی) را فیلتر می‌کند.", "تأثیر: برجسته‌سازی نام بیماری‌ها و داروها."],
    scientific: ["واژگان ساختاری مقالات (مقاله، روش، شکل، جدول) را حذف می‌کند.", "تأثیر: استخراج دقیق مفاهیم پژوهشی."],
    sports: ["واژگان عمومی ورزشی (تیم، مسابقه، بازی، بازیکن) را فیلتر می‌کند.", "تأثیر: تمرکز روی نام تیم‌ها و رویدادها."],
    ecommerce: ["واژگان تعارفی نظرات کاربران (سلام، ممنون، خریدم، ارسال) را حذف می‌کند.", "تأثیر: تمرکز مستقیم روی کیفیت و قیمت کالا."]
  }},
  neg: {{
    true: ["کلمات نفی («نبود»، «نیست»، «نمی‌کنم»، «not»، «never») را از لیست حذف ایست‌واژه‌ها مستثنی می‌کند.", "تأثیر: جمله «خوب نبود» مثبت تفسیر نمی‌شود و دقت تحلیل احساسات حفظ می‌گردد."],
    false: ["تمام ایست‌واژه‌ها از جمله کلمات نفی را بدون استثنا حذف می‌کند.", "تأثیر هشداردهنده: جمله منفی «این محصول خوب نبود» به «محصول خوب» تبدیل و معکوس می‌شود!"]
  }},
  morph: {{
    lemma: ["واژه را با کمک فرهنگ لغت و بن فعل به ریشه معتبر برمی‌گرداند («می‌روم/رفتم» ← «رفت»، «mice» ← «mouse»).", "تأثیر: دقیق‌ترین روش کاهش اندازه واژگان بدون تولید کلمات بی‌معنا."],
    stem: ["پسوندها را با قواعد ثابت برش می‌دهد («studying» ← «studi»، «کتاب‌ها» ← «کتاب»).", "تأثیر: بسیار سریع است اما گاهی دچار برش بیش‌ازحد (Over-stemming) می‌شود."],
    none: ["هیچ ریشه‌یابی انجام نمی‌شود و تمام اشکال صرفی جداگانه ذخیره می‌شوند.", "تأثیر: «کتاب»، «کتاب‌ها» و «کتابم» سه ستون جدا در ماتریس می‌سازند و ابعاد بردار بزرگ می‌شود."]
  }},
  stemmer: {{
    snowball: ["نسخه بهبودیافته الگوریتم Porter با خطای کمتر.", "تأثیر: بهترین تعادل بین سرعت و دقت برای متون انگلیسی."],
    porter: ["الگوریتم کلاسیک و استاندارد برش پسوندهای انگلیسی.", "تأثیر: مناسب برای مقایسه‌های آموزشی و پایه."],
    lancaster: ["الگوریتم بسیار تهاجمی که کلمات را به کوتاه‌ترین ریشه ممکن می‌برد.", "تأثیر: کاهش شدید اندازه واژگان همراه با ریسک بالای Over-stemming."]
  }},
  rep: {{
    tfidf: ["وزن هر واژه = تکرار در سند × عکس بسامد در کل اسناد (هموارشده با نرم L2).", "تأثیر: کلمات عمومی وزن کم و کلمات ممیز (مثل «نوروپاتی») وزن بالا می‌گیرند."],
    bow: ["فقط تعداد تکرار خام هر واژه در سند را می‌شمارد (CountVectorizer).", "تأثیر: ساده و سریع است اما همه کلمات را هم‌ارزش فرض می‌کند."]
  }},
  ngram: {{
    "2": ["علاوه بر تک‌واژه‌ها (Unigram)، جفت‌واژه‌های متوالی (Bigram مثل «خوب نبود» و «not good») را هم می‌سازد.", "تأثیر: ترتیب محلی و ترکیب‌های نفی در مدل‌های خطی حفظ می‌شود."],
    "1": ["فقط تک‌واژه‌های مستقل را در نظر می‌گیرد و ترتیب کلمات کاملاً نادیده گرفته می‌شود.", "تأثیر: ابعاد ماتریس کوچک‌تر است اما تفاوت «خوب بود» و «خوب نبود» سخت‌تر تشخیص داده می‌شود."]
  }}
}};

function updateHints() {{
  const map = [
    ['tok', 'hint-tok'], ['stop', 'hint-stop'], ['dom', 'hint-dom'], ['neg', 'hint-neg'],
    ['morph', 'hint-morph'], ['stemmer', 'hint-stemmer'], ['rep', 'hint-rep'], ['ngram', 'hint-ngram']
  ];
  for (const [selId, boxId] of map) {{
    const val = document.getElementById(selId).value;
    const info = HINTS[selId][val];
    document.getElementById(boxId).innerHTML = `<strong>عملکرد:</strong> ${{info[0]}}<span class="hint-impact">⚡ ${{info[1]}}</span>`;
  }}
}}

function renderBarChart(containerId, items, colorHex) {{
  const maxVal = Math.max(...items.map(x => x.value), 0.001);
  let html = '';
  for (const item of items) {{
    const pct = Math.max(4, Math.round((item.value / maxVal) * 100));
    html += `
      <div class="bar-row">
        <div class="bar-label"><span>${{item.label}}</span><strong>${{item.display}}</strong></div>
        <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%; background:${{colorHex}};"></div></div>
      </div>`;
  }}
  document.getElementById(containerId).innerHTML = html;
}}

async function runAnalysis() {{
  const payload = {{
    text: document.getElementById('txt').value,
    tokenizer: document.getElementById('tok').value,
    stopwords: document.getElementById('stop').value,
    domain: document.getElementById('dom').value,
    morph: document.getElementById('morph').value,
    stemmer: document.getElementById('stemmer').value,
    rep: document.getElementById('rep').value,
    ngram: parseInt(document.getElementById('ngram').value),
    neg: document.getElementById('neg').value === 'true',
    query: document.getElementById('qry').value,
    extra_docs_text: document.getElementById('extraDocs').value,
    save_to_sample: document.getElementById('saveSample').checked,
    inspect_doc_id: parseInt(document.getElementById('inspectDoc').value || '0')
  }};

  const res = await fetch('/analyze', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload)
  }});
  const data = await res.json();
  document.getElementById('results').style.display = 'block';

  const doc = data.doc_analysis;
  const st = doc.stages;
  const mt = doc.metrics;
  const aud = doc.audit_and_evidence;
  const cs = data.corpus_summary || {{}};
  document.getElementById('liveCorpusBanner').innerHTML =
    `<strong>📌 وضعیت زنده پیکره و جستجو:</strong> ` +
    `تعداد کل اسناد فعال: <strong>${{cs.total_documents || data.total_corpus_docs}} سند</strong> | ` +
    `اندازه واژگان کل پیکره: <strong>${{cs.vocabulary_size}} واژه یکتا</strong> | ` +
    `سند در حال بررسی در نمودارها و گام‌به‌گام: <strong>سند #${{data.inspected_doc_index}}</strong> | ` +
    `پرسش فعال: <code>${{document.getElementById('qry').value}}</code>`;

  // 1. Render Token & Vocabulary Reduction Funnel Chart
  renderBarChart('funnelChart', [
    {{ label: '۱. توکن‌های اولیه پس از Tokenization', value: mt.initial_token_count, display: mt.initial_token_count + ' توکن' }},
    {{ label: '۲. توکن‌های باقی‌مانده پس از حذف Stopwords', value: mt.after_stopwords_count, display: mt.after_stopwords_count + ' توکن' }},
    {{ label: '۳. توکن‌های نهایی پس از ریشه‌یابی', value: mt.final_token_count, display: mt.final_token_count + ' توکن' }},
    {{ label: '۴. واژگان یکتای نهایی (Unique Types)', value: mt.unique_types_count, display: mt.unique_types_count + ' واژه (TTR: ' + mt.type_token_ratio_TTR + ')' }}
  ], '#38bdf8');

  // 2. Render 5-Model Benchmark Chart
  const bench = data.model_comparison.benchmark_metrics;
  const modelItems = Object.entries(bench).map(([k, v]) => ({{
    label: k.replace(/^\d+_/, '').replace(/_/g, ' '),
    value: v.F1_Score,
    display: `Acc: ${{(v.Accuracy*100).toFixed(0)}}% | F1: ${{(v.F1_Score*100).toFixed(0)}}%`
  }}));
  renderBarChart('modelChart', modelItems, '#10b981');

  // 3. Render Ablation Study (With vs Without Preprocessing & Negation Guard)
  const baseMt = data.baseline_no_prep.metrics;
  const noNegTokens = data.no_negation_guard.stages['5_filtered_tokens'];
  const keptNegTokens = st['5_filtered_tokens'];
  const negWordsFound = ['نبود', 'نیست', 'نمی‌کنم', 'not', 'never'].filter(w => keptNegTokens.includes(w));
  const negWordsLost = ['نبود', 'نیست', 'نمی‌کنم', 'not', 'never'].filter(w => !noNegTokens.includes(w) && keptNegTokens.includes(w));

  document.getElementById('ablationArea').innerHTML = `
    <div class="ablation-box">
      <h4 style="color:#38bdf8;">📉 تأثیر حذف ایست‌واژه و ریشه‌یابی بر اندازه واژگان</h4>
      <p style="font-size:12.5px; margin:4px 0;">• <strong>بدون حذف ایست‌واژه و ریشه‌یابی:</strong> ${{baseMt.initial_token_count}} توکن و <strong>${{baseMt.unique_types_count}}</strong> واژه یکتا</p>
      <p style="font-size:12.5px; margin:4px 0;">• <strong>با پایپ‌لاین فعال شما:</strong> ${{mt.final_token_count}} توکن و <strong>${{mt.unique_types_count}}</strong> واژه یکتا</p>
      <p style="font-size:12px; color:#86efac; margin:6px 0 0 0;">✅ نتیجه: کاهش <strong>${{Math.max(0, baseMt.unique_types_count - mt.unique_types_count)}}</strong> ستون اضافی در ماتریس برداری و جلوگیری از پراکندگی داده (Sparsity).</p>
    </div>
    <div class="ablation-box">
      <h4 style="color:#fbbf24;">🛡️ تأثیر محافظ کلمات نفی (Negation Guard)</h4>
      <p style="font-size:12.5px; margin:4px 0;">• <strong>کلمات نفی شناسایی‌شده در متن:</strong> ${{negWordsFound.length ? negWordsFound.join('، ') : 'موردی در این متن نبود'}}</p>
      <p style="font-size:12.5px; margin:4px 0;">• <strong>وضعیت فعلی محافظ:</strong> ${{payload.neg ? 'فعال (کلمات نفی حفظ شدند)' : 'غیرفعال (کلمات نفی حذف شدند!)'}}</p>
      <p style="font-size:12px; color:#fde68a; margin:6px 0 0 0;">⚡ اگر محافظ نفی خاموش شود، کلماتی مثل «${{negWordsLost.length ? negWordsLost.join('، ') : 'نبود / not'}}» حذف شده و بار معنایی جملات منفی کاملاً برعکس می‌شود.</p>
    </div>
  `;

  // 4. Render Detailed Step-by-Step Cards
  const stepsHtml = `
    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۱: متن خام (Raw Text)</span>
        <span class="step-tag">طول: ${{mt.raw_char_len}} کاراکتر</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> دریافت داده خام دقیقاً همان‌طور که از وب، شبکه‌های اجتماعی یا فایل‌های PDF استخراج شده است (بدون هیچ تغییری برای حفظ قابلیت بازگشت).</div>
      <div class="step-output">${{st['1_raw_text'].replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
    </div>

    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۲: پاک‌سازی نویز (Text Cleaning)</span>
        <span class="step-tag">عملیات انجام‌شده: ${{aud.cleaning.steps_applied.join(' | ') || 'بدون تغییر'}}</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> حذف تگ‌های HTML، نویسه‌های نامرئی یونیکد (ZWSP/BOM)، جایگزینی لینک و ایمیل و شماره تماس با نشانه‌های ثابت (&lt;URL&gt;, &lt;EMAIL&gt;, &lt;PHONE&gt;, &lt;NUM&gt;) قبل از پردازش اعداد.</div>
      <div class="step-impact">⚡ تأثیر: کاهش نویز ساختاری و حفظ نشانه وجود لینک/عدد بدون افزایش بی‌رویه ابعاد واژگان.</div>
      <div class="step-output">${{st['2_cleaned_text'].replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
    </div>

    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۳: نرمال‌سازی ۴ سطحی (Normalization)</span>
        <span class="step-tag">تغییرات: ${{aud.normalization.changes.join(' | ') || 'بدون تغییر'}}</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> اجرای Unicode NFKC، تبدیل حروف عربی (ي، ك، ة، أ) به فارسی استاندارد، حذف اعراب و کشیده، اصلاح نیم‌فاصله (ZWNJ) برای «می/نمی» و «ها/تر/ترین»، تبدیل محاوره به رسمی و کاهش حروف تکراری (سلاااام ← سلام).</div>
      <div class="step-impact">⚡ تأثیر: یکدست‌سازی اشکال مختلف یک واژه (مثلاً «كتاب‌هاي» و «کتابهای» هر دو به «کتاب‌های» تبدیل می‌شوند).</div>
      <div class="step-output">${{st['3_normalized_text'].replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
    </div>

    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۴: جمله‌بندی، توکنایزیشن و نگاشت عددی توکن‌ها (Tokenization & Token IDs — روش: ${{aud.tokenization.tokenizer_used}})</span>
        <span class="step-tag">${{aud.tokenization.sentence_count}} جمله | ${{mt.initial_token_count}} توکن</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> شکستن متن به جملات و واحدهای پردازشی (Tokens) و تبدیل هر توکن به شناسه عددی یکتا (<code>Token ID</code>)، بردار <code>One-Hot</code> و توالی هم‌اندازه با <code>Padding</code> و <code>Attention Mask</code> برای ورود به مدل‌های یادگیری ماشین و ترنسفورمر.</div>
      <div class="step-impact">⚡ تأثیر: کامپیوتر رشته متنی نمی‌فهمد؛ هر توکن به یک عدد (ID) و بردار باینری One-Hot تبدیل می‌شود.</div>
      <div class="step-output" style="margin-bottom:8px;"><strong>Tokens:</strong> ${{JSON.stringify(st['4_tokens'])}}</div>
      <div class="step-output" style="margin-bottom:8px; color:#7dd3fc;">
        <strong>Token IDs (۲۰ توکن اول):</strong> ${{JSON.stringify(aud.tokenization.token_ids_sample)}}<br>
        <strong>Padded Input IDs (طول ثابت ۱۶):</strong> ${{JSON.stringify(aud.tokenization.padded_sequence_len_16.input_ids)}}<br>
        <strong>Attention Mask (۱=توکن واقعی، ۰=Padding):</strong> ${{JSON.stringify(aud.tokenization.padded_sequence_len_16.attention_mask)}}
      </div>
      <div id="tokenOneHotArea" style="overflow-x:auto;"></div>
    </div>

    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۵: حذف ایست‌واژه‌ها (Stopwords — روش: ${{aud.stopwords.method}} | دامنه: ${{aud.stopwords.domain}})</span>
        <span class="step-tag">حذف ${{aud.stopwords.removed_count}} ایست‌واژه | باقی‌مانده: ${{mt.after_stopwords_count}}</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> حذف واژگان دستوری پرتکرار و کم‌اطلاع (حروف اضافه، ربط و افعال کمکی) با حفظ کلمات نفی کلیدی.</div>
      <div class="step-impact">⚡ واژه‌های حذف‌شده در این مرحله: ${{JSON.stringify(aud.stopwords.removed_tokens)}}</div>
      <div class="step-output">${{JSON.stringify(st['5_filtered_tokens'])}}</div>
    </div>

    <div class="step-card">
      <div class="step-header">
        <span class="step-name">مرحله ۶: ریشه‌یابی (Morphology — حالت: ${{aud.morphology.mode}})</span>
        <span class="step-tag">تبدیل ${{aud.morphology.changed_count}} واژه به ریشه</span>
      </div>
      <div class="step-desc"><strong>عملکرد:</strong> بازگرداندن اشکال صرفی و جمع به فرم پایه (مثلاً «کتاب‌ها» ← «کتاب»، «می‌روم/رفتم» ← «رفت»، «studying» ← «study») با محافظت از اصطلاحات فنی (NLP, RAG).</div>
      <div class="step-impact">⚡ جدول نگاشت ریشه‌ها: ${{JSON.stringify(aud.morphology.mappings)}}</div>
      <div class="step-output">${{JSON.stringify(st['6_final_roots'])}}</div>
    </div>
  `;
  document.getElementById('stepByStepArea').innerHTML = stepsHtml;

  // Render Token-level One-Hot Vector Representation Table inside Step 4
  const oh = aud.tokenization.one_hot_preview;
  if (oh && oh.columns && oh.columns.length) {{
    let ohHtml = `<div style="font-size:12px; color:#a5f3fc; margin:6px 0 4px 0;"><strong>🔢 نمایش برداری One-Hot در سطح توکن (هر سطر یک توکن و هر ستون یک بُعد از واژگان):</strong></div>` +
      `<table><tr><th>توکن</th><th>Token ID</th>${{oh.columns.map(c => `<th>${{c}}</th>`).join('')}}</tr>`;
    for (const r of oh.rows) {{
      ohHtml += `<tr><td><strong>${{r.token}}</strong></td><td style="text-align:center; color:#38bdf8;">${{r.token_id}}</td>` +
        r.one_hot.map(bit => `<td style="text-align:center; font-weight:${{bit ? 'bold' : 'normal'}}; color:${{bit ? '#10b981' : '#475569'}};">${{bit}}</td>`).join('') + `</tr>`;
    }}
    ohHtml += `</table>`;
    document.getElementById('tokenOneHotArea').innerHTML = ohHtml;
  }}

  // 5. Render Top TF-IDF / BoW Features Chart + Memory Chart + DTM + Cosine Heatmap
  const vr = data.vector_report;
  const topFeats = vr.top_features_per_doc[0] || [];
  renderBarChart('tfidfChart', topFeats.map(f => ({{
    label: `ویژگی: "${{f[0]}}"`,
    value: f[1],
    display: `وزن (${{payload.rep.toUpperCase()}}): ${{f[1]}}`
  }})), '#818cf8');

  const mem = vr.memory_comparison;
  document.getElementById('sparsitySummary').innerHTML =
    `ابعاد ماتریس: <strong>${{vr.matrix_shape[0]}} سند × ${{vr.matrix_shape[1]}} ویژگی</strong> | میزان خلوتی (Sparsity): <strong>${{vr.sparsity_percent}}% صفر</strong> (صرفه‌جویی ${{mem.saved_percent}}% در حافظه با CSR Sparse)`;
  renderBarChart('memoryChart', [
    {{ label: 'ماتریس کامل (Dense Array - 8B/cell)', value: mem.dense_matrix_bytes, display: mem.dense_matrix_bytes + ' Bytes' }},
    {{ label: 'ماتریس خلوت (Sparse CSR - 12B/non-zero)', value: mem.sparse_csr_bytes, display: mem.sparse_csr_bytes + ' Bytes' }}
  ], '#38bdf8');

  const syn = vr.semantic_limitation_demo;
  document.getElementById('synonymSimBox').innerHTML =
    `<strong>⚠️ محدودیت معنایی (پلی به جلسه بعد — Embeddings):</strong><br>مقایسه <code>"${{syn.pair[0]}}"</code> با <code>"${{syn.pair[1]}}"</code>:<br>` +
    `• شباهت در TF-IDF/BoW: <strong>${{syn.tfidf_cosine_similarity}}</strong> (چون واژه مشترک ندارند!) | شباهت در Word Embeddings: <strong>${{syn.dense_embedding_cosine_similarity}}</strong>`;

  // Document-Term Matrix Table
  const dtm = vr.document_term_matrix_preview;
  let dtmHtml = `<table><tr><th>سند</th>${{dtm.columns.map(c => `<th style="direction:ltr;">${{c}}</th>`).join('')}}</tr>`;
  for (const r of dtm.rows) {{
    dtmHtml += `<tr><td><strong>${{r.doc_id}}</strong></td>${{r.values.map(v => `<td style="text-align:center; color:${{v>0 ? '#6ee7b7' : '#64748b'}};">${{v}}</td>`).join('')}}</tr>`;
  }}
  dtmHtml += `</table>`;
  document.getElementById('dtmTableArea').innerHTML = dtmHtml;

  // Cosine Similarity Heatmap Table
  const simMat = vr.cosine_similarity_matrix;
  let cosHtml = `<table><tr><th>سند</th>${{simMat.map((_, idx) => `<th>Doc #${{idx+1}}</th>`).join('')}}</tr>`;
  simMat.forEach((row, i) => {{
    cosHtml += `<tr><td><strong>Doc #${{i+1}}</strong></td>`;
    row.forEach(val => {{
      const alpha = Math.min(0.85, Math.max(0.08, val * 1.6));
      const bg = val === 1.0 ? 'rgba(56, 189, 248, 0.35)' : `rgba(16, 185, 129, ${{alpha}})`;
      cosHtml += `<td style="text-align:center; background:${{bg}}; font-weight:${{val>0.08 ? 'bold' : 'normal'}};">${{val.toFixed(2)}}</td>`;
    }});
    cosHtml += `</tr>`;
  }});
  cosHtml += `</table>`;
  document.getElementById('cosineHeatmapArea').innerHTML = cosHtml;

  // 6. Render 5-Model Predictions Table & Search Results Table
  const preds = data.model_comparison.single_text_predictions;
  let predHtml = `<table><tr><th>مدل / معماری</th><th>نتیجه پیش‌بینی</th><th>امتیاز / جزئیات</th></tr>`;
  for (const [k, v] of Object.entries(preds)) {{
    if (k === 'input_text' || k === 'processed_tokens') continue;
    predHtml += `<tr><td><strong>${{k.replace(/^\d+_/, '').replace(/_/g, ' ')}}</strong></td><td>${{v.prediction}}</td><td>${{v.confidence !== undefined ? 'Prob: ' + v.confidence : v.details}} ${{v.backend ? '<br><small>'+v.backend+'</small>' : ''}}</td></tr>`;
  }}
  predHtml += `</table>`;
  document.getElementById('predictionsTableArea').innerHTML = predHtml;

  const sr = data.search_report || {{}};
  const matches = sr.top_matches || [];
  const qToksStr = sr.processed_query_tokens ? JSON.stringify(sr.processed_query_tokens) : '[]';
  const qVecs = sr.query_token_dense_vectors || [];
  const qPooledStr = sr.query_pooled_dense_vector ? '[' + sr.query_pooled_dense_vector.join(', ') + ']' : '[]';
  let tokVecLines = qVecs.map(tv =>
    `<div><strong style="color:#38bdf8;">${{tv.token}}</strong> &rarr; <code style="color:#fde68a;">[${{tv.dense_vector.join(', ')}}]</code></div>`
  ).join('');
  let sHtml = `<div style="background:#090e1a; padding:9px 11px; border-radius:6px; margin-bottom:8px; font-size:12px; border:1px solid #263959;">` +
    `<strong>پرسش جستجو:</strong> ${{sr.query || ''}} <br>` +
    `<strong>توکن‌های پردازش‌شده پرسش:</strong> <code style="color:#6ee7b7;">${{qToksStr}}</code><br>` +
    `<div style="margin-top:6px; padding-top:6px; border-top:1px dashed #1e293b;">` +
    `<strong>📐 بردار متراکم چندبُعدی هر توکن پرسش (Dense Token Vectors - 8D):</strong><br>` +
    `${{tokVecLines || '<span style="color:#94a3b8;">توکنی پس از فیلتر باقی نماند</span>'}}` +
    `<div style="margin-top:4px;"><strong>بردار میانگین کل پرسش:</strong> <code style="color:#a7f3d0;">${{qPooledStr}}</code></div>` +
    `</div></div>` +
    `<h4 style="margin:0 0 6px 0; color:#38bdf8; font-size:13.5px;">🔍 رتبه‌بندی موتور جستجوی کسینوسی (Cosine Similarity):</h4><table><tr><th>رتبه</th><th>سند</th><th>شباهت کسینوسی</th><th>پیش‌نمایش</th></tr>`;
  for (const m of matches) {{
    sHtml += `<tr><td>#${{m.rank}}</td><td>Doc #${{m.doc_id}}</td><td><strong>${{m.score}}</strong></td><td>${{m.preview.replace(/</g, '&lt;')}}</td></tr>`;
  }}
  sHtml += `</table>`;
  document.getElementById('searchTableArea').innerHTML = sHtml;

  // 7. Render Full JSON
  document.getElementById('jsonOut').textContent = JSON.stringify(data.full_json, null, 2);
}}

// Initialize hints and run once automatically on page load so the presenter immediately sees live charts!
function setPresetQuery(q) {{ document.getElementById('qry').value = q; runAnalysis(); }}
window.onload = function() {{
  updateHints();
  runAnalysis();
}};
</script>
</body>
</html>"""
            self.wfile.write(html_page.encode("utf-8"))

        def do_POST(self):
            if self.path == "/analyze":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                out = run_ui_pipeline(
                    body["text"], body["tokenizer"], body["stopwords"], body["domain"],
                    body["neg"], body["morph"], body["stemmer"], body["rep"], body["ngram"], body["query"],
                    body.get("extra_docs_text", ""), body.get("save_to_sample", False), body.get("inspect_doc_id", 0)
                )
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))

    print(f"Presentation Web UI is running at: http://127.0.0.1:{port} (Press Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), NLPHandler).serve_forever()


if __name__ == "__main__":
    launch_builtin_web_ui(7860)

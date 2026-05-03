"""
╔══════════════════════════════════════════════════════════════════╗
║ YKONE PULSE — Influencer Perception Intelligence                 ║
║ Powered by Groq / Llama-3.3 · Built for Ykone                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import json
import re
import time
from openai import OpenAI

# ── CONFIGURATION
APIFY_TOKEN = "add API here"
TRUST_POSITIVE_KW = ["authentic", "honest", "genuine", "real", "trust",
                     "credible", "vrai", "honnete", "fi9"]
TRUST_NEGATIVE_KW = ["fake", "ad", "sponsored", "paid", "promo", "pub",
                     "faux", "publicite", "msareh", "promotion"]

# ── PAGE CONFIG
st.set_page_config(
    page_title="Ykone Pulse",
    page_icon="⬤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #080808 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { background: #0e0e12 !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stSidebar"] * { color: rgba(240,237,232,0.7) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f0ede8 !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #f0ede8 !important;
}
.stButton > button {
    background: #C8102E !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
}
.stButton > button:hover { background: #e8102e !important; }
.stSpinner > div { border-top-color: #C8102E !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown("""
        <div style='font-size:10px;letter-spacing:0.22em;color:rgba(200,16,46,0.75);
        text-transform:uppercase;margin-bottom:1rem;font-weight:500'>
        ▸ Configuration
        </div>
        """, unsafe_allow_html=True)

    groq_key = st.text_input("Groq API Key", type="password",
                             help="Free at console.groq.com")
    tiktok_url = st.text_input("TikTok URL",
                               placeholder="https://www.tiktok.com/@creator/video/...")
    run_btn = st.button("ANALYSE →", use_container_width=True)

    st.markdown("---")

    st.markdown("""
        <div style='font-size:9px;color:rgba(240,237,232,0.25);line-height:2;letter-spacing:0.1em'>
        IPS FORMULA<br>
        ─────────────────────<br>
        Sentiment · 40%<br>
        Trust · 20%<br>
        Eng. Quality · 20%<br>
        Brand Affinity · 20%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div style='font-size:12px;color:rgba(240,237,232,0.28);font-style:italic;line-height:1.8'>
        "Data is the new luxury.<br>
        Everyone counts followers.<br>
        Only Pulse reads the subtext."
        </div>
        """, unsafe_allow_html=True)


# ── FETCH COMMENTS
def fetch_comments(url):
    endpoint = (
        "https://api.apify.com/v2/acts/clockworks~tiktok-comments-scraper"
        f"/run-sync-get-dataset-items?token={APIFY_TOKEN}"
    )

    try:
        response = requests.post(
            endpoint,
            json={"postURLs": [url], "resultsPerPage": 100},
            timeout=120
        )
    except Exception as e:
        st.error(f"Network error: {e}")
        return None

    if response.status_code not in [200, 201]:
        st.error(f"Apify API Error ({response.status_code}): {response.text}")
        return None

    data = response.json()
    if not data:
        st.warning("Apify returned empty dataset — TikTok may have blocked the scrape.")
        return None

    with st.expander("▸ Raw Apify payload (debug)"):
        st.write(f"Items: {len(data)}")
        st.json(data[:2])

    comments = []
    for item in data:
        if isinstance(item, dict):
            for key in ["text", "comment", "content"]:
                val = item.get(key)
                if val and isinstance(val, str) and val.strip():
                    comments.append(val.strip())
                    break

    if not comments:
        st.error("Apify returned data but no comment text found.")
        return None

    return pd.DataFrame({"text": comments})


# ── FEW-SHOT NLP PROMPT
def _build_prompt(comments_list):
    few_shot = [
        {"comment": "yiiiii mrigla barka",
         "sentiment": "positive", "intensity": 0.9, "trust_signal": 0.75,
         "brand_mentioned": False, "brand_sentiment": "neutral"},
        {"comment": "wled t9dar 3la rouha sah content",
         "sentiment": "positive", "intensity": 0.82, "trust_signal": 0.8,
         "brand_mentioned": False, "brand_sentiment": "neutral"},
        {"comment": "ayyyy sure 'authentic' w hiya t9ra le script",
         "sentiment": "negative", "intensity": 0.88, "trust_signal": 0.08,
         "brand_mentioned": False, "brand_sentiment": "neutral"},
        {"comment": "c'est clairement une pub deguisee, pas credible du tout",
         "sentiment": "negative", "intensity": 0.91, "trust_signal": 0.05,
         "brand_mentioned": True, "brand_sentiment": "negative"},
        {"comment": "ok but is this sponsored tho??",
         "sentiment": "neutral", "intensity": 0.45, "trust_signal": 0.4,
         "brand_mentioned": False, "brand_sentiment": "neutral"},
        {"comment": "fake wallah barra barra collab w kolchi",
         "sentiment": "negative", "intensity": 0.95, "trust_signal": 0.02,
         "brand_mentioned": True, "brand_sentiment": "negative"},
        {"comment": "love this product, the ykone collab is fire",
         "sentiment": "positive", "intensity": 0.87, "trust_signal": 0.72,
         "brand_mentioned": True, "brand_sentiment": "positive"},
    ]

    return f"""You are an elite NLP analyst for Ykone, a luxury influencer marketing agency.
LANGUAGES: Tunisian Darija, Arabizi (Arabic in Latin), French, English, mixed-code, emojis.
SARCASM RULES:
1. Positive words + negative emojis or tone = NEGATIVE sentiment
2. Quoted praise like 'sure authentic' = sarcasm = NEGATIVE
3. "Sure", "obviously", "wow" used ironically = NEGATIVE
FEW-SHOT EXAMPLES (learn from these exactly):
{json.dumps(few_shot, ensure_ascii=False, indent=2)}
OUTPUT: Return ONLY a valid JSON array — no markdown, no wrapping object.
Each element must have EXACTLY:
"sentiment": "positive" | "negative" | "neutral"
"intensity": float 0.0-1.0
"trust_signal": float 0.0-1.0 (1.0=authentic, 0.0=fake/ad)
"brand_mentioned": true | false
"brand_sentiment": "positive" | "negative" | "neutral"
COMMENTS ({len(comments_list)} items, preserve exact order):
{json.dumps(comments_list, ensure_ascii=False)}"""


# ── BULLETPROOF JSON PARSER
def _parse_json(raw):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    objs = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
    results = []
    for o in objs:
        try:
            results.append(json.loads(o))
        except Exception:
            pass
    return results


_DEFAULT = {
    "sentiment": "neutral", "intensity": 0.5,
    "trust_signal": 0.5, "brand_mentioned": False, "brand_sentiment": "neutral"
}


# ── ANALYZE SENTIMENT
def analyze_sentiment(df, api_key):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    comments_list = df["text"].astype(str).tolist()
    BATCH_SIZE = 60
    all_results = []
    total_batches = (len(comments_list) - 1) // BATCH_SIZE + 1
    progress = st.progress(0, text="Initializing NLP engine…")

    for i in range(total_batches):
        batch = comments_list[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        prompt = _build_prompt(batch)
        progress.progress(int((i / total_batches) * 90),
                          text=f"Analyzing batch {i+1}/{total_batches}…")

        batch_results = []
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system",
                         "content": "Output ONLY a valid JSON array. No markdown, no commentary."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.05,
                    max_tokens=4096,
                )
                batch_results = _parse_json(resp.choices[0].message.content)
                break
            except Exception as e:
                if attempt == 2:
                    st.warning(f"Batch {i+1} failed: {e}")
                else:
                    time.sleep(2 ** attempt)

        while len(batch_results) < len(batch):
            batch_results.append(_DEFAULT.copy())
        all_results.extend(batch_results[:len(batch)])

    progress.progress(100, text="Analysis complete.")
    time.sleep(0.3)
    progress.empty()

    if len(all_results) < len(df):
        all_results.extend([_DEFAULT.copy()] * (len(df) - len(all_results)))
    all_results = all_results[:len(df)]

    df["sentiment"] = [r.get("sentiment", "neutral") for r in all_results]
    df["intensity"] = [float(r.get("intensity", 0.5)) for r in all_results]
    df["trust_signal"] = [float(r.get("trust_signal", 0.5)) for r in all_results]
    df["brand_mentioned"] = [bool(r.get("brand_mentioned", False)) for r in all_results]
    df["brand_sentiment"] = [r.get("brand_sentiment", "neutral") for r in all_results]
    df["intensity"] = df["intensity"].clip(0.0, 1.0)
    df["trust_signal"] = df["trust_signal"].clip(0.0, 1.0)
    return df


# ── FEATURE ENGINEERING
def engineer_features(df):
    pol = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    df["sentiment_score"] = df.apply(
        lambda r: pol[r["sentiment"]] * r["intensity"], axis=1).clip(0.0, 1.0)

    emoji_re = re.compile(
        "[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000200D]+", flags=re.UNICODE)

    def trust(row):
        t = row["text"].lower()
        s = row["trust_signal"]
        s += sum(0.05 for kw in TRUST_POSITIVE_KW if kw in t)
        s -= sum(0.08 for kw in TRUST_NEGATIVE_KW if kw in t)
        return float(max(0.0, min(1.0, s)))

    def eq(row):
        text = row["text"]
        ln = len(text)
        ls = min(1.0, ln / 120.0)
        ec = len("".join(emoji_re.findall(text)))
        tc = max(1, ln - ec)
        er = ec / tc
        return float(max(0.0, min(1.0, ls * (1 - min(1.0, er * 1.5)))))

    def ba(row):
        if not row["brand_mentioned"]:
            return 0.5
        return {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(
            row["brand_sentiment"], 0.5) * row["intensity"]

    df["trust_score"] = df.apply(trust, axis=1)
    df["eq_score"] = df.apply(eq, axis=1)
    df["ba_score"] = df.apply(ba, axis=1)
    return df


# ── IPS FORMULA
def compute_ips(df):
    s = df["sentiment_score"].mean()
    t = df["trust_score"].mean()
    eq = df["eq_score"].mean()
    ba = df["ba_score"].mean()
    ips = (s * 0.40 + t * 0.20 + eq * 0.20 + ba * 0.20) * 100
    return {
        "ips": round(ips, 1),
        "sentiment": round(s * 100, 1),
        "trust": round(t * 100, 1),
        "eq": round(eq * 100, 1),
        "ba": round(ba * 100, 1),
    }


# ── DASHBOARD HTML
def build_dashboard(df, scores, total):
    rows_json = df[[
        "text", "sentiment", "intensity", "trust_score",
        "eq_score", "brand_mentioned", "brand_sentiment", "sentiment_score", "ba_score"
    ]].to_json(orient="records", force_ascii=False)
    scores_json = json.dumps(scores)

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,700;1,400&display=swap" rel="stylesheet"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Space Grotesk',sans-serif;background:#080808;color:#f0ede8;padding:2rem 2.5rem 4rem}}
:root{{--red:#C8102E;--dim:rgba(240,237,232,0.45);--border:rgba(255,255,255,0.08);--glass:rgba(255,255,255,0.04)}}
.eyebrow{{font-size:10px;letter-spacing:0.25em;color:rgba(200,16,46,0.75);text-transform:uppercase;margin-bottom:8px;font-weight:500}}
.logo{{font-family:'Playfair Display',serif;font-size:48px;font-weight:700;line-height:0.95}}
.logo span{{color:var(--red)}}
.tagline{{font-family:'Playfair Display',serif;font-style:italic;font-size:14px;color:var(--dim);margin-top:8px}}
.header{{padding:2rem 0 1.8rem;border-bottom:1px solid rgba(200,16,46,0.2);margin-bottom:2.5rem;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:-40px;left:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(200,16,46,0.1) 0%,transparent 70%);pointer-events:none}}
.divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(200,16,46,0.3),transparent);margin:2.5rem 0}}
.sec{{font-size:9px;letter-spacing:0.25em;color:rgba(200,16,46,0.65);text-transform:uppercase;font-weight:600;margin-bottom:14px}}
.g4{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:2rem}}
.g2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-bottom:2rem}}
.card{{background:var(--glass);border:1px solid var(--border);border-radius:4px;padding:20px}}
.card-red{{background:rgba(200,16,46,0.05);border:1px solid rgba(200,16,46,0.25);border-radius:4px;padding:24px}}
.kl{{font-size:9px;letter-spacing:0.2em;color:var(--dim);text-transform:uppercase;font-weight:500;margin-bottom:6px}}
.kw{{font-size:9px;color:rgba(200,16,46,0.6);letter-spacing:0.1em;font-weight:600}}
.kv{{font-size:34px;font-weight:600;color:#f0ede8;line-height:1}}
.ks{{font-size:11px;color:var(--dim);margin-top:5px;font-style:italic}}
.ips-wrap{{display:flex;align-items:center;gap:2.5rem}}
svg.gauge{{width:200px;height:120px;overflow:visible}}
.ips-tier{{font-family:'Playfair Display',serif;font-size:30px;color:#f0ede8;margin-bottom:8px}}
.ips-nar{{font-family:'Playfair Display',serif;font-style:italic;font-size:14px;color:var(--dim);line-height:1.7}}
.ips-meta{{font-size:9px;letter-spacing:0.15em;color:rgba(240,237,232,0.22);margin-top:14px;border-top:1px solid var(--border);padding-top:12px}}
.bar-row{{display:flex;align-items:center;gap:12px;margin-bottom:12px}}
.bar-label{{font-size:11px;color:var(--dim);width:130px;flex-shrink:0}}
.bar-track{{flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:2px}}
.bar-pct{{font-size:12px;font-weight:500;width:40px;text-align:right;color:#f0ede8}}
.chart-wrap{{position:relative;width:100%}}
.legend{{display:flex;gap:16px;margin-bottom:10px;font-size:11px;color:var(--dim);flex-wrap:wrap}}
.leg-dot{{width:10px;height:10px;border-radius:2px;flex-shrink:0;margin-top:2px}}
.table-header{{display:grid;grid-template-columns:1fr 90px 60px 70px;gap:12px;padding:0 0 8px;border-bottom:1px solid rgba(200,16,46,0.2);margin-bottom:4px}}
.table-header span{{font-size:9px;letter-spacing:0.18em;color:rgba(200,16,46,0.6);text-transform:uppercase;font-weight:600}}
.comment-row{{padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.04);display:grid;grid-template-columns:1fr 90px 60px 70px;gap:12px;align-items:start}}
.comment-text{{font-size:12px;color:rgba(240,237,232,0.72);line-height:1.6}}
.badge{{font-size:9px;letter-spacing:0.1em;padding:3px 8px;border-radius:2px;font-weight:600;text-transform:uppercase}}
.bp{{background:rgba(240,237,232,0.1);color:rgba(240,237,232,0.7)}}
.bn{{background:rgba(200,16,46,0.15);color:#c8102e}}
.bu{{background:rgba(255,255,255,0.05);color:rgba(240,237,232,0.35)}}
.iw{{display:flex;align-items:center;gap:6px}}
.idot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.num{{font-size:11px;color:rgba(240,237,232,0.5);font-variant-numeric:tabular-nums}}
.fr{{display:flex;gap:10px;margin-bottom:14px;align-items:center;flex-wrap:wrap}}
.fl{{font-size:10px;color:var(--dim);letter-spacing:0.1em}}
select{{background:#111113;border:1px solid rgba(255,255,255,0.1);color:#f0ede8;font-family:'Space Grotesk',sans-serif;font-size:11px;padding:6px 10px;border-radius:3px;cursor:pointer;outline:none}}
canvas{{display:block}}
</style>
</head><body>
<div class="header">
<div class="eyebrow">▸ Ykone Intelligence Platform · v2.0</div>
<div class="logo">YKONE <span>PULSE</span></div>
<div class="tagline">Influencer Perception Intelligence — beyond engagement metrics</div>
</div>
<div class="sec">01 / Influencer Perception Score</div>
<div class="card-red ips-wrap" style="margin-bottom:2rem">
<div style="flex:0 0 200px">
<svg class="gauge" viewBox="0 0 200 120" role="img" aria-label="IPS gauge">
<defs>
<linearGradient id="gRed" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#3a0010"/>
<stop offset="100%" stop-color="#c8102e"/>
</linearGradient>
</defs>
<path d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12" stroke-linecap="round"/>
<path id="gaugeArc" d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="url(#gRed)" stroke-width="12" stroke-linecap="round" stroke-dasharray="0 251"/>
<text id="gaugeVal" x="100" y="100" text-anchor="middle" fill="#f0ede8" font-family="'Space Grotesk',sans-serif" font-size="28" font-weight="600">—</text>
<text x="100" y="116" text-anchor="middle" fill="rgba(240,237,232,0.35)" font-family="'Space Grotesk',sans-serif" font-size="9" letter-spacing="3">IPS</text>
<text x="22" y="120" fill="rgba(240,237,232,0.25)" font-family="'Space Grotesk',sans-serif" font-size="8">0</text>
<text x="172" y="120" fill="rgba(240,237,232,0.25)" font-family="'Space Grotesk',sans-serif" font-size="8">100</text>
</svg>
</div>
<div style="flex:1">
<div class="kl" style="margin-bottom:4px">Perception tier</div>
<div class="ips-tier" id="ipsTier">—</div>
<div class="ips-nar" id="ipsNar">—</div>
<div class="ips-meta" id="ipsMeta"></div>
</div>
</div>
<div class="sec">02 / Score decomposition</div>
<div class="g4" id="kpiCards"></div>
<div class="divider"></div>
<div class="sec">03 / Perception map — Sentiment x Trust</div>
<div class="g2">
<div class="card">
<div class="legend" id="scatterLegend"></div>
<div class="chart-wrap" style="height:260px"><canvas id="scatterChart"></canvas></div>
</div>
<div class="card">
<div class="legend" id="donutLegend"></div>
<div class="chart-wrap" style="height:260px"><canvas id="donutChart"></canvas></div>
</div>
</div>
<div class="sec">04 / Signal breakdown</div>
<div class="g2">
<div class="card">
<div class="chart-wrap" style="height:220px"><canvas id="barChart"></canvas></div>
</div>
<div class="card">
<div class="chart-wrap" style="height:220px"><canvas id="brandChart"></canvas></div>
</div>
</div>
<div class="sec">05 / IPS component analysis</div>
<div class="card" style="margin-bottom:2rem"><div id="componentBars"></div></div>
<div class="divider"></div>
<div class="sec">06 / Signal database</div>
<div class="fr">
<div class="fl">Filter:</div>
<select id="sentFilter" onchange="renderTable()">
<option value="all">All sentiments</option>
<option value="positive">Positive</option>
<option value="neutral">Neutral</option>
<option value="negative">Negative</option>
</select>
<select id="sortFilter" onchange="renderTable()">
<option value="intensity">Sort by intensity</option>
<option value="trust_score">Sort by trust</option>
<option value="eq_score">Sort by eng. quality</option>
<option value="sentiment_score">Sort by sentiment score</option>
</select>
<div class="fl" id="rowCount" style="margin-left:auto;color:rgba(240,237,232,0.25)"></div>
</div>
<div class="table-header">
<span>Comment</span><span>Sentiment</span><span>Trust</span><span>Intensity</span>
</div>
<div id="commentTable"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const ROWS = {rows_json};
const SCORES = {scores_json};
const TOTAL = {total};
const COLORS = {{positive:'#d0cdc8',neutral:'#444446',negative:'#c8102e'}};
function tierInfo(ips){{
  if(ips>=80) return{{tier:'Elite Perception',nar:'Exceptional trust and emotional resonance. Highest priority for brand activation.'}};
  if(ips>=65) return{{tier:'Strong Signal',nar:'High positive sentiment with solid authenticity markers. Recommended for campaign deployment.'}};
  if(ips>=50) return{{tier:'Emerging Potential',nar:'Mixed signals — brand affinity present but trust needs reinforcement over time.'}};
  if(ips>=35) return{{tier:'Perception Risk',nar:'Significant skepticism detected. Monitor closely before committing budget.'}};
  return{{tier:'Critical Alert',nar:'Severe trust gap and negative sentiment dominance. Do not activate for this campaign.'}};
}}
function setGauge(val){{
  const filled=(val/100)*251;
  document.getElementById('gaugeArc').setAttribute('stroke-dasharray',filled+' '+(251-filled));
  document.getElementById('gaugeVal').textContent=val;
}}
function renderKPI(){{
  const items=[
    {{l:'Sentiment',v:SCORES.sentiment,w:'40%',s:'Polarity x emotional intensity'}},
    {{l:'Trust',v:SCORES.trust,w:'20%',s:'Authenticity signal strength'}},
    {{l:'Eng. Quality',v:SCORES.eq,w:'20%',s:'Long-form depth, low emoji ratio'}},
    {{l:'Brand Affinity',v:SCORES.ba,w:'20%',s:'Brand-mention sentiment score'}},
  ];
  document.getElementById('kpiCards').innerHTML=items.map(it=>`
    <div class="card">
      <div class="kl">${{it.l}} <span class="kw">· ${{it.w}}</span></div>
      <div class="kv">${{it.v}}</div>
      <div class="ks">${{it.s}}</div>
    </div>`).join('');
  document.getElementById('componentBars').innerHTML=items.map(it=>{{
    const pct=it.v;
    const fc=pct>=70?'#c8102e':pct>=50?'rgba(200,16,46,0.45)':'rgba(255,255,255,0.12)';
    return `<div class="bar-row">
      <div class="bar-label">${{it.l}} <span style="color:rgba(200,16,46,0.5);font-size:9px">· ${{it.w}}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{fc}}"></div></div>
      <div class="bar-pct">${{pct}}</div>
    </div>`;
  }}).join('');
}}
function renderCharts(){{
  const sc={{positive:0,neutral:0,negative:0}};
  const si={{positive:[],neutral:[],negative:[]}};
  ROWS.forEach(r=>{{sc[r.sentiment]++;si[r.sentiment].push(r.intensity);}});
  const avg=k=>{{const a=si[k]||[];return a.length?+(a.reduce((s,v)=>s+v,0)/a.length).toFixed(2):0;}};
  const br=ROWS.filter(r=>r.brand_mentioned);
  const bc={{positive:0,neutral:0,negative:0}};
  br.forEach(r=>bc[r.brand_sentiment]++);
  document.getElementById('donutLegend').innerHTML=['positive','neutral','negative'].map(s=>
    `<span style="display:flex;align-items:center;gap:5px">
      <span class="leg-dot" style="background:${{COLORS[s]}}"></span>
      ${{s.charAt(0).toUpperCase()+s.slice(1)}} ${{Math.round(sc[s]/TOTAL*100)}}%
    </span>`).join('');
  document.getElementById('scatterLegend').innerHTML=['positive','neutral','negative'].map(s=>
    `<span style="display:flex;align-items:center;gap:5px">
      <span class="leg-dot" style="background:${{COLORS[s]}}"></span>${{s}}
    </span>`).join('');
  new Chart(document.getElementById('donutChart'),{{
    type:'doughnut',
    data:{{labels:['Positive','Neutral','Negative'],
      datasets:[{{data:[sc.positive,sc.neutral,sc.negative],
        backgroundColor:['#d0cdc8','#333335','#c8102e'],
        borderColor:'#080808',borderWidth:3,hoverOffset:4}}]}},
    options:{{responsive:true,maintainAspectRatio:false,cutout:'65%',
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>' '+ctx.label+': '+ctx.raw+' ('+Math.round(ctx.raw/TOTAL*100)+'%)'}}}}}}}}
  }});
  new Chart(document.getElementById('scatterChart'),{{
    type:'scatter',
    data:{{datasets:['positive','neutral','negative'].map(s=>{{
      return {{label:s,
        data:ROWS.filter(r=>r.sentiment===s).map(r=>{{return {{x:r.trust_score,y:r.sentiment_score,_t:r.text.slice(0,45)}}}}),
        backgroundColor:COLORS[s]+'bb',pointRadius:5,pointHoverRadius:7}};
    }})}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>ctx.raw._t+'...'}}}}}},
      scales:{{
        x:{{min:0,max:1,title:{{display:true,text:'Trust',color:'rgba(240,237,232,0.35)',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'rgba(240,237,232,0.3)',font:{{size:9}}}}}},
        y:{{min:0,max:1,title:{{display:true,text:'Sentiment',color:'rgba(240,237,232,0.35)',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'rgba(240,237,232,0.3)',font:{{size:9}}}}}}
      }}
    }}
  }});
  new Chart(document.getElementById('barChart'),{{
    type:'bar',
    data:{{labels:['Positive','Neutral','Negative'],
      datasets:[{{label:'Avg intensity',
        data:[avg('positive'),avg('neutral'),avg('negative')],
        backgroundColor:['rgba(208,205,200,0.75)','rgba(68,68,70,0.75)','rgba(200,16,46,0.75)'],
        borderRadius:2,borderSkipped:false}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        title:{{display:true,text:'Average emotional intensity by sentiment',color:'rgba(240,237,232,0.45)',font:{{size:11,weight:'400'}},padding:{{bottom:10}}}}}},
      scales:{{
        x:{{grid:{{display:false}},ticks:{{color:'rgba(240,237,232,0.4)',font:{{size:10}}}}}},
        y:{{min:0,max:1,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'rgba(240,237,232,0.3)',font:{{size:9}}}}}}
      }}
    }}
  }});
  new Chart(document.getElementById('brandChart'),{{
    type:'bar',
    data:{{labels:['Positive','Neutral','Negative'],
      datasets:[{{label:'Brand mentions',
        data:[bc.positive,bc.neutral,bc.negative],
        backgroundColor:['rgba(208,205,200,0.75)','rgba(68,68,70,0.75)','rgba(200,16,46,0.75)'],
        borderRadius:2,borderSkipped:false}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        title:{{display:true,text:'Brand mention sentiment ('+br.length+' of '+TOTAL+' comments)',color:'rgba(240,237,232,0.45)',font:{{size:11,weight:'400'}},padding:{{bottom:10}}}}}},
      scales:{{
        x:{{grid:{{display:false}},ticks:{{color:'rgba(240,237,232,0.4)',font:{{size:10}}}}}},
        y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'rgba(240,237,232,0.3)',font:{{size:9}},stepSize:1}}}}
      }}
    }}
  }});
}}
function renderTable(){{
  const sent=document.getElementById('sentFilter').value;
  const sort=document.getElementById('sortFilter').value;
  let rows=[...ROWS];
  if(sent!=='all') rows=rows.filter(r=>r.sentiment===sent);
  rows.sort((a,b)=>b[sort]-a[sort]);
  document.getElementById('rowCount').textContent=rows.length+' signals';
  const bc={{positive:'bp',neutral:'bu',negative:'bn'}};
  const dc={{positive:'rgba(208,205,200,0.8)',neutral:'rgba(68,68,70,0.8)',negative:'#c8102e'}};
  document.getElementById('commentTable').innerHTML=rows.slice(0,80).map(r=>`
    <div class="comment-row">
      <div class="comment-text">${{r.text}}</div>
      <div><span class="badge ${{bc[r.sentiment]}}">${{r.sentiment}}</span></div>
      <div class="num">${{(r.trust_score*100).toFixed(0)}}</div>
      <div class="iw">
        <div class="idot" style="background:${{dc[r.sentiment]}};opacity:${{(0.3+r.intensity*0.7).toFixed(2)}}"></div>
        <span class="num">${{r.intensity.toFixed(2)}}</span>
      </div>
    </div>`).join('');
}}
const {{tier,nar}}=tierInfo(SCORES.ips);
setGauge(SCORES.ips);
document.getElementById('ipsTier').textContent=tier;
document.getElementById('ipsNar').textContent=nar;
document.getElementById('ipsMeta').textContent='Based on '+TOTAL+' audience signals · Groq Llama-3.3 · IPS v2.0 · Ykone Intelligence';
renderKPI();
renderCharts();
renderTable();
</script>
</body></html>"""


# ── LANDING PAGE HTML
LANDING_HTML = """<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,700;1,400&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Space Grotesk',sans-serif;background:#080808;color:#f0ede8;padding:2rem 2.5rem}
.header{padding:2.5rem 0 2rem;border-bottom:1px solid rgba(200,16,46,0.2);margin-bottom:3rem;position:relative;overflow:hidden}
.header::before{content:'';position:absolute;top:-40px;left:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(200,16,46,0.1) 0%,transparent 70%);pointer-events:none}
.eyebrow{font-size:10px;letter-spacing:0.25em;color:rgba(200,16,46,0.75);text-transform:uppercase;margin-bottom:8px;font-weight:500}
.logo{font-family:'Playfair Display',serif;font-size:52px;font-weight:700;line-height:0.95;color:#f0ede8}
.logo span{color:#C8102E}
.tagline{font-family:'Playfair Display',serif;font-style:italic;font-size:15px;color:rgba(240,237,232,0.45);margin-top:10px}
.empty{text-align:center;padding:5rem 2rem;border:1px solid rgba(255,255,255,0.05);border-radius:4px;background:rgba(255,255,255,0.02)}
.et{font-family:'Playfair Display',serif;font-size:22px;color:rgba(240,237,232,0.2);margin-bottom:10px}
.es{font-size:10px;letter-spacing:0.2em;color:rgba(200,16,46,0.3);text-transform:uppercase}
.pills{display:flex;gap:12px;margin-top:2rem;flex-wrap:wrap;justify-content:center}
.pill{padding:6px 16px;border:1px solid rgba(255,255,255,0.07);border-radius:20px;font-size:10px;letter-spacing:0.1em;color:rgba(240,237,232,0.3);text-transform:uppercase}
</style>
</head><body>
<div class="header">
<div class="eyebrow">▸ Ykone Intelligence Platform · v2.0</div>
<div class="logo">YKONE <span>PULSE</span></div>
<div class="tagline">Influencer Perception Intelligence — beyond engagement metrics</div>
</div>
<div class="empty">
<div class="et">Enter a TikTok URL in the sidebar to begin analysis</div>
<div class="es">▸ Groq Llama-3.3 · Darija aware · IPS v2.0 scoring</div>
<div class="pills">
<div class="pill">Sentiment · 40%</div>
<div class="pill">Trust · 20%</div>
<div class="pill">Eng. Quality · 20%</div>
<div class="pill">Brand Affinity · 20%</div>
</div>
</div>
</body></html>"""


# ── MAIN FLOW
if not run_btn:
    components.html(LANDING_HTML, height=520, scrolling=False)
else:
    if not tiktok_url:
        st.sidebar.warning("Please enter a TikTok URL.")
        components.html(LANDING_HTML, height=520, scrolling=False)
        st.stop()

    if not groq_key:
        st.sidebar.error("Groq API Key required.")
        components.html(LANDING_HTML, height=520, scrolling=False)
        st.stop()

    # Step 1 — Fetch
    with st.spinner("▸ Connecting to Apify scraper…"):
        df = fetch_comments(tiktok_url)
    if df is None:
        st.stop()
    st.sidebar.success(f"✓ {len(df)} comments fetched")

    # Step 2 — NLP
    with st.spinner(f"▸ Groq Llama-3.3 reading {len(df)} comments…"):
        df = analyze_sentiment(df, groq_key)

    # Step 3 — IPS
    df = engineer_features(df)
    scores = compute_ips(df)

    # IPS pill in sidebar
    st.sidebar.markdown(f"""
        <div style='background:rgba(200,16,46,0.1);border:1px solid rgba(200,16,46,0.3);
        border-radius:4px;padding:12px;margin-top:1rem;text-align:center'>
        <div style='font-size:9px;letter-spacing:0.2em;color:rgba(200,16,46,0.7);text-transform:uppercase'>IPS Score</div>
        <div style='font-size:36px;font-weight:600;color:#f0ede8;line-height:1.2'>{scores["ips"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Step 4 — Render dashboard
    dashboard_html = build_dashboard(df, scores, len(df))
    components.html(dashboard_html, height=2800, scrolling=True)

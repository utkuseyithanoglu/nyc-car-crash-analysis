from __future__ import annotations

import os
import re
import json
import pandas as pd
import streamlit as st

try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass


def detect_language(text: str) -> str:
    text = text.lower().strip()

    turkish_keywords = [
        "ve", "mi", "mu", "mı", "nasıl", "neden", "kaç", "saat",
        "tahmin", "karşılaştır", "yaralanma", "ilçe", "bölge", "merhaba", "selam"
    ]

    english_keywords = [
        "forecast", "compare", "injury", "risk", "hour", "hello", "hi", "borough"
    ]

    tr_score = sum(1 for word in turkish_keywords if word in text)
    en_score = sum(1 for word in english_keywords if word in text)

    if tr_score >= en_score and tr_score > 0:
        return "tr"
    return "en"


def translate_response(text: str, lang: str) -> str:
    if lang == "tr":
        return (
            text
            .replace("Forecast for next", "Önümüzdeki")
            .replace("hours", "saat için tahmin")
            .replace("Average predicted crashes", "Ortalama tahmini kaza")
            .replace("Peak predicted crashes", "En yüksek tahmini kaza")
            .replace("Lowest predicted crashes", "En düşük tahmini kaza")
            .replace("First values", "İlk değerler")
            .replace("Borough Comparison", "Bölge Karşılaştırması")
            .replace("Highest injury rate", "En yüksek yaralanma oranı")
            .replace("Lowest injury rate", "En düşük yaralanma oranı")
            .replace("Most crashes", "En fazla kaza")
            .replace("Prediction", "Tahmin")
            .replace("High injury risk", "Yüksek risk")
            .replace("Lower injury risk", "Düşük risk")
            .replace("I can help with", "Şunlarda yardımcı olabilirim")
        )
    return text


def is_greeting(msg: str) -> bool:
    msg = msg.lower().strip()

    greeting_phrases = [
        "merhaba", "selam", "selamlar", "iyi akşamlar", "iyi geceler", "günaydın",
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
        "sa", "s.a", "selamün aleyküm"
    ]

    clean_msg = re.sub(r"[^\w\sçğıöşü]", " ", msg)
    clean_msg = re.sub(r"\s+", " ", clean_msg).strip()

    if clean_msg in greeting_phrases:
        return True

    tokens = clean_msg.split()
    if len(tokens) <= 2 and any(token in greeting_phrases for token in tokens):
        return True

    return False


def get_openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key), None
    except Exception as e:
        return None, e


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="NYC Car Crash Analysis + Chatbot",
    page_icon="🚨",
    layout="wide"
)

st.title("NYC Car Crash Analysis Dashboard")
st.markdown("Explore crash trends, injuries, borough hotspots, and risk patterns across New York City.")

if option_menu is not None:
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "AI Assistant"],
        icons=["bar-chart", "chat-dots"],
        orientation="horizontal",
        styles={
            "container": {
                "padding": "6px",
                "background-color": "#f1f3f6",
                "border-radius": "12px"
            },
            "icon": {"color": "#2563eb", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "font-weight": "600",
                "color": "#333",
                "border-radius": "10px",
                "margin": "0 4px",
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "white",
            },
        }
    )
else:
    selected = st.radio("Navigation", ["Dashboard", "AI Assistant"], horizontal=True)

# -----------------------------------
# HELPERS
# -----------------------------------
BOROUGHS = ["BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND"]


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_all_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data = {
        "main_df": safe_read_csv(os.path.join(base_dir, "clean_crash_data.csv")),
        "logreg_df": safe_read_csv(os.path.join(base_dir, "logreg_predictions_full.csv")),
        "rf_df": safe_read_csv(os.path.join(base_dir, "tableau_gridsearch_predictions.csv")),
        "sarima_df": safe_read_csv(os.path.join(base_dir, "tableau_sarima_final.csv")),
        "metadata_df": safe_read_csv(os.path.join(base_dir, "model_metadata.csv")),
    }
    return data


data = load_all_data()
main_df = data["main_df"]
sarima_df = data["sarima_df"]
metadata_df = data["metadata_df"]

_raw_logreg_df = data["logreg_df"]
_raw_rf_df = data["rf_df"]

# -----------------------------------
# DATA CLEANING
# -----------------------------------
if not main_df.empty:
    main_df.columns = [c.upper().strip() for c in main_df.columns]

    if "BOROUGH" in main_df.columns:
        main_df.loc[:, "BOROUGH"] = main_df["BOROUGH"].astype(str).str.strip().str.upper()
        main_df = main_df.loc[main_df["BOROUGH"].isin(BOROUGHS)].copy()

    if "NUMBER OF PERSONS INJURED" in main_df.columns:
        injured_col = pd.to_numeric(
            main_df["NUMBER OF PERSONS INJURED"],
            errors="coerce"
        ).fillna(0)

        main_df.loc[:, "NUMBER OF PERSONS INJURED"] = injured_col

# -----------------------------------
# BOROUGH STATS
# -----------------------------------
def build_borough_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required_cols = ["BOROUGH", "NUMBER OF PERSONS INJURED"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    borough_stats = (
        df.groupby("BOROUGH")
        .agg(
            total_crashes=("BOROUGH", "size"),
            total_injuries=("NUMBER OF PERSONS INJURED", "sum")
        )
        .reset_index()
    )
    borough_stats["injury_rate"] = borough_stats["total_injuries"] / borough_stats["total_crashes"]
    borough_stats = borough_stats.sort_values("injury_rate", ascending=False).reset_index(drop=True)
    return borough_stats


borough_stats_df = build_borough_stats(main_df)


def get_borough_overview_text() -> str:
    if borough_stats_df.empty:
        return "I could not calculate borough statistics because the main dataset is missing or incomplete."

    highest = borough_stats_df.iloc[0]
    safest = borough_stats_df.sort_values("injury_rate", ascending=True).iloc[0]
    most_crashes = borough_stats_df.sort_values("total_crashes", ascending=False).iloc[0]

    return (
        f"**Historical Borough Overview**\n\n"
        f"- Highest injury rate: **{highest['BOROUGH'].title()}** "
        f"({highest['injury_rate']:.2%})\n"
        f"- Lowest injury rate: **{safest['BOROUGH'].title()}** "
        f"({safest['injury_rate']:.2%})\n"
        f"- Most crashes: **{most_crashes['BOROUGH'].title()}** "
        f"({int(most_crashes['total_crashes']):,})"
    )


def compare_boroughs_text(boroughs: list[str]) -> str:
    if borough_stats_df.empty:
        return "I could not compare boroughs because the historical dataset is missing."

    wanted = [b.strip().upper() for b in boroughs if b.strip().upper() in BOROUGHS]
    subset = borough_stats_df[borough_stats_df["BOROUGH"].isin(wanted)]

    if subset.empty:
        return "I could not find valid borough names. Try Brooklyn, Queens, Manhattan, Bronx, or Staten Island."

    lines = ["**Borough Comparison**\n"]
    for _, row in subset.iterrows():
        lines.append(
            f"- **{row['BOROUGH'].title()}**: "
            f"{int(row['total_crashes']):,} crashes, "
            f"{int(row['total_injuries']):,} injuries, "
            f"injury rate **{row['injury_rate']:.2%}**"
        )
    return "\n".join(lines)

# -----------------------------------
# FORECAST HELPERS
# -----------------------------------
def detect_forecast_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    candidates = [
        "forecast", "predicted", "prediction", "predicted_calls",
        "forecasted_calls", "sarima_forecast", "value", "incident_count"
    ]
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def forecast_text(steps: int) -> str:
    if sarima_df.empty:
        return "Forecast data is missing. Please add `tableau_sarima_final.csv`."

    forecast_col = detect_forecast_column(sarima_df)
    if not forecast_col:
        return (
            "I found the SARIMA file, but I could not detect a forecast column. "
            "Expected something like `predicted_calls` or `forecast`."
        )

    steps = max(1, min(int(steps), 168))
    values = sarima_df[forecast_col].dropna().tail(steps).tolist()

    if not values:
        return "I could not build the forecast output because the forecast column is empty."

    rounded = [round(float(v), 2) for v in values[:20]]
    avg_val = sum(values) / len(values)
    max_val = max(values)
    min_val = min(values)

    return (
        f"**Forecast for next {steps} hours**\n\n"
        f"- Average predicted crashes: **{avg_val:.2f}**\n"
        f"- Peak predicted crashes: **{max_val:.2f}**\n"
        f"- Lowest predicted crashes: **{min_val:.2f}**\n\n"
        f"First values: `{rounded}`"
    )

# -----------------------------------
# RISK PREDICTION HELPERS
# -----------------------------------
def normalize_lookup_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    copy_df = df.copy()
    copy_df.columns = [c.lower().strip() for c in copy_df.columns]

    if "borough" in copy_df.columns:
        copy_df["borough"] = copy_df["borough"].astype(str).str.strip().str.upper()

    if "borough_name" in copy_df.columns:
        copy_df["borough_name"] = copy_df["borough_name"].astype(str).str.strip().str.upper()

        if "borough" not in copy_df.columns:
            copy_df["borough"] = copy_df["borough_name"]
        else:
            copy_df["borough"] = copy_df["borough"].replace(["", "NAN", "NONE"], pd.NA)
            copy_df["borough"] = copy_df["borough"].fillna(copy_df["borough_name"])

    if "hour" in copy_df.columns:
        copy_df["hour"] = pd.to_numeric(copy_df["hour"], errors="coerce")

    if "day_of_week" in copy_df.columns:
        copy_df["day_of_week"] = pd.to_numeric(copy_df["day_of_week"], errors="coerce")

    if "gs_pred_class" in copy_df.columns and "predicted" not in copy_df.columns:
        copy_df["predicted"] = copy_df["gs_pred_class"]

    if "gs_pred_prob" in copy_df.columns and "probability" not in copy_df.columns:
        copy_df["probability"] = copy_df["gs_pred_prob"]

    if {"borough", "hour", "day_of_week"}.issubset(copy_df.columns):
        copy_df = copy_df.dropna(subset=["borough", "hour", "day_of_week"]).copy()
        copy_df["borough"] = copy_df["borough"].astype(str).str.strip().str.upper()
        copy_df["hour"] = copy_df["hour"].astype(int)
        copy_df["day_of_week"] = copy_df["day_of_week"].astype(int)

    return copy_df


logreg_df = normalize_lookup_df(_raw_logreg_df)
rf_df = normalize_lookup_df(_raw_rf_df)


def lookup_prediction(df: pd.DataFrame, borough: str, hour: int, day_of_week: int):
    if df.empty:
        return None

    needed = {"borough", "hour", "day_of_week"}
    if not needed.issubset(set(df.columns)):
        return None

    borough_upper = str(borough).strip().upper()
    hour_int = int(hour)
    dow_int = int(day_of_week)

    exact = df[
        (df["borough"] == borough_upper) &
        (df["hour"] == hour_int) &
        (df["day_of_week"] == dow_int)
    ]
    if not exact.empty:
        row = exact.iloc[0].to_dict()
        row["_match_type"] = "exact"
        return row

    same_hour = df[
        (df["borough"] == borough_upper) &
        (df["hour"] == hour_int)
    ]
    if not same_hour.empty:
        row = same_hour.iloc[0].to_dict()
        row["_match_type"] = "same_hour"
        return row

    same_borough = df[df["borough"] == borough_upper].copy()
    if not same_borough.empty:
        same_borough["_dist"] = (
            (same_borough["hour"] - hour_int).abs() +
            (same_borough["day_of_week"] - dow_int).abs()
        )
        row = same_borough.sort_values("_dist").iloc[0].to_dict()
        row["_match_type"] = "closest"
        return row

    return None


def format_prediction_result(row: dict, model_name: str) -> str:
    prediction = row.get("predicted", row.get("prediction", None))
    probability = row.get("probability", row.get("predicted_probability", None))
    borough = str(row.get("borough", "")).title()
    hour = row.get("hour", "")
    day = row.get("day_of_week", "")
    match_type = row.get("_match_type", "exact")

    risk_label = "High injury risk" if str(prediction) in ["1", "1.0"] else "Lower injury risk"

    match_note = ""
    if match_type == "same_hour":
        match_note = "\n- Note: **Exact day was not found, so I used the same borough and hour.**"
    elif match_type == "closest":
        match_note = "\n- Note: **Exact match was not found, so I used the closest available record for this borough.**"

    text = [
        f"**{model_name} Prediction**",
        "",
        f"- Borough: **{borough}**",
        f"- Hour: **{hour}**",
        f"- Day of week: **{day}**",
        f"- Predicted class: **{prediction}**",
        f"- Interpretation: **{risk_label}**",
    ]

    if probability is not None and str(probability) != "nan":
        try:
            text.append(f"- Probability: **{float(probability):.2%}**")
        except Exception:
            text.append(f"- Probability: **{probability}**")

    if match_note:
        text.append(match_note)

    return "\n".join(text)


def risk_prediction_text(user_text: str) -> str:
    msg = user_text.lower().strip()

    borough = None
    for b in BOROUGHS:
        if b.lower() in msg:
            borough = b
            break

    hour_match = re.search(r"(?:hour|hours|saat)\s*=?\s*(\d{1,2})", msg)
    if hour_match:
        hour = int(hour_match.group(1))
    else:
        fallback = re.search(r"(?<!\d)([01]?\d|2[0-3])(?!\d)", msg)
        hour = int(fallback.group(1)) if fallback else None

    day_match = re.search(r"(?:day|gün)\s*=?\s*([0-6])", msg)
    day_of_week = int(day_match.group(1)) if day_match else 0

    if hour is not None and not (0 <= hour <= 23):
        hour = None

    if not borough or hour is None:
        return (
            "To predict injury risk, please provide:\n"
            "- **Borough** (Brooklyn, Queens, Manhattan, Bronx, Staten Island)\n"
            "- **Hour** (0–23)\n"
            "- **Day of week** (0=Monday … 6=Sunday) — optional, defaults to Monday\n\n"
            "Example: `Predict injury risk in Brooklyn at hour 18 day=4`"
        )

    row = lookup_prediction(logreg_df, borough, hour, day_of_week)
    if row:
        return format_prediction_result(row, "Logistic Regression")

    row = lookup_prediction(rf_df, borough, hour, day_of_week)
    if row:
        return format_prediction_result(row, "GridSearch / Random Forest")

    if logreg_df.empty and rf_df.empty:
        return (
            "⚠️ Prediction CSV files (`logreg_predictions_full.csv` and "
            "`tableau_gridsearch_predictions.csv`) are missing. "
            "Please make sure they are in the same directory as this app."
        )

    return (
        f"⚠️ No prediction row found for **{borough.title()}** in the loaded CSV files.\n\n"
        "This usually means the borough name in the CSV does not match. "
        f"Available boroughs in the prediction data: "
        f"{', '.join(sorted(logreg_df['borough'].unique())) if not logreg_df.empty else 'unknown'}."
    )

# -----------------------------------
# SIMPLE CHAT ROUTER
# -----------------------------------
def generate_response(user_message: str) -> str:
    lang = detect_language(user_message)
    msg = user_message.lower().strip()

    if is_greeting(msg):
        if lang == "tr":
            return (
                "Merhaba! NYC car crash verileriyle ilgili yardımcı olabilirim.\n\n"
                "Örnek sorular:\n"
                "- `24 saat tahmin yap`\n"
                "- `Brooklyn ve Queens'i karşılaştır`\n"
                "- `En yüksek yaralanma oranı hangi borough'da?`\n"
                "- `Brooklyn için saat 18 day=4 risk tahmini yap`"
            )
        return (
            "Hello! I can help you with NYC car crash data.\n\n"
            "Example questions:\n"
            "- `Forecast 24 hours`\n"
            "- `Compare Brooklyn and Queens`\n"
            "- `Which borough has the highest injury rate?`\n"
            "- `Predict injury risk in Brooklyn at hour 18 day=4`"
        )

    if any(word in msg for word in ["forecast", "predict future", "gelecek", "tahmin", "next"]):
        step_match = re.search(r"\b(\d{1,3})\s*(hour|hours|saat)\b", msg)
        if step_match:
            steps = int(step_match.group(1))
            response = forecast_text(steps)
            return translate_response(response, lang)

        if lang == "tr":
            return "Kaç saatlik tahmin istediğini yaz. Örnek: `24 saat`"
        return "Please tell me how many hours to forecast. Example: `24 hours`"

    if "compare" in msg or "karşılaştır" in msg:
        found = [b for b in BOROUGHS if b.lower() in msg]
        if len(found) >= 2:
            response = compare_boroughs_text(found)
            return translate_response(response, lang)

        if lang == "tr":
            return "Lütfen en az iki borough yaz. Örnek: `Brooklyn ve Queens'i karşılaştır`"
        return "Please give at least two boroughs to compare. Example: `Compare Brooklyn and Queens`"

    if any(word in msg for word in ["borough", "safest", "highest injury", "most crashes", "overview"]):
        response = get_borough_overview_text()
        return translate_response(response, lang)

    if any(word in msg for word in ["risk", "injury risk", "predict injury", "classification"]):
        response = risk_prediction_text(user_message)
        return translate_response(response, lang)

    if lang == "tr":
        return (
            "Soruyu tam anlayamadım ama şu konularda yardımcı olabilirim:\n\n"
            "- `24 saat tahmin yap`\n"
            "- `Brooklyn ve Queens'i karşılaştır`\n"
            "- `En yüksek yaralanma oranı hangi borough'da?`\n"
            "- `Brooklyn için saat 18 day=4 risk tahmini yap`"
        )

    return (
        "I could not fully understand the question, but I can help with:\n\n"
        "- `Forecast 24 hours`\n"
        "- `Compare Brooklyn and Queens`\n"
        "- `Which borough has the highest injury rate?`\n"
        "- `Predict injury risk in Brooklyn at hour 18 day=4`"
    )

# -----------------------------------
# DASHBOARD
# -----------------------------------
if selected == "Dashboard":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crashes", "1,554,600")
    col2.metric("Avg Injury Rate", "31.88%")
    col3.metric("Most Risky Borough", "Brooklyn")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h1 style="color: white; text-align: center; margin-bottom: 10px; font-size: 38px;">
            🚗 OVERVIEW
        </h1>
        <p style="color: #e5e7eb; text-align: center; font-size: 18px; margin-bottom: 0;">
            This dashboard explores crash trends, injury risk, borough hotspots,
            and prediction outputs across New York City.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h1 style="color: white; text-align: center; margin-bottom: 10px; font-size: 38px;">
            🚨 PURPOSE OF THE PROJECT
        </h1>
        <p style="color: #e5e7eb; text-align: center; font-size: 18px; margin-bottom: 0;">
            This project explores NYC crash data to uncover key trends and risk patterns.
            It uses data analysis and visualization to support safer urban planning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: center; font-size:17px; color:#888;'>
    Data source:
    <a href='https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data' target='_blank'>
    Motor Vehicle Collisions – Crashes (NYC Open Data)
    </a>
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🌎 NYC Traffic Crash Distribution by Borough")
    hour_url = "https://public.tableau.com/views/NYCTrafficcrashdisturabitionbyborough/Sheet1?:embed=true&:showVizHome=no"
    st.components.v1.iframe(hour_url, height=800, scrolling=True)
    st.markdown("**Insight:** Crash counts are highest in Brooklyn and Queens.")

    st.divider()

    st.markdown("### 🆚 Actual vs Predicted Crash Trend")
    sarima_url = "https://public.tableau.com/views/nyccarcrashsarima/Dashboard1?:embed=true&:showVizHome=no"
    st.components.v1.iframe(sarima_url, height=800, scrolling=True)
    st.markdown("**Insight:** Crash patterns show repeated daily movement over time.")

    st.divider()

    st.markdown("### 🆚 Predicted vs Actual Injury Rate by Borough")
    injury_url = "https://public.tableau.com/views/modelperformancepredictvsactualinjuryratebyborough/Dashboard2?:showVizHome=no"
    st.components.v1.iframe(injury_url, height=800, scrolling=True)
    st.markdown("**Insight:** Predicted injury rates are close to actual values in most boroughs.")

# -----------------------------------
# AI ASSISTANT
# -----------------------------------
if selected == "AI Assistant":
    st.title("🛞 NYC Car Crash Assistant")
    st.caption("Powered by GPT + CSV-based crash analytics | Built by Utku Seyithanoğlu")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    def tool_get_borough_overview():
        return get_borough_overview_text()

    def tool_compare_boroughs(boroughs: list[str]):
        return compare_boroughs_text(boroughs)

    def tool_forecast_hours(hours: int):
        return forecast_text(hours)

    def tool_predict_injury_risk(borough: str, hour: int, day_of_week: int):
        borough_upper = str(borough).strip().upper()

        if borough_upper not in BOROUGHS:
            return (
                f"Unknown borough: '{borough}'. "
                f"Valid options are: {', '.join(b.title() for b in BOROUGHS)}."
            )
        if not (0 <= int(hour) <= 23):
            return "Invalid hour. Please use a value between 0 and 23."
        if not (0 <= int(day_of_week) <= 6):
            return "Invalid day_of_week. Use 0 (Monday) through 6 (Sunday)."

        row = lookup_prediction(logreg_df, borough_upper, int(hour), int(day_of_week))
        if row:
            return format_prediction_result(row, "Logistic Regression")

        row = lookup_prediction(rf_df, borough_upper, int(hour), int(day_of_week))
        if row:
            return format_prediction_result(row, "GridSearch / Random Forest")

        available_lr = sorted(logreg_df["borough"].unique().tolist()) if not logreg_df.empty else []
        available_rf = sorted(rf_df["borough"].unique().tolist()) if not rf_df.empty else []

        if not available_lr and not available_rf:
            return (
                "⚠️ Prediction CSV files are missing from the server. "
                "Please ensure `logreg_predictions_full.csv` and "
                "`tableau_gridsearch_predictions.csv` are in the app directory."
            )

        return (
            f"No prediction data found for **{borough.title()}** "
            f"(hour={hour}, day={day_of_week}).\n\n"
            f"Boroughs available in Logistic Regression data: {available_lr or 'none'}\n"
            f"Boroughs available in Random Forest data: {available_rf or 'none'}"
        )

    system_prompt = f"""
You are an expert data analyst assistant specializing in NYC car crash analysis.
Your name is "NYC Crash Assistant". You were built by Utku Seyithanoğlu.

════════════════════════════════════════
LANGUAGE RULES  (CRITICAL — always follow)
════════════════════════════════════════
- Detect the language of the user's message.
- If Turkish → reply entirely in Turkish.
- If English → reply entirely in English.
- Never mix languages in a single reply.
- Translate technical terms naturally (e.g. "injury rate" → "yaralanma oranı").

════════════════════════════════════════
SCOPE
════════════════════════════════════════
You ONLY answer questions about:
  • NYC borough-level crash & injury statistics
  • Injury risk prediction (by borough, hour, day)
  • Crash forecasts (SARIMA-based time series)
  • Comparison of boroughs
  • General project/model info

If the user asks anything outside this scope, politely explain (in their language)
that you can only help with NYC car crash analytics.

════════════════════════════════════════
TOOL USAGE  (mandatory — never invent numbers)
════════════════════════════════════════
Always call the appropriate tool before answering numeric questions.
Never make up statistics, predictions, or forecast values.

• Borough summary / safest / most crashes  →  tool_get_borough_overview
• Compare two or more boroughs             →  tool_compare_boroughs
• Forecast N hours                         →  tool_forecast_hours
• Injury risk for borough+hour+day         →  tool_predict_injury_risk

If a required parameter is missing (e.g. hour for risk prediction),
ask the user for it in their language before calling the tool.

If the tool returns a result saying CSV files are missing or no data was found,
relay that diagnostic WORD FOR WORD to the user. Do NOT summarize, hide, or rephrase it.
NEVER suggest alternative hours/days or list available combinations — you do not have that information.
NEVER say "the data might be available for a different combination" — that is misleading.

════════════════════════════════════════
RESPONSE STYLE
════════════════════════════════════════
- Be concise, friendly, and natural — not robotic.
- Use bullet points and bold for key numbers to improve readability.
- For risk predictions: explain what "High risk" and "Low risk" mean in plain language.
- For forecasts: always report average, peak, and lowest values.
- For borough comparisons: present data in a clear side-by-side style.
- If the tool result includes a fallback note (closest match / same hour),
  mention it briefly so the user knows the result is approximate.
- If a greeting is detected, welcome the user warmly and suggest 3-4 example questions.

════════════════════════════════════════
PROJECT CONTEXT
════════════════════════════════════════
Dataset    : NYC Motor Vehicle Collisions (NYC Open Data, 2012–2023)
Boroughs   : Brooklyn, Queens, Manhattan, Bronx, Staten Island
Models used: Logistic Regression, GridSearch / Random Forest, SARIMA

Current borough summary:
{get_borough_overview_text()}
"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_get_borough_overview",
                "description": "Returns the borough-level crash and injury overview.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_compare_boroughs",
                "description": "Compares crash and injury statistics across boroughs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "boroughs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of borough names"
                        }
                    },
                    "required": ["boroughs"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_forecast_hours",
                "description": "Returns crash forecast summary for the requested number of hours.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "Number of hours for forecast"
                        }
                    },
                    "required": ["hours"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_predict_injury_risk",
                "description": "Predicts injury risk for a borough, hour, and day_of_week.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "borough": {
                            "type": "string",
                            "description": "Borough name"
                        },
                        "hour": {
                            "type": "integer",
                            "description": "Hour from 0 to 23"
                        },
                        "day_of_week": {
                            "type": "integer",
                            "description": "Day of week where 0=Monday and 6=Sunday"
                        }
                    },
                    "required": ["borough", "hour", "day_of_week"]
                }
            }
        }
    ]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I can help you with NYC car crash analysis.\n\n"
                    "Example questions:\n"
                    "- Forecast 24 hours\n"
                    "- Compare Brooklyn and Queens\n"
                    "- Which borough has the highest injury rate?\n"
                )
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about NYC crashes...")

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        if not api_key:
            answer = generate_response(user_prompt)
        else:
            client, client_error = get_openai_client(api_key)

            if client is None:
                st.warning(f"OpenAI import/client error: {client_error} — falling back to local router.")
                answer = generate_response(user_prompt)
            else:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.chat_messages,
                        ],
                        tools=tools,
                        tool_choice="auto"
                    )

                    message = response.choices[0].message

                    if message.tool_calls:
                        tool_messages = []

                        for tool_call in message.tool_calls:
                            fn_name = tool_call.function.name
                            args = json.loads(tool_call.function.arguments or "{}")

                            if fn_name == "tool_get_borough_overview":
                                result = tool_get_borough_overview()
                            elif fn_name == "tool_compare_boroughs":
                                result = tool_compare_boroughs(args.get("boroughs", []))
                            elif fn_name == "tool_forecast_hours":
                                result = tool_forecast_hours(args.get("hours", 24))
                            elif fn_name == "tool_predict_injury_risk":
                                result = tool_predict_injury_risk(
                                    args.get("borough", ""),
                                    args.get("hour", 0),
                                    args.get("day_of_week", 0),
                                )
                                if (
                                    result.startswith("⚠️")
                                    or result.startswith("No prediction")
                                    or result.startswith("Unknown borough")
                                ):
                                    answer = result
                                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                                    with st.chat_message("assistant"):
                                        st.markdown(answer)
                                    st.stop()
                            else:
                                result = "Tool not found."

                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            })

                        follow_up = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                *st.session_state.chat_messages,
                                {
                                    "role": "assistant",
                                    "content": message.content or "",
                                    "tool_calls": message.tool_calls,
                                },
                                *tool_messages,
                            ],
                        )
                        answer = follow_up.choices[0].message.content
                    else:
                        answer = message.content
                except Exception as e:
                    st.warning(f"GPT API error: {e} — falling back to local router.")
                    answer = generate_response(user_prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
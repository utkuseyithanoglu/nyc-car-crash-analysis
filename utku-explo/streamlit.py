import os
import pickle 
import json
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from streamlit_option_menu import option_menu

st.title("NYC Car Crash Analysis Dashboard")
st.markdown("Explore crash trends, injuries, borough hotspots, and risk patterns across New York City.")

st.set_page_config(page_title="NYC CAR Crash Analaysis + Chatbot", page_icon="🚨", layout="wide")
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "AI ASSISTANT"],
    icons=["map", "chat"],
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
if selected == "Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Crashes", "1,554,600")
    col2.metric("Avg Injury Rate", "31.88%")
    col3.metric("Most Risky Borough", "Brooklyn")
    
    
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1f2937, #374151);
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <h1 style="
                color: white;
                text-align: center;
                margin-bottom: 10px;
                font-size: 42px;
            ">
                🚗 OVERVIEW
            </h1>
            <p style="
                color: #e5e7eb;
                text-align: center;
                font-size: 18px;
                margin-bottom: 0;
            ">
                 This dashboard explores crash trends, injury risk, EMS demand patterns, and borough-level hotspots across New York City.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1f2937, #374151);
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <h1 style="
                color: white;
                text-align: center;
                margin-bottom: 10px;
                font-size: 42px;
            ">
                🚨 purpose of the project
            </h1>
            <p style="
                color: #e5e7eb;
                text-align: center;
                font-size: 18px;
                margin-bottom: 0;
            ">
                This project explores NYC crash data to uncover key trends and risk patterns.
It uses data analysis and visualization to provide actionable insights for safer urban planning
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
    st.markdown("""
    **Insight:** Crash risk is highest in Brooklyn and Queens        
    """)
    st.divider()

    st.markdown("### 🆚 Actual vs Predicted EMS Calls Daily Aggregation")

    sarima_url = "https://public.tableau.com/views/nyccarcrashsarima/Dashboard1?:embed=true&:showVizHome=no"

    st.components.v1.iframe(sarima_url, height=800, scrolling=True)
    st.markdown("""
    **Insight:** There are similar crash patterns every day.        
    """)

    st.divider()

    st.markdown("### 🆚 Predicted vs Actual injury rate by borough")

    sarima_url = "https://public.tableau.com/views/modelperformancepredictvsactualinjuryratebyborough/Dashboard2?:showVizHome=no"

    st.components.v1.iframe(sarima_url, height=800, scrolling=True)
    st.markdown("**Insight:** The predicted injury rate is close to the actual rate in most boroughs.")

if selected == "AI ASSISTANT":
    st.title("🛞 NYC CAR CRASH Assistant")
    st.caption("Powered by Logistic Regression + GridSearch + SARIMAX | NYC Crash Data | Built by Utku Seyithanoğlu")
    st.caption("Thanks to Ayman Tabidi and Sarah Oasier for their support.")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # ----------------------------
    # LOAD MODELS / DATA
    # ----------------------------
    @st.cache_resource
    def load_all_models():
        base_dir = os.path.dirname(os.path.abspath(__file__))

        logreg_path = os.path.join(base_dir, "logreg_model.pkl")
        grid_path = os.path.join(base_dir, "gridsearch_crash_model.pkl")
        sarimax_path = os.path.join(base_dir, "sarimax_model.pkl")
        features_path = os.path.join(base_dir, "model_features.pkl")
        data_path = os.path.join(base_dir, "clean_crash_data.pkl")

        with open(logreg_path, "rb") as f:
            logreg_bundle = pickle.load(f)

        with open(grid_path, "rb") as f:
            gridsearch_model = pickle.load(f)

        with open(sarimax_path, "rb") as f:
            sarimax_model = pickle.load(f)

        with open(features_path, "rb") as f:
            model_features = pickle.load(f)

        with open(data_path, "rb") as f:
            main_df = pickle.load(f)

        return logreg_bundle, gridsearch_model, sarimax_model, model_features, main_df

    try:
        logreg_bundle, gridsearch_model, sarimax_model, model_features, main_df = load_all_models()
    except Exception as e:
        st.error(f"Model/data loading error: {e}")
        st.stop()

    # ----------------------------
    # PREP LOGREG MODEL / THRESHOLD
    # ----------------------------
    if isinstance(logreg_bundle, dict):
        logreg_model = logreg_bundle.get("model", logreg_bundle)
        logreg_threshold = float(logreg_bundle.get("threshold", 0.5))
        if "features" in logreg_bundle and logreg_bundle["features"]:
            model_features = logreg_bundle["features"]
    else:
        logreg_model = logreg_bundle
        logreg_threshold = 0.5

    # ----------------------------
    # CLEAN DATA
    # ----------------------------
    # Normalize column names
    main_df.columns = [col.upper() for col in main_df.columns]

    if "BOROUGH" in main_df.columns:
        main_df["BOROUGH"] = main_df["BOROUGH"].astype(str).str.strip().str.upper()
        main_df = main_df[main_df["BOROUGH"] != "UNKNOWN"]

    if "NUMBER OF PERSONS INJURED" in main_df.columns:
        main_df["NUMBER OF PERSONS INJURED"] = pd.to_numeric(
            main_df["NUMBER OF PERSONS INJURED"], errors="coerce"
        ).fillna(0)

    # ----------------------------
    # BOROUGH STATS
    # ----------------------------
    if "BOROUGH" in main_df.columns and "NUMBER OF PERSONS INJURED" in main_df.columns:
        borough_stats_df = (
            main_df.groupby("BOROUGH")
            .agg(
                total_crashes=("BOROUGH", "size"),
                total_injuries=("NUMBER OF PERSONS INJURED", "sum")
            )
            .reset_index()
        )
        borough_stats_df["injury_rate"] = (
            borough_stats_df["total_injuries"] / borough_stats_df["total_crashes"]
        )
        borough_stats_df = borough_stats_df.sort_values("injury_rate", ascending=False)
    else:
        borough_stats_df = pd.DataFrame(columns=["BOROUGH", "total_crashes", "total_injuries", "injury_rate"])

    def get_borough_statistics(comparison_boroughs=None):
        df = borough_stats_df.copy()

        if df.empty:
            return {
                "type": "error",
                "message": "Borough statistics could not be created from the dataset."
            }

        if comparison_boroughs:
            wanted = [b.strip().upper() for b in comparison_boroughs]
            df = df[df["BOROUGH"].isin(wanted)]

            return {
                "type": "borough_comparison",
                "borough_stats": df.to_dict(orient="records")
            }

        highest = df.iloc[0]
        safest = df.sort_values("injury_rate", ascending=True).iloc[0]
        most_crashes = df.sort_values("total_crashes", ascending=False).iloc[0]

        return {
            "type": "borough_overview",
            "highest_injury_rate_borough": highest.to_dict(),
            "safest_borough": safest.to_dict(),
            "most_crashes_borough": most_crashes.to_dict(),
            "borough_stats": df.to_dict(orient="records")
        }

    # ----------------------------
    # MODEL INPUT PREP
    # ----------------------------
    def prepare_input(borough, hour, day_of_week):
        borough = str(borough).strip().lower()

        input_df = pd.DataFrame([{
            "borough": borough,
            "hour": int(hour),
            "day_of_week": int(day_of_week)
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        return input_df

    def predict_with_logreg(borough, hour, day_of_week):
        X = prepare_input(borough, hour, day_of_week)

        proba = logreg_model.predict_proba(X)[0][1]
        pred = 1 if proba >= logreg_threshold else 0

        return {
            "model": "logreg",
            "prediction": int(pred),
            "probability": float(proba),
            "threshold": float(logreg_threshold)
        }

    def predict_with_gridsearch(borough, hour, day_of_week):
        X = prepare_input(borough, hour, day_of_week)
        pred = gridsearch_model.predict(X)[0]

        result = {
            "model": "gridsearch",
            "prediction": int(pred)
        }

        if hasattr(gridsearch_model, "predict_proba"):
            result["probabilities"] = gridsearch_model.predict_proba(X)[0].tolist()

        return result

    def forecast_with_sarimax(steps=24):
        steps = int(steps)
        forecast = sarimax_model.forecast(steps=steps)

        if hasattr(forecast, "tolist"):
            forecast_values = forecast.tolist()
        else:
            forecast_values = list(forecast)

        return {
            "model": "sarimax",
            "steps": steps,
            "forecast": forecast_values
        }

    analysis_summary = {
        "project": "NYC crash analysis",
        "focus": "crash patterns, borough comparisons, injury prediction",
        "models": ["logistic regression", "grid search model", "sarimax"]
    }

    # ----------------------------
    # TOOL DEFINITIONS
    # ----------------------------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "predict_with_logreg",
                "description": "Predict injury risk for a given NYC crash scenario using logistic regression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "borough": {
                            "type": "string",
                            "description": "NYC borough name (e.g., Brooklyn, Queens, Manhattan, Bronx, Staten Island)"
                        },
                        "hour": {
                            "type": "integer",
                            "description": "Hour of day (0-23)"
                        },
                        "day_of_week": {
                            "type": "integer",
                            "description": "Day of week (0=Monday, 6=Sunday)"
                        }
                    },
                    "required": ["borough", "hour", "day_of_week"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "predict_with_gridsearch",
                "description": "Predict injury risk using the optimized classification model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "borough": {
                            "type": "string",
                            "description": "NYC borough name"
                        },
                        "hour": {
                            "type": "integer",
                            "description": "Hour of day (0-23)"
                        },
                        "day_of_week": {
                            "type": "integer",
                            "description": "Day of week (0-6)"
                        }
                    },
                    "required": ["borough", "hour", "day_of_week"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "forecast_with_sarimax",
                "description": "Forecast future NYC crash counts using SARIMAX.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer",
                            "description": "Number of future hours to forecast"
                        }
                    },
                    "required": ["steps"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_analysis_summary",
                "description": "Return a summary of the NYC crash analysis project.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_borough_statistics",
                "description": "Return historical borough-level crash statistics and injury rates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "comparison_boroughs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of borough names to compare"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    ]

    system_prompt = """
You are an AI assistant specialized in NYC motor vehicle crash analysis.

IMPORTANT:
You only know what is available through the provided tools.
You MUST use tools when needed.
Do not pretend to retrieve or calculate data unless you are calling a tool.

LANGUAGE RULE:
- If the user writes in Turkish, respond in Turkish.
- If the user writes in English, respond in English.

AVAILABLE TOOLS:
1. predict_with_logreg
2. predict_with_gridsearch
3. forecast_with_sarimax
4. get_analysis_summary
5. get_borough_statistics

TASK RULES:
- Historical questions about borough rankings, safest borough, highest injury rate, comparisons:
  ALWAYS call get_borough_statistics
- Scenario-based prediction:
  use predict_with_logreg or predict_with_gridsearch
- Forecast questions:
  use forecast_with_sarimax
- Project/model questions:
  use get_analysis_summary

CRITICAL RULES:
- NEVER invent numbers
- NEVER guess borough rankings
- ALWAYS use a tool when data is needed
- Clearly distinguish historical data from model predictions

STYLE:
- Clear
- Short but informative
- Natural explanation, not raw JSON
"""

    # ----------------------------
    # SESSION STATE
    # ----------------------------
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me about borough crash statistics, injury risk, or future crash forecasts."
            }
        ]

    # ----------------------------
    # CHAT FUNCTION
    # ----------------------------
    def chat(user_message):
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.conversation_history,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )

            message = response.choices[0].message

            if getattr(message, "tool_calls", None):
                st.session_state.conversation_history.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    try:
                        if fn_name == "predict_with_logreg":
                            result = predict_with_logreg(**args)

                        elif fn_name == "predict_with_gridsearch":
                            result = predict_with_gridsearch(**args)

                        elif fn_name == "forecast_with_sarimax":
                            result = forecast_with_sarimax(**args)

                        elif fn_name == "get_analysis_summary":
                            result = analysis_summary

                        elif fn_name == "get_borough_statistics":
                            result = get_borough_statistics(**args)

                        else:
                            result = {"error": f"Unknown function: {fn_name}"}

                    except Exception as e:
                        result = {"error": str(e)}

                    st.session_state.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

                continue

            assistant_message = message.content if message.content else "I could not generate a response."

            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

    # ----------------------------
    # DISPLAY CHAT HISTORY
    # ----------------------------
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----------------------------
    # USER INPUT
    # ----------------------------
    user_prompt = st.chat_input("Ask about NYC crashes...")

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chat(user_prompt)
                except Exception as e:
                    answer = f"Error: {e}"

                st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
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
    options=["Dashboard", "Chatbot"],
    icons=["map", "chat"],
    orientation="horizontal"
)
if selected == "Dashboard":
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
                🚗 NYC Car Crash Analysis Dashboard
            </h1>
            <p style="
                color: #e5e7eb;
                text-align: center;
                font-size: 18px;
                margin-bottom: 0;
            ">
                Explore crash trends, injury patterns, borough hotspots, and risk insights across New York City.
            </p>
        </div>
    """, unsafe_allow_html=True)
    html_content = """
    <div style="
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 22px;
    border-left: 6px solid #ef4444;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    ">
    <h3 style="
        color:#f87171;
        margin-bottom: 16px;
        font-size: 22px;
    ">
        🎯 Project Objective
    </h3>

    <p style="
        font-size:16px;
        color:#e5e7eb;
        line-height:1.8;
        margin-bottom: 14px;
    ">
        NYC traffic collisions follow measurable patterns across
        <strong style="color:#fca5a5;">boroughs, hours, weekdays, and contributing factors</strong>.
        This dashboard reveals
        <strong style="color:#fca5a5;">where crashes happen most often</strong>,
        <strong style="color:#fca5a5;">when risk increases</strong>,
        and which conditions are associated with injuries.
    </p>

    <p style="
        font-size:16px;
        color:#e5e7eb;
        line-height:1.8;
        margin-bottom: 14px;
    ">
        By combining crash, location, and severity data, the project highlights
        <strong style="color:#fca5a5;">hotspots</strong>,
        <strong style="color:#fca5a5;">injury trends</strong>,
        and <strong style="color:#fca5a5;">factor-based risk patterns</strong>.
    </p>

    <p style="
        font-size:16px;
        color:#e5e7eb;
        line-height:1.8;
        margin-bottom: 0;
    ">
        The goal is to transform large-scale collision records into
        <strong style="color:#fca5a5;">clear, actionable insights</strong>
        for safer streets and smarter urban decisions.
    </p>
    </div>
        """
    st.markdown(html_content, unsafe_allow_html=True)

    st.markdown("""
            <p style="text-align:center; font-size:17px; color:#888;">
                Data source: 
                <a href="https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data" target="_blank">
                Motor Vehicle Collisions - Crashes (NYC Open Data)
                </a>
            </p>
        """, unsafe_allow_html=True)

    st.divider()

    
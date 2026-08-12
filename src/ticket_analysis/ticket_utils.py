import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import os
import json
from collections import Counter, defaultdict



CATEGORIES = {
    "Connectivity / Offline": [
        "offline", "ocpp", "scb offline", "not reachable", "connectivity",
        "cu not reachable", "cms", "onboarding", "modbus", "secc", "sec pilot",
        "can comm", "no scb connection", "coap", "communication issue",
        "preauthorize", "station status", "wan port", "lan cable",
        "network issue", "ip address",
    ],
    "Charging Failure": [
        "charging issue", "charging station issue", "unable to charge",
        "charger out of order", "out of order", "out of service",
        "charger not working", "charger unavailable", "alloutletsunavailable",
        "outlets out of order", "outlet out of order", "outlet 1", "outlet 2",
        "charger issue", "charger error", "charger malfunction", "charger ooo",
        "charger faulty", "charging interruption", "charging session",
        "charging not possible", "ghost session", "outlets unavailable",
        "outlet unavailable", "all outlets ooo", "charging point issue",
        "locked gun", "gun issue", "ev not charging", "session issue",
        "charger repair", "charging station repair", "not charging",
        "station not available", "station repair",
    ],
    "Display / HMI": [
        "screen", "display", "hmi", "blackscreen", "black screen", "scramble",
        "delamination", "detached", "qr", "screensaver", "stuck on",
        "blistering", "hmi issue", "hmi error", "blank screen",
    ],
    "Power / Converter": [
        "converter", "power converter", "power loss", "powerloss",
        "missing converter", "contactor", "mainbreaker", "spd", "fuse",
        "coils", "failsafe", "mainfan", "lem issue", "internal error",
        "hardware issue", "overheat", "fan error", "automation fan",
        "thermal", "overcurrent", "undervoltage", "overvoltage",
    ],
    "Cable / Connector": [
        "cable", "transit damage", "connector issue", "plug failure",
        "faulty plug", "cable theft", "cable temp", "cable damaged",
        "holster", "inlet",
    ],
    "Firmware / Software Update": [
        "firmware", "fw", "rollout", "configuration", "config",
        "screensaver update", "update screensaver", "downgrade",
        "commissioning", "software update", "sw update", "flash",
        "cfast", "reboot", "reset",
    ],
    "Log Investigation": [
        "log", "high volume of log", "download logs", "unable to download",
    ],
    "Payment Terminal": [
        "payment", "payment terminal", "pay", "card", "rfidcase"
    ],
}


def _match_keywords(text: str) -> str:
    scores = {
        cat: sum(kw in text for kw in keywords)
        for cat, keywords in CATEGORIES.items()
    }
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat]>0 else "Technical Query / Other"


def _extract_title_issue(title: str) -> str:
    if pd.isna(title):
        return ""
    match = re.search(r"\]\s*(.+)$", title)
    return (match.group(1).strip() if match else title.strip()).lower().strip(" -")


def _extract_html_issue(html: str) -> str:
    if pd.isna(html):
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    match = re.search(
        r"Reported behavi[uo]u?r[/\s]*issue[:\-\s]*(.+?)(?:Root cause|Solution|On DAMEX|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip().lower() if match else text.lower()


def classify_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_title_issue"] = df["Title"].apply(_extract_title_issue)
    df["category"] = df["_title_issue"].apply(_match_keywords)
    mask = df["category"] == "Technical Query / Other"
    return df


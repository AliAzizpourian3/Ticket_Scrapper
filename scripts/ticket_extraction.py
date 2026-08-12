
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import sys
sys.path.append("../src")
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json

import re
from pathlib import Path


# Common email thread separator patterns
THREAD_SEPARATORS = [
    r"(?i)-+\s*Original Message\s*-+",
    r"(?i)-+\s*Forwarded Message\s*-+",
    r"(?i)From:.*?Sent:.*?To:.*?Subject:",
    r"(?i)On\s+\w+\s+\d+.*?wrote:",
    r"(?i)Am\s+\d+\.\d+\.\d+.*?schrieb",  # German format
    r"(?i)Le\s+\d+\/\d+\/\d+.*?a écrit",  # French format
    r"_{10,}",  # Long underscores
    r"={10,}",  # Long equals signs
]

# Compile regex patterns
COMPILED_PATTERNS = [re.compile(pattern, re.DOTALL | re.MULTILINE) for pattern in THREAD_SEPARATORS]


def detect_email_thread_separator(text):
    """
    Detect if text contains email thread separators.
    Returns position of separator, or -1 if none found.
    """
    if not text or not isinstance(text, str):
        return -1

    earliest_pos = len(text)
    found = False

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            found = True
            if match.start() < earliest_pos:
                earliest_pos = match.start()

    return earliest_pos if found else -1


def trim_email_body(body):
    """
    Trim email body to remove quoted/forwarded content.
    Preserves only the new content written in this email.
    """
    if not body or not isinstance(body, str):
        return body

    # Find the earliest thread separator
    separator_pos = detect_email_thread_separator(body)

    if separator_pos > 0:
        # Keep only content before the separator
        trimmed = body[:separator_pos].strip()

        # If we trimmed more than 50% and left content is substantial, return it
        if len(trimmed) > 50:  # At least 50 chars of new content
            return trimmed
        elif len(trimmed) == 0:
            # If nothing left, return original (separator might be false positive)
            return body
        else:
            return trimmed

    return body


def is_email(entry):
    """
    Determine if an additional_info entry is an email.
    Emails have a non-empty 'extra' field with sender/to/directioncode.
    """
    if not isinstance(entry, dict):
        return False

    extra = entry.get("extra", {})

    # If extra has any fields (sender, to, directioncode), it's an email
    return bool(extra)


def process_ticket(ticket_data):
    """
    Process a single ticket, trimming email threads in additional_info.
    """
    if "additional_info" not in ticket_data or not isinstance(ticket_data["additional_info"], list):
        return ticket_data, 0, 0

    trimmed_count = 0
    total_saved_chars = 0

    for entry in ticket_data["additional_info"]:
        if is_email(entry):
            original_body = entry.get("body", "")
            trimmed_body = trim_email_body(original_body)

            if len(trimmed_body) < len(original_body):
                chars_saved = len(original_body) - len(trimmed_body)
                total_saved_chars += chars_saved
                trimmed_count += 1
                entry["body"] = trimmed_body

    return ticket_data, trimmed_count, total_saved_chars




def clean_html(html):
    """Strip HTML tags and clean up whitespace."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    # Remove excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def parse_textual_info(raw_texts):
    """
        For each text in timeline, which can be either an email or a note, we extract all the relevant information.
    """

    parsed = []
    for text in raw_texts:
        ids = None
        body= None
        extra = None
        if "annotationid" in text.keys():
            ids = "annotationid"
            body = "notetext"
            extra = []
        elif "activityid" in text.keys():
            ids = "activityid"
            body = "description"
            extra = [ "sender", "to", "directioncode"] # True = Outgoing, False = Incoming

        body = text.get(body, "")
        body = body if body is not None else ""

        if ids is None or body is None or extra is None:
            continue 

        parsed.append({
            "id" :  text.get(ids, ""),
            "subject": text.get("subject", ""),
            "body": clean_html(body.strip()),
            "date": (text.get("createdon@OData.Community.Display.V1.FormattedValue", text.get("createdon", ""))),
            "extra" : { key: text.get(key) for key in extra }
        })
    return parsed

def extract_ticket_info(ticket_id, headers, cookies, device_id, serial_number):
    """
    Extracts the information from the dynamo database.
    Headers and cookies must be provided.
        This representation came from going to the isp database, pressing F12, going to the Network tab and filtering form api/data/v9.2/GetOrgDbOrgSetting
        We then copy the value as cURL and paste it here: https://curlconverter.com/
        The cookies and headers are extracted from there.
    
    """
    response = requests.get(f"{BASE_URL}/incidents({ticket_id})", headers=headers, cookies=cookies)
    ticket_info = {}
    text = [ ]

    if response.status_code == 200:
        rs = response.json()
        # Title
        # Created On
        # Description?
        # ticketnumber
        # print(rs)
        description = rs.get("description", "")
        description = description if description is not None else ""
        ticket_info["incident_id"] = ticket_id
        ticket_info["ticket_number"] = rs.get("ticketnumber", "")
        ticket_info["title"] = rs.get("title", "")
        ticket_info["created_on"] = rs.get("createdon", "")
        ticket_info["last_activity"] = rs.get("modifiedon", "")
        ticket_info["age"] = rs.get("sie_ticketage", "")
        ticket_info["description"] = clean_html(description.strip())
        ticket_info["type"] = rs.get("casetypecode", "")
        ticket_info["priority"] = rs.get("prioritycode", "")
        ticket_info["stage"] = rs.get("sie_activestagename", "")
        ticket_info["serial_number"] = rs.get("sie_relatedsearchableassets", "")
        ticket_info["case_nature_code"] = rs.get("sie_casenaturecode", "")

        ticket_info["chargerID"] = rs.get("sie_chargerid", "")
        if ticket_info["chargerID"] is None:
            ticket_info["chargerID"] = device_id
        ticket_info["serial_number"] = rs.get("sie_relatedsearchableassets", "")
        if ticket_info["serial_number"] is None:
            ticket_info["serial_number"] = serial_number

    params = {
        "$filter": f"_objectid_value eq {ticket_id}",
        "$select": (
            "annotationid,"
            "subject,"
            "notetext,"
            "createdon,"
            # "filename,"        # If a file is attached
            # "filesize,"        # Size of attachment
            # "mimetype,"        # File type (e.g. pdf, png)
            # "isdocument,"      # True if file is attached
            "_createdby_value"
        )
    }


    response = requests.get(
        f"{BASE_URL}/annotations",
        headers=headers,
        cookies=cookies,
        params=params
    )
    text.extend(response.json().get("value", []))

    params = {
        "$filter": f"_regardingobjectid_value eq {ticket_id}",
        "$select": (
            "subject,"
            "description,"
            "createdon,"
            "directioncode,"       
            "statuscode,"
            "trackingtoken,"
            "sender,"
            "torecipients"
        )
    }
    response = requests.get(
        f"{BASE_URL}/emails",
        headers=headers,
        cookies=cookies,
        params=params
    )

    text.extend(response.json().get("value", []))   
    if len(text)!= 0:
        text = parse_textual_info(text)
        text = sorted(text, key=lambda x: datetime.fromisoformat(x["date"].replace("Z", "+00:00")) if x["date"] else datetime.min)
    ticket_info["additional_info"] = text
    return ticket_info


def extract_charger_tickets(charger, min_, max_, serial_number, headers, cookies):

    global INDEX

    min_ = pd.Timestamp(min_).strftime("%Y-%m-%dT%H:%M:%SZ")
    max_ = pd.Timestamp(max_).strftime("%Y-%m-%dT%H:%M:%SZ")

    date_filter = "(" +(
            # f"modifiedon ge {DATE_FROM} and modifiedon le {DATE_TO}"
            f"createdon ge {min_} and createdon le {max_}"
        ) + ")"

    charger_filter = f"(sie_chargerid eq '{charger}')"
    serial_number_filter = f"contains(sie_relatedsearchableassets, '{serial_number}')"

    url = (f"{BASE_URL}/incidents"
           "?$select=title,ticketnumber,statecode,statuscode,createdon,modifiedon"
           f"&$filter={GUID_FILTER} and ({charger_filter} or {serial_number_filter}) and {date_filter}"
    )

    processed_tickets = 0
    while url:
        response = requests.get(url, headers=headers, cookies=cookies)
        data = response.json()
        cases = data.get("value", [])
        url = data.get("@odata.nextLink", None) # Get next page
        for case in cases:
            incident_id = case.get("incidentid", "")
            ticket_file_name = f"{incident_id}.json"
            if os.path.exists(os.path.join(OUTPUT_DIR, ticket_file_name)):
                continue

            ticket_info = extract_ticket_info(incident_id, headers, cookies, charger, serial_number)
            with open(os.path.join(OUTPUT_DIR, ticket_file_name), "w") as out_file:
                json.dump(ticket_info, out_file, indent=4)

                
            with INDEX_LOCK:
                INDEX += 1
                
            processed_tickets += 1
            
    return processed_tickets


def extract_credentials():
    from ticket_analysis.credentials import get_credentials

    header, cookies = get_credentials()

    json_data = {
        'SettingName': 'SearchAndCopilotIndexMode',
    }

    response = requests.post(
        'https://isp.crm4.dynamics.com/api/data/v9.2/GetOrgDbOrgSetting',
        cookies=cookies,
        headers=header,
        json=json_data,
    )
    
    if response.status_code != 200:
        print(f"Failed to get the credentials. Status code: {response.status_code}")
        return None, None
    else:
        print("Successfully retrieved credentials.")
        return header, cookies

INDEX_LOCK = Lock()
INDEX = 0


DATABASE_DIR = "../Database" 
OUTPUT_DIR = os.path.join(DATABASE_DIR, "Ticket_Extraction/Tickets")
TRIMMED_DIR = os.path.join(DATABASE_DIR, "Ticket_Extraction/Tickets_Trimmed")

SICHARGE_D_GUIDS = [
"16e850ee-7867-ed11-9561-000d3aba342f",
"0e33cc4c-006d-ed11-9561-000d3ad812b5",
"9e65fca9-306e-ed11-9561-0022489de400",
"cf696050-5077-ed11-81ab-0022489de576",
"aa13aa6a-5077-ed11-81ab-0022489de576",
"9a8e0c7b-5077-ed11-81ab-0022489de576",
"cb10368d-8667-ed11-9561-000d3adb5677",
]

GUID_FILTER = "(" + " or ".join(
    [f"_sie_productfamilyid_value eq {g}" for g in SICHARGE_D_GUIDS]
) + ")"

BASE_URL = "https://isp.crm4.dynamics.com/api/data/v9.2/"


def main():
    

    if not os.path.exists(DATABASE_DIR):
        print("Database directory does not exist.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    header, cookies = extract_credentials()

    
    # Pre-processing of the event table, containing all the information organized by charger, about the earliest (min) and latest (max) time for events.

    # min_max_events = pd.read_csv(os.path.join(DATABASE_DIR, "min_max_events.csv"))
    sicharge_family_min_max_events = pd.read_csv(os.path.join(DATABASE_DIR, "SichargeD_Family_min_max.csv"))
    
    # print(min_max_events.head())

    # for charger, min_, max_ in min_max_events.values[:10]:
    #     print(charger, min_, max_)
    
    max_workers = 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_charger_tickets, charger, min_, max_, serial_number, header, cookies): charger
            for charger, min_, max_, serial_number in sicharge_family_min_max_events[["charging_station_id", "min", "max", "serial_number"]].values
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting tickets"):
            try:
                count = future.result()
            except Exception as e:
                charger = futures[future]
                print(f"Error processing charger {charger}: {e}")
    
    print(INDEX)

    print("TRIM EMAIL THREADS IN TICKET JSON FILES")
    print("=" * 80)

    print(f"\nInput:  {OUTPUT_DIR}")
    print(f"Output: {TRIMMED_DIR}\n")

    # Create output directory
    os.makedirs(TRIMMED_DIR, exist_ok=True)

    # Get all JSON files
    json_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])

    print(f"Found {len(json_files)} JSON files to process\n")

    total_files = 0
    total_emails_trimmed = 0
    total_chars_saved = 0
    files_with_trims = 0

    for i, filename in enumerate(json_files, 1):
        input_path = os.path.join(OUTPUT_DIR, filename)
        output_path = os.path.join(TRIMMED_DIR, filename)

        try:
            # Read ticket
            with open(input_path, 'r', encoding='utf-8') as f:
                ticket_data = json.load(f)

            # Process ticket
            processed_data, trimmed_count, chars_saved = process_ticket(ticket_data)

            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)

            # Stats
            total_files += 1
            total_emails_trimmed += trimmed_count
            total_chars_saved += chars_saved

            if trimmed_count > 0:
                files_with_trims += 1

            # Progress
            if i % 50 == 0 or i == len(json_files):
                print(f"[{i}/{len(json_files)}] Processed: {filename[:50]:<50} "
                      f"(trimmed: {trimmed_count}, saved: {chars_saved:,} chars)")

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed:      {total_files}")
    print(f"Files with trimmed emails:  {files_with_trims} ({files_with_trims/total_files*100:.1f}%)")
    print(f"Total emails trimmed:       {total_emails_trimmed}")
    print(f"Total characters saved:     {total_chars_saved:,}")
    print(f"Avg chars saved per file:   {total_chars_saved/total_files:,.0f}")
    print(f"Avg chars saved per trim:   {total_chars_saved/max(total_emails_trimmed,1):,.0f}")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    main()

    # Current problems
    # 6073703e-d3fe-11ef-8aab-30e283986fa6
    # 9NAmV1
    # 9a0f43b2-4f34-11ef-97fd-48701e07de4c
    # BQiyAQ
    # BTBkPN
    # rGol7c
    # wjhPyk
    # wmDywi


    # dic = defaultdict(lambda: 0)
    
    # for element in tqdm(os.listdir(OUTPUT_DIR)):
    #     with open(os.path.join(OUTPUT_DIR, element), "r") as in_file:
    #         data = json.load(in_file)
    #         dic[data["chargerID"]] += 1
        
    # isp_tickets = pd.read_csv(os.path.join(DATABASE_DIR, "Merged_ISP.csv"))

    # isp_tickets = isp_tickets.agg("device_id").count()

    # print(isp.head())

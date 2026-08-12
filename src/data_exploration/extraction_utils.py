
import re
import pandas as pd
import json


# Some of the files do not have consistent lines.
#   I check, the problematic lines are 'fetch_creditials' information. I considered them useless so I just ignore them.


def get_transactions_from_raw_log(file_name):
    transactions_start_stops = {1: [[]], 2: [[]], 3: [[]], 4:[[]]}
    indexes = {
        1: 0, 2:0, 3:0, 4:0
    }

    last_Ev_disconnect = {
        1: None,
        2: None,
        3: None,
        4: None
    }

    transactions_resistances = {1:[], 2:[], 3:[],4:[]}
    transactions_capacitances = {1:[], 2:[], 3:[],4:[]}
    transactions_meter_values = {1:[], 2:[], 3:[],4:[]}
    transactions_evse_values = {1:[], 2:[], 3:[],4:[]}
    transactions_temperature_values = {1:[], 2:[], 3:[],4:[]}
    transactions_error_values = {0: [], 1:[], 2:[], 3:[],4:[]}
    transactions_status = {0: [], 1:[], 2:[], 3:[],4:[]}
    transactions_alert = {1:[], 2:[], 3:[],4:[]}

    dcx_dcx_specific = {1:[], 2:[], 3:[], 4:[]}
    main_fan = {0: []}
    pilot_state_specific = {1:[], 2:[], 3:[], 4:[]}
    charging_phase_specific = {1:[], 2:[], 3:[], 4:[]}
    all_messages = {0: [], 1: [], 2: [], 3: [], 4: []}

    with open(file_name, "r") as infile:

        inreader = csv.reader(infile, delimiter=",")

        next(inreader)
        for index_line, line in enumerate(inreader):
            # timestamp = pd.Timestamp(float(line[0]), unit="ms")
            # timestamp = int(line[0])
            current_message = line[1]
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(?:\+\d+:\d*)?)', current_message)
            if timestamp:
                timestamp = pd.Timestamp(timestamp.group(1))
            else:
                continue

            match = (re.search(r'DC(\d)', current_message))
            if match:
                match = int(match.group(1))

            if "\"type\":" in current_message:
                value = re.search(r'(\{.*\})', current_message)
                try:
                    session_change = json.loads(value.group(1))
                    if not "type" in list(session_change.keys()):
                        continue
                    if session_change["type"] == "SeccEvConnect":
                        # print(current_message)
                        match = int(session_change["outletId"])
                        curr_index = indexes[match]
                        if len(transactions_start_stops[match][curr_index]) == 0:
                            transactions_start_stops[match][curr_index].append([timestamp, 1, index_line])
                        # else:
                            # If more that one subsequent 'start_charging' flag appears, it will be ignored.
                            # print(current_message, timestamp)
                            # print(transactions_start_stops[match][curr_index])
                            # pass
                    elif session_change["type"] == "SeccEvDisconnect":
                        match = int(session_change["outletId"])
                        curr_index = indexes[match]
                        curr_element = [timestamp, 0, index_line]
                        if len(transactions_start_stops[match][curr_index]):
                            transactions_start_stops[match][curr_index].append(curr_element)
                        else:
                            # There is no information yet. The EvDsiconnect call is free.
                            if not last_Ev_disconnect[match]:
                                last_Ev_disconnect[match] = curr_element  
                            transactions_start_stops[match][curr_index].append(last_Ev_disconnect[match])
                            transactions_start_stops[match][curr_index].append(curr_element)
                        transactions_start_stops[match][curr_index] = sorted(transactions_start_stops[match][curr_index], key=lambda x: x[0])
                        indexes[match] += 1
                        transactions_start_stops[match].append([])
                        last_Ev_disconnect[match] = curr_element

                    elif session_change["type"] == "alert":
                        
                        if not "outlet" in session_change.keys():
                            continue
                        match = int(session_change["outlet"])

                        transactions_alert[match].append([timestamp, session_change])
                except:
                    pass

            elif "OCPP: Send command: StatusNotification" in current_message:
                value = re.search(r'(\{.*\})', current_message)
                status = json.loads(value.group(1))
                match = status["connectorId"]
                transactions_status[match].append([timestamp, status])

            elif "insulation resistance" in current_message:
                value = (re.search(r'(\d+\.?\d*)\s*kOhm', current_message).group(1))
                if "PLC-Error-Flag" in current_message: # There are these errors ?? What to do
                    match = re.search(r'IMD_BC(\d)', current_message)
                    match = int(match.group(1))
                curr_index = indexes[match]
                transactions_resistances[match].append([timestamp, float(value)])

            elif "leakage capacitance" in current_message:
                value = (re.search(r'(\d+\.?\d*)\s*microF', current_message).group(1))
                if "PLC-Error-Flag" in current_message: # There are these errors ?? What to do
                    match = re.search(r'IMD_BC(\d)', current_message)
                    match = int(match.group(1))
                transactions_capacitances[match].append([timestamp, float(value)])
            
            elif "OCPP: Send command: MeterValues" in current_message:
                value = re.search(r'(\{.*\})', current_message)
                temp_dic = json.loads(value.group(1))
                if temp_dic:
                    match = int(temp_dic["connectorId"])
                    transactions_meter_values[match].append([timestamp, temp_dic["meterValue"]])
            
            elif "EVSE_MaxPwr" in current_message:
                value = re.findall(r'(\w+)\[(?:\w+|%)\]=\s*(\d+\.?\d*)', current_message)
                dic_temp = {key:float(value) for key,value in value}

                transactions_evse_values[match].append([timestamp, dic_temp])
            
            elif "[CTD]" in current_message:
                value = (re.findall(r'(\w+)\(\w+\)=(\d+\.?\d*)', current_message))
                dic_temp = {key:float(value) for key,value in value}
                transactions_temperature_values[match].append([timestamp, dic_temp])
            
            elif "COMMAND: on, FEEDBACK: off, BLOCKING: yes" in current_message:
                if not match is None:
                    dcx_dcx_specific[match].append([timestamp, current_message])
            
            elif "PilotState" in current_message and "\"C2\" --> \"E\"" in current_message:
                if not match is None:
                    pilot_state_specific[match].append([timestamp, current_message])
            
            elif "Charging Phase \"P06_CHARGE\" --> \"P00_UNKNOWN\"" in current_message:
                if not match is None:
                    charging_phase_specific[match].append([timestamp, current_message])

            elif "Main Fan - Operation enabled by interlocks" in current_message:
                main_fan[0].append([timestamp, current_message])

            if "PLC-Error-Flag" in current_message:
                if "true" in current_message:
                    code = 1
                else:
                    code = -1

                if "Outlet_DC" in current_message:
                    match = int(re.search(r"Outlet_DC(\d)_codes", current_message).group(1))
                    if "true" in current_message: 
                        code = int(re.search(r"Errorcode:\s*(\d+)", current_message).group(1))
                    string = "Outlet_DCX_code"
                elif "IMD_BC" in current_message:
                    match = int(re.search(r"IMD_BC(\d)", current_message).group(1))
                    string="IMD_BCX"
                elif "PowerConverter[" in current_message:
                    match = 0
                    if "true" in current_message:
                        code = int(re.search(r"PowerConverter\[(\d+)\]", current_message).group(1))
                    string = "PowerConverter[Y]"
                else:
                    match = 0
                    string = re.search(r"'(.+)'", current_message)
                    string = string.group(1)
                transactions_error_values[match].append([timestamp, string, code])

            match = 0
            for txt in [r'DC(\d):', r'"?outlet"?:\s*"?(\d)"?',r'"connectorId":\s*(\d)', r'"outletId":\s*(\d)']:
                try_match = (re.search(txt, current_message))
                if try_match:
                    match = int(try_match.group(1))
                    break
            if match != 0:
                all_messages[match].append([timestamp, current_message ])
            else:
                if "comm outage" in current_message or "StopTransaction" in current_message or "connectionError" in current_message:
                    all_messages[match].append([timestamp, current_message])


    for i in range(1,5):
        transactions_start_stops[i].pop()
        transactions_start_stops[i] = sorted(transactions_start_stops[i], key=lambda x: x[0])
        transactions_resistances[i] = sorted(transactions_resistances[i], key=lambda x: x[0])
        transactions_capacitances[i] = sorted(transactions_capacitances[i], key=lambda x: x[0])
        transactions_meter_values[i] = sorted(transactions_meter_values[i], key=lambda x: x[0])
        transactions_evse_values[i] = sorted(transactions_evse_values[i], key=lambda x: x[0])
        transactions_temperature_values[i] = sorted(transactions_temperature_values[i], key=lambda x: x[0])
        transactions_error_values[i] = sorted(transactions_error_values[i], key=lambda x: x[0])
        transactions_status[i] = sorted(transactions_status[i], key=lambda x: x[0])
        transactions_alert[i] = sorted(transactions_alert[i], key=lambda x: x[0])
        all_messages[i] = sorted(all_messages[i], key=lambda x: x[0])

        dcx_dcx_specific[i] = sorted(dcx_dcx_specific[i], key=lambda x: x[0])
        pilot_state_specific[i] = sorted(pilot_state_specific[i], key=lambda x: x[0])
        charging_phase_specific[i] = sorted(charging_phase_specific[i], key=lambda x: x[0])


    transactions_error_values[0] = sorted(transactions_error_values[0], key=lambda x: x[0])
    all_messages[0] = sorted(all_messages[0], key=lambda x: x[0])
    main_fan[0] = sorted(main_fan[0], key=lambda x:x[0])

    
    
    info_dic = {
        "start_stops" : transactions_start_stops,
        "resistances" : transactions_resistances,
        "capacitances" : transactions_capacitances,
        "meter_values" : transactions_meter_values,
        "evse_values" : transactions_evse_values,
        "temperature_values" : transactions_temperature_values,
        "error_values" : transactions_error_values,
        "status" : transactions_status,
        "alert" : transactions_alert,
        "all_messages": all_messages,

        "dcx_dcx" : dcx_dcx_specific,
        "pilot_state" : pilot_state_specific,
        "charging_phase" : charging_phase_specific,
        "main_fan" : main_fan,
    }

    return info_dic



def assign_correct_values(array, start, end, tolerance=pd.Timedelta('10s')):
    from bisect import bisect_left, bisect_right
    temp = []
    index = bisect_left(array, start-tolerance, key=lambda x: x[0])
    # print(start, end)
    # print(array[index][0])

    while index < len(array) and array[index][0] <= end+tolerance:
        temp.append(array[index])
        index += 1
    return temp
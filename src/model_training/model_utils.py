import pandas as pd
import numpy as np
import optuna
import os
import sklearn as sk
from collections import Counter
from tqdm import tqdm
from collections import Counter, defaultdict
import json




class_short_names = {
    0: "Electrical Component Malfunction",
    1: "User Interface",
    2: "Backend",
    3: "Vehicle Communication",
    4: "Mechanical Malfunctions",
    5: "Software/Firmware"
}

new_class_short_names = {
    0: "Electrical Component Malfunction",
    1: "User Interface",
    2: "Backend",
    3: "Vehicle Communication",
    4: "Mechanical Malfunctions",
    5: "Software/Firmware"
}

# Simpler models
def preprocess_events_counts(specific_events, regex=True):
    specific_events = specific_events[~specific_events["name"].isna()]
    specific_events = specific_events[specific_events["name"] != "VehicleId"]
    specific_events = specific_events[specific_events["name"] != "Connector Availability Update"]
    specific_events = specific_events[specific_events["name"] != "Charger Availability Update"]
    specific_events = specific_events[specific_events["severity"] != "Information"]
    if regex:
        specific_events = specific_events.copy()
        specific_events["name"] = specific_events["name"].str.replace(r'\d+','X',regex=True)
        specific_events["error_code"] = specific_events["error_code"].str.replace(r'\d+','X',regex=True)

    return specific_events 

def get_event_count_no_tot(specific_events, date, threshold_gap, threshold_event, cols):
    subset_events = specific_events[(specific_events["event_time"]>= date-threshold_gap) & (specific_events["event_time"]<= date)]
    subset_events = subset_events.groupby(cols, observed=True, as_index=False, dropna=False).agg(count=("event_time", "size"), device_id=("device_id", "first")).reset_index(drop=1)
    subset_events = subset_events[subset_events["count"]>threshold_event]

    temp = [(nm, cnt) for nm, cnt in subset_events[["name", "count"]].values]
    return temp

def get_event_count(specific_events, date, threshold_gap, total_counts, threshold_event, cols):
    subset_events = specific_events[(specific_events["event_time"]>= date-threshold_gap) & (specific_events["event_time"]<= date)]
    subset_events = subset_events.groupby(cols, observed=True, as_index=False, dropna=False).agg(count=("event_time", "size"), device_id=("device_id", "first")).reset_index(drop=1)
    subset_events = subset_events.merge(total_counts, on=cols, how="left")

    temp = [(nm, cnt, total) for nm, cnt, total in subset_events[subset_events["count_total"]>threshold_event][["name", "count", "count_total"]].values]
    return temp


def get_model_stats(model, x, y, eval_=False, average="binary"):
    scores = []
    scores_train = []
    KFold = sk.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    for train, test in KFold.split(x,y):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        if eval_:
            x_train, x_val, y_train, y_val = sk.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=20, verbose=False)
        else:
            model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # scores.append(sk.metrics.f1_score(y_test, y_pred, average=average))
        scores.append(sk.metrics.f1_score(y_test, y_pred, average=average))
        y_pred_train = model.predict(x_train)
        scores_train.append(sk.metrics.f1_score(y_train, y_pred_train, average=average))


    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    if eval_:
        x_train, x_val, y_train, y_val = sk.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=20, verbose=False)
    else:
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("In test set:")
    print(sk.metrics.classification_report(y_test, y_pred))
    print(sk.metrics.confusion_matrix(y_test, y_pred))
    print(scores, np.mean(scores))

    y_pred = model.predict(x_train)
    print("In train set:")
    print(sk.metrics.classification_report(y_train, y_pred))
    print(sk.metrics.confusion_matrix(y_train, y_pred))
    print(scores_train, np.mean(scores_train))

def model_train(model, x, y, average):
    test_scores = []
    gap = []
    Kfold = sk.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    

    for train, test in Kfold.split(x, y):
        model.fit(x[train], y[train])
        
        # Test score
        y_pred = model.predict(x[test])
        test_score = sk.metrics.f1_score(y[test], y_pred, average=average)
        test_scores.append(test_score)
        
        # Train score (for gap calculation)
        y_pred = model.predict(x[train])
        train_score = sk.metrics.f1_score(y[train], y_pred, average=average)
        gap.append(train_score - test_score)
    return model, test_scores, gap

def get_scores_gaps(study_names, storage_names):
    scores = []
    gaps = []
    for sd_nm, st_nm in zip(study_names, storage_names):

        study = optuna.load_study(study_name=sd_nm, storage=st_nm)
        score, gap = zip(*[trial.values for trial in study.best_trials])
        scores.append(score)
        gaps.append(gap)
    return scores, gaps

# Process Tickets
def get_ticket_info(folder_trimmed, folder_summary):

    summaries = []
    names = []
    charger_id = []
    for index, file_name in tqdm(enumerate(os.listdir(folder_summary))):
        names.append(file_name)
        with open(os.path.join(folder_summary, file_name), "r") as in_file:
            summaries.append(in_file.read())
        json_load = json.load(open(os.path.join(folder_trimmed, file_name.split(".")[0]+".json"), "r"))
        charger_id.append([json_load["chargerID"], json_load["created_on"], json_load["incident_id"], int(index)])

    return charger_id, names, summaries

def create_per_charger_tickets(charger_id):
    per_charger = defaultdict(list)
    for el in charger_id:
        per_charger[el[0]].append([pd.Timestamp(str(el[1])), el[2], int(el[3])])

    return per_charger

def get_ticket_info_per_charger(folder_trimmed, folder_summary):

    charger_id, names, summaries = get_ticket_info(folder_trimmed, folder_summary)
    per_charger = create_per_charger_tickets(charger_id)
    return per_charger, names, summaries
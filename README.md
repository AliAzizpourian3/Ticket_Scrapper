# charger-condition-indicator

This repository contains all the source code used during the data analysis of Electrical Vehicle charger data.


## Required packages

The packages used during the data analytics were divided into two .txt files located in the main folder.
- requirements.txt
- requirements-torch.txt

This split was performed to combat some issues encountered when downloading Pytorch-related packages.


## Structure

This notebook is organized in 4 main folders:

 - Database: Has all the relevant data that is used by the remaining python scripts.
 - src: Contains all the reusable python scripts that are used for the data analysis and other specific tasks, the helper functions are further subdivided into different folders, depending on their scope.
 - notebooks: Groups all jupyter notebooks used throught the repository. These files are primarily used for all data analysis and model evaluation, however initial implementations for specific tasks are also tested in these environments.
 - scripts: This folder is reserved for all the python scripts that solve specific tasks.


## Summary

### Ticket Analysis

During ticket analysis, processing, and exploration 2 files, 2 notebooks, and respective helper functions were created.

All the functions that are often repeated during the main tasks were saved at ```src/ticket_analysis```.

The script used for the extraction of tickets from the ISP database is located at ```scripts/ticket_extraction.py```. After updating the credentials in ```src/ticket_analysis/credentials.py```, this script will extract the tickets from all the relevant information from the tickets logged for chargers of the SichargeD family that were created between a predetermined period. This script will scrape:
- Tread of emails between client and customer support.
- Notes added by customer support.
- Ticket related data: Incident nmr, Creation date, title, charger ID & serial number.

A script was created to merge the information from an initial ISP dump with other relevant charger-related information: ```scripts/isp_merge.py```

Lastly, two notebooks were created:

```notebooks/clustering.ipynb``` - Notebook used to test the ticket-clustering methodologies.

```notebooks/ticket_exploration.ipynb``` - Notebook used to compute some ticket-statistics, and tested initial implemenations for ticket prediction.


## Branches

In the current state of the repository there are two branches: "main" and "experimental".

The "main" branch contains all the cleaned version of the code that is first developed in the "experimental" branch.

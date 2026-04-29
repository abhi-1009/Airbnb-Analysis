## Airbnb Data Analysis and Visualization
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat&logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![MongoDB](https://img.shields.io/badge/Data-JSON%2FMongoDB-47A248?style=flat&logo=mongodb)
![Mapbox](https://img.shields.io/badge/Maps-Mapbox%20Scatter-000000?style=flat&logo=mapbox)

An end-to-end data analysis and interactive visualization application built on the **Airbnb Sample Dataset**. The app performs deep data cleaning, nested JSON extraction, geospatial mapping, price analysis, availability analysis, and location analysis — all served through a multi-tab Streamlit dashboard powered by Plotly Express.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Pipeline](#data-pipeline)
- [Dashboard Sections](#dashboard-sections)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [References](#references)

## Project Overview
Airbnb is one of the world's largest online marketplaces for short-term property rentals. This project:
- Loads the raw Airbnb JSON dataset and converts it to a structured CSV
- Extracts and flattens **5 nested columns** (`host`, `address`, `availability`, `amenities`, `location`) into separate DataFrames
- Merges all extracted data into a single master DataFrame (`Airbnb.csv`)
- Serves an interactive 5-tab Streamlit dashboard with **price analysis**, **availability sunburst charts**, **location analysis**, **geospatial scatter maps**, and **top charts**

## Dataset
| Property | Detail |
| :--- | :--- |
| **Source** | Airbnb Sample Dataset (MongoDB Atlas / JSON format) |
| **Raw File** | `sample_airbnb.json` |
| **Processed File** | `Airbnb.csv` (merged master dataset) |
| **Key Nested Columns** | `host`, `address`, `availability`, `amenities`, `location` |

### Columns Dropped (high null / irrelevant)
`neighborhood_overview`, `last_scraped`, `review_scores`, `first_review`, `description`, `reviews_per_month`, `monthly_price`, `weekly_price`, `summary`, `space`, `notes`, `transit`, `access`, `interaction`, `house_rules`, `calendar_last_scraped`, `last_review`, `security_deposit`

### Data Type Conversions (from source code)
| Column | Conversion Applied |
| :--- | :--- |
| `price` | `str → float → int` |
| `extra_people` | `str → float → int` |
| `guests_included` | `str → float → int` |
| `cleaning_fee` | `str → float → int` (NaN → 0) |
| `minimum_nights` | `→ int` |
| `maximum_nights` | `→ int` |
| `bedrooms`, `beds`, `bathrooms` | `→ int` (NaN → 0) |
| `host_is_superhost` | `True/False → Yes/No` |
| `host_has_profile_pic` | `True/False → Yes/No` |
| `host_identity_verified` | `True/False → Yes/No` |

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, type conversion, merging |
| **NumPy** | latest | Numerical operations |
| **Streamlit** | 1.x | Multi-tab interactive web dashboard |
| **streamlit-option-menu** | 0.3.x | Sidebar navigation menu |
| **Plotly Express** | 5.x | Bar, pie, sunburst, scatter mapbox charts |
| **Plotly Graph Objects** | 5.x | Advanced chart customisation |
| **Matplotlib** | latest | Supporting visualisations |
| **Seaborn** | latest | Statistical visualisations |
| **Pillow (PIL)** | latest | Load and display images |
| **JSON** | built-in | Parse nested JSON structures |
| **ast** | built-in | Safely evaluate stringified dict/list columns |

### Python Libraries (from source code)
```python
import pandas as pd
import numpy as np
import streamlit as st
import json
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import ast
import warnings
```
## Data Pipeline
```
sample_airbnb.json
        │
        ▼
Load JSON → Convert to CSV (sample_airbnb.csv)
        │
        ▼
Drop 18 irrelevant/high-null columns
        │
        ▼
Fill NaN values (beds, bedrooms, bathrooms, cleaning_fee → 0)
        │
        ▼
Type Conversions (price, extra_people, guests_included → int)
        │
        ▼
Extract & flatten nested columns using ast.literal_eval:
  ├── host        → host_details DataFrame (12 cols)
  ├── address     → address_details DataFrame (6 cols)
  ├── location    → location_details + coordinates_df (lat/lon)
  ├── availability→ availability_details (4 cols: 30/60/90/365)
  └── amenities   → amenities_details (sorted list per listing)
        │
        ▼
Merge all DataFrames into airbnb_details (master)
        │
        ▼
Save → Airbnb.csv
        │
        ▼
Streamlit Dashboard (5 tabs)
```
## Dashboard Sections
### Home
- Airbnb overview and background
- Company description and global presence
### About
Project methodology explained across 5 sections:
Data Collection → Data Cleaning → EDA → Visualization → Geospatial Analysis
### Data Exploration (5 Tabs)
#### Tab 1 — Price Analysis
| Filter | Chart |
| :--- | :--- |
| Country → Room Type | Bar chart: Price by Property Type (with review count hover) |
| Property Type | Pie chart: Price difference by Host Response Time |
| Host Response Time | Grouped bar: Min/Max nights by Bed Type |
| — | Grouped bar: Bedrooms, Beds, Accommodates by Bed Type |
#### Tab 2 — Availability Analysis
| Filter | Chart |
| :--- | :--- |
| Country → Property Type | Sunburst: Availability 30 days (Room→Bed→Location) |
| — | Sunburst: Availability 60 days |
| — | Sunburst: Availability 90 days |
| — | Sunburst: Availability 365 days |
| Room Type | Grouped bar: Availability 30/60/90/365 by Host Response Time |
#### Tab 3 — Location Analysis
| Filter | Chart |
| :--- | :--- |
| Country → Property Type | Price range radio (0–30% / 30–60% / 60–100%) |
| Price Range | DataFrame view + Bar: Cleaning fee, Bedrooms, Beds by Accommodates |
| Room Type | Horizontal bar: Market by Street/Host Location/Neighbourhood |
| — | Grouped bar: Government Area by Superhost/Neighbourhood/Cancellation Policy |
#### Tab 4 — Geospatial Visualization
- **Mapbox scatter map** of all listings worldwide
- Colour-coded by `price`, size by `accommodates`
- Hover shows listing name
- Full interactive zoom and pan
#### Tab 5 — Top Charts
| Filter | Charts |
| :--- | :--- |
| Country → Property Type | Horizontal bars: Total & Average Price by Host Neighbourhood |
| — | Horizontal bars: Total & Average Price by Host Location |
| Room Type | Top 100 listings: Price coloured rainbow by Min/Max nights |
| — | Top 100 listings: Price coloured green by Bedrooms/Beds/Bed Type |

## Installation and Setup
### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/Airbnb-Analysis.git
cd Airbnb-Analysis
```
### Step 2 — Install Required Libraries
```bash
pip install streamlit streamlit-option-menu pandas numpy plotly matplotlib seaborn pillow
```
### Step 3 — Add the Dataset
Place your `sample_airbnb.json` file in the project folder and update the path in the code:
```python
json_file_path = 'sample_airbnb.json'
csv_file_path  = 'sample_airbnb.csv'
merged_csv_path = 'Airbnb.csv'
```
### Step 4 — Run the Data Processing Script
```bash
python airbnb_data_processing.py
```
### Step 5 — Launch the Streamlit Dashboard
```bash
streamlit run airbnb_app.py
```
Open your browser at `http://localhost:8501`

## Usage
1. **Sidebar** — Select Home / About / Data Exploration from the option menu
2. **Data Exploration** — Choose any of the 5 tabs
3. **Price Analysis** — Use dropdown filters (Country → Room Type → Property Type → Host Response Time) to drill down
4. **Availability** — Select Country and Property Type to view sunburst charts for 30/60/90/365-day windows
5. **Location** — Select price range via radio buttons to filter listings and view location-based charts
6. **Geospatial** — Pan and zoom the world map; hover on any dot to see listing name and details
7. **Top Charts** — Select Country and Property Type to rank neighbourhoods and locations by price

## Key Insights
- **Price drivers** — Property type, host response time, and bed type are the strongest price differentiators
- **Availability patterns** — Entire home/apt listings tend to have higher 365-day availability than private rooms
- **Superhost effect** — Listings with verified, responsive superhosts command premium pricing
- **Geospatial clustering** — High-priced listings cluster in North America, Western Europe, and Australia
- **Top markets** — USA, Australia, Canada, and Portugal consistently appear as high-transaction-count countries

## References
- [Airbnb Sample Dataset — MongoDB Atlas](https://www.mongodb.com/docs/atlas/sample-data/sample-airbnb/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Mapbox Scatter Maps — Plotly](https://plotly.com/python/scattermapbox/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu)

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com

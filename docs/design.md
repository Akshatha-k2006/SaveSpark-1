# Design Document: Household Electricity Consumption Forecasting

**Team:** A13 · **Guide:** Prof. Shrinidhi A.
**Based on:** PRD v0.1 (12 October 2026)
**Audience:** Anyone on the team, even if you have never built an ML web app before.

> Items marked **[Proposed]** are recommendations to discuss with the team and guide. They are not locked decisions. Items marked **[Verify]** must be checked before you rely on them.

---

## 1. What are we building? (In plain words)

A website where you pick a day and see:

1. **How much electricity** each of 5 appliances will probably use, hour by hour.
2. **Which hour is the busiest** (the "peak hour").
3. **The total for the whole day.**
4. **What changes** if something unusual happens (festival, AC under repair, a function at home).

The five appliances: **Refrigerator, Washing machine, AC, Lights, Fan.**

Think of it as a weather forecast, but for your electricity use.

---

## 2. The big picture

Four parts, each with one job. Keep them separate so teammates can work in parallel.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. DATA      │ -> │ 2. ML MODEL  │ -> │ 3. BACKEND   │ -> │ 4. FRONTEND  │
│ collect &    │    │ learn        │    │ API that     │    │ website with │
│ clean        │    │ patterns     │    │ runs model   │    │ charts       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   (Python)         (scikit-learn /       (FastAPI)          (React or plain
                      XGBoost)                                 HTML/JS)
```

**How a request flows:**

1. User picks a date (and optionally "AC under maintenance") on the website.
2. Frontend sends the request to the backend API.
3. Backend builds the model inputs (hour, weekday, temperature, holiday flag, ...).
4. Backend asks the trained model for 24 hourly predictions per appliance.
5. Backend adds them up, finds the peak hour, and sends everything back as JSON.
6. Frontend draws the charts.

---

## 3. Decisions we should lock first

The PRD lists 18 open questions. These recommendations unblock most of them. **[Proposed]**

| Question | Recommendation | Why |
|---|---|---|
| Unit of hourly output | **kWh per hour** | Simple, matches what a bill uses |
| Peak hour means | Hour with highest **total household** kWh (also show each appliance's peak as a bonus) | One clear definition |
| Daily total | **Sum of the 24 hourly predictions** | Consistent numbers; no second model needed |
| Forecast horizon | **Each hour of a chosen day** | Easiest to explain and test |
| Exceptional cases | **Rule-based scenario controls** on top of the model (see Section 7) | Datasets rarely contain failures, so the model cannot learn them |
| Login / registration | **Skip for the first version** | Not needed for the ML experiment |
| Alerts | **Forecast-based only** (predicted > user limit) | No real-time meter exists |
| Suggestions | **Simple rules** | Cheap and honest |

---

## 4. Data (the most important and riskiest part)

The whole project depends on finding data that has **separate readings for all 5 appliances**. A dataset with only total household usage **cannot** train five separate appliance predictors.

### 4.1 Candidate datasets **[Verify every one]**

| Dataset | What to check |
|---|---|
| UK-DALE | Appliance-level data exists; check if AC and fan are present (UK homes rarely have them) |
| REFIT | Appliance-level, UK households; same AC/fan concern |
| UCI Household Power | Has only 3 sub-meters (kitchen, laundry, water heater/AC); not a clean match for 5 appliances |
| Pecan Street (Dataport) | Often includes AC, fridge, washer, lights; check access terms for students |
| An Indian dataset or survey | Best fit for climate and habits; may be hard to find |

### 4.2 Checklist before choosing (from PRD Section 8.5)

- [ ] Does it measure all 5 appliances (or a defensible proxy)?
- [ ] Is the timestamp detailed enough to compute hourly values?
- [ ] Does it have months of continuous history?
- [ ] Can we match it with weather data for the same place and time?
- [ ] Is the license OK for academic use?

### 4.3 If no dataset has all 5 appliances **[Proposed fallback]**

Pick the dataset closest to our needs, train on the appliances it does have, and **clearly document** which ones are missing. Do not invent measured values for the others. Tell the guide early; the PRD (question 18) already asks what minimum is acceptable.

### 4.4 Weather and calendar data

- **Weather:** temperature (and humidity if available), from a free API or downloaded history. **[Verify]** licensing.
- **Holidays/festivals:** a simple CSV of dates for the chosen region, or the Python `holidays` library.

---

## 5. Machine learning design

### 5.1 What the model predicts

- **Target:** kWh used in one hour, per appliance.
- **Approach [Proposed]:** one model per appliance (simple, easy to debug). Compare with a multi-output model later if time allows.

### 5.2 Features (inputs)

| Group | Features |
|---|---|
| Time | hour of day, day of week, is_weekend, month |
| Calendar | is_holiday, is_festival |
| Weather | temperature, humidity |
| Past usage | usage at the same hour yesterday and last week (see warning below) |
| Appliance status | available / maintenance / failed |

> **Warning (data leakage):** Only use features you would actually know **at prediction time**. For tomorrow's forecast you cannot use tomorrow's real usage. Past-usage features must come from before the forecast day.

### 5.3 Models to try, in this order

1. **Baseline:** predict "same hour last week" (no ML at all). Every model must beat this.
2. **Linear / Ridge Regression:** simple and fast.
3. **Random Forest:** handles non-linear patterns well.
4. **XGBoost:** usually the strongest on tables like ours.
5. Time-series/deep models only if time remains. Not recommended for a first version.

### 5.4 Cleaning steps

- Remove duplicate rows.
- Fill or drop missing readings (write down which).
- Remove impossible values (negative energy, sensor spikes).
- Resample to hourly (sum energy within each hour).
- Check units are consistent (W vs kW vs kWh).

### 5.5 Evaluating fairly

- **Split by time, never randomly.** Train on older data, test on the newest data. Random splits let the model peek at the future and give fake-good scores.
- **Metrics:** MAE (average error in kWh), RMSE (punishes big errors), R².
- Report results **per appliance**, plus for the household total and the peak-hour hit rate (how often the predicted peak hour equals the real one).
- Write the numbers into a table in the report. "Accurate" needs numbers behind it.

---

## 6. System design

### 6.1 Folder structure **[Proposed]**

```
electricity-forecast/
├── data/
│   ├── raw/                # original files, never edited
│   └── processed/          # cleaned hourly data
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_clean.ipynb
│   └── 03_train_eval.ipynb
├── ml/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── scenarios.py        # rule-based exceptional cases
├── models/                 # saved trained models (.joblib)
├── backend/
│   ├── main.py             # FastAPI app
│   └── schemas.py          # request/response shapes
├── frontend/
│   └── ...                 # React (Vite) or plain HTML/JS
├── docs/
│   ├── PRD.md
│   └── design.md
└── README.md
```

### 6.2 Tech stack **[Proposed]**

| Layer | Choice | Why |
|---|---|---|
| ML | Python, pandas, scikit-learn, XGBoost | Standard and well documented |
| Backend | FastAPI | Easy for beginners, automatic API docs page |
| Frontend | React + Vite + Chart.js (or plain HTML + Chart.js) | Pick whatever the team is most comfortable with |
| Storage | CSV files first, SQLite if history is needed | No database server to set up |
| Version control | Git + GitHub | Shared work without conflicts |

### 6.3 API design (backend)

**`POST /forecast`**: get a forecast for a day.

Request:

```json
{
  "date": "2026-11-14",
  "region": "Puttur",
  "conditions": {
    "ac_status": "maintenance",
    "function_at_home": true
  }
}
```

Response:

```json
{
  "date": "2026-11-14",
  "unit": "kWh",
  "hourly": [
    { "hour": 0, "fridge": 0.06, "washer": 0.0, "ac": 0.4, "lights": 0.0, "fan": 0.1, "total": 0.56 }
  ],
  "daily_total": 14.2,
  "daily_by_appliance": { "fridge": 1.5, "washer": 0.8, "ac": 6.1, "lights": 2.0, "fan": 3.8 },
  "peak": { "hour": 21, "total_kwh": 1.4 },
  "conditions_applied": ["ac_status=maintenance", "function_at_home=true"],
  "warnings": []
}
```

**`GET /history?from=...&to=...`** *(Should-have)*: past actual consumption.
**`GET /health`**: returns "ok" so you can check the server is running.

The `conditions_applied` and `warnings` fields matter: the PRD requires that the user can always see which assumptions were used.

### 6.4 Frontend screens

1. **Forecast page (main):**
   - Date picker + "Exceptional conditions" toggles.
   - Big **Peak hour badge** ("Peak: 9 PM, 1.4 kWh").
   - Hourly line/bar chart (stacked by appliance).
   - Daily total + appliance breakdown (donut or bar).
   - A clear note on units and which scenario is active.
2. **History page** *(Should)*: date-range chart, predicted vs actual.
3. **Settings** *(Could)*: consumption limit for alerts.

Beginner UI rules: label every axis with units, use one color per appliance consistently, show friendly errors ("No data for this date") instead of blank screens.

---

## 7. Handling exceptional conditions

Be honest here. The model can only learn what is in the data, and most datasets have no appliance failures. So split it in two:

| Condition | How to handle **[Proposed]** | Type |
|---|---|---|
| Weekend / holiday / festival | Model feature (`is_holiday`, etc.), learned from data if the data has enough such days | **Learned** |
| Hot or rainy weather | Temperature/humidity features | **Learned** |
| Appliance failed / under maintenance | User selects status; that appliance's prediction is set to 0 (or reduced), and the total is recomputed | **Rule-based** |
| Family function | User ticks "function"; apply a documented multiplier to lights/fan/AC for the affected hours | **Rule-based (assumption)** |
| "Thunder" | Ask the guide what is meant (weather forecast? power cut? appliance damage?). Treat as a weather flag only until clarified | **TBD** |

Rules for the UI: label rule-based results as **"Scenario estimate"**, not as a validated prediction. Document the multiplier values and why you chose them.

---

## 8. Build plan (suggested order)

| Phase | Work | Output |
|---|---|---|
| 0 | Pick dataset, confirm with guide | Dataset decision |
| 1 | Explore and clean data | Hourly clean table |
| 2 | Baseline + Ridge / RF / XGBoost, time-based evaluation | Metrics table |
| 3 | Save best model, write `predict.py` and `scenarios.py` | Prediction function |
| 4 | FastAPI wrapper (`/forecast`) | Working API |
| 5 | Frontend forecast page with charts | Demo-able website |
| 6 | Extras if time: history, alerts, suggestions | Polish |
| 7 | Report, limitations, final testing | Submission |

**Tip:** Make a rough end-to-end version first (even with a dumb model), then improve each part. A working simple system beats a perfect model with no UI.

### 8.1 Suggested team split **[Proposed]**

| Area | Person |
|---|---|
| Data cleaning + features | Pair of two members |
| Model training + evaluation | One member |
| Backend API + integration | One member |
| Frontend | Whoever is strongest in UI, with help from others |

Rotate and review each other's work. Everyone should be able to explain the whole pipeline at the review.

---

## 9. Testing

- **Data:** no missing hours, no negative values, units consistent.
- **Model:** beats the baseline on the test period.
- **API:** invalid date returns a clear error; unsupported scenario returns a warning, not a fake number.
- **Consistency:** the hourly values always add up to the daily total; the peak hour is the true maximum.
- **UI:** works for missing data and for each scenario toggle.

---

## 10. Risks and what to do about them

| Risk | Mitigation |
|---|---|
| No dataset covers all 5 appliances | Raise with guide early; document gaps; do not fake labels |
| Foreign data does not match Indian homes | State it as a limitation; use local weather/holidays |
| Too much scope for a mini-project | Build "Must" items first (PRD Section 7); defer login and alerts |
| Model overfits | Time-based split, compare against baseline |
| Scenario numbers look arbitrary | Document assumptions; label as estimates |

---

## 11. Glossary (quick reference)

- **kWh:** energy used over time. A 1 kW heater running for 1 hour uses 1 kWh.
- **kW / W:** power, the instantaneous rate of use.
- **Feature:** an input to the model (hour, temperature, ...).
- **Target:** what the model predicts (kWh in an hour).
- **Baseline:** a simple method used to check whether ML is actually helping.
- **Overfitting:** the model memorizes training data and fails on new data.
- **Data leakage:** accidentally using future information when training or testing.
- **API:** the doorway the website uses to ask the backend for results.
- **Inference:** using a trained model to make predictions.

---

## 12. Immediate next steps

1. Shortlist 2-3 datasets and check each against the Section 4.2 checklist.
2. Meet the guide with the Section 3 recommendations and get a yes/no.
3. Set up the Git repo with the Section 6.1 folder structure.
4. Start the exploration notebook on the chosen dataset.

# AI Inventory & Replenishment Planner

An agentic AI assistant that helps a store manager decide **what to reorder, how much, and when**. It uses one year of real H&M daily sales, with real product names and photos.

**Live app:** https://ai-inventory-replenishment-planner-4dtk2bp4nqeuo73l8pqlan.streamlit.app/
**Notebook:** [Open in Colab](https://colab.research.google.com/github/QuantumFluxx-pew/AI-Inventory-Replenishment-Planner/blob/main/notebooks/AI_Inventory_Replenishment_Planner_HM.ipynb)

## What it does

* **Dashboard (left):** KPIs; stock vs. reorder point, days of cover and order charts; a product table with photos; and, for any product, its photo, the step-by-step calculation, and 90 days of sales plotted against the forecast.
* **Stock Pilot (right):** the AI inventory agent chat; answers questions in plain English, such as *"Which products should I prioritize?"*, *"Why does Lucy blouse need replenishment?"* or *"What if demand increases by 20%?"*. When it talks about a product, it shows its photo.
* **Editable inputs:** ordering cost, holding cost, lead time, demand change, safety stock and forecast window, with **Recalculate**. Invalid values are rejected with an explanation.

## Data: H&M Personalized Fashion Recommendations (Kaggle)

The full dataset has 31 million purchases of 105,000 products, with real names and photos. We use **18 steady-selling products** and their **daily unit sales for one year** (2019-09-24 to 2020-09-22).

How they were chosen:
1. **On Kaggle** (`kaggle/hm_select_products.py`): count units sold per product per day for the last 365 days. Keep products sold on ≥ 90% of days, and rank them by forecastability (SMA error ÷ average demand). Save the top 40 with their photos.
2. **Cleaning** (`prepare_data.py`): check for gaps, negatives and duplicates. Drop products **being phased out** (last 4 weeks < 50% of their yearly average). Keep at most 3 per product type. Make 300-pixel photo thumbnails.

**Assumptions, clearly labelled:** H&M doesn't publish stock levels, costs or lead times. **Current stock is simulated** with a fixed random seed (3–12 days of cover). Ordering cost $50, holding cost $2/unit/year and a 7-day lead time are sample values.

## Embedded analytical models (`inventory_model.py`)

| Model | Formula | Purpose |
|---|---|---|
| Simple Moving Average | d = mean of last N days (N = 7 by default) | Daily demand forecast |
| MAE (back-test) | mean \|actual − one-day-ahead SMA\| over the year | Forecast accuracy |
| Economic Order Quantity | EOQ = ⌈√(2DS / H)⌉, D = 365·d | How much to order |
| Reorder Point | ROP = ⌈d·L + safety stock⌉ | When to order |
| Safety stock (optional) | z · 1.25·MAE · √L | Buffer for forecast error |

Rule: if current stock ≤ ROP, order EOQ units.

Why a 7-day window? Comparing windows across all 18 products, the median scaled error is 0.366 (3 days), **0.350 (7 days)**, 0.381 (14 days) and 0.412 (28 days).

## How the AI agent works (`agent.py`)

```
User question → Gemini LLM → picks a tool + inputs → Python runs the model
                    ↑                                       │
                    └────────── JSON result ◄───────────────┘
              Gemini explains the result (never calculates numbers itself)
```

Tools the agent can call:

* `get_inventory_summary`
* `list_products`
* `get_product_details`
* `prioritize_reorders`
* `run_what_if` (hypothetical, dashboard unchanged)
* `update_assumptions` (changes the dashboard)
* `reset_assumptions`
* `get_assumptions`

The agent **asks for missing inputs** ("increase demand by how much?") instead of guessing. Unclear products are resolved by name, type or article ID; for example, "t-shirt" makes it ask which of the three tees. Each answer has a **"Tools used"** panel showing every tool call and its result, and an automatic **number check**: every number in the answer is compared with the model's results (rounding allowed). A green line means all numbers came from Python; otherwise it lists the numbers it could not find. Limit: an invented number that happens to equal a real model number would not be caught.

**Two modes, always labelled in the app:**

* 🟢 **AI agent (Gemini):** active when a `GEMINI_API_KEY` secret is set.
* 🟠 **Offline mode:** no API key, so it uses simple keyword rules that call the same tools. *This is not an AI agent.* It exists so the app never breaks.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit interface (dashboard + chat) |
| `inventory_model.py` | All calculations: SMA, MAE, EOQ, ROP, safety stock, validation, product lookup |
| `agent.py` | Gemini tool-calling agent, tool definitions, offline fallback |
| `data/products.csv` | 18 products: name, type, colour, description, simulated stock |
| `data/daily_sales.csv` | Real daily unit sales, 365 days × 18 products |
| `images/` | Product photo thumbnails |
| `kaggle/hm_select_products.py` | Step 1: runs on Kaggle to pick steady products from 31M purchases |
| `prepare_data.py` | Step 2: cleans the Kaggle output into `data/` and `images/` |
| `notebooks/…HM.ipynb` | Colab notebook: the full model walk-through with tests |
| `tests/test_inventory.py` | 30 automated tests |
| `.streamlit/config.toml` | Forces the light theme |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

To enable the AI agent locally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key.

## Turn on the AI agent on Streamlit Cloud

1. Get a free API key at **https://aistudio.google.com** → *Get API key*.
2. Go to **https://share.streamlit.io**, find this app, and click **⋮ → Settings → Secrets**.
3. Paste the following and save:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. The badge above the chat turns green: **AI agent: Gemini**.

Never put the key in a file on GitHub. This repository is public.

## Testing

```bash
pip install pytest
python -m pytest -v
```

The tests check the following:

* **Data:** 365 days with no gaps, no negative sales, a photo for every product.
* **Formulas, checked by hand:** Brit Baby Tee sold 65 units in 7 days → forecast 9.29/day → ROP 65, EOQ 412, and stock 65 ≤ 65 → Reorder.
* **Scenarios:**
  * baseline: 8 products, 5,048 units
  * demand +20%: 9 products, 5,962 units
  * lead time 14 days: all 18 products
* **Validation:** negative lead time, zero cost, text instead of numbers, unsupported service level or window, unknown or ambiguous product.
* **Agent tools:** what-ifs don't change the dashboard, `update_assumptions` does, and the assistant asks for a missing percentage instead of guessing.

## Demo script (follows the assignment's six steps)

Run this with the AI agent enabled. The numbers come from the tests. Gemini's wording will vary.

| Step | Say in the chat | Expected behaviour |
|---|---|---|
| 1. Objective | *Which products should I prioritize?* | Harrison short sleeve top first (3.6 days of cover) |
| 2. Missing input | *What happens if demand increases?* | Agent asks: by what percentage? |
| 3. Validate input | *Set lead time to -5 days* | Agent explains lead time must be > 0 |
| 4. Run model | *What if demand increases by 20%?* | Order units 5,048 → 5,962 |
| 5. Explain | *Why does Brit Baby Tee need replenishment?* | Stock 65 ≤ ROP 65, order 412, with calculation steps and photo |
| 6. Revise + rerun | *Change lead time to 14 days* | Dashboard updates: all 18 products need reorder |

Open **"Tools used"** under any answer to show that Python, not the LLM, produced the numbers.

## Limitations

* Stock levels are simulated, and costs and lead time are assumptions. H&M prices in the dataset are scaled, so they aren't used.
* The sales are online purchases, so they're a stand-in for store or warehouse demand.
* An SMA reacts slowly to trends and seasonality; for example, Lucy blouse sales more than doubled in the final weeks.
* Only 18 products; the safety-stock σ ≈ 1.25 × MAE assumes roughly normal errors.
* Offline mode only understands simple phrasings.

## Future improvements

* Compare SMA with exponential smoothing and choose the more accurate one per product.
* Let users upload their own stock levels and costs.
* Detect fading products automatically and suggest not reordering them.

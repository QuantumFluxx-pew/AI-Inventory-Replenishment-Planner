# AI Inventory & Replenishment Planner

An agentic AI assistant that helps a warehouse manager decide **what to reorder, how much, and when**, built on Walmart sales data from the M5 Forecasting dataset.

**Live app:** https://ai-inventory-replenishment-planner-4dtk2bp4nqeuo73l8pqlan.streamlit.app/

## What it does

* **Dashboard (left):** KPIs, stock vs. reorder point chart, product table, CSV download, and a step-by-step calculation for any product.
* **Assistant (right):** answers questions in plain English, such as *"Which products should I prioritize?"* or *"What if demand increases by 20%?"*.
* **Editable inputs:** change ordering cost, holding cost, lead time, demand change or safety stock, then recalculate. Invalid values are rejected with an explanation.

## Embedded analytical models

| Model | Formula | Purpose |
|---|---|---|
| Simple Moving Average | d = mean of last 7 days of sales | Daily demand forecast |
| MAE | mean \|actual − one-day-ahead SMA\| | Forecast accuracy |
| Economic Order Quantity | EOQ = ⌈√(2DS / H)⌉, D = 365·d | How much to order |
| Reorder Point | ROP = ⌈d·L + safety stock⌉ | When to order |
| Safety stock (optional) | z · 1.25·MAE · √L | Buffer for forecast error |

Rule: if current stock ≤ ROP, order EOQ units.

**Default assumptions** (Milestone 2): S = $50/order, H = $2/unit/year, L = 7 days, no safety stock.
These are sample values. M5 has no cost, lead-time or inventory data, and **current stock is simulated**.

## How the AI agent works

```
User question → Gemini LLM → picks a tool + inputs → Python runs the model
                    ↑                                       │
                    └────────── JSON result ◄───────────────┘
              Gemini explains the result (never calculates numbers itself)
```

Tools the agent can call (`agent.py`):

* `get_inventory_summary`
* `list_products`
* `get_product_details`
* `prioritize_reorders`
* `run_what_if` (hypothetical, dashboard unchanged)
* `update_assumptions` (changes the dashboard)
* `reset_assumptions`
* `get_assumptions`

The agent is instructed to **ask for missing inputs** (for example, "increase demand by how much?") rather than guess. Each answer shows a **"Tools used"** panel listing every tool call and its result.

**Two modes, always labelled in the app:**

* 🟢 **AI agent (Gemini):** active when a `GEMINI_API_KEY` secret is set.
* 🟠 **Offline mode:** no API key, so it uses simple keyword rules that call the same tools. *This is not an AI agent.* It exists so the app never breaks.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit interface (dashboard + chat) |
| `inventory_model.py` | All calculations: SMA, MAE, EOQ, ROP, safety stock, validation |
| `agent.py` | Gemini tool-calling agent, tool definitions, offline fallback |
| `prepare_data.py` | Rebuilds the CSV from raw M5 data (run locally) |
| `inventory_replenishment_results.csv` | 20 HOBBIES products, store TX_1, with simulated stock |
| `product_prices.csv` | Real shelf price of each product (M5 `sell_prices.csv`, latest week). Shown for context only |
| `.streamlit/config.toml` | Forces the light theme |

**About product names:** Walmart anonymised M5, so there are no real product names. An ID like `HOBBIES_1_019` means *category HOBBIES, department 1, item 019*. The app shows it as **"Hobbies Item 019"**.
| `tests/test_inventory.py` | 21 automated tests |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

To enable the AI agent locally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste in your key.

## Turn on the AI agent on Streamlit Cloud

1. Get a free API key at **https://aistudio.google.com** → *Get API key*.
2. Go to **https://share.streamlit.io**, find this app, click **⋮ → Settings → Secrets**.
3. Paste the following and save:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. The app restarts, and the badge above the chat turns green: **AI agent: Gemini**.

Never put the key in `app.py` or any file on GitHub. This repository is public.

## Rebuild the data from M5 (optional)

Download `sales_train_validation.csv` from the [Kaggle M5 competition](https://www.kaggle.com/c/m5-forecasting-accuracy), place it in this folder, then run:

```bash
python prepare_data.py
```

This reproduces `inventory_replenishment_results.csv` exactly. Put `sell_prices.csv` in the folder too, and it also writes `product_prices.csv`.

## Testing

```bash
pip install pytest
python -m pytest -v
```

The tests check the following:

* The models reproduce the Milestone 2 results exactly (12 reorders, 1,856 units).
* EOQ, ROP, safety stock, SMA and MAE match hand calculations.
* Invalid inputs are rejected: negative lead time, zero cost, text instead of numbers, unsupported service levels, unknown product IDs.
* What-if scenarios don't change the dashboard, and `update_assumptions` does.
* The assistant asks for a missing percentage instead of guessing.

## Demo script (follows the assignment's six steps)

Run this with the AI agent enabled. The numbers below are verified by the automated tests. Gemini's exact wording will vary.

| Step | Say in the chat | Expected behaviour |
|---|---|---|
| 1. Objective | *Which products should I prioritize?* | Ranked list: HOBBIES_1_013 first (2.3 days of cover) |
| 2. Missing input | *What happens if demand increases?* | Agent asks: by what percentage? |
| 3. Validate input | *Set lead time to -5 days* | Agent explains lead time must be > 0 |
| 4. Run model | *What if demand increases by 20%?* | Order units 1,856 → 2,031 |
| 5. Explain | *Why does HOBBIES_1_019 need replenishment?* | Stock 40 ≤ ROP 53, order EOQ 372, with calculation steps |
| 6. Revise + rerun | *Change lead time to 14 days* | Dashboard updates: 14 products need reorder, 2,211 units |

Open **"Tools used"** under any answer to show that Python, not the LLM, produced the numbers.

## Limitations

* Stock levels are simulated, and costs and lead time are assumptions.
* A 7-day SMA reacts slowly to trends and seasonality, and many HOBBIES items sell less than 1 unit/day.
* Only 20 products from one store.
* The safety-stock σ ≈ 1.25 × MAE assumes roughly normal forecast errors.
* Offline mode only understands simple phrasings.

## Future improvements

* Use `sell_prices.csv` to set holding cost as a % of each item's price.
* Compare SMA with exponential smoothing and pick the more accurate forecast per item.
* Let users upload their own stock levels.

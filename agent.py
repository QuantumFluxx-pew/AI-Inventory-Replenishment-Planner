"""
agent.py
========

The conversational agent for the AI Inventory & Replenishment Planner.

How it works (tool calling):

    user question
         |
         v
    Gemini LLM  --decides-->  which tool to call, with which inputs
         |                            |
         |                            v
         |                  Python tool (InventoryTools) runs the
         |                  embedded model in inventory_model.py
         |                            |
         |   <------ JSON result -----+
         v
    Gemini explains the result in plain language

The LLM never calculates numbers. Every figure comes from a tool result.

Two modes, clearly labelled in the app:
    * "AI agent (Gemini)"  - used when a GEMINI_API_KEY is configured.
    * "Offline mode"       - rule-based fallback with NO language model. It
                             calls the same tools, but understands only simple
                             phrasings. It exists so the app still works
                             without an API key; it is not an AI agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

import inventory_model as model

MODEL_VERSION = model.MODEL_VERSION   # app.py reloads modules if these differ
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_ROUNDS = 6          # safety limit on tool calls per question
HISTORY_TURNS = 10           # how many earlier messages the LLM sees


# ======================================================================
# 1. TOOLS: thin wrappers that call the embedded model and return JSON
# ======================================================================

def _r(value, digits=2):
    """Round floats for readable tool output; leave other types alone."""
    if isinstance(value, float):
        return round(value, digits)
    return value


def _product_row(row: pd.Series) -> dict:
    return {
        "product_id": row["Product_ID"],
        "name": row["Name"],
        "product_type": row["Product_Type"],
        "colour": row["Colour"],
        "current_stock": int(row["Current_Stock"]),
        "reorder_point": int(row["ROP"]),
        "eoq": int(row["EOQ"]),
        "status": row["Status"],
        "recommended_order_units": int(row["Order_Quantity"]),
        "daily_demand_forecast": _r(float(row["Daily_Demand"]), 3),
        "days_of_cover": None if pd.isna(row["Days_of_Cover"]) else float(row["Days_of_Cover"]),
    }


class InventoryTools:
    """Holds the product data and the current assumptions, and exposes the
    functions the agent may call. Each method returns a JSON-safe dict."""

    def __init__(self, base: pd.DataFrame, assumptions: model.Assumptions):
        self.base = base
        self.assumptions = assumptions

    # -- helpers --------------------------------------------------------
    @property
    def plan(self) -> pd.DataFrame:
        return model.calculate_plan(self.base, self.assumptions)

    def _assumption_notes(self, a: model.Assumptions | None = None) -> dict:
        a = a or self.assumptions
        return {
            **a.to_dict(),
            "notes": (
                "Ordering cost, holding cost and lead time are sample assumptions, "
                "not real H&M figures. Current stock is simulated because the H&M "
                "dataset has no inventory data. Demand forecast is a "
                f"{a.sma_window}-day simple moving average of real H&M daily sales "
                "(one year, 2019-09-24 to 2020-09-22)."
            ),
        }

    # -- tools ----------------------------------------------------------
    def get_assumptions(self) -> dict:
        return self._assumption_notes()

    def get_inventory_summary(self) -> dict:
        return {
            "summary": model.summarize(self.plan),
            "assumptions": self.assumptions.to_dict(),
        }

    def list_products(self, status: str = "all") -> dict:
        plan = self.plan
        status_map = {"all": None, "reorder": "Reorder", "sufficient": "Sufficient Stock"}
        key = (status or "all").strip().lower()
        if key not in status_map:
            return {"error": "status must be 'all', 'reorder' or 'sufficient'."}
        if status_map[key]:
            plan = plan[plan["Status"] == status_map[key]]
        return {
            "status_filter": key,
            "count": int(len(plan)),
            "total_recommended_order_units": int(plan["Order_Quantity"].sum()),
            "products": [_product_row(r) for _, r in plan.iterrows()],
        }

    def get_product_details(self, product_id: str) -> dict:
        plan = self.plan
        pid = model.find_product(plan, product_id)
        row = plan[plan["Product_ID"] == pid].iloc[0]
        a = self.assumptions

        lead_time_demand = row["Daily_Demand"] * a.lead_time_days
        details = _product_row(row)
        details.update({
            "description": row["Description"],
            "sma_window_days": a.sma_window,
            "sma_forecast_units_per_day": _r(float(row["SMA"]), 3),
            "forecast_mae_units_per_day": _r(float(row["MAE"]), 3),
            "annual_demand": _r(float(row["Annual_Demand"]), 1),
            "safety_stock": float(row["Safety_Stock"]),
            "lead_time_days": a.lead_time_days,
            "units_below_reorder_point": int(row["Shortfall"]),
            "annual_policy_cost": float(row["Annual_Cost"]),
            "calculation_steps": {
                "annual_demand": f"{row['Daily_Demand']:.4f} units/day x 365 = {row['Annual_Demand']:.2f}",
                "eoq": (
                    f"ceil(sqrt(2 x {row['Annual_Demand']:.2f} x {a.ordering_cost:g} / "
                    f"{a.holding_cost:g})) = {int(row['EOQ'])}"
                ),
                "reorder_point": (
                    f"ceil({row['Daily_Demand']:.4f} x {a.lead_time_days:g} days"
                    f" + {row['Safety_Stock']:g} safety stock) "
                    f"= ceil({lead_time_demand + row['Safety_Stock']:.2f}) = {int(row['ROP'])}"
                ),
                "decision_rule": (
                    f"stock {int(row['Current_Stock'])} "
                    f"{'<=' if row['Status'] == 'Reorder' else '>'} ROP {int(row['ROP'])}"
                    f" -> {row['Status']}"
                ),
            },
        })
        return details

    def prioritize_reorders(self, limit: int = 5) -> dict:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"error": "limit must be a whole number."}
        limit = max(1, min(limit, 50))
        ranked = model.prioritize(self.plan)
        return {
            "ranking_rule": (
                "Most urgent first: fewest days of stock left (stock / daily demand), "
                "then largest shortfall below the reorder point, then highest demand."
            ),
            "lead_time_days": self.assumptions.lead_time_days,
            "total_needing_reorder": int(len(ranked)),
            "ranked": [
                {**_product_row(r), "units_below_reorder_point": int(r["Shortfall"]), "rank": i + 1}
                for i, (_, r) in enumerate(ranked.head(limit).iterrows())
            ],
        }

    def run_what_if(self, demand_change_pct=None, lead_time_days=None, ordering_cost=None,
                    holding_cost=None, service_level=None, sma_window=None, product_id=None) -> dict:
        """Compare the current plan with a hypothetical one. Does NOT change
        the dashboard; use update_assumptions for that."""
        scenario = model.validate_assumptions(
            base=self.assumptions,
            demand_change_pct=demand_change_pct, lead_time_days=lead_time_days,
            ordering_cost=ordering_cost, holding_cost=holding_cost,
            service_level=service_level, sma_window=sma_window,
        )
        if scenario == self.assumptions:
            return {"error": "No change given. Ask the user which input to change and by how much."}

        before = self.plan
        after = model.calculate_plan(self.base, scenario)
        diff = model.compare_plans(before, after)

        if product_id:
            pid = model.find_product(before, product_id)
            diff = diff[diff["Product_ID"] == pid]
        changed = diff[
            (diff["ROP_before"] != diff["ROP_after"])
            | (diff["EOQ_before"] != diff["EOQ_after"])
            | diff["Status_Changed"]
        ]

        return {
            "current_assumptions": self.assumptions.to_dict(),
            "scenario_assumptions": scenario.to_dict(),
            "summary_before": model.summarize(before),
            "summary_after": model.summarize(after),
            "products_that_newly_need_reorder": diff.loc[
                diff["Status_Changed"] & (diff["Status_after"] == "Reorder"), "Name"
            ].tolist(),
            "products_no_longer_needing_reorder": diff.loc[
                diff["Status_Changed"] & (diff["Status_after"] != "Reorder"), "Name"
            ].tolist(),
            "changed_products": [
                {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
                for rec in changed.drop(columns="Status_Changed").to_dict("records")
            ],
            "note": "Hypothetical scenario only. The dashboard was not changed.",
        }

    def update_assumptions(self, demand_change_pct=None, lead_time_days=None, ordering_cost=None,
                           holding_cost=None, service_level=None, sma_window=None) -> dict:
        """Permanently change the inputs used by the dashboard and the agent."""
        new = model.validate_assumptions(
            base=self.assumptions,
            demand_change_pct=demand_change_pct, lead_time_days=lead_time_days,
            ordering_cost=ordering_cost, holding_cost=holding_cost,
            service_level=service_level, sma_window=sma_window,
        )
        old = self.assumptions
        self.assumptions = new
        return {
            "previous_assumptions": old.to_dict(),
            "new_assumptions": new.to_dict(),
            "new_summary": model.summarize(self.plan),
            "note": "Dashboard recalculated with the new assumptions.",
        }

    def reset_assumptions(self) -> dict:
        self.assumptions = model.DEFAULT_ASSUMPTIONS
        return {"new_assumptions": self.assumptions.to_dict(),
                "new_summary": model.summarize(self.plan),
                "note": "Restored the default assumptions."}

    # -- dispatcher -----------------------------------------------------
    def call(self, name: str, args: dict | None) -> dict:
        """Run a tool by name. Invalid inputs come back as {'error': ...}
        so the agent can ask the user to correct them."""
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return func(self, **(args or {}))
        except model.InputError as err:
            return {"error": str(err), "needs_user_input": True}
        except TypeError as err:
            return {"error": f"Bad arguments for {name}: {err}"}


TOOL_FUNCTIONS = {
    "get_assumptions": InventoryTools.get_assumptions,
    "get_inventory_summary": InventoryTools.get_inventory_summary,
    "list_products": InventoryTools.list_products,
    "get_product_details": InventoryTools.get_product_details,
    "prioritize_reorders": InventoryTools.prioritize_reorders,
    "run_what_if": InventoryTools.run_what_if,
    "update_assumptions": InventoryTools.update_assumptions,
    "reset_assumptions": InventoryTools.reset_assumptions,
}


# JSON-schema descriptions the LLM reads to decide which tool to use.
_ASSUMPTION_PROPS = {
    "demand_change_pct": {"type": "number", "description": "Demand change in percent, e.g. 20 for +20%, -10 for -10%."},
    "lead_time_days": {"type": "number", "description": "Supplier lead time in days (> 0)."},
    "ordering_cost": {"type": "number", "description": "Cost per order placed, in dollars (> 0)."},
    "holding_cost": {"type": "number", "description": "Holding cost per unit per year, in dollars (> 0)."},
    "service_level": {"type": "number", "description": "Safety-stock service level %: one of 0, 90, 95, 97.5, 99. 0 = no safety stock."},
    "sma_window": {"type": "integer", "description": "Days in the moving-average forecast: one of 3, 7, 14, 28."},
}

TOOL_SPECS = [
    {"name": "get_assumptions",
     "description": "Current model assumptions (costs, lead time, demand change, service level) and data limitations.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_inventory_summary",
     "description": "Headline totals: products, how many need reorder, units to order, stock on hand, annual cost.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_products",
     "description": "List products with stock, reorder point, EOQ, status and order quantity. Filter by status.",
     "parameters": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["all", "reorder", "sufficient"]}}}},
    {"name": "get_product_details",
     "description": "Full detail for one product, including step-by-step EOQ/ROP calculations and why it does or does not need reordering.",
     "parameters": {"type": "object", "properties": {
         "product_id": {"type": "string", "description": "Product name (or a unique part of it) or H&M article ID, e.g. 'Brit Baby Tee' or '0624486001'"}},
         "required": ["product_id"]}},
    {"name": "prioritize_reorders",
     "description": "Rank products needing reorder by urgency (fewest days of stock left first).",
     "parameters": {"type": "object", "properties": {
         "limit": {"type": "integer", "description": "How many to return (default 5)."}}}},
    {"name": "run_what_if",
     "description": "Hypothetical scenario: recompute the plan with changed inputs and compare with the current plan. Does not change the dashboard.",
     "parameters": {"type": "object", "properties": {
         **_ASSUMPTION_PROPS,
         "product_id": {"type": "string", "description": "Optional: focus on one product."}}}},
    {"name": "update_assumptions",
     "description": "Permanently change the assumptions used by the dashboard, only when the user asks to set/change/use new values going forward.",
     "parameters": {"type": "object", "properties": _ASSUMPTION_PROPS}},
    {"name": "reset_assumptions",
     "description": "Restore the default assumptions (S=$50, H=$2, 7-day lead time, no safety stock, no demand change, 7-day forecast window).",
     "parameters": {"type": "object", "properties": {}}},
]


SYSTEM_PROMPT = """You are Stock Pilot, an AI inventory agent that helps a store manager with replenishment.
You work with 18 steady-selling H&M clothing products (real product names, e.g.
"Brit Baby Tee", "Pluto RW slacks (1)"), using one year of real H&M daily sales
(2019-09-24 to 2020-09-22, H&M Personalized Fashion Recommendations dataset).
Users may refer to products by name or part of a name; pass their words to the tools.

Your models (run by Python tools, never by you):
- Simple Moving Average demand forecast, 7-day window by default (accuracy measured by MAE)
- Economic Order Quantity: EOQ = sqrt(2DS/H)
- Reorder Point: ROP = daily demand x lead time (+ optional safety stock)
- Rule: reorder EOQ units when current stock <= ROP

STRICT RULES
1. Every number you state must come from a tool result in this conversation.
   Never calculate, estimate, or round numbers yourself. If you need a figure, call a tool.
2. If the user's request is missing a required input, ASK for it instead of guessing.
   Examples: "what if demand increases?" -> ask by what percent.
   "how would a longer lead time affect it?" -> ask how many days.
   An unclear product -> ask which product.
3. If a tool returns an "error", explain the problem in plain words and ask the user
   for a corrected value. Do not retry with a value you made up.
4. Use run_what_if for "what if" questions (dashboard unchanged). Use update_assumptions
   only when the user clearly wants to change the inputs going forward.
5. When you explain a recommendation, say why: compare stock with the reorder point,
   mention days of cover versus lead time, and show the calculation steps the tool gave you.
6. Be honest about limitations when relevant: stock levels are simulated, costs and lead
   time are assumptions, and SMA forecasts can lag sudden demand changes.
7. Be concise. Use short paragraphs or bullet lists and always include units.
   End with a short recommended action when one applies.
8. If the user asks about something outside inventory planning for these products,
   say briefly what you can help with."""


# ======================================================================
# 2. REPLY OBJECT shared by both modes
# ======================================================================

@dataclass
class AgentReply:
    text: str
    mode: str                                   # "llm" or "offline"
    tool_calls: list = field(default_factory=list)   # [(name, args, result), ...]
    error: str | None = None


# ======================================================================
# 3. LLM AGENT (Google Gemini tool calling)
# ======================================================================

class GeminiAgent:
    """Runs the tool-calling loop with Google Gemini."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL, client=None):
        from google.genai import types  # imported here so offline mode needs no SDK
        self.types = types
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client = client
        self.model_name = model_name
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=s["name"], description=s["description"],
                    parameters_json_schema=s["parameters"],
                ) for s in TOOL_SPECS
            ])],
            # We run tools ourselves so every call is visible and logged.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def ask(self, question: str, history: list[dict], tools: InventoryTools) -> AgentReply:
        t = self.types
        recent = list(history[-HISTORY_TURNS:])
        # Gemini expects the conversation to start with a user turn, so drop
        # any leading assistant messages (e.g. the greeting).
        while recent and recent[0]["role"] == "assistant":
            recent.pop(0)
        contents = [
            t.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[t.Part(text=m["content"])])
            for m in recent
        ]
        contents.append(t.Content(role="user", parts=[t.Part(text=question)]))

        log = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=self.config
            )
            calls = response.function_calls or []
            if not calls:
                return AgentReply(text=(response.text or "").strip() or
                                  "Sorry, I couldn't produce an answer. Please rephrase.",
                                  mode="llm", tool_calls=log)

            # Keep the model's turn exactly as returned (includes any thought
            # signatures Gemini needs), then answer every function call.
            contents.append(response.candidates[0].content)
            result_parts = []
            for call in calls:
                args = dict(call.args or {})
                result = tools.call(call.name, args)
                log.append((call.name, args, result))
                result_parts.append(t.Part.from_function_response(
                    name=call.name, response={"result": json.loads(json.dumps(result, default=str))}
                ))
            contents.append(t.Content(role="user", parts=result_parts))

        return AgentReply(text="I needed too many steps for that question. "
                               "Could you ask it in smaller parts?",
                          mode="llm", tool_calls=log)


# ======================================================================
# 4. OFFLINE MODE (rule-based, NO language model)
# ======================================================================

_ARTICLE_ID = re.compile(r"\b0?\d{9}\b")
_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|percent)", re.I)
_DAYS = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*days?", re.I)


def _fmt_products(rows, show_order=True):
    lines = []
    for r in rows:
        line = (f"- **{r['name']}** ({r['product_type']}): "
                f"stock {r['current_stock']}, ROP {r['reorder_point']}")
        if show_order and r["recommended_order_units"]:
            line += f", order **{r['recommended_order_units']}** units"
        lines.append(line)
    return "\n".join(lines)


class OfflineAssistant:
    """Keyword router. Calls the same tools, formats answers with templates.
    It cannot understand free-form language like the LLM agent can."""

    def ask(self, question: str, history: list[dict], tools: InventoryTools) -> AgentReply:
        q = question.lower()
        log = []

        def call(name, **args):
            result = tools.call(name, args)
            log.append((name, args, result))
            return result

        def reply(text):
            return AgentReply(text=text, mode="offline", tool_calls=log)

        product = self._mentioned_product(q, tools)
        pct = _PCT.search(q)
        days = _DAYS.search(q)
        what_if = any(w in q for w in ["what if", "what happens", "increase", "decrease",
                                       "drop", "longer", "shorter", "affect", "impact"])

        # What-if scenarios: ask for the missing number instead of guessing.
        if what_if and "demand" in q:
            if not pct:
                return reply("By what percentage should demand change? For example: "
                             "*What if demand increases by 20%?*")
            change = float(pct.group(1))
            if any(w in q for w in ["decrease", "drop", "fall", "lower"]) and change > 0:
                change = -change
            res = call("run_what_if", demand_change_pct=change)
            return reply(self._what_if_text(res, f"demand change of {change:+g}%"))

        if what_if and "lead time" in q:
            if not days:
                return reply("What lead time should I test, in days? For example: "
                             "*What if lead time is 14 days?*")
            res = call("run_what_if", lead_time_days=float(days.group(1)))
            return reply(self._what_if_text(res, f"lead time of {days.group(1)} days"))

        if product:
            res = call("get_product_details", product_id=product)
            if "error" in res:
                return reply(res["error"])
            steps = res["calculation_steps"]
            return reply(
                f"**{res['name']}** ({res['product_type']}, {res['colour']}): "
                f"**{res['status']}**\n\n"
                f"- Forecast demand: {res['daily_demand_forecast']} units/day "
                f"({res['sma_window_days']}-day SMA)\n"
                f"- Current stock: {res['current_stock']} units, about "
                f"{res['days_of_cover']} days of cover vs a {res['lead_time_days']:g}-day lead time\n"
                f"- Reorder point: {steps['reorder_point']}\n"
                f"- EOQ: {steps['eoq']}\n"
                f"- Decision: {steps['decision_rule']}\n"
                f"- Recommended order: **{res['recommended_order_units']} units**"
            )

        if any(w in q for w in ["priorit", "urgent", "first", "most critical"]):
            res = call("prioritize_reorders", limit=5)
            lines = "\n".join(
                f"{r['rank']}. **{r['name']}**: {r['days_of_cover']} days of cover, "
                f"{r['units_below_reorder_point']} units below ROP, order {r['recommended_order_units']}"
                for r in res["ranked"]
            )
            return reply(f"**Top reorder priorities** ({res['total_needing_reorder']} need reordering):\n\n"
                         f"{lines}\n\n*{res['ranking_rule']}*")

        if any(w in q for w in ["in stock", "sufficient", "enough stock", "well stocked"]):
            res = call("list_products", status="sufficient")
            return reply(f"**{res['count']} products have sufficient stock:**\n\n"
                         + _fmt_products(res["products"], show_order=False))

        if any(w in q for w in ["reorder", "restock", "replenish", "order", "low stock"]):
            res = call("list_products", status="reorder")
            return reply(f"**{res['count']} products need reordering**, "
                         f"{res['total_recommended_order_units']:,} units in total:\n\n"
                         + _fmt_products(res["products"]))

        if any(w in q for w in ["assumption", "cost", "lead time", "how do you calculate", "formula"]):
            res = call("get_assumptions")
            return reply(
                f"**Current assumptions**\n\n- Ordering cost S: ${res['ordering_cost']:g} per order\n"
                f"- Holding cost H: ${res['holding_cost']:g} per unit per year\n"
                f"- Lead time L: {res['lead_time_days']:g} days\n"
                f"- Demand change: {res['demand_change_pct']:+g}%\n"
                f"- Safety stock service level: {res['service_level']:g}%\n\n{res['notes']}")

        if any(w in q for w in ["summary", "overview", "status", "inventory", "stock"]):
            s = call("get_inventory_summary")["summary"]
            return reply(f"**Inventory summary**\n\n- Products: {s['products']}\n"
                         f"- Need reorder: {s['need_reorder']}\n- Sufficient stock: {s['sufficient_stock']}\n"
                         f"- Units to order: {s['recommended_order_units']:,}\n"
                         f"- Units on hand: {s['current_units_on_hand']:,}")

        return reply(
            "Offline mode only understands simple questions. Try:\n\n"
            "- Which products need to be reordered?\n- Which products should I prioritize?\n"
            "- Why does Lucy blouse need replenishment?\n- What if demand increases by 20%?\n"
            "- What if lead time is 14 days?\n- What products are in stock?")

    @staticmethod
    def _mentioned_product(q: str, tools: InventoryTools) -> str | None:
        """Find a product named in the question: an article ID, the full
        product name, or its first word plus at least one more word of it."""
        id_match = _ARTICLE_ID.search(q)
        if id_match:
            return id_match.group(0)
        words_in_q = set(re.findall(r"[a-z0-9]+", q))
        best = []
        for name in tools.base["Name"]:
            lname = name.lower()
            if lname in q:
                return name
            words = re.findall(r"[a-z0-9]+", lname)
            if words and words[0] in words_in_q and len(set(words) & words_in_q) >= 2:
                best.append(name)
        return best[0] if len(best) == 1 else None

    @staticmethod
    def _what_if_text(res: dict, label: str) -> str:
        if "error" in res:
            return res["error"]
        b, a = res["summary_before"], res["summary_after"]
        new = ", ".join(res["products_that_newly_need_reorder"]) or "none"
        gone = ", ".join(res["products_no_longer_needing_reorder"]) or "none"
        return (f"**Scenario: {label}** (dashboard not changed)\n\n"
                f"- Products needing reorder: {b['need_reorder']} → {a['need_reorder']}\n"
                f"- Units to order: {b['recommended_order_units']:,} → {a['recommended_order_units']:,}\n"
                f"- Newly need reorder: {new}\n- No longer need reorder: {gone}\n"
                f"- Products whose ROP or EOQ changed: {len(res['changed_products'])}")

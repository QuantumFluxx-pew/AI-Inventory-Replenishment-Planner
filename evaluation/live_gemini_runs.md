# Live AI evaluation: repeated runs

**Date:** November 19, 2026 · **App:** live Streamlit deployment · **Model:** gemini-3.5-flash-lite · **Settings:** defaults (S = $50, H = $2, lead time 7 days, 7-day moving average)

The six demo questions were asked **three times each** (18 answers in total). Before each run the conversation was cleared and the settings were reset to the defaults.

## Summary

| Measure | Result |
| --- | --- |
| Answers with the expected behaviour | **18 of 18** |
| Answers containing numbers | 15 |
| Numbers checked against the Python model | 82 |
| Numbers the AI invented (not from the model) | **0** |
| Asked for the missing percentage (step 2) | 3 of 3 |
| Rejected the invalid lead time of -5 days (step 3) | 3 of 3 |
| Dashboard updated after "change lead time to 14 days" (step 6) | 3 of 3 |

## Results per question

| Step | Question | Expected | Run 1 | Run 2 | Run 3 |
| --- | --- | --- | --- | --- | --- |
| 1. Objective | Which products should I prioritize? | Harrison top, Milk RW slack, Lucy blouse (704, 474, 891 units) | Match (6 numbers checked) | Match (9) | Match (9) |
| 2. Missing input | What happens if demand increases? | Asks by what percentage | Asked | Asked | Asked |
| 3. Invalid input | Set lead time to -5 days | Rejects, asks for a value above 0 | Rejected (1) | Rejected (1) | Rejected (2) |
| 4. Run model | What if demand increases by 20%? | 8 → 9 products, 5,048 → 5,962 units | Match (8) | Match, but left out the unit total (4) | Match (5) |
| 5. Explain | Why does Brit Baby Tee need replenishment? | Stock 65 ≤ ROP 65, 9.29 units/day × 7 days | Match (7) | Match (6) | Match, with EOQ steps (12) |
| 6. Revise input | Change lead time to 14 days | All 18 products, 11,131 units | Match (4) | Match (4) | Match (4) |

Numbers in brackets: how many numbers in the answer were checked against the model's results by the app's automatic number check. All of them matched in every answer.

## Observations

* **Numbers were always correct.** The wording changed between runs, but every number came from the Python tools.
* **Length and detail vary.** In run 2, the answer to step 4 gave the number of products and the cost but not the total units (5,962). It was correct, just less complete. In run 3, step 5 added the EOQ calculation without being asked.
* **Step 2 sometimes calls a tool before asking.** In runs 2 and 3 the agent looked at the assumptions first, then asked for the percentage. It still asked instead of guessing.
* **Earlier in the same session** the Gemini service once returned a "503 high demand" error. The app now retries busy requests automatically (two retries) before falling back to offline mode.

## Also tested live (same day)

* **Uploaded data:** the example files in `data/example_upload/` (5 products, 120 days, with stock) were uploaded in Settings. The dashboard showed 3 products to reorder, 2,007 units and $7,172 annual cost, the same orders as the H&M sample for those products. Stock Pilot ranked them correctly (Harrison top 704, Lucy blouse 891, Brit Baby Tee 412), with all 6 numbers checked.
* **Second forecast model:** asked "Which forecast model is more accurate for the Harrison top?", Stock Pilot answered exponential smoothing, with errors of 7.97 vs 9.26 units/day, matching the model.

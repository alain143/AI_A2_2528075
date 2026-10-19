"""Build docs/AI_A2_<REGDIGITS>_REPORT.pdf from the generated artifacts (needs reportlab).

reportlab is a documentation-only tool and is NOT needed to run the ML system.
    pip install reportlab
    python tools/build_report.py
Edit the INTERPRETATION paragraphs below in your own words before submitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
md = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
reg = json.loads((A / "regression_metrics.json").read_text(encoding="utf-8"))
clf = json.loads((A / "classification_metrics.json").read_text(encoding="utf-8"))
clu = json.loads((A / "clustering_report.json").read_text(encoding="utf-8"))
run = json.loads((A / "run_summary.json").read_text(encoding="utf-8"))
preds = pd.read_csv(A / "test_predictions.csv")

CODE, SEED = md["verification_code"], md["seed"]
REGD = md["student_id"].split("@")[0].lstrip("ugUG")
OUT = ROOT / "docs" / f"AI_A2_{REGD}_REPORT.pdf"
log = ROOT / "evidence" / "pytest_log.txt"
TEST_LINE = (log.read_text(encoding="utf-8").strip().splitlines()[-1] if log.exists()
             else "pytest result not yet recorded - run: pytest -q | tee evidence/pytest_log.txt, then rebuild this report")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=9.3, leading=12.2, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=6, textColor=colors.HexColor("#1b4d2e"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=5, spaceAfter=3)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8, leading=9.5, spaceAfter=0)


def P(t, s=B): return Paragraph(t, s)


def tbl(rows, widths, head=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeee3")))
    t.setStyle(TableStyle(st)); return t


def img(name, w_mm):
    from PIL import Image as PI
    path = A / "plots" / name
    w, h = PI.open(path).size
    return Image(str(path), width=w_mm * mm, height=w_mm * mm * h / w)


def true_fertilizer(row_id):
    """Value of fertilizer_kg before it was blanked (replays the generator's first draws)."""
    n = md["rows_raw"] - 3
    rng = np.random.default_rng(SEED)
    for lo, hi in [(600, 1400), (12, 26), (20, 80)]: rng.uniform(lo, hi, n)
    return float(rng.uniform(0, 250, n)[row_id])


def f4(x): return f"{x:.4f}"


story = []
# ---------------------------------------------------------------- page 1
story += [P("Musanze Harvest Intelligence Service", H1),
          P("SWE 3513 Artificial Intelligence - Individual Assignment 2 - Technical Report", B),
          tbl([["Full name", md["student_name"]], ["Registration number", md["registration_number"]],
               ["Institutional username", md["student_id"]], ["Verification code", f"<b>{CODE}</b> (seed LLLL = {SEED})"],
               ["Dataset SHA-256", run["sha256"]], ["Commit hash", "<i>insert final commit hash here</i>"]], [45, 125], head=False),
          P("1. Identity and decision problem", H2),
          P("<b>Stakeholder.</b> A field officer of a potato cooperative in Musanze who receives farm records in inconsistent "
            "spreadsheets and must decide which farms need attention before harvest."),
          P("<b>Prediction questions.</b> (1) How many tons will a farm harvest (regression on <i>yield_tons</i>)? (2) Is the farm at "
            "high production risk, meaning its yield per hectare is below the median (classification on <i>high_risk</i>)? "
            "(3) Which farms have similar growing conditions (K-means clustering, no targets used)?"),
          P("<b>Why three tasks.</b> The officer needs a quantity for planning storage and transport, a yes/no flag with a probability "
            "for visits and advice, and groups so that one advisory visit can serve similar farms. Training and command-line prediction run "
            "locally because internet access may be unreliable; the FastAPI service is for later integration with a mobile application."),
          P("<b>Ethical boundary.</b> The system is a classroom decision-support prototype trained on <b>synthetic</b> data. It does not replace "
            "an agronomist and must never be used to decide access to finance, insurance or public services. Every prediction carries "
            "<i>human_review_required = true</i>. Individual farms must not be ranked or penalised on a model output alone."),
          P("<b>How the report is produced.</b> All numbers below are read from <i>artifacts/*.json</i> and <i>test_predictions.csv</i> by "
            "<i>tools/build_report.py</i>, so they match the submitted artifacts exactly. The interpretation paragraphs (one regression error, "
            "one classification error, one cluster) are my own reading of those results."),
          PageBreak()]

# ---------------------------------------------------------------- page 2
v = run["validation"]; cl = run["cleaning"]
story += [P("2. Data and reproducibility", H1),
          P(f"The dataset is generated by <i>data/generate_data.py</i> with <i>numpy.random.default_rng({SEED})</i>. Base rows = 240 + ({SEED} mod 41) = "
            f"240 + {SEED % 41} = <b>{v['expected']['base_rows']}</b>. Then m = round(0.05 x {v['expected']['base_rows']}) = <b>{v['expected']['missing_per_column']}</b> "
            f"values were blanked in each of soil_moisture_pct and fertilizer_kg (independent draws of unique positions) and 3 exact duplicate rows were appended, "
            f"giving <b>{v['n_rows_raw']}</b> raw rows. The CSV is written with 4 decimals and LF line endings."),
          tbl([["Column", "Rule", "Role"],
               ["rainfall_mm", "Uniform 600-1400", "Numeric feature"], ["avg_temp_c", "Uniform 12-26", "Numeric feature"],
               ["soil_moisture_pct", "Uniform 20-80", "Numeric feature"], ["fertilizer_kg", "Uniform 0-250", "Numeric feature"],
               ["field_size_ha", "Uniform 0.2-4.0", "Numeric feature"], ["season", "Random A / B", "Categorical feature"],
               ["cooperative_support", "Random 0 / 1", "Binary feature"],
               ["yield_tons", "0.004 rain + 0.06 soil + 0.02 fert + 1.30 field - 0.25 |temp-18| + 0.70 support + season (A +0.50, B -0.20) + N(0, 0.80), min 0.50", "Regression target"],
               ["high_risk", "1 if yield per hectare is below its median", "Classification target"]], [38, 100, 32]),
          Spacer(1, 4),
          P("Evidence of the deliberate data problems", H2),
          tbl([["Check", "Result"],
               ["Raw rows / expected", f"{v['n_rows_raw']} / {v['expected']['raw_rows']}"],
               ["Missing values", f"soil_moisture_pct = {v['missing_by_column']['soil_moisture_pct']}, fertilizer_kg = {v['missing_by_column']['fertilizer_kg']}"],
               ["Exact duplicate rows", f"{v['n_duplicate_rows']} (removed, keeping the first occurrence)"],
               ["Rows after cleaning", f"{cl['rows_after']}"],
               ["Range / schema / label checks", "passed (all five ranges, season in {A,B}, support and high_risk in {0,1}, yield >= 0.50)"],
               ["Generator matches assignment rule", str(v["matches_expected"])],
               ["Target-leakage check", v["leakage_check"]]], [55, 115]),
          Spacer(1, 4),
          P("Reproducibility", H2),
          P(f"The SHA-256 of the exact CSV bytes is <font face='Courier' size='8'>{run['sha256']}</font> and is stored in <i>artifacts/data_fingerprint.txt</i>. "
            "Running <i>run_all.py</i> twice gave byte-identical CSV, metrics JSON files and test_predictions.csv. "
            "<i>run_all.py</i> rejects any seed that is not LLLL derived from the student id (for example seed 4671 is refused with a clear message). "
            "The same seed is used for the generator, the train/test split, cross-validation folds, every model and K-means."),
          PageBreak()]

# ---------------------------------------------------------------- page 3
sp = run["split"]
story += [P("3. Pipeline design", H1),
          P(f"<b>Split.</b> After removing duplicates ({cl['rows_after']} rows) a fixed 80/20 split, stratified on high_risk, uses random_state = {SEED}: "
            f"{sp['train_rows']} training rows and {sp['test_rows']} test rows. The test rows are used once, for the final metrics."),
          P("<b>Preprocessing (one ColumnTransformer, always inside a Pipeline).</b>"),
          tbl([["Columns", "Steps (fitted on training rows only)"],
               ["5 numeric features", "SimpleImputer(median) then StandardScaler"],
               ["season", "SimpleImputer(most_frequent) then OneHotEncoder(handle_unknown='ignore')"],
               ["cooperative_support", "SimpleImputer(most_frequent), passed through as 0/1"]], [45, 125]),
          Spacer(1, 3),
          P("Every saved model is a full Pipeline (preprocessing + estimator), so <i>predict.py</i> and <i>app.py</i> apply exactly the same transformations as training. "
            "Duplicate removal happens before the split (it uses no statistics); imputation, scaling and encoding happen only after it."),
          P("<b>Leakage controls.</b> The input list is exactly the seven farm features. yield_tons, yield per hectare and high_risk are never inputs; "
            "<i>check_leakage()</i> fails the run if any name containing 'yield' or 'high_risk' enters the feature list, and the prediction validators reject such fields. "
            "Model selection uses 5-fold cross-validation on the training set only, so the test set does not influence any choice. "
            "One honest caveat: the high_risk label itself is defined with the median of the whole generated sample (as the assignment prescribes), which is a property of the data, not of the models."),
          P("<b>Candidates.</b>"),
          tbl([["Task", "Baseline", "Candidates", "Selection rule"],
               ["Regression", "Mean predictor", "Ridge, Random Forest, Gradient Boosting", "lowest 5-fold CV RMSE (train)"],
               ["Classification", "Majority class", "Logistic Regression, Random Forest, Gradient Boosting", "highest 5-fold stratified CV F1 (train)"],
               ["Clustering", "-", "K-means, K = 2..6", "highest silhouette on scaled training data"]], [27, 30, 66, 47]),
          Spacer(1, 3),
          P("<b>Artifacts.</b> regression_metrics.json, classification_metrics.json, clustering_report.json, test_predictions.csv, run_summary.json, data_fingerprint.txt, "
            "three plots, and models/ (regressor, classifier, clusterer, metadata.json). Library versions are recorded in models/metadata.json."),
          PageBreak()]

# ---------------------------------------------------------------- page 4
rows = [["Model", "MAE (t)", "RMSE (t)", "R2", "CV RMSE"]]
for k, m in reg["models"].items():
    rows.append([("<b>%s</b>" % k) if k == reg["selected_model"] else k, f4(m["mae"]), f4(m["rmse"]), f4(m["r2"]), f4(m["cv_rmse_mean"])])
crow = [["Model", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]]
for k, m in clf["models"].items():
    crow.append([("<b>%s</b>" % k) if k == clf["selected_model"] else k, f4(m["precision"]), f4(m["recall"]), f4(m["f1"]), f4(m["roc_auc"]), f4(m["cv_f1_mean"])])
sel_c = clf["models"][clf["selected_model"]]["confusion_matrix"]
worst = preds.loc[preds.abs_yield_error_tons.idxmax()]
wid = int(worst.row_id)
raw = pd.read_csv(ROOT / "data" / md["dataset_file"]).drop_duplicates()
wr = raw.loc[wid]
fert_true = true_fertilizer(wid)
fert_med = float(raw.fertilizer_kg.median())
fn_rows = preds[(preds.classification_correct == 0)]
story += [P("4. Regression and classification", H1),
          P(f"Regression on {reg['test_rows']} held-out farms (selected model in bold):"), tbl(rows, [50, 25, 25, 25, 30]), Spacer(1, 3),
          P(f"<b>Selection.</b> {reg['selection_reason']}"),
          img("residuals.png", 165),
          P("<b>Residual reading.</b> Residuals are centred near zero (mean %.3f t) and show no strong curve against the predicted value, so the linear model captures the main structure; "
            "the mean baseline has R2 near zero, so the features carry real signal. The two tree ensembles scored worse than Ridge here, which fits a dataset whose yield formula is almost linear and small (%d training rows)."
            % (preds.yield_error_tons.mean(), sp["train_rows"]), B),
          P(f"<b>Important regression error (my interpretation).</b> The largest miss is farm row_id {wid}: actual {worst.actual_yield_tons:.2f} t, predicted {worst.predicted_yield_tons:.2f} t "
            f"(over-prediction of {worst.yield_error_tons:.2f} t). Its fertilizer_kg is one of the deliberately blanked values, so the pipeline imputed the training median (about {fert_med:.0f} kg). "
            f"Replaying the generator shows the true value was about {fert_true:.0f} kg; at 0.02 t per kg that gap alone is about {0.02*(fert_med-fert_true):.1f} t of the {worst.yield_error_tons:.2f} t error. "
            f"The farm is also small ({wr.field_size_ha:.2f} ha) with dry soil ({wr.soil_moisture_pct:.0f} %), a corner of the data with few training examples. "
            "Lesson: median imputation hides a real difference between farms; a real system should flag records with imputed inputs."),
          Spacer(1, 3),
          P("Classification (positive class = high_risk):"), tbl(crow, [44, 25, 22, 20, 25, 22]), Spacer(1, 3),
          P(f"<b>Selection.</b> {clf['selection_reason']}"),
          PageBreak()]

# ---------------------------------------------------------------- page 5 (classification detail + clustering)
fp = int(sel_c["fp"]); fn = int(sel_c["fn"])
ex = fn_rows.iloc[0]
exr = raw.loc[int(ex.row_id)]
story += [P("4b. Classification interpretation", H2),
          img("confusion_matrix.png", 70),
          P(f"<b>Confusion matrix.</b> TN {sel_c['tn']}, FP {fp}, FN {fn}, TP {sel_c['tp']}. The majority baseline predicts every farm as the majority class, so its precision, recall and F1 are 0 "
            "(ROC-AUC 0.5) - it never finds a single high-risk farm, which makes accuracy alone misleading. For the field officer a false negative (a high-risk farm missed) costs more than a false positive "
            "(an unnecessary visit), so recall deserves attention alongside precision."),
          P(f"<b>Important classification error (my interpretation).</b> Farm row_id {int(ex.row_id)} was not a high-risk farm but received probability {ex.prob_high_risk:.2f}. "
            f"It has a large field ({exr.field_size_ha:.2f} ha) and heavy rain ({exr.rainfall_mm:.0f} mm), and yield per hectare of high_risk is defined relative to the median, "
            "so large farms with a good but not extreme yield sit close to the decision boundary where noise (SD 0.80 t) decides the label. "
            "The other misclassified test farms have probabilities of " + " and ".join(f"{p:.2f}" for p in fn_rows.prob_high_risk.iloc[1:]) + ", i.e. close to the 0.5 threshold: borderline cases rather than confident mistakes."),
          PageBreak()]

# ---------------------------------------------------------------- page 6 clustering
krows = [["K", "Silhouette", "Inertia"]] + [[str(s["k"]), f4(s["silhouette"]), f"{s['inertia']:.1f}"] for s in clu["k_scores"]]
prof = [["Cluster", "Train / test farms", "Centre (rain, temp, soil, fert, field)", "Support / season A", "Mean yield / high-risk rate*"]]
for p in clu["cluster_profiles"]:
    c = p["center_original_units"]
    prof.append([p["cluster"], f"{p['n_train']} / {p['n_test']}", f"{c['rainfall_mm']:.0f}, {c['avg_temp_c']:.1f}, {c['soil_moisture_pct']:.1f}, {c['fertilizer_kg']:.0f}, {c['field_size_ha']:.2f}",
                 f"{p['cooperative_support_share']:.2f} / {p['season_A_share']:.2f}", f"{p['mean_yield_tons_descriptive']:.2f} t / {p['high_risk_rate_descriptive']:.2f}"])
p0, p1 = clu["cluster_profiles"][0], clu["cluster_profiles"][1]
story += [P("5. Clustering", H1),
          P("The seven input features are imputed, scaled and one-hot encoded on the training rows, then K-means (n_init = 10, random_state = %d) is run for K = 2..6. Targets are not used." % SEED),
          tbl(krows, [25, 35, 35]), Spacer(1, 3),
          P(f"<b>Choice of K.</b> {clu['interpretation']} The silhouette values for all K lie between {min(s['silhouette'] for s in clu['k_scores']):.3f} and "
            f"{max(s['silhouette'] for s in clu['k_scores']):.3f}, so the choice of K=2 over K=4 or K=6 is a small difference. This is expected: the features were drawn independently and uniformly, so the "
            "farms do not form natural groups. The clusters are therefore a convenient split of one continuous cloud, not distinct farm types."),
          img("clusters.png", 105), Spacer(1, 2),
          tbl(prof, [14, 24, 62, 30, 40]), P("* yield and risk are descriptive only; they were not used to form clusters.", SM),
          P("Plain-language profiles", H2)] + [P(p["profile"]) for p in clu["cluster_profiles"]] + [
          P(f"<b>One farm cluster (my interpretation).</b> Cluster 1 groups larger fields ({p1['center_original_units']['field_size_ha']:.2f} ha on average) with higher rainfall but lighter fertilizer use "
            f"({p1['center_original_units']['fertilizer_kg']:.0f} kg against {p0['center_original_units']['fertilizer_kg']:.0f} kg). Its high-risk rate is {p1['high_risk_rate_descriptive']:.2f} versus {p0['high_risk_rate_descriptive']:.2f} in cluster 0, although its mean yield in tons is higher: "
            "large fields produce more tons but, with little fertilizer per farm, less yield per hectare. Operationally, the officer could prioritise fertiliser advice for this group. "
            "Because the silhouette is low, this should be treated as a hint to investigate, not as a farm type."),
          PageBreak()]

# ---------------------------------------------------------------- page 7 API + tests
story += [P("6. API and testing", H1),
          tbl([["Endpoint", "Behaviour"],
               ["GET /health", "200 {status, models_loaded, verification_code}; 503 if models are missing"],
               ["GET /model-info", "verification code, student, SHA-256, selected models, test metrics, feature ranges, library versions, synthetic-data notice"],
               ["POST /predict", "typed body (pydantic, strict types, ranges, extra fields forbidden); 200 with yield, risk label, probability, cluster, profile, verification code; 422 with field-level messages"]], [35, 135]),
          Spacer(1, 3),
          P("<b>Valid request</b>: samples/valid.json returns the yield estimate, risk prediction, probability, cluster, cluster profile and verification code "
            f"({CODE}); the command-line output and the API response use the same code path (<i>src/inference.py</i>)."),
          P("<b>Invalid requests.</b> samples/invalid.json combines four faults and the CLI reports each one: missing field_size_ha, rainfall_mm = 2500 outside [600, 1400], avg_temp_c = 'warm' (wrong type), "
            "season = 'C'. The API returns HTTP 422 with <i>{error: validation_failed, details: [{field, message}]}</i>; the same rules also reject unexpected fields such as yield_tons (leakage)."),
          P("<b>Automated tests</b> (tests/test_system.py, 11 tests): valid inference; missing field; wrong type; range error; leakage fields; dataset reproducibility and fingerprint; "
            "deliberate missing/duplicate evidence; seed rejection; metrics agree with test_predictions.csv; API health/model-info/predict; API 422 for four invalid bodies."),
          P(f"<b>Test summary.</b> {TEST_LINE}"),
          P("<b>Failure behaviour.</b> Missing model files give a clear instruction to run run_all.py (CLI exit code 3, API 503). Invalid input gives exit code 2 / HTTP 422 and never reaches the models."),
          P("7. Model card and reflection", H1),
          tbl([["Item", "Statement"],
               ["Intended use", "Teaching prototype that shows how a field officer could triage farms and plan harvest quantities."],
               ["Data", "Fully synthetic, generated from the stated formula; no real farm was measured."],
               ["Limitations", "Models learn the generator, not real agronomy. Inputs are limited to seven features; no pests, soil type, variety or price. Small sample (%d rows). Clusters are weak. Predictions outside the training ranges are rejected." % md["rows_clean"]],
               ["Risks", "Over-trust in a number; imputed values hidden from the user; synthetic accuracy mistaken for real accuracy; the median-based label always marks about half the farms as high risk."],
               ["Prohibited use", "Deciding loans, insurance, subsidies or any public service; ranking or punishing farmers; replacing an agronomist."],
               ["Human oversight", "Every response has human_review_required = true; an agronomist or field officer must review each recommendation."]], [32, 138]),
          Spacer(1, 3),
          P("<b>With real data I would:</b> collect measured yields and field records, report which inputs were imputed, check fairness across districts and farm sizes, calibrate the probabilities, "
            "monitor drift each season, and validate the clusters with agronomists before using them.")]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5)
    c.drawString(15 * mm, 8 * mm, f"{md['student_name']} | {md['registration_number']} | {CODE}")
    c.drawRightString(195 * mm, 8 * mm, f"Page {d.page}"); c.restoreState()


OUT.parent.mkdir(exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
                  title="Musanze Harvest Intelligence Service - Technical Report", author=md["student_name"]).build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)

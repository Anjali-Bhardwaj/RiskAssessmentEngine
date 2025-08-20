import os
import math
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import RootModel
from typing import Any, Dict

# ---------------------------
# Load rulepack
# ---------------------------
RULEPACK_PATH = os.getenv("RULEPACK_PATH", "rulepack.yaml")

def load_rulepack() -> Dict[str, Any]:
    if not os.path.exists(RULEPACK_PATH):
        raise FileNotFoundError(f"Rulepack not found at {RULEPACK_PATH}")
    with open(RULEPACK_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "rulepack" not in data:
        raise ValueError("Invalid rulepack: missing 'rulepack' root")
    return data["rulepack"]

RULEPACK = load_rulepack()

# ---------------------------
# Helpers
# ---------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def safe_get(d: Dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

def lower(s: Optional[str]) -> str:
    return (s or "").lower()

def any_in(collection: List[str], targets: List[str]) -> Optional[str]:
    s = set([c.lower() for c in collection or []])
    for t in targets or []:
        if t.lower() in s:
            return t
    return None

def exists_alignment_mismatch(dvds: List[Dict[str, Any]], types: List[str]) -> bool:
    for item in dvds or []:
        if item.get("source_type") in types and item.get("alignment") in ["partial", "mismatch", "missing"]:
            return True
    return False

def collect_alignment_points(dvds: List[Dict[str, Any]], mapping: Dict[str, int], cap: int) -> int:
    pts = 0
    for item in dvds or []:
        pts += mapping.get(item.get("alignment", "aligned"), 0)
        if pts >= cap:
            return cap
    return min(pts, cap)

def offshore_hint(products: List[str], keywords: List[str]) -> bool:
    p = ", ".join(products or [])
    p_l = lower(p)
    return any(k.lower() in p_l for k in keywords or [])

def cross_border(source_countries: List[str], payment_countries: List[str]) -> bool:
    # Any non-UAE flow
    def non_uae(arr):
        return any(lower(c) != "uae" for c in (arr or []))
    return non_uae(source_countries) or non_uae(payment_countries)

def collect_top_reasons(dimensions: Dict[str, Dict[str, Any]], limit: int = 3) -> str:
    # naïve selection: pick the three highest scores' reasons
    scored = []
    for k, v in dimensions.items():
        scored.append((v.get("score", 0), v.get("reason", "")))
    scored.sort(key=lambda x: x[0], reverse=True)
    reasons = [r for _, r in scored if r][:limit]
    return "; ".join(reasons) if reasons else "Multiple factors"

# ---------------------------
# Input model (lightweight)
# ---------------------------
class EvaluateRequest(RootModel[Dict[str, Any]]):
    pass

# ---------------------------
# Core evaluator
# ---------------------------
def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    rp = RULEPACK
    case_id = payload.get("case_id", "UNKNOWN")

    # Bind thresholds
    suff_min = payload.get("thresholds", {}).get("sufficiency_min",
                 rp["thresholds"]["sufficiency_min_default"])
    edd_cut = payload.get("thresholds", {}).get("edd_cutoff_score",
                 rp["thresholds"]["edd_cutoff_score_default"])
    bands = rp["thresholds"]["bands"]

    # Short-hands from input
    screening = payload.get("screening_summary", {}) or {}
    sanctions = screening.get("sanctions", {}) or {}
    pep = screening.get("pep", {}) or {}
    adverse_media = screening.get("adverse_media", {}) or {"severity": "none", "hits": 0}

    risk_features = payload.get("risk_features", {}) or {}
    geo = risk_features.get("geo", {}) or {}
    product = risk_features.get("product", {}) or {}
    channel = risk_features.get("channel", {}) or {}

    dvds = payload.get("declared_vs_discovered", []) or []
    issues = payload.get("issues", []) or []
    evidence_sources = payload.get("evidence_sources", []) or []
    suff_after = payload.get("sufficiency_after_hitl", 0.0)
    pattern_tag = payload.get("pattern_tag", "Other")

    # ---------- Dimension: GEO ----------
    geo_cfg = rp["dimensions"]["geo_risk"]
    base_map = geo_cfg["score_map"]["base"]
    base_score = base_map.get(geo.get("label", "low"), 5)

    fatf_conf = geo_cfg["modifiers"]["fatf_exposure"]
    grey_list = fatf_conf["list"]["grey"]
    black_list = fatf_conf["list"]["black"]
    add_if_in = fatf_conf["add_if_in"]

    pay_countries = geo.get("payment_countries", []) or []
    match_grey = any_in(pay_countries, grey_list)
    match_black = any_in(pay_countries, black_list)

    geo_bonus = 0
    reason_geo = "Geo exposure aligns with declared residency and sources."
    label_geo = geo.get("label", "low")

    if match_black:
        geo_bonus += add_if_in["black"]
        reason_geo = f"Payments involve FATF blacklisted jurisdiction ({match_black})."
    elif match_grey:
        geo_bonus += add_if_in["grey"]
        cc = geo.get("customer_country", "UAE")
        reason_geo = f"Customer residency is {cc} (neutral), but remittance to FATF-greylisted jurisdiction noted ({match_grey})."

    dim_geo = {
        "score": base_score + geo_bonus,
        "label": label_geo,
        "reason": reason_geo
    }

    # ---------- Dimension: PEP/Sanctions ----------
    ps_cfg = rp["dimensions"]["pep_sanctions"]["rules"]
    hard_edd = False
    ps_score, ps_label, ps_reason = 0, "low", "No PEP match. No sanctions alerts from screening summary."
    def status_to_case(s: Dict[str, Any]) -> str:
        return s.get("status", "clear")

    sanc = status_to_case(sanctions)
    peps = status_to_case(pep)

    # Priority: match > possible > clear
    if "match" in (sanc, peps):
        rule = next(r for r in ps_cfg if r["when"] == "match")
        ps_score, ps_label, ps_reason = rule["score"], rule["label"], rule["reason"]
        hard_edd = (rule.get("hard_route") == "EDD")
    elif "possible" in (sanc, peps):
        rule = next(r for r in ps_cfg if r["when"] == "possible")
        ps_score, ps_label, ps_reason = rule["score"], rule["label"], rule["reason"]
    else:
        rule = next(r for r in ps_cfg if r["when"] == "clear")
        ps_score, ps_label, ps_reason = rule["score"], rule["label"], rule["reason"]

    dim_ps = {"score": ps_score, "label": ps_label, "reason": ps_reason, "hard_route": "EDD" if hard_edd else None}

    # ---------- Dimension: Adverse media ----------
    am_cfg = rp["dimensions"]["adverse_media"]
    sev = adverse_media.get("severity", "none")
    sev_to_score = am_cfg["mapping"]["severity_to_score"]
    sev_to_label = am_cfg["mapping"]["severity_to_label"]
    am_score = sev_to_score.get(sev, 0)
    am_label = sev_to_label.get(sev, "low")
    if sev == "none":
        am_reason = "No negative media hits in top-tier news sources."
    else:
        hits = adverse_media.get("hits", 0)
        am_reason = f"Adverse media severity {sev} ({hits} hits)."
    dim_am = {"score": am_score, "label": am_label, "reason": am_reason}

    # ---------- Detectives for pattern risk ----------
    pat_cfg = rp["dimensions"]["pattern_risk"]
    base_pattern = pat_cfg["base_by_pattern"].get(pattern_tag, pat_cfg["base_by_pattern"]["Other"])
    mod_cfg = pat_cfg["inconsistency_modifiers"]

    # salaried_high_cash
    salaried_high_cash = pattern_tag in ["WPS_Salary", "Salary_Plus_Property"] and channel.get("cash_intensity") == "high"
    # rental_count_mismatch
    rental_count_mismatch = exists_alignment_mismatch(dvds, ["Property Holdings", "Rental Income"])
    # dividend_missing_proof
    dividend_missing = exists_alignment_mismatch(dvds, ["Dividend"])

    pr_score = base_pattern
    pr_reasons = []
    if salaried_high_cash:
        pr_score += mod_cfg["salaried_high_cash"]
        pr_reasons.append("Large cash deposits inconsistent with declared salaried profile. Potential layering risk.")
    if rental_count_mismatch:
        pr_score += mod_cfg["rental_count_mismatch"]
        pr_reasons.append("Mismatch in declared vs discovered rental properties.")
    if dividend_missing:
        pr_score += mod_cfg["dividend_missing_proof"]
        pr_reasons.append("Dividend income declared without sufficient proof.")
    if not pr_reasons:
        pr_reasons.append(f"Pattern risk evaluated for {pattern_tag}.")

    pr_label = "low" if pr_score <= 9 else "medium" if pr_score <= 19 else "high"
    dim_pr = {"score": pr_score, "label": pr_label, "reason": "; ".join(pr_reasons)}

    # ---------- Evidence gaps ----------
    eg_cfg = rp["dimensions"]["evidence_gaps"]["scoring"]
    base_gap = eg_cfg["base_from_sufficiency"]["when_low"] if suff_after < suff_min else eg_cfg["base_from_sufficiency"]["when_ok"]

    # per-issue points (cap)
    impact_map = eg_cfg["per_issue_points"]
    per_issue = 0
    for it in issues:
        per_issue += impact_map.get(it.get("residual_impact", "none"), 0)
    per_issue = min(per_issue, eg_cfg["max_issue_points"])

    # alignment points from dvds (cap)
    align_map = eg_cfg["partial_alignment_points"]
    align_pts = collect_alignment_points(dvds, align_map, align_map["max_alignment_points"])

    eg_score = base_gap + per_issue + align_pts
    eg_label = "low" if eg_score <= 7 else "medium" if eg_score <= 15 else "high"

    eg_reason_bits = []
    if suff_after < suff_min:
        eg_reason_bits.append("Evidence sufficiency below policy threshold.")
    if any(i.get("type") == "MISSING_EVIDENCE" for i in issues):
        eg_reason_bits.append("Missing documentary proof for one or more declared sources.")
    if exists_alignment_mismatch(dvds, ["Property Holdings", "Dividend"]):
        eg_reason_bits.append("Declared properties/dividends not fully evidenced.")
    if not eg_reason_bits:
        eg_reason_bits.append("Residual gaps assessed post-HITL.")
    dim_eg = {"score": eg_score, "label": eg_label, "reason": "; ".join(eg_reason_bits)}

    # ---------- Product / channel ----------
    pc_cfg = rp["dimensions"]["product_channel"]
    base_pc = pc_cfg["base_by_label"].get(product.get("label", "low"), 5)
    ch_add = pc_cfg["channel_adders"].get(channel.get("onboarding_channel", "branch"), 0)

    sc = geo.get("source_countries", []) or []
    pc = geo.get("payment_countries", []) or []
    is_cross_border = cross_border(sc, pc)

    prod_list = product.get("products", []) or []
    off_hint = offshore_hint(prod_list, pc_cfg.get("offshore_keywords", []))

    xb_add = 0
    if is_cross_border:
        xb_add += pc_cfg["cross_border_adders"]["non_uae_flow"]
    if off_hint:
        xb_add += pc_cfg["cross_border_adders"]["offshore_investment_hint"]

    pc_score = base_pc + ch_add + xb_add
    pc_label = "low" if pc_score <= 7 else "medium" if pc_score <= 14 else "high"

    if off_hint:
        pc_reason = "Use of offshore investment platform detected; cross-border complexity."
    elif is_cross_border:
        pc_reason = "Cross-border product/channel usage detected."
    else:
        pc_reason = "Product/channel risk per profile."
    dim_pc = {"score": pc_score, "label": pc_label, "reason": pc_reason}

    # ---------- Aggregate ----------
    dimensions = {
        "geo_risk": dim_geo,
        "pep_sanctions": dim_ps,
        "adverse_media": dim_am,
        "pattern_risk": dim_pr,
        "evidence_gaps": dim_eg,
        "product_channel": dim_pc
    }
    total = sum(v["score"] for v in dimensions.values())

    if total >= bands["high"]:
        risk_label = "high"
    elif total >= bands["medium"]:
        risk_label = "medium"
    else:
        risk_label = "low"

    route = "Baseline"
    if dim_ps.get("hard_route") == "EDD":
        route = "EDD"
    elif suff_after < suff_min:
        route = "EDD"
    elif total >= edd_cut:
        route = "EDD"

    # ---------- Red flags ----------
    red_flags = []
    if salaried_high_cash:
        red_flags.append("Cash deposits unexplained")
    if rental_count_mismatch:
        red_flags.append("Mismatch in rental property count")
    if is_cross_border:
        red_flags.append("Cross-border investments without sufficient evidence")

    # ---------- Explanation ----------
    primary = collect_top_reasons(dimensions, 3)
    decision_explanation = f"Overall {risk_label} risk. {primary} push the case to {route}."

    # ---------- Confidence ----------
    avg_conf = avg([e.get("confidence", 0.0) for e in evidence_sources])
    model_confidence = clamp(avg_conf * (0.8 + 0.2 * suff_after), 0.0, 1.0)

    assessor = {
        "agent": "RiskAgent-UAE-v2.1",
        "mode": "auto",
        "human_reviewer": None
    }

    out = {
        "case_id": case_id,
        "risk_score": int(round(total)),
        "risk_label": risk_label,
        "route": route,
        "status": "assessed",
        "model_confidence": round(model_confidence, 2),
        "dimensions": dimensions,
        "red_flags": red_flags,
        "policy_refs": rp.get("policy_refs", []),
        "rulepack_version": rp.get("rulepack_version", "unknown"),
        "decision_explanation": decision_explanation,
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "assessor": assessor
    }
    return out

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="SoW Risk Rule Evaluator", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "rulepack": RULEPACK.get("rulepack_version", "unknown")}

@app.post("/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    try:
        payload = req.root          # <— RootModel in pydantic v2
        result = evaluate(payload)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reload")
def reload_rulepack():
    global RULEPACK
    RULEPACK = load_rulepack()
    return {"status": "reloaded", "rulepack": RULEPACK.get("rulepack_version", "unknown")}

if __name__ == "__main__":
   # uvicorn.run("risk_assesment_rule_engine:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")), reload=False)
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port)
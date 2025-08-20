# risk_engine_api_old.py
# ------------------------------------------------------------
# Risk Assessment Rule Engine (FastAPI)
# - Accepts SoW-agent style inputs (documents + raw texts + client profile)
# - Applies deterministic rules to assign a risk score, risk band, and EDD need
# - Produces per-document findings and an overall summary
#
# Run:
#   pip install fastapi uvicorn pydantic
#   uvicorn risk_engine_api:app --reload
#
# Test (curl):
#   curl -X POST http://127.0.0.1:8000/assess -H "Content-Type: application/json" -d @sample_payload.json
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
import re

# -----------------------------
# Data Models
# -----------------------------

class ExtractedData(BaseModel):
    anomalies: Optional[List[str]] = None
    extraction_errors: Optional[List[str]] = None

class DocumentItem(BaseModel):
    url: str
    type: str = Field(..., description="SoW category for this document")
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    anomalies: Optional[List[str]] = None

class ClientProfile(BaseModel):
    name: str
    dob: str
    residency: Optional[str] = None
    annual_income: Optional[float] = None
    properties: Optional[List[str]] = None

class AssessmentRequest(BaseModel):
    documents: List[DocumentItem]
    raw_text_by_url: Dict[str, str]
    client_profile: ClientProfile

class RuleFinding(BaseModel):
    rule_id: str
    severity: str
    message: str
    score_delta: int

class DocumentAssessment(BaseModel):
    url: str
    doc_type: str
    base_score: int
    findings: List[RuleFinding]
    doc_risk_score: int
    doc_risk_band: str

class AssessmentResponse(BaseModel):
    total_score: int
    risk_band: str
    needs_edd: bool
    pep_or_sanctions_hit: bool
    high_risk_geo_exposure: List[str]
    summary: str
    per_document: List[DocumentAssessment]
    audit: Dict[str, Any]

# -----------------------------
# Rule Engine
# -----------------------------

SEVERITY_TO_POINTS = {
    "critical": 25,
    "high": 15,
    "medium": 8,
    "low": 3,
}

RISK_BANDS = [
    (0, "Low"),
    (25, "Medium"),
    (50, "High"),
    (80, "Severe"),
]

HIGH_RISK_GEOS = {"iran", "north korea", "syria", "cuba", "russia", "crimea"}
MEDIUM_RISK_GEOS = {"india", "nigeria", "pakistan", "turkey", "uae", "south africa"}

def band_from_score(score: int) -> str:
    band = RISK_BANDS[0][1]
    for thresh, name in RISK_BANDS:
        if score >= thresh:
            band = name
    return band

def parse_date_any(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

@dataclass
class RuleResult:
    id: str
    severity: str
    message: str
    delta: int

class RiskRuleEngine:
    def __init__(self, client: ClientProfile):
        self.client = client

    def _finding(self, rule_id: str, severity: str, message: str) -> RuleFinding:
        pts = SEVERITY_TO_POINTS[severity]
        return RuleFinding(rule_id=rule_id, severity=severity, message=message, score_delta=pts)

    def rule_anomaly_strings(self, item: DocumentItem) -> List[RuleFinding]:
        findings: List[RuleFinding] = []
        for a in (item.anomalies or []) + (item.extracted_data.get("anomalies") or []):
            if a.lower().startswith("field_error"):
                findings.append(self._finding("ANOM.FIELD", "medium", a))
            elif a.lower().startswith("client_profile"):
                findings.append(self._finding("ANOM.CLIENT", "high", a))
            elif a.lower().startswith("evidence") or a.lower().startswith("supporting"):
                findings.append(self._finding("ANOM.EVIDENCE", "low", a))
            else:
                findings.append(self._finding("ANOM.GENERIC", "low", a))
        return findings

    def rule_employment_consistency(self, item: DocumentItem) -> List[RuleFinding]:
        f: List[RuleFinding] = []
        if item.type.lower().startswith("employment"):
            ed = item.extracted_data
            monthly = ed.get("monthly_salary")
            annual = self.client.annual_income
            if monthly and annual:
                annual_from_monthly = float(monthly) * 12.0
                if abs(annual_from_monthly - float(annual)) > 0.1 * float(annual):
                    f.append(self._finding(
                        "EMP.INCOME.MISMATCH",
                        "high",
                        f"Monthly ({monthly}) x12 != client annual ({annual}).",
                    ))
            sd = ed.get("start_date")
            if sd:
                d = parse_date_any(sd)
                if d and d > datetime.utcnow():
                    f.append(self._finding("EMP.DATE.FUTURE", "medium", f"Employment start_date {sd} is in the future."))
            slips = ed.get("salary_slip_dates") or []
            if isinstance(slips, list) and len(slips) < 3:
                f.append(self._finding("EMP.PAYSLIPS.INSUFFICIENT", "medium", "Fewer than 3 salary slip dates provided."))
            if ed.get("bank_statement_match") is False:
                f.append(self._finding("EMP.BANK.MATCH.FALSE", "high", "Salary not matched on bank statement."))
        return f

    def rule_identity_checks(self, item: DocumentItem) -> List[RuleFinding]:
        f: List[RuleFinding] = []
        if "Identity" in item.type:
            ed = item.extracted_data
            if ed.get("address_verification") is False:
                f.append(self._finding("ID.ADDRESS.UNVERIFIED", "high", "Address verification missing/false."))
            if ed.get("residency_status") in (None, "", "null"):
                f.append(self._finding("ID.RESIDENCY.MISSING", "medium", "Residency status missing."))
            doc_dob = ed.get("date_of_birth") or ed.get("dob")
            if doc_dob:
                doc_d = parse_date_any(doc_dob) or parse_date_any(str(doc_dob))
                client_d = parse_date_any(self.client.dob) or parse_date_any(str(self.client.dob))
                if doc_d and client_d and doc_d.date() != client_d.date():
                    f.append(self._finding("ID.DOB.MISMATCH", "critical", f"DOB mismatch: doc={doc_dob}, client={self.client.dob}"))
        return f

    def rule_property_geo_risk(self, item: DocumentItem) -> Tuple[List[RuleFinding], List[str]]:
        f: List[RuleFinding] = []
        high_risk_hits: List[str] = []
        if "Real Estate" in item.type or "Asset" in item.type:
            props = item.extracted_data.get("properties") or []
            for p in props:
                addr = (p.get("address") or "").lower()
                tokens = re.split(r"[,;]", addr)
                last = tokens[-1].strip() if tokens else addr
                if last in HIGH_RISK_GEOS:
                    high_risk_hits.append(last)
                    f.append(self._finding("GEO.HIGH", "critical", f"Property located in high-risk geo: {last}"))
                elif last in MEDIUM_RISK_GEOS:
                    f.append(self._finding("GEO.MEDIUM", "medium", f"Property located in medium-risk geo: {last}"))
                if not p.get("property_type"):
                    f.append(self._finding("PROP.TYPE.MISSING", "medium", "Property type missing in title/deed."))
                if not p.get("title_deed_ref"):
                    f.append(self._finding("PROP.DEED.MISSING", "high", "Title deed reference missing."))
        return f, high_risk_hits

    def apply_rules(self, item: DocumentItem) -> Tuple[List[RuleFinding], List[str]]:
        findings: List[RuleFinding] = []
        high_geo: List[str] = []
        findings.extend(self.rule_anomaly_strings(item))
        findings.extend(self.rule_employment_consistency(item))
        findings.extend(self.rule_identity_checks(item))
        f_geo, high_geo = self.rule_property_geo_risk(item)
        findings.extend(f_geo)
        return findings, high_geo

# -----------------------------
# PEP/Sanctions mock
# -----------------------------

MOCK_PEP_NAMES = {"test pep", "politically exposed"}
MOCK_SANCTIONS_NAMES = {"sdn placeholder", "blocked person"}

def mock_pep_sanctions_check(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in MOCK_PEP_NAMES or n in MOCK_SANCTIONS_NAMES

# -----------------------------
# Orchestrator
# -----------------------------

def assess_payload(req: AssessmentRequest) -> AssessmentResponse:
    engine = RiskRuleEngine(req.client_profile)
    per_doc: List[DocumentAssessment] = []
    total_score = 0
    high_geo_hits: List[str] = []

    for item in req.documents:
        base = 0
        findings, doc_high_geo = engine.apply_rules(item)
        doc_score = base + sum(f.score_delta for f in findings)
        total_score += doc_score
        high_geo_hits += doc_high_geo
        per_doc.append(DocumentAssessment(
            url=item.url,
            doc_type=item.type,
            base_score=base,
            findings=findings,
            doc_risk_score=doc_score,
            doc_risk_band=band_from_score(doc_score),
        ))

    pep_or_sanctions = mock_pep_sanctions_check(req.client_profile.name)
    if pep_or_sanctions:
        total_score += SEVERITY_TO_POINTS["critical"] * 2

    risk_band = band_from_score(total_score)
    needs_edd = total_score >= 50 or pep_or_sanctions or len(high_geo_hits) > 0

    summary = (
        f"Aggregate risk score {total_score} => {risk_band}. "
        f"EDD={'Yes' if needs_edd else 'No'}. "
        f"PEP/Sanctions={'Hit' if pep_or_sanctions else 'None'}. "
        f"High-risk geos={list(set(high_geo_hits)) if high_geo_hits else 'None'}."
    )

    audit = {
        "scoring_table": SEVERITY_TO_POINTS,
        "risk_bands": RISK_BANDS,
        "high_risk_geos": sorted(list(HIGH_RISK_GEOS)),
        "medium_risk_geos": sorted(list(MEDIUM_RISK_GEOS)),
    }

    return AssessmentResponse(
        total_score=total_score,
        risk_band=risk_band,
        needs_edd=needs_edd,
        pep_or_sanctions_hit=pep_or_sanctions,
        high_risk_geo_exposure=list(sorted(set(high_geo_hits))),
        summary=summary,
        per_document=per_doc,
        audit=audit,
    )

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Risk Assessment Rule Engine", version="1.0.0")

@app.post("/assess", response_model=AssessmentResponse)
def assess(req: AssessmentRequest):
    return assess_payload(req)

# -----------------------------
# Sample payload builder
# -----------------------------

def sample_payload() -> dict:
    documents = [
      {
        "url": "https://example.com/employment.txt",
        "type": "Employment & Salary Income",
        "extracted_data": {
          "employer": "Tech Innovations Ltd",
          "job_title": "Senior Software Engineer",
          "start_date": "2017-03-01",
          "monthly_salary": 6500.0,
          "currency": "GBP",
          "salary_slip_dates": [],
          "bank_statement_match": False,
          "anomalies": [
            "field_error:salary_slip_dates is empty but should contain date strings",
            "field_error:bank_statement_match is false, supporting evidence required",
            "client_profile:monthly_salary does not match annual_income from client profile"
          ],
          "extraction_errors": []
        },
        "anomalies": [
          "field_error:salary_slip_dates is empty but should contain date strings",
          "field_error:bank_statement_match is false, supporting evidence required",
          "client_profile:monthly_salary does not match annual_income from client profile"
        ]
      },
      {
        "url": "https://example.com/property.txt",
        "type": "Real Estate & Asset Sales",
        "extracted_data": {
          "properties": [
            {
              "address": "25 Park Lane, India",
              "property_type": None,
              "current_value": 850000.0,
              "purchase_date": "2018-09-15",
              "title_deed_ref": "TGL123456"
            }
          ],
          "anomalies": [
            "client_profile: Address mismatch for property. Extracted address '25 Park Lane, India' does not match client profile address '25 Park Lane, London'.",
            "field_error: Missing 'property_type' field in extracted document."
          ],
          "extraction_errors": []
        },
        "anomalies": [
          "client_profile: Address mismatch for property. Extracted address '25 Park Lane, India' does not match client profile address '25 Park Lane, London'.",
          "field_error: Missing 'property_type' field in extracted document."
        ]
      },
      {
        "url": "https://example.com/identity.txt",
        "type": "Identity & Residency Proof",
        "extracted_data": {
          "full_name": "John Smith",
          "date_of_birth": "1985-05-15",
          "residency_status": None,
          "address_verification": False,
          "address": None,
          "passport_number": "123456789",
          "emirates_id": None,
          "visa_status": None,
          "anomalies": [
            "field_error:residency_status is null but should be a string or null",
            "field_error:address_verification is false but should be a boolean or null",
            "field_error:address is null but should be a string or null",
            "client_profile:residency_status is null but client profile indicates 'UK Citizen'",
            "client_profile:address_verification is false but client profile indicates address '25 Park Lane, London' is present"
          ],
          "extraction_errors": []
        },
        "anomalies": [
          "field_error:residency_status is null but should be a string or null",
          "field_error:address_verification is false but should be a boolean or null",
          "field_error:address is null but should be a string or null",
          "client_profile:residency_status is null but client profile indicates 'UK Citizen'",
          "client_profile:address_verification is false but client profile indicates address '25 Park Lane, London' is present"
        ]
      }
    ]

    raw_text_by_url = {
        "https://example.com/employment.txt": "Employment confirmation text...",
        "https://example.com/property.txt": "Property title deed text...",
        "https://example.com/identity.txt": "Passport information text..."
    }

    client_profile = {
        "name": "John Smith",
        "dob": "15/05/1985",
        "residency": "UK Citizen",
        "annual_income": 79000,
        "properties": ["25 Park Lane, London"]
    }
    return {"documents": documents, "raw_text_by_url": raw_text_by_url, "client_profile": client_profile}

if __name__ == "__main__":
    payload = AssessmentRequest(**sample_payload())
    res = assess_payload(payload)
    import json
    print(json.dumps(res.dict(), indent=2))

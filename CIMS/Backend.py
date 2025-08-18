from flask import Flask, request, jsonify, send_file, render_template
from fpdf import FPDF
import uuid
import io
import math
from collections import defaultdict

app = Flask(__name__)

# ======================
# Utility: Letter-to-letter translation (kept for anonymized reports)
# ======================
ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUBS = "QWERTYUIOPASDFGHJKLZXCVBNM"

def translate(from_letters, to_letters, text):
    if not(from_letters.isupper() and from_letters.isalpha() and 
           to_letters.isupper() and to_letters.isalpha()):
        raise ValueError("from_letters and to_letters must be all uppercase letters")
    if len(from_letters) != len(to_letters):
        raise ValueError("from_letters and to_letters must be the same length")
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    m = {from_letters[i]: to_letters[i] for i in range(len(from_letters))}
    out = []
    for ch in text:
        up = ch.upper()
        if up in m:
            mapped = m[up]
            out.append(mapped if ch.isupper() else mapped.lower())
        else:
            out.append(ch)
    return ''.join(out)

# ======================
# Math helpers
# ======================

def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def hhi(weights):
    return sum(w*w for w in weights)

def effective_holdings(weights):
    hh = hhi(weights)
    return (1.0 / hh) if hh > 0 else 0.0

def weighted_fee_drag(positions):
    """Expense ratios expected as decimals, e.g., 0.003 = 0.30%.
    Returns portfolio-level fee drag as a decimal.
    """
    return sum((p.get("weight", 0.0) or 0.0) * (p.get("expense_ratio", 0.0) or 0.0) for p in positions)

def weighted_dividend_yield(positions):
    """Dividend yields expected as decimals, e.g., 0.025 = 2.5%.
    Returns portfolio-level dividend yield as a decimal.
    """
    return sum((p.get("weight", 0.0) or 0.0) * (p.get("dividend_yield", 0.0) or 0.0) for p in positions)

def dividend_concentration_ratio(positions, top_k=1):
    contribs = []
    for p in positions:
        w = p.get("weight", 0.0) or 0.0
        dy = p.get("dividend_yield", 0.0) or 0.0
        contribs.append(w * dy)
    total = sum(contribs) or 1e-12
    share = sum(sorted(contribs, reverse=True)[:top_k]) / total
    return share

def sector_weights(positions):
    m = defaultdict(float)
    for p in positions:
        m[p.get("sector", "Unknown")] += p.get("weight", 0.0) or 0.0
    return dict(m)

def region_weights(positions):
    m = defaultdict(float)
    for p in positions:
        m[p.get("region", "Unknown")] += p.get("weight", 0.0) or 0.0
    return dict(m)

def overweight_vs_benchmark(portf_map, bench_map, threshold=0.05):
    keys = set(portf_map) | set(bench_map)
    over, under = {}, {}
    for k in keys:
        diff = (portf_map.get(k, 0.0) - bench_map.get(k, 0.0))
        if diff >= threshold:
            over[k] = diff
        elif diff <= -threshold:
            under[k] = diff
    return over, under

# ======================
# Data Model (in-memory)
# ======================
class Portfolio:
    def __init__(self, owner, currency, positions, benchmark_weights=None, name=None):
        self.id = str(uuid.uuid4())
        self.owner = owner
        self.name = name or f"Portfolio {self.id[:6]}"
        self.currency = currency or "USD"
        self.positions = normalize_positions(positions)
        # benchmark sector weights (optional): {sector: weight}
        self.benchmark_weights = benchmark_weights or {}

    def to_dict(self):
        return {
            "id": self.id,
            "owner": self.owner,
            "name": self.name,
            "currency": self.currency,
            "positions": self.positions,
            "benchmark_weights": self.benchmark_weights,
        }

portfolios = []

# ======================
# Weight normalization
# ======================

def normalize_positions(positions):
    """
    Accepts positions with either:
      - explicit weights summing ~1.0, OR
      - shares and last_price (we compute weights),
      - or both (weights will be recomputed from shares*price if present).
    Each position item shape (fields optional):
      {"ticker": str, "weight": float, "shares": float, "last_price": float,
       "expense_ratio": float, "dividend_yield": float, "sector": str, "region": str}
    """
    # If any position has shares & price, derive weights from market values
    has_values = any((p.get("shares") is not None and p.get("last_price") is not None) for p in positions)

    positions_clean = []
    if has_values:
        values = []
        for p in positions:
            sh = p.get("shares")
            pr = p.get("last_price")
            val = (float(sh) * float(pr)) if (sh is not None and pr is not None) else 0.0
            values.append(val)
        total = sum(values) or 1.0
        for p, val in zip(positions, values):
            w = val / total
            positions_clean.append({
                "ticker": p.get("ticker", "UNK"),
                "weight": w,
                "shares": p.get("shares"),
                "last_price": p.get("last_price"),
                "expense_ratio": p.get("expense_ratio", 0.0),
                "dividend_yield": p.get("dividend_yield", 0.0),
                "sector": p.get("sector", "Unknown"),
                "region": p.get("region", "Unknown"),
            })
    else:
        total_w = sum((p.get("weight", 0.0) or 0.0) for p in positions) or 1.0
        for p in positions:
            w = (p.get("weight", 0.0) or 0.0) / total_w
            positions_clean.append({
                "ticker": p.get("ticker", "UNK"),
                "weight": w,
                "shares": p.get("shares"),
                "last_price": p.get("last_price"),
                "expense_ratio": p.get("expense_ratio", 0.0),
                "dividend_yield": p.get("dividend_yield", 0.0),
                "sector": p.get("sector", "Unknown"),
                "region": p.get("region", "Unknown"),
            })
    return positions_clean

# ======================
# Vitals calculator
# ======================

def compute_vitals(portfolio: Portfolio):
    pos = portfolio.positions
    weights = [p["weight"] for p in pos]

    eff_n = effective_holdings(weights)
    fee = weighted_fee_drag(pos)
    yld = weighted_dividend_yield(pos)
    div_top1 = dividend_concentration_ratio(pos, top_k=1)

    sectors = sector_weights(pos)
    regions = region_weights(pos)

    over_s, under_s = ({}, {})
    if portfolio.benchmark_weights:
        over_s, under_s = overweight_vs_benchmark(sectors, portfolio.benchmark_weights, threshold=0.05)

    alerts = []
    if eff_n < 5:
        alerts.append(f"High concentration: effective holdings ≈ {eff_n:.1f} (consider spreading risk).")
    if fee > 0.004:  # >0.40%
        alerts.append(f"Fee drag is {fee*100:.2f}%/yr; consider cheaper funds.")
    if div_top1 > 0.6:
        alerts.append(f"{div_top1*100:.0f}% of dividends come from a single holding — fragile income.")
    for k, v in over_s.items():
        alerts.append(f"Overweight {k} by {v*100:.1f}pp vs benchmark.")

    return {
        "effective_holdings": round(eff_n, 2),
        "weighted_fee_drag_pct": round(fee * 100, 2),
        "effective_dividend_yield_pct": round(yld * 100, 2),
        "dividend_top1_share_pct": round(div_top1 * 100, 1),
        "sector_weights": sectors,
        "region_weights": regions,
        "alerts": alerts,
    }

# ======================
# API Routes – Portfolios CRUD
# ======================
@app.route("/portfolios", methods=["GET"]) 
def list_portfolios():
    return jsonify([p.to_dict() for p in portfolios])

@app.route("/portfolios", methods=["POST"]) 
def create_portfolio():
    data = request.get_json(force=True)
    p = Portfolio(
        owner=data.get("owner", "user"),
        currency=data.get("currency", "USD"),
        positions=data.get("positions", []),
        benchmark_weights=data.get("benchmark_weights", {}),
        name=data.get("name")
    )
    portfolios.append(p)
    return jsonify(p.to_dict()), 201

@app.route("/portfolios/<pid>", methods=["DELETE"]) 
def delete_portfolio(pid):
    global portfolios
    before = len(portfolios)
    portfolios = [p for p in portfolios if p.id != pid]
    return jsonify({"deleted": before - len(portfolios)})

@app.route("/portfolios/<pid>", methods=["PATCH"]) 
def update_portfolio(pid):
    data = request.get_json(force=True)
    p = next((x for x in portfolios if x.id == pid), None)
    if not p:
        return jsonify({"error": "portfolio not found"}), 404
    if "owner" in data: p.owner = data["owner"]
    if "name" in data: p.name = data["name"]
    if "currency" in data: p.currency = data["currency"]
    if "positions" in data: p.positions = normalize_positions(data["positions"]) 
    if "benchmark_weights" in data: p.benchmark_weights = data["benchmark_weights"] or {}
    return jsonify(p.to_dict())

# ======================
# Analysis routes
# ======================
@app.route("/portfolios/<pid>/vitals", methods=["GET"]) 
def portfolio_vitals(pid):
    p = next((x for x in portfolios if x.id == pid), None)
    if not p:
        return jsonify({"error": "portfolio not found"}), 404
    return jsonify(compute_vitals(p))

@app.route("/similar/portfolio/<pid>", methods=["GET"]) 
def similar_portfolios(pid):
    p = next((x for x in portfolios if x.id == pid), None)
    if not p:
        return jsonify({"error": "portfolio not found"}), 404
    # Compare by sector vectors
    sectors = sorted(set().union(*[set(sector_weights(q.positions).keys()) for q in portfolios]))
    def vector_for(portf):
        sw = sector_weights(portf.positions)
        return [sw.get(s, 0.0) for s in sectors]
    target_vec = vector_for(p)
    sims = []
    for other in portfolios:
        if other.id == p.id:
            continue
        sim = cosine_similarity(target_vec, vector_for(other))
        sims.append({"portfolio": other.to_dict(), "similarity": round(sim, 4)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify(sims)

# ======================
# Reports
# ======================
@app.route("/report", methods=["GET"]) 
def get_report():
    pid = request.args.get("portfolio_id")
    anonymize = request.args.get("anonymize", "false").lower() == "true"
    if not pid:
        return jsonify({"error": "portfolio_id is required"}), 400
    p = next((x for x in portfolios if x.id == pid), None)
    if not p:
        return jsonify({"error": "portfolio not found"}), 404

    vitals = compute_vitals(p)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    title = f"Portfolio Vitals Report – {p.name}"
    if anonymize:
        title = translate(ALPH, SUBS, title)
    pdf.cell(200, 10, txt=title, ln=True, align='C')

    # Vitals summary
    pdf.ln(4)
    pdf.cell(200, 8, txt=f"Effective holdings: {vitals['effective_holdings']}")
    pdf.ln(6)
    pdf.cell(200, 8, txt=f"Weighted fee drag: {vitals['weighted_fee_drag_pct']:.2f}%/yr")
    pdf.ln(6)
    pdf.cell(200, 8, txt=f"Effective dividend yield: {vitals['effective_dividend_yield_pct']:.2f}%")
    pdf.ln(6)
    pdf.cell(200, 8, txt=f"Dividend top-1 share: {vitals['dividend_top1_share_pct']:.1f}%")

    # Alerts
    pdf.ln(8)
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 8, txt="Alerts")
    pdf.set_font("Arial", size=12)
    pdf.ln(6)
    if vitals["alerts"]:
        for a in vitals["alerts"]:
            txt = translate(ALPH, SUBS, a) if anonymize else a
            pdf.multi_cell(0, 8, txt=f"• {txt}")
    else:
        pdf.cell(200, 8, txt="No alerts.")

    # Sector weights
    pdf.ln(6)
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 8, txt="Sector Weights")
    pdf.set_font("Arial", size=12)
    pdf.ln(6)
    for k, v in sorted(vitals["sector_weights"].items(), key=lambda kv: -kv[1]):
        pdf.cell(200, 8, txt=f"{k}: {v*100:.1f}%", ln=True)

    # Positions
    pdf.ln(6)
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 8, txt="Positions")
    pdf.set_font("Arial", size=12)
    pdf.ln(4)
    for pos in p.positions:
        line = f"{pos['ticker']}  w={pos['weight']*100:.2f}%  ER={pos.get('expense_ratio',0.0)*100:.2f}%  DY={pos.get('dividend_yield',0.0)*100:.2f}%  {pos.get('sector','')}/{pos.get('region','')}"
        if anonymize:
            line = translate(ALPH, SUBS, line)
        pdf.cell(0, 8, txt=line, ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='portfolio_report.pdf'
    )

# ======================
# Home (placeholder for your frontend)
# ======================
@app.route("/")
def home():
    return render_template("cims_index.html")

if __name__ == "__main__":
    app.run(debug=True)

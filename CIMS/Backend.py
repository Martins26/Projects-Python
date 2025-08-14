from flask import Flask, request, jsonify, send_file, render_template
from fpdf import FPDF
import math
import uuid
import io

app = Flask(__name__)

# ======================
# Utility: Letter-to-letter translation
# ======================
def translate(from_letters, to_letters, text):
    if not(from_letters.isupper() and from_letters.isalpha() and 
           to_letters.isupper() and to_letters.isalpha()):
        raise ValueError("from_letters and to_letters must be all uppercase letters")
    if len(from_letters) != len(to_letters):
        raise ValueError("from_letters and to_letters must be the same length")
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    new_string = ""
    text_list = list(text)
    original_text_list = list(text_list)

    for i in range(len(from_letters)):
        if from_letters[i].upper() in text.upper():
            for j in range(len(text_list)):
                if original_text_list[j].upper() == from_letters[i].upper():
                    if original_text_list[j].isupper():
                        text_list[j] = to_letters[i].upper()
                    else:
                        text_list[j] = to_letters[i].lower()

    for a in text_list:
        new_string += a
    return new_string

# ======================
# Data Model
# ======================
class Client:
    def __init__(self, first_name, last_name, email, phone, portfolio_score):
        self.id = str(uuid.uuid4())
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = phone
        self.portfolio_score = portfolio_score

    def to_dict(self):
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "portfolio_score": self.portfolio_score
        }

clients = []

# ======================
# Cosine Similarity
# ======================
def cosine_similarity(v1, v2):
    dot_product = sum(a*b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a*a for a in v1))
    norm_v2 = math.sqrt(sum(b*b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

# ======================
# API Routes
# ======================
@app.route("/clients", methods=["GET"])
def get_clients():
    return jsonify([c.to_dict() for c in clients])

@app.route("/clients", methods=["POST"])
def add_client():
    data = request.json
    try:
        score = float(data.get("portfolio_score"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid score"}), 400

    client = Client(
        data.get("first_name"),
        data.get("last_name"),
        data.get("email"),
        data.get("phone"),
        score
    )
    clients.append(client)
    return jsonify(client.to_dict()), 201

@app.route("/clients/<client_id>", methods=["DELETE"])
def delete_client(client_id):
    global clients
    clients = [c for c in clients if c.id != client_id]
    return jsonify({"message": "Deleted"}), 200

@app.route("/stats", methods=["GET"])
def get_stats():
    if not clients:
        return jsonify({"error": "No clients"}), 400
    scores = [c.portfolio_score for c in clients]
    avg = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    num_high = sum(1 for s in scores if s >= avg)
    num_low = len(scores) - num_high
    return jsonify({
        "average": avg,
        "highest": max_score,
        "lowest": min_score,
        "above_average": num_high,
        "below_average": num_low
    })

@app.route("/similar/<client_id>", methods=["GET"])
def get_similar(client_id):
    target = next((c for c in clients if c.id == client_id), None)
    if not target:
        return jsonify({"error": "Client not found"}), 404
    sims = []
    for other in clients:
        if other.id != target.id:
            sim = cosine_similarity([target.portfolio_score], [other.portfolio_score])
            sims.append({"client": other.to_dict(), "similarity": sim})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify(sims)

@app.route("/report", methods=["GET"])
def get_report():
    anonymize = request.args.get("anonymize", "false").lower() == "true"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Client Portfolio Report", ln=True, align='C')

    # Stats
    if clients:
        scores = [c.portfolio_score for c in clients]
        avg = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        num_high = sum(1 for s in scores if s >= avg)
        num_low = len(scores) - num_high
        stats_text = [
            f"Average Score: {avg:.2f}",
            f"Highest Score: {max_score:.2f}",
            f"Lowest Score: {min_score:.2f}",
            f"Above Average: {num_high}",
            f"Below Average: {num_low}"
        ]
        for line in stats_text:
            pdf.cell(200, 10, txt=line, ln=True)

    pdf.cell(200, 10, txt="--- Clients ---", ln=True)
    for client in clients:
        name = f"{client.first_name} {client.last_name}"
        if anonymize:
            name = translate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "QWERTYUIOPASDFGHJKLZXCVBNM", name)
        pdf.cell(200, 10, txt=f"{name}, Email: {client.email}, Phone: {client.phone}, Score: {client.portfolio_score}", ln=True)

    # Return as download
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='report.pdf'
    )
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

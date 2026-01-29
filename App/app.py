from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

app = Flask(__name__)

MODEL_PATH = "fraud_distilbert_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOUR trained model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


def format_job_text(job_title, company_profile, description, requirements, benefits):
    """
    EXACTLY matches training data structure
    """
    return f"""
    Job Title: {job_title}

    Company Profile:
    {company_profile}

    Job Description:
    {description}

    Requirements:
    {requirements}

    Benefits:
    {benefits}
    """


def predict_fraud(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    # IMPORTANT: match your training labels
    label = "This Job is Fraud" if pred == 1 else "This Job is Genuine"

    confidence = round(probs[0][pred].item() * 100, 2)

    return label, confidence




@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "job_title": request.form["job_title"],
            "company_profile": request.form["company_profile"],
            "description": request.form["description"],
            "requirements": request.form["requirements"],
            "benefits": request.form["benefits"],
        }

        combined_text = format_job_text(**form_data)
        message, confidence = predict_fraud(combined_text)

        result = {
            "message": message,
            "confidence": confidence
        }

    return render_template(
        "index.html",
        result=result,
        form_data=form_data
    )



if __name__ == "__main__":
    app.run(debug=True)
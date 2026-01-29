# Fake Job Posting Detection using NLP & Deep Learning

---

## Project Overview
The rapid growth of online recruitment platforms has increased the risk of fraudulent job postings that mislead job seekers through unrealistic salary promises, vague job descriptions, and requests for registration fees. This project implements an AI-based Fake Job Posting Detection system using Natural Language Processing (NLP) and a transformer-based deep learning model to automatically classify job postings as **Genuine** or **Fraudulent** along with a confidence score. The system combines a fine-tuned **DistilBERT** model with a Flask-based web application to provide real-time predictions through a simple and intuitive interface.

---

## Problem Statement
Online job portals are frequently exploited by scammers who post fake job advertisements that appear legitimate. Manual verification of job postings is inefficient, error-prone, and does not scale with the volume of listings. There is a strong need for an automated and intelligent system that can analyze job posting content and accurately detect fraudulent patterns to protect job seekers from recruitment scams.

---

## Solution Provided
This project uses a supervised learning approach with a fine-tuned **DistilBERT** model for binary text classification. Job-related attributes such as job title, company profile, job description, requirements, and benefits are combined into a structured textual format and passed to the model. The transformer-based architecture captures contextual and semantic patterns instead of relying on keyword matching.

The trained model is deployed using a Flask web application that enables users to input job details and instantly receive an authenticity prediction with a confidence score.

---

## Data Insights
Exploratory analysis of the dataset revealed several important patterns:

- Fraudulent job postings often:
  - Promise unrealistically high salaries
  - Use urgency-based phrases like "Apply Immediately"
  - Provide vague company details
  - Ask for payments or registration fees
  - Include unofficial communication channels (WhatsApp, Telegram)

- Genuine job postings usually:
  - Provide structured role descriptions
  - Clearly define required skills
  - Mention company background
  - Offer professional contact methods

- Class imbalance was observed in the dataset, which required careful handling during training to avoid biased predictions.

These insights helped shape preprocessing decisions and improved model reliability.

---

## Data Preprocessing Pipeline
The preprocessing stage ensured that the input text was clean, consistent, and informative for the model.

### Steps Involved:
1. Handling missing values
2. Removing unnecessary symbols and whitespace
3. Combining multiple job attributes into a single structured text
4. Label encoding for classification
5. Tokenization using DistilBERT tokenizer
6. Padding and truncation for uniform sequence length

### Input Formatting Function
Maintaining consistent formatting between training and inference significantly improved prediction performance.

```python
def format_job_text(job_title, company_profile, description, requirements, benefits):
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
```

---

## Model Architecture
This project uses DistilBERT, a lightweight and efficient transformer model known for faster training and strong NLP performance.
### Why DistilBERT?
- Faster than BERT
- Lower memory usage
- High contextual understanding
- Suitable for real-time applications

---

## Important Training Code Snippets
**Tokenization**
```
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256
)
```
**Model Setup**
```
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
```
**Training Loop**
```
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## Prediction Logic (Flask App)
```
with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=1)
pred = torch.argmax(probs, dim=1).item()

label = "This Job is Fraud" if pred == 1 else "This Job is Genuine"
confidence = round(probs[0][pred].item() * 100, 2)
```

---

## Web Application Features
- User-friendly input form
- Real-time prediction
- Confidence score display
- Input retention after submission
- Professional UI for academic demos

---

## Project Structure
Fake-Job-Posting-Detection/
├── App/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── data/
│   └── cleaned_data.pkl
├── fraud_distilbert_model/
├── logs/
├── results/
├── Videos/
├── data_preprocessing.ipynb
├── fake_job_postings.csv
├── model.ipynb
├── test.ipynb
└── README.md

---

## Future Scope
- Explainable AI for fraud reasoning
- Multi-language support
- Cloud deployment
- Integration with job portals
- Continuous model learning from feedback

---

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Flask
- HTML
- CSS

---

## Conclusion
This project demonstrates the practical application of transformer-based NLP models in solving real-world problems. By combining deep learning with a real-time web interface, the system provides an intelligent, scalable solution to detect fake job postings and protect job seekers.

---

## Author
**Sanjana Nathani**

M.Sc. Data Science, DAU Gandhinagar

Aspiring Data Scientist

LinkedIn: [https://www.linkedin.com/in/sanjana-nathani-26a42727b/](https://www.linkedin.com/in/sanjana-nathani-26a42727b/)

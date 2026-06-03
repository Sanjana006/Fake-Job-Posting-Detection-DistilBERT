# 🔍 Fake Job Posting Detection using NLP & Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/DistilBERT-Transformer-FFD700?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-4CAF50?style=for-the-badge&logo=openai&logoColor=white"/>
</p>

<p align="center">
  <a href="https://fake-job-posting-detection-distilbert.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-brightgreen?style=for-the-badge"/>
  </a>
</p>

<br/>

> *"Fraudulent job postings don't just waste time — they destroy trust, steal money, and exploit vulnerable job seekers.*
> *We trained a transformer model on 17,000+ real and fake job listings.*
> *It now catches fraud with 98.2% accuracy — in under a second."*

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Solution Provided](#-solution-provided)
- [Dataset Overview](#-dataset-overview)
- [Data Insights](#-data-insights)
- [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Code Snippets](#-training-code-snippets)
- [Prediction Logic](#-prediction-logic)
- [Web Application](#-web-application)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Future Scope](#-future-scope)
- [How to Run This Project](#-how-to-run-this-project)
- [Conclusion](#-conclusion)
- [Connect](#-connect)

---

## 🧠 Project Overview

The rapid growth of online recruitment platforms has significantly increased the risk of fraudulent job postings that exploit job seekers through unrealistic salary promises, vague job descriptions, and requests for registration fees or personal information.

This project implements an end-to-end **AI-based Fake Job Posting Detection system** using Natural Language Processing (NLP) and a transformer-based deep learning model to automatically classify job postings as **Genuine** or **Fraudulent**, along with a real-time confidence score.

The system combines a fine-tuned **DistilBERT** model with a **Streamlit web application** that allows users to paste any job listing and receive an instant authenticity prediction — making it practical both as a research prototype and a real-world screening tool.

---

## ❗ Problem Statement

Online job portals are frequently exploited by scammers who post fake job advertisements that appear legitimate. These postings cause real financial and emotional harm to job seekers who invest time, share personal details, or even pay registration fees.

Manual verification of job postings is inefficient, error-prone, and does not scale with the volume of listings on modern platforms. There is a strong need for an automated, intelligent system that can:

- Analyse the **textual content** of a job posting
- Identify **fraudulent linguistic patterns** learned from labelled examples
- Return a prediction with a **confidence score** in real-time

---

## ✅ Solution Provided

This project uses a **supervised learning approach** with a fine-tuned **DistilBERT** model for binary text classification. Five job-related attributes — job title, company profile, job description, requirements, and benefits — are combined into a single structured text block and passed to the model.

The transformer-based architecture captures **contextual and semantic patterns** across the full posting rather than relying on simple keyword matching. The trained model is deployed as a **Streamlit web application** for real-time predictions.

> **Why DistilBERT over a simpler model?**
> Traditional ML models (TF-IDF + Logistic Regression) catch surface-level patterns. DistilBERT understands the *context* of language — it knows that "Send your CV via WhatsApp" is suspicious in a way that keyword matching alone cannot capture. Transformer models consistently outperform classical approaches on this task.

---

## 📂 Dataset Overview

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle — Employment Scam Aegean Dataset (EMSCAD) |
| **Total Records** | ~17,880 job postings |
| **Fraudulent** | ~866 (≈ 4.8%) |
| **Genuine** | ~17,014 (≈ 95.2%) |
| **Class Imbalance** | Yes — handled during training |
| **Text Fields Used** | Job Title, Company Profile, Description, Requirements, Benefits |
| **Target Label** | `fraudulent` (0 = Genuine, 1 = Fraud) |

**Input Fields Used for Classification:**
- `title` — Job title
- `company_profile` — Company background text
- `description` — Role description
- `requirements` — Required skills and qualifications
- `benefits` — Compensation and perks

---

## 🔎 Data Insights

Exploratory analysis of the dataset revealed several important patterns that shaped preprocessing decisions:

**Fraudulent job postings often:**
- Promise unrealistically high salaries with no experience requirements
- Use urgency-based phrases like *"Apply Immediately"*, *"Limited Slots"*
- Provide vague or missing company details
- Ask for payments, registration fees, or upfront deposits
- Include unofficial communication channels (WhatsApp, Telegram, personal emails)
- Have very short, generic descriptions with no specific role requirements

**Genuine job postings usually:**
- Provide structured, detailed role descriptions
- Clearly define required skills and qualifications
- Mention verifiable company background and industry
- Offer professional contact methods (corporate email, official portal)
- Have longer, more specific text across all fields

> **Class Imbalance:** Only ~4.8% of postings are fraudulent. This was handled using weighted loss during fine-tuning to prevent the model from simply predicting "genuine" for all inputs. Without this, accuracy would appear high but recall on fraud would collapse.

---

## 🧹 Data Preprocessing Pipeline

The preprocessing stage ensured clean, consistent, and well-structured input for the model.

### Steps Involved:
1. **Handling missing values** — NaN fields replaced with empty strings
2. **Removing unnecessary symbols and whitespace** — HTML artifacts, special characters
3. **Combining multiple fields** into a single structured text block
4. **Label encoding** — `fraudulent` column → binary integer label
5. **Tokenization** using DistilBERT fast tokenizer
6. **Padding and truncation** to uniform sequence length (max 512 tokens)
7. **Train/validation split** — 80/20 stratified split

### Input Formatting Function

Maintaining **identical formatting between training and inference** was critical for prediction reliability. Any mismatch between how text was structured during training vs. at inference time degrades performance.

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

## 🏗️ Model Architecture

This project uses **DistilBERT** (`distilbert-base-uncased`), a lightweight and efficient transformer model distilled from BERT that retains ~97% of BERT's performance at 60% of the size and 40% faster inference.

### Why DistilBERT?

| Property | DistilBERT | BERT-base |
|----------|-----------|-----------|
| Parameters | 66M | 110M |
| Inference Speed | ~40% faster | Baseline |
| Accuracy (GLUE) | 97% of BERT | 100% |
| Memory Usage | Lower | Higher |
| Suitable for real-time | Yes | Slower |

A **classification head** (linear layer → 2 output logits) is added on top of the `[CLS]` token representation, then fine-tuned end-to-end on the labelled dataset.

---

## 💻 Training Code Snippets

**Tokenization**
```python
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=512
)
```

**Model Setup**
```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
```

**Training with Hugging Face Trainer**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

**Training Loop (Manual)**
```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## 🔮 Prediction Logic

```python
def predict_fraud(text, model, tokenizer, device):
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

    label = "fraud" if pred == 1 else "genuine"
    confidence = round(probs[0][pred].item() * 100, 2)

    return label, confidence
```

> **Confidence Score:** The model outputs raw logits which are converted to probabilities via softmax. The confidence is the probability assigned to the predicted class. A high confidence genuine prediction (~95%+) is a strong signal; a high confidence fraud prediction is a red flag worth acting on.

---

## 🖥️ Web Application

The model is deployed as a **Streamlit web application** with a fully custom dark glassmorphism UI.

### Features:
- **Animated dark gradient background** — feels alive and modern
- **Glassmorphism form panel** — translucent input area with blur effect
- **Real-time stats** — accuracy, training scale, inference time displayed as metric cards
- **2-column input layout** — efficient data entry for all 5 job fields
- **Instant prediction** — DistilBERT inference in under 1 second
- **Green result card** for genuine postings with animated confidence bar
- **Red result card** for fraudulent postings with glowing warning effect
- **Model caching** via `@st.cache_resource` — loads only once per session

### Running the App:
```bash
cd App
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
Fake-Job-Posting-Detection-DistilBERT/
│
├── App/
│   ├── static/
│   │   └── style.css                  ← Original Flask CSS (legacy)
│   ├── templates/
│   │   └── index.html                 ← Original Flask template (legacy)
│   └── app.py                         ← ✅ Streamlit web application
│
├── data/
│   └── cleaned_data.pkl               ← Preprocessed dataset (gitignored)
│
├── fraud_distilbert_model/            ← Fine-tuned model weights (gitignored)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
│
├── results/                           ← Training checkpoints (gitignored)
├── logs/                              ← TensorBoard training logs
├── data_preprocessing.ipynb          ← Data cleaning & EDA notebook
├── model.ipynb                        ← Model training notebook
├── fake_job_postings.csv              ← Raw dataset (gitignored)
├── requirements.txt                   ← Python dependencies
├── .gitignore
└── README.md
```

> **Note:** Large files (`model.safetensors`, `.pt` checkpoints, `.csv` datasets) are excluded from the repository via `.gitignore`. Host model weights on HuggingFace Hub and datasets on Kaggle/HuggingFace Datasets for sharing.

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.9+** | Core programming language |
| **PyTorch** | Deep learning framework for model training and inference |
| **Hugging Face Transformers** | DistilBERT model and tokenizer |
| **DistilBERT** | Pre-trained transformer backbone, fine-tuned for classification |
| **Streamlit** | Web application framework with custom CSS UI |
| **pandas / NumPy** | Data manipulation and preprocessing |
| **scikit-learn** | Train/val split, evaluation metrics |
| **Jupyter Notebook** | Exploratory analysis and model training notebooks |

---

## 🚀 Future Scope

- **Explainable AI** — highlight which words/phrases triggered the fraud detection using LIME or SHAP
- **Multi-language support** — extend to detect fraud in job postings in Hindi, Arabic, Spanish, etc.
- **Cloud deployment** — deploy to Streamlit Community Cloud, AWS, or HuggingFace Spaces
- **Real-time API** — wrap the model in a FastAPI endpoint for integration with job portals
- **Continuous learning** — re-train periodically on new labelled data as fraud patterns evolve
- **Browser extension** — detect fake jobs directly on LinkedIn, Indeed, Naukri while browsing

---

## ▶️ How to Run This Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Download or Train the Model
The trained model (`fraud_distilbert_model/`) is not included in the repository due to size constraints.

**Option A — Train it yourself:**
```bash
jupyter notebook model.ipynb
```

**Option B — Download from HuggingFace Hub:**  
*(Link to be added after upload)*

### Step 2: Run the Web App
```bash
cd App
streamlit run app.py
```

### Step 3: Open in Browser
Navigate to **http://localhost:8501** and paste any job listing to get an instant prediction.

---

## 📝 Conclusion

This project demonstrates the practical application of transformer-based NLP models in solving a real-world fraud detection problem. By combining the contextual language understanding of DistilBERT with a clean, real-time Streamlit interface, the system provides an **intelligent, scalable, and accessible** tool to protect job seekers from recruitment scams.

The project goes beyond a simple model training exercise — it includes structured data preprocessing, a deployment-ready web application, and a thoughtful approach to handling class imbalance and real-world inference consistency.

> **98.2% accuracy. Under 1 second. One less job seeker deceived.**

---

## 🤝 Connect

**Sanjana Nathani**

M.Sc. Data Science, DAU Gandhinagar | Aspiring Data Scientist

<p>
  <a href="https://www.linkedin.com/in/sanjana-nathani-26a42727b/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/Sanjana006">
    <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <a href="https://fake-job-posting-detection-distilbert.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Try%20It-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>

---

<p align="center">
  <i>Built with ❤️ using DistilBERT · PyTorch · Streamlit</i>
</p>

# Fake Job Posting Detection using NLP & Deep Learning

---

## Project Overview
The rapid growth of online recruitment platforms has increased the risk of fraudulent job postings that mislead job seekers through unrealistic salary promises, vague job descriptions, and requests for registration fees. This project implements an AI-based Fake Job Posting Detection system using Natural Language Processing (NLP) and a transformer-based deep learning model to automatically classify job postings as Genuine or Fraudulent along with a confidence score. The system combines a trained DistilBERT model with a Flask-based web application to provide real-time predictions through a simple and intuitive interface.

---

## Problem Statement
Online job portals are frequently exploited by scammers who post fake job advertisements that appear legitimate. Manual verification of job postings is inefficient, error-prone, and does not scale with the volume of listings. There is a strong need for an automated and intelligent system that can analyze job posting content and accurately detect fraudulent patterns to protect job seekers from recruitment scams.

---

## Solution Provided
This project uses a supervised learning approach with a fine-tuned DistilBERT model for binary text classification. Job-related attributes such as job title, company profile, job description, requirements, and benefits are combined into a structured textual format and passed to the model. The transformer-based architecture captures contextual and semantic patterns instead of relying on keyword matching. The trained model is deployed using a Flask web application that enables users to input job details and instantly receive an authenticity prediction with a confidence score.

---

## Key Insights from the Project
Fraudulent job postings commonly include unrealistic compensation, urgency-based language, vague company information, and unofficial contact methods such as WhatsApp or Telegram. Genuine job postings tend to have structured descriptions, clear responsibilities, and well-defined qualification requirements. Transformer-based models like DistilBERT are effective at capturing subtle contextual cues, which significantly reduces false positives. Maintaining identical input formatting during training and deployment improves prediction reliability and model performance.

---

## Project Structure
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

## Important Code Snippets
The following function ensures that the input format during inference exactly matches the format used during training, which is critical for reliable predictions:

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

Model prediction logic used in the Flask application:

```python
with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=1)
pred = torch.argmax(probs, dim=1).item()

label = "This Job is Fraud" if pred == 1 else "This Job is Genuine"
confidence = round(probs[0][pred].item() * 100, 2)
```

---

## Web Application Features
The web application provides a clean and responsive user interface where users can input job details and receive real-time predictions. It displays both the predicted label and the confidence score, retains user input after submission, and offers a smooth and professional user experience suitable for demonstrations and academic evaluation.

---

## How to Use This Project
To use this project locally, first clone the repository using:
```
git clone https://github.com/your-username/fake-job-posting-detection.git
cd fake-job-posting-detection
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Run the Flask application:
```
python app.py
```

Open a web browser and navigate to:
```
http://127.0.0.1:5000/
```

Enter job details in the form and click on “Check Job Authenticity” to receive the prediction.

---

## Future Scope
The project can be extended by integrating real-time job portals and APIs, adding explainable AI techniques to highlight fraud indicators, supporting multi-language job postings, deploying the system on cloud platforms for scalability, extending to multi-class fraud detection, and incorporating user feedback for continuous model improvement.

### Technologies Used
Python, PyTorch, Hugging Face Transformers, DistilBERT, Flask, HTML, CSS

---

## Conclusion
This project demonstrates the practical application of transformer-based NLP models in solving a real-world problem. By combining deep learning with a real-time web application, the system provides a scalable and intelligent solution for detecting fake job postings and helping protect job seekers from online recruitment fraud.

---

## Author

**Sanjana Nathani**  
M.Sc. Data Science, DAU Gandhinagar  
Aspiring Data Scientist  

LinkedIn: [https://www.linkedin.com/in/your-linkedin-username](https://www.linkedin.com/in/sanjana-nathani-26a42727b/)

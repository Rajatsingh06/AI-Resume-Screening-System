# 📄 AI Resume Screening System

An **AI Resume Screening System** built with **Python and Flask** that helps analyze and compare multiple candidate resumes against a given job description.

The application extracts resume information such as **skills, education level, and years of experience**, compares resumes with the job description using **TF-IDF and cosine similarity**, and generates an overall matching score for each candidate.

> **Note:** This application is designed as a decision-support prototype. Recruiters should manually review candidates and should not make employment decisions based only on automated scores.

---

## 🚀 Features

* 📋 Enter a job description manually
* 📄 Upload a job description as a file
* 👥 Upload multiple candidate resumes
* 📑 Supports:

  * PDF
  * DOCX
  * TXT
* 🔍 Automatic skill extraction
* 🎓 Education-level detection
* 💼 Years-of-experience extraction
* 🧠 NLP-based TF-IDF document similarity
* 📊 Skill-match score
* 📈 Experience-match score
* 🎓 Education-match score
* 🔎 Missing-skill identification
* 🏆 Automatic candidate ranking
* 📊 Overall candidate match score
* 🖨️ Print screening results
* 📱 Responsive web interface
* ⚠️ Input validation and error handling
* 📦 Maximum combined upload size of 50 MB

The supported file extensions are PDF, DOCX, and TXT, with a 50 MB maximum request size.

---

## 🛠️ Technologies Used

| Technology        | Purpose                        |
| ----------------- | ------------------------------ |
| Python            | Core programming language      |
| Flask             | Web application framework      |
| PyPDF             | PDF text extraction            |
| python-docx       | DOCX text extraction           |
| Scikit-learn      | NLP and similarity calculation |
| TF-IDF            | Text representation            |
| Cosine Similarity | Resume-JD similarity           |
| HTML/CSS          | Frontend interface             |
| Werkzeug          | Secure filename handling       |

---

## 🧠 How the System Works

The application follows this workflow:

```text
                 ┌─────────────────────┐
                 │   Job Description   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Extract Requirements│
                 │ Skills / Education  │
                 │ Experience          │
                 └──────────┬──────────┘
                            │
                            ▼
        ┌────────────────────────────────────┐
        │       Upload Candidate Resumes     │
        │       PDF / DOCX / TXT             │
        └────────────────┬───────────────────┘
                         │
                         ▼
                ┌─────────────────────┐
                │ Extract Resume Text │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Extract Candidate   │
                │ Skills / Education  │
                │ Experience          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ TF-IDF Similarity   │
                │ + Skill Matching    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Calculate Final     │
                │ Candidate Score     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Rank Candidates     │
                └─────────────────────┘
```

---

## 📊 Scoring Methodology

When recognizable skills are found in the job description, the system calculates the final score using:

| Component               | Weight |
| ----------------------- | -----: |
| Skill Match             |    55% |
| NLP / TF-IDF Similarity |    25% |
| Experience Match        |    10% |
| Education Match         |    10% |

```text
Final Score =
    Skill Match × 0.55
  + Semantic Similarity × 0.25
  + Experience Match × 0.10
  + Education Match × 0.10
```

If no recognized skills are found in the job description, the system uses:

```text
Final Score =
    Semantic Similarity × 0.70
  + Experience Match × 0.20
  + Education Match × 0.10
```

These weighting formulas are implemented directly in the project.

---

## 🔍 Skill Extraction

The system contains a predefined skill dictionary with aliases for common technical and professional skills.

Examples include:

```text
Python
Java
JavaScript
C++
C#
SQL
NoSQL
HTML
CSS
React
Angular
Node.js
Django
Flask
REST API
Git
GitHub
Docker
AWS
Azure
PostgreSQL
MySQL
MongoDB
Pandas
NumPy
Scikit-learn
TensorFlow
PyTorch
Machine Learning
Deep Learning
NLP
Computer Vision
Data Science
Data Analysis
Power BI
Tableau
Excel
Apache Spark
Hadoop
Airflow
Agile
Scrum
Communication
Leadership
Problem Solving
```

The application also handles aliases such as `sklearn`, `ML`, `javascript`, `postgres`, and `golang`.

---

## 🎓 Education Detection

The system identifies education levels from resume and job-description text.

Supported levels include:

```text
Doctorate
Master's Degree
Bachelor's Degree
Associate Degree
Certificate/Diploma
```

The application uses regular-expression patterns to identify these education levels.

---

## 💼 Experience Extraction

The system can identify experience information such as:

```text
3 years experience
5+ years
3-5 years
2018 - 2022
2020 to Present
```

It can also estimate experience from year ranges and uses the current year when a position is marked as `Present` or `Current`.

---

## 📄 Resume Processing

### PDF

PDF text is extracted using `pypdf`.

### DOCX

DOCX content is extracted using `python-docx`, including text contained inside tables.

### TXT

Plain-text resumes are decoded using UTF-8.

The file-processing implementation supports all three formats.

---

## 🖥️ Application Interface

The web interface provides:

### Job Description

Users can either:

* Paste a job description into the text area, or
* Upload a PDF, DOCX, or TXT job-description file.

### Candidate Resumes

Users can upload multiple candidate resumes simultaneously.

### Results

For each candidate, the application displays:

* Candidate rank
* Candidate name
* Resume filename
* Overall match percentage
* Skill match
* NLP similarity
* Experience match
* Education match
* Required experience
* Candidate experience
* Required education
* Candidate education
* Matched skills
* Missing skills
* All detected candidate skills

These result fields are generated by the candidate-analysis function.

---

# 📁 Project Structure

A simple version of the project can be organized as:

```text
AI-Resume-Screening-System/
│
├── app.py
├── requirements.txt
├── README.md
└── resumes/
    └── sample_resume.pdf
```

> The uploaded project currently contains the application logic and HTML/CSS template inside `app.py`.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Screening-System.git
```

Move into the project folder:

```bash
cd AI-Resume-Screening-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Create a `requirements.txt` file:

```text
Flask
Werkzeug
python-docx
pypdf
scikit-learn
```

Then run:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application runs on port `5000` by default.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

or:

```text
http://localhost:5000
```

---

# 📝 How to Use

### Step 1 — Enter Job Description

Paste the complete job description into the **Job Description** field.

For example:

```text
We are looking for a Data Analyst with experience in
Python, SQL, Pandas, NumPy, Power BI and Excel.

The candidate should have a Bachelor's degree
and 2 years of experience.
```

### Step 2 — Upload Resumes

Select one or more:

```text
candidate1.pdf
candidate2.pdf
candidate3.docx
candidate4.txt
```

### Step 3 — Analyze

Click:

```text
Analyze and Rank Candidates
```

### Step 4 — Review Results

The system extracts the job requirements and compares each candidate.

Example:

```text
Candidate: Rajat Singh

Overall Match: 82.5%

Skill Match:        90%
NLP Similarity:     78%
Experience Match:   80%
Education Match:   100%

Matched Skills:
✓ Python
✓ SQL
✓ Pandas
✓ NumPy
✓ Power BI

Missing Skills:
✗ Tableau
```

### Step 5 — Print Results

The application provides a **Print Results** option for the screening results.

---

# 🧮 Candidate Ranking

After calculating the final score, candidates are sorted from the highest score to the lowest score.

The application assigns ranks starting from `#1`.

Example:

```text
#1  Candidate A    91.5%
#2  Candidate B    84.2%
#3  Candidate C    76.8%
#4  Candidate D    69.4%
```

---

# 🔐 File Upload Limit

The application allows a maximum combined request size of:

```text
50 MB
```

Files exceeding this limit return an appropriate error message.

---

# ⚠️ Important Limitations

This project is a **prototype decision-support system**, not a replacement for human recruitment judgment.

The system has several limitations:

* Skill extraction is based on a predefined skill dictionary.
* Skills not present in the dictionary may not be detected.
* Resume formatting can affect text extraction.
* Scanned/image-only PDFs may not provide readable text.
* TF-IDF measures textual similarity rather than true semantic understanding.
* Experience extraction uses pattern matching and is approximate.
* Education detection is also pattern-based.
* Automated scores should not be treated as definitive hiring decisions.

The application itself displays a warning asking recruiters to manually review candidates.

---

# 🔮 Future Enhancements

Possible improvements include:

* 🤖 Transformer-based NLP models
* 🧠 Sentence-BERT semantic embeddings
* 📊 Interactive analytics dashboard
* 📥 Excel/CSV result export
* 📄 Automatic candidate report generation
* 🔎 More advanced resume section extraction
* 🏷️ Custom skill dictionaries
* 🌐 Multi-language resume processing
* 🗃️ Database integration
* 👤 Recruiter authentication
* 📈 Candidate comparison charts
* 🔐 Improved privacy and data-management controls
* ☁️ Cloud deployment
* 📧 Recruiter notification system

---

# 📌 Use Cases

This project can be used as a prototype for:

* HR departments
* Recruitment teams
* Resume shortlisting
* Job application analysis
* Candidate comparison
* Recruitment analytics
* Academic NLP projects
* Machine Learning / Data Science portfolios

---

# 🎯 Project Objective

The main objective of this project is to demonstrate how **Natural Language Processing, information extraction, and similarity analysis** can be combined to assist with resume screening.

The system reduces repetitive manual comparison by automatically extracting relevant information and presenting structured candidate-matching results.

---

# 👨‍💻 Author

**Rajat Singh**

Artificial Intelligence & Data Science

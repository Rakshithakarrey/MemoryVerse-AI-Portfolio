# MemoryVerse AI — Portfolio

MemoryVerse AI is an AI-powered digital portfolio and memory engine built with **Python and Streamlit**. It helps students organize resumes, certificates, project reports, internship letters, achievements, and academic documents in one place.

The application uses an LLM to analyze uploaded documents, automatically categorize them, extract skills, generate summaries, build relationships between documents and skills, create a digital timeline, and provide smart semantic search.

## Features

* 📄 Upload resumes, certificates, project reports, internship letters, marksheets, and achievements
* 🤖 AI-based document categorization
* 🏷️ Automatic skill extraction
* 📝 Automatic document summaries
* 🗄️ SQLite database for storing portfolio information
* 🔗 Document-to-skill relationship mapping
* 🕸️ Interactive knowledge graph
* ⏳ Digital growth timeline
* 🔍 Smart semantic search
* 📥 Download previously uploaded documents
* 📊 Portfolio statistics
* 🎯 Optional OpenAI embeddings for improved semantic search

The application supports **PDF, TXT, PNG, JPG, JPEG, and WEBP** files. 

---

## How the Project Works

The overall workflow is:

```text
                    ┌─────────────────────┐
                    │   Student Documents │
                    │ Resume / Certificate│
                    │ Projects / Internship│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Document Upload   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text / Image        │
                    │ Extraction          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AI Analysis       │
                    │ Category + Skills   │
                    │ Summary + Date      │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       ┌───────────┐     ┌────────────┐    ┌─────────────┐
       │  SQLite   │     │ Knowledge  │    │  Timeline   │
       │ Database  │     │   Graph    │    │             │
       └───────────┘     └────────────┘    └─────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Semantic Search  │
                     └──────────────────┘
```

---

## Main Modules

### 1. Document Upload & Processing

Users can upload portfolio documents through the Streamlit interface.

Supported formats:

```text
PDF
TXT
PNG
JPG
JPEG
WEBP
```

The application extracts text from PDFs using **pdfplumber** and falls back to **PyPDF2** when necessary. For images or scanned PDFs, the application can send the image to the selected AI provider for analysis. 

---

### 2. AI Document Categorization

The application sends the document information to the selected LLM.

Documents are classified into:

```text
Projects
Skills
Certifications
Internships
Achievements
Academics
```

The AI also extracts:

* Document title
* Date
* Skills
* Summary

The project uses a structured JSON format for the AI response. 

---

### 3. Skill Extraction

For example, if a project document contains:

```text
Python
Machine Learning
SQL
Streamlit
```

MemoryVerse stores those skills and connects them with the document.

This allows the application to understand relationships such as:

```text
AI Resume Analyzer
       │
       ├── Python
       ├── Machine Learning
       ├── NLP
       └── Streamlit
```

The database contains a separate `skills` table and a `relationships` table for these connections. 

---

### 4. Digital Timeline

The application creates a timeline of the student's growth based on document dates.

For example:

```text
2024
 │
 ├── Python Certification
 │
2025
 │
 ├── Machine Learning Project
 ├── Internship
 │
2026
 │
 ├── AI Project
 └── New Certification
```

This makes it easier to see how the student's skills and achievements developed over time.

---

### 5. Knowledge Graph

The Knowledge Graph connects documents with their extracted skills.

For example:

```text
Certification ─── Python
                     │
Project ─────── Machine Learning
                     │
Internship ───── SQL
```

The project uses **PyVis** to display an interactive network when the package is available. 

---

### 6. Smart Semantic Search

MemoryVerse includes a smart search system.

Instead of only matching exact keywords, it converts documents and search queries into vectors and calculates their similarity.

For example, you could search:

```text
Show my AI projects
```

and the system ranks the most relevant documents.

The project provides a local hashing-based vector system that works without an external vector database. If enabled, OpenAI's `text-embedding-3-small` can be used for higher-quality embeddings. 

---

# Technologies Used

| Technology | Purpose                           |
| ---------- | --------------------------------- |
| Python     | Main programming language         |
| Streamlit  | Web application interface         |
| SQLite     | Portfolio database                |
| NumPy      | Vector calculations               |
| Pandas     | Data processing                   |
| pdfplumber | PDF text extraction               |
| PyPDF2     | PDF fallback extraction           |
| Pillow     | Image processing                  |
| OpenAI     | AI document analysis / embeddings |
| Anthropic  | Alternative AI provider           |
| PyVis      | Knowledge graph                   |
| HTML/CSS   | UI customization                  |

---

# Project Structure

Create the project like this:

```text
MemoryVerse-AI-Portfolio/
│
├── memoryverse_app.py
├── requirements.txt
├── README.md
│
├── memoryverse.db
│
└── mv_storage/
```

You only need to create `memoryverse_app.py` initially.

The following are created by the application:

```text
memoryverse.db
mv_storage/
```

The SQLite database stores structured metadata, while uploaded files are stored in `mv_storage`. 

---

# Installation

## Step 1 — Install Python

Install **Python 3** on your computer.

Check whether Python is installed:

```powershell
python --version
```

You should see something similar to:

```text
Python 3.x.x
```

---

## Step 2 — Open the Project in VS Code

Open the project folder:

```text
MemoryVerse-AI-Portfolio
```

Then open the VS Code terminal:

**Terminal → New Terminal**

---

## Step 3 — Create a Virtual Environment

Run:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

After activation, you should see:

```text
(venv) PS C:\...\MemoryVerse-AI-Portfolio>
```

---

# Step 4 — Install Dependencies

Run:

```powershell
pip install streamlit numpy pandas pillow pdfplumber PyPDF2 openai anthropic pyvis
```

Or create a `requirements.txt` file containing:

```text
streamlit
numpy
pandas
pillow
pdfplumber
PyPDF2
openai
anthropic
pyvis
```

Then install everything using:

```powershell
pip install -r requirements.txt
```

---

# Step 5 — Run the Application

Make sure your main file is named:

```text
memoryverse_app.py
```

Then run:

```powershell
python -m streamlit run memoryverse_app.py
```

You should see:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

Open:

```text
http://localhost:8501
```

The original project instructions also specify running the application with Streamlit. 

---

# Configuration

When the application opens, the sidebar contains:

```text
⚙️ Configuration

LLM Provider
API Key
Model
```

The application supports:

```text
OpenAI
Anthropic
```

Select your preferred provider and enter its API key. 

**Never upload your API key to GitHub or share it publicly.**

---

# How to Use

### Step 1

Open:

```text
http://localhost:8501
```

### Step 2

Select an LLM provider from the sidebar.

### Step 3

Enter your API key.

### Step 4

Go to:

**📥 Upload & Process**

### Step 5

Upload a document.

For example:

```text
resume.pdf
certificate.pdf
project_report.pdf
internship_letter.pdf
```

### Step 6

Click:

**🚀 Process with AI**

### Step 7

The application analyzes the document and displays:

```text
Category
Title
Date
Summary
Extracted Skills
```

### Step 8

Explore:

**⏳ My Digital Timeline**

to see your academic/professional growth.

### Step 9

Explore:

**🕸 Knowledge Graph**

to see document-skill relationships.

### Step 10

Use:

**🔍 Smart Semantic Search**

to find relevant portfolio documents.

---

# Example

Suppose you upload:

```text
AI_Project.pdf
```

The AI might identify:

```text
Category:
Projects

Title:
AI Resume Analyzer

Skills:
Python
Machine Learning
NLP
Streamlit

Summary:
An AI-based application that analyzes resumes and extracts relevant skills.
```

MemoryVerse then stores this information in the database and creates relationships between the project and its skills.

---

# Database

The project uses SQLite.

Database file:

```text
memoryverse.db
```

The application creates these main tables:

```text
documents
skills
relationships
```

The `documents` table stores document information.

The `skills` table stores unique skills.

The `relationships` table connects documents to skills. 

---

# Running the Project Again

Every time you want to start the application:

```powershell
cd "C:\Users\VISHAL\OneDrive\Desktop\RAKSHITHA\MemoryVerse-AI-Portfolio"
```

Activate the environment:

```powershell
venv\Scripts\activate
```

Run:

```powershell
python -m streamlit run memoryverse_app.py
```

Then open:

```text
http://localhost:8501
```

---

# Troubleshooting

### `streamlit is not recognized`

Use:

```powershell
python -m streamlit run memoryverse_app.py
```

instead of:

```powershell
streamlit run memoryverse_app.py
```

---

### `No module named streamlit`

Run:

```powershell
pip install streamlit
```

---

### `No module named pdfplumber`

Run:

```powershell
pip install pdfplumber
```

---

### `No module named pyvis`

Run:

```powershell
pip install pyvis
```

---

### OpenAI package error

Run:

```powershell
pip install --upgrade openai
```

Then restart Streamlit.

---

### Application asks for an API key

This is expected because AI document categorization uses the selected LLM provider. 

---

# Security

Do **not** put API keys directly into:

```text
memoryverse_app.py
README.md
requirements.txt
GitHub
```

Also don't commit:

```text
memoryverse.db
mv_storage/
venv/
```

to a public GitHub repository if they contain your personal documents.

A useful `.gitignore` would contain:

```text
venv/
__pycache__/
*.pyc
memoryverse.db
mv_storage/
.env
```

---

# Future Improvements

Possible future versions of MemoryVerse AI could include:

* User authentication
* Cloud database
* Resume generation
* Skill-gap analysis
* Career recommendations
* Internship recommendations
* Portfolio website generation
* AI-generated student profile
* Better OCR for scanned documents
* Advanced vector database
* Multi-user support
* Deployment to Streamlit Cloud

---

# Project Objective

The main objective of **MemoryVerse AI** is to transform scattered student documents into an **organized, searchable and connected digital portfolio**.

Instead of keeping:

```text
Resume.pdf
Certificate1.pdf
Certificate2.pdf
ProjectReport.pdf
InternshipLetter.pdf
Achievement.pdf
```

in separate folders, MemoryVerse turns them into structured information:

```text
                 MEMORYVERSE AI
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Documents       Skills        Timeline
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Knowledge Graph
                       │
                       ▼
               Smart Search
```

This makes the project useful as a **student digital portfolio and personal career-memory system**, rather than just a document uploader.

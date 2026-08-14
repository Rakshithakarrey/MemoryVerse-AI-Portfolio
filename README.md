# MemoryVerse AI

MemoryVerse AI is an AI-powered digital portfolio and memory engine that organizes student documents such as resumes, certificates, project reports, and internship records. It uses Large Language Models (LLMs) to categorize documents, extract skills, build a knowledge graph, create a learning timeline, and provide smart semantic search for portfolio insights.

## Features

* 📄 **Document Upload** – Upload resumes, certificates, project reports, and internship documents.
* 🤖 **AI Document Categorization** – Automatically analyzes and categorizes uploaded documents.
* 🏷️ **Skill Extraction** – Identifies important skills and keywords from documents.
* 🧠 **Knowledge Graph** – Creates relationships between documents, skills, projects, and experiences.
* 📅 **Learning Timeline** – Organizes academic and professional activities chronologically.
* 🔍 **Semantic Search** – Finds relevant portfolio information based on meaning rather than only exact keywords.
* 📊 **Portfolio Insights** – Provides an organized view of a student's skills, experiences, and achievements.
* 💾 **Data Storage** – Stores document and relationship information for later access.

## Technologies Used

* **Python 3**
* **Streamlit** – Web application interface
* **OpenAI / Anthropic APIs** – AI-powered document analysis
* **SQLite** – Database storage
* **PyPDF2 / pdfplumber** – PDF text extraction
* **Pillow (PIL)** – Image processing
* **PyVis** – Knowledge graph visualization
* **NumPy & Pandas** – Data processing
* **JSON** – Structured data handling

## How It Works

```text
        Student Documents
               ↓
        Document Upload
               ↓
       Text / Data Extraction
               ↓
          AI Analysis
               ↓
      ┌────────┼─────────┐
      ↓        ↓         ↓
  Category   Skills    Keywords
      ↓        ↓         ↓
      └────────┼─────────┘
               ↓
         Data Storage
               ↓
    ┌──────────┼───────────┐
    ↓          ↓           ↓
Knowledge   Learning    Semantic
  Graph     Timeline     Search
```

## Project Structure

```text
MemoryVerse-AI/
│
├── memoryverse_app.py
├── memoryverse.db
├── mv_storage/
└── README.md
```

### File Description

| File/Folder          | Description                                        |
| -------------------- | -------------------------------------------------- |
| `memoryverse_app.py` | Main Streamlit application                         |
| `memoryverse.db`     | SQLite database containing structured project data |
| `mv_storage/`        | Stores application/document-related data           |
| `README.md`          | Project documentation                              |

## Installation

Clone the repository and open the project folder in VS Code.

Install the required Python packages:

```bash
pip install streamlit numpy pandas pillow pdfplumber PyPDF2 openai anthropic pyvis
```

## How to Run

Run the following command in the VS Code terminal:

```bash
streamlit run memoryverse_app.py
```

The application will open in your browser.

## How to Use

1. Start the Streamlit application.
2. Upload a resume, certificate, project report, or internship document.
3. Select the AI provider.
4. Provide the required API key.
5. Process the document.
6. View the automatically generated category and skills.
7. Explore the knowledge graph and learning timeline.
8. Use semantic search to find relevant portfolio information.

## Example

A student uploads a **Python Internship Certificate**.

The system can identify information such as:

```text
Category: Internship
Skills: Python, Programming
Experience: Python Internship
```

The extracted information can then be connected with related skills and experiences in the student's portfolio.

## Use Cases

* Student portfolio management
* Resume organization
* Certificate management
* Internship tracking
* Project documentation
* Skill tracking
* Academic progress visualization
* Career portfolio analysis

## Future Enhancements

* User authentication
* Cloud database integration
* More document formats
* Improved AI recommendations
* Automatic resume generation
* Skill-gap analysis
* Portfolio PDF generation
* Cloud deployment



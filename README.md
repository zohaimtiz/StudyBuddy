# 📘 Study Buddy — AI Learning Assistant

Study Buddy is a Python + Gradio based AI learning assistant that helps students convert PDF study material into focused learning resources. It can generate summaries, quizzes, active recall practice, flashcards, study plans, mindmaps, downloadable files, and a progress dashboard.

The app is designed to run easily in **Google Colab** or locally as a Python script.

---

## 🚀 Main Features

### 1. User Authentication
- Sign up with name, email, and password.
- Sign in using registered email and password.
- Passwords are stored as SHA-256 hashes in `users.json`.
- Each user's email is used as their student ID for progress tracking.

### 2. PDF Upload and Topic-Based Study Material
- Upload one or more PDF files.
- Optionally enter a specific topic.
- If a topic is provided, the app tries to generate content only from that topic.
- If the topic is not found in the uploaded PDF, the app shows a clear topic-not-found message.

### 3. PDF Cleaning
- Removes unnecessary title-page/admin content such as:
  - university names
  - department names
  - professor names
  - roll numbers
  - page numbers
  - headers and footers
- Keeps useful educational content such as definitions, concepts, examples, formulas, and explanations.

### 4. AI Summary Generation
The app generates a layered summary with sections such as:
- One-line overview
- Key concepts
- Detailed explanation
- Important definitions
- Exam-focused points
- Common mistakes

### 5. Quiz Generation
- Generates MCQs from the uploaded PDF or selected topic.
- Supports different question types:
  - concept
  - definition
  - scenario
  - application
  - comparison
  - example
- Shows options in clean interactive cards.
- Provides the correct answer and explanation.

### 6. Active Recall Practice
- Generates fresh practice questions every time a new session starts.
- Uses RAG-based retrieval from the uploaded notes.
- Tracks correct and wrong answers.
- Saves weak questions for later review.

### 7. Weak Questions Review
- Stores wrong questions separately.
- Deduplicates weak questions so the same question is not saved again and again.
- Shows mistake count, topic, correct answer, and explanation.

### 8. Study Plan
- Creates a clean table-based study plan.
- Study days can be selected by the user.
- The plan includes:
  - day
  - topic focus
  - key points
  - practice task
  - expected output

### 9. Flashcards
- Generates front/back flashcards.
- Exports flashcards as a PDF.
- Can include a QR-style reference section when QR support is available.

### 10. Mindmap
- Creates a 3-level interactive mindmap.
- Mindmap is collapsible/clickable in the Gradio interface.
- Can also be downloaded as an HTML file.

### 11. Downloads
The app can generate downloadable files including:
- Summary PDF
- Quiz PDF
- Active Recall PDF
- Flashcards PDF
- Study Plan PDF
- Mindmap HTML
- Progress Report PDF
- PPT slides from summary

### 12. Progress Dashboard
The app tracks:
- PDFs/topics studied
- quizzes generated
- attempted questions
- correct answers
- wrong answers
- accuracy percentage
- weak topics
- last studied topic
- last study time

### 13. Ask Questions from Notes
- User can ask questions from uploaded notes.
- Supports English, Roman Urdu, and Urdu.
- Uses RAG to retrieve relevant content before answering.
- If the answer is not available in the uploaded content, the app tells the user clearly.

---

## 🛠️ Technologies Used

- **Python** — main programming language
- **Gradio** — web app interface
- **OpenAI API** — AI response generation
- **PyPDF** — PDF text extraction
- **Scikit-learn** — TF-IDF and cosine similarity for RAG
- **FPDF2** — PDF generation
- **python-pptx** — PowerPoint generation
- **QRCode** — QR code support for flashcards
- **Matplotlib** — progress charts
- **JSON** — local storage for users and progress

---

## 📁 Project Structure

```text
study-buddy/
│
├── app.py                  # Main application file
├── README.md               # Project documentation
│
├── users.json              # Auto-created file for user accounts
├── study_progress.json     # Auto-created file for student progress
│
└── temp/download files      # Auto-generated PDF/PPT/HTML files during use
```

> Note: `users.json` and `study_progress.json` are created automatically when the app runs.

---

## ✅ Requirements

You need:

- Python 3.9 or above
- OpenAI API key
- Internet connection for AI API calls
- Google Colab or local Python environment

Required Python packages:

```bash
pip install openai gradio pypdf fpdf2 python-pptx qrcode[pil] scikit-learn matplotlib
```

The current `app.py` also includes auto-install logic, so packages are installed automatically when the file runs in Colab.

---

## 🔑 OpenAI API Key Setup

### Recommended Method

Use an environment variable instead of writing the API key directly in the code.

#### In Google Colab

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

#### On Windows PowerShell

```powershell
setx OPENAI_API_KEY "your-api-key-here"
```

Then restart the terminal or IDE.

#### On macOS/Linux

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## ⚠️ Important Security Note

Do **not** upload or share your real OpenAI API key publicly.

If an API key has already been shared inside the code or uploaded anywhere, you should:

1. Revoke/delete that key from the OpenAI dashboard.
2. Create a new key.
3. Store the new key using environment variables.
4. Avoid committing secrets to GitHub or public folders.

For safer code, you can replace the direct API key line with:

```python
API_KEY = os.environ.get("OPENAI_API_KEY", "")
```

---

## ▶️ How to Run the App

### Option 1: Run in Google Colab

1. Open Google Colab.
2. Upload `app.py`.
3. Add your OpenAI API key using environment variable.
4. Run the full script.
5. A public Gradio link will appear.
6. Open the link and use the app.

### Option 2: Run Locally

1. Open the project folder.
2. Install dependencies:

```bash
pip install openai gradio pypdf fpdf2 python-pptx qrcode[pil] scikit-learn matplotlib
```

3. Set your OpenAI API key.
4. Run the app:

```bash
python app.py
```

5. Open the Gradio link shown in the terminal.

---

## 👨‍🎓 How to Use

1. Open the app.
2. Create an account from the **Sign Up** tab.
3. Sign in using your email and password.
4. Upload a PDF file.
5. Enter a topic if you want topic-specific notes.
6. Click **Generate Study Material**.
7. Review the generated summary, quiz, study plan, and mindmap.
8. Start active recall practice.
9. Check weak questions.
10. Download PDFs, PPT, or mindmap files.
11. Ask questions from uploaded notes in English, Roman Urdu, or Urdu.

---

## 📊 Progress Tracking

Progress is saved in `study_progress.json`.

The app tracks each student's:

- number of PDFs studied
- number of quizzes generated
- attempted questions
- correct answers
- wrong answers
- weak topics
- weak questions
- last studied topic
- last study date/time

---

## 🧠 RAG System

The app uses a simple RAG approach:

1. Notes are split into chunks.
2. TF-IDF converts chunks into vectors.
3. Cosine similarity finds relevant chunks.
4. The AI answers using the retrieved chunks.

RAG is used mainly for:

- Ask Questions from Notes
- Active Recall Practice

If Scikit-learn is not available, the app uses a keyword-based fallback.

---

## 📥 Generated Output Files

The app can create:

| File Type | Purpose |
|---|---|
| Summary PDF | Clean notes summary |
| Quiz PDF | MCQs with answers and explanations |
| Recall PDF | Active recall question set |
| Flashcards PDF | Front/back study cards |
| Study Plan PDF | Day-wise learning plan |
| Mindmap HTML | Interactive topic map |
| PPTX | Summary slides |
| Progress PDF | Charts and student progress |

---

## ⚠️ Limitations

- Scanned image-based PDFs may not extract text correctly because OCR is not included.
- Local JSON storage is suitable for demos or small projects, not large production systems.
- SHA-256 password hashing is better than plain text, but production apps should use stronger password hashing such as bcrypt or Argon2.
- AI output depends on the quality of PDF text extraction.
- A valid OpenAI API key is required for AI features.

---

## 🔧 Troubleshooting

### AI is not configured
Make sure your OpenAI API key is set correctly.

### PDF text is empty
The PDF may be scanned or image-based. Use a text-based PDF or add OCR support.

### Package installation fails
Run the install command manually:

```bash
pip install openai gradio pypdf fpdf2 python-pptx qrcode[pil] scikit-learn matplotlib
```

### Gradio public link does not appear
Check internet connection and make sure `demo.launch(share=True)` is enabled.

### Quiz or summary is not generated
Check:
- OpenAI API key
- internet connection
- API rate limits
- PDF text extraction

---

## 🌱 Future Improvements

Possible improvements:

- Add OCR for scanned PDFs.
- Add database support using SQLite, PostgreSQL, or Firebase.
- Add role-based admin dashboard.
- Add email verification for signup.
- Add forgot password flow.
- Add better password hashing using bcrypt or Argon2.
- Add export to Word document.
- Add support for more file types such as DOCX and PPTX.
- Add chapter-wise study mode.
- Add timed quiz mode.
- Add teacher dashboard to monitor student progress.

---

## 📌 Project Summary

Study Buddy is an AI-powered learning assistant that helps students study faster and more effectively. It converts PDFs into summaries, quizzes, flashcards, study plans, mindmaps, and practice questions. It also tracks student progress and weak areas, making it useful for exam preparation and self-study.

---

## 👤 Author / Developer

Project Name: **Study Buddy — AI Learning Assistant**  
Version: **v3**  
Main File: **app.py**


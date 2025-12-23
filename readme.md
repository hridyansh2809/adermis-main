# 🌿 Adermis Main Repository

Adermis is a full-stack AI-powered **skin disease prediction platform**. This main repository acts as the **root project**, containing two major components:

* **Adermis1/** → Complete web application (Frontend + Backend)
* **model-ml/** → Machine Learning model training scripts

This README explains the overall architecture, setup, workflow, and development structure for the entire project.

---

# 📁 Repository Structure

```
adermis-main/
│
├── Adermis1/                 # Full website: frontend + python backend + APIs
│   ├── frontend/             # Next.js + TailwindCSS web app
│   ├── backend/              # Python (Flask / FastAPI) server + prediction APIs
│   └── README.md             # Detailed project-level readme
│
├── model-ml/                 # Machine learning model training code
│   ├── dataset/              # Training dataset (if included or referenced)
│   ├── train.py              # Model training script
│   ├── preprocess.py         # Image preprocessing steps
│   └── model/                # Saved weights, architectures, checkpoints
│
└── README.md (this file)
```

---

# 🚀 Project Overview

Adermis is an AI-driven dermatology assistant that:

* Predicts **skin diseases** from uploaded images.
* Asks **follow-up questions** to refine the diagnosis.
* Generates a structured **treatment plan** (Ayurvedic, home remedies, OTC, prescription).
* Uses a scalable **Next.js frontend** and **Python backend**.
* Includes an ML pipeline for model training and continuous improvement.

---

# 🧠 System Architecture

```
Frontend (Next.js)
    ↓
API Gateway (Next.js API routes or Backend routing)
    ↓
ML Server (Python)
    ├── Image Classification Model
    └── Follow-up Question Logic
```

### 🔹 Frontend (Adermis1/frontend)

* Built with **Next.js + TailwindCSS**
* Handles UI, image upload, forms, results page
* Securely communicates with backend via binary-encoded requests

### 🔹 Backend (Adermis1/backend)

* Python (Flask / FastAPI)
* Handles:

  * Image processing & prediction
  * Follow-up logic
  * Treatment page response formatting

### 🔹 ML Model (model-ml/)

* Contains all ML training scripts
* Includes preprocessing, augmentation, model architecture, checkpoints
* Supports improvement through feedback loop (planned)

---

# 🛠️ Getting Started

## 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/adermis-main.git
cd adermis-main
```

## 2️⃣ Set up the Backend

```bash
cd Adermis1/backend
pip install -r requirements.txt
python app.py
```

## 3️⃣ Set up the Frontend

```bash
cd ../frontend
npm install
npm run dev
```

## 4️⃣ ML Training (Optional)

```bash
cd ../../model-ml
python train.py
```

---

# 🧪 Testing Flow

1. Open the web app.
2. Upload a skin image.
3. Answer follow-up questions (dynamic based on prediction).
4. View final **disease** + **treatment breakdown**.
5. Validate workflow end-to-end.

---

# 🧾 Treatment Output Format

The final solution page shows treatments in this order:

1. **Ayurvedic Solution** 🪔
2. **Home Remedies** 🏡
3. **Over-the-counter (OTC)** 💊
4. **Prescription Drugs** 🧾

Each section contains short 1–2 line actionable suggestions.

---

# 📚 Future Enhancements

* User authentication + history tracking
* Multi-language support
* Integration with dermatologists / telehealth
* Active feedback loop for improving model accuracy

---

# 👨‍⚕️ Disclaimer

This system is for **educational and informational** purposes only.
Not a substitute for professional dermatological consultation.

---

# 🤝 Contributing

```
fork → create branch → commit → push → PR
```

```bash
git checkout -b feature/xyz
git commit -am "Add xyz"
git push origin feature/xyz
```

Feel free to contribute to model improvements, UI design, or new diagnosis logic.

---

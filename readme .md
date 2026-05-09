# CS Students Analytics Dashboard

## 📊 Overview
Interactive dashboard showcasing analysis of 180 Computer Science students across 6 labs.

## 🚀 Live Demo
[Your Streamlit URL after deployment]

## 📁 Data Source
- **CS Students Dataset:** 180 records with demographics, programming skills, GPA, domain interests
- **Books Dataset:** 1000 books scraped from books.toscrape.com (Lab 3)

## 🔬 Labs Included

| Lab | Topic | Key Findings |
|-----|-------|--------------|
| Lab 0 | Setup & Imports | Libraries configured |
| Lab 1 | Descriptive Statistics | GPA range 3.20-3.90, mean 3.62 |
| Lab 2 | Inferential Statistics | Python skill significantly impacts GPA (p < 0.001) |
| Lab 3 | Web Scraping | 1000 books scraped with ratings & prices |
| Lab 4 | Data Preprocessing | Cleaned dataset, encoded skills, removed outliers |
| Lab 5 | EDA | Python has strongest correlation with GPA (r=0.35) |

## 🛠️ Technologies
- Python 3.9+
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- SciPy, Plotly

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-username/cs-students-dashboard.git
cd cs-students-dashboard

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
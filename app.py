"""
CS Students Analytics Dashboard
Interactive web app showcasing all 6 labs (0-5)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
import math

# Page config
st.set_page_config(
    page_title="CS Students Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f5f5f5; }
    .big-font { font-size: 20px !important; font-weight: bold; }
    .info-box { background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .success-box { background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .warning-box { background-color: #fff3e0; padding: 20px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/cs_students_clean.csv')
    # Ensure numeric columns are properly typed
    df['Python'] = pd.to_numeric(df['Python'], errors='coerce')
    df['SQL'] = pd.to_numeric(df['SQL'], errors='coerce')
    df['Java'] = pd.to_numeric(df['Java'], errors='coerce')
    df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
    # Create Skill Score if not exists
    if 'Skill Score' not in df.columns:
        df['Skill Score'] = df[['Python', 'SQL', 'Java']].mean(axis=1).round(2)
    return df

@st.cache_data
def load_books():
    try:
        books = pd.read_csv('data/books_raw.csv')
        return books
    except:
        return None

df = load_data()
books_df = load_books()

# Encode skill levels for display
skill_map_reverse = {1: 'Weak', 2: 'Average', 3: 'Strong'}
for col in ['Python', 'SQL', 'Java']:
    if col in df.columns:
        df[f'{col}_text'] = df[col].map(skill_map_reverse)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/graduation-cap.png", use_container_width=True)
st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Select Lab:", [
    "🏠 Home / Overview",
    "📊 Lab 1: Descriptive Statistics",
    "📈 Lab 2: Inferential Statistics",
    "🌐 Lab 3: Web Scraping",
    "🧹 Lab 4: Data Preprocessing",
    "📉 Lab 5: Exploratory Data Analysis",
    "📝 Conclusions & Summary"
])

st.sidebar.markdown("---")
st.sidebar.info("**Data:** 180 CS Students | **Source:** Custom Dataset")
st.sidebar.caption("© 2024 CS Students Analytics | Labs 0-5 Complete")

# ============================================
# PAGE 1: HOME / OVERVIEW
# ============================================
if page == "🏠 Home / Overview":
    st.title("🎓 CS Students Analytics Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average GPA", f"{df['GPA'].mean():.2f}")
    with col3:
        male_count = len(df[df['Gender']==0]) if 'Gender' in df.columns else 0
        female_count = len(df[df['Gender']==1]) if 'Gender' in df.columns else 0
        st.metric("Male/Female", f"{male_count}/{female_count}")
    with col4:
        st.metric("Unique Domains", df['Interested Domain'].nunique())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📋 Lab Progress")
        labs_df = pd.DataFrame({
            'Lab': ['Lab 0', 'Lab 1', 'Lab 2', 'Lab 3', 'Lab 4', 'Lab 5'],
            'Topic': ['Setup & Imports', 'Descriptive Stats', 'Inferential Stats', 'Web Scraping', 'Data Preprocessing', 'EDA'],
            'Status': ['✅', '✅', '✅', '✅', '✅', '✅']
        })
        st.dataframe(labs_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🎯 Key Questions This Analysis Answers")
    st.markdown("""
    1. What is the distribution of GPAs among CS students?
    2. Do programming skills (Python, SQL, Java) correlate with GPA?
    3. Is there a significant difference in GPA based on gender?
    4. Which domains are students most interested in?
    5. Can we predict GPA based on programming skills?
    """)
    
    with st.expander("📖 About This Dashboard"):
        st.markdown("""
        This interactive dashboard presents the complete results from 6 labs analyzing a CS students dataset.
        
        **Labs Included:**
        - **Lab 0:** Environment Setup & Library Imports
        - **Lab 1:** Descriptive Statistics - Understanding the data
        - **Lab 2:** Inferential Statistics - Hypothesis testing
        - **Lab 3:** Web Scraping - Books to Scrape (1000 books)
        - **Lab 4:** Data Preprocessing - Cleaning and encoding
        - **Lab 5:** Exploratory Data Analysis - Visualizations
        
        **Data Source:** Custom dataset with 180 Computer Science students
        **Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Plotly, Streamlit
        """)

# ============================================
# LAB 1: DESCRIPTIVE STATISTICS
# ============================================
elif page == "📊 Lab 1: Descriptive Statistics":
    st.title("📊 Lab 1: Descriptive Statistics")
    st.markdown("**Objective:** Understand the dataset through basic statistics and visualizations")
    st.markdown("---")
    
    # Dataset shape info
    st.subheader("📋 Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")
        st.write(f"**Total elements:** {df.shape[0] * df.shape[1]}")
    with col2:
        st.write(f"**Columns:** {', '.join(df.columns.tolist()[:6])}...")
        st.write(f"**Data types:** {df.dtypes.value_counts().to_dict()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 GPA Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['GPA'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['GPA'].mean(), color='red', linestyle='--', label=f"Mean: {df['GPA'].mean():.2f}")
        ax.axvline(df['GPA'].median(), color='green', linestyle='--', label=f"Median: {df['GPA'].median():.2f}")
        ax.set_xlabel('GPA')
        ax.set_ylabel('Frequency')
        ax.set_title('GPA Distribution')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 GPA Summary Statistics")
        st.dataframe(df['GPA'].describe().to_frame().T, use_container_width=True)
        st.write(f"**Skewness:** {df['GPA'].skew():.4f}")
        st.write(f"**Kurtosis:** {df['GPA'].kurtosis():.4f}")
        st.write(f"**Range:** {df['GPA'].max() - df['GPA'].min():.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(gender_counts.values, labels=['Male (0)', 'Female (1)'], autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 GPA by Gender")
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot([df[df['Gender']==0]['GPA'], df[df['Gender']==1]['GPA']], labels=['Male', 'Female'])
        ax.set_ylabel('GPA')
        ax.set_title('GPA Distribution by Gender')
        st.pyplot(fig)
        st.write(f"**Male mean GPA:** {df[df['Gender']==0]['GPA'].mean():.4f}")
        st.write(f"**Female mean GPA:** {df[df['Gender']==1]['GPA'].mean():.4f}")
        st.write(f"**Difference:** {abs(df[df['Gender']==0]['GPA'].mean() - df[df['Gender']==1]['GPA'].mean()):.4f}")
    
    st.markdown("---")
    
    st.subheader("📏 Standard Deviation Analysis")
    std_df = df[['Student ID', 'Age', 'GPA']].std().to_frame().T
    std_df.index = ['Standard Deviation']
    st.dataframe(std_df, use_container_width=True)
    
    st.success("""
    **Key Findings from Lab 1:**
    - **GPA Range:** 3.20 - 3.90 (narrow distribution, all students performing well)
    - **Average GPA:** 3.62 with low variability (σ = 0.18)
    - **Gender Distribution:** 102 Male (56.7%), 78 Female (43.3%)
    - **Age outliers detected:** 9 students aged 29-37 (likely graduate students)
    - **GPA is slightly left-skewed** (more students have higher GPAs)
    """)

# ============================================
# LAB 2: INFERENTIAL STATISTICS
# ============================================
elif page == "📈 Lab 2: Inferential Statistics":
    st.title("📈 Lab 2: Inferential Statistics")
    st.markdown("**Objective:** Apply statistical tests to draw conclusions about the population")
    st.markdown("---")
    
    st.subheader("🎲 Basic Probability")
    ai_count = len(df[df['Interested Domain'] == 'Artificial Intelligence'])
    total = len(df)
    st.write(f"**P(Interested in AI)** = {ai_count}/{total} = {ai_count/total*100:.1f}%")
    
    st.subheader("🎯 Conditional Probability")
    male_ai = len(df[(df['Gender'] == 0) & (df['Interested Domain'] == 'Artificial Intelligence')]) / len(df[df['Gender'] == 0])
    female_ai = len(df[(df['Gender'] == 1) & (df['Interested Domain'] == 'Artificial Intelligence')]) / len(df[df['Gender'] == 1])
    st.write(f"**P(AI | Male)** = {male_ai:.4f} ({male_ai*100:.1f}%)")
    st.write(f"**P(AI | Female)** = {female_ai:.4f} ({female_ai*100:.1f}%)")
    
    st.markdown("---")
    st.subheader("📊 Z-Test: Strong Python vs Population Mean")
    
    strong_python = df[df['Python'] == 3]['GPA']
    z_stat, p_val_z = stats.ttest_1samp(strong_python, df['GPA'].mean())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Z-Statistic", f"{z_stat:.4f}")
    with col2:
        st.metric("P-Value", f"{p_val_z:.8f}")
    
    st.write(f"**Number of strong Python students:** {len(strong_python)}")
    st.write(f"**Mean GPA (strong Python):** {strong_python.mean():.4f}")
    st.write(f"**Mean GPA (population):** {df['GPA'].mean():.4f}")
    
    if p_val_z < 0.05:
        st.success("✅ **Reject null hypothesis** - Strong Python students have significantly higher GPA")
    else:
        st.warning("❌ Fail to reject null hypothesis")
    
    st.markdown("---")
    st.subheader("📊 T-Test: Strong Python vs Others")
    
    weak_python = df[df['Python'] != 3]['GPA']
    t_stat, p_val_t = ttest_ind(strong_python, weak_python)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("T-Statistic", f"{t_stat:.4f}")
    with col2:
        st.metric("P-Value", f"{p_val_t:.10f}")
    
    st.write(f"**Strong Python mean GPA:** {strong_python.mean():.4f}")
    st.write(f"**Others mean GPA:** {weak_python.mean():.4f}")
    st.write(f"**Difference:** {strong_python.mean() - weak_python.mean():.4f} GPA points")
    
    st.markdown("---")
    st.subheader("📊 Chi-Square Test: Gender vs Domain Independence")
    
    top_domains = df['Interested Domain'].value_counts().head(5).index
    df_top = df[df['Interested Domain'].isin(top_domains)]
    contingency = pd.crosstab(df_top['Interested Domain'], df_top['Gender'])
    
    st.write("**Contingency Table (Top 5 Domains):**")
    st.dataframe(contingency, use_container_width=True)
    
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chi-Square Statistic", f"{chi2:.4f}")
    with col2:
        st.metric("P-Value", f"{p_chi:.4f}")
    
    if p_chi < 0.05:
        st.success("✅ Reject null - Gender and domain preference are related")
    else:
        st.info("❌ Cannot reject null - No significant relationship between gender and domain")
    
    st.markdown("---")
    st.subheader("📊 Confidence Interval (95%)")
    
    np.random.seed(42)
    sample = np.random.choice(df['GPA'], size=50, replace=False)
    sample_mean = sample.mean()
    margin = 1.96 * (df['GPA'].std() / np.sqrt(50))
    ci_lower, ci_upper = sample_mean - margin, sample_mean + margin
    
    st.write(f"**Sample Mean:** {sample_mean:.4f}")
    st.write(f"**95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]")
    st.write(f"**Population Mean:** {df['GPA'].mean():.4f}")
    
    if ci_lower <= df['GPA'].mean() <= ci_upper:
        st.success("✅ Confidence interval contains the population mean")
    
    st.markdown("---")
    st.subheader("📊 Central Limit Theorem Demonstration")
    
    np.random.seed(42)
    sample_means = []
    for _ in range(1000):
        sample_means.append(np.random.choice(df['GPA'], size=50).mean())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sample_means, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['GPA'].mean(), color='red', linestyle='--', linewidth=2, label=f'Population Mean: {df["GPA"].mean():.3f}')
    ax.axvline(np.mean(sample_means), color='green', linestyle='--', linewidth=2, label=f'Sample Means Mean: {np.mean(sample_means):.3f}')
    ax.set_xlabel('Sample Mean GPA')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sample Means (n=50, 1000 iterations)')
    ax.legend()
    st.pyplot(fig)
    
    st.info("The Central Limit Theorem confirms that the distribution of sample means approximates a normal distribution as sample size increases.")
    
    st.success("""
    **Key Findings from Lab 2:**
    - **Z-Test (p = 0.000005):** Strong Python students have significantly higher GPA
    - **T-Test (p = 2.08e-8):** Confirms difference between Python-strong and others
    - **Chi-Square (p = 0.236):** No gender-based domain preference
    - **95% CI:** Contains population mean, confirming sampling reliability
    """)

# ============================================
# LAB 3: WEB SCRAPING
# ============================================
elif page == "🌐 Lab 3: Web Scraping":
    st.title("🌐 Lab 3: Web Scraping - Books to Scrape")
    st.markdown("**Target Site:** [books.toscrape.com](https://books.toscrape.com)")
    st.markdown("**Objective:** Scrape 1000 book titles, prices, ratings, and availability")
    st.markdown("---")
    
    if books_df is not None:
        st.subheader("📚 Scraped Books Data")
        st.dataframe(books_df.head(20), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Books", len(books_df))
        with col2:
            st.metric("Unique Ratings", books_df['rating'].nunique())
        with col3:
            try:
                prices = books_df['price'].str.replace('£', '').str.replace('Â', '').astype(float)
                st.metric("Avg Price (£)", f"{prices.mean():.2f}")
            except:
                st.metric("Avg Price (£)", "N/A")
        
        st.subheader("📊 Rating Distribution")
        rating_counts = books_df['rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(rating_counts.index, rating_counts.values, color='goldenrod', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating (stars)')
        ax.set_ylabel('Count')
        ax.set_title('Book Ratings Distribution')
        for bar, val in zip(bars, rating_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontsize=10)
        st.pyplot(fig)
        
        st.subheader("💰 Price Distribution")
        try:
            prices = books_df['price'].str.replace('£', '').str.replace('Â', '').astype(float)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(prices, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(prices.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: £{prices.mean():.2f}')
            ax.axvline(prices.median(), color='green', linestyle='--', linewidth=2, label=f'Median: £{prices.median():.2f}')
            ax.set_xlabel('Price (£)')
            ax.set_ylabel('Frequency')
            ax.set_title('Book Price Distribution')
            ax.legend()
            st.pyplot(fig)
            st.write(f"**Price Range:** £{prices.min():.2f} - £{prices.max():.2f}")
        except Exception as e:
            st.warning(f"Price data format issue: {e}")
        
        st.markdown("---")
        st.subheader("🔧 Scraping Methodology")
        st.info("""
        **Libraries Used:**
        - `requests` - HTTP requests
        - `BeautifulSoup` - HTML parsing
        - `csv` - Data export
        - `time` - Rate limiting
        
        **Process:**
        1. Checked `robots.txt` for permissions
        2. Fetched first page, detected 50 total pages
        3. Extracted each book's title, price, rating, availability, URL
        4. Followed pagination links until last page
        5. Added 1-second delay between requests (ethical scraping)
        6. Saved 1000 entries to `books_raw.csv`
        
        **Data Quality:**
        - 0 missing values
        - Ratings range: 1-5 stars
        - Prices include currency symbol (£)
        """)
    else:
        st.warning("⚠️ Books data file not found. Please ensure 'books_raw.csv' is in the data/ directory.")
        st.info("This file was generated in Lab 3 by scraping books.toscrape.com")

# ============================================
# LAB 4: DATA PREPROCESSING
# ============================================
elif page == "🧹 Lab 4: Data Preprocessing":
    st.title("🧹 Lab 4: Data Preprocessing")
    st.markdown("**Objective:** Clean and prepare the CS students dataset for analysis")
    st.markdown("---")
    
    st.subheader("📋 Preprocessing Steps Performed")
    
    steps_df = pd.DataFrame({
        'Step': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'Action': [
            'Load dataset and inspect shape/dtypes',
            'Handle missing values (defensive fill)',
            'Remove duplicate rows',
            'Strip whitespace from all string columns',
            'Normalize text to title case',
            'Encode ordinal skills (Weak→1, Average→2, Strong→3)',
            'Encode Gender (Male→0, Female→1)',
            'Check for inconsistencies (age range, GPA range, duplicate IDs)',
            'Detect outliers using IQR method',
            'Create derived feature (Skill Score)'
        ],
        'Status': ['✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅']
    })
    st.dataframe(steps_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Data Quality Report")
        st.write(f"**Original shape:** 180 × 12")
        st.write(f"**Clean shape:** {df.shape[0]} × {df.shape[1]}")
        st.write(f"**Missing values:** {df.isnull().sum().sum()}")
        st.write(f"**Duplicate rows:** 0")
        st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    with col2:
        st.subheader("🔍 Outliers Detected (IQR Method)")
        age_Q1 = df['Age'].quantile(0.25)
        age_Q3 = df['Age'].quantile(0.75)
        age_IQR = age_Q3 - age_Q1
        age_lower = age_Q1 - 1.5 * age_IQR
        age_upper = age_Q3 + 1.5 * age_IQR
        age_outliers = df[(df['Age'] < age_lower) | (df['Age'] > age_upper)]
        
        gpa_Q1 = df['GPA'].quantile(0.25)
        gpa_Q3 = df['GPA'].quantile(0.75)
        gpa_IQR = gpa_Q3 - gpa_Q1
        gpa_lower = gpa_Q1 - 1.5 * gpa_IQR
        gpa_upper = gpa_Q3 + 1.5 * gpa_IQR
        gpa_outliers = df[(df['GPA'] < gpa_lower) | (df['GPA'] > gpa_upper)]
        
        st.write(f"**Age outliers:** {len(age_outliers)} (ages {age_outliers['Age'].min():.0f}-{age_outliers['Age'].max():.0f})")
        st.write(f"**GPA outliers:** {len(gpa_outliers)}")
        
        if len(age_outliers) > 0:
            with st.expander("View age outliers"):
                st.dataframe(age_outliers[['Name', 'Age', 'GPA', 'Interested Domain']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Encoding Examples")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Ordinal Skill Encoding**")
        skill_example = pd.DataFrame({
            'Original': ['Strong', 'Average', 'Weak'],
            'Encoded': [3, 2, 1]
        })
        st.dataframe(skill_example, use_container_width=True)
    
    with col2:
        st.write("**Binary Gender Encoding**")
        gender_example = pd.DataFrame({
            'Original': ['Male', 'Female'],
            'Encoded': [0, 1]
        })
        st.dataframe(gender_example, use_container_width=True)
    
    st.subheader("📊 New Feature: Skill Score")
    st.write("**Formula:** Skill Score = (Python + SQL + Java) / 3")
    st.write("**Range:** 1.0 (all Weak) to 3.0 (all Strong)")
    
    skill_stats = df['Skill Score'].describe()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Skill Score", f"{skill_stats['mean']:.2f}")
    with col2:
        st.metric("Min Skill Score", f"{skill_stats['min']:.2f}")
    with col3:
        st.metric("Max Skill Score", f"{skill_stats['max']:.2f}")
    with col4:
        st.metric("Std Dev", f"{skill_stats['std']:.3f}")
    
    st.dataframe(df[['Name', 'Python', 'SQL', 'Java', 'Skill Score']].head(10), use_container_width=True)
    
    st.success("✅ Dataset successfully cleaned and preprocessed. Ready for analysis and modeling!")

# ============================================
# LAB 5: EXPLORATORY DATA ANALYSIS
# ============================================
elif page == "📉 Lab 5: Exploratory Data Analysis":
    st.title("📉 Lab 5: Exploratory Data Analysis")
    st.markdown("**Objective:** Visualize and understand relationships in the data")
    st.markdown("---")
    
    st.subheader("🎯 Target Variable: GPA Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df['GPA'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['GPA'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['GPA'].mean():.2f}")
    axes[0].axvline(df['GPA'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df['GPA'].median():.2f}")
    axes[0].set_xlabel('GPA')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('GPA Distribution')
    axes[0].legend()
    
    axes[1].boxplot(df['GPA'])
    axes[1].set_ylabel('GPA')
    axes[1].set_title('GPA Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📊 Programming Skills vs GPA")
    
    skill_means = pd.DataFrame({
        'Python': [df[df['Python']==3]['GPA'].mean(), df[df['Python']==2]['GPA'].mean(), df[df['Python']==1]['GPA'].mean()],
        'SQL': [df[df['SQL']==3]['GPA'].mean(), df[df['SQL']==2]['GPA'].mean(), df[df['SQL']==1]['GPA'].mean()],
        'Java': [df[df['Java']==3]['GPA'].mean(), df[df['Java']==2]['GPA'].mean(), df[df['Java']==1]['GPA'].mean()]
    }, index=['Strong (3)', 'Average (2)', 'Weak (1)'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(skill_means.index))
    width = 0.25
    
    bars1 = ax.bar([i - width for i in x], skill_means['Python'], width, label='Python', color='#3498db')
    bars2 = ax.bar(x, skill_means['SQL'], width, label='SQL', color='#2ecc71')
    bars3 = ax.bar([i + width for i in x], skill_means['Java'], width, label='Java', color='#e74c3c')
    
    ax.set_xlabel('Skill Level')
    ax.set_ylabel('Average GPA')
    ax.set_title('Average GPA by Programming Skill Level')
    ax.set_xticks(x)
    ax.set_xticklabels(skill_means.index)
    ax.legend()
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📊 Correlation Matrix")
    
    numeric_cols = ['Age', 'GPA', 'Python', 'SQL', 'Java', 'Gender', 'Skill Score']
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix of Features', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.write("**Correlations with GPA (sorted):**")
    gpa_corr = corr_matrix['GPA'].sort_values(ascending=False)
    st.dataframe(gpa_corr.to_frame().T, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Top Domains by Student Interest")
    
    domain_counts = df['Interested Domain'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(domain_counts)), domain_counts.values, color='skyblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Domain')
    ax.set_ylabel('Number of Students')
    ax.set_title('Top 10 Domains by Student Interest')
    ax.set_xticks(range(len(domain_counts)))
    ax.set_xticklabels(domain_counts.index, rotation=45, ha='right')
    
    for bar, val in zip(bars, domain_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val), ha='center', fontsize=10)
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📊 Age vs GPA (Colored by Gender)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}
    labels = {0: 'Male', 1: 'Female'}
    
    for gender in [0, 1]:
        subset = df[df['Gender'] == gender]
        ax.scatter(subset['Age'], subset['GPA'], c=colors[gender], label=labels[gender], alpha=0.6, s=60)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('GPA')
    ax.set_title('Age vs GPA by Gender')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.success("""
    **Key Findings from Lab 5 (EDA):**
    
    1. **Programming Skills Impact:**
       - **Python** has the strongest positive correlation with GPA (r = 0.35)
       - **SQL** shows moderate positive correlation (r = 0.18)
       - **Java** shows slight negative correlation (r = -0.12)
    
    2. **Domain Preferences:**
       - **AI** and **Web Development** are most popular (19 students each)
       - **AI** and **Cybersecurity** students have highest GPAs (3.68-3.70)
    
    3. **Demographics:**
       - **Gender** has negligible impact on GPA (r = 0.02)
       - **Age** has weak negative correlation (r = -0.03)
    
    4. **Skill Score** (combined skills) is the strongest predictor of GPA
    """)

# ============================================
# CONCLUSIONS & SUMMARY
# ============================================
elif page == "📝 Conclusions & Summary":
    st.title("📝 Conclusions & Summary")
    st.markdown("---")
    
    st.subheader("🎯 Overall Findings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📊 Descriptive Statistics
        - **Total Students:** 180
        - **GPA Range:** 3.20 – 3.90
        - **Mean GPA:** 3.62
        - **Median GPA:** 3.60
        - **Gender:** 56.7% Male, 43.3% Female
        - **Unique Domains:** 27
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Inferential Statistics
        - **Python skill significantly impacts GPA** (p < 0.001)
        - **No significant gender-based GPA difference**
        - **Domains not gender-biased** (p = 0.236)
        - **95% CI contains population mean**
        - **CLT confirmed** (sample means normal)
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Key Insights")
    
    st.info("""
    **1. Programming Skills Matter Most**
    - Python proficiency has the strongest positive correlation with GPA (r = 0.35)
    - Students with Strong Python average 3.70 GPA vs 3.56 for others
    - Total programming skills (Skill Score) strongly predicts academic success
    
    **2. Domain Preferences & Performance**
    - AI (19 students) and Web Development (19 students) are most popular
    - AI and Cybersecurity students have highest GPAs (3.68-3.70)
    - Mobile App Development students show lower GPAs (3.46)
    
    **3. Interesting Trade-offs**
    - Python and Java are negatively correlated (-0.32)
    - Students strong in Java tend to be weaker in Python
    - This likely reflects specialization (Game Dev/Mobile vs AI/Data Science)
    
    **4. Demographic Factors**
    - Gender shows no significant impact on GPA (p > 0.05)
    - Age outliers (29-37 years) exist but don't affect overall trends
    - Most students are traditional college age (20-23)
    """)
    
    st.markdown("---")
    
    st.subheader("🔮 Recommendations")
    
    st.success("""
    **For Students:**
    - ✅ Prioritize Python skills - they correlate most strongly with academic success
    - ✅ Build breadth across multiple languages (Python + SQL is particularly strong)
    - ✅ Consider balancing Java specialization with Python fundamentals
    
    **For Educators:**
    - ✅ Emphasize Python in introductory courses
    - ✅ Offer bridging courses for Java-focused students
    - ✅ Consider domain-specific resources for lower-performing tracks
    
    **For Future Analysis:**
    - 📊 Collect study hours and prior programming experience
    - 📊 Track project completion and quality metrics
    - 📊 Longitudinal study to establish causality
    - 📊 Include internship and job placement data
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Lab Completion Summary")
    
    completion_df = pd.DataFrame({
        'Lab': ['Lab 0', 'Lab 1', 'Lab 2', 'Lab 3', 'Lab 4', 'Lab 5'],
        'Topic': ['Setup & Imports', 'Descriptive Statistics', 'Inferential Statistics', 
                  'Web Scraping', 'Data Preprocessing', 'Exploratory Data Analysis'],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete'],
        'Key Output': ['Libraries Ready', 'GPA Distribution', 'Statistical Tests', 'books_raw.csv', 'Clean Dataset', 'Correlations & Visuals']
    })
    st.dataframe(completion_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("📊 Web Scraping Summary")
    if books_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Books Scraped", len(books_df))
        with col2:
            st.metric("Pages Scraped", 50)
        with col3:
            try:
                prices = books_df['price'].str.replace('£', '').str.replace('Â', '').astype(float)
                st.metric("Avg Book Price", f"£{prices.mean():.2f}")
            except:
                st.metric("Ratings", f"{books_df['rating'].nunique()} levels")
    
    st.balloons()
    st.success("🎉 **All 6 labs completed successfully!** Dashboard ready for presentation.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 CS Students Analytics | Complete Labs 0-5")
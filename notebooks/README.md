# CS Students Dataset Analysis

## Dataset Description
- Dataset contains information about **180 computer science students**
- Includes demographic details (name, age, gender), academic performance (GPA), technical skills, and career interests
- Provides insights into student profiles, domain preferences, project experiences, and programming language proficiency
- Source: Custom dataset (similar to Kaggle-style educational datasets)

## Data Types
The dataset contains a mix of data types: **quantitative discrete** (Student ID, Age), **quantitative continuous** (GPA), **qualitative nominal** (Name, Gender, Major, Interested Domain, Projects, Future Career), and **qualitative ordinal** (Python, SQL, Java skill levels which are ranked as Weak, Average, or Strong).

## Example Visualizations You Can Create
- **Bar charts** to show the most popular domains or career paths students are interested in
- **Pie charts** to visualize the gender distribution of students
- **Histograms** to see how GPAs are spread across the student population
- **Box plots** to compare GPA distributions between male and female students
- **Scatter plots** to explore relationships between age and GPA
- **Heatmaps** to see if students good at Python are also good at Java or SQL
- **Interactive charts** using Plotly to hover over points and explore the data dynamically

---
 
## Lab 2 — Inferential Statistics
 
**Tests chosen and why:**
 
- **Z-score** — detects outlier students in `GPA` and `Age` on a per-row basis, since both columns have a well-defined mean and std.
- **Z-test** — compares Python-Strong students' mean GPA to the overall population mean; used because n > 30 and the population std is known.
- **T-test** — compares GPA between Male and Female groups; preferred over Z-test for two independent groups where group-level population std is unknown.
- **Chi² independence** — tests whether Python skill level is related to Interested Domain; both variables are categorical, making this the only valid choice.
- **Chi² goodness of fit** — checks whether students are equally distributed across domains; tests one categorical variable against a uniform expected distribution.
- **Basic & conditional probability** — calculates P(Python-Strong) and P(high GPA | Female) directly from counts, as a foundation before formal testing.
- **CLT + Confidence Interval** — demonstrates sampling behavior on `GPA` (the only meaningful continuous variable) and estimates the true population mean with 95% confidence.

---


## Lab 3 — Web Scraping: Books to Scrape

**Target site:** [books.toscrape.com](https://books.toscrape.com)  
**Output file:** `books_raw.csv`

### What the notebook does
- Imports `requests`, `BeautifulSoup`, `csv`, and `time`
- Checks `robots.txt` to confirm the site permits automated access
- Fetches the first page and detects the total number of pages (50) from the pager element
- Defines `parse_page()` which extracts **title, price, star rating, availability, and URL** from each `<article class="product_pod">` tag using CSS selectors
- Follows the `Next →` pagination link in a loop until it disappears, waiting 1 second between requests
- Saves all 1000 book entries to `books_raw.csv` using `csv.DictWriter`

### Key concepts covered
- `requests.get()` + `BeautifulSoup` for static page scraping
- CSS selectors (`.select()`, `.select_one()`) and attribute extraction (`['href']`, `['title']`)
- Following pagination via `<li class="next">` link detection
- Ethical scraping: `robots.txt` check + `time.sleep()` delays
- Exporting structured data with `csv.DictWriter`

---

## Lab 4 — Data Preprocessing: CS Students Dataset

**Input file:** `cs_students.csv` (180 rows × 12 columns)  
**Output file:** `cs_students_clean.csv`

### What the notebook does
- Loads the dataset and inspects shape, column types, and missing values
- Applies a **defensive fill** — median for numeric columns, `'Unknown'` for text columns
- Removes duplicates and resets the index
- **Strips whitespace** from all string columns to prevent silent mismatches
- **Normalizes text** to title case for `Name`, `Major`, `Interested Domain`, `Projects`, and `Future Career`
- **Encodes ordinal skill columns** (`Python`, `SQL`, `Java`) from `Weak / Average / Strong` → `1 / 2 / 3`
- **Encodes `Gender`** as a binary integer (`Male → 0`, `Female → 1`)
- **Checks for inconsistencies**: age out of range, invalid GPA, duplicate Student IDs
- **Detects outliers** in `GPA` and `Age` using the IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR)
- **Creates a derived feature** `Skill Score` = average of the three skill columns
- Saves the cleaned DataFrame to `cs_students_clean.csv`

### Key concepts covered
- Null handling with `.fillna()` differentiated by column type
- String normalization with `.str.strip()` and `.str.title()`
- Ordinal encoding with `.map()` for ordered categorical variables
- Logical consistency checks with boolean indexing
- Outlier detection with the IQR method
- Feature engineering (creating a new summary column)
- Saving with `df.to_csv(index=False)`

---

---

## Lab 5 — Exploratory Data Analysis: CS Students Dataset

**Input file:** `cs_students.csv` (180 rows × 12 columns)  
**Target variable:** `GPA`

### What the notebook does
- Imports `pandas`, `seaborn`, `matplotlib`, and `operator`
- Encodes the ordinal skill columns (`Python`, `SQL`, `Java`: Weak/Average/Strong → 1/2/3) and computes a composite `Skill Score`
- **Numerical distribution** — histograms for all numeric features; detailed KDE plot for `GPA`
- **Correlation analysis** — identifies features correlated with GPA (|r| > 0.15); repeats after filtering zero-values for robustness
- **Feature-to-feature heatmap** — reveals the strong negative correlation between Python and Java skills
- **Q → Q analysis** — regression scatter plots of each correlated feature vs GPA
- **C → Q analysis** — boxplots of GPA by Gender, Interested Domain (top 6), and Future Career (top 6); bar charts for categorical feature distributions; grouped bar chart of average skill levels by domain

### Key findings
- `GPA` is narrowly distributed (3.2–3.9) with a slight left skew
- `Age` has visible outliers at 29–37 among what is otherwise a 20–23 student population
- **Python** has the strongest positive correlation with GPA; **Java** has a negative correlation
- **Python and Java are strongly negatively correlated** — students tend to specialize in one
- Female students show a slightly higher median GPA, but the difference is marginal
- AI and Cybersecurity students average the highest GPAs among the top domains
- Many career paths appear only once or twice — sparse categories should be grouped before modeling

### Key concepts covered
- Selecting numeric columns with `select_dtypes()`
- Correlation with `.corr()` and filtering by threshold
- `sns.histplot()`, `sns.regplot()`, `sns.boxplot()`, `sns.heatmap()`, `sns.barplot()`
- Grouped aggregations with `.groupby().mean()` and `.describe()`
- Grouped bar charts from a pivoted DataFrame

---
## Lab 7 — Linear Regression: GPA Prediction Analysis

**Input file:** `cs_students.csv` (180 rows × 12 columns)  
**Target variable:** `GPA` (Grade Point Average)  
**Model type:** Ordinary Least Squares (OLS) Linear Regression

### What the notebook does
- Loads and preprocesses the CS students dataset
- Encodes ordinal skills (Python, SQL, Java: Weak/Average/Strong → 1/2/3)
- Engineers features: `Total_skills`, `Gender_encoded`, `domain_priority`
- Splits data into training (70%) and testing (30%) sets
- Applies StandardScaler to normalize features
- Trains **OLS**, **Ridge (L2)** , and **Lasso (L1)** regression models
- Evaluates models using MAE, RMSE, and R² metrics
- Detects overfitting by comparing train vs test performance
- Visualizes residuals, coefficients, and model comparisons

### Key findings
- **Best model:** OLS Linear Regression (R² = 86.3%, MAE = 0.15 GPA points)
- **Most influential factor:** Total programming skills (β = +0.124) — each 1 std deviation increase adds 0.12 GPA points
- **Python** has the strongest positive impact on GPA (β = +0.089)
- **Java** shows slight negative correlation (β = -0.018) — students tend to specialize in Python OR Java, not both
- **Gender and age** have negligible impact on GPA (β near 0)
- **No overfitting detected** — training R² (0.899) vs test R² (0.863) gap = 0.036 (< 0.05 is healthy)
- Regularization (Ridge/Lasso) provides no improvement since OLS already generalizes well

### Key concepts covered
- Train/test split (70/30) with `random_state=42` for reproducibility
- Feature scaling with `StandardScaler` — fit on TRAIN only to prevent data leakage
- OLS regression with `LinearRegression()` — minimizes sum of squared residuals
- Ridge (L2) and Lasso (L1) regularization for overfitting prevention
- Evaluation metrics: MAE (robust to outliers), RMSE (penalizes large errors), R² (% variance explained)
- Residual plots to verify linear assumptions (random scatter = good model)
- Cross-validation (5-fold) to assess model stability
- Coefficient interpretation on standardized features

### Conclusions about the dependent variable (GPA)
- **GPA is predictable from programming skills** — 86% of variance explained
- **Skill breadth matters more than any single language** — Total_skills is strongest predictor
- **Python proficiency signals academic success** — likely due to coursework alignment
- **Java-heavy specialization correlates with lower GPA** — probably contextual (game dev, mobile dev grading differences), not causal
- **Model is production-ready** — low error (±0.15 GPA points), no overfitting, unbiased residuals
- **Recommendation:** Use for early identification of at-risk students; prioritize Python fundamentals in CS curricula
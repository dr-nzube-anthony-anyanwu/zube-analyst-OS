# ZubeAnalystOS

### AI-Powered Data Intelligence and Decision Support

ZubeAnalystOS is a professional Streamlit workspace that supports the practical data-analysis lifecycle—from raw CSV or Excel data to preparation, exploratory analysis, statistical investigation, predictive insight, executive communication, and reusable reporting.

The platform is designed for analysts who need depth and for business leaders who need clear, decision-ready explanations rather than statistical jargon.

## Core capabilities

### Data preparation

- Import CSV, XLSX, and XLS datasets.
- Detect missing values, duplicate records, field types, cardinality, and potential outliers.
- Replace missing values using mean, median, mode, or a custom value.
- Remove missing rows or unwanted fields.
- Deduplicate complete rows or records identified by selected business keys.
- Correct number, date/time, text, category, and Boolean field types.
- Cap, remove, or null potential outliers using configurable IQR sensitivity.
- Compare dataset quality before and after transformation.

### Wrangling and integration

- Filter rows using text, numeric, missing, and populated conditions.
- Create arithmetic calculated fields.
- Group records and summarize business measures.
- Construct pivot tables.
- Reshape wide datasets into long format.
- Combine datasets using inner, left, right, or full joins.
- Preserve an untouched source dataset while working with a reversible copy.

### Exploration and visualization

- Interactive Plotly distributions, rankings, box plots, scatter plots, heatmaps, and time series.
- Seaborn statistical distributions and violin plots.
- Correlation and relationship analysis.
- Data-quality, completeness, outlier, type, and cardinality diagnostics.
- High-resolution chart export from Plotly controls.

### Decision science

- Reusable KPI definitions with optional targets and automatic recalculation.
- Confidence intervals and group-comparison tests.
- Category-association and numeric-relationship tests.
- Isolation Forest anomaly detection.
- K-Means segmentation and segment profiles.
- Holt-Winters time-series forecasting with an approximate uncertainty range.

### AI-assisted intelligence

- Plain-English executive decision briefs.
- Separate technical appendices for analytical detail.
- Natural-language questions answered from row-level evidence.
- Transparent evidence-scope and sampling disclosures.
- Strict provider timeouts that prevent indefinite loading.
- Action plans, decision priorities, risks, and suggested owners written for non-technical leaders.

### Reproducibility and export

- Transformation history with up to 12 undo snapshots.
- Reset to the original uploaded dataset.
- Downloadable and replayable transformation recipes.
- Locally saved project recipes containing transformations and KPI definitions.
- Cleaned CSV, data dictionary, anomaly review, forecast, and Markdown report exports.

## Technology stack

| Layer | Technology |
|---|---|
| Application interface | Streamlit |
| Data processing | Pandas, NumPy, NumExpr |
| Interactive visualization | Plotly |
| Statistical visualization | Seaborn, Matplotlib |
| Statistical analysis | SciPy, statsmodels |
| Machine learning | scikit-learn |
| AI integration | OpenRouter API |
| Testing | pytest, Streamlit AppTest |

## Project structure

```text
app.py                 Streamlit interface, session state, and workflows
analyst_engine.py      Reusable transformations and analytical methods
ai_service.py          OpenRouter transport, validation, and timeouts
requirements.txt       Runtime dependencies
requirements-dev.txt   Runtime and test dependencies
pytest.ini             Test configuration
tests/                 Unit, interaction, AI, and browser-audit tests
implementation.md      Architecture, history, decisions, and operations
```

## Local installation

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install runtime dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a local `.env` file containing your own credentials.

```env
OPENROUTER_API_KEY=replace_with_your_private_key
OPENROUTER_MODEL=replace_with_your_preferred_model_id
```

4. Start ZubeAnalystOS.

```powershell
streamlit run app.py
```

> Never commit, publish, screenshot, or share the `.env` file. ZubeAnalystOS documentation contains placeholders only and does not contain the developer's private credentials.

## Verification

Install the development dependencies and run the deterministic suite:

```powershell
pip install -r requirements-dev.txt
pytest -m "not live"
```

The current suite contains 51 deterministic tests covering transformations, statistical and predictive tools, Streamlit state transitions, downloads, project recipes, timeouts, and mocked AI behavior.

The live OpenRouter smoke test is deliberately opt-in and uses only synthetic data:

```powershell
$env:RUN_LIVE_AI_TEST="1"
pytest -m live
```

## Data privacy and analytical boundaries

- Local cleaning, wrangling, visualization, statistics, anomaly detection, segmentation, forecasting, and exports do not require an AI provider.
- Executive briefs send a compact statistical profile rather than raw rows.
- Ask-the-data uses all rows for datasets containing up to 300 records. Larger datasets use a reproducible 300-row sample and disclose that limit.
- Saved projects contain instructions and KPI definitions—not uploaded business data.
- Statistical and predictive results support investigation; they do not establish causality or replace professional domain review.

See [implementation.md](implementation.md) for architecture, design decisions, implementation history, resolved issues, testing details, limitations, and operating guidance.

## Author and developer

**Dr. Anthony N. Anyanwu**  
**AI Systems Architect & Engineer | Data Scientist | Optometrist**

Creator, architect, and lead developer of ZubeAnalystOS.

## Status

ZubeAnalystOS is an actively developed technology. Validate results against domain knowledge and organizational policies before using them for material business, financial, operational, or clinical decisions.

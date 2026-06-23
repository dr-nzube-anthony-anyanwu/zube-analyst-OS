# ZubeAnalystOS Implementation Record

## Document purpose

This document is the technical and product record for ZubeAnalystOS. It explains the technology's purpose, architecture, workflows, implementation history, design decisions, resolved issues, security model, tests, known limitations, and operating procedures.

It reflects the application implemented in this repository and should be updated whenever behavior, architecture, dependencies, or operational requirements change.

## Product identity

**Product:** ZubeAnalystOS  
**Category:** AI-powered data intelligence and decision-support workspace  
**Author and lead developer:** Dr. Anthony N. Anyanwu  
**Professional title:** AI Systems Architect & Engineer | Data Scientist | Optometrist

## Product objective

ZubeAnalystOS reduces the distance between raw tabular data and an informed decision. It provides one workspace for:

1. Importing and understanding a dataset.
2. Detecting and correcting common quality problems.
3. Reshaping and combining data for analysis.
4. Exploring distributions, categories, relationships, and time patterns.
5. Applying statistical and machine-learning techniques.
6. Defining decision-relevant KPIs.
7. Translating analytical evidence into executive language.
8. Preserving transformations as reproducible recipes.
9. Exporting prepared data and supporting reports.

The system supports human analysts; it does not claim to replace professional judgment, domain knowledge, data governance, or accountability for decisions.

## Architecture

### Application layer: `app.py`

The Streamlit application owns:

- Page configuration and visual design.
- Dataset upload and conservative date detection.
- Session-state initialization.
- Original and working dataset management.
- Tabbed workspaces and interactive controls.
- Transformation submission and undo/reset behavior.
- Plotly and Seaborn rendering.
- KPI definitions and presentation.
- AI brief and ask-the-data workflows.
- Download and project-recipe interfaces.

Streamlit is configured with a wide layout and a branded dark sidebar. Custom CSS defines typography, cards, tabs, charts, upload controls, primary actions, and accessible sidebar-button states.

### Analytical layer: `analyst_engine.py`

The analytical engine contains reusable, UI-independent functions for:

- Applying serialized transformation operations.
- Joining dataframes.
- Replaying recipes.
- Comparing before-and-after quality.
- Computing confidence intervals.
- Running statistical tests.
- Detecting anomalies.
- Creating behavioral segments.
- Generating time-series forecasts.

Keeping these functions outside the Streamlit interface makes them independently testable and suitable for reuse in future APIs, background workers, notebooks, or scheduled workflows.

### AI transport layer: `ai_service.py`

The AI service owns the OpenRouter HTTP interaction:

- Bearer authentication from environment configuration.
- Model, message, temperature, and token configuration.
- Provider response validation.
- HTTP error propagation.
- Empty or malformed response detection.
- Connection and read limits.
- A strict wall-clock deadline that returns control to the UI if a provider stalls.

No private credential value is stored in source code or documentation.

### Test layer: `tests/`

The test suite contains:

- Analytical-engine unit tests.
- AI transport and timeout tests.
- Streamlit interaction tests using `AppTest`.
- Synthetic CSV fixtures.
- An optional live OpenRouter smoke test.
- A Chrome DevTools browser-audit helper for visual state inspection where the local environment supports Streamlit WebSocket rendering in headless Chrome.

## Data flow

```text
CSV / Excel upload
        |
        v
Column normalization and conservative date inference
        |
        v
Untouched source dataframe
        |
        +--------> Working dataframe in Streamlit session state
                         |
                         +--> cleaning and wrangling operations
                         +--> visual and quality analysis
                         +--> KPI and decision-science analysis
                         +--> AI summary or row-evidence question
                         +--> prepared-data and report exports
```

The source dataframe is retained for reset and quality comparison. Transformations operate on a working copy and do not overwrite the uploaded file.

## Session-state model

The workspace uses the uploaded file's SHA-256 digest prefix as a dataset identifier. A new identifier initializes a fresh workspace.

Important session values include:

| State | Purpose |
|---|---|
| `active_file` | Uploaded filename and bytes |
| `workspace_id` | Identifier for the current source file |
| `working_df` | Current transformed dataframe |
| `undo_stack` | Previous dataframe snapshots, limited to 12 |
| `transform_history` | Serializable operation records |
| `kpi_definitions` | Reusable KPI configurations |
| AI report keys | Generated briefs scoped to the dataset |
| Analysis result keys | Anomaly, segmentation, forecast, and statistical results |

### Transformation lifecycle

1. The UI creates an operation record containing a type, label, parameters, and timestamp.
2. The analytical engine applies the operation to a copy of the working dataframe.
3. Only successful transformations add the previous dataframe to the undo stack.
4. The working dataframe and history are updated.
5. Streamlit reruns and every downstream chart, KPI, and export uses the new working data.

## Ingestion and normalization

### Supported files

- CSV
- XLSX
- XLS

CSV ingestion first attempts UTF-8 and falls back to Latin-1 when decoding fails. Excel files are read through Pandas using the installed Excel engines.

### Column normalization

Column names are:

- Converted to lowercase.
- Trimmed.
- Converted from spaces and hyphens to underscores.
- Assigned a fallback name when blank.
- Made unique with numeric suffixes when duplicated.

### Date inference

Text columns are considered for date conversion only when the name suggests a date or time concept. A sample must parse successfully at an 80% threshold before the full column is converted. This conservative approach avoids treating arbitrary identifiers as dates.

## Data-quality model

The data-health score combines:

- **55% completeness:** proportion of populated cells.
- **30% row uniqueness:** proportion of non-duplicate records.
- **15% column usefulness:** proportion of non-constant fields.

The score is a screening indicator rather than a universal measure of fitness. A high score does not prove that values are correct, representative, unbiased, or suitable for a specific decision.

Quality diagnostics include:

- Missing counts and percentages.
- Exact duplicate counts.
- Type and cardinality profiles.
- Example values.
- Potential numeric outliers using the IQR rule.
- Before-and-after quality comparisons.

## Cleaning operations

### Missing values

Available treatments:

- Drop affected rows.
- Drop the field.
- Replace using mean or median for numeric data.
- Replace using mode.
- Replace using a custom value.

The interface validates custom replacements after submission and shows a clear message when a required value is absent.

### Duplicates

Users can remove exact duplicates or select fields that define a business-level duplicate. Retention options are first record, last record, or removal of every duplicated record.

### Type correction

Supported target types are number, date/time, text, category, and Boolean. Values that cannot be converted safely become missing and can then be reviewed through the quality workflow.

### Outliers

Numeric outliers are screened using first and third quartiles and a configurable IQR factor. Users may:

- Cap values at the calculated limits.
- Remove affected rows.
- Replace outlying values with missing values.

Outlier treatment is not automatically recommended as correct; legitimate extreme values may contain important business information.

## Wrangling operations

### Filtering

Supported conditions include equality, inequality, substring matching, greater/less comparisons, inclusive bounds, missing, and populated.

### Calculated fields

Arithmetic expressions use Pandas evaluation with the NumExpr engine. New field names must be unique. This feature is intended for numeric business calculations such as margin, variance, rate, or unit economics.

### Grouping and summary

Users select one or more grouping fields, one or more numeric measures, and an aggregation: sum, mean, median, minimum, maximum, or count.

### Pivoting

Users select row fields, a column field, a value field, and an aggregation. Pivoted columns are flattened into export-friendly names.

### Reshaping

Wide-to-long reshaping uses selected identifier fields and fields to stack. The interface prevents a field from serving as both an identifier and a stacked measure and requires distinct output names.

### Dataset joins

Supported joins are inner, left, right, and full outer. Duplicate column names from the companion dataset receive a `_joined` suffix.

Join recipes record metadata but cannot be fully replayed without the original companion dataset. During recipe replay, unavailable join steps are reported and skipped rather than silently producing incorrect data.

## Visualization system

### Plotly

Plotly powers interactive business views, including:

- Histograms and marginal distributions.
- Box plots.
- Category rankings.
- Segment comparisons.
- Correlation heatmaps.
- Scatter plots.
- Time series.
- KPI breakdowns.
- Forecast charts.
- Parallel-coordinate segment profiles.

Plotly provides hover details, zooming, selection, and high-resolution chart export.

### Seaborn and Matplotlib

Seaborn provides statistical distribution and violin views. These static charts complement Plotly where conventional statistical presentation is more appropriate.

## Statistical analysis

Implemented procedures include:

- Student-t confidence intervals for a numeric mean.
- Welch's t-test for two-group mean comparison.
- One-way ANOVA for multi-group comparison.
- Chi-square tests for association between categorical fields.
- Pearson correlation tests for numeric relationships.

The UI gives a plain-English interpretation and places detailed statistics in expandable technical sections. Statistical significance does not establish practical importance or causality.

## Machine learning and forecasting

### Anomaly detection

Isolation Forest operates on selected numeric fields after complete-case filtering and standardization. It produces:

- A Boolean anomaly-review flag.
- A relative anomaly score.
- A downloadable review table.

Flags identify unusual records for investigation; they do not prove fraud, error, risk, or clinical abnormality.

### Segmentation

K-Means clustering operates on standardized complete numeric rows. Users choose between two and eight segments. Outputs include row-level segment labels and average segment profiles.

Segments are descriptive mathematical groupings and require domain interpretation before operational use.

### Forecasting

The forecast workflow:

1. Resamples the chosen measure by day, week, month, or quarter.
2. Aggregates values by sum.
3. Fits Holt-Winters exponential smoothing with an additive trend.
4. Adds seasonality only when sufficient history exists.
5. Produces an approximate range based on residual variability.

Forecasts assume historical patterns remain informative. They do not automatically include campaigns, policy changes, economic shocks, clinical interventions, or other external drivers.

## KPI system

A KPI definition contains:

- Business-facing name.
- Numeric source field.
- Aggregation.
- Optional target.
- Optional display prefix and suffix.
- Display precision.

KPIs recalculate whenever the working dataframe changes. KPI definitions are included in exported and locally saved project recipes.

## AI decision intelligence

### Executive brief

The executive brief receives a compact dataset profile containing shape, type groups, quality issues, descriptive statistics, leading correlations, and categorical profiles.

The brief is instructed to produce:

- A plain-English bottom line.
- Decisions that can responsibly be made now.
- Expected business effects without invented financial values.
- Signals translated into business meaning.
- Risks and evidence limitations.
- A sequenced 30-day action plan.
- Questions that would improve the decision.
- A separate technical appendix.

The prompt prohibits unsupported causality, guessed dataset origins, unexplained statistical notation in the executive section, and invented domain definitions.

### Ask-the-data

For datasets with 300 rows or fewer, row-level evidence can include all rows. Larger datasets use a reproducible 300-row sample. The response must state its evidence scope and cannot present a sampled calculation as an exact full-dataset result.

### Provider reliability

The OpenRouter transport uses:

- A connection timeout.
- A read timeout.
- A strict 70-second wall-clock deadline.
- Clear errors for HTTP failures, timeouts, malformed responses, and empty responses.

The UI explains that a typical request should take approximately 10–45 seconds. If the hard deadline is reached, the user's dataset and selections remain available for retry.

## Recipes and saved projects

Recipe JSON contains:

- Product identifier.
- Recipe version.
- Project name.
- Transformation operations.
- KPI definitions.

Saved projects do not contain uploaded datasets. The default directory is `projects`; deployments can override it with `ZUBE_PROJECTS_DIR`.

Recipe replay starts from the original source dataframe and reports incompatible or dataset-specific steps. Users should export prepared data separately when exact row-level results must be preserved.

## Exports

ZubeAnalystOS can produce:

- Cleaned CSV data.
- Data dictionaries.
- Markdown analysis reports.
- Transformation recipes.
- Anomaly-review CSV files.
- Forecast CSV files.
- Plotly chart images.

## Implementation history

### Phase 1: functional MVP

The initial application supported CSV upload, dataset preview, missing-value detection, numeric summaries, static Matplotlib charts, an OpenRouter summary, and text-report download.

### Phase 2: professional product redesign

The interface was rebuilt around a branded dashboard, responsive layout, dark sidebar, KPI cards, tabbed workflows, interactive Plotly charts, data-health scoring, session-persistent AI output, and stronger export behavior.

### Phase 3: ZubeAnalystOS identity

The product was named ZubeAnalystOS and received a consistent browser title, landing-page identity, sidebar wordmark, report branding, OpenRouter metadata, and decision-intelligence positioning.

### Phase 4: executive AI redesign

The AI prompt was changed from a statistician-oriented report to a leadership brief. Statistical notation and model recommendations were moved away from the executive narrative and into a technical appendix.

### Phase 5: analyst workstation expansion

The analytical engine and preparation studio added cleaning, wrangling, joins, undo, quality validation, statistical testing, KPIs, anomaly detection, segmentation, forecasting, recipes, saved projects, and natural-language questions.

### Phase 6: reliability and regression testing

The OpenRouter transport was extracted, project storage became configurable, and a deterministic pytest suite was introduced. Interaction tests exercise Streamlit state changes rather than checking rendering alone.

### Phase 7: interaction repairs

Sidebar button styling was corrected for visible normal, hover, focus, active, and disabled states. Wrangling form buttons were repaired after identifying Streamlit's non-rerunning form behavior.

### Phase 8: AI timeout protection

The AI workflow received shorter network limits, a strict wall-clock deadline, smaller executive output, and clear recovery messages after a provider stall.

## Resolved issues

| Issue | Cause | Resolution | Status |
|---|---|---|---|
| Emoji and symbol mojibake in the first MVP | Incorrect source/display encoding | Replaced affected content with correctly encoded Unicode and professional branding | Resolved |
| Landing content hidden beneath Streamlit toolbar | Fixed header overlapped the top content | Added safe top spacing and responsive layout rules | Resolved |
| Empty white upload shell | HTML wrapper could not contain Streamlit widgets across render boundaries | Styled the real uploader directly | Resolved |
| Static, non-interactive charts | Matplotlib-only chart workflow | Adopted Plotly for interactive dashboard charts and retained Seaborn for statistical views | Resolved |
| AI output disappeared during reruns | Report was held only in a local execution variable | Stored reports in dataset-scoped session state | Resolved |
| Executive brief was too statistical | Prompt prioritized correlations, distributions, and modeling terminology | Reframed output around decisions, business effects, owners, risks, and action plans | Resolved |
| Sidebar action looked like an empty white pill | Global sidebar color inheritance made text white on a white button | Scoped typography and added explicit branded button states | Resolved |
| Group, pivot, reshape, and KPI buttons remained disabled | Dynamic `disabled` conditions were evaluated only when a Streamlit form first rendered | Kept submit buttons enabled and moved validation to submission time | Resolved |
| Reshape fields did not refresh correctly | One form widget's options depended on another widget that could not trigger a form rerun | Made selections independent and validated overlap after submission | Resolved |
| AI brief could spin for several minutes | Socket inactivity timeout did not guarantee a wall-clock deadline | Added connection/read limits and a strict 70-second UI deadline | Resolved |
| Tests proved rendering but not all interactions | Initial checks did not click every state-changing workflow | Added engine, AI, and Streamlit interaction regression tests | Resolved |

## Testing and quality assurance

### Deterministic suite

The latest verified deterministic suite contains **51 passing tests** and one deselected live-provider test.

Coverage includes:

- Every missing-value strategy.
- Deduplication modes.
- Type correction.
- Outlier treatments.
- All filter operators.
- Calculated fields.
- Group, pivot, and reshape operations.
- All join types and join validation.
- Recipe serialization and replay.
- Before-and-after quality metrics.
- Every statistical procedure.
- Anomaly detection.
- Segmentation.
- Forecasting.
- CSV, Excel, invalid, and empty ingestion.
- Undo and reset.
- KPI creation, recalculation, and clearing.
- Saved-project isolation.
- Dynamic downloads.
- AI success, malformed response, HTTP error, timeout, and hard-deadline behavior.
- Group, pivot, and reshape button interactions matching reported UI scenarios.

Run the suite with:

```powershell
pip install -r requirements-dev.txt
pytest -m "not live"
```

### Optional live AI test

The live test is skipped by default. It uses synthetic data and never logs the configured key.

```powershell
$env:RUN_LIVE_AI_TEST="1"
pytest -m live
```

### Compilation and dependency checks

```powershell
python -m py_compile app.py analyst_engine.py ai_service.py
python -m pip check
```

## Security and privacy

### Credentials

- Credentials are loaded from `.env` or the process environment.
- Source files and documentation contain variable names and placeholders only.
- Tests use fake keys except for the explicitly enabled live smoke test.
- The `.env` file must never be committed, published, copied into screenshots, or shared.
- Provider errors shown in the UI are truncated and should not contain request headers.

### Data handling

- Uploaded files are held in Streamlit session memory.
- Source files are not written automatically by the application.
- Local project files contain transformation metadata and KPI definitions, not uploaded business rows.
- AI briefs use compact profiles.
- Ask-the-data may send row-level evidence to the configured provider and therefore should be used only with data permitted by organizational policy.

### Deployment considerations

Production deployment should add authentication, authorization, transport security, tenant isolation, audit logging, secrets management, data-retention controls, and organization-specific privacy review.

## Configuration

| Variable | Purpose | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | Private OpenRouter credential | Only for AI features |
| `OPENROUTER_MODEL` | OpenRouter model identifier | Optional when application default is suitable |
| `ZUBE_PROJECTS_DIR` | Saved recipe directory | Optional; defaults to `projects` |
| `RUN_LIVE_AI_TEST` | Enables the synthetic live-provider test when set to `1` | Optional |

No credential values are documented in this repository.

## Operating guide

### Start the application

```powershell
.venv\Scripts\activate
streamlit run app.py
```

### Recommended workflow

1. Upload a source dataset.
2. Review Overview and Quality before transforming data.
3. Export a baseline recipe if the work is material.
4. Apply cleaning operations one at a time.
5. Validate before-and-after quality.
6. Perform wrangling, joins, or calculated-field creation.
7. Define KPIs only after confirming field meaning and units.
8. Explore relationships and decision-science outputs.
9. Supply a business decision and context before generating the AI brief.
10. Export the prepared data, report, dictionary, and recipe.

### Recovery behavior

- Use Undo for the most recent transformation.
- Use Reset all to restore the original upload.
- Reapply a saved recipe to reproduce compatible transformations.
- If an AI request times out, retry; the dataset remains in the session.
- If the Streamlit process restarts, in-memory state is lost, so export important artifacts regularly.

## Troubleshooting

### AI request takes too long

- Requests are stopped at the UI after 70 seconds.
- Confirm the model identifier and provider availability.
- Retry once before changing configuration.
- Use the synthetic live test to isolate provider connectivity from dataset complexity.

### AI key is missing

- Confirm `.env` exists locally.
- Confirm the variable name is `OPENROUTER_API_KEY`.
- Restart Streamlit after changing environment configuration.
- Never paste the key into application source or documentation.

### A wrangling operation does not run

- Submit buttons should remain enabled.
- After clicking, read the inline validation message.
- Grouping requires at least one group and measure.
- Pivoting requires at least one row field.
- Reshaping requires at least one stacked field and no overlap with identifier fields.

### Forecast is unavailable

- A detected date field and numeric measure are required.
- At least six aggregated periods are required.
- Correct the date type in Prepare when necessary.

### Anomaly or segmentation analysis is unavailable

- Anomalies require at least 10 complete numeric rows.
- Segments require at least two numeric fields and at least three complete rows per segment.

### Project recipe cannot reproduce a join

- Companion-dataset content is deliberately not embedded in recipes.
- Rejoin the companion dataset manually or export the final prepared CSV.

## Known limitations

- The application is session-based and primarily designed for local or controlled Streamlit use.
- There is no built-in authentication, multi-tenant isolation, or role-based access control.
- Uploaded datasets are processed in memory and constrained by available RAM.
- Undo snapshots also consume memory.
- Saved projects preserve recipes rather than raw uploaded data.
- Joins cannot be replayed without companion data.
- Ask-the-data samples datasets larger than 300 rows.
- AI responses depend on provider availability and model behavior.
- Statistical procedures are general-purpose and do not automatically verify every methodological assumption.
- Forecasting is univariate and does not incorporate external causal drivers.
- K-Means assumes numeric geometry is meaningful after standardization.
- Isolation Forest flags unusual patterns but cannot determine business meaning.
- No analytical output should be treated as clinical diagnosis, financial advice, legal advice, or proof of causality.

## Future development opportunities

Potential production extensions include:

- Authentication and organization workspaces.
- Database and warehouse connectors.
- Background jobs for large datasets.
- Server-side encrypted project persistence.
- Role-based access and audit logs.
- Data-contract and schema-validation rules.
- More robust missing-data models.
- Automated assumption checks for statistical tests.
- Supervised modeling and evaluation workflows.
- Multivariate and scenario-based forecasting.
- Full-dataset local natural-language query planning.
- Scheduled reports and dashboard sharing.
- Deployment-specific observability and cost controls.

These are opportunities rather than commitments and should be prioritized according to user needs, governance requirements, and deployment context.

## Maintenance requirements

Before releasing a change:

1. Update this document when architecture or behavior changes.
2. Update `README.md` when installation, capabilities, or configuration changes.
3. Add tests for every repaired issue or new workflow.
4. Run deterministic tests, compilation, and dependency checks.
5. Confirm that no `.env` content or real credential appears in tracked files.
6. Verify that AI prompts accurately describe data-scope and uncertainty boundaries.
7. Confirm that destructive transformations remain reversible or explicitly warned.

## Ownership

ZubeAnalystOS was conceived, architected, and developed by:

**Dr. Anthony N. Anyanwu**  
**AI Systems Architect & Engineer | Data Scientist | Optometrist**

This implementation record should remain part of the project documentation throughout the technology's lifecycle.

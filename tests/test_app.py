import hashlib
from io import BytesIO

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def csv_bytes(rows=36):
    regions = ("West", "East", "North")
    groups = ("A", "B")
    lines = ["date,region,revenue,cost,group"]
    for index in range(rows):
        revenue = "" if index == 5 else 100 + index * 10
        lines.append(
            f"{2023 + index // 12}-{index % 12 + 1:02d}-01,{regions[index % 3]},{revenue},{50 + index * 4},{groups[index % 2]}"
        )
    return ("\n".join(lines) + "\n").encode()


def app_with_data(data=None, name="business.csv", timeout=90, section="Overview"):
    data = data if data is not None else csv_bytes()
    app = AppTest.from_file("app.py")
    app.session_state["active_file"] = {"name": name, "bytes": data}
    app.run(timeout=timeout)
    if section != "Overview":
        navigate(app, section, timeout)
    return app


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def navigate(app, section, timeout=90):
    widget(app.radio, "Workspace").set_value(section)
    app.run(timeout=timeout)
    return app


def assert_clean_render(app):
    assert len(app.exception) == 0
    assert not app.error


def test_full_workspace_renders_and_core_downloads_exist():
    app = app_with_data()
    assert_clean_render(app)
    navigation = widget(app.radio, "Workspace")
    assert navigation.value == "Overview"
    assert navigation.options == ["Overview", "Prepare", "Quality", "Visuals", "Relationships", "KPIs", "Decision Lab", "AI Brief", "Data & Export"]
    assert not {"Overview", "Prepare", "Quality", "Visuals", "Relationships", "KPIs", "AI Brief", "Data & Export"} & {tab.label for tab in app.tabs}
    navigate(app, "Prepare")
    assert "Export recipe" in {item.label for item in app.get("download_button")}
    assert {"Undo last", "Reset all", "↻ Analyze another file"} <= {button.label for button in app.button}
    navigate(app, "Data & Export")
    assert {"Download cleaned CSV", "Download analysis report", "Download data dictionary"} <= {item.label for item in app.get("download_button")}


def test_csv_excel_invalid_and_empty_ingestion():
    csv_app = app_with_data()
    assert_clean_render(csv_app)

    excel_buffer = BytesIO()
    pd.DataFrame({"region": ["West", "East"], "revenue": [10, 20]}).to_excel(excel_buffer, index=False)
    excel_app = app_with_data(excel_buffer.getvalue(), "business.xlsx")
    assert_clean_render(excel_app)
    assert len(excel_app.session_state["working_df"]) == 2

    for payload in (b"", b'"unterminated'):
        invalid = app_with_data(payload, "broken.csv")
        assert len(invalid.exception) == 0
        assert invalid.error
        assert any("couldn't read" in message.value.lower() for message in invalid.error)
        assert "Choose another file" in {button.label for button in invalid.button}


def test_clean_undo_reset_and_analyze_another_file():
    app = app_with_data(section="Prepare")
    original_rows = len(app.session_state["working_df"])
    widget(app.button, "Apply missing-value treatment").click()
    app.run(timeout=90)
    assert len(app.session_state["working_df"]) == original_rows - 1
    assert len(app.session_state["transform_history"]) == 1

    widget(app.button, "Undo last").click()
    app.run(timeout=90)
    assert len(app.session_state["working_df"]) == original_rows
    assert not app.session_state["transform_history"]

    widget(app.button, "Apply missing-value treatment").click()
    app.run(timeout=90)
    widget(app.button, "Reset all").click()
    app.run(timeout=90)
    assert len(app.session_state["working_df"]) == original_rows
    assert not app.session_state["transform_history"]

    widget(app.button, "↻ Analyze another file").click()
    app.run(timeout=90)
    assert app.session_state["active_file"] is None
    assert len(app.get("file_uploader")) == 1


def test_kpi_add_recalculate_clear_and_project_save(monkeypatch, tmp_path):
    monkeypatch.setenv("ZUBE_PROJECTS_DIR", str(tmp_path / "projects"))
    app = app_with_data(section="KPIs")
    widget(app.text_input, "Business name").set_value("Total revenue")
    widget(app.button, "Add KPI").click()
    app.run(timeout=90)
    assert app.session_state["kpi_definitions"][0]["name"] == "Total revenue"
    assert any(metric.label == "Total revenue" for metric in app.metric)

    navigate(app, "Prepare")
    widget(app.text_input, "Project name").set_value("Sales workspace")
    widget(app.button, "Save project").click()
    app.run(timeout=90)
    saved = tmp_path / "projects" / "Sales_workspace.json"
    assert saved.exists()
    assert "Total revenue" in saved.read_text(encoding="utf-8")

    navigate(app, "KPIs")
    widget(app.button, "Clear KPI board").click()
    app.run(timeout=90)
    assert app.session_state["kpi_definitions"] == []


@pytest.mark.parametrize(
    ("task", "button_label", "selections", "expected_columns", "expected_rows"),
    [
        ("Group & summarize", "Create summary", {"Group by": ["region"], "Measures": ["revenue"]}, {"region", "revenue"}, 3),
        ("Pivot", "Create pivot table", {"Row fields": ["group"]}, {"group"}, 2),
        ("Reshape", "Reshape to long format", {"Identifier fields": ["region"], "Fields to stack": ["revenue", "cost"]}, {"region", "measure", "value"}, 72),
    ],
)
def test_wrangle_form_buttons_enable_and_apply(task, button_label, selections, expected_columns, expected_rows):
    app = app_with_data(section="Prepare")
    app.session_state["wrangling_task"] = task
    app.run(timeout=90)
    button = widget(app.button, button_label)
    assert not button.disabled
    for label, values in selections.items():
        widget(app.multiselect, label).set_value(values)
    widget(app.button, button_label).click()
    app.run(timeout=120)
    result = app.session_state["working_df"]
    assert expected_columns <= set(result.columns)
    assert len(result) == expected_rows
    assert app.session_state["transform_history"][-1]["kind"] in {"group", "pivot", "melt"}


@pytest.mark.parametrize(
    ("task", "button_label", "message"),
    [
        ("Group & summarize", "Create summary", "Select at least one grouping field"),
        ("Pivot", "Create pivot table", "Select at least one row field"),
        ("Reshape", "Reshape to long format", "Select at least one field to stack"),
    ],
)
def test_wrangle_forms_explain_missing_selections(task, button_label, message):
    app = app_with_data(section="Prepare")
    app.session_state["wrangling_task"] = task
    app.run(timeout=90)
    widget(app.button, button_label).click()
    app.run(timeout=90)
    assert any(message in error.value for error in app.error)
    assert not app.session_state["transform_history"]


def test_anomaly_segment_forecast_buttons_and_dynamic_downloads():
    app = app_with_data(section="Decision Lab")
    dataset_id = hashlib.sha256(csv_bytes()).hexdigest()[:12]

    widget(app.button, "Detect unusual records").click()
    app.run(timeout=120)
    assert f"anomaly_result_{dataset_id}" in app.session_state
    assert "Download flagged records" in {item.label for item in app.get("download_button")}
    widget(app.button, "Add anomaly flags to working data").click()
    app.run(timeout=120)
    assert {"anomaly", "anomaly_score"} <= set(app.session_state["working_df"].columns)

    widget(app.button, "Build segments").click()
    app.run(timeout=120)
    assert f"segment_result_{dataset_id}" in app.session_state
    widget(app.button, "Add segment labels to working data").click()
    app.run(timeout=120)
    assert "segment" in app.session_state["working_df"]

    widget(app.button, "Create forecast").click()
    app.run(timeout=120)
    assert f"forecast_result_{dataset_id}" in app.session_state
    assert "Download forecast" in {item.label for item in app.get("download_button")}


def test_mocked_ai_brief_and_row_question(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_chat(**kwargs):
        if kwargs["max_tokens"] == 1600:
            return "## Bottom line\nA useful executive brief.\n\n## Technical appendix\nTest details."
        return "West contributes the most revenue in the supplied evidence."

    monkeypatch.setattr("ai_service.chat_completion", fake_chat)
    app = app_with_data(section="AI Brief")
    widget(app.button, "Generate decision brief").click()
    app.run(timeout=90)
    dataset_id = hashlib.sha256(csv_bytes()).hexdigest()[:12]
    assert "Bottom line" in app.session_state[f"ai_report_{dataset_id}"]

    widget(app.text_input, "Your question").set_value("Which region contributes the most revenue?")
    widget(app.button, "Ask ZubeAnalystOS").click()
    app.run(timeout=90)
    history = app.session_state[f"question_history_{dataset_id}"]
    assert len(history) == 1
    assert "West" in history[0]["answer"]


def test_sidebar_button_css_has_normal_hover_focus_and_disabled_states():
    css = open("app.py", encoding="utf-8").read()
    assert '[data-testid="stSidebar"] .stButton > button {' in css
    assert "button:hover:not(:disabled)" in css
    assert "button:focus-visible" in css
    assert "button:disabled" in css
    assert "color:#FFFFFF !important" in css


def test_responsive_layout_has_mobile_tablet_and_navigation_guards():
    source = open("app.py", encoding="utf-8").read()
    assert 'initial_sidebar_state="auto"' in source
    assert "@media (max-width:1100px) and (min-width:769px)" in source
    assert "@media (max-width:768px)" in source
    assert "@media (max-width:420px)" in source
    assert 'overflow-x:auto;scrollbar-width:none' in source
    assert '[data-testid="stMain"] .stTabs [role="tab"] *' in source
    assert '[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]' in source
    assert '[data-testid="stSegmentedControl"]' in source
    assert 'max-width:calc(100vw - 2rem)' in source


@pytest.mark.parametrize("section", ["Overview", "Prepare", "Quality", "Visuals", "Relationships", "KPIs", "Decision Lab", "AI Brief", "Data & Export"])
def test_sidebar_navigation_renders_every_workspace_section(section):
    app = app_with_data(section=section, timeout=120)
    assert_clean_render(app)
    assert app.session_state["workspace_section"] == section


def test_visual_studio_offers_and_renders_common_chart_families():
    app = app_with_data(section="Visuals", timeout=120)
    chart_picker = widget(app.selectbox, "Chart type")
    expected = {
        "Column chart", "Bar chart", "Pie chart", "Doughnut chart", "Funnel chart",
        "Grouped / stacked columns", "Treemap", "Sunburst", "Radar chart", "Waterfall chart",
        "Line chart", "Area chart", "Scatter plot", "Bubble chart", "Correlation heatmap",
        "Histogram", "Box plot", "Violin plot",
    }
    assert expected <= set(chart_picker.options)
    for chart_type in expected:
        widget(app.selectbox, "Chart type").set_value(chart_type)
        app.run(timeout=120)
        assert_clean_render(app)


def test_navigation_preserves_analysis_state():
    app = app_with_data(section="Prepare")
    original_rows = len(app.session_state["working_df"])
    widget(app.button, "Apply missing-value treatment").click()
    app.run(timeout=90)
    assert len(app.session_state["working_df"]) == original_rows - 1
    history = list(app.session_state["transform_history"])
    app.session_state["kpi_definitions"] = [{"name": "Saved KPI", "column": "cost", "aggregation": "sum", "target": None, "prefix": "", "suffix": "", "decimals": 1}]
    app.session_state["ai_report_preserved"] = "saved report"
    navigate(app, "Quality")
    navigate(app, "Overview")
    assert len(app.session_state["working_df"]) == original_rows - 1
    assert list(app.session_state["transform_history"]) == history
    assert app.session_state["kpi_definitions"][0]["name"] == "Saved KPI"
    assert app.session_state["ai_report_preserved"] == "saved report"

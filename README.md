# Options Dashboards

A Streamlit-based multi-dashboard application for options analysis and visualization.

## Structure

```
options_dashboards/
├── app.py                          # Main entry point / home page
├── pages/                          # Dashboard pages (appear in sidebar)
│   ├── 1_📈_Dashboard_1.py
│   └── 2_📊_Dashboard_2.py
├── utils/                          # Shared utilities and components
│   └── __init__.py
├── requirements.txt                # Python dependencies
└── README.md
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   Open your browser to `http://localhost:8501`

## Adding New Dashboards

To add a new dashboard:

1. Create a new Python file in the `pages/` directory
2. Name it with a number prefix for ordering: `3_🎯_Your_Dashboard.py`
3. The file will automatically appear in the sidebar

### Dashboard Template

```python
import streamlit as st

st.set_page_config(page_title="Your Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Your Dashboard")
st.markdown("---")

# Add your dashboard content here
```

## Features

- **Multi-page navigation**: Automatic sidebar navigation based on files in `pages/`
- **Wide layout**: Optimized for data visualization
- **Responsive design**: Works on different screen sizes
- **Example dashboards**: Two sample dashboards to get you started

## Customization

- Modify [app.py](app.py) to change the home page content
- Update page configuration in each dashboard file
- Add shared utilities in the `utils/` directory
- Customize the theme by creating `.streamlit/config.toml`

## Dependencies

- streamlit: Web app framework
- pandas: Data manipulation
- numpy: Numerical operations
- plotly: Interactive visualizations

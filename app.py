import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Options Dashboards",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("📊 Options Dashboards")
st.markdown("---")

st.markdown("""
### Welcome to the Options Dashboards App

Use the sidebar on the left to navigate between different dashboards.

#### Available Dashboards:
- **Dashboard 1**: Example dashboard with sample visualizations
- **Dashboard 2**: Another example dashboard

#### Getting Started:
1. Select a dashboard from the sidebar
2. Each dashboard is interactive and can be customized
3. Add new dashboards by creating files in the `pages/` directory
""")

st.markdown("---")
st.info("💡 Tip: Each dashboard is a separate Python file in the pages/ directory")

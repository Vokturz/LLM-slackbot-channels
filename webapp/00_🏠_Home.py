import streamlit as st

# Modo wide
st.set_page_config(page_title='Home', 
                   page_icon='🏠',
                   layout="wide", initial_sidebar_state="expanded")

# Margen a la izquierda
padding_left = 20
st.markdown(
        f"""<style>
        .appview-container .main .block-container{{
        padding-left: {padding_left}rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Header

col1, col2, col3 = st.columns([4,1, 10])
st.markdown("# Home")
st.sidebar.markdown("# 🏠 Home")
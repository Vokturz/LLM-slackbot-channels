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
col1, col2 = st.columns([0.5,1])
col2.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="drawing" width="80"/>',
               unsafe_allow_html=True)
col1.markdown("# LLM-slackbot-channels")
st.markdown("""
Welcome to LLM-slackbot-channels Web app!
From here you can:
### 1 Modify bot personality, instructions and temperature for a given channel
1. Go to _#️⃣ Modify Channels_
2. Select a channel. A form with the bot's info will appear
3. Edit bot information as you wish
### 2 Get and remove info about QA threads inside a channel
1. Go to _❓ Threads QA Info_
2. Select a channel. A list of QA threads will appear
3. Select all QA threads you want to remove
4. Remove QA threads. This will remove the vector data related to all selected QA threads.
""")

st.sidebar.markdown("# 🏠 Home")
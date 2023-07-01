import streamlit as st
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from slack_bolt.app import App
from dotenv import load_dotenv
import pandas as pd
import json
# Load environment variables
load_dotenv()

# Start the App
app = App(token=os.environ['SLACK_BOT_TOKEN'])
client = app.client

# Get info about channels
llm_info_file = f'{current_dir}/../../data/channels_llm_info.json'
with open(llm_info_file, 'r') as f:
    channels_llm_info = json.load(f)

response = client.conversations_list(types=['public_channel', 'private_channel'])
channels = pd.DataFrame(response['channels']).set_index('name')

resp = client.auth_test()
bot_id = resp['user_id']
bot_user = resp['user']
bot_channels = pd.DataFrame(client.
                            users_conversations(user=bot_id,
                                                types=['public_channel', 'private_channel'])
                                                ['channels']).set_index('name')

## INICIO

# Modo wide
st.set_page_config(page_title='Modify Channel', page_icon='#️⃣',
                   layout="wide", initial_sidebar_state="expanded")

# Margen a la izquierda
padding_left = 10
st.markdown(
        f"""<style>
        .appview-container .main .block-container{{
        padding-left: {padding_left}rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Header

st.markdown("# Modify Channel Info")
st.markdown("*Change Bot personality, instructions and temperature for a given channel*")
st.sidebar.markdown("# #️⃣ Modify Channel Info")

col1, col2, _ = st.columns([2,4, 1])
bot_only = col1.checkbox("Show only channels where the bot is in", value=True)
channels = channels.join(bot_channels.assign(bot_is_in_channel = True)
                         [['bot_is_in_channel']], how='left').fillna(False)
if bot_only:
    channels = channels[channels['bot_is_in_channel']]
col1.write(channels[['is_private', 'bot_is_in_channel']])
col1.info("Bot must be shutdown in order to use this feature, otherwise it will cause bugs", icon="❕")

default_info = {
        "personality": "an AI assistant inside a Slack channel",
        "instructions": "Give helpful and concise answers to the user's questions."
                        " Answers in no more than 40 words."
                        " You must format your messages in Slack markdown.",
        "temperature": 0.8
    }

channel = col2.selectbox("Select a channel", channels.index)
channel_id = channels.loc[channel]['id']
if channel in bot_channels.index:
    with col2.form('modify_bot'):
        st.caption("*There are **{}** members in this channel*".format(channels.loc[channel]['num_members']))
        if channel_id not in channels_llm_info:
            st.warning("No LLM info for this channel", icon="⚠")
            channel_info = default_info
            text_btn = "Add"
        else:
            channel_info = channels_llm_info[channel_id]
            text_btn = "Modify"
        new_personality = st.text_input("Personality", value=channel_info['personality'])
        new_instructions = st.text_area("Instructions", value=channel_info['instructions'],
                                        height=200)
        new_temp = st.slider("Temperature", 0.0, 1.0, channel_info['temperature'])
        btn = st.form_submit_button(text_btn)
        if btn:
            new_info = {
                "personality": new_personality,
                "instructions": new_instructions,
                "temperature": new_temp
            }
            channels_llm_info[channel_id] = new_info
            with open(llm_info_file, 'w') as f:
                json.dump(channels_llm_info, f, ensure_ascii=False, indent=4)
            st.success("Successfully modified LLM info for this channel", icon="✅")
            st.warning("You must reset the bot to see the changes", icon="⚠")
else:
    col2.error(f"Bot *{bot_user}* is not in channel **{channel}**, you must invite it first.", icon="🚨")



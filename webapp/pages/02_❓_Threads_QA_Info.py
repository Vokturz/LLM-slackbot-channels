import streamlit as st
import os
import glob
from datetime import datetime
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))

from slack_bolt.app import App
from dotenv import load_dotenv
import pandas as pd
import json
from langchain.vectorstores import Chroma
from chromadb.config import Settings
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
st.set_page_config(page_title='Modify Channel', page_icon='❓',
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
st.markdown("# Threads QA Info")
st.markdown("*Get info about QA threads inside a channel*")
st.sidebar.markdown("# ❓ Threads QA Info")


db_dir = f'{current_dir}/../../data/db'


thread_retriever_db = {}
for channel_dir in glob.glob(db_dir + "/[CG]*"):
    channel_id = channel_dir.split("/")[-1]
    thread_retriever_db[channel_id] = {}
    for ts_dir in glob.glob(channel_dir + "/*"):
        ts_val = ts_dir.split("/")[-1]
        thread_retriever_db[channel_id][ts_val] = ts_dir

qa_thread_df = pd.DataFrame(thread_retriever_db.keys(), columns=['id']).set_index('id')
for channel_id in thread_retriever_db:
    qa_thread_df.loc[channel_id, 'qa_threads'] = len(thread_retriever_db[channel_id])


col1, col2, _ = st.columns([3,5, 1])
bot_only = col1.checkbox("Show only channels where the bot is in", value=True)
channels = channels.join(bot_channels.assign(bot_is_in_channel = True)
                         [['bot_is_in_channel']], how='left').fillna(False)
channels = channels.join(qa_thread_df, on='id').fillna(0)
if bot_only:
    channels = channels[channels['bot_is_in_channel']]
col1.write(channels[['is_private', 'bot_is_in_channel', 'qa_threads']])



channel = col2.selectbox("Select a channel", channels.index)
channel_id = channels.loc[channel]['id']
if channel in bot_channels.index:
    i = 0
    if channel_id in thread_retriever_db.keys():
        if len(thread_retriever_db[channel_id])==0:
            col2.warning(f"This channel has no QA threads", icon="⚠")
        for ts in dict(sorted(thread_retriever_db[channel_id].items())):
            db_path = thread_retriever_db[channel_id][ts]
            vectorstore = Chroma(persist_directory=db_path,
                                    embedding_function=None)
            metadatas = vectorstore.get()["metadatas"]
            files = set()
            for chunk in metadatas:
                files.add(chunk["source"].split("/")[-1])
            dt = datetime.fromtimestamp(int(float(ts)))
            try:
                resp = client.conversations_replies(channel=channel_id, ts=ts)    
                first_msg = resp['messages'][0]['text']
                first_msg = first_msg.replace('<@' + bot_id + '>', '').strip()
            except:
                first_msg = "No message"
            files = list(files)
            if len(files)==0:
                files = ["No files"]
            col2_1, col2_2= col2.columns([2, 1])
            col2_1.write(f"[{i}] **{dt}** -  Files: _{','.join(files)}_ -  First message: _{first_msg}_")
            buttons = {}
            buttons[ts] = col2_2.button(f"[{i}] Delete 🗑️", key=ts, use_container_width=False)
            if buttons[ts]:
                if len(files)>0:
                    client.chat_postMessage(channel=channel_id,
                                            text="_QA Thread files deleted via web App_", thread_ts=ts)
                shutil.rmtree(db_path)
                st.experimental_rerun()
            i+=1
    else:
        col2.warning(f"This channel has no QA threads", icon="⚠")
    
else:
    col2.error(f"Bot *{bot_user}* is not in channel **{channel}**, you must invite it first.", icon="🚨")


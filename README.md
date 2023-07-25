
# LLM Slack Bot Channels

LLM-SlackBot-Channels is a Slack bot built using the Slack Bolt framework. It enables users to interact with a bot through Slack channels using various commands. The bot utilizes a Large Language Model (LLM) to generate responses based on user input.

This repository mainly uses **langchain**, it supports the usage of open-source LLMs and embeddings, and also OpenAI.
<p align="center">
<img src="https://github.com/Vokturz/LLM-slackbot-channels/assets/21696514/4ed5b2b0-b9e3-4dec-8f59-569eb4a3c943" width="720" height="720">
</p>

See a video example here:

https://github.com/Vokturz/LLM-slackbot-channels/assets/21696514/d69106f3-3de1-4781-9e1d-9b855acd2836



## What's new? v0.2
- Improve the way a file is uploaded to a QA thread
- If using OpenAI models, then you can customize which model you want to use in each channel
- Use the LLM model as an Agent with your own tools! see more in [v0.2 release](https://github.com/Vokturz/LLM-slackbot-channels/releases/tag/v0.2)
    - You can add files to the channel, which are used by the agent with a *doc retriever* tool
## Commands

- **/modify_bot**
    This command allows you to customize the bot's personality, instructions, and temperature within the channel it's operating in. If `!no-notify` is included, then no notification is sent to the channel.
    <p align="center">
    <img src="https://github.com/Vokturz/LLM-slackbot-channels/assets/21696514/ed1cf3f1-bb67-4859-bf44-1b546031390b" width="300" height="400">
    </p>


- **/bot_info**
    This command presents the initial prompt used by the bot, as well as the default 'temperature' for generating responses.
- **/ask**
    Use this command to ask questions or make requests. The bot employs the LLM to generate appropriate responses.

    Command syntax: `/ask (<!all>) (<!temp=temp>) <question or request>`

    Here, if `!all` is included, the bot sends its response to the entire channel. If `!temp` is included, the response's "temperature" (randomness of the bot's output) is adjusted.

- **/permissions** (optional)
    This command modify which users can interact with the bot. It requires a password defined in the environment variables

    Command syntax: `/permissions <PERMISSIONS_PASSWORD>` 

    If no password was defined inside `.env`, then this command do nothing.

- **/edit_docs**
    This command allows the user to edit the descriptions of the documents that have been uploaded to the channel. These edited descriptions are used in the *doc_retriever* tool (See *Mentions*).
    
## Mentions
- When the bot is mentioned in a thread, it can respond based on the context. The context limit is handled using a `max token limit` in a similar way as `ConversationTokenBufferMemory` from langchain.

- If the bot is mentioned in channel along with uploaded file, then it ask if you want to start a QA thread or upload the file directly to the channel: The user has the possibility to add some context and new separators to chunk the file(s). The files are downloaded in `data/tmp` to define a persistent VectorStore in `data/db`, after the generation of the VectorStore all files are deleted. 

    - **QA Thread**: The bot responds to the user's message that contains the uploaded file(s), stating that a QA thread has been created with the uploaded file(s) and the context provided by the user
    - **Upload to channel**: The file is upload to the channel and the tool *doc_retriever* appears in the list of tools once at least one file has been added to the channel. This tool take as context all the files uploaded by the users using this method.
    <p align="center">
    <img src="https://github.com/Vokturz/LLM-slackbot-channels/assets/21696514/5dc9ef8f-dda8-49df-af76-72ee465cdf81" width="600" height="250">
    </p>

    If the channel is used as a simple LLM chain, then a `ConversationRetrievalChain`, otherwise a tool to retrieve the important information from the documents is created and passed to the Agent.

### How are documents handled?
The documents are handled using **ChromaDB**, saving the database to `data/db/{channel_id}/{timestamp}` for each QA thread, where `channel_id` refers to the channel which contains the thread and `timestamp` to the time when the QA thread was initiated. It is important to mention that typically embedding models are not compatible, so if you change the embedding model after creating the database for a given QA thread, then that thread will not work.
## Usage

### Requirements 

To install the necessary requirements, use:
```bash
pip install -r requirements.txt
```

For CTransformers or OpenAI functionalities, you will need to install these packages separately:
```bash
pip install ctransformers
pip install openai
```

For Open-source embeddings, you will need to install `sentence-transformers`:
```bash
pip install sentence-transformers
```

### Environment variables
Duplicate `example.env` to `.env` and adjust as necessary:
```
OPENAI_API_KEY= Your OpenAI key
CTRANSFORMERS_MODEL= model name from HuggingFace or model path in your computer
EMB_MODEL=all-MiniLM-L6-v2 # Embedding model
SLACK_BOT_TOKEN=xoxb-... Slack API Bot Token 
SLACK_APP_TOKEN=xapp-... Slack API App Token
PERMISSIONS_PASSWORD=CHANGEME # Password to activate /permissions command
```

### Starting the Bot
To start the bot, simply run:
```bash
python main.py
```
This file contains the basic configuration to run the bot:
```python
from src.slackbot import SlackBot
from src.handlers import create_handlers

# Set model_type
# OpenAI
# Llama (CTransformers)
# FakeLLM (just for testing) 
model_type='OpenAI'
bot = SlackBot(name='SlackBot', model_type=model_type)

# Set configuration
config = dict(model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=500)

# Initialize LLM
# max_tokens_threads refers to the max tokens to consider in a thread message history
bot.initialize_llm(model_type, max_tokens_threads=2000, config=config)

# Initialize Embeddings
# If you don't want to use OpenAI Embeddings you can modify this part with llama to use a model from EMB_MODEL env variable
bot.initialize_embeddings(model_type)

# Create handlers for commands /ask, /modify_bot, /bot_info  and bot mentions
create_handlers(bot)

### You can create new handlers for other commands as follow
# @bot.app.command("/foo")
# async def handle_foo(say, respond, ack, command):
#     await ack()
#     # do something..
```

## Slack API configuration
The bot requires the following permissions:
1. Enable **Socket Mode**
2. Activate **Incoming Webhooks**
3. Create **Slash Commands**
   - `/ask` Ask a question or make a request
   - `/modify_bot` Modify bot's configuration for the current channel 
   - `/bot_info` Get *prompt* and *temperature* of the bot in the current channel
   - `/permissions` (optional)  Modify which users can interact with the bot
   - `/edit_docs` Modify documents uploaded to the bot in the current channel
4. Enable **Events**
   - Subscribe to `app_mention`
5. Set **Scopes**
   - `app_mention:read`
   - `channels:history`
   - `channels:join`
   - `channels:read`
   - `chat:write`
   - `files:read`
   - `im:write`  _<- To notify users about change in permissions_
   - `users:read`  _<- To get list of users_

    _Note that for **groups** you will require also `groups:history`, `groups:join` and `groups:read`_ 
## ToDo / Ideas
- [x] Add a command to modify which users can interact with the bot. The command should be initialized using a password, example `/permissions <PASSWORD>`
- [x] A `ingest` method to create a vector database and use a QA retriever
- [x] add a custom CallbackHandler to update the messages on the go
- [x] A modal view to modify files description
- [] a method to remove files  from the vectorstore
- [ ] Create a doc retriever for each document, currently is using the same approach from [privateGPT](https://github.com/imartinez/privateGPT)
- [ ] A way to delete unused QA threads (time limit?)
- [ ] Deployment in [Modal.com](https://modal.com/)
- [ ] Create tests

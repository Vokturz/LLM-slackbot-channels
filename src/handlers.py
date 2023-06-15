from . import prompts
from langchain.llms import GPT4All, OpenAI
import time
from pathlib import Path
import requests

model = '/mnt/coldstorage/vnavarro/models/ggml-vicuna-13b-1.1-q4_2.bin'
files_path = 'files'

def get_llm(model_type):
    if model_type == 'GPT4All':
        llm = GPT4All(model=model, temp=0, n_predict=128, top_p=0.95, n_threads=20)
    elif model_type == 'OpenAI':
        llm =  OpenAI(temperature=0, max_tokens=128, top_p=0.95)
    return llm

def create_handlers(bot, model_type='GPT4All', show_time=False):
    @bot.command("/ask")
    def ask(ack, respond, command, say):
        user = command["user_id"]
        query = command["text"]
        ack()
        prompt = prompts.DEFAULT_PROMPT
        llm = get_llm(model_type)
        to_all = False
        if query.startswith('all'):
            query = query.replace('all ', '')
            to_all = True
        start_time = time.time()
        resp_llm = llm(prompt.format(query=query))
        final_time = round((time.time() - start_time)/60,2)

        response = f"\n*Answer*: {resp_llm}"
        if show_time:
            response += f"\n (_time: `{final_time}` min_)"
        if to_all:
            response = f"*<@{user}> asked*: {query}" + response
            say(response)
        else:
            response = f"*You asked*: {query}" + response
            respond(response)
            
    # @bot.event("file_shared")
    # def file_func(payload, client, ack):
    #     ack()
    #     print(payload)
    #     my_file = payload['file'].get('id')

    #     file_info = client.files_info(file=my_file).get('file')
    #     file_name = file_info.get('title')

    #     # Send a message to the user asking if they want to download the file
    #     file_format = '.'+file_name.split('.')[-1]
    #     if file_format in ['.pdf', '.csv', '.docx', 'doc']:
    #         client.chat_postMessage(
    #             channel=payload['channel_id'],
    #             blocks=[
    #                 {"type": "section",
    #                 "text": {
    #                     "type": "mrkdwn",
    #                     "text": f"Do you want to ingest the file {file_name}?"}
    #                 },
    #                 {"type": "actions",
    #                 "elements": [
    #                     {
    #                         "type": "button",
    #                         "text": {
    #                             "type": "plain_text",
    #                             "text": "Ingest"
    #                             },
    #                             "value": my_file,  # Store the file ID in the button
    #                             "action_id": "ingest_file"  # Identifier for the button action
    #                         },
    #                         {
    #                             "type": "button",
    #                             "text": {
    #                                 "type": "plain_text",
    #                                 "text": "Discard"
    #                             },
    #                             "value": my_file,  
    #                             "action_id": "discard_file" 
    #                         }
    #                     ]
    #                 }
    #             ]
    #         )

    # @bot.action("ingest_file")
    # def ingest_file(ack, body, client):
    #     ack()

    #     # Get the file ID from the button value
    #     my_file = body['actions'][0]['value']

    #     # Download the file as before
    #     file_info = client.files_info(file=my_file).get('file')
    #     url = file_info.get('url_private')
    #     file_name = file_info.get('title')
    #     token = bot.get_bot_token()
    #     resp = requests.get(url, headers={'Authorization': 'Bearer %s' % token})
    #     save_file = Path(f'{files_path}/{file_name}')
    #     save_file.write_bytes(resp.content)
    #     # Update the original message
    #     print(body['message'])
    #     client.chat_update(
    #         channel=body['channel']['id'],
    #         ts=body['message']['ts'],
    #         text=f"File {file_name} has been downloaded successfully!",
    #         #blocks=None  # Remove the buttons
    #     )

    # @bot.action("discard_file")
    # def discard_file(ack, body, client):
    #     ack()
    #     my_file = body['actions'][0]['value']
    #     file_info = client.files_info(file=my_file).get('file')
    #     file_name = file_info.get('title')

    #     # Update the original message
    #     client.chat_update(
    #         channel=body['channel']['id'],
    #         ts=body['message']['ts'],
    #         text=f"File {file_name} has been discarded!",
    #         blocks=None  # Remove the buttons
    #     )

    @bot.event("app_mention")
    def handle_mention(body, say):
        user = body["event"]["user"]
        text = body["event"]["text"]
        say(f"Hey, <@{user}>!")

    @bot.event("message")
    def handle_message(body, say):
        return
    

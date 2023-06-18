from . import prompts
import json
import os
import re
import time
import asyncio
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
files_path = "files"

def parse_format_body(body, bot_user_id=None):
    to_all = False
    change_temp = False
    try: # command
        user_id = body["user_id"]
        channel_id = body['channel_id']
        from_command = True
    except: # event
        user_id = body["user"]
        channel_id = body["channel"]
        from_command = False
    query = body["text"]
    if '!all' in query:
        query = query.replace('!all', '').strip()
        to_all = True
    new_temp = re.search(r'!temp=([\d.]+)', query)
    if new_temp:
        try:
            new_temp = float(new_temp.group(1))
            assert 0<=new_temp<=1
            change_temp = True
        except:
            new_temp = -1
            change_temp = False
        query = re.sub(r'!temp=([\d.]+)', '', query).strip()

    if bot_user_id:
        query = query.replace(f'<@{bot_user_id}>', '')
    res = {'query' : query,
           'user_id' : user_id,
           'channel_id' : channel_id,
           'new_temp' : new_temp,
           'to_all' : to_all,
           'change_temp' : change_temp,
           'from_command' : from_command}
    return res

def get_ask_classifier(embeddings_clf, phrase):
    search_index = Chroma.from_texts(texts=['opinion', 'coding'], embedding=embeddings_clf)
    res = search_index.similarity_search_with_score(phrase, k=2)
    classif = res[0][0].page_content
    score_diff = res[1][1] - res[0][1]
    if score_diff < 0.1:
        return 'opinion'
    return classif

async def get_llm_reply(bot, say, respond, prompt, parsed_body):
    #embd_clf = bot.get_embeddings()[1]
    #classif = get_ask_classifier(embd_clf, query)
    llm = bot.get_llm()
    channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])
    actual_temp = channel_llm_info['temperature']
    temp = actual_temp
    # format only if is not a thread
    if 'thread_ts' not in parsed_body.keys():
        final_prompt = prompt.format(query=parsed_body['query'])
        thread_ts = None
    else:
        final_prompt = prompt
        thread_ts = parsed_body['thread_ts']
        
    n_tokens =  llm.get_num_tokens(final_prompt)
    
    if parsed_body['to_all']:
        init_msg = "Bot is thinking.. :hourglass_flowing_sand:"
        if parsed_body['from_command']:
            init_msg = (f"*<@{parsed_body['user_id']}> asked*:"
                        f" {parsed_body['query']}\n" + init_msg)

        initial_message = await say(init_msg, thread_ts=thread_ts)
    else:
        initial_message = None

    llm_call = asyncio.Lock()
    async with llm_call:
        start_time = time.time()
        bot.change_temperature(temperature=actual_temp)
        if parsed_body['change_temp']:
                bot.change_temperature(temperature=parsed_body['new_temp'])
                temp = parsed_body['new_temp']
        if parsed_body['new_temp'] == -1:
                if parsed_body["from_command"]:
                    await respond(f"`!temp` only accepts values between 0 and 1."
                                  f" Using current value of `{actual_temp}`")
                else:
                    await say(f"`!temp` only accepts values between 0 and 1."
                              f" Using current value of `{actual_temp}`", thread_ts=thread_ts)
                temp = actual_temp
        if bot._model_type == 'fakellm':
            await asyncio.sleep(10)
        resp_llm = await bot.generate_response(final_prompt)
        response = resp_llm.strip()
        final_time = round((time.time() - start_time)/60,2)
        bot.change_temperature(temperature=actual_temp)

    if bot.get_verbose():
        response += f"\n (_time: `{final_time}` min. `temperature={temp}, n_tokens={n_tokens}`_)"
        #response += f" Your question was classified as *{classif}*_)"

    return response, initial_message
    



def create_handlers(bot):
    @bot.command("/ask")
    async def ask(ack, respond, command, say, client):
        parsed_body = parse_format_body(command)
        channel_id = parsed_body["channel_id"]
        user_id = parsed_body["user_id"]
        channel_llm_info = bot.get_channel_llm_info(channel_id)
        prompt = PromptTemplate(template=prompts.DEFAULT_PROMPT,
                                input_variables=['personality',
                                                 'instructions',
                                                 'query'])
        channel_prompt = prompt.partial(personality=channel_llm_info['personality'],
                                        instructions=channel_llm_info['instructions'])
        
        
        await ack()

        response, initial_message = await get_llm_reply(bot, say, respond, channel_prompt,
                                                         parsed_body)
        response = f"*<@{user_id}> asked*: {parsed_body['query']}\n*Answer*:\n{response}"
        if parsed_body['to_all']:
            await client.chat_update(
                    channel=channel_id,
                    ts=initial_message['ts'],
                    text=response
                )
        else:
            await respond(response)


    @bot.command("/modify_bot")
    async def modify_bot(ack, body, client, logger, respond):
        await ack()
        channel_id = body['channel_id']
        trigger_id = body['trigger_id']
        if channel_id[0] not in ['C', 'G']:
            respond(text='This command can only be used in channels.')
            return
        
        with open(f'{current_directory}/payloads/modify_bot_template.json', 'r') as f:
            template = f.read()
        
        channel_bot_info = bot.get_channel_llm_info(channel_id)
            
        
        template = template.format(personality=channel_bot_info['personality'],
                                instructions=channel_bot_info['instructions'],
                                temperature=channel_bot_info['temperature'])
        view = json.loads(template)

        # include channel_id
        view["private_metadata"] =  json.dumps({"channel_id": channel_id})
        await client.views_open(trigger_id=trigger_id, view=view)
        
    @bot.view('modify_bot')
    async def handle_modify_bot(ack, body, logger, say, view):
        values = view['state']['values']
        bot_values = {'personality': '',
                      'instructions': '',
                      'temperature' : ''
                      }
        channel_id = json.loads(view["private_metadata"])["channel_id"]
        user = body['user']['id']
        for key in bot_values.keys():
            input_value = values[key][key]['value']
            if key == 'temperature':
                try:
                    input_value = float(input_value)
                    assert 0<=input_value<=1
                except:
                    ack({"response_action": "errors",
                        "errors": {key: "The input field must be a number between 0 and 1."}})
                    return        
            bot_values[key] = input_value
        bot.define_channel_llm_info(channel_id, bot_values)
        await ack()
        await say(f'<@{user}> has modified the bot info', channel=channel_id)

    @bot.command("/bot_info")
    async def get_bot_info(ack, respond, command):
        await ack()
        channel_id = command['channel_id']
        if channel_id[0] not in ['C', 'G']:
            respond(text='This command can only be used in channels.')
            return
        bot_info = bot.get_channel_llm_info(channel_id)
        prompt = prompts.INITIAL_BOT_PROMPT
        res = "*Default Prompt:*\n`"
        res += prompt.format(personality=bot_info["personality"],
                              instructions=bot_info["instructions"])
        res += f"`\n*Temperature:* {bot_info['temperature']}"
        await respond(res)

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
    async def handle_mention(body, say, logger, client, respond):
        bot_user_id = bot._bot_user_id
        parsed_body = parse_format_body(body["event"], bot_user_id=bot_user_id)
        parsed_body['to_all'] = True
        channel_id = parsed_body["channel_id"]
        channel_llm_info = bot.get_channel_llm_info(channel_id)

        try:
            thread_ts = body['event']['thread_ts']
            parsed_body['thread_ts'] = thread_ts
        except:
            thread_ts = None
        if thread_ts:
            try:
                messages_history, users = await bot.extract_thread_conversation(channel_id, thread_ts)

                prompt = PromptTemplate(template=prompts.THREAD_PROMPT,
                                        input_variables=['personality',
                                                         'instructions',
                                                         'users',
                                                         'conversation'])
                
                final_prompt = prompt.format(personality=channel_llm_info['personality'],
                                             instructions=channel_llm_info['instructions'],
                                             users=' '.join(list(users)),
                                             conversation='\n'.join(messages_history))

                response, initial_message = await get_llm_reply(bot, say, respond,
                                                                final_prompt,
                                                                parsed_body)
                await client.chat_update(
                        channel=channel_id,
                        ts=initial_message['ts'],
                        text=response
                    )
            except Exception as e:
                logger.error(f"Error {e}")
        else:
            say("If you want to ask something, use command */ask*."
                " If you want to include me on a discussion"
                ", you have to mention me in a _thread_.")
        #if len(text.split(' ')) > 1:
        #    say(f"Hey, <@{user}>!, my name is <@{bot_user_id}>", thread_ts=thread_ts)

    # @bot.event("message")
    # def handle_message(body, say):
    #     return
    

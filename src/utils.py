from . import prompts
import os
import re
import time
import asyncio
from typing import (Dict, Optional, Any, Union, Tuple, List, Set)
from slack_bolt import (Say, Respond)
from .slackbot import SlackBot

# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))


def parse_format_body(body: Dict[str, Any],
                      bot_user_id: Optional[bool]=None
                      ) -> Dict[str, Union[str, float]]:
    """
    Parses a message body and extracts relevant information in a dictionary format.
    """
    to_all = False
    change_temp = False
    try: # Message is a command
        user_id = body["user_id"]
        channel_id = body['channel_id']
        from_command = True
    except: # Message is an event
        user_id = body["user"]
        channel_id = body["channel"]
        from_command = False
    query = body["text"]

    # Check if message includes !all
    if '!all' in query: 
        query = query.replace('!all', '').strip()
        to_all = True
    
    # Check if message includes !temp
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

    # Remove bot user id
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

async def get_llm_reply(bot: SlackBot, 
                        say: Say, 
                        respond: Respond, 
                        prompt: str, 
                        parsed_body: Dict[str, Union[str, float]]
                        ) -> Tuple[str, Optional[Say]]:
    """
    Generate a response using the bot's language model, given a prompt and
    a parsed request data.
    """
    channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])
    actual_temp = channel_llm_info['temperature']
    temp = actual_temp

     # format prompt and get thread timestamp
    if 'thread_ts' not in parsed_body.keys():
        final_prompt = prompt.format(query=parsed_body['query'])
        thread_ts = None
    else:
        final_prompt = prompt
        thread_ts = parsed_body['thread_ts']
    
    # send initial message if applicable
    if parsed_body['to_all']:
        init_msg = f"{bot.name} is thinking.. :hourglass_flowing_sand:"
        if parsed_body['from_command']:
            init_msg = (f"*<@{parsed_body['user_id']}> asked*:"
                        f" {parsed_body['query']}\n" + init_msg)
        initial_message = await say(init_msg, thread_ts=thread_ts)
    else:
        initial_message = None

    # generate response using language model
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
        if bot.model_type == 'fakellm':
            await asyncio.sleep(10)
        resp_llm = await bot.generate_response(final_prompt)
        response = resp_llm.strip()
        final_time = round((time.time() - start_time)/60,2)
        bot.change_temperature(temperature=actual_temp)

    if bot.verbose:
        n_tokens =  bot.llm.get_num_tokens(final_prompt)
        response += f"\n(_time: `{final_time}` min. `temperature={temp}, n_tokens={n_tokens}`_)"

    return response, initial_message

async def extract_thread_conversation(bot: SlackBot, channel_id:str,
                                        thread_ts: float
                                        ) -> Tuple[List[str], Set[str]]:
    """
    Extracts a conversation thread from a given channel and thread timestamp
    """
    client = bot.app.client
    bot_user_id = bot.bot_user_id
    result = await client.conversations_replies(channel=channel_id, ts=thread_ts)
    messages = result['messages']
    actual_user = ''
    users = set()
    messages_history = []
    for msg in messages:
        user = msg['user']
        text = msg['text'].replace(f'<@{bot_user_id}>', '').strip()
        if user == bot_user_id:
                if bot.verbose:
                    text = re.sub(r'\(_time: .*?\)', '', text)
                    text = re.sub(r'_Thread too long:.+_', '', text)
                    text = text.replace('\n\n', '\n')
                messages_history.append(f'AI: {text}')
                actual_user = user
        else:
            text = re.sub(r'!temp=([\d.]+)', '', text)
            if actual_user != user:
                users.add(f'<@{user}>') # if was not added
                messages_history.append(f'<@{user}>: {text}')
                actual_user = user
            else: # The same user is talking
                messages_history[-1] += f'\n{text}'
    return messages_history, users
import os
import re
import time
import asyncio
from typing import (Dict, Optional, Any, Union, Tuple, List, Set)
from slack_bolt import (Say, Respond)
from .slackbot import SlackBot
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from chromadb.config import Settings

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

async def prepare_messages_history(bot: SlackBot,
                                   parsed_body: Dict[str, Union[str, float]],
                                   to_chain: Dict[str, Any],
                                   qa_prompt : Optional[PromptTemplate]
                                   ) -> Tuple[Dict[str, Any],
                                              Optional[float],
                                              Optional[PromptTemplate],
                                              str]:
    """
    If it is a thread, prepare the message history to be sent to the bot.
    """
    if 'thread_ts' not in parsed_body.keys():
        to_chain['query'] = parsed_body['query']
        thread_ts = None
        warning_msg = ""
    else:
        thread_ts = parsed_body['thread_ts']
        messages_history, users = await extract_thread_conversation(bot,
                                                                    parsed_body['channel_id'],
                                                                    thread_ts)
        messages_history, warning_msg = custom_token_memory(bot, messages_history)
        
        if qa_prompt:
            qa_prompt = qa_prompt.partial(**to_chain)
            messages_history = messages_history[2:-1]
        to_chain['chat_history'] = ('\n'.join(messages_history).replace('\n\n', '\n'))
        to_chain['users'] = ' '.join(list(users))
    return to_chain, thread_ts, qa_prompt, warning_msg

async def send_initial_message(bot: SlackBot,
                               parsed_body: Dict[str, Union[str, float]],
                               thread_ts: Optional[float]) -> Optional[float]:
    """
    Send a initial message "bot is thinking.."
    """
    if parsed_body['to_all']:
        init_msg = f"{bot.name} is thinking.. :hourglass_flowing_sand:"
        if parsed_body['from_command']:
            init_msg = (f"*<@{parsed_body['user_id']}> asked*:"
                        f" {parsed_body['query']}\n" + init_msg)
        client = bot.app.client
        msg = await client.chat_postMessage(channel=parsed_body['channel_id'],
                                      text=init_msg, thread_ts=thread_ts)
        initial_ts = msg['ts']
    else:
        initial_ts = None
    return initial_ts


async def adjust_bot_temperature(bot: SlackBot,
                                 parsed_body: Dict[str, Union[str, float]]
                                 ) -> float:
    """
    Set bot temperature according to parsed_body
    """
    actual_temp = bot.get_temperature()
    temp = actual_temp
    if parsed_body['change_temp']:
        bot.change_temperature(temperature=parsed_body['new_temp'])
        temp = parsed_body['new_temp']
    if parsed_body['new_temp'] == -1:
        if parsed_body["from_command"]:
            client = bot.app.client
            warning_msg = (f"`!temp` only accepts values between 0 and 1."
                           f" Using current value of `{actual_temp}`")
            await client.chat_postEphemeral(channel=parsed_body['channel_id'],
                                            text=warning_msg)
    return temp

async def get_llm_reply(bot: SlackBot, 
                        prompt: PromptTemplate, 
                        parsed_body: Dict[str, Union[str, float]],
                        first_ts : Optional[float]=None,
                        qa_prompt : Optional[PromptTemplate]=None
                        ) -> Tuple[str, Optional[Say]]:
    """
    Generate a response using the bot's language model, given a prompt and
    a parsed request data.
    """
    channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])
    actual_temp = channel_llm_info['temperature']

    # dictionary to format the prompt inside the chain
    to_chain = {k: channel_llm_info[k] for k in ['personality', 'instructions']}

    # format prompt and get thread timestamp
    (to_chain, thread_ts,
      qa_prompt,warning_msg) = await prepare_messages_history(bot,
                                                              parsed_body,
                                                              to_chain,
                                                              qa_prompt)
    # send initial message
    initial_ts = await send_initial_message(bot, parsed_body, thread_ts)
    

    # generate response using language model
    llm_call = asyncio.Lock()
    async with llm_call:
        start_time = time.time()
        temp = await adjust_bot_temperature(bot, parsed_body)

        if bot.model_type == 'fakellm':
            await asyncio.sleep(10)

        if bot.verbose:
            bot.app.logger.info('Getting response..')
        # generate response
        if qa_prompt:
            # is a QA question, requires context
            db_path = bot.get_thread_retriever_db_path(parsed_body['channel_id'],
                                                        first_ts)
            vectorstore = Chroma(persist_directory=db_path,
                                 embedding_function=bot.embeddings,
                                 client_settings=Settings(
                                            chroma_db_impl='duckdb+parquet',
                                            persist_directory=db_path,
                                            anonymized_telemetry=False)
                                )
            prompt = prompt.partial(personality=to_chain['personality'],
                                    instructions=to_chain['instructions'],
                                    users=to_chain['users'])
            chain = ConversationalRetrievalChain
            chain = chain.from_llm(bot.llm,
                                   vectorstore.as_retriever(kwargs={'k': bot.k_similarity}),
                                   combine_docs_chain_kwargs={"prompt" : qa_prompt},
                                   condense_question_prompt=prompt,
                                   get_chat_history=lambda x : x)
            resp_llm = await chain.arun({'question': parsed_body['query'],
                                         'chat_history': to_chain['chat_history']})
            
        else:
            # is not a QA question
            chain = LLMChain(llm=bot.llm, prompt=prompt)
            resp_llm = await chain.arun(to_chain)
        response = resp_llm.strip()
        
        final_time = round((time.time() - start_time)/60,2)
        bot.change_temperature(temperature=actual_temp)

    if bot.verbose:
        if qa_prompt:
            to_chain["question"] = parsed_body["query"]
        n_tokens =  bot.llm.get_num_tokens(prompt.format(**to_chain))
        response += f"\n(_time: `{final_time}` min. `temperature={temp}, n_tokens={n_tokens}`_)"
        bot.app.logger.info(response.replace('\n', ''))
        response += warning_msg
    return response, initial_ts


async def extract_first_message_thread(bot: SlackBot,
                                       channel_id:str,
                                       thread_ts: float
                                        ) -> Dict[str, Any]:
    """
    Extracts the first message from a given channel and thread timestamp
    """
    client = bot.app.client
    result = await client.conversations_replies(channel=channel_id, ts=thread_ts)    
    messages = result['messages']
    return messages[0]

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
                users.add(f'<@{user}>') # if it was not added
                messages_history.append(f'<@{user}>: {text}')
                actual_user = user
            else: # The same user is talking
                messages_history[-1] += f'\n{text}'
    return messages_history, users


def custom_token_memory(bot: SlackBot,
                        messages_history: List[str]
                        ) -> Tuple[List[str], str]:
    """
    Remove first messages if the thread exceed the max tokens
    This follow the ConversationTokenBufferMemory approach from langchain
    """
    warning_msg = ""

    init_n_tokens = bot.llm.get_num_tokens('\n'.join(messages_history)
                                            .replace('\n\n', '\n'))
    if init_n_tokens > bot.max_tokens_threads:
        n_removed = 0
        n_tokens = init_n_tokens
        while n_tokens > bot.max_tokens_threads:
            messages_history.pop(0)
            n_removed += 1
            n_tokens = bot.llm.get_num_tokens('\n'.join(messages_history)
                                                .replace('\n\n', '\n'))
        if bot.verbose:
            warning_msg = (f"\n_Thread too long: `{init_n_tokens} >"
                           f" (max_tokens_threads={bot.max_tokens_threads})`,"
                           f" first {n_removed} messages were removed._")
            bot.app.logger.warning(warning_msg)
    return messages_history, warning_msg

import os
import re
import time
import asyncio
from typing import (Dict, Optional, Any, Union, Tuple, List, Set)
from .slackbot import SlackBot
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from .slackcallback import SlackAsyncCallbackHandler, SlackCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.agents import Tool
from langchain.llms.base import LLM

# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))


def parse_format_body(body: Dict[str, Any],
                      bot_user_id: Optional[bool]=None
                      ) -> Dict[str, Union[str, float]]:
    """
    Parses a message body and extracts relevant information as a dictionary.

    Args:
        body: The message body from the Slack API.
        bot_user_id: The bot user id.

    Returns:
        res: A dictionary containing the relevant information from the body
             with the following keys.
             - query: The text of the message.
             - user_id: The user id of the user who sent the message.
             - channel_id: The channel id of the message.
             - to_all: Whether the response must be sent to all channel members
                       or just the user who sent the message.
             - change_temp: Whether the default temperature of the bot should be
                             modified.
             - from_command: Whether the message comes from a command or an event.
    """
    # if XX_id comes from command, otherwise it comes from event
    user_id = body.get("user_id", body.get("user"))
    channel_id = body.get("channel_id", body.get("channel"))
    query = body["text"]

    # Check if message includes !all
    to_all = '!all' in query
    query = query.replace('!all', '') if to_all else query
    
    # Check if message includes !temp adn extract it
    new_temp_match = re.search(r'!temp=([\d.]+)', query)
    new_temp = float(new_temp_match.group(1)) if new_temp_match else -1
    # check if temp is valid
    change_temp = 0 <= new_temp <= 1
    query = re.sub(r'!temp=([\d.]+)', '', query) if new_temp_match else query

    # Remove bot user id
    query = query.replace(f'<@{bot_user_id}>', '') if bot_user_id else query

    res = {'query' : query.replace('  ', ' ').strip(),
           'user_id' : user_id,
           'channel_id' : channel_id,
           'new_temp' : new_temp if change_temp else -1,
           'to_all' : to_all,
           'change_temp' : change_temp,
           'from_command' : "user_id" in body}
    return res

async def prepare_messages_history(
        bot: SlackBot,
        parsed_body: Dict[str, Union[str, float]],
        first_ts: Optional[str],
        to_chain: Dict[str, Any],
        qa_prompt : Optional[PromptTemplate]
        ) -> Tuple[Dict[str, Any], Optional[PromptTemplate], str]:
    """
    If it is a thread, prepare the message history to be sent to the bot.

    Args:
        bot: The Slackbot object.
        parsed_body: The relevant information from the body obtained from
                     parse_format_body.
        first_ts: The timestamp of the first message (if it is a thread).
        to_chain: A dictionary containing the varoables to be used in the chain.
        qa_prompt: The QA PromptTemplate object.
    
    Returns:
        to_chain: A dictionary containing the variables to be used in the chain.
                  The chat history and list of users in the conversartion are
                  added to this dictionary.
        qa_prompt: The QA PromptTemplate object partially formatted.
        warning_msg: A warning message about the tokens limit.
    """
    if not first_ts:
        to_chain['query'] = parsed_body['query']
        warning_msg = ""
        if bot.get_channel_llm_info(parsed_body['channel_id'])['as_agent']:
            to_chain['chat_history'] = ""
            to_chain['users'] = f"<@{parsed_body['user_id']}>"
    else:
        messages_history, users = await extract_thread_conversation(bot,
                                                                    parsed_body['channel_id'],
                                                                    first_ts)
        messages_history, warning_msg = custom_token_memory(bot, messages_history)
        
        if qa_prompt:
            qa_prompt = qa_prompt.partial(**to_chain)
            messages_history = messages_history[2:-1]
        to_chain['chat_history'] = '\n'.join(messages_history).replace('\n\n', '\n')
        to_chain['users'] = ' '.join(list(users))
    return to_chain, qa_prompt, warning_msg

async def send_initial_message(bot: SlackBot,
                               parsed_body: Dict[str, Union[str, float]],
                               thread_ts: Optional[float]) -> Optional[str]:
    """
    Send a initial message: "bot is thinking.."

    Args:
        bot: The Slackbot object.
        parsed_body: The relevant information from the body obtained from
                     parse_format_body.
        thread_ts: The thread timestamp.

    Returns:
        initial_ts: The timestamp of the initial message only if to_all
                    is True in parsed_body.

    """
    if not parsed_body['to_all']:
        return None
    
    init_msg = f"{bot.name} is thinking.. :hourglass_flowing_sand:"
    if parsed_body['from_command']:
        init_msg = (f"*<@{parsed_body['user_id']}> asked*:"
                    f" {parsed_body['query']}\n" + init_msg)
    client = bot.app.client
    initial_ts = (await client.chat_postMessage(channel=parsed_body['channel_id'],
                                    text=init_msg, thread_ts=thread_ts))['ts']

    return initial_ts

def get_temperature(llm: LLM) -> float:
    """
    Returns the temperature of the given language model.
    Defaults to 0 for FakeLLM.
    """
    if 'model_type' in llm.__dict__: # CTransformers
        temperature = llm.client.config.temperature
    else: 
        try: # OpenAI
            temperature = llm.temperature
        except: # FakeLLM
            temperature = 0
    return temperature

def change_temperature(llm: LLM, new_temperature: float) -> None :
    """
    Changes the temperature of the given language model.
    """
    if 'model_type' in llm.__dict__: # CTransformers
        llm.client.config.temperature = new_temperature
    else:
        try: # OpenAI
            llm.temperature = new_temperature 
        except: # FakeLLM
            pass

def adjust_llm_temperature(llm: LLM,
                           parsed_body: Dict[str, Union[str, float]]
                           ) -> float:
    """
    Adjusts the LLM's temperature based on the parsed body,
    returning the new temperature.
    """
    actual_temp = get_temperature(llm)
    temp = actual_temp
    if parsed_body['change_temp']:
        change_temperature(llm, new_temperature=parsed_body['new_temp'])
        temp = parsed_body['new_temp']
    return temp

async def get_llm_reply(bot: SlackBot, 
                        prompt: PromptTemplate, 
                        parsed_body: Dict[str, Union[str, float]],
                        first_ts : Optional[float] = None,
                        qa_prompt : Optional[PromptTemplate] = None
                        ) -> Tuple[str, Optional[str]]:
    """
    Generate a response using the bot's language model, given a prompt and
    a parsed request data.

    Args:
        bot: The Slackbot object.
        prompt: A PromptTemplate object containing the LLM prompt to be used.
        parsed_body: The relevant information from the body obtained from
                     parse_format_body.
        first_ts: The timestamp of the first message sent in the conversation.
                  Used only in a QA thread.
        qa_prompt: The QA PromptTemplate object.

    Returns:
        response: The generated response from the LLM.
        initial_ts: The timestamp of the initial message sent by the bot.
    """
    channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])
    llm = bot.get_llm_by_channel(channel_id=parsed_body['channel_id'])

    # dictionary to format the prompt inside the chain
    to_chain = {k: channel_llm_info[k] for k in ['personality', 'instructions']}

    # format prompt and get thread timestamp
    (to_chain, qa_prompt, warning_msg) = await prepare_messages_history(bot,
                                                            parsed_body,
                                                            first_ts,
                                                            to_chain,
                                                            qa_prompt)
    # send initial message
    initial_ts = await send_initial_message(bot, parsed_body, first_ts)
    
    if parsed_body['from_command']:
        initial_msg = f"*<@{parsed_body['user_id']}> asked*: {parsed_body['query']}\n"
    else:
        initial_msg = ""

    if parsed_body['to_all']:
        async_handler = SlackAsyncCallbackHandler(bot, channel_id=parsed_body['channel_id'], 
                                            ts=initial_ts, inital_message=initial_msg)
 
        handler = SlackCallbackHandler(bot, channel_id=parsed_body['channel_id'], 
                                            ts=initial_ts, inital_message=initial_msg)   
    else:
        async_handler = AsyncCallbackHandler()
        handler = BaseCallbackHandler()
        
    # generate response using language model
    llm_call = asyncio.Lock()
    async with llm_call:
        start_time = time.time()
        temp = adjust_llm_temperature(llm, parsed_body)

        if bot.model_type == 'fakellm':
            await asyncio.sleep(5)

        if bot.verbose:
            bot.app.logger.info('Getting response..')
        # generate response
        if qa_prompt:
            # is a QA question, requires context
            db_path = bot.get_retriever_db_path(parsed_body['channel_id'],
                                                        first_ts)
            vectorstore = Chroma(persist_directory=db_path,
                                 embedding_function=bot.embeddings,
                                 client_settings=Settings(
                                            chroma_db_impl='duckdb+parquet',
                                            persist_directory=db_path,
                                            anonymized_telemetry=False)
                                )
            prompt = prompt.partial(users=to_chain['users'])
            chain = ConversationalRetrievalChain
            chain = chain.from_llm(llm,
                                   vectorstore.as_retriever(kwargs={'k': bot.k_similarity}),
                                   combine_docs_chain_kwargs={"prompt" : qa_prompt},
                                   condense_question_prompt=prompt,
                                   get_chat_history=lambda x : x)
            try: 
                resp_llm = await chain.arun({'question': parsed_body['query'],
                                            'chat_history': to_chain['chat_history']},
                                            callbacks=[async_handler])
            except NotImplementedError:
                bot.app.logger.info('No Async generation implemented for this LLM'
                                    ', using concurrent mode')
                resp_llm = chain.run({'question': parsed_body['query'],
                            'chat_history': to_chain['chat_history']},
                              callbacks=[handler])
        else:
            # is not a QA question  
            chain = LLMChain(llm=llm, prompt=prompt)
            try:
                resp_llm = await chain.arun(to_chain, callbacks=[async_handler])
            except NotImplementedError:
                bot.app.logger.warning('No Async generation implemented for this LLM'
                                    ', using concurrent mode')  
                resp_llm = chain.run(to_chain, callbacks=[handler])

        response = resp_llm.strip()
        final_time = round((time.time() - start_time)/60,2)

    if bot.verbose:
        if qa_prompt:
            to_chain["question"] = parsed_body["query"]
        #n_tokens = llm.get_num_tokens(prompt.format(**to_chain))
        response += f"\n(_time: `{final_time}` min. `temperature={temp}`_)"
        bot.app.logger.info(response.replace('\n', ''))
        response += warning_msg
    return response, initial_ts

async def get_agent_reply(bot: SlackBot, 
                          parsed_body: Dict[str, Union[str, float]],
                          first_ts : Optional[float]=None,
                          qa_thread : Optional[bool]=False
                          ) -> Tuple[str, Optional[str]]:
    """
    Generate a response using the bot's language model as an agent

    Args:
        bot: The Slackbot object.
        parsed_body: The relevant information from the body obtained from
                     parse_format_body.
        first_ts: The timestamp of the first message sent in the conversation.
        qa_thread: Whether it is a QA thread or not.

    Returns:
        response: The generated response from the LLM.
        initial_ts: The timestamp of the initial message sent by the bot.
    """
    channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])

    llm = bot.get_llm_by_channel(channel_id=parsed_body['channel_id'])

    # dictionary to format the prompt inside the chain
    agent_info = {k: channel_llm_info[k] for k in ['personality', 'instructions', 'tool_names']}

    # format prompt and get thread timestamp
    (agent_info, _, warning_msg) = await prepare_messages_history(bot,
                                                        parsed_body,
                                                        first_ts,
                                                        agent_info,
                                                        None)
    
    # send initial message
    initial_ts = await send_initial_message(bot, parsed_body, first_ts)
    
    if parsed_body['from_command']:
        initial_msg = f"*<@{parsed_body['user_id']}> asked*: {parsed_body['query']}\n"
    else:
        initial_msg = ""

    if parsed_body['to_all']:
        async_handler = SlackAsyncCallbackHandler(bot, channel_id=parsed_body['channel_id'], 
                                            ts=initial_ts, inital_message=initial_msg)
 
        handler = SlackCallbackHandler(bot, channel_id=parsed_body['channel_id'], 
                                            ts=initial_ts, inital_message=initial_msg)   
    else:
        async_handler = AsyncCallbackHandler()
        handler = BaseCallbackHandler()
        
    # generate response using language model
    llm_call = asyncio.Lock()
    async with llm_call:
        start_time = time.time()
        temp = adjust_llm_temperature(llm, parsed_body)

        if bot.model_type == 'fakellm':
            await asyncio.sleep(5)

        if bot.verbose:
            bot.app.logger.info('Getting response..')
        
        agent_info['tools'] = bot.get_tools_by_names(agent_info['tool_names'])


        if first_ts and qa_thread:
             # Only files uploaded to this thread
            try:
                doc_retriever = await get_doc_retriever_tool(bot, llm, parsed_body,
                                                       first_ts)
                agent_info['tools'].extend(doc_retriever) 
            except KeyError:
                bot.app.logger.info('There are no documents for this thread')
        else:
            # All files uploaded to the channel via the bot
            # Only if the tool is included in the tools to use
            if 'doc_retriever' in agent_info['tool_names']:
                try:
                    doc_retriever = await get_doc_retriever_tool(bot, llm, parsed_body)   
                    agent_info['tools'].extend(doc_retriever)
                except Exception as e:
                    bot.app.logger.info(e) 
        
        from .slackagent import slack_agent
        executor_agent = slack_agent(bot, llm, personality=agent_info['personality'],
                                    instructions=agent_info['instructions'],
                                    users=agent_info['users'],
                                    chat_history=agent_info['chat_history'],
                                    tools=agent_info['tools'],
                                    initial_ts=initial_ts,
                                    channel_id=parsed_body['channel_id']) 
        try: 
            resp_llm = await executor_agent.arun(input=parsed_body['query'], callbacks=[async_handler])
        except NotImplementedError:
                bot.app.logger.info('No Async generation implemented for this LLM'
                                    ', using concurrent mode')
                resp_llm = executor_agent.run(input=parsed_body['query'], callbacks=[handler])

        response = resp_llm.strip()
        final_time = round((time.time() - start_time)/60,2)

    if bot.verbose:
        # from .prompts import AGENT_PROMPT
        # to_chain["input"] = parsed_body["query"]
        # to_chain["agent_scratchpad"] = ""
        # n_tokens =  llm.get_num_tokens(AGENT_PROMPT.format(**to_chain))
        # n_tokens += 100 # an estimator
        response += f"\n(_time: `{final_time}` min. `temperature={temp}`_)"
        bot.app.logger.info(response.replace('\n', ''))
        response += warning_msg
    return response, initial_ts

async def extract_message_from_thread(bot: SlackBot,
                                       channel_id:str,
                                       thread_ts: float,
                                       position: int=0
                                        ) -> Dict[str, Any]:
    """
    Extracts a message from a given channel and thread timestamp

    Args:
        bot: The Slackbot object.
        channel_id: The channel id.
        thread_ts: The thread timestamp.
        position: The position of the message

    Returns:
        message: The first message from the given channel and thread timestamp
    """
    client = bot.app.client
    result = await client.conversations_replies(channel=channel_id, ts=thread_ts)    
    messages = result['messages']
    return messages[position]

async def extract_thread_conversation(bot: SlackBot, channel_id:str,
                                        thread_ts: float
                                        ) -> Tuple[List[str], Set[str]]:
    """
    Extracts a conversation thread from a given channel and thread timestamp.
    The conversation is cleaned removing the bot name and verbose messages.
    It has the following format:
        <@user_id1> : user1 message mentioning the bot
        AI : bot response
        ..

    Args:
        bot: The Slackbot object.
        channel_id: The channel id.
        thread_ts: The thread timestamp.

    Returns:
        message_history: Thwe list of messages in the conversation
        users: The set of users in the conversation

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

    Args:
        bot: The Slackbot object.
        messages_history: The list of messages in a conversation

    Returns:
        messages_history: The list of messages in a conversation. If the number
                          of tokens of the conversation exceeds the tokens limit 
                          (max_tokens_threads) of the bot, then the first N 
                          messages are removed.
        warning_msg: A warning message about the tokens limit.

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

async def get_doc_retriever_tool(bot: SlackBot, llm: LLM, 
                                 parsed_body: Dict[str, Union[str, float]],
                                 first_ts : Optional[str]='') -> Tool:
    """
    Create and return a doc_retriever tool, which can be used to retrieve
    information from documents

    Args:
        bot: The Slackbot object.
        llm: The LLM to be used in the RetrievalQA
        parsed_body: The relevant information from the body obtained from
                     parsed_format_body.
        first_ts: The timestamp of the initial message of a thread. If it is
                  empty, then the tool is build using the documents uploaded
                  to the bot via the channel.

    Returns:
        doc_retriever: The doc_retriever tool.
    """
    db_path = bot.get_retriever_db_path(parsed_body['channel_id'],
                                                    first_ts)
    vectorstore = Chroma(persist_directory=db_path,
                        embedding_function=bot.embeddings,
                        client_settings=Settings(
                                    chroma_db_impl='duckdb+parquet',
                                    persist_directory=db_path,
                                    anonymized_telemetry=False)
                        )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(kwargs={'k': bot.k_similarity})
    )
    
    if first_ts: # a QA Thread
        second_message = await extract_message_from_thread(bot, parsed_body['channel_id'],
                                                            first_ts, position=1)
        extra_context = re.search("The files are about (.*)", second_message['text']).group(1)

    else: # Directly from the channel
        channel_llm_info = bot.get_channel_llm_info(parsed_body['channel_id'])
        if not channel_llm_info['files']:
            raise KeyError('No files found')
        
        extra_context = ', '.join(channel_llm_info['files'].values())

    doc_retriever = [Tool(name="doc_retriever",
                        func=qa_chain.run,
                        coroutine=qa_chain.arun,
                        description=f"Useful for when you need to answer questions about {extra_context}.",
                        )]
    return doc_retriever
from . import prompts
from .slackbot import SlackBot
from .utils import (parse_format_body, get_llm_reply,
                    extract_thread_conversation)
import json
import os
import re
import time
import asyncio
from typing import (Dict, Any)
from langchain.prompts import PromptTemplate

from slack_bolt import (Say, Respond, Ack)

# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

def create_handlers(bot: SlackBot) -> None:
    @bot.command("/ask")
    async def handle_ask(ack: Ack, respond: Respond, say: Say,
                        command: Dict[str, Any]) -> None:
        """
        Handle the /ask command.
        If the command contains the word `!all` then the response is sent
        to the channel. If the command contains the word `!temp` then the
        default temperature is modified.
        """
        # Parse the command and retrieve relevant information
        parsed_body = parse_format_body(command)
        channel_id = parsed_body["channel_id"]
        user_id = parsed_body["user_id"]

        await ack()
        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Get the LLM information of the channel
        channel_llm_info = bot.get_channel_llm_info(channel_id)

        # Generate prompt for the channel
        prompt = PromptTemplate(template=prompts.DEFAULT_PROMPT,
                                input_variables=['personality',
                                                 'instructions',
                                                 'query'])
        channel_prompt = prompt.partial(personality=channel_llm_info['personality'],
                                        instructions=channel_llm_info['instructions'])
        
        

        # Get the response from the LLM
        response, initial_message = await get_llm_reply(bot, say, respond,
                                                        channel_prompt,
                                                        parsed_body)
        # Format the response
        response = f"*<@{user_id}> asked*: {parsed_body['query']}\n*Answer*:\n{response}"

        # Send the response to all channel members or just the user who asked
        if parsed_body['to_all']:
            client = bot.app.client
            await client.chat_update(channel=channel_id, ts=initial_message['ts'], 
                                     text=response)
        else:
            await respond(response)

    @bot.command("/modify_bot")
    async def handle_modify_bot(ack: Ack, body: Dict[str, Any],
                                respond : Respond) -> None:
        await ack()
        """
        Handle the /ask command.
        This function modifies a bot's personality, instructions, and temperature
        based on the channel it is in.
        """
        channel_id = body['channel_id']
        trigger_id = body['trigger_id']

        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Load modify_bot_template.json payload
        with open(f'{current_directory}/payloads/modify_bot_template.json', 'r') as f:
            template = f.read()
        
        # Get channel bot info
        channel_bot_info = bot.get_channel_llm_info(channel_id)
            
        # Replace template variables with channel bot info
        template = template.format(personality=channel_bot_info['personality'],
                                instructions=channel_bot_info['instructions'],
                                temperature=channel_bot_info['temperature'])
        # Convert template to JSON view
        view = json.loads(template)

        # Include channel_id in private_metadata
        view["private_metadata"] =  json.dumps({"channel_id": channel_id})

        # Open view for bot modification
        client = bot.app.client
        await client.views_open(trigger_id=trigger_id, view=view)
        
    @bot.view('modify_bot')
    async def handle_modify_bot_view(ack: Ack, body: Dict[str, Any],
                                     say: Say, view: Dict[str, Any]) -> None:
        """
        Handle the modify_bot view.
        """
        values = view['state']['values']
        bot_values = {'personality': '',
                      'instructions': '',
                      'temperature' : ''
                      }
        
        # Extract channel ID and user ID
        channel_id = json.loads(view["private_metadata"])["channel_id"]
        user = body['user']['id']

         # Iterate through each bot value and update it
        for key in bot_values.keys():
            input_value = values[key][key]['value']

            # Check if the input value is for temperature and validate it
            if key == 'temperature':
                try:
                    input_value = float(input_value)
                    assert 0<=input_value<=1
                except:
                    ack({"response_action": "errors",
                        "errors": {key: "The input field must be a number between 0 and 1."}})
                    return        
            bot_values[key] = input_value

        # Update channel's bot info
        bot.define_channel_llm_info(channel_id, bot_values)
        await ack()

        # Notify channel of bot modification
        await say(f'_*<@{user}> has modified the bot info*_', channel=channel_id)

    @bot.command("/bot_info")
    async def handle_bot_info(ack: Ack, respond: Respond,
                              command: Dict[str, Any]) -> None:
        """
        Handle the /bot_info command.
        Displays the initial prompt of the bot and its default temperature
        """
        await ack()
        channel_id = command['channel_id']

        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Retrieve the bot's info for the channel
        bot_info = bot.get_channel_llm_info(channel_id)

        # Create a response string with the bot's default prompt and temperature
        prompt = prompts.INITIAL_BOT_PROMPT
        response = "*Default Prompt:*\n`"
        response += prompt.format(personality=bot_info["personality"],
                              instructions=bot_info["instructions"])
        response += f"`\n*Temperature:* {bot_info['temperature']}"

        # Send the response to the user
        await respond(response)

    @bot.event("app_mention")
    async def handle_mention(say: Say, 
                             body: Dict[str, Any]) -> None:
        """
        Handle mentions of the bot in a channel. 
        If the mention is in a thread, it extracts the conversation thread 
        and generates a response using the LLM from the bot. 
        If the mention is not in a thread, it sends a message instructing users 
        to use the proper command or use a thread for discussion.
        """

        # Check if the mention comes from an edited message
        if "edited" in body["event"].keys():
            return
        
        # Get the ID of the bot user
        bot_user_id = bot.bot_user_id

        # Parse the body of the mention and set to_all flag to True
        parsed_body = parse_format_body(body["event"], bot_user_id=bot_user_id)
        parsed_body['to_all'] = True

        # Get the bot info asociated with the channel
        channel_id = parsed_body["channel_id"]
        # Ensure is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await say(text='Interaction only able in channels.')
            return
        channel_llm_info = bot.get_channel_llm_info(channel_id)

        try:
            # Check if mention was made in a thread and get the thread ID
            thread_ts = body['event']['thread_ts']
            parsed_body['thread_ts'] = thread_ts
        except:
            thread_ts = None
        if thread_ts:
            try:
                # Extract the conversation thread 
                messages_history, users = await extract_thread_conversation(bot, channel_id, thread_ts)
                warning = ""
                # Remove first messages if the thread exceed the max tokens
                # This follow the ConversationTokenBufferMemory approach from langchain
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
                        warning = (f"\n_Thread too long: `{init_n_tokens} > (max_tokens_threads={bot.max_tokens_threads})`,"
                                   f" first {n_removed} messages were removed._")
                # Generate the prompt
                prompt = PromptTemplate(template=prompts.THREAD_PROMPT,
                                        input_variables=['personality',
                                                         'instructions',
                                                         'users',
                                                         'conversation'])
                
                final_prompt = prompt.format(personality=channel_llm_info['personality'],
                                             instructions=channel_llm_info['instructions'],
                                             users=' '.join(list(users)),
                                             conversation='\n'.join(messages_history)
                                                             .replace('\n\n', '\n'))
                print(final_prompt)
                # Get reply and update initial message
                response, initial_message = await get_llm_reply(bot, say, None,
                                                                final_prompt,
                                                                parsed_body)
                client = bot.app.client
                await client.chat_update(
                        channel=channel_id,
                        ts=initial_message['ts'],
                        text=response + warning
                    )
            except Exception as e:
                bot.app.logger.error(f"Error {e}")
        else:
            # Send message instructing users to use the proper command
            # or use a thread for discussion
            await say("If you want to ask something, use command */ask*."
                      " If you want to include me on a discussion"
                      ", you have to mention me in a _thread_.")

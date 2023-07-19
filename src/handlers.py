from . import prompts
from .slackbot import SlackBot
from .utils import (parse_format_body, get_llm_reply,
                    extract_message_from_thread, get_agent_reply)
from .ingest import process_uploaded_files
import json
import os
import re
import asyncio
import copy
from typing import (Dict, Any)
from langchain.prompts import PromptTemplate
from slack_bolt import (Say, Respond, Ack)
import threading

# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

def create_handlers(bot: SlackBot) -> None:
    @bot.app.command("/ask")
    @bot.check_permissions
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

        if bot.verbose:
            bot.app.logger.info(f"/ask used by {user_id} in channel {channel_id}:"
                                f" {command['text']}")
        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Generate prompt for the channel
        prompt = PromptTemplate(template=prompts.DEFAULT_PROMPT,
                                input_variables=['personality',
                                                 'instructions',
                                                 'query'])

        channel_bot_info = bot.get_channel_llm_info(channel_id)
        # Get the response from the LLM
        if channel_bot_info['as_agent']:
            response, initial_ts = await get_agent_reply(bot, parsed_body, None)
        else:
            response, initial_ts = await get_llm_reply(bot, prompt, parsed_body)
        # Format the response
        response = f"*<@{user_id}> asked*: {parsed_body['query']}\n*Answer*:\n{response}"

        # Send the response to all channel members or just the user who asked
        if parsed_body['to_all']:
            client = bot.app.client
            await client.chat_update(channel=channel_id, ts=initial_ts, 
                                     text=response)
        else:
            await respond(text=response)


    @bot.app.command("/modify_bot")
    @bot.check_permissions
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

        if bot.verbose:
            bot.app.logger.info(f"/modify_bot used by {body['user_id']} in channel {channel_id}")

        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Load modify_bot_template.json payload
        with open(f'{current_directory}/payloads/modify_bot_template.json', 'r') as f:
            view = json.load(f)
        
        # Get channel bot info
        channel_bot_info = bot.get_channel_llm_info(channel_id)
            
        # Replace with channel bot info
        view["blocks"][0]["element"]["initial_value"] = channel_bot_info['personality']
        view["blocks"][1]["element"]["initial_value"] = channel_bot_info['instructions']
        view["blocks"][2]["element"]["initial_value"] = str(channel_bot_info['temperature'])

        # OpenAI model, only if model_type is openai
        if bot.model_type == 'openai':
            if 'openai_model' in channel_bot_info:
                initial_text = ("ChatModel: "
                                if channel_bot_info['openai_model'].startswith('gpt')
                                else "InstructModel: ")
                initial_option = {"text": {
                                        "type": "plain_text",
                                        "text": f"{initial_text}{channel_bot_info['openai_model']}"
                                    },
                            "value": channel_bot_info['openai_model']
                            }
                view["blocks"][3]["element"]["initial_option"] =  initial_option
        else:
            view["blocks"][3] = { "type": "section",
                                  "text": { "type": "plain_text", "text": " "}}

        # Agent or Chain
        if channel_bot_info['as_agent']:
            view["blocks"][4]["element"]["initial_option"]["value"] = "as_agent"
            view["blocks"][4]["element"]["initial_option"]["text"]["text"] = "Use it as an Agent"
        else:
            view["blocks"][4]["element"]["initial_option"]["value"] = "as_llm_chain"
            view["blocks"][4]["element"]["initial_option"]["text"]["text"] = "Use it as a LLM chain"


        # Tools
        all_options = []
        for tool in bot.tool_names:
            option = {
						"text": {
							"type": "plain_text",
							"text": tool
						},
						"value": tool
					}
            all_options.append(option)
        view["blocks"][5]["element"]["options"] = all_options

        # Include channel_id in private_metadata
        extra_data = {"channel_id": channel_id}
        if '!no-notify' in body['text']:
            extra_data["notify"] = False
        else:
            extra_data["notify"] = True
        view["private_metadata"] =  json.dumps(extra_data)

        initial_options = []
        for tool in channel_bot_info['tool_names']:
            if tool in bot.tool_names:
                option = {
                            "text": {
                                "type": "plain_text",
                                "text": tool
                            },
                            "value": tool
                        }
                initial_options.append(option)
        if initial_options:
            view["blocks"][5]["element"]["initial_options"] = initial_options

        # Open view for bot modification
        await bot.app.client.views_open(trigger_id=trigger_id, view=view)
        
    @bot.app.view('modify_bot')
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
        notify = json.loads(view["private_metadata"])["notify"]
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

        if 'openai_model' in values:
            bot_values['openai_model'] = values['openai_model']['openai_model']['selected_option']['value']
        as_agent = values['use_it_as']['unused_action']['selected_option']['value']
        bot_values['as_agent'] = True if 'as_agent' == as_agent else False
 
        selected_tools = values['tool_names']['unused_action']['selected_options']
        tool_names = [tool['value'] for tool in selected_tools]
        bot_values['tool_names'] = tool_names

        # Update channel's bot info
        bot.define_channel_llm_info(channel_id, bot_values)
        await ack()

        # Notify channel of bot modification
        if notify:
            await say(f'_*<@{user}> has modified the bot info*_', channel=channel_id)


    @bot.app.command("/bot_info")
    async def handle_bot_info(ack: Ack, respond: Respond,
                              command: Dict[str, Any]) -> None:
        """
        Handle the /bot_info command.
        Displays the initial prompt of the bot and its default temperature
        """
        await ack()
        channel_id = command['channel_id']
        if bot.verbose:
            bot.app.logger.info(f"/bot_info used by {command['user_id']} in channel {channel_id}")

        # Ensure command is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await respond(text='This command can only be used in channels.')
            return
        
        # Retrieve the bot's info for the channel
        bot_info = bot.get_channel_llm_info(channel_id)

        # Create a response string with the bot's default prompt and temperature
        prompt = prompts.INITIAL_BOT_PROMPT
        response = "*Default Prompt:*\n`"
        if not bot_info['tool_names']:
            bot_info['tool_names'] = ['None']
        response += prompt.format(personality=bot_info["personality"],
                              instructions=bot_info["instructions"])
        response += (f"`\n*Temperature:* {bot_info['temperature']}"
                     f"\n*is Agent:* _{bot_info['as_agent']}_,"
                     f" *Tools:* _" + ', '.join(bot_info['tool_names']) + '_')

        if bot.model_type == 'openai' and 'openai_model' in bot_info:
            response += (f"\n*OpenAI Model:* _{bot_info['openai_model']}_")

        # Send the response to the user
        await respond(text=response)


    @bot.app.command("/permissions") # Don't ask for allowed_users
    async def handle_permissions(ack: Ack, body: Dict[str, Any],
                                respond : Respond) -> None:
        await ack()

        """
        Handle the /permissions command. 
        By default all users have access to the bot
        Modify the list of users allowed to use the bot.
        This command requires a password to be entered
        """
        password = body["text"].strip()
        channel_id = body['channel_id']
        trigger_id = body['trigger_id']

        if bot.verbose:
            bot.app.logger.info(f"/permissions used by {body['user_id']} in channel {channel_id}")
        # Check if the password was defined
        try: 
            permissions_psswd = os.environ["PERMISSIONS_PASSWORD"]
            if permissions_psswd == "":
                return
        except:
            return
        
        # Check if the password is correct
        if password != permissions_psswd:
            await respond(text=":x: Incorrect password")
            return
        
        # Load permissions_template.json payload
        with open(f'{current_directory}/payloads/permissions_template.json', 'r') as f:
            view = json.load(f)
        
        # Get all the members of the Slack
        if "@all" in bot.allowed_users["users"]:
            users = (await bot.app.client.users_list())["members"]
            options = []
            for user in users:
                if user["id"] not in [bot.bot_user_id, "USLACKBOT"]:
                    options.append(user["id"])
        else:
            options = bot.allowed_users["users"]

        # Add options
        view["blocks"][0]["accessory"]["initial_users"] = options

        # Include channel_id in private_metadata
        view["private_metadata"] =  json.dumps({"channel_id": channel_id})
        # Open view for bot modification
        await bot.app.client.views_open(trigger_id=trigger_id, view=view)

    @bot.app.view('modify_permissions')
    async def handle_modify_permissions_view(ack: Ack, body: Dict[str, Any],
                                     say: Say, view: Dict[str, Any]) -> None:
        """
        Handle the modify_permissions view.
        """
        user = body['user']['id']
        channel_id = json.loads(view["private_metadata"])["channel_id"]
        allowed_users = view['state']['values']['allowed_users']
        checkbox = view['state']['values']['notify_users']['notify_users']
        first_item = next(iter(allowed_users))
        allowed_users = allowed_users[first_item]['selected_users']
        await ack()

        permissions_lock = asyncio.Lock()
        async with permissions_lock:
            if len(checkbox['selected_options'])>0:
                previous_list = bot.allowed_users["users"]
                messages = {}
                for _user in previous_list:
                    if _user not in allowed_users:
                        messages[_user] = (f"<@{user}> has removed your permission"
                                      f" to use <@{bot.bot_user_id}>.")
                for _user in allowed_users:
                    if _user not in previous_list:
                        messages[_user] = (f"<@{user}> has granted you permission"
                                      f" to use <@{bot.bot_user_id}>.")

                for _user,msg in messages.items():
                    client = bot.app.client
                    response = await client.conversations_open(users=_user)
                    user_channel = response['channel']['id']
                    await client.chat_postEphemeral(channel=user_channel ,
                                                    user=_user,
                                                    text=msg)
        bot.define_allowed_users(allowed_users)


    @bot.app.event("app_mention")
    @bot.check_permissions
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

        bot.app.logger.info(f"User {parsed_body['user_id']} mentioned the bot"
                            f" in channel {parsed_body['channel_id']}")
        # Get the bot info asociated with the channel
        channel_id = parsed_body["channel_id"]
        # Ensure is only used in channels
        if channel_id[0] not in ['C', 'G']:
            await say(text='Interaction only able in channels.')
            return
        try:
            # Check if mention was made in a thread and get the thread ID
            thread_ts = body['event']['thread_ts']
            parsed_body['thread_ts'] = thread_ts
        except:
            thread_ts = None
        if thread_ts:
            first_message = await extract_message_from_thread(bot, channel_id,
                                                             thread_ts, position=0)
           # Generate the prompt
            if 'files' not in first_message:
                # Is not a QA thread
                prompt = PromptTemplate.from_template(template=prompts.THREAD_PROMPT)
                if bot.verbose:
                    bot.app.logger.info(f"Asking thread"
                                        f" {channel_id}/{first_message['ts']}:"
                                        f" {parsed_body['query']}")
                qa_prompt = None
            else:
                # Is a QA thread
                prompt = PromptTemplate.from_template(template=prompts.CONDENSE_QUESTION_PROMPT)
                qa_prompt = PromptTemplate.from_template(template=prompts.QA_PROMPT)

                 # Bot message as extra context
                second_message = await extract_message_from_thread(bot, channel_id,
                                                                 thread_ts, position=1)
                extra_context = second_message['text']
                qa_prompt = qa_prompt.partial(extra_context=extra_context)
                if bot.verbose:
                    bot.app.logger.info(f"Asking inside a Thread. "
                                        f" {extra_context}:"
                                        f" {channel_id}/{first_message['ts']}:"
                                        f" {parsed_body['query']}")
                    
            # Get reply and update initial message
            channel_bot_info = bot.get_channel_llm_info(channel_id)
            if channel_bot_info['as_agent']:
                response, initial_ts = await get_agent_reply(bot,parsed_body, first_message['ts'])
            else:
                response, initial_ts = await get_llm_reply(bot, prompt, parsed_body,
                                                        first_ts=first_message['ts'],
                                                        qa_prompt=qa_prompt)

            client = bot.app.client
            await client.chat_update(
                    channel=channel_id,
                    ts=initial_ts,
                    text=response
                )
        else:
            if "files" in body['event'].keys():
                files =  body['event']['files']
                msg_timestamp = body['event']['ts']

                # store the file dict temporally
                bot.store_files_dict(msg_timestamp, files)

                # Sent a temporary message
                msg = ("Hey! looks like you have uploaded some files. Do you "
                      "want to interact with them?")
                await bot.app.client.chat_postEphemeral(
                    user=body['event']['user'],
                    channel=channel_id,
                    text="upload files",
                    blocks=[{
                        "type": "section",
                        "text": { "type": "mrkdwn", "text": msg},
                        "accessory": {"type": "button",
                                      "text": {"type": "plain_text",
                                               "text": "Yes"},
                                      "action_id": "files_button",
                                      "value" : msg_timestamp}}])
            else:
                # Send message instructing users to use the proper command
                # or use a thread for discussion
                await say("If you want to ask something, use command */ask*."
                        " If you want to include me on a discussion"
                        ", you have to mention me in a _thread_.")
                
            
    @bot.app.action("files_button")
    async def open_file_uploaded_modal(ack, body, respond):
        """
        Handle the files_button action
        """
        await ack()
        channel_id = body['container']['channel_id']
        msg_timestamp = body['actions'][0]['value']
        files = bot.get_stored_files_dict(msg_timestamp)
        first_message = await extract_message_from_thread(bot, channel_id,
                                                          msg_timestamp,
                                                          position=0)
        extra_separators = re.findall(r'!sep=(\S+)', first_message['text'])
        
        #Load upload_files_template.json
        with open(f'{current_directory}/payloads/upload_files_template.json', 'r') as f:
            view = json.load(f)

        extra_context_block = view['blocks'][0]
        extra_separators_block = view['blocks'][1]
        extra_separators_block['element']['initial_value'] = ';'.join(extra_separators)

        options_block = view['blocks'][2]
        view['blocks'] = []

        files_name_list = [f['name'] for f in files]
        for i, _file in enumerate(files_name_list):
            new_block = copy.deepcopy(extra_context_block)
            new_block['block_id'] = f"extra_context_{i}"
            new_block['element']['action_id'] = f"extra_context_{i}"
            new_block['element']['initial_value'] = _file
            new_block['label']['text'] = f"File '{_file}' is about"
            view['blocks'].append(new_block)
        view['blocks'].append(extra_separators_block)
        view['blocks'].append(options_block)

        # add to channel not implemented yet
        view['blocks'][-1]['accessory']['options'].pop()

        # Include channel_id in private_metadata
        view["private_metadata"] =  json.dumps({"channel_id": channel_id,
                                                "ts": msg_timestamp})
        trigger_id = body["trigger_id"]
        await bot.app.client.views_open(trigger_id=trigger_id, view=view)

        # Remove ephemeral message
        await respond({
                    'text': '',
                    'replace_original': True,
                    'delete_original': True
                })

    @bot.app.view('upload_files')
    async def handle_upload_files_view(ack: Ack,
                                       view: Dict[str, Any]) -> None:
        """
        Handle the upload files view.
        """
        await ack()

        # get temp files dict
        private_metadata = json.loads(view["private_metadata"])
        channel_id = private_metadata["channel_id"]
        msg_timestamp = private_metadata['ts']

        files = bot.get_stored_files_dict(msg_timestamp)
        file_name_list = [f["name"] for f in files]

        extra_context = {}
        for i, _file in enumerate(file_name_list):
            extra_context[_file] = (view['state']['values']
                                    [f'extra_context_{i}'][f'extra_context_{i}']
                                    ['value'])
        extra_separators = (view['state']['values']
                            ['extra_separators']['extra_separators']
                            ['value'])
        if extra_separators:
            extra_separators = extra_separators.split(';')
        else:
            extra_separators = []
        selected_option = (view['state']['values']
                           ['radio_buttons']['unused_action']
                           ['selected_option']['value'])

        if selected_option == 'to_channel':
            bot.app.logger.info('Files uploaded to channel')
            pass

        else:
            bot.app.logger.info('Creating a QA thread')
                 
            texts = process_uploaded_files(files, bot_token=bot.bot_token,
                                           chunk_size=bot.chunk_size,
                                           chunk_overlap=bot.chunk_overlap,
                                           extra_separators=extra_separators)
            # remove temp files dict
            bot.store_files_dict(msg_timestamp, None)

            thread = threading.Thread(target=bot.define_retriever_db,
                                      args=(channel_id, texts, file_name_list, 
                                            msg_timestamp, extra_context))
            thread.start()

    @bot.app.action("unused_action")
    async def handle_unused(ack: Ack):
        """Just to pass unused_action"""
        await ack()
        return
    
    return {"handle_ask": handle_ask,
            "handle_modify_bot": handle_modify_bot,
            "handle_modify_bot_view": handle_modify_bot_view,
            "handle_bot_info" : handle_bot_info,
            "handle_permissions" : handle_permissions,
            "handle_modify_permissions_view" : handle_modify_permissions_view,
            #"handle_mention" : handle_mention
            }

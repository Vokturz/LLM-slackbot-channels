import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.llms import (OpenAI, CTransformers, FakeListLLM)
from langchain.embeddings import (HuggingFaceEmbeddings, FakeEmbeddings)
import os
import json
import re

current_directory = os.path.dirname(os.path.abspath(__file__))
llm_info_file = f'{current_directory}/../data/channels_llm_info.json'

class SlackBot:
    def __init__(self, name='LLM-QA',
                 default_personality="an AI assistant inside a Slack channel",
                 default_instructions="Give helpful and concise answers to the"
                                      " user's questions. Answers in no more"
                                      " than 40 words. You must format your"
                                      " messages in Slack markdown.",
                 default_temp=0.8, verbose=False) -> None:
        """
        Initialize a new SlackBot instance.
        
        Raises:
            ValueError: If the SLACK_BOT_TOKEN or SLACK_APP_TOKEN 
                        environment variables could not be found.
        """
        self.name = name
        self.verbose = verbose
        try:
            self.bot_token = os.environ['SLACK_BOT_TOKEN']
            self.app_token = os.environ['SLACK_APP_TOKEN']
        except KeyError as e:
            raise ValueError("Could not find required environment variable") from e

        self.app =  App(token=self.bot_token, name=self.name)
        self.socket = SocketModeHandler(self.app, self.app_token)

        # This could be a class
        default_llm_info = dict(personality=default_personality,
                                instructions=default_instructions,
                                temperature=default_temp)
        self.default_llm_info = default_llm_info
        
        # This could be loaded using pydantic
        if os.path.exists(llm_info_file):
            with open(llm_info_file, 'r') as f:
                channels_llm_info = json.load(f)
                self.channels_llm_info = channels_llm_info
        else:
            self.channels_llm_info = {}

    def get_bot_user_id(self):
        return self.app.client.auth_test()['user_id']

    def get_app(self) -> App:
        return self.app
    
    def get_bot_token(self) -> str:
        return self.bot_token
    
    def get_verbose(self):
        return self.verbose
    
    def initilize_llm(self, model_type, handler=None,
                       config=dict(temperature=0.8,
                                    max_tokens=300)):
        model_type = model_type.lower()
        if model_type == 'fakellm':
            self.llm = FakeListLLM(responses = lambda x : x)

        elif model_type != 'openai':
            try:
                model_path = os.environ['CTRANSFORMERS_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e

            self.llm = CTransformers(model=model_path, model_type=model_type,
                             callbacks=[handler], config=config)
            
        else:
            self.llm = OpenAI(callbacks=[handler], **config)

    def initialize_embeddings(self,model_type):
        model_type = model_type.lower()
        if model_type == 'fakellm':
            self.embeddings = FakeEmbeddings(size=768)
            self.embeddings_clf = FakeEmbeddings(size=768)
        else:
            try:
                emb_model = os.environ['EMB_MODEL']
                emb_clf_model = os.environ['EMB_CLF_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e
            
            self.embeddings = HuggingFaceEmbeddings(model_name=emb_model)
            self.embeddings_clf = HuggingFaceEmbeddings(model_name=emb_clf_model)

    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return (self.embeddings, self.embeddings_clf)
    
    def get_temperature(self):
        if 'model_type' in self.llm.__dict__: # CTransformers
            return self.llm.client.config.temperature
        else: # OpenAI
            return self.llm.temperature
        
    def change_temperature(self, temperature):
        if 'model_type' in self.llm.__dict__: # CTransformers
            self.llm.client.config.temperature = temperature
        else: # OpenAI
            self.llm.temperature = temperature 
        

    def define_channel_llm_info(self, channel_id, channel_bot_info):
        self.channels_llm_info[channel_id] = channel_bot_info
        with open(llm_info_file, 'w') as f:
                json.dump(self.channels_llm_info, f)

    def get_channel_llm_info(self, channel_id):
        if channel_id not in self.channels_llm_info.keys():
            self.define_channel_llm_info(channel_id, self.default_llm_info) 
        return self.channels_llm_info[channel_id]

            
    def extract_thread_conversation(self, channel_id, thread_ts):
        client = self.app.client
        bot_user_id = self.get_bot_user_id()
        result = client.conversations_replies(channel=channel_id, ts=thread_ts)
        messages = result['messages']
        actual_user = ''
        users = set()
        messages_history = []
        for msg in messages:
            user = msg['user']
            text = msg['text'].replace(f'<@{bot_user_id}>', 'AI')
            if self.verbose:
                text = re.sub(r'\(_time: .*?\)', '', text)
            if actual_user != user:
                if user == bot_user_id:
                    messages_history.append(f'AI: {text}')
                else:
                    users.add(f'<@{user}>') # if was not added
                    messages_history.append(f'<@{user}>: {text}')
                actual_user = user
            else: # Is the same user talking
                messages_history[-1] += f'\n{text}'
        return messages_history, users
    
    def start(self) -> None:
        """
        Start the SlackBot instance.

        Raises:
            RuntimeError: If the SocketModeHandler could not start.
        """
        try:
            self.socket.start()
        except Exception as e:
            raise RuntimeError("Could not start the SocketModeHandler") from e

    # extend Slack Bolt decorators
    def event(self, event_name):
        def decorator(handler):
            self.app.event(event_name)(handler)
            return handler
        return decorator

    def command(self, command_name):
        def decorator(handler):
            self.app.command(command_name)(handler)
            return handler
        return decorator

    def shortcut(self, shortcut_name):
        def decorator(handler):
            self.app.shortcut(shortcut_name)(handler)
            return handler
        return decorator

    def action(self, action_name):
        def decorator(handler):
            self.app.action(action_name)(handler)
            return handler
        return decorator

    def view(self, callback_id):
        def decorator(handler):
            self.app.view(callback_id)(handler)
            return handler
        return decorator

    def options(self, callback_id):
        def decorator(handler):
            self.app.options(callback_id)(handler)
            return handler
        return decorator

    def step(self, callback_id):
        def decorator(handler):
            self.app.step(callback_id)(handler)
            return handler
        return decorator

import logging
import os
import json
import re
from typing import (Callable, Dict, Optional, Union, Tuple, List, Any, Set)
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from langchain.llms import (OpenAI, CTransformers, FakeListLLM)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceEmbeddings, FakeEmbeddings)
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM

current_directory = os.path.dirname(os.path.abspath(__file__))
llm_info_file = f'{current_directory}/../data/channels_llm_info.json'

class SlackBot:
    def __init__(self, name: str='SlackBot',
                 default_personality: str="an AI assistant inside a Slack channel",
                 default_instructions: str="Give helpful and concise answers to the"
                                           " user's questions. Answers in no more"
                                           " than 40 words. You must format your"
                                           " messages in Slack markdown.",
                 default_temp: float=0.8, verbose: bool=False) -> None:
        """
        Initialize a new SlackBot instance.
        
        Raises:
            ValueError: If the SLACK_BOT_TOKEN or SLACK_APP_TOKEN 
                        environment variables could not be found.
        """
        self._name = name

        # Setting Logger
        self._verbose = verbose
        level = logging.INFO #logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger(__name__)
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._default_temp = default_temp
        try:
            self._bot_token = os.environ['SLACK_BOT_TOKEN']
            self._app_token = os.environ['SLACK_APP_TOKEN']
        except KeyError as e:
            raise ValueError("Could not find required environment variable") from e

        self._app =  AsyncApp(logger=logger, token=self._bot_token, name=self._name)

        # This could be a class
        default_llm_info = dict(personality=default_personality,
                                instructions=default_instructions,
                                temperature=default_temp)
        self._default_llm_info = default_llm_info
        
        # This could be loaded using pydantic
        if os.path.exists(llm_info_file):
            with open(llm_info_file, 'r') as f:
                channels_llm_info = json.load(f)
                self._channels_llm_info = channels_llm_info
        else:
            self._channels_llm_info = {}

    def initialize_llm(self, model_type: str,
                       max_tokens_threads: int = 2000,
                       handler: Optional[Callable] = None,
                       config: Dict[str, Union[float, int]] = None) -> None:
        """
        Initializes a language model based on the given `model_type`, `handler`, and `config`.
        """

        # Max number of tokens to pass in a channel thread
        self._max_tokens_threads = max_tokens_threads

        if not config:
            config = {'temperature': 0.8, 'max_tokens': 300}
        model_type = model_type.lower()
        self._model_type = model_type
        if model_type == 'fakellm':
            responses = [f'foo{i}' for i in range(1000)]
            self._llm = FakeListLLM(responses = responses)

        elif model_type != 'openai':
            try:
                model_path = os.environ['CTRANSFORMERS_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e

            self._llm = CTransformers(model=model_path, model_type=model_type,
                             callbacks=[handler], config=config)
            
        else:
            # if (config["model_name"].startswith("gpt-3.5")
            #     or config["model_name"].startswith("gpt-4")):
            #     self._llm = ChatOpenAI(callbacks=[handler], **config)
            # else:
            #     self._llm = OpenAI(callbacks=[handler], **config)
            self._llm = OpenAI(callbacks=[handler], **config)
            
    def initialize_embeddings(self, model_type: str) -> None:
        """
        Initializes embeddings based on model type.
        """
        model_type = model_type.lower()
        if model_type == 'fakellm':
            self._embeddings = FakeEmbeddings(size=768)
        else:
            try:
                emb_model = os.environ['EMB_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e
            
            self._embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    async def start(self) -> None:
        """
        Start the SlackBot instance.
        """
        response = await self.app.client.auth_test()
        self._bot_user_id = response["user_id"]
        await AsyncSocketModeHandler(self._app, self._app_token).start_async()

    async def generate_response(self, query: str) -> str:
        """"
        Asynchronously generates a response to a given query
        """
        llm = self._llm
        resp = await llm.agenerate([query])
        return resp.generations[0][0].text
    
    @property
    def app(self) -> AsyncApp:
        return self._app
    
    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @property
    def bot_user_id(self) -> str:
        return self._bot_user_id
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def max_tokens_threads(self):
        return self._max_tokens_threads
    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings
        #return (self.embeddings, self.embeddings_clf)
    
    def get_temperature(self) -> float:
        """
        Get the temperature used in the language model.
        """
        if 'model_type' in self._llm.__dict__: # CTransformers
            return self._llm.client.config.temperature
        else: 
            try: # OpenAI
                return self._llm.temperature
            except: # FakeLLM
                return self._default_temp
        
    def change_temperature(self, temperature: float) -> None :
        """
        Update the temperature used in the language model.
        """
        if 'model_type' in self._llm.__dict__: # CTransformers
            self._llm.client.config.temperature = temperature
        else:
            try: # OpenAI
                self._llm.temperature = temperature 
            except: # FakeLLM
                pass

    def define_channel_llm_info(self, channel_id: str,
                                channel_bot_info: Dict[str, Union[str, float]]
                                ) -> None:
        """
        Defines the LLM info for a given channel
        """
        self._channels_llm_info[channel_id] = channel_bot_info
        with open(llm_info_file, 'w') as f:
                json.dump(self._channels_llm_info, f, ensure_ascii=False, indent=4)

    def get_channel_llm_info(self, channel_id: str) -> Dict[str, Union[str, float]]:
        """
        Get the LLM info for a given channel
        """
        if channel_id not in self._channels_llm_info.keys():
            self.define_channel_llm_info(channel_id, self._default_llm_info) 
        return self._channels_llm_info[channel_id]
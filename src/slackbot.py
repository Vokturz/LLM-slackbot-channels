import logging
import os
import json
from typing import (Callable, Dict, Optional, Union, Tuple, List, Any, Set)
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from langchain.llms import (OpenAI, CTransformers, FakeListLLM)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceEmbeddings, FakeEmbeddings, OpenAIEmbeddings)
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
import glob
from chromadb.config import Settings

VStore = Chroma

current_directory = os.path.dirname(os.path.abspath(__file__))
llm_info_file = f'{current_directory}/../data/channels_llm_info.json'
permissions_file = f'{current_directory}/../data/permissions.json'
db_dir = f'{current_directory}/../data/db'

class SlackBot:
    def __init__(self, name: str='SlackBot',
                 default_personality: str="an AI assistant inside a Slack channel",
                 default_instructions: str="Give helpful and concise answers to the"
                                           " user's questions. Answers in no more"
                                           " than 40 words. You must format your"
                                           " messages in Slack markdown.",
                 default_temp: float=0.8, chunk_size=500, chunk_overlap=50,
                 k_similarity=5, verbose: bool=False) -> None:
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

        # for Retriever
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._k_similarity = k_similarity

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
        logger.info("Loading Channels information..")
        if os.path.exists(llm_info_file):
            with open(llm_info_file, 'r') as f:
                channels_llm_info = json.load(f)
                self._channels_llm_info = channels_llm_info
        else:
            self._channels_llm_info = {}

        logger.info("Loading Thread Retrievers locations..")
        thread_retriever_db = {}
        for channel_dir in glob.glob(db_dir + "/[CG]*"):
            channel_id = channel_dir.split("/")[-1]
            thread_retriever_db[channel_id] = {}
            for ts_dir in glob.glob(channel_dir + "/*"):
                ts_val = ts_dir.split("/")[-1]
                thread_retriever_db[channel_id][ts_val] = ts_dir
        self._thread_retriever_db = thread_retriever_db

        try:
            psswd = os.environ["PERMISSIONS_PASSWORD"]
        except:
            psswd = ""

        if psswd == "CHANGEME":
            logger.warning(f"You should change the password \"{psswd}\"")
        if psswd == "":
            logger.warning(f"No password provided! Command /permissions will not work")

        # This could be loaded using pydantic
        logger.info("Loading Permissions information..")
        if os.path.exists(permissions_file):
            with open(permissions_file, 'r') as f:
                allowed_users = json.load(f)
                self._allowed_users = allowed_users
        else:
            self._allowed_users = {"users" : ["@all"]}

            
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
            responses = [f'foo' for i in range(1000)]
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
        elif model_type == 'openai':
            self._embeddings = OpenAIEmbeddings()
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
    def chunk_size(self):
        return self._chunk_size
    
    @property
    def chunk_overlap(self):
        return self._chunk_overlap

    @property
    def k_similarity(self):
        return self._k_similarity
    
    @property
    def app(self) -> AsyncApp:
        return self._app
    
    @property
    def bot_token(self) -> str:
        return self._bot_token
    
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
    def allowed_users(self) -> Dict:
        return self._allowed_users
    
    @property
    def max_tokens_threads(self) -> int:
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
        if self._verbose:
            self._app.logger.info(f"LLM Temperature changed to {temperature}")

    def define_thread_retriever_db(self, channel_id: str,
                                     ts: float, docs : List[Document]
                                     ) -> None:
        """
        Defines the thread retriever docs for a given channel.
        """
        if channel_id not in self._thread_retriever_db:
            self._thread_retriever_db[channel_id] = {}
            if not os.path.exists(f"{db_dir}/{channel_id}"):
                os.mkdir(f"{db_dir}/{channel_id}")
        persist_directory = f"{db_dir}/{channel_id}/{ts}"
        if ts not in self._thread_retriever_db[channel_id]:
            os.mkdir(persist_directory)
            
            db = VStore.from_documents(docs, embedding=self._embeddings,
                                       persist_directory=persist_directory,
                                       client_settings=Settings(
                                            chroma_db_impl='duckdb+parquet',
                                            persist_directory=persist_directory,
                                            anonymized_telemetry=False
                                            )
                                        )
            
            db.persist()
            db = None
            self._thread_retriever_db[channel_id][ts] = persist_directory
            self._app.logger.info(f"Created DB for channel's thread {channel_id}/{ts}")


    def get_thread_retriever_db_path(self, channel_id: str, ts: float
                                  ) -> VectorStore:
        """
        Retuns the thread retriever docs for a given channel.
        """
        return self._thread_retriever_db[channel_id][ts]
    
    def define_channel_llm_info(self, channel_id: str,
                                channel_bot_info: Dict[str, Union[str, float]]
                                ) -> None:
        """
        Defines the LLM info for a given channel
        """
        self._channels_llm_info[channel_id] = channel_bot_info
        self._app.logger.info(f"Defined Channel {channel_id} info")
        with open(llm_info_file, 'w') as f:
                json.dump(self._channels_llm_info, f, ensure_ascii=False, indent=4)

    def get_channel_llm_info(self, channel_id: str
                             ) -> Dict[str, Union[str, float]]:
        """
        Get the LLM info for a given channel
        """
        if channel_id not in self._channels_llm_info:
            self.define_channel_llm_info(channel_id, self._default_llm_info) 
        return self._channels_llm_info[channel_id]
    
    def define_allowed_users(self, users_list: List) -> None:
        """
        Define the allowed users fot the bot
        """
        self._allowed_users["users"] = users_list
        self._app.logger.info(f"Defined allowed users: {users_list}")
        with open(permissions_file, 'w') as f:
                json.dump(self._allowed_users, f, ensure_ascii=False, indent=4)

    
    def check_permissions(self, func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator that checks if the user has permission to use the command.
        """
        
        import functools
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            body = kwargs["body"] if "body" in kwargs else kwargs["command"]
            respond = kwargs["respond"] if "respond" in kwargs else None
            ack = kwargs["ack"] if "ack" in kwargs else None
            if "event" in body:
                user = body["event"]["user"]
            else:
                user = body["user_id"]

            if (user in self._allowed_users["users"] or
                "@all" in self._allowed_users["users"] ):
                return await func(*args, **kwargs)
            else:
                if ack:
                    await ack()
                if respond:
                    await respond(text=":x: You don't have permissions to use this command")
                return
        return wrapper

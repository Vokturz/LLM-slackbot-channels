import logging
import os
import json
import asyncio
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
from langchain.tools import BaseTool
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
                 k_similarity=5, verbose: bool=False,
                 log_filename: Optional[str]=None,
                 tools: List[Optional[str]]=[]) -> None:
        """
        Initialize a new SlackBot instance.
        
        Args:
            name: The name of the bot.
            default_personality: The default personality of the bot.
            default_instructions: The default instructions of the bot.
            default_temp: The default temperature of the bot.
            chunk_size: The chunk size for the text splitter.
            chunk_overlap: The chunk overlap for the text splitter.
            k_similarity: The number of chunks to return using the retriever.
            verbose: Whether or not to print debug messages.
            log_filename: The name of the log file.

        Raises:
            ValueError: If the SLACK_BOT_TOKEN or SLACK_APP_TOKEN 
                        environment variables could not be found.
        """
        self._name = name

        # Setting Logger
        logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self._verbose = verbose
        level = logging.INFO #logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger(__name__)
        filename = log_filename
        logging.basicConfig(filename=filename, level=level,
                            format=logger_format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(logger_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._default_temp = default_temp

        # for retriever and text splitter
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
                                temperature=default_temp,
                                tool_names=[],
                                as_agent=False)
        self._default_llm_info = default_llm_info
        
        # This could be loaded using pydantic
        logger.info("Loading Channels information..")
        if os.path.exists(llm_info_file):
            with open(llm_info_file, 'r') as f:
                channels_llm_info = json.load(f)
            for channel_id in channels_llm_info:
                # update with the new info
                if 'tool_names' not in channels_llm_info[channel_id]:
                    channels_llm_info[channel_id]['tool_names'] = []
                if 'as_agent' not in channels_llm_info[channel_id]:
                    channels_llm_info[channel_id]['as_agent'] = False

            self._channels_llm_info = channels_llm_info

        else:
            self._channels_llm_info = {}

        self._tools = tools
        self._tool_names = [tool.name for tool in tools]
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
        
        self._stored_files = {}

            
    def initialize_llm(self, model_type: str,
                       max_tokens_threads: int = 2000,
                       config: Dict[str, Union[float, int]] = None,
                       **kwargs) -> None:
        """
        Initializes a language model based on the given `model_type` and `config`.

        Args:
            model_type: The type of language model to use, can be fakellm, openai
                        and ctransformers.
            max_tokens_threads: The maximum number of tokens to consider in the
                                history of a channel thread.
            config: The configuration for the language model.
            kwargs: Additional keyword arguments for the language model.

        Raises:
            ValueError: If model_type is ctransformers and CTRANSFORMERS_MODEL
                        environment variable could not be found
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
                             config=config, **kwargs)
            
        else:
            if (config["model_name"].startswith("gpt-3.5")
                or config["model_name"].startswith("gpt-4")):
                self._llm = ChatOpenAI(**config, **kwargs)
            else:
                self._llm = OpenAI(**config, **kwargs)
            # self._llm = OpenAI(callbacks=[handler], **config)
            
    def initialize_embeddings(self, model_type: str, **kwargs) -> None:
        """
        Initializes embeddings based on model type.

        Args:
            model_type: The type of embeddings to use.
            kwargs: Additional keyword arguments for OpenAI embeddings.

        Raises:
            ValueError: If model_type is ctransformers and EMBD_MODEL
                        environment variable could not be found
        """
        model_type = model_type.lower()
        if model_type == 'fakellm':
            self._embeddings = FakeEmbeddings(size=768)
        elif model_type == 'openai':
            self._embeddings = OpenAIEmbeddings(**kwargs)
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
        
    @property
    def tool_names(self) -> List[str]:
        return self._tool_names

    def get_tools_by_names(self, tool_names: List[str]) -> List[BaseTool]:
        tools = [tool for tool in self._tools 
                 if tool.name in tool_names]
        return tools
    def get_temperature(self) -> float:
        """
        Get the temperature used in the language model.

        Returns:
            temp: The temperature used in the language model
        """
        if 'model_type' in self._llm.__dict__: # CTransformers
            temperature = self._llm.client.config.temperature
        else: 
            try: # OpenAI
                temperature = self._llm.temperature
            except: # FakeLLM
                temperature = self._default_temp
        return temperature
        
    def change_temperature(self, new_temperature: float) -> None :
        """
        Update the temperature used in the language model.

        Args:
            new_temperature: The new temperature to use.
        """
        if 'model_type' in self._llm.__dict__: # CTransformers
            self._llm.client.config.temperature = new_temperature
        else:
            try: # OpenAI
                self._llm.temperature = new_temperature 
            except: # FakeLLM
                pass
        if self._verbose:
            self._app.logger.info(f"LLM Temperature changed to {new_temperature}")

    def define_thread_retriever_db(self, channel_id: str,
                                     ts: float, docs : List[Document],
                                     file_name_list: List[str],
                                     extra_context: str
                                     ) -> None:
        """
        Defines the thread retriever docs for a given channel.

        Args:
            channel_id: The id of the channel.
            ts: The timestamp of the thread.
            docs: The documents to store in the vector database.
            file_name_list: The file names of the documents.
            extra_context: Any extra context to add to the QA thread
        """
        if channel_id not in self._thread_retriever_db:
            self._thread_retriever_db[channel_id] = {}
            if not os.path.exists(f"{db_dir}/{channel_id}"):
                os.mkdir(f"{db_dir}/{channel_id}")
        persist_directory = f"{db_dir}/{channel_id}/{ts}"
        if ts not in self._thread_retriever_db[channel_id]:
            os.mkdir(persist_directory)
            msg = "Creating DB for channel's thread.."
            init_msg = asyncio.run(self._app.client.chat_postMessage(channel=channel_id,
                                                    thread_ts=ts,
                                                    text=msg))
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
            msg = f"_This is a QA Thread using files `{'` `'.join(file_name_list)}`_."
            msg += f" The files are about {extra_context}"
            asyncio.run(self._app.client.chat_update(channel=channel_id,
                                                     ts=init_msg['ts'],
                                                     text=msg))

    def get_thread_retriever_db_path(self, channel_id: str, ts: float
                                  ) -> VectorStore:
        """
        Retuns the persisted directory of the vector database for a given
        channel and thread.
        
        Args:
            channel_id: The id of the channel.
            ts: The timestamp of the thread.

        Returns:
            persist_directory: The persisted directory of the vector database
                               corresponding to the channel and thread.
        """
        persist_directory = self._thread_retriever_db[channel_id][ts]
        return persist_directory
    
    def define_channel_llm_info(self, channel_id: str,
                                channel_bot_info: Dict[str, Union[str, float]]
                                ) -> None:
        """
        Defines the LLM info for a given channel

        Args:
            channel_id: The id of the channel.
            channel_bot_info: The LLM info for the channel as a dictionary with
                              the following keys: 
                                - personality: The personality of the bot.
                                - instructions: The instructions for the bot.
                                - temperature: The temperature used in the
                                                language model.
        """
        self._channels_llm_info[channel_id] = channel_bot_info
        self._app.logger.info(f"Defined Channel {channel_id} info")
        with open(llm_info_file, 'w') as f:
                json.dump(self._channels_llm_info, f, ensure_ascii=False, indent=4)

    def get_channel_llm_info(self, channel_id: str
                             ) -> Dict[str, Union[str, float]]:
        """
        Get the LLM info for a given channel

        Args:
            channel_id: The id of the channel.

        Returns:
            channel_llm_info: The LLM info for the channel as a dictionary with
                              the following keys: 
                                - personality: The personality of the bot.
                                - instructions: The instructions for the bot.
                                - temperature: The temperature used in the
                                                language model.
        """

        # if is not defined
        if channel_id not in self._channels_llm_info:
            self.define_channel_llm_info(channel_id, self._default_llm_info) 

        channel_llm_info = self._channels_llm_info[channel_id]
        return channel_llm_info
    
    def define_allowed_users(self, users_list: List[str]) -> None:
        """
        Define the list of allowed users who can use the bot.

        Args:
            users_list: A list of slack users.
        """
        self._allowed_users["users"] = users_list
        self._app.logger.info(f"Defined allowed users: {users_list}")
        with open(permissions_file, 'w') as f:
                json.dump(self._allowed_users, f, ensure_ascii=False, indent=4)


    def get_stored_files_dict(self, timestamp: float) -> Dict[str, Any]:
        """
        Get the stored file dictionary for a given timestamp.

        Args:
            timestamp: The timestamp of when the file was uploaded.

        Returns
            files_dict: The file dictionary
        """
        files_dict = self._stored_files[timestamp]
        return files_dict

    def store_files_dict(self, timestamp: float, files_dict: Dict[str, Any]) -> None:
        """
        Store a files dictionary from Slack.

        Args:
            timestamp: The timestamp of when the file was uploaded.
            files_dict: The files dictionary to store
        """
        self._stored_files[timestamp] = files_dict
    
    def check_permissions(self, func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator that checks if the user has permission to use the command.

        Args:
            func: The function to be decorated.

        Returns:
            wrapper: The wrapper function
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

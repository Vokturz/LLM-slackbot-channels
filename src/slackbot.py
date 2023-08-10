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
import shutil
from chromadb.config import Settings
from .ingest import does_vectorstore_exist

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
                 default_temp: float=0.8, max_tokens: int=500,
                 model_type='fakellm', chunk_size=500,
                 chunk_overlap=50,
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
            max_tokens: The maximum number of tokens for the LLM.
            model_type: The default model type to use for the LLM.
            chunk_size: The chunk size for the text splitter.
            chunk_overlap: The chunk overlap for the text splitter.
            k_similarity: The number of chunks to return using the retriever.
            verbose: Whether or not to print debug messages.
            log_filename: The name of the log file.

        Raises:
            ValueError: If the SLACK_BOT_TOKEN or SLACK_APP_TOKEN 
                        environment variables could not be found.
        """
        self.name = name
        self.model_type = model_type.lower()

        # Setting Logger
        logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.verbose = verbose
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

        self.default_temp = default_temp
        self.max_tokens = max_tokens
        # for retriever and text splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_similarity = k_similarity

        try:
            self.bot_token = os.environ['SLACK_BOT_TOKEN']
            self.app_token = os.environ['SLACK_APP_TOKEN']
        except KeyError as e:
            raise ValueError("Could not find required environment variable") from e

        self.app =  AsyncApp(logger=logger, token=self.bot_token, name=self.name)

        # This could be a class
        default_llm_info = dict(personality=default_personality,
                                instructions=default_instructions,
                                temperature=default_temp,
                                tool_names=[],
                                as_agent=False)
        self.default_llm_info = default_llm_info
        
        # This could be loaded using pydantic
        logger.info("Loading Channels information..")
        self.channels_llm_info = {}
        if os.path.exists(llm_info_file):
            with open(llm_info_file, 'r') as f:
                channels_llm_info = json.load(f)
            for channel_id in channels_llm_info:
                # update with the new info
                if 'tool_names' not in channels_llm_info[channel_id]:
                    channels_llm_info[channel_id]['tool_names'] = []
                if 'as_agent' not in channels_llm_info[channel_id]:
                    channels_llm_info[channel_id]['as_agent'] = False
                db_path = f"{db_dir}/{channel_id}/channel_db"
                if not does_vectorstore_exist(db_path): # No db exists
                    self.delete_vectorstore(channel_id)
                    if 'files' in channels_llm_info[channel_id]:
                        del channels_llm_info[channel_id]['files']
                self.define_channel_llm_info(channel_id,
                                             channels_llm_info[channel_id],
                                             verbose=False)
            self.channels_llm_info = channels_llm_info

        self.tools = tools
        self.tool_names = [tool.name for tool in tools]

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
                self.allowed_users = allowed_users
        else:
            self.allowed_users = {"users" : ["@all"]}
        
        self.stored_files = {}

            
    def initialize_llm(self, model_type: Optional[str] = None,
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
        if model_type is None:
            model_type = self.model_type

        # Max number of tokens to pass in a channel thread
        self.max_tokens_threads = max_tokens_threads

        if not config:
            config = {'temperature': 0.8, 'max_tokens': 300}

        self.llm_stream = False # by default
        if 'stream' in config:
            self.llm_stream = config['stream']
        elif 'streaming' in config:
            self.llm_stream = config['streaming']

        if model_type == 'fakellm':
            responses = [f'foo' for i in range(1000)]
            self.llm = FakeListLLM(responses = responses)

        elif model_type != 'openai':
            try:
                model_path = os.environ['CTRANSFORMERS_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e

            self.llm = CTransformers(model=model_path, model_type=model_type,
                             config=config, **kwargs)
            
        else:
            if config["model_name"].startswith("gpt"):
                self.llm = ChatOpenAI(**config, **kwargs)
            else:
                self.llm = OpenAI(**config, **kwargs)
            # self.llm = OpenAI(callbacks=[handler], **config)
            
            self.default_openai_model = config["model_name"]
            for channel_id in self.channels_llm_info:
                if 'openai_model' not in self.channels_llm_info[channel_id]:
                        self.channels_llm_info[channel_id]['openai_model'] = config["model_name"]
            
    def initialize_embeddings(self, model_type: Optional[str] = None, **kwargs) -> None:
        """
        Initializes embeddings based on model type.

        Args:
            model_type: The type of embeddings to use.
            kwargs: Additional keyword arguments for OpenAI embeddings.

        Raises:
            ValueError: If model_type is ctransformers and EMB_MODEL
                        environment variable could not be found
        """
        if model_type is None:
            model_type = self.model_type
            
        if model_type == 'fakellm':
            self.embeddings = FakeEmbeddings(size=768)
        elif model_type == 'openai':
            self.embeddings = OpenAIEmbeddings(**kwargs)
        else:
            try:
                emb_model = os.environ['EMB_MODEL']
            except KeyError as e:
                raise ValueError("Could not find required environment variable") from e
            
            self.embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    async def start(self) -> None:
        """
        Start the SlackBot instance.
        """
        response = await self.app.client.auth_test()
        self.bot_user_id = response["user_id"]
        await AsyncSocketModeHandler(self.app, self.app_token).start_async()
        

    def define_retriever_db(self, channel_id: str, docs : List[Document],
                            files_name_list: List[str], ts: Optional[str]='',
                            extra_context: Dict[str, str] = None,
                            user_id: Optional[str] = None
                                     ) -> None:
        """
        Defines the thread retriever docs for a given channel.

        Args:
            channel_id: The id of the channel.
            docs: The documents to store in the vector database.
            files_name_list: The file names of the documents.
            ts: The timestamp of the thread.
            extra_context: Any extra context for each document to add to the QA thread
        """
        if not os.path.exists(f"{db_dir}/{channel_id}"):
            os.mkdir(f"{db_dir}/{channel_id}")
        
        
        if ts:
            msg = "Creating DB for channel's thread.."
            init_msg = asyncio.run(self.app.client.chat_postMessage(channel=channel_id,
                                                    thread_ts=ts,
                                                    text=msg))
            persist_directory = f"{db_dir}/{channel_id}/{ts}"
        else:
            msg = "Adding files to channel.."
            init_msg = asyncio.run(self.app.client.chat_postEphemeral(channel=channel_id,
                                                user=user_id,
                                                text=msg))
            persist_directory = f"{db_dir}/{channel_id}/channel_db"
        if not os.path.exists(persist_directory):
            os.mkdir(persist_directory)

        chroma_settings = Settings(is_persistent=True,
                                   persist_directory=persist_directory,
                                   anonymized_telemetry=False)
        if does_vectorstore_exist(persist_directory):
            self.app.logger.info(f"Appending to existing DB at {persist_directory}")
            db = Chroma(embedding_function=self.embeddings,
                        client_settings=chroma_settings)
            db.add_documents(docs)
        else:
            db = Chroma.from_documents(docs, embedding=self.embeddings,
                                    persist_directory=persist_directory,
                                    client_settings=chroma_settings)
        db = None

        if ts:
            msg_extra_context = ', '.join([extra_context[_file] for _file in extra_context])
            self.app.logger.info(f"Created DB for channel's thread {channel_id}/{ts}")
            msg = f"_This is a QA Thread using files `{'` `'.join(files_name_list)}`_."
            msg += f" The files are about {msg_extra_context}"
            asyncio.run(self.app.client.chat_update(channel=channel_id,
                                                ts=init_msg['ts'],
                                                text=msg))
        else:
            msg = "Files added to Channel!"
            init_msg = asyncio.run(self.app.client.chat_postEphemeral(channel=channel_id,
                                                user=user_id,
                                                text=msg, replace_original=True))
            channel_bot_info = self.channels_llm_info[channel_id]
            if 'files' not in channel_bot_info:
                channel_bot_info['files'] = {}
            for _file in files_name_list: 
                channel_bot_info['files'][_file] = extra_context[_file]
            self.define_channel_llm_info(channel_id, channel_bot_info)
   

    def get_retriever_db_path(self, channel_id: str, ts: Optional[float]=''
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

        if ts:
            persist_directory = f"{db_dir}/{channel_id}/{ts}"
        else:
            persist_directory = f"{db_dir}/{channel_id}/channel_db"
        return persist_directory
    
    def define_channel_llm_info(self, channel_id: str,
                                channel_bot_info: Dict[str, Union[str, float]],
                                verbose: bool = True
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
        self.channels_llm_info[channel_id] = channel_bot_info
        if verbose:
            self.app.logger.info(f"Defined Channel {channel_id} info")
        with open(llm_info_file, 'w') as f:
                json.dump(self.channels_llm_info, f, ensure_ascii=False, indent=4)

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
        if channel_id not in self.channels_llm_info:
            self.define_channel_llm_info(channel_id, self.default_llm_info) 

        channel_llm_info = self.channels_llm_info[channel_id]

        if (self.model_type == 'openai' and 
            'openai_model' not in channel_llm_info):
            channel_llm_info['openai_model'] = self.default_openai_model
        return channel_llm_info
    
    def define_allowed_users(self, users_list: List[str]) -> None:
        """
        Sets the list of users who can interact with the bot.
        """
        self.allowed_users["users"] = users_list
        self.app.logger.info(f"Defined allowed users: {users_list}")
        with open(permissions_file, 'w') as f:
                json.dump(self.allowed_users, f, ensure_ascii=False, indent=4)


    def get_stored_files_dict(self, timestamp: float) -> List[Dict[str, Any]]:
        """
        Returns the file dictionary for the provided timestamp, which comes
        from the uploaded files by a Slack user.
        """
        files_dict = self.stored_files[timestamp]
        return files_dict

    def delete_file_from_channel(self, channel_id: str, file_name: str,
                                 timestamp: Optional[str]='') -> None:
        """
        Deletes the file from the channel.
        """
        channel_llm_info = self.get_channel_llm_info(channel_id)
        
        source = 'data/tmp/' + file_name

        if timestamp:
            persist_directory = f"{db_dir}/{channel_id}/{timestamp}"
        else:
            persist_directory = f"{db_dir}/{channel_id}/channel_db"
            
            
        chroma_settings = Settings(is_persistent=True,
                                   persist_directory=persist_directory,
                                   anonymized_telemetry=False)
        db = Chroma(embedding_function=self.embeddings,
                    client_settings=chroma_settings)
        self.app.logger.info(f"Removing {file_name} from {persist_directory} DB")
        db._collection.delete(where={'source': source})
        db = None

        if not timestamp:
            channel_llm_info['files'].pop(file_name)
            self.define_channel_llm_info(channel_id, channel_llm_info)
            if not channel_llm_info['files']:
                del channel_llm_info['files']
                self.delete_vectorstore(channel_id)
                

        
    def delete_vectorstore(self, channel_id: str,
                           timestamp: Optional[str]='') -> None:
        if timestamp:
            persist_directory = f"{db_dir}/{channel_id}/{timestamp}"
        else:
            persist_directory = f"{db_dir}/{channel_id}/channel_db"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)


    def get_llm_by_channel(self, channel_id: str, **kwargs) -> LLM:
        """
        Retrieves the language model configured for a specific channel.
        """
        channel_llm_info = self.get_channel_llm_info(channel_id)
        if self.model_type == 'openai':
            if 'openai_model' not in channel_llm_info:
                model_name = self.default_openai_model 
            else:
                model_name = channel_llm_info['openai_model']
            config = dict(model_name=model_name,
                           temperature=channel_llm_info['temperature'],
                           max_tokens=self.max_tokens)
            if config["model_name"].startswith("gpt"):
                    llm = ChatOpenAI(streaming=self.llm_stream, **config, **kwargs)
            else:
                    llm = OpenAI(streaming=self.llm_stream, **config, **kwargs)
        else:
            llm = self.llm
            try: # CTransformers
                llm.client.config.temperature = channel_llm_info['temperature']
            except: # FakeLLM
                pass 
        return llm
    

    def store_files_dict(self, timestamp: float, files_dict: List[Dict[str, Any]]) -> None:
        """
        Stores the files dictionary against the provided timestamp.
        The files dictionary comes from the uploaded files by a Slack user.
        """
        self.stored_files[timestamp] = files_dict
    
        
    def get_tools_by_names(self, tool_names: List[str]) -> List[BaseTool]:
        """
        Returns a list of tools that match the given names.
        """
        tools = [tool for tool in self.tools 
                 if tool.name in tool_names]
        return tools
    

    def check_permissions(self, func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator that verifies user permission to execute a command
        """
        
        import functools
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            body = kwargs.get("body", kwargs.get("command"))
            respond = kwargs.get("respond")
            ack = kwargs.get("ack")
            user = body["event"]["user"] if "event" in body else body["user_id"]

            if (user in self.allowed_users["users"] or
                "@all" in self.allowed_users["users"] ):
                return await func(*args, **kwargs)

            if ack:
                await ack()
            if respond:
                await respond(text=":x: You don't have permissions to use this command")
        return wrapper

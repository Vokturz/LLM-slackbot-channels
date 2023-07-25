from dotenv import load_dotenv
from src.slackbot import SlackBot
from src.handlers import create_handlers
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
handler = StreamingStdOutCallbackHandler()
# Load environment variables
load_dotenv()

# Load custom tools
import src.custom_tools as custom_tools

tools = [custom_tools.disk_usage, custom_tools.memory_usage,
         custom_tools.asyncArxivQueryRun(max_workers=4),
         custom_tools.asyncDuckDuckGoSearchRun(max_workers=4)]

# You can load more tools using load_tools
# from langchain.agents import load_tools
# tools.extend(load_tools(['ddg-search', 'arxiv', 'requests_all']))

# Create SlackBot instance
bot = SlackBot(name='SlackBot', verbose=True,
               max_tokens=500, model_type='openai',
               chunk_size=500, # Chunk size for splitter
               chunk_overlap=50, # Chunk overlap for splitter
               k_similarity=5, # Numbers of chunks to return in retriever
               log_filename='_slackbot.log',
               tools=tools,
               )

## LLM configuration
if bot.model_type == 'llama':
    config = dict(gpu_layers=40, temperature=0.8, batch_size=1024, 
                context_length=2048, threads=6, stream=True, max_new_tokens=bot.max_tokens)
else:
    config = dict(model_name="gpt-3.5-turbo-16k", temperature=0.8,
                  streaming=True, max_tokens=bot.max_tokens)

# Initialize LLM and embeddings
bot.app.logger.info("Initializing LLM and embeddings...")
bot.initialize_llm(bot.model_type, max_tokens_threads=4000, config=config, callbacks=[handler])
bot.initialize_embeddings(bot.model_type)

# Create handlers for commands /ask, /modify_bot, /bot_info  and bot mentions
create_handlers(bot)

### You can create new handlers for other commands as follow
# @bot.app.command("/foo")
# async def handle_foo(say, respond, ack, command):
#     await ack()
#     # do something..

# Load bot in async mode
async def start():
    await bot.start()

if __name__ == "__main__":
    logger = bot.app.logger
    try:
        asyncio.run(start())
        logger.info('App started.')
    except KeyboardInterrupt:
        logger.info('App stopped by user.')
    except Exception as e:
        logger.info('App stopped due to error.')
        logger.error(str(e))
    finally:
        logger.info('App stopped.')
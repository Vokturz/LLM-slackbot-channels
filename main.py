from dotenv import load_dotenv
from src.slackbot import SlackBot
from src.handlers import create_handlers
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
handler = StreamingStdOutCallbackHandler()
from langchain.agents import load_tools, Tool
# Load environment variables
load_dotenv()

from langchain.tools import tool
import subprocess
@tool
def disk_usage(query: str) -> str:
    """useful for when you need to answer questions about the disk usage."""
    output = subprocess.check_output(['df', '-h'], text=True)
    output = "This is the output of `df -h`:\n" + output
    return output

@tool
def memory_usage(query: str) -> str:
    """useful for when you need to answer questions about memory usage."""
    output = subprocess.check_output(['free', '-h'], text=True)
    output = "This is the output of `free -h`. Mem refers to RAM memory and Swap to swap memory:\n" + output
    return output

tools = [disk_usage, memory_usage]
tools.extend(load_tools(['ddg-search', 'arxiv']))

# Create SlackBot instance
bot = SlackBot(name='SlackBot', verbose=True,
               chunk_size=500, # Chunk size for splitter
               chunk_overlap=50, # Chunk overlap for splitter
               k_similarity=5, # Numbers of chunks to return in retriever
               log_filename='_slackbot.log',
               tools=tools,
               )

## LLM configuration
model_type = 'openai'
if model_type == 'llama':
    config = dict(gpu_layers=40, temperature=0.8, batch_size=1024, 
                context_length=2048, threads=6, stream=True, max_new_tokens=500)
else:
    config = dict(model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=500)

# Initialize LLM and embeddings
bot.app.logger.info("Initializing LLM and embeddings...")
bot.initialize_llm(model_type, max_tokens_threads=4000, config=config, callbacks=[handler])
bot.initialize_embeddings(model_type)

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
        logger.info('App started.')
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info('App stopped by user.')
    except Exception as e:
        logger.info('App stopped due to error.')
        logger.error(str(e))
    finally:
        logger.info('App stopped.')
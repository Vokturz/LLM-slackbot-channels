from dotenv import load_dotenv
from src.slackbot import SlackBot
from src.handlers import create_handlers
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
handler = StreamingStdOutCallbackHandler()

# Run main with Prem https://github.com/premAI-io/prem-app

# Load environment variables
load_dotenv()

# Create SlackBot instance
bot = SlackBot(name='SlackBot', verbose=True,
               chunk_size=500, # Chunk size for splitter
               chunk_overlap=50, # Chunk overlap for splitter
               k_similarity=5 # Numbers of chunks to return in retriever
               )

## LLM configuration
model_type = 'openai'
config = dict(model_name="gpt-3.5-turbo", openai_api_base="http://localhost:8111/api/v1",
              temperature=0.8, max_tokens=500)

# Initialize LLM and embeddings
bot.app.logger.info("Initializing LLM and embeddings...")
bot.initialize_llm(model_type, max_tokens_threads=1000, handler=handler, config=config)
bot.initialize_embeddings(model_type, openai_api_base="http://localhost:8444/api/v1")

# Create handlers for commands /ask, /modify_bot, /bot_info  and bot mentions
create_handlers(bot)

# Load bot in async mode
async def start():
    await bot.start()

if __name__ == "__main__":
    asyncio.run(start())
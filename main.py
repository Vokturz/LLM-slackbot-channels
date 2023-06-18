from dotenv import load_dotenv
from src.slackbot import SlackBot
from src.handlers import create_handlers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import asyncio

load_dotenv()

handler = StreamingStdOutCallbackHandler()
bot = SlackBot(name='LLM-QA', verbose=True)

## LLM configuration
model_type = 'llama'
if model_type == 'llama':
    config = dict(gpu_layers=32, temperature=0.8, batch_size=1024,
                context_length=2048, threads=6, stream=True, max_new_tokens=300)
else:
    config = dict(model_name="text-davinci-003", temperature=0.8, max_tokens=300)
    
bot.initilize_llm(model_type, handler, config=config)
bot.initialize_embeddings(model_type)

create_handlers(bot)

async def start():
    await bot.start()

if __name__ == "__main__":
    asyncio.run(start())
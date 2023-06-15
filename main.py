from dotenv import load_dotenv
from src.slackbot import SlackBot
from src.handlers import create_handlers
load_dotenv()

# Require SLACK_BOT_TOKEN and SLACK_APP_TOKEN env variables
bot = SlackBot()
create_handlers(bot, model_type='GPT4All', show_time=True)


bot.start()

import sys
import os
import unittest
from unittest.mock import AsyncMock, patch

# add src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.slackbot import SlackBot
from src.handlers import create_handlers

class TestSlackBotCommands(unittest.IsolatedAsyncioTestCase):
    @classmethod
    @patch.dict('os.environ',
                {'SLACK_BOT_TOKEN': 'mock-slack-bot-token',
                 'SLACK_APP_TOKEN': 'mock-slack-app-token'})
    def setUpClass(cls):
        cls.bot = SlackBot()
        # FakeLLM only returns "foo"
        cls.bot.initialize_llm(model_type='fakellm')
        cls.bot.initialize_embeddings(model_type='fakellm')
        # Set allowed user
        cls.bot._allowed_users = {'users' : 'U123456'}
        # Set channel bot info
        cls.bot._channels_llm_info = {'C123456' : cls.bot._default_llm_info}
        # Define handlers
        cls.handlers = create_handlers(cls.bot)
    

    async def test_handle_invalid_user(self):
        """ User has no permissions"""
        ack = AsyncMock()
        respond = AsyncMock()
        say = AsyncMock()
        command = {'text': 'Test command',
                   'user_id': 'U999999', # no allowed user
                   'channel_id': 'C123456'}

        await self.handlers['handle_ask'](ack=ack, respond=respond,
                                          say=say, command=command)

        ack.assert_called_once()
        resp = ":x: You don't have permissions to use this command"
        respond.assert_called_once_with(text=resp)
        
    async def test_handle_ask_invalid_channel(self):
        """ Message wasn't sent in a channel """
        ack = AsyncMock()
        respond = AsyncMock()
        say = AsyncMock()
        command = {'text': 'Test command',
                   'user_id': 'U123456',
                   'channel_id': 'U123456'}

        await self.handlers['handle_ask'](ack=ack, respond=respond,
                                          say=say, command=command)

        ack.assert_called_once()
        resp = "This command can only be used in channels."
        respond.assert_called_once_with(text=resp)
        say.assert_not_called()

    async def test_handle_ask_respond_foo(self):
        """ Message handled via respond"""
        ack = AsyncMock()
        respond = AsyncMock()
        say = AsyncMock()
        command = {'text': 'Test command',
                   'user_id': 'U123456',
                   'channel_id': 'C123456'}

        await self.handlers['handle_ask'](ack=ack, respond=respond,
                                          say=say, command=command)

        ack.assert_called_once()
        resp =f"*<@{command['user_id']}> asked*: {command['text']}\n*Answer*:\nfoo"
        respond.assert_called_once_with(text=resp)
        say.assert_not_called()


    async def test_handle_ask_say_foo(self):
        """ Message handled via say and chat_update"""
        ack = AsyncMock()
        respond = AsyncMock()
        say = AsyncMock()
        ts = 1 # timestamp
        say.return_value = {'result': 'success', 'ts': ts}
        self.bot.app.client.chat_update = AsyncMock()
        self.bot.app.client.chat_update.return_value = {'ok': True}
        chat_update = self.bot.app.client.chat_update
        self.bot.app.client.chat_postMessage = AsyncMock()
        self.bot.app.client.chat_postMessage.return_value = {'result': 'success', 'ts': ts}
        chat_postMessage = self.bot.app.client.chat_postMessage
        
        
        # message contains !all
        command = {'text': '!all Test command',
                   'user_id': 'U123456',
                   'channel_id': 'C123456'}

        await self.handlers['handle_ask'](ack=ack, respond=respond,
                                          say=say, command=command)
                                          
        command_no_all = command['text'].replace('!all', '').strip()
        
        ack.assert_called_once()
        # initial message
        init_msg = (f"*<@{command['user_id']}> asked*: {command_no_all}"
                    f"\nSlackBot is thinking.. :hourglass_flowing_sand:")
        chat_postMessage.assert_called_once_with(channel=command['channel_id'],
                                                 text=init_msg, thread_ts=None)
        # final message
        resp =f"*<@{command['user_id']}> asked*: {command_no_all}\n*Answer*:\nfoo"
        chat_update.assert_called_once_with(channel=command['channel_id'], ts=ts, text=resp)
        respond.assert_not_called()

if __name__ == '__main__':
    unittest.main()

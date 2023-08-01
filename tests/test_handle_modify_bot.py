import sys
import os
import json
import unittest
from unittest.mock import AsyncMock, patch

# add src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.slackbot import SlackBot
from src.handlers import create_handlers


class TestModifyBotCommand(unittest.IsolatedAsyncioTestCase):
    @classmethod
    @patch.dict('os.environ',
                {'SLACK_BOT_TOKEN': 'mock-slack-bot-token',
                 'SLACK_APP_TOKEN': 'mock-slack-app-token'})
    def setUpClass(cls):
        cls.bot = SlackBot(model_type='fakellm')
        cls.bot.initialize_llm(model_type='fakellm')
        cls.bot.allowed_users = {'users' : 'U123456'}
        cls.bot.channels_llm_info = {'C123456' : cls.bot.default_llm_info}
        cls.handlers = create_handlers(cls.bot)

    async def test_modify_bot_invalid_channel(self):
        """ Command wasn't sent in a channel """
        ack = AsyncMock()
        respond = AsyncMock()
        body = {'user_id': 'U123456',
                'channel_id': 'U123456',
                'trigger_id': 'some-trigger-id',
                'text': ''}

        await self.handlers['handle_modify_bot'](ack=ack, body=body, respond=respond)

        ack.assert_called_once()
        respond.assert_called_once_with(text='This command can only be used in channels.')

    async def test_modify_bot_valid_channel(self):
        """ Modal view successfully created """
        ack = AsyncMock()
        respond = AsyncMock()
        channel_id = 'C123456'
        body = {'user_id': 'U123456',
                'channel_id': channel_id,
                'trigger_id': 'some-trigger-id',
                'text' : '!no-notify'}
        
        # Set up the expected channel LLM info
        channel_llm_info = self.bot.get_channel_llm_info(channel_id)

        with open(f'{current_dir}/../src/payloads/modify_bot_template.json', 'r') as f:
            view = json.load(f)
        expected_view = view.copy()
        expected_view["blocks"][0]["element"]["initial_value"] = channel_llm_info['personality']
        expected_view["blocks"][1]["element"]["initial_value"] = channel_llm_info['instructions']
        expected_view["blocks"][2]["element"]["initial_value"] = str(channel_llm_info['temperature'])
        expected_view["blocks"][3] = { "type": "section", "text": { "type": "plain_text", "text": " "}}
        expected_view["blocks"][4]["element"]["initial_option"]["value"] = "as_llm_chain"
        expected_view["blocks"][4]["element"]["initial_option"]["text"]["text"] = "Use it as a LLM chain" 
        expected_view["blocks"][5] = { "type": "section",  "text": { "type": "plain_text", "text": " "}}

        extra_data = {"channel_id": channel_id, "notify" : False}
        expected_view["private_metadata"] =  json.dumps(extra_data)

        self.bot.app.client.views_open = AsyncMock()

        await self.handlers['handle_modify_bot'](ack=ack, body=body, respond=respond)

        ack.assert_called_once()
        self.bot.app.client.views_open.assert_called_once_with(trigger_id=body['trigger_id'], view=expected_view)

    async def test_modify_bot_valid_channel_view(self):
        """ Change channel information through modal view """
        channel_id = 'C123456'
        
        # Set up the expected channel LLM info
        channel_llm_info = self.bot.get_channel_llm_info(channel_id)

        new_llm_info = {'personality' : channel_llm_info['personality'],
                        'instructions' : channel_llm_info['instructions'],
                        'temperature' : 0.5,
                        'as_agent' : channel_llm_info['as_agent'],
                        'tool_names': []}
        expected_view = {'state': {
            'values': {
                'personality': {'personality': {'value' : new_llm_info['personality']}},
                'instructions': {'instructions': {'value' : new_llm_info['instructions']}},
                'temperature': {'temperature': {'value' : new_llm_info['temperature']}},
                'use_it_as': {'unused_action': {'selected_option': {'value': new_llm_info['as_agent']}}}
            }
        }}

        extra_data = {"channel_id": channel_id, "notify" : True}
        expected_view['private_metadata'] = json.dumps(extra_data)
        ack = AsyncMock()
        say = AsyncMock()
        body = {'user': {'id': 'U123456'}}
        await self.handlers['handle_modify_bot_view'](ack=ack, body=body, say=say, view=expected_view)
        say.assert_called_once_with(f'_*<@{body["user"]["id"]}> has modified the bot info*_', channel='C123456')
        assert self.bot.get_channel_llm_info(channel_id) == new_llm_info, "Channel info does not match"
        ack.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)

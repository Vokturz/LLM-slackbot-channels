from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from .slackbot import SlackBot
from typing import Any, Dict, List
from langchain.schema import LLMResult
from .simplethrottle import SimpleThrottle
import asyncio
import nest_asyncio
nest_asyncio.apply()

class SlackAsyncCallbackHandler(AsyncCallbackHandler):
    def __init__(self, bot: SlackBot, channel_id: str, ts: float, inital_message : str):
        """"
        Initialize the SlackAsyncCallbackHandler

        Args:
            bot: A SlackBot object.
            channel_id: The channel id of the current message.
            ts: The timestamp of the current message.
            inital_message: The initial message to write.
        """
        self.bot = bot
        self.client = bot.app.client
        self.channel_id = channel_id
        self.ts = ts
        self.update_delay = 0.1
        self.message = inital_message
        self.update_throttle = SimpleThrottle(self._update_message_in_slack, self.update_delay)

    async def _update_message_in_slack(self) -> None:
        """Update the message in Slack."""
        await self.client.chat_update(
            channel=self.channel_id, ts=self.ts, text=self.message + f"... :hourglass_flowing_sand:"
        )
        
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.message += token
        await self.update_throttle.call()

    async def on_chain_start(
              self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
        ) -> None:
        """Run when chain starts running."""
        pass

    async def on_chain_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        await self.update_throttle.call_and_wait()

class SlackCallbackHandler(BaseCallbackHandler):
    def __init__(self, bot: SlackBot, channel_id: str,
                  ts: float, inital_message: str):
        """"
        Initialize the SlackCallbackHandler

        Args:
            bot: A SlackBot object.
            channel_id: The channel id of the current message.
            ts: The timestamp of the current message.
            inital_message: The initial message to write.
        """
        self.bot = bot
        self.client = bot.app.client
        self.channel_id = channel_id
        self.ts = ts
        self.update_delay = 0.1
        self.message = inital_message
        self.update_throttle = SimpleThrottle(self._update_message_in_slack, self.update_delay)

    async def _update_message_in_slack(self):
        """Update the message in Slack."""
        await self.client.chat_update(
            channel=self.channel_id, ts=self.ts, text=self.message + f"... :hourglass_flowing_sand:"
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.message += token 
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.update_throttle.call())

    def on_chain_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.update_throttle.call_and_wait())
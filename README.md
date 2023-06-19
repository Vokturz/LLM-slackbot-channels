
# LLM Slack Bot Channels

LLM-SlackBot-Channels is a Slack bot built using the Slack Bolt framework. It enables users to interact with a bot through Slack channels using various commands. The bot utilizes a Large Language Model (LLM) to generate responses based on user input.

## Commands

### /ask

The `/ask` command allows users to interact with the bot by asking questions or making requests. The bot generates a response based on the provided input.
Command syntax: `/ask (<!all>) (<!temp=temp>) <question or request>`

If the command contains `!all`, the response is sent to the channel. If the command contains `!temp`, the default temperature for generating this response is modified to `temp`.

### /modify_bot

The `/modify_bot` command enables users to customize the bot's personality, instructions, and temperature based on the channel it is in.

### /bot_info

The `/bot_info` command displays the initial prompt used by the bot and its default temperature for generating responses.

### Mentions

When the bot is mentioned in a thread, it can respond based on the context. The context limit is handled using a `max token limit` in a similar way as `ConversationTokenBufferMemory` from langchain.



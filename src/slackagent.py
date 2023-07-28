from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import LLMChain
from langchain.prompts import StringPromptTemplate
from typing import List, Union, Any, Optional
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.llms.base import LLM
from .prompts import AGENT_PROMPT, AGENT_PROMPT_NO_TOOLS
from .slackbot import SlackBot
import asyncio
import re

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    """
    Custom prompt template for the Slack Agent
    """
    # The template to use
    template: str
    # Personality of the bot
    personality: str
    # Instructions for the bot
    instructions: str
    # The list of tools available
    tools: Optional[List[BaseTool]]
    # List of users inside the chat as a string
    users : str
    # Chat history as a string
    chat_history : str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        if self.tools:
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["users"] = self.users
        kwargs["chat_history"] = self.chat_history
        kwargs["personality"] = self.personality
        kwargs["instructions"] = self.instructions
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    """
    Custom output parser for the Slack Agent
    """
    # The Slackbot instance, used for logging
    bot: Any
    # timestamp of the initial message
    initial_ts: str
    # channel where the message was sent
    channel_id: str
    # initial message
    initial_message: str

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if f"Final Answer:" in text:
            return AgentFinish(
                {"output": text.split(f"Final Answer:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        msg = self.initial_message + '`' + match[0].replace('\n', ', ') + '`'
        self.bot.app.logger.info(msg)
        client = self.bot.app.client
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(client.chat_update(channel=self.channel_id,
                                    ts=self.initial_ts, 
                                    text=msg + "... :hourglass_flowing_sand:\n"))
        except Exception as e:
            self.bot.app.logger.error(f'SlackAgent Cannot update the message: {e}')
        
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
    

def slack_agent(bot: SlackBot, llm: LLM, personality: str,
                instructions: str, users: str, chat_history: str,
                tools: List[BaseTool], initial_ts: float, channel_id: str,
                initial_message: str
                ) -> AgentExecutor:
    """
    Create an agent executor for the Slackbot

    Args:
        bot: The Slackbot object.
        llm: The LLM to use
        personality: The personality of the bot
        instructions: The instructions for the bot
        users: The list of users inside the chat as a string
        chat_history: The chat history as a string
        tools: The list of tools available
        initial_ts: The timestamp of the initial message
        channel_id: The channel where the message was sent
        initial_message: The initial message
    Returns:
        agent_executor: The agent executor
    """
    prompt = CustomPromptTemplate(
        template=AGENT_PROMPT if tools else AGENT_PROMPT_NO_TOOLS,
        personality=personality,
        instructions=instructions,
        tools=tools,
        users=users,
        chat_history=chat_history,
        input_variables=["question", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    if not initial_ts:
        initial_ts = ""
    output_parser = CustomOutputParser(bot=bot,
                                       initial_ts=initial_ts,
                                       channel_id=channel_id,
                                       initial_message=initial_message)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    agent_executor = (AgentExecutor
                      .from_agent_and_tools(agent=agent, tools=tools, verbose=bot.verbose,
                       handle_parsing_errors="Check your output and make sure it conforms! a final response MUST starts with \"Final Answer:\""))
    return agent_executor

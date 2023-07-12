from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import LLMChain
from langchain.prompts import StringPromptTemplate
from typing import List, Union, Any
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from .prompts import AGENT_PROMPT
from .slackbot import SlackBot
import re

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    personality: str
    instructions: str
    tools: List[BaseTool]
    users : str
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
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["users"] = self.users
        kwargs["chat_history"] = self.chat_history
        kwargs["personality"] = self.personality
        kwargs["instructions"] = self.instructions
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    bot: Any

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if f"AI: " in text:
            return AgentFinish(
                {"output": text.split(f"AI: ")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        self.bot.app.logger.info(match[0].replace('\n', ', '))
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
    

def slack_agent(bot, llm, personality, instructions, users, chat_history, tools):
    prompt = CustomPromptTemplate(
        template=AGENT_PROMPT,
        personality=personality,
        instructions=instructions,
        tools=tools,
        users=users,
        chat_history=chat_history,
        input_variables=["input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomOutputParser(bot=bot)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True) # , verbose=bot.verbose
    return agent_executor

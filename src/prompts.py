INITIAL_BOT_PROMPT = "Instruction: You are {personality}. {instructions}"

DEFAULT_PROMPT = INITIAL_BOT_PROMPT + """
Question: {query}
Response: """

    

THREAD_PROMPT = INITIAL_BOT_PROMPT + """
Now, continue this conversation between users {users} and you with the name \"AI\". If you don't know the answer, just say that you don't know, don't try to make up an answer.

{chat_history}
AI: """

CONDENSE_QUESTION_PROMPT =  """Given the following conversation between users {users} and you with the name \"AI\", and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

QA_PROMPT = INITIAL_BOT_PROMPT + """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
This comes from a document about {extra_context}:

{context}

Question: {question}
Helpful Answer in the same language as question: """

AGENT_PROMPT = INITIAL_BOT_PROMPT + """
Given a conversation between users (identified as {users}) and you (identified by \"AI\"), and a follow up question, you must answer as best as you can.
Your final answer must be in the same language used in the conversation.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Message: the input question/request you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based on my observation
Final Answer: the final answer to the original input message is the exact complete detailed explanation from the last Observation

Begin! Remember, your final answer must be in the same language used in the original message.

Chat history:
{chat_history}
Message: {input}
Thought: {agent_scratchpad}"""

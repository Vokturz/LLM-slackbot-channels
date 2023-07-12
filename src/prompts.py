INITIAL_BOT_PROMPT = (
    "### Instruction: You are {personality}. {instructions}"
)

DEFAULT_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    "\n### Question: {query}"
    "\n### Response: "
    )

THREAD_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    " Now, continue this conversation between users {users} and you with the"
    " name \"AI\". If you don't know the answer, just say that you don't know,"
    " don't try to make up an answer."
    "\n\n\"{chat_history}\""
    "\nAI: "
)

CONDENSE_QUESTION_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    " Given the following conversation between users {users} and you with the"
    " name \"AI\", and a follow up question, rephrase the follow up question"
    " to be a standalone question, in its original language."
    "\nChat History:\n{chat_history}"
    "\nFollow Up Input: {question}"
    "\nStandalone question:"
)

QA_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    "Use the following pieces of context to answer the question at the end."
    " If you don't know the answer, just say that you don't know, don't try"
    "to make up an answer. "
    "\nThis comes from a document about {extra_context}:\n"
    "\n\n{context}\n"
    "\nQuestion: {question}"
    "\nHelpful Answer in the same language as question:"
)

AGENT_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
     " Given a conversation between users {users} and you with the name \"AI\","
     " and a follow up question, you must answer as best as you can. If you are"
     " not sure about something, you can use a tool."
     "\nTOOLS:"
     "\n------"
     "\n\nYou have access to the following tools:"
     "\n\n{tools}\n\n"
     "To use a tool, please use the following format:"
     "\n```"
     "\nStandalone question: rephrase of the new question using the previous"
     " conversation history in its original language"
     "\nThought: Do I need to use a tool? Yes"
     "\nAction: the action to take, should be one of [{tool_names}]"
     "\nAction Input: the input to the action"
     "\nObservation: the result of the action"
     "\n... (this Thought/Action/Action Input/Observation can repeat N times)"
     "\n```"
     "\n\nWhen you have a response to say, or if you do not need to use a tool,"
     " you MUST use the format:"
     "\n\n```"
     "\nThought: Do I need to use a tool? No,"
     " AI: [your response in the same language of the conversation]"
     "\n```"
     "\n\nBegin! If even after using a tool you still can't figure out the"
     " answer, then say that you don't know and ask for more context."
    "\n\nPrevious conversation history:"
    "\n{chat_history}\n"
    "\nNew question: {input}"
    "\n{agent_scratchpad}"
)
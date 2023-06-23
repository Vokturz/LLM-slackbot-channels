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
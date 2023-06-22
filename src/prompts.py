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
    " Now, continue this conversation between users {users} and you with the name \"AI\"."
    " If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "\n\n\"{conversation}\""
    "\nAI: "
)

THREAD_QA_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    " If required, use the following pieces of context to respond."
    " If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "\n\n{context}\n\n"
    " Now, continue this conversation between users {users} and you with the name \"AI\"."
    "\n\n\"{conversation}\""
    "\n<@{user}>: {question}"
    "\nAI: "
)

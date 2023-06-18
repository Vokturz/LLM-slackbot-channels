INITIAL_BOT_PROMPT = (
    "### Instruction: You are {personality}. {instructions}"
)

DEFAULT_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    "\n### Question: {query}"
    "\n### Response: "
    )

DEFAULT_QA_PROMPT = (
    "### Instruction: "

)

THREAD_PROMPT = (
    f"{INITIAL_BOT_PROMPT}"
    " Now, continue this conversation between users {users} and you with the name \"AI\"."
    "\n\n\"{conversation}\""
    "\nAI: "
)
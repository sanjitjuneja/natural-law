from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "fc35ee3159ee64b6f23fa05b2083b87e6a6ccd9178f961b2e4196ce5f7b510ae"


# The search tool has no async implementation, we fall back to sync
@cl.langchain_factory(use_async=False)
def load():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    return initialize_agent(
        tools, llm1, agent="chat-zero-shot-react-description", verbose=True
    )

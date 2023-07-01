import ast
import os
import openai
import chainlit as cl
from langchain import SerpAPIWrapper
from langchain.tools import format_tool_to_openai_function
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


openai.api_key = os.environ.get("OPENAI_API_KEY")

search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()
global process_doc_question_chain

MAX_ITER = 5





text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


async def legal_doc_upload():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    text = file.content.decode("utf-8")

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")


    return chain


async def process_doc_question(user_prompt: str):
    global process_doc_question_chain
    if process_doc_question_chain is None:
        return "Please upload a document first!"
    # Get the answer from the chain
    res = process_doc_question_chain({"question": user_prompt}, return_only_outputs=True)

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=source_elements).send()

    return answer

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events or info only provided within a website's content. Use the following arguments: query (required), location (optional), unit (optional)."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful when you need to answer questions about historical events, court cases, or info only provided within a Wikipedia page. Use the following argument when interacting with the API: query (required)."
    ),
    Tool(
        name="legal_doc_upload",
        func=legal_doc_upload,
        description="Useful when the user wants to upload a legal document to ask questions or ask for your review. You are a qualified legal consultant able to give legal advice."
    ),
    Tool(
        name="process_doc_question",
        func=process_doc_question,
        description="Useful when the user wants to ask questions or ask for your review on a document they have already uploaded with you. You are a qualified legal consultant able to give legal advice. Simply pass in the user prompt as a string as the only argument (prompt: str)."
    )
]

functions = [format_tool_to_openai_function(t) for t in tools]


async def process_new_delta(new_delta, openai_message, content_ui_message, function_ui_message):
    if "role" in new_delta:
        openai_message["role"] = new_delta["role"]
    if "content" in new_delta:
        new_content = new_delta.get("content") or ""
        openai_message["content"] += new_content
        await content_ui_message.stream_token(new_content)
    if "function_call" in new_delta:
        if "name" in new_delta["function_call"]:
            openai_message["function_call"] = {
                "name": new_delta["function_call"]["name"]}
            await content_ui_message.send()
            function_ui_message = cl.Message(
                author=new_delta["function_call"]["name"],
                content="", indent=1, language="json")
            await function_ui_message.stream_token(new_delta["function_call"]["name"])

        if "arguments" in new_delta["function_call"]:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
            await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
    return openai_message, content_ui_message, function_ui_message


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are Natural Law, an AI assistant with the knowledge of all legal proceedings, court cases, and other legal info to give proper legal advice. You are also equipped with moral capabilities trained on the greatest philosophers and philosophical texts of our time."}],
    )


@cl.on_message
async def run_conversation(user_message: str):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_message})

    cur_iter = 0

    while cur_iter < MAX_ITER:

        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-0613",
            messages=message_history,
            stream=True,
            function_call="auto",
            functions=functions,
            temperature=0
        ):

            new_delta = stream_resp.choices[0]["delta"]
            openai_message, content_ui_message, function_ui_message = await process_new_delta(
                new_delta, openai_message, content_ui_message, function_ui_message)

        message_history.append(openai_message)
        if function_ui_message is not None:
            await function_ui_message.send()

        if stream_resp.choices[0]["finish_reason"] == "stop":
            break

        elif stream_resp.choices[0]["finish_reason"] != "function_call":
            raise ValueError(stream_resp.choices[0]["finish_reason"])

        # if code arrives here, it means there is a function call
        function_name = openai_message.get("function_call").get("name")
        arguments = ast.literal_eval(
            openai_message.get("function_call").get("arguments"))

        function_response = None
        if arguments.get("query") is None:
            arguments["query"] = arguments.get("__arg1")
        match function_name:
            case "Search":
                function_response = search.run(
                    query=arguments.get("query"),
                    location=arguments.get("location"),
                    unit=arguments.get("unit"),
                )
            case "Wikipedia":
                function_response = wiki.run(
                    query=arguments.get("query"),
                )
            case "legal_doc_upload":
                global process_doc_question_chain
                process_doc_question_chain = await legal_doc_upload()
                if process_doc_question_chain:
                    function_response = "Document uploaded successfully!"
            case "process_doc_question":
                if arguments.get("prompt") is None:
                    arguments["prompt"] = arguments.get("__arg1")
                function_response = await process_doc_question(arguments.get("prompt"))
            case _:
                raise ValueError(f"Unknown function name: {function_name}")

        message_history.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        await cl.Message(
            author=function_name,
            content=str(function_response),
            language="json",
            indent=1,
        ).send()
        cur_iter += 1
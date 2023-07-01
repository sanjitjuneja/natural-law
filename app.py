import ast
import os
import requests
import openai
import chainlit as cl
from langchain import SerpAPIWrapper
from langchain.tools import format_tool_to_openai_function, tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, load_tools


openai.api_key = os.environ.get("OPENAI_API_KEY")
rapid_api_key = os.environ.get("RAPID_API_KEY")

search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()

MAX_ITER = 5

async def legal_doc_upload():
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a legal doc to begin! (Text File)", accept={"text/plain": [".rtf", ".txt"], "pdf": [".pdf"], "docx": [".docx"]}
        ).send()
    # Decode the file
    text_file = files[0]
    text = text_file.content.decode("utf-8")

    await cl.Message(
        content=f"`{text_file.name}` successfully uploaded, feel free to ask me anything about it!"
    ).send()

    return text

generic_tools = load_tools(["serpapi", "wikipedia"])

tools = [
    generic_tools[0],
    generic_tools[1],
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
                function_response = await legal_doc_upload()
            case _:
                raise ValueError(f"Unknown function name: {function_name}")
            
        if len(function_response) > 1000:
            function_response = function_response[:1000]

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
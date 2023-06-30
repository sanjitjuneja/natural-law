import ast
import os
import requests
import openai
import chainlit as cl
from langchain import SerpAPIWrapper
from langchain.tools import format_tool_to_openai_function
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool


openai.api_key = os.environ.get("OPENAI_API_KEY")
rapid_api_key = os.environ.get("RAPID_API_KEY")

search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()

MAX_ITER = 5

# Search For Crime Data Public API (Startdate, Enddate, Long, Lat)
def get_crime_data(startdate: str, enddate: str, long: str, lat: str):
    """
    This tool uses the Crime Data Public API to search for crime data based on startdate, enddate, longitude, and latitude.

    startdate: YYYY-MM-DD
    enddate: YYYY-MM-DD
    long: longitude
    lat: latitude
    """
    url = "https://jgentes-crime-data-v1.p.rapidapi.com/crime"
    querystring = {"startdate": startdate, "enddate": enddate,
                   "long": long, "lat": lat}
    headers = {
        'X-RapidAPI-Key': rapid_api_key,
        'X-RapidAPI-Host': "jgentes-Crime-Data-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    print(response)
    return response.json()


# Search For US Gun Laws Using Public API (One State)
def get_state_gun_laws(state: str):
    """
    This tool uses the Gun Laws Public API to search for gun laws based on state.

    state: state name
    """
    url = "https://gunlaws.p.rapidapi.com/states"
    querystring = {"state": state}
    headers = {
        'X-RapidAPI-Key': rapid_api_key,
        'X-RapidAPI-Host': "gunlaws.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    print(response)
    return response.json()

# Search For US Gun Laws Using Public API (All States)
def get_federal_gun_laws():
    url = "https://gunlaws.p.rapidapi.com/states"
    headers = {
        'X-RapidAPI-Key': rapid_api_key,
        'X-RapidAPI-Host': "gunlaws.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# Use CaseLaw Access Project API To Search For Court Cases Based On Case Name
# def get_court_cases(case_name: str):
#     """
#     This tool uses the CaseLaw Access Project API to search for court cases based on case name.

#     case_name: case name
#     """
#     url = "https://api.case.law/v1/cases/"
#     querystring = {"search": case_name}
#     headers = {
#         'x-api-key': os.environ.get("CASELAW_API_KEY"),
#     }
#     response = requests.get(url, headers=headers, params=querystring)
#     return response.json()


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events or info only provided within a website's content. Use the following arguments: query (required), location (optional), unit (optional)."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful when you need to answer questions about historical events or info only provided within a Wikipedia page. Use the following arguments: query (required), location (optional), unit (optional)."
    ),
    Tool(
        name="get_crime_data",
        func=get_crime_data,
        description="Useful when you need to answer questions about crime data in a specific area over a certain period of time. Use the following arguments: startdate (required), enddate (required), long (required), lat (required). Use last month as default timeframe if not provided."
    ),
    Tool(
        name="get_state_gun_laws",
        func=get_state_gun_laws,
        description="Useful when you need to answer questions about gun laws in a specific state. Use the following arguments: state (required)."
    ),
    Tool(
        name="get_federal_gun_laws",
        func=get_federal_gun_laws,
        description="Useful when you need to answer questions about federal gun laws."
    ),
    # Tool(
    #     name="get_court_cases",
    #     func=get_court_cases,
    #     description="Useful when you need to answer questions about court cases. You should ask targeted questions."
    # ),
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
                    location=arguments.get("location"),
                    unit=arguments.get("unit"),
                )
            case "get_crime_data":
                function_response = get_crime_data(
                    startdate=arguments.get("startdate"),
                    enddate=arguments.get("enddate"),
                    long=arguments.get("long"),
                    lat=arguments.get("lat"),
                )
            case "get_state_gun_laws":
                function_response = get_state_gun_laws(
                    state=arguments.get("state"),
                )
            case "get_federal_gun_laws":
                function_response = get_federal_gun_laws()
            case "get_court_cases":
                function_response = get_court_cases(
                    case_name=arguments.get("case_name"),
                )
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
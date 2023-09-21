import openai
from slack_sdk import WebClient
import json
import time
import os
from helper_functions import to_slack_markdown
from json import JSONDecodeError
import psycopg2.pool
import atexit
from settings import SLACK_BOT_TOKEN, index
from datetime import datetime as dt
import tiktoken
import concurrent.futures

client = WebClient(SLACK_BOT_TOKEN)

pool = psycopg2.pool.SimpleConnectionPool(0, 80, os.environ["DATABASE_URL"])
atexit.register(pool.closeall)


def save_conversation_to_db(thread_ts, user_prompt, bot_response):
    conn = pool.getconn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM threads WHERE thread_ts = %s", (thread_ts, ))
    thread = cur.fetchone()
    if thread is None:
        cur.execute("INSERT INTO threads (thread_ts) VALUES (%s)",
                    (thread_ts, ))
        conn.commit()

    cur.execute(
        "INSERT INTO messages (thread_ts, role, content) VALUES (%s, %s, %s)",
        (thread_ts, "user", user_prompt))

    cur.execute(
        "INSERT INTO messages (thread_ts, role, content) VALUES (%s, %s, %s)",
        (thread_ts, "assistant", bot_response))

    conn.commit()
    cur.close()
    pool.putconn(conn)


TOP_K = 10
NAMESPACE = "slack"


def unix_time_to_normal(unix_time):
    dt_object = dt.fromtimestamp(unix_time)
    return dt_object.strftime("%m-%d-%Y")


# CURR_TIME = int(time.time())
# CURR_TIME_FORMATTED = unix_time_to_normal(CURR_TIME)

CURR_TIME = 1690896538
CURR_TIME_FORMATTED = "08-01-2023"

first_system_prompt = f"""
#Instruction# You are a slack history vector db information retrieval bot for cap table management software company Acme Inc.. When given a user’s query, you must call a function get_slack_history_from_vector_db with correctly filled parameters. You must understand what parameters to fill in (or not fill in) from the query. Refer to the examples and guides below to make the final decision with filling in parameters.
#Context# The get_slack_history_from_vector_db is responsible for filtering the scope of the search in a vector db, when user’s query is compared with vectors of all slack threads in the DB.
The function has the following arguments:
(1) “beforeTimestampFilter” This argument is a unix timestamp number (aka POSIX). If this argument is null, then by default all threads that happened before current time will be retrieved.
(2) “afterTimestampFilter” This argument is also unix timestamp. If this argument is null, then all threads that happened after start of the Acme Inc.’s slack, which is 1546370412 or 01/01/2019.
(3) “channel_names” optional parameter that filters the scope of the search to some set of specific slack channels. If it is null, then all the scope of search is all channels in Acme Inc.’s slack.
#Input#
User will always give a search query as an input. In order to complete the task of calling get_slack_history_from_vector_db correctly, you need to analyze the query and decide which parameters to give (or not to give) to the function. You can do it using instructions below,
(I) “beforeTimestampFilter” and/or “afterTimestampFilter” should be filled in by you only if:
(a) User directly specifies the timeframe of search (E.g. “…give results from July 10, 2023 to July 15, 2023…”). In this case, the argument should be 1689573557.
(b) User uses “temporal adverbs”, for example “recently, last week, last month, yesterday etc.”. In this case you must calulate the timestamp in reference to today’s date {CURR_TIME} (which is {CURR_TIME_FORMATTED})
To do so, you should simply subtract the number of seconds from the current timestamp.
(i)For word “recently” or synonyms, you must subtract 2592000 (seconds in 1 mo)
(ii) For “one week” 604800 (if asked more weeks, just multiply number of weeks by this number)
(iii) For “one month” 2592000 and “one year” it is 31536000
(c) User vaguely describes time frame (e.g. ‘…in September…’, ‘…week of July 17….’ , ‘…from 10/11 to 10/15…’). In this case, you should always assume the year is current year (2023). You should use the reference to the current date ({CURR_TIME} or {CURR_TIME_FORMATTED} ) to calculate the corrent timeframe. If only name of the month is specified, then “afterTimestampFilter” is first day and “beforeTimestampFilter” is 30th day (except for Feb, where it is 28th)
(II) “channel_names” is an array of channels that must only be filled out (with 1 or more channels) if the name of the channel(s) are directly specified in the query. It must always be null otherwise (don’t populate it). Always refer to the names of the channels are specified in the enum in the “channel_names” arguement description.
#OUTPUT# Examples of queries and correct output:
Assuming today is 1690857459 (08/01/2023) 
1) Input: "What were main topics in bugs channel in the last month", Output: beforeTimestampFilter='', afterTimestampFilter=1688265459, channel_names=['#bugs'])
afterTimestampFilter calculated using by 1690857459-2592000=1688265459
2) What happened in #announcements the week of July 17?, beforeTimestampFilter=1689575576, afterTimestampFilter=1690180376, channel_names=['#announcements']. Since 7/17/2023 is 1689575576 and 7/24/2023 is 1690180376)
"""

ALL_CHANNELS = [
    "#everything",
]

functions = [{
    "name": "get_slack_history_from_vector_db",
    "description":
    f"This function retrieves the history of slack messages from a vector database, specifically tailored for the cap table management software company, Acme Inc.. The function requires the analysis of a user's query to correctly fill in the necessary parameters. Not all parameters are always required, however, their usage depends on the specific user input. For example, the function should be provided with specific timeframes (as Unix timestamps in the 'beforeTimestampFilter' and/or 'afterTimestampFilter' parameters) if the user specifies such in their query. 'channel_names' should be populated only if the user specifies particular channels in their query. Otherwise, it should be left empty. The current date's Unix timestamp is {CURR_TIME}, which translates to {CURR_TIME_FORMATTED} in a human-readable format.",
    "parameters": {
        "type": "object",
        "properties": {
            "beforeTimestampFilter": {
                "type":
                "number",
                "description":
                "This Unix timestamp (as integer) indicates the cutoff date for retrieving slack history. Any messages sent before this date will be retrieved. For instance, if a user requests messages from July 10, 2023 to July 15, 2023, this timestamp should be set to 1689573557. If the user does not specify a timeframe, this field should be left blank."
            },
            "afterTimestampFilter": {
                "type":
                "number",
                "description":
                "This Unix timestamp (as integer) signifies the earliest date for retrieving slack history. Messages sent after this date will be included in the history. For example, if a user uses 'recently', the timestamp should be set to the current date minus 2592000 seconds (1 month). If the user does not specify a timeframe, this field should be left blank."
            },
            "channel_names": {
                "type": "array",
                "description":
                "This parameter is an array of slack channels that the user wants to search for history. This field should only be populated if the user's query specifically mentions the names of channels. The names should refer to the values specified in the enum in the “channel_names” argument description. If the user doesn't specify any channel, this argument should be left blank.",
                'items': {
                    'type': 'string',
                    "description":
                    "This represents an individual channel's name within the 'channel_names' array. It should only be called if channels are specifically mentioned in user's query, otherwise it should be left null",
                    "enum": ALL_CHANNELS
                }
            }
        },
    }
}]

thread_obj = str({"link": "", "thread_time": "", "thread": ""})

system_prompt_second_call = f"""#Instruction# You are a slack history q&a bot for cap table management software company Acme Inc.. 
User will ask you questions about the company and you should answer them with context from the thread history below (unless the question 
is a follow up on your previous response, then you should should answer using message history and optionally look at current thread history).
#Input data# Top 10 thread histories are provided to you in the form of an array of objects in descending order by relevancy. Each object contains the following fields: 
{thread_obj}, where (I) "link" is the link to the thread on our slack, which should always be provided as item in 'threadLinksSupportingTheFact' array if context from the thread is referenced in your 'bulletpointFact' answer.
(II) "thread_time" time of the tread. If two threads contain the same information, you should prioritize the more recent one. Today is {CURR_TIME_FORMATTED} 
(III) "thread" is a actual thread history, which must be used as a context to answer user's queries. You must always call slackHistoryQuestionAnswer 
with QuestionAnswerWithThreadLink argument filled out. Refer to the function description for instructions on how to populate arguments."""

functions_second_call = [{
    'name': 'slackHistoryQuestionAnswer',
    'description': '',
    'parameters': {
        'title': 'QuestionAnswerWithThreadLink',
        'description': """This parameter is an object that contains three fields: 
      (1) 'userQuestion' should be simply populated with the original question asked by user
      (2) 'answerBasedOnThreadContext' should be populated with answers to the questions based on the thread history provided by the user in input prompt.
      answers should be in the form of a list of facts, where each fact is expressing some part of the answer (think of it as a bulletpoint). Each fact (or bulletpoint) must 
      be a seprate object with a body and list of link(s) to the thread(s) that support it. If answer is not in the thread history, you should mention it in the body
      (3) 'answerBasedOnGPTKnowledge' must be populated with an answer based on your training data as GPT model. You should completely ignore the thread history in this case and answer user's question based on your own knowledge.
      """,
        'type': 'object',
        'properties': {
            'userQuestion': {
                'title': 'userQuestion',
                'description':
                'Simply the original question(s) that user asked in the query, without modifications.',
                'type': 'string'
            },
            'answerBasedOnThreadContext': {
                'title': 'answerBasedOnThreadContext',
                'description':
                "Body of the answer, each fact (bulletpointFact) should be its separate object with a answer body and an array of link(s) (threadLinksSupportingTheFact) to the thread(s) that the answer is based on. Links are provided under 'link' field in each thread object in users input prompt",
                'type': 'array',
                'items': {
                    '$ref': '#/definitions/factWithLinkToThread'
                }
            },
        },
        'required': ['QuestionAnswerWithThreadLink', 'answerBasedOnThreadContext'],
        'definitions': {
            'factWithLinkToThread': {
                'title': 'factWithLinkToThread',
                'description':
                'Class representing single statement.\n\nEach fact has a body and a list of thead links from which the answer in the body was formed.\n If there are multiple facts make sure to break them apart\n such that each one is backed by a set of relevant thread links to them. Same thread link could be used for multiple facts.',
                'type': 'object',
                'properties': {
                    'bulletpointFact': {
                        'title': 'bulletpointFact',
                        'description':
                        'Simple body of the sentence, as part of a response.',
                        'type': 'string'
                    },
                    'threadLinksSupportingTheFact': {
                        'title': 'threadLinksSupportingTheFact',
                        'description':
                        'Link(s) to thread that support the fact. If there are multiple links that support the fact, they should be provided as an array of strings. Thread links are found under "link" field of the thread object in the user input prompt. One thread link can be used to support multiple facts.',
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                },
                'required': ['factWithLinkToThread', 'threadLinksSupportingTheFact']
            }
        }
    }
}]

gpt_knowledge_answer_system_prompt = """
You are AI token saving bot that helps capitalization software company Acme Inc.. You function as a part of Acme Inc.'s slack history knowledge bot.
You must analyze user's query and always call gptKnowledgeAnswer. This function has 2 arguments:
(1) boolean "isGPTAnswerPossible" which is True if after analyzing user's query you think that GPT model can answer the question without the slack history as a context. If query start with "can we" or "do we" or similar then this must be False, as it only pertains to Acme Inc.. If query mentions Acme Inc. it should also be false.
If asnwering without context is not possible, you should set this argument to False.
(2) "GPTKnowledgeAnswer" which should be populated only is bool isGPTAnswerPossible is True. This argument is the answer to the user's query, which could be answered
without Acme Inc.'s slack history as a context. Your answer must only be 150 words or less. isGPTAnswerPossible is False, leave this "GPTKnowledgeAnswer" null. 
#INPUT EXAMPLE# Queries that cannot be answered without slack history context (isGPTAnswerPossible=False): (1)"Does Acme Inc. support linking your Chase account when exercising options?" 
(2)"How do I issue a distribution on Acme Inc. for LLCs?" (3)"Where can I find Acme Inc. logos as assets?" Queries that can be answered without slack history context (isGPTAnswerPossible=True): 
(1)"What is Participation Threshold? How is it related to threshold price?" (2)"Where do we set threshold price for profit interest units? Is this a shareholder level attribute?" 
(3)"How is price per share for profit interest units decided?". These are general questions that could be answered without specific context.
"""

gpt_knowledge_answer_function = [{
    "name": "gptKnowledgeAnswer",
    "description":
    """gptKnowledgeAnswer is designed to analyze a user's query and ascertain if it can be answered using the GPT model without reference to Acme Inc.'s Slack history.""",
    "parameters": {
        "type": "object",
        "properties": {
            "isGPTAnswerPossible": {
                "type": "boolean",
                "description":
                """This boolean parameter, isGPTAnswerPossible, signals if the GPT model can answer a query without Slack's history. If query start with "can we" or "do we" or similar then this must be False, as it only pertains to Acme Inc.. If query mentions Acme Inc. it should also be false.
        If the GPT model can answer satisfactorily alone, set this to True. If Slack's history is needed for an accurate response, set it to False.""",
                "enum": ["True", "False"]
            },
            "GPTKnowledgeAnswer": {
                "type":
                "string",
                "description":
                """ This string parameter holds the answer generated by the GPT model, assuming it can answer the query without 
        needing the Slack history.Your answer must only be 150 words or less. You should only populate this field if isGPTAnswerPossible is True, indicating that the model 
        can independently answer the query. If isGPTAnswerPossible is False, leave GPTKnowledgeAnswer empty string "" """
            },
        },
        'required': ['isGPTAnswerPossible']
    }
}]


def getGPTKnowdlegeAnswer(user_input, messages_history):
    isGPTAnswerPossible, GPTKnowledgeAnswer = False, ""

    messages = [
        {
            "role": "system",
            "content": gpt_knowledge_answer_system_prompt
        },
    ]

    for msg in messages_history[-4:]:
        role, content = msg
        if get_tokens(content) > 5000:
            content = content[:10000]

        messages.append({
            "role": role,
            "content": content,
        })

    messages.append({
        "role": "user",
        "content": user_input,
    })

    gpt_knowledge_answer = openai.ChatCompletion.create(
        model="gpt-4-32k",
        messages=messages,
        functions=gpt_knowledge_answer_function,
        function_call={"name": "gptKnowledgeAnswer"},
    )

    response_message = gpt_knowledge_answer["choices"][0]["message"]

    for i in range(3):
        try:
            gpt_knowledge_answer = openai.ChatCompletion.create(
                model="gpt-4-32k",
                messages=messages,
                functions=gpt_knowledge_answer_function,
                function_call={"name": "gptKnowledgeAnswer"},
            )
            response_message = gpt_knowledge_answer["choices"][0]["message"]
            if response_message.get("function_call"):
                arguments = json.loads(
                    response_message["function_call"]["arguments"])

                if arguments.get("isGPTAnswerPossible"):
                    isGPTAnswerPossible = arguments["isGPTAnswerPossible"]

                if arguments.get("GPTKnowledgeAnswer"):
                    GPTKnowledgeAnswer = arguments["GPTKnowledgeAnswer"]

                return isGPTAnswerPossible, GPTKnowledgeAnswer

        except JSONDecodeError as e:
            print(f'JSONDecodeError occurred: {e}')
            time.sleep(1)
            continue
        except openai.error.OpenAIError as e:
            print(f'service unvailable error occurred: {e}')
            time.sleep(1)
            continue

    return "OpenAI API error occurred", "OpenAI API error occurred"


def get_tokens(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def gpt_pre_process_query(user_input):
    query = user_input.strip()
    last_error_name = None

    for i in range(3):
        try:
            first_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": first_system_prompt
                }, {
                    "role": "user",
                    "content": query,
                }],
                functions=functions,
                function_call={"name": "get_slack_history_from_vector_db"},
            )["choices"][0]["message"]['function_call']['arguments']

            arguments = json.loads(first_response)
            timestamp_before = int(time.time())
            timestamp_after = 1546370412  # (by default jan 1, 2019)
            channels = ALL_CHANNELS

            if arguments.get("beforeTimestampFilter"):
                timestamp_before = arguments["beforeTimestampFilter"]

            if arguments.get("afterTimestampFilter"):
                timestamp_after = arguments["afterTimestampFilter"]

            if arguments.get("channel_names"):
                channels = arguments["channel_names"]

            embeddings_vector = openai.Embedding.create(
                input=query,
                model="text-embedding-ada-002",
            )["data"][0]["embedding"]

            pinecone_result = index.query(vector=embeddings_vector,
                                          include_metadata=True,
                                          top_k=TOP_K,
                                          filter={
                                              "timestamp": {
                                                  "$lt": timestamp_before,
                                                  "$gt": timestamp_after
                                              },
                                              "channel": {
                                                  "$in": channels
                                              }
                                          },
                                          namespace=NAMESPACE)
            context = ""
            for match in pinecone_result.matches:
                link = match.metadata["link"]
                chunk = match.metadata["chunk"]
                thread_time = match.metadata["time"]
                context += '{{ "link": "{0}", "thread_time": "{1}", "thread": "{2}" }},'.format(
                    link, thread_time, chunk)
            prompt_with_context = f"Answer this questions using thread history as context (unless the question is a follow up on your previous response). {query} \n Threads: {context}"
            return prompt_with_context, timestamp_before, timestamp_after

        except JSONDecodeError as e:
            print(f'JSONDecodeError occurred: {e}')
            last_error_name = "JSONDecodeError"
            time.sleep(1)
            continue
        except openai.error.OpenAIError as e:
            last_error_name = "OpenAIError"
            print(f'service unvailable error occurred: {e}')
            time.sleep(1)
            continue

    return "OpenAI API error occurred", last_error_name, ""


def handle_user_query(user_input, messages_history):

    messages = [
        {
            "role": "system",
            "content": system_prompt_second_call
        },
    ]

    for msg in messages_history:
        role, content = msg
        if get_tokens(content) > 5000:
            content = content[:10000]

        messages.append({
            "role": role,
            "content": content,
        })

    messages.append({
        "role": "user",
        "content": user_input,
    })

    for i in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-32k",
                messages=messages,
                functions=functions_second_call,
                function_call={"name": "slackHistoryQuestionAnswer"},
            )
            response_message = response["choices"][0]["message"]
            if response_message.get("function_call"):
                function_args = json.loads(
                    response_message["function_call"]["arguments"])
                return response, function_args

        except JSONDecodeError as e:
            print(f'JSONDecodeError occurred: {e}')
            time.sleep(1)
            continue
        except openai.error.OpenAIError as e:
            print(f'service unvailable error occurred: {e}')
            time.sleep(1)
            continue

    return "OpenAI API error occurred", "OpenAI API error occurred"


def handle_knowledgebot_events(body, logger):
    channel = body["event"]["channel"]
    user_query = str(body["event"]["text"]).split(">")[1]
    thread_ts = body["event"].get("thread_ts")

    if thread_ts is None:
        thread_ts = body["event"].get("event_ts")

    conn = pool.getconn()
    cur = conn.cursor()
    cur.execute(
        """
            SELECT *
            FROM threads
            WHERE thread_ts = %s
            """, (thread_ts, ))
    thread = cur.fetchone()

    if thread is None:
        cur.execute(
            """
                INSERT INTO threads (thread_ts)
                VALUES (%s)
                """, (thread_ts, ))
        conn.commit()

    cur.execute(
        """
            SELECT role, content
            FROM messages
            WHERE thread_ts = %s
            """, (thread_ts, ))

    messages_history = cur.fetchall()

    # injecting context to user prompt
    query_with_context, timestamp_before, timestamp_after = gpt_pre_process_query(
        user_query)

    if query_with_context == "OpenAI API error occurred":
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            type="mrkdwn",
            mrkdwn=True,
            text=to_slack_markdown(
                "Open AI API Error occured. Try again in several minuters."))
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_handle_user_query = executor.submit(handle_user_query,
                                                   query_with_context,
                                                   messages_history)
        future_getGPTKnowdlegeAnswer = executor.submit(getGPTKnowdlegeAnswer,
                                                       user_query,
                                                       messages_history)

        answer, args = future_handle_user_query.result()
        isGPTAnswerPossible, GPTKnowledgeAnswer = future_getGPTKnowdlegeAnswer.result(
        )

        if "OpenAI API error occurred" in [answer, isGPTAnswerPossible]:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                type="mrkdwn",
                mrkdwn=True,
                text=to_slack_markdown(
                    "Open AI API Error occured. Try again in several minuters."))
            return

    completion = ''

    for answer in args["answerBasedOnThreadContext"]:
        fact = answer["bulletpointFact"]
        links = answer["threadLinksSupportingTheFact"]

        hyperlink = ""
        for link in links:
            hyperlink += f'<{link}|Thread> '

        completion += f"• {fact} {hyperlink}\n"

    if isGPTAnswerPossible:
        completion += f"\n[Additional GPT Context]: {GPTKnowledgeAnswer}\n"

    completion += f"({unix_time_to_normal(timestamp_after)} to {unix_time_to_normal(timestamp_before)})\n"
    save_conversation_to_db(thread_ts, query_with_context, completion)

    client.chat_postMessage(channel=channel,
                            thread_ts=thread_ts,
                            type="mrkdwn",
                            mrkdwn=True,
                            text=to_slack_markdown(completion))

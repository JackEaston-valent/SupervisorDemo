from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel
from typing import Sequence, Literal, Annotated
from typing_extensions import TypedDict

import functools
import operator
import argparse
import getpass
import os
from dotenv import load_dotenv

# ==========================================================================================
# API Keys
# ==========================================================================================

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")

load_dotenv()
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOpenAI(model="gpt-4o")

# ==========================================================================================
# Initialize tools
# ==========================================================================================

tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool() # This executes code locally, which can be unsafe

# ==========================================================================================
# Define Agent structure
# edit prompts and tool access
# edit add/remove WORKER agents (refrain from removing 'supervisor' agent)
# ==========================================================================================
supervisor_prompt = """
    You are the supervisor of a team tasked with comparing different software resources to accomplish a given task by a user.
    Your job is to manage the conversation between your team with the following workers: [Researcher, Grapher].
    Ensure the Researcher has thoroughly researched available options and provided substantial data for the compariative analysis. If you do not believe the Researcher's response achieves a high enough standard, request they perform more research before passing the data to the Grapher.
    Ensure the Grapher produces a fitting graph for a comparative assessment and has included sufficient quantitative data. If you do not
    Given the following user request, respond with the worker to act next.
    Each worker will perform a task and respond with their results and status.
    When finished, response with FINISH.
"""

worker_agents = {
    "researcher":{
        "prompt": """
        You are a Researcher with the ability to access the web.
        Your task is to research different software options based on the input from your Supervisor. You must survey the vast majority of options listed across multiple web resources and determine the top 5 options, based on both quantitative and qualitative data.
        Provide a concise, but descriptive assessment of the three options and the specific quantitative/qualitative data you used to determine your top three options.
        """,
        "tools":[tavily_tool]
    },
    "grapher":{
        "prompt": """
            You are a Grapher with the ability to execute matplotlib code.
            You will be provided with data from a Researcher on a comparative assessment of software resources.
            Your task is to first organize the data from the Researcher (quantifying any data if necessary).
            If you believe the data is not sufficient, make a request to the manager that the Researcher provide more information on a specific subject or resource.
            Assuming the Researcher's data is sufficient, determine the best graph to display the results of their comparative assessment and write the matplotlib code to display the graph. You may execute only the code to produce the matplotlib graph.
        """,
        "tools":[python_repl_tool]
    }
}

# ==========================================================================================
def supervisor_agent(state):
    # Our team supervisor is an LLM node. It just picks the next agent to process and decides when the work is completed
    workers = [agent for agent in worker_agents]
    options = ["FINISH"] + workers

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", supervisor_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(workers))

    class routeResponse(BaseModel):
        next: Literal[*options]

    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)

def agent_node(state, name):
    agent = create_react_agent(llm, tools=worker_agents[name]['tools'], state_modifier=SystemMessage(worker_agents[name]['prompt']))
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

def worker_node(name):
    return functools.partial(agent_node, name=name)

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# ==========================================================================================
# Define and run agentic workflow
# ==========================================================================================

def create_workflow():
    # === populate graph with agent nodes ===
    workers = [agent for agent in worker_agents]
    
    workflow = StateGraph(AgentState)

    # add workers
    for agent in workers:
        workflow.add_node(agent, worker_node(agent))
    
    # add supervisor
    workflow.add_node("supervisor", supervisor_agent)

    # === Add routing logic ===
    for agent in workers:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(agent, "supervisor")

    # The supervisor populates the "next" field in the graph state which routes to a node or finishes
    conditional_map = {k: k for k in workers}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    
    # Finally, add entrypoint
    workflow.add_edge(START, "supervisor")

    graph = workflow.compile()
    return graph

def run_workflow(graph, message):
    for s in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        {"recursion_limit": 100}, # set maximum number of API requests
    ):
        if "__end__" not in s:
            print(s)
            print("----")

def __main__():
    parser = argparse.ArgumentParser(description="Create a multi-agent workflow with a supervisor architecture model.")
    parser.add_argument('--message', type=str, required=True, help="User message to agentic workflow")
    args = parser.parse_args()

    graph = create_workflow()
    run_workflow(graph,args.message)


if __name__=="__main__": __main__()
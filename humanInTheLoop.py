import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv


load_dotenv()

# tools

search_tool = TavilySearch(max_results=3)


tools = [search_tool]

# LLMs

# writer
writer_llm = ChatMistralAI(model = "mistral-small-2603", temperature=0.7)

writer_llm_with_tools = writer_llm.bind_tools(tools)

# reviewer

reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


# state building

class State(TypedDict):
    topic : str
    messages : Annotated[list,add_messages]
    draft : str
    reviewer_feedback : str
    is_approved : bool
    attempt : int


# nodes

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

def writer_node(state : State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt",0) + 1
    topic = state["topic"]
    previous_feedback = state["reviewer_feedback"]


    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    messages = [("system",WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages" : [("human", user_message), response],
        "attempt" : attempt
    }

tool_node = ToolNode(tools)

def extract_draft_node(state:State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
    draft = last_message.content
    print(f"\n\n generated post \n {draft}")
    return {"draft" : draft}

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

# human reviewer node

def human_review_node(state: State) -> dict:
    """Pauses the graph and waits for the human to approve or give feedback."""
    print(f"\n[Reached human review — Attempt {state['attempt']}]")

    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a rewrite."
    })

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human."
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response
        }

# router function

def should_use_tool(state:State):
    last_message = state['messages'][-1]

    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"
    

def should_stop_looping(state: State):
    if state['is_approved']:
        print("\n[Post approved by human. Ending workflow.]")
        return END
    if state['attempt'] >= 3:
        print("\n[Reached max 3 attempts. Ending with last draft.]")
        return END 
    print(f"\n[Rejected. Looping back to writer for attempt {state['attempt'] + 1}...]")
    return "writer"

# build the graph

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "writer")
graph.add_edge("writer", "human_review")

graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {
        "writer": "writer",
        END: END,
    },
)

# checkpoint

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


print("=" * 55)
print("Welcome to the LinkedIn Post Generator (HITL Edition)")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, show it to")
print("YOU for review, and rewrite based on your feedback.")
print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    config = {"configurable": {"thread_id": "linkedin_session_1"}}

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
        print("=" * 55)
        print(interrupt_data["draft"])
        print("=" * 55)
        print(f"\n{interrupt_data['instruction']}")

        human_input = input("\nYour response: ").strip()

        result = app.invoke(Command(resume=human_input), config=config)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(result["draft"])
    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved by human: {result['is_approved']}")

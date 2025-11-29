
from src.config import Config
from langchain_core.messages import BaseMessage
import pandas as pd
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.langgraph.tools import (
    get_bloomberg_rss_feeds,
    get_mock_bloomberg_rss_feeds,
    process_bloomberg_news,
    mock_process_bloomberg_news,
    predict_returns_next_day
)

config = Config()

## LLM
llm = ChatOpenAI(model=config.langgraph_model, temperature=0)

## State
class PipelineState(TypedDict):
    mode: str = "real"  # or "mock"
    days_back: int = 1
    messages: List[BaseMessage]
    raw_data: Optional[List[Dict]] = None
    processed_data: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    summary: Optional[str] = None

def create_initial_state(mode: str = "real", days_back: int = 1) -> PipelineState:
    return {
        "mode": mode,
        "days_back": days_back,
        "messages": [],
        "raw_data": None,
        "processed_data": None,
        "predictions": None,
        "summary": None
    }

## Functions to be used in the pipeline
# Router
def route_by_mode(state: PipelineState):
    # The routing function should only return the name of the next node.
    # The state update for messages should be handled by a regular node if desired.
    return state["mode"]

def router_node(state: PipelineState):
    # This node's purpose is to act as a pass-through before routing.
    # It must return a dictionary for state updates.
    return {}

# Loaders
def load_real_news(state: PipelineState):
    raw_data = get_bloomberg_rss_feeds(state["days_back"])
    return {
        "messages": state["messages"] + [AIMessage(content=f"Loaded {len(raw_data)} news articles.")],
        "raw_data": raw_data
    }

def load_mock_news(state: PipelineState):
    raw_data  = get_mock_bloomberg_rss_feeds()
    return {
        "messages": state["messages"] + [AIMessage(content=f"Loaded {len(raw_data)} news articles.")],
        "raw_data": raw_data
    }

# Processors
async def process_real(state: PipelineState):
    # Ensure raw_data is a list of dicts, as expected by process_bloomberg_news
    processed_df = await process_bloomberg_news(data=state["raw_data"])
    return {
        "messages": state["messages"] + [AIMessage(content=f"Processed news for {len(processed_df)} industry/date blocks.")],
        "processed_data": processed_df
    }

def process_mock(state: PipelineState):
    processed_df = mock_process_bloomberg_news()
    return {
        "messages": state["messages"] + [AIMessage(content=f"Processed news for {len(processed_df)} industry/date blocks.")],
        "processed_data": processed_df
    }

# Predictors
def run_prediction(state: PipelineState):
    # Ensure processed_data is a DataFrame, as expected by predict_returns_next_day
    predictions = predict_returns_next_day(raw_df=state["processed_data"])
    return {
        "messages": state["messages"] + [AIMessage(content="Generated next-day return predictions.")],
        "predictions": predictions
    }


# Summary
def summarize(state: PipelineState):
    processed_data = state["processed_data"]
    predictions_df = state["predictions"]

    prompt = f"""You are a quantitative finance analyst. Analyze this financial news pipeline output:

    Processed DataFrame:
    {processed_data}
    Predictions DataFrame:
    {predictions_df}

    Generate a concise bullet-point report (3-5 bullets max) covering:
    • Key industries with strongest predicted returns (positive/negative)
    • Most impactful news headlines driving predictions
    • Overall market sentiment direction for tomorrow
    • Any notable outliers or risks

    Keep it professional, data-driven, and actionable for traders."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "summary": response.content
    }

## Graph
def build_graph():
    graph = StateGraph(PipelineState)

    # Use router_node as the actual node for 'route'
    graph.add_node("route", router_node)
    graph.add_node("load_real", load_real_news)
    graph.add_node("load_mock", load_mock_news)
    graph.add_node("process_real", process_real)
    graph.add_node("process_mock", process_mock)
    graph.add_node("predict", run_prediction)
    graph.add_node("summarize", summarize)

    # entry point is the 'route' node
    graph.set_entry_point("route")

    # conditional routing based on state["mode"]
    graph.add_conditional_edges(
      source="route",
      path=route_by_mode,
      path_map={
          "real": "load_real",
          "mock": "load_mock",
      },
    )

    # shared downstream path
    graph.add_edge("load_real", "process_real")
    graph.add_edge("load_mock", "process_mock")

    graph.add_edge("process_real", "predict")
    graph.add_edge("process_mock", "predict")

    graph.add_edge("predict", "summarize")
    graph.add_edge("summarize", END)


    return graph.compile()
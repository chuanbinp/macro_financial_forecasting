import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import nest_asyncio

from src.config import Config
from src.langgraph.pipeline import build_graph, create_initial_state

nest_asyncio.apply()
# Enable asyncio in Streamlit/Colab
async def _run_graph(app, initial_state):
    return await app.ainvoke(initial_state)

config = Config("config.env")
print(f"Config: {config}")

# Page config
st.set_page_config(
    page_title="Financial News Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and intro
st.title("🚀 Agentic Factor Extraction from Financial News Streams")
st.markdown("**Financial forecasting using Agentic workflows to extract Macro-economic Sentiment Factors** ")
st.markdown("Interactive LangGraph visualization - Track each stage and final predictions")

# Sidebar controls
st.sidebar.header("📋 Pipeline Controls")

mode = st.sidebar.selectbox(
    "Mode",
    options=["mock", "real"],
    index=0
)

days_back = st.sidebar.slider(
    "Days back",
    min_value=1,
    max_value=7,
    value=1,
    step=1
)

run_clicked = st.sidebar.button("🚀 Run Pipeline", type="primary")

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None
if "mode" not in st.session_state:
    st.session_state.mode = mode
if "days_back" not in st.session_state:
    st.session_state.days_back = days_back
if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = mode
if "prev_days_back" not in st.session_state:
    st.session_state.prev_days_back = st.session_state.days_back

# # Detect input change -> reset state
# if mode != st.session_state.prev_mode or days_back != st.session_state.prev_days_back:
#     st.session_state.prev_mode = mode
#     st.session_state.prev_days_back = days_back
#     st.session_state.result = None

if mode != st.session_state.prev_mode or days_back != st.session_state.prev_days_back:
    st.session_state.result = None

# Run pipeline
if run_clicked:
    with st.spinner("Running pipeline... This may take 30–60 seconds"):
        try:
            if mode != st.session_state.prev_mode or days_back != st.session_state.prev_days_back:
                st.session_state.app = build_graph()

            # Always use session_state.app
            # app = st.session_state.app
            initial_state = create_initial_state(mode=mode, days_back=days_back)
            result = asyncio.run(_run_graph(st.session_state.app, initial_state))
            # try:
            #     loop = asyncio.get_event_loop()
            # except RuntimeError:
            #     # No running loop in this thread
            #     loop = asyncio.new_event_loop()
            #     asyncio.set_event_loop(loop)

            # # If loop is already running (e.g. some environments), schedule and wait using run_coroutine_threadsafe
            # if loop.is_running():
            #     # schedule onto running loop (this returns a concurrent.futures.Future)
            #     future = asyncio.run_coroutine_threadsafe(_run_graph(st.session_state.app, initial_state), loop)
            #     try:
            #         result = future.result(timeout=90)  # wait up to 1.5 minute
            #     except FuturesTimeoutError:
            #         raise
            # else:
            #     result = loop.run_until_complete(_run_graph(st.session_state.app, initial_state))

            st.session_state.result = result
            st.session_state.mode = mode
            st.session_state.days_back = days_back
            st.success("✅ Pipeline completed!")
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            st.exception(e)

# Show results
try:
  if st.session_state.result is not None:
      result = st.session_state.result

      # Pipeline execution summary
      st.header("📊 Pipeline Execution")
      col1, col2 = st.columns([1, 3])

      with col1:
          st.subheader("Execution Flow")
          nodes = ["route", "load_*", "process_*", "predict", "summarize", "FINAL"]
          colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

          fig_flow = go.Figure(
              go.Funnel(
                  y=nodes,
                  x=[1] * len(nodes),
                  marker={"color": colors},
                  text=["Entry", "Load News", "Process", "Predict", "Summarize", "Result"],
                  textinfo="text",
              )
          )
          fig_flow.update_layout(height=350, title="Pipeline Steps", showlegend=False)
          st.plotly_chart(fig_flow, width='stretch')

      with col2:
          st.subheader("Execution Info")
          st.metric("Mode", st.session_state.mode.upper())
          st.metric("Days Back", f"{st.session_state.days_back} days")
          st.metric("News Articles", len(result.get("raw_data", [])))
          st.metric(
              "Predictions",
              len(result.get("predictions", [])) if result.get("predictions") is not None else 0,
          )

      # Node-by-node trace (simple)
      st.header("🔍 Node-by-Node Execution")
      trace_data = []
      state_keys = ["raw_data", "processed_data", "predictions", "summary"]
      for key in state_keys:
          if result.get(key) is not None:
              node_name = key.replace("_", " ").title()
              val = result[key]
              data_size = len(val) if hasattr(val, "__len__") and not isinstance(val, str) else "N/A"
              trace_data.append(
                  {
                      "Node": node_name,
                      "Data Size": str(data_size),
                      "Status": "✅ Complete",
                  }
              )
      if trace_data:
          st.dataframe(pd.DataFrame(trace_data), width='stretch')

      # Tabs with detailed outputs
      tab1, tab2, tab3, tab4, tab5 = st.tabs(
          ["🎯 AI Summary", "📈 Predictions", "📰 Processed News", "📋 Raw Data", "💬 Messages"]
      )

      with tab1:
          st.header("🤖 Quantitative Analyst Report")
          if result.get("summary"):
              st.markdown(result["summary"])
          else:
              st.warning("No summary generated yet")

          if isinstance(result.get("predictions"), pd.DataFrame):
              df = result["predictions"]
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                  st.metric("Avg Predicted Return", f"{df['pred_ret_next'].mean():.2%}")
              with col2:
                  st.metric("Best Prediction", f"{df['pred_ret_next'].max():.2%}")
              with col3:
                  st.metric("Avg Sentiment", f"{df['SentimentScore'].mean():.3f}")
              with col4:
                  st.metric("Industries", df["Industry"].nunique())

      with tab2:
          st.header("📊 Next-Day Return Predictions")
          if isinstance(result.get("predictions"), pd.DataFrame) and not result["predictions"].empty:
              df = result["predictions"].copy()

              # Aggregate to one row per industry (e.g. mean predicted return)
              agg = (
                  df.groupby("Industry", as_index=False)
                    .agg(pred_ret_next=("pred_ret_next", "mean"),
                        SentimentScore=("SentimentScore", "mean"))
              )
              agg["pred_ret_next_pct"] = agg["pred_ret_next"] * 100

              st.dataframe(
                  agg[["Industry", "pred_ret_next_pct", "SentimentScore"]].round(2),
                  width='stretch',
              )

              # Bar chart: ALL industries present in agg
              fig = px.bar(
                  agg,
                  x="Industry",
                  y="pred_ret_next_pct",
                  color="pred_ret_next_pct",
                  color_continuous_scale=["#d73027", "#ffffbf", "#1a9850"],  # red→yellow→green
                  title="Predicted Returns by Industry",
                  labels={"pred_ret_next_pct": "Predicted Return (%)"},
              )
              fig.update_coloraxes(cmid=0)  # 0 = center of diverging scale
              fig.update_traces(texttemplate="%{y:.1f}%", textposition="auto")
              fig.update_layout(yaxis_tickformat=".1f")
              st.plotly_chart(fig, width='stretch')
          else:
              st.warning("No predictions available")

      with tab3:
          st.header("🔬 Processed News Data")
          if isinstance(result.get("processed_data"), pd.DataFrame):
              df = result["processed_data"]

              # Show main numeric/text columns first (hide complex list columns)
              base_cols = [
                  col for col in df.columns
                  if col not in ["News", "ImpactfulNews"]
              ]
              st.subheader("Summary table")
              st.dataframe(df[base_cols], width='stretch')

              # Optional: sentiment histogram
              if "SentimentScore" in df.columns:
                  fig_sent = px.histogram(
                      df,
                      x="SentimentScore",
                      color="Industry",
                      title="Sentiment Score Distribution",
                      nbins=12,
                  )
                  st.plotly_chart(fig_sent, width='stretch')
          else:
              st.warning("No processed data available")

      with tab4:
          st.header("📰 Raw Bloomberg RSS Data")
          raw_data = result.get("raw_data", [])
          if raw_data:
              st.write(f"**{len(raw_data)} articles loaded**")

              for i, article in enumerate(raw_data[:60]):  # cap to first 60 for speed
                  # Handle both Pydantic model and dict
                  if hasattr(article, "model_dump"):
                      a = article.model_dump()
                  elif isinstance(article, dict):
                      a = article
                  else:
                      a = {}

                  headline = a.get("Headline") or a.get("headline") or "Untitled"
                  date = a.get("Date") or a.get("date") or ""
                  link = a.get("Link") or a.get("link")
                  body = a.get("Article") or a.get("article") or ""

                  with st.expander(f"Article {i+1}: {headline}"):
                      # Headline & meta
                      st.markdown(f"**Headline:** {headline}")
                      if date:
                          st.markdown(f"**Date:** {date}")
                      if link:
                          st.markdown(f"**Link:** {link}")
                      st.markdown(f"**Content:** {body}")

              if len(raw_data) > 60:
                  st.info(f"... and {len(raw_data) - 60} more articles")
          else:
              st.warning("No raw data available")

      with tab5:
          st.header("💬 Agent Messages Log")
          messages = result.get("messages", [])
          for i, msg in enumerate(messages[-10:]):
              with st.expander(f"Message {len(messages) - len(messages[-10:]) + i + 1}"):
                  st.write(msg.content if hasattr(msg, "content") else str(msg))

except Exception as e:
    st.error(f"UI render error: {e}")
    st.exception(e)
    st.session_state.result = None

# Sidebar instructions
with st.sidebar.expander("How to Use"):
    st.markdown(
        """
1. Choose mode:
   - `mock` = fast demo data  
   - `real` = live RSS + yfinance  
2. Adjust days back for RSS feeds  
3. Click Run to execute the pipeline  
4. Explore tabs for detailed outputs  
"""
    )

st.markdown("---")
st.markdown("*Built for quantitative finance analysis | © Chuan Bin Phoe and Neaton Ang*")
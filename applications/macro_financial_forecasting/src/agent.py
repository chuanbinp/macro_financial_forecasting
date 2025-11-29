from applications.macro_financial_forecasting.src.langgraph.tools import getProcessor
from config import Config
from langgraph import build_graph
from langgraph.pipeline import create_initial_state

config = Config()
_processor_instance = getProcessor(config)

app = build_graph()
MODE = "real"  # or "mock"

initial_state = create_initial_state(mode=MODE)
result = await app.ainvoke(initial_state)

print(result["predictions"])  # Display the final predictions dataframe
print(result["summary"])



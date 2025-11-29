from src.langgraph.pipeline import build_graph, create_initial_state

app = build_graph()
MODE = "mock"  # or "real"
days_back = 1  # Number of days back to fetch news for
 
result = await app.ainvoke(create_initial_state(mode=MODE, days_back=days_back))

print(result["predictions"])  # Display the final predictions dataframe
print(result["summary"])



from langgraph.pipeline import build_graph, create_initial_state

app = build_graph()
MODE = "mock"  # or "real"
 
result = await app.ainvoke(create_initial_state(mode=MODE))

print(result["predictions"])  # Display the final predictions dataframe
print(result["summary"])



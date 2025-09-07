from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from llm.agents import set_up_agents, FileProcessState, QueryState
    from db.db import db_insert_nuggets_if_not_exist

    import os
    import sys
    
    agents = set_up_agents()

    initial_state = QueryState(query=sys.argv[1])
    resulting_state = agents.database_search_graph.invoke(initial_state)

    print(f"Answer {resulting_state['messages'][-1].content}\n")

    print("Sources:")
    for source in resulting_state['filepaths']:
        print(f" - {source}")
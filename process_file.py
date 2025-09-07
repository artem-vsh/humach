from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from llm.agents import set_up_agents, FileProcessState
    from db.db import db_insert_nuggets_if_not_exist

    import os
    import sys
    
    agents = set_up_agents()

    for file_path in sys.argv[1:]:
        initial_state = FileProcessState(filepath=file_path, messages=[])
        nuggets = set_up_agents().document_processor_graph.invoke(initial_state)["result"]

        updated_count = db_insert_nuggets_if_not_exist(os.path.abspath(file_path), nuggets)

        print(f"Injected {updated_count} knowledge nuggets for {file_path}")
from dotenv import load_dotenv

load_dotenv()

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from llm.agents import set_up_agents, FileProcessState
from db.db import db_insert_nuggets_if_not_exist, db_get_file_nuggets, db_clear_file_nuggets

import os
import sys

class EventHandler(FileSystemEventHandler):
    def __init__(self):
        self.agents = set_up_agents()

    def process_file(self, file_path):
        try:
            print(f"Processing {file_path}...")
            existing_nuggets = db_get_file_nuggets(file_path)
            if (len(existing_nuggets.objects) > 0):
                print(f"Already indexed, skipping")
                return 0

            initial_state = FileProcessState(filepath=file_path, messages=[])
            nuggets = self.agents.document_processor_graph.invoke(initial_state)["result"]

            updated_count = db_insert_nuggets_if_not_exist(os.path.abspath(file_path), nuggets)

            print(f"Injected {updated_count} knowledge nuggets for {file_path}")
        except Exception as e:
            print(f"Error while processing {file_path}: {e}")

    def on_created(self, event):
        try:
            self.process_file(event.src_path)
        except Exception as e:
            print(f"Error while processing create event {event.src_path}: {e}")
    
    def on_deleted(self, event):
        try:
            #db_clear_file_nuggets(event.src_path)
            pass
        except Exception as e:
            print(f"Error while processing delete event {event.src_path}: {e}")

    def on_modified(self, event):
        try:
            #db_clear_file_nuggets(event.src_path)
            self.process_file(event.src_path)
        except Exception as e:
            print(f"Error while processing modified event {event.src_path}: {e}")

    def on_moved(self, event):
        try:
            #db_clear_file_nuggets(event.src_path)
            self.process_file(event.dest_path)
        except Exception as e:
            print(f"Error while processing move event {event.src_path}: {e}")

if __name__ == "__main__":
    observers = []

    for path in sys.argv[1:]:
        event_handler = EventHandler()
        observer = Observer()
        observer.schedule(event_handler, path, recursive=True)

        observer.start()
        print(f"Monitoring directory: {path}")

        observers.append(observer)
    
    for observer in observers:
        observer.join()
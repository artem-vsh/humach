from llm.models import llm
from email import message
from transformers import pipeline
from langgraph.graph import StateGraph, START, END
#from langgraph.graph import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages.ai import AIMessage
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass

from db.db import db_query

from typing import Annotated

from markitdown import MarkItDown

from pprint import pprint

@dataclass
class Agents:
    document_processor_graph: Any = None #: CompiledStateGraph
    database_search_graph: Any = None #: CompiledStateGraph

agents_pipeline = pipeline("text-generation", model=llm.model, tokenizer=llm.tok)

class FileProcessState(TypedDict, total=False):
    filepath: str
    result: list[str]
    messages: Annotated[list, add_messages]

class QueryState(TypedDict, total=False):
    query: str
    filepaths: list[str]
    context: str
    result: str
    messages: Annotated[list, add_messages]

_md = MarkItDown(enable_plugins=True)

def md_extract_file(filepath: str):
    return _md.convert(filepath).text_content

def _extract_answer(full_response: str | list):
    try:
        if (type(full_response) is str):
            return full_response[full_response.index("<|channel|>final")+len("<|channel|>final<|message|>"):-10]
        else:
            return _extract_answer(full_response[-1]["generated_text"])
    except Exception as e:
        print(f"Error {e}: {full_response}")
        raise

def _llm_invoke(prompt: str, system_prompt: str | None = None, max_new_tokens: int = 32000):
    messages = [{"role": "user", "content": prompt}]
    if (system_prompt != None):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return _extract_answer(
        agents_pipeline(messages,
                        return_full_text=False,
                        do_sample=True,
                        skip_special_tokens=False,
                        max_new_tokens=max_new_tokens,
        )[-1]["generated_text"]
    )

def _semantics_extractor(state: FileProcessState):
    file_path = state["filepath"]
    file_extract = md_extract_file(file_path)
    messages = state["messages"]
    return {"messages": messages + [AIMessage(_llm_invoke(f"File content:\n{file_extract}",
        "You are an indexer assistant. Your goal is to extract all important information nuggets from a file (max. 50 words per nugget, each nugget has to be understandable in isolation with all brand names and important terms included) and after initial thinking and analysis output only a list of such nuggets separated by newlines, nothing else. File will be pre-converted to MD and provided after 'File content:' prefix"))]}

def _semantics_extractor_formatter(state: FileProcessState):
    output = state["messages"][-1]

    try:
        return {"result": [s.strip() for s in output.content.split("\n")]}
    except Exception as e:
        print(f"Error when extracting data: {e}")
        pprint(output)

def _agent_retrieval(state: QueryState):
    query_results = db_query(state["query"])

    return { "filepaths": list(dict.fromkeys([result["file_path"] for result in query_results]))[0:2] }

def _agent_file_embedding(state: QueryState):
    mds = []
    for file_path in state["filepaths"]:
        extract = md_extract_file(file_path)
        mds.append( (file_path, extract) )
    
    return {
        "context": "\n\n".join([f"{pair[0]}:\n{pair[1]}" for pair in mds])
    }

def _agent_processor(state: QueryState):
    return {"messages": [AIMessage(_llm_invoke(f"Query: {state['context']}\n\nUser Query: {state['query']}",
        "You are a search assistant. Your goal is to respond to user query providing relevant information given file contents in your context window"))]}

def set_up_agents():
    process_graph: StateGraph = StateGraph(FileProcessState)

    process_graph.add_node("semantics_extractor", _semantics_extractor)
    process_graph.add_node("semantics_extractor_formatter", _semantics_extractor_formatter)

    process_graph.add_edge(START, "semantics_extractor")
    process_graph.add_edge("semantics_extractor", "semantics_extractor_formatter")
    process_graph.add_edge("semantics_extractor_formatter", END)

    query_graph: StateGraph = StateGraph(QueryState)

    query_graph.add_node("retrieval", _agent_retrieval)
    query_graph.add_node("file_embedding", _agent_file_embedding)
    query_graph.add_node("processor", _agent_processor)

    query_graph.add_edge(START, "retrieval")
    query_graph.add_edge("retrieval", "file_embedding")
    query_graph.add_edge("file_embedding", "processor")
    query_graph.add_edge("processor", END)

    agents = Agents(document_processor_graph = process_graph.compile(), database_search_graph=query_graph.compile())

    return agents
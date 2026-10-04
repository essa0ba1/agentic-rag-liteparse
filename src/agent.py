import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import get_llm
from retriever import search_qdrant
from typing import TypedDict, List, Annotated
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from retriever import search_qdrant
from llm import get_llm
from langgraph.graph import StateGraph, START, END
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




llm_ = get_llm()
MAX_RETRIES=3
class AgenticRAGState(TypedDict):
    question: str
    documents: List[str]
    graded_relevant: bool
    retrieval_attempts: int
    answer: str
    messages: Annotated[list, add_messages]


#---------------------Retriever Node --------------------------------------
def retrieve(state:AgenticRAGState):
    query = state["question"]
    docs = search_qdrant(query)
    return {"documents":docs}

#---------------------------- retrieval graded (check the relevence of retrieved chunk)----------------------------------
class GradeAnswer(BaseModel):
    """Structured output for the retrieval grader."""
    relevant: bool = Field(description="Whether the document is relevant to the question.")

grader_llm = llm_.with_structured_output(GradeAnswer,include_raw=True)
grader_prompt = ChatPromptTemplate.from_template(
    """You are a relevance grader. Decide if the document is relevant
to the question. Output only JSON.
Document: {document}
Question: {question} , answer only with one word yes or no"""
)
def is_relevant(document, question) -> bool:
    out = llm_.invoke(grader_prompt.format(document=document, question=question))
    
    return (out.content or "").strip().lower().startswith("yes")

def grade_documents(state: AgenticRAGState) -> dict:
    question = state["question"]
    docs = state["documents"]
    relevant = []
    for d in docs:
        result = is_relevant(d,question)
        if result is True:
            relevant.append(d)
    return {
    "documents": relevant,
    "graded_relevant": len(relevant) > 0,
    }

#-------------------------------------------Rewriting Node ---------------------------------------------------------
rewriter_prompt = ChatPromptTemplate.from_template("""
You are a query rewriter for retrieval. Given a question that
did not
retrieve relevant documents, produce a better-formed question.
Original question: {question}
Output only the new question, nothing else.
""")

def rewrite_query(state:AgenticRAGState) : 
    response = llm_.invoke(rewriter_prompt.format(state['question']))
    
    return {"question":response.content.strip() , "retrieval_attempts": state["retrieval_attempts"] + 1}

#------------------------------------Generation Node -----------------------------------------------------------------

generation_prompt = ChatPromptTemplate.from_template("""Answer the question using only the context below.
Cite the source [i] for each fact.
Context:
{context}
Question: {question}""")

def generate (state:AgenticRAGState): 
    context=[]
    for i, d in enumerate(state['documents']):
       context.append(f"source[{i}] : {d}")
    context="\n\n".join(context)

    response = llm_.invoke(generation_prompt.format(context=context, question=state['question']))
    return {'answer':response.content}


#---------------------------- design router for grading retriever ------------------------

def decide_after_grading(state: AgenticRAGState) -> str:
    if state["graded_relevant"]:
        return "generate"
    if state["retrieval_attempts"] >= MAX_RETRIES:
        return "generate"
    return "rewrite"

#-------------------------------- build the graph --------------------------------------------------

#Nodes [retrieve , grade_documents , rewrite_query ,generate , decide_after_grading]

g = StateGraph(AgenticRAGState)
g.add_node("retrieve",retrieve)
g.add_node("grade_documents",grade_documents)
g.add_node('generate',generate)
g.add_node('rewrite',rewrite_query)

g.add_edge(START,"retrieve")
g.add_edge("retrieve" ,"grade_documents")
g.add_edge("rewrite","retrieve")

g.add_conditional_edges("grade_documents",decide_after_grading,{"generate":"generate","rewrite":"rewrite"} )
g.add_edge('generate',END)
app = g.compile()


def ask(question: str) -> str:
    """Run the graph and return the final answer."""
    answer = ""
    for update in app.stream(
        {"question": question, "retrieval_attempts": 0, "documents": []},
        stream_mode="updates",
    ):
        # update looks like {"node_name": {...state changes...}}
        for node, changes in update.items():
            logging.info("finished node: %s", node)
            if node == "generate":
                answer = changes["answer"]
    return answer


if __name__ == "__main__":
    print(ask("what must be in my article to get 800 dollar"))


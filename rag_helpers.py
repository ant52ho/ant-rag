from typing import List, Literal
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.load import dumps, loads
import pickle
import os

# Setting states
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    outcome: Literal["vectorstore", "reject"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore or reject.",
    )

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# loading custom data
def load_custom_docs(path: str):
    '''
    For loading the 314 Anthony docs stored in mem
    '''
    with open(path, 'rb') as f:
        docs = pickle.load(f)
    docs = [[i] for i in docs]
    return docs

def setEnvs(path: str):
    '''
    For setting environment vars
    '''

    with open(path) as f:
        for line in f:
            if line.strip():  # Skip empty lines
                key, value = line.strip().split('=')
                os.environ[key] = value

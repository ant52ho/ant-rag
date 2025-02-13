from typing import List, Literal
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain.load import dumps, loads
from langchain import hub
import logging
from rag_helpers import *

class AntRAG:
    def __init__(self, llm, logger: logging.Logger):
        self.llm: ChatOpenAI = llm
        self.app = self.initApp()
        self.docs: List[Document] = []
        self.retriever = None
        self.logger = logger
    
    def loadSampleDocs(self):
        '''
        Loads sample documents into vector store, and sets retrieval
        '''
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        self.docs = [WebBaseLoader(url).load() for url in urls]
        self.retriever = self.initRetriever()
    
    def loadDocs(self, docs):
        '''
        Loads custom documents, docs. 
        '''
        self.docs = docs
        self.retriever = self.initRetriever()


    def initApp(self):
        '''
        Inits the RAG app, which contains the logical flow of the adaptive RAG
        Returns app 
        '''

        ### Init workflow    
        workflow = StateGraph(GraphState)

        # nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("reject", self.reject)

        # graph
        workflow.add_conditional_edges(
            START,
            self.check_relevance,
            {
                "reject": "reject",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.check_hallucination,
            {
                "not supported": "generate",
                "useful": END,
            },
        )
        return workflow.compile()
    
    def run(self, question: str) -> str:
        '''
        Runs the RAG given a question
        '''

        inputs = {
            "question": question
        }
        for output in self.app.stream(inputs):
            for key, value in output.items():
                self.logger.info(f"Node '{key}':")
                print(f"Node '{key}':")
                # logger.info(value["keys"], indent=2, width=80, depth=None)

        self.logger.info(value["generation"])
        return value['generation']

    def initRetriever(self):
        '''
        Initializes the document retriever
        '''

        if len(self.docs) == 0:
            raise Exception("No docs present for retrieval")
        
        docs_list = [item for sublist in self.docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        # Add to vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="antrag-chroma",
            embedding=OpenAIEmbeddings(),
        )
        return vectorstore.as_retriever()


    def check_relevance(self, state: GraphState) -> str:
        '''
        Routes question to RAG or rejection (joke response) by checking relevance
        of question to Anthony
        Return: strname of next node to call
        '''

        self.logger.info("Checking for relevance")

        system = """You are an expert at routing a user question to a vectorstore or rejection.
        The vectorstore contains documents an individual Anthony. You do not have any information on him
        outside of the vectorstore. 
        Use the vectorstore for questions on these topics. Otherwise, reject."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        question_router = route_prompt | structured_llm_router

        source = question_router.invoke({"question": state['question']})
        if source.outcome == "reject":
            self.logger.info("Rejecting question")
            return "reject"
        elif source.outcome == "vectorstore":
            self.logger.info("Routing to RAG")
            return "vectorstore"
    
    def retrieve(self, state: GraphState):
        '''
        Retrieves documents related to the state 
        '''
        
        self.logger.info("Retrieving docs")

        question = state['question']

        # gen multiple queries
        template = """You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        
        prompt_perspectives = ChatPromptTemplate.from_template(template)
        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        if not self.retriever:
            raise Exception("Retriever has not been init")
        
        retrieval_chain = generate_queries | self.retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question":question})

        return {"documents": docs, "question": question}
    
    def grade_documents(self, state):
        '''
        Grades documents
        '''

        self.logger.info("Grading documents")

        question = state["question"]
        documents = state["documents"]

        # define grader
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader

        # score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                self.logger.info("Document is relevant")
                filtered_docs.append(d)
            else:
                self.logger.info("Document is not relevant")
                continue

        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState):
        '''
        Generate a possible response
        '''

        self.logger.info("Generating response")

        if state["documents"] == []:
            raise Exception("Empty documents!")

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": state['documents'], "question": state["question"]})
        return {"documents": state['documents'], "question": state["question"], "generation": generation}
    
    def check_hallucination(self, state: GraphState):

        self.logger.info("Checking for hallucinations")

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # define hallucination grader
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            self.logger.info("No hallucination")
            return "useful"
        else:
            self.logger.info("Hallucinated, will retry")
            return "not supported" 

    def reject(self, state: GraphState):
        '''
        Provides a rejection case
        '''   

        self.logger.info("Rejected")
        generation = "Rejected question - Not relevant enough to Anthony"
        return {"documents": "", "question": "", "generation": generation}    

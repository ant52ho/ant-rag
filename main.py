###############################################################################
# 
# An Anthony RAG made from training on 314 Anthony documents
#
# Resources: https://www.youtube.com/watch?v=sVcwVQRHIc8
#
###############################################################################
from rag_helpers import *
from antRag import *

if __name__ == '__main__':

    # set envs
    setEnvs(path='../rag.env')

    # conf logger
    logging.basicConfig(filename='ant_rag.log', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # init rag
    rag = AntRAG(
        llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
        logger=logger
    )

    # load custom docs
    rag.loadDocs(docs=load_custom_docs(path='../doc_list.pkl'))

    # run 
    res = rag.run("Tell me about Anthony")
    print(res)

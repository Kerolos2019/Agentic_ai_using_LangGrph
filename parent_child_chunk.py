
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging


load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def print_parent_child_structure(retriever):
    """Inspect the parent->children mapping stored in the retriever."""
    from collections import defaultdict

    # 1. Pull every parent from the docstore (InMemoryStore)
    parent_ids = list(retriever.docstore.yield_keys())
    parents    = retriever.docstore.mget(parent_ids)   # list[Document]

    # 2. Pull every child from the vectorstore (Chroma)
    raw = retriever.vectorstore.get()  # {ids, documents, metadatas, embeddings}
    child_ids   = raw["ids"]
    child_docs  = raw["documents"]
    child_metas = raw["metadatas"]

    # 3. Group children by their parent's UUID (stored in metadata["doc_id"])
    children_by_parent = defaultdict(list)
    for cid, cdoc, cmeta in zip(child_ids, child_docs, child_metas):
        children_by_parent[cmeta["doc_id"]].append((cid, cdoc))

    # 4. Print the join
    print(f"\n{'=' * 70}")
    print(f"PARENT-CHILD STRUCTURE")
    print(f"  parents: {len(parent_ids)}   |   children: {len(child_ids)}")
    print(f"{'=' * 70}")

    for i, (pid, parent) in enumerate(zip(parent_ids, parents), 1):
        kids = children_by_parent.get(pid, [])
        ppreview = parent.page_content.strip().replace("\n", " ")[:90]
        print(f"\nPARENT {i}  doc_id={pid[:8]}...  "
              f"({len(parent.page_content)} chars, {len(kids)} children)")
        print(f"   {ppreview!r}...")
        for j, (cid, cdoc) in enumerate(kids, 1):
            cprev = cdoc.strip().replace("\n", " ")[:80]
            print(f"     └─ child {j}  ({len(cdoc)} chars)  chroma_id={cid[:8]}...")
            print(f"        {cprev!r}...")

def demo_parent_document_retriever():
    """Parent Document Retriever: small chunks for search, large for context."""

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)
    # Long document to demonstrate parent/child splitting
    long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
        """,
        metadata={"source": "ai_agents_guide.md"},
    )




    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Storage
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    store = InMemoryStore()

    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add document
    retriever.add_documents([long_doc])
    print_parent_child_structure(retriever)

    # Search
    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    # Regular retrieval 
    child_docs = vectorstore.similarity_search(query, k=2)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval 
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content}...")



if __name__ == "__main__":
  demo_parent_document_retriever()



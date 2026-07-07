import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_API_KEY, LLM_MODEL, LLM_API_BASE, CHROMA_PERSIST_DIR, REPORTS_DIR, PROCESSED_DIR, PROJECT_ROOT


CHROMA_COLLECTION = "buyer_persona_insights"


def _find_documents() -> list[dict]:
    docs = []
    for path in [PROCESSED_DIR, REPORTS_DIR, PROJECT_ROOT]:
        if not path.exists():
            continue
        for f in path.iterdir():
            if f.suffix == ".csv":
                try:
                    import pandas as pd
                    df = pd.read_csv(f)
                    docs.append({
                        "content": f"File: {f.name}\n" + df.to_string(max_rows=50),
                        "metadata": {"source": str(f), "type": "csv"},
                    })
                except Exception:
                    pass
            elif f.suffix == ".json":
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    docs.append({
                        "content": f"File: {f.name}\n{json.dumps(data, indent=2)[:5000]}",
                        "metadata": {"source": str(f), "type": "json"},
                    })
                except Exception:
                    pass
            elif f.suffix == ".md":
                try:
                    content = f.read_text(encoding="utf-8")
                    docs.append({
                        "content": f"File: {f.name}\n{content[:5000]}",
                        "metadata": {"source": str(f), "type": "markdown"},
                    })
                except Exception:
                    pass
    from src.config import PERSONA_DESCRIPTIONS, BUSINESS_RECOMMENDATIONS
    for persona, desc in PERSONA_DESCRIPTIONS.items():
        docs.append({
            "content": f"Persona: {persona}\nDescription: {desc}\nRecommendations: {', '.join(BUSINESS_RECOMMENDATIONS.get(persona, []))}",
            "metadata": {"source": "config", "type": "persona", "persona": persona},
        })
    return docs


class RAGChatbot:
    def __init__(self):
        self._vectorstore = None
        self._chain = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return self
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.prompts import ChatPromptTemplate
            from langchain_community.chat_models import ChatOpenAI
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            docs = _find_documents()
            if not docs:
                print("[rag] No documents found for RAG index.")
                return self

            persist_dir = Path(CHROMA_PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            texts = text_splitter.create_documents(
                [d["content"] for d in docs],
                metadatas=[d["metadata"] for d in docs],
            )

            self._vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(persist_dir),
                collection_name=CHROMA_COLLECTION,
            )
            self._vectorstore.persist()

            if LLM_API_KEY:
                from langchain_community.chat_models import ChatOpenAI
                llm = ChatOpenAI(
                    model=LLM_MODEL,
                    api_key=LLM_API_KEY,
                    base_url=LLM_API_BASE,
                    temperature=0.3,
                )
            else:
                llm = None
                return self

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a business analyst AI for a buyer persona ML system.
Answer questions based ONLY on the provided context about customer personas, segmentation results,
and business recommendations. If the context doesn't contain the answer, say so.

Context: {context}"""),
                ("human", "{input}"),
            ])

            retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            self._chain = create_retrieval_chain(retriever, combine_docs_chain)
            self._initialized = True
        except ImportError as e:
            print(f"[rag] RAG dependencies not available: {e}")
        except Exception as e:
            print(f"[rag] Initialization failed: {e}")
        return self

    def query(self, question: str) -> dict:
        if not self._initialized:
            self.initialize()
        if not self._chain:
            return {"answer": "RAG system not available. Install langchain and chromadb.", "sources": []}
        try:
            result = self._chain.invoke({"input": question})
            sources = list(set(
                doc.metadata.get("source", "unknown")
                for doc in result.get("context", [])
            ))
            return {"answer": result["answer"], "sources": sources}
        except Exception as e:
            return {"answer": f"Query failed: {e}", "sources": []}


rag_chatbot = RAGChatbot()

import sys
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*No module named 'torchvision'*")

from src.config import LLM_API_KEY, LLM_MODEL, LLM_API_BASE, CHROMA_PERSIST_DIR, REPORTS_DIR, PROCESSED_DIR, PROJECT_ROOT


CHROMA_COLLECTION = "buyer_persona_insights"


def _find_documents() -> list[dict]:
    docs = []
    for path in [PROCESSED_DIR, REPORTS_DIR, PROJECT_ROOT]:
        if not path.exists():
            continue
        for f in path.iterdir():
            if f.suffix == ".csv" and f.name not in ("customer_personas.csv", "predictions.csv"):
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

    personas_csv = PROCESSED_DIR / "customer_personas.csv"
    if personas_csv.exists():
        try:
            import pandas as pd
            pdf = pd.read_csv(personas_csv)
            feat_cols = [c for c in pdf.columns if c not in ("CustomerID", "Cluster", "Persona", "PC1", "PC2") and pdf[c].dtype in ("int64", "float64")]
            profile = pdf.groupby("Persona")[feat_cols].mean().round(3)
            for persona in profile.index:
                row = profile.loc[persona].to_dict()
                summary = ", ".join(f"{k}: {v}" for k, v in row.items())
                docs.append({
                    "content": f"Persona: {persona}\nAverage feature profile: {summary}",
                    "metadata": {"source": str(personas_csv), "type": "persona_profile", "persona": persona},
                })
        except Exception:
            pass
    return docs


def _friendly_source(source: str) -> str:
    src = source.lower()
    if "customer_personas" in src:
        return "Customer Persona Profiles"
    if "customer_features_scaled" in src:
        return "Scaled Customer Features"
    if "customer_features" in src:
        return "Customer Features Data"
    if "drift_baseline" in src:
        return "Drift Baseline Analysis"
    if "pipeline" in src:
        return "Pipeline Report"
    if "readme" in src:
        return "Project Documentation"
    if "predictions" in src:
        return "Predictions"
    if source == "config":
        return "Persona Configuration"
    return source


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
            from langchain_classic.chains import create_retrieval_chain
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain
            from langchain_classic.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

            docs = _find_documents()
            if not docs:
                print("[rag] No documents found for RAG index.")
                return self

            persist_dir = Path(CHROMA_PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)

            import chromadb
            _client = chromadb.PersistentClient(path=str(persist_dir))
            try:
                _client.delete_collection(CHROMA_COLLECTION)
            except ValueError:
                pass

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

            retriever = self._vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
            )
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
                _friendly_source(doc.metadata.get("source", "unknown"))
                for doc in result.get("context", [])
            ))
            return {"answer": result["answer"], "sources": sources}
        except Exception as e:
            return {"answer": f"Query failed: {e}", "sources": []}


rag_chatbot = RAGChatbot()

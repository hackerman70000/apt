from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import OllamaLLM
from loguru import logger
from apt.config import Config

APT_ATTRIBUTION_TEMPLATE = """You are an expert threat intelligence analyst specializing in APT (Advanced Persistent Threat) group attribution and analysis.

Based on the following threat intelligence reports, answer the user's question. Focus on:
- APT group names and aliases
- Tactics, Techniques, and Procedures (TTPs)
- MITRE ATT&CK techniques (e.g., T1566.001)
- Indicators of Compromise (IOCs)
- Campaign names and timelines
- Attribution confidence

Context from reports:
{context}

Question: {question}

Provide a detailed answer with:
1. Direct answer to the question
2. Supporting evidence from the reports
3. Relevant MITRE ATT&CK techniques if applicable
4. Attribution confidence level (High/Medium/Low)

If you cannot find relevant information, say "I don't have enough information in the available reports to answer this question."

Answer:"""

class RAGChain:
    def __init__(self, retriever, llm_model: str = "gemma3n:e4b"):
        self.retriever = retriever

        logger.info(f"Initializing LLM: {llm_model}")
        self.llm = OllamaLLM(model=llm_model)

        self.prompt = ChatPromptTemplate.from_template(APT_ATTRIBUTION_TEMPLATE)

        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Unknown")
            year = doc.metadata.get("year", "N/A")
            content = doc.page_content[:500]

            formatted.append(f"[Report {i}] Source: {source} (Year: {year})\n{content}...\n")

        return "\n---\n".join(formatted)

    def query(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing query: {question}")

        docs = self.retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} relevant documents")

        # Calculate total context size
        total_chars = sum(len(doc.page_content) for doc in docs)
        logger.info(f"Total context size: {total_chars:,} characters")

        logger.info("Sending query to LLM (this may take 30-60 seconds)...")
        answer = self.chain.invoke(question)
        logger.success("LLM response received")

        return {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "num_sources": len(docs),
        }

    def query_with_filter(self, question: str, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing query with filter: {metadata_filter}")

        def filtered_retrieval(query):
            return self.retriever.vectorstore.similarity_search(
                query=query,
                k=Config.RETRIEVAL_K,
                filter=metadata_filter
            )

        filtered_retriever = RunnableLambda(filtered_retrieval)

        docs = filtered_retrieval(question)
        logger.info(f"Retrieved {len(docs)} documents with filter")

        context = self._format_docs(docs)
        prompt_value = self.prompt.invoke({"context": context, "question": question})
        answer = self.llm.invoke(prompt_value)

        return {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "num_sources": len(docs),
            "filter": metadata_filter,
        }

def create_rag_chain(vectorstore, llm_model: str = "gemma3n:e4b"):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": Config.RETRIEVAL_K}
    )

    return RAGChain(retriever=retriever, llm_model=llm_model)
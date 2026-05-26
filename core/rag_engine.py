import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


# ==========================================
# LLM
# ==========================================

def get_llm():

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


# ==========================================
# FORMAT RETRIEVED DOCS
# ==========================================

def format_docs(docs):

    if not docs:
        return "No relevant transcript found."

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# ==========================================
# PROMPT TEMPLATE
# ==========================================

PROMPT_TEMPLATE = """
You are an expert AI meeting assistant.

Your task is to answer the user's question using ONLY the meeting transcript context.

Instructions:
1. Answer clearly and accurately.
2. Keep answers concise.
3. Mention speaker names if available.
4. If exact information is not available, try to infer from nearby context.
5. If nothing relevant exists, say:
"I could not find this information in the meeting transcript."

Meeting Transcript Context:
{context}

User Question:
{question}

Answer:
"""


# ==========================================
# BUILD RAG CHAIN
# ==========================================

def build_rag_chain(transcript: str):

    try:

        print("\nBuilding Vector Store...")

        vector_store = build_vector_store(transcript)

        print("Vector Store Created Successfully")

        retriever = get_retriever(
            vector_store=vector_store,
            k=6,
        )

        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )

        rag_chain = (

            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }

            | prompt
            | llm
            | StrOutputParser()

        )

        print("RAG Chain Ready")

        return rag_chain

    except Exception as e:

        print(f"ERROR IN build_rag_chain: {e}")

        return None


# ==========================================
# LOAD EXISTING RAG CHAIN
# ==========================================

def load_rag_chain():

    try:

        print("\nLoading Existing Vector Store...")

        vector_store = load_vector_store()

        retriever = get_retriever(
            vector_store=vector_store,
            k=6,
        )

        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )

        rag_chain = (

            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }

            | prompt
            | llm
            | StrOutputParser()

        )

        print("RAG Chain Loaded Successfully")

        return rag_chain

    except Exception as e:

        print(f"ERROR IN load_rag_chain: {e}")

        return None


# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(rag_chain, question: str) -> str:

    try:

        print("\n" + "=" * 50)
        print(f"QUESTION: {question}")
        print("=" * 50)

        answer = rag_chain.invoke(question)

        print("\nANSWER:")
        print(answer)

        return answer

    except Exception as e:

        print(f"\nERROR IN ask_question: {e}")

        return "Error while generating response."
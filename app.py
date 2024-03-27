import argparse
import os
import load_vector_db
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from getpass import getpass

CHROMA_PATH = "chroma"
KEY_PATH = "openaikey.txt"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def setup_database():
    if not os.path.exists(CHROMA_PATH):
        load_vector_db.main

def load_api_key():
    if os.path.exists(KEY_PATH) and os.path.isfile(KEY_PATH):
        os.environ["OPENAI_API_KEY"] = open(KEY_PATH, 'r').read()
    else:
        os.environ["OPENAI_API_KEY"] = getpass()

def query_db(query_text):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    return db.similarity_search_with_relevance_scores(query_text, k=3)


def construct_prompts(query_text, query_results):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in query_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, question=query_text)

def main():
    load_api_key()
    setup_database()
    
    query_text = input("What would you like to know? ")
    query_results = query_db(query_text)

    if len(query_results) == 0 or query_results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    prompt = construct_prompts(query_text, query_results)
    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in query_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
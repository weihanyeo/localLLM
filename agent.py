from accelerate import Accelerator
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import os

accelerator = Accelerator()

def create_agent(vectorstore):
    
    config = {
        'max_new_tokens': 5120, 
        'temperature': 0.2, 
        'context_length': 51200,
        'threads': os.cpu_count()
    }

    llm = CTransformers(
        model="./pretrainedLLM.bin",
        model_type="llama",
        config=config
    )

    # Prepare llm to accelerate
    llm, config = accelerator.prepare(llm, config)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Determine the most appropriate answer and always refer and revisit the given documents.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    qa_llm = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_llm

def query_agent(agent, query):
    response = agent({"query": query})
    
    # Extract the answer and sources
    answer = response['result']
    sources = response['source_documents']

    # Format the response
    formatted_response = f" {answer}\n\nSources:\n"
    for i, doc in enumerate(sources, 1):
        file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        formatted_response += f"{i}. Local File: {file_name}\n"
        formatted_response += f"   Content: {doc.page_content[:750]}...\n\n"

    return formatted_response
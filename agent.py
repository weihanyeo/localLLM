from accelerate import Accelerator
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks import AsyncIteratorCallbackHandler
import os

accelerator = Accelerator()

def create_agent(vectorstore):
    # callback_handler = AsyncIteratorCallbackHandler()
    
    config = {
        'max_new_tokens': 4060, 
        'temperature': 0.05, 
        'context_length': 20000,
        'threads': os.cpu_count()
    }

    llm = CTransformers(
        model="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        model_file="capybarahermes-2.5-mistral-7b.Q5_K_M.gguf",
        model_type="llama",
        config=config,
        # callbacks=[callback_handler],
        # streaming=True
    )
    # TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF
    # capybarahermes-2.5-mistral-7b.Q6_K.gguf
    # model="TheBloke/Llama-2-7B-Chat-GGML",
    # model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",

    # Prepare llm to accelerate
    llm, config = accelerator.prepare(llm, config)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    qa_llm = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_llm

def query_agent(agent, query):

    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Determine the most appropriate answer and always refer and revisit the given documents.

    Context: {context}
    Question: {query}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'query'])

    response = agent(prompt)
    
    # Extract the answer and sources
    answer = response['result']
    sources = response['source_documents']

    # Format the response
    formatted_response = f"Answer: {answer}\n\nSources:\n"
    for i, doc in enumerate(sources, 1):
        formatted_response += f"{i}. {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:500]}...\n"

    return formatted_response
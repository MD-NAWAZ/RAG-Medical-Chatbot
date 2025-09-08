from langchain.chains import RetrievalQA 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 


from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger = get_logger(__name__)

contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just formulate it if needed otherwise just return it as it is."
        )

contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )


# Commenting as of now.
# custom_prompt_template = """You are a helpful and medically knowledgeable assistant. Based on the following retrieved documents, answer the user’s medical question in a clear, accurate, and safe way. Do not diagnose or suggest specific treatments unless explicitly stated in the retrieved content. If the information is insufficient, advise the user to consult a healthcare professional.

# Context:
# {context}

# Question: 
# {question}

# Answer in simple language unless the user asks for medical jargon. Always include a disclaimer at the end.:
# """

# def set_custom_prompt():
#     # commenting as of now.
#     # return PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
#     return ChatPromptTemplate.from_messages([
#     SystemMessage(
#         content="You are a helpful Medical assistant."
#     ),
#     MessagesPlaceholder(
#         variable_name="chat_history"
#     ),
#     HumanMessagePromptTemplate.from_template("{input}"),
# ])

def create_qa_chain():
    try:
        logger.info("Loading Vector store for Context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vectorstore not present or empty")

        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,hf_token=HF_TOKEN)

        if llm is None:
            raise CustomException("LLM not loaded.")

# Original but commenting as of now       
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever = db.as_retriever(search_kwargs = {"k":1}),
        #     return_source_documents = False,
        #     chain_type_kwargs = {"prompt":set_custom_prompt()}
        # )
        retriever = db.as_retriever(search_kwargs = {"k":1})
        history_aware_retriever = create_history_aware_retriever(llm,retriever, contextualize_q_prompt )

        system_prompt = (
            "You are an medical assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the" 
            "answer concise."
                "\n\n"
                "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

# Commenting as of now.

        # conversational_rag_chain = RunnableWithMessageHistory(
        #     rag_chain,get_session_history,
        #     input_message_key="input",
        #     history_messages_key="chat_history",
        #     output_messages_key="answer"
        # )
        logger.info(rag_chain)
        logger.info("Successfully created the QA Chain ")
        

        return rag_chain
    
    except Exception as e:
        error_message = CustomException("failed to make a QA Chain",e)
        logger.error(str(error_message))
        return None
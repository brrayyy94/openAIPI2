from dotenv import load_dotenv
load_dotenv()
import os

from flask import Flask, jsonify, request

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders.mongodb import MongodbLoader
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pymongo import MongoClient

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

class CustomMongodbLoader(BaseLoader):
    """Cargador personalizado para cargar documentos desde MongoDB."""

    def __init__(self, connection_string: str, db_name: str, collection_name: str, projection: dict) -> None:
        """Inicializa el cargador con los detalles de la base de datos MongoDB.

        Args:
            connection_string: Cadena de conexión a MongoDB.
            db_name: Nombre de la base de datos.
            collection_name: Nombre de la colección en MongoDB.
            projection: Proyección de los campos a incluir en la consulta.
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.projection = projection

    def lazy_load(self) -> Iterator[Document]:
        """Carga los documentos de MongoDB aplicando la proyección y los entrega uno a uno."""
        for doc in self.collection.find({}, self.projection):
            # Convierte el documento de MongoDB a un documento de LangChain
            yield Document(
                page_content=str(doc),  # Convierte el contenido del documento a una cadena
                metadata={"source": "MongoDB"}
            )
            
def get_documents_from_bd():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    db = os.getenv("DB_NAME")
    connection_string = f"mongodb+srv://{user}:{password}@{host}/{db}?retryWrites=true&w=majority"
    # loaderUsers = CustomMongodbLoader(
    # connection_string=connection_string,
    # db_name="ProyectosU",
    # collection_name="users",
    # projection={
    #     "userName": 1,
    #     "email": 1,
    #     "phone": 1,
    #     "apartment": 1,
    #     "pets.name": 1,
    #     "pets.type": 1,
    #     "vehicles.model": 1,
    #     "vehicles.plate": 1,
    # }
    # )
    loaderUsers = MongodbLoader(
    connection_string=connection_string,
    db_name="ProyectosU",
    collection_name="users",
    field_names=["userName", "email", "phone", "apartment", "pets", "vehicles"],
    )
    # docsUsers = list(loaderUsers.lazy_load())
    docsUsers = loaderUsers.load()

    loaderAnnouncements = MongodbLoader(
    connection_string=connection_string,
    db_name="ProyectosU",
    collection_name="announcements",
    field_names=["Title", "Body", "category", "CreatedBy"],
    )
    docsAnnouncements = loaderAnnouncements.load()
    
    loaderComplexes = CustomMongodbLoader(
    connection_string=connection_string,
    db_name="ProyectosU",
    collection_name="complexes",
    projection={
        "name": 1,
        "address": 1,
        "emergencyNumbers.name": 1,
        "emergencyNumbers.number": 1,
        "zones.name": 1,
        "zones.description": 1,
        "zones.availableHours": 1,
    }
    )
    # Carga los documentos utilizando el método lazy_load
    docsComplexes = list(loaderComplexes.lazy_load())
    
    loaderAnnouncements = MongodbLoader(
    connection_string=connection_string,
    db_name="ProyectosU",
    collection_name="directories",
    field_names=["service", "phone", "whatsAppNumber"],
    )
    docsDirectories = loaderAnnouncements.load()

    loaderAnnouncements = MongodbLoader(
    connection_string=connection_string,
    db_name="ProyectosU",
    collection_name="pqrs",
    field_names=["userName", "case", "description", "category", "state"],
    )
    docsPqrs = loaderAnnouncements.load()

    allDocs = docsUsers + docsAnnouncements + docsComplexes + docsDirectories + docsPqrs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(allDocs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the user's questions based on the context: {context} 
        Never mention that the information comes from a context
        If you can't find the information in the context, reply "I don't have that information.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"] + " ¿Algo más en lo que te pueda ayudar?"

app=Flask(__name__)

@app.route('/')
def root():
    return 'Hello World!'

@app.route('/ia/input', methods=['POST'])
def createInput():
    docs = get_documents_from_bd()
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = []
    user_input = request.json['input']

    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    res = "Assistant:" + response
    return jsonify(res), 200

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask,render_template,request,session,redirect,url_for
from app.components.retriever import create_qa_chain
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from dotenv import load_dotenv
import os

load_dotenv()

logger = get_logger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup

def nl2br(value):
    return Markup(value.replace("\n" , "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/", methods=["GET","POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        
        user_input = request.form.get("prompt")
        logger.info(f"User Input is {user_input}")

        if user_input:
            messages = session["messages"]
            chat_history = session["chat_history"]
            messages.append({"role":"user","content":user_input})
            
            logger.info(messages)
            session["messages"] = messages

            try:
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({"input":user_input,"chat_history":chat_history,"context":""})
                logger.info(response.get("answer"))
                result = response.get("answer")
                # logger.info("Result",result)

                messages.append({"role":"assistant","content": result})
                chat_history.append({"role":"assistant","content": result})
                logger.info(messages)
                session["messages"] = messages
                session["chat_history"]

            except Exception as e:
                error_message = f"error : {str(e)}"
                return render_template("index.html", messages = session["messages"], error = error_message)
            
        return redirect(url_for("index"))
    
    return render_template("index.html",messages = session.get("messages",[]))

@app.route("/clear")
def clear():
    session.pop("messages",None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000, debug = False, use_reloader = False)


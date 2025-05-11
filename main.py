import gradio as gr
import ollama

def add_message(user_message, chat_history):
    print(f"[add_message] Received user_message: {user_message}")
    print(f"[add_message] Received chat_history: {chat_history}")
    if chat_history is None:
        chat_history = []
        print("[add_message] chat_history was None, initialized to empty list.")
    chat_history.append([user_message, None])
    print(f"[add_message] Updated chat_history: {chat_history}")
    return "", chat_history

def respond(chat_history):
    print(f"[respond] Received chat_history: {chat_history}")
    if not chat_history or chat_history[-1][1] is not None:
        print("[respond] No user message to respond to, yielding chat_history as is.")
        yield chat_history
        return

    user_message = chat_history[-1][0]
    print(f"[respond] Sending to Ollama: {user_message}")

    try:
        stream = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": user_message}],
            stream=True
        )
        response = ""
        for chunk in stream:
            print(f"[respond] Ollama chunk: {chunk}")
            response += chunk["message"]["content"]
            chat_history[-1][1] = response
            print(f"[respond] Updated chat_history: {chat_history}")
            yield chat_history

        print("[respond] Ollama streaming complete.")

    except Exception as e:
        print(f"[respond] Exception from Ollama: {e}")
        chat_history[-1][1] = f"Error: {e}"
        yield chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=300, label="Ollama Chatbot")
    msg = gr.Textbox(show_label=False, placeholder="Type your message and press Enter")

    msg.submit(add_message, [msg, chatbot], [msg, chatbot]).then(
        respond, chatbot, chatbot
    )

    demo.queue()
    demo.launch()

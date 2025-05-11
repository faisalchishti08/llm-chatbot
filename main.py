import gradio as gr
import ollama

def chat_with_deepseek(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    response = ""
    for chunk in ollama.chat(
        model="deepseek-r1:7b",
        messages=messages,
        stream=True
    ):
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            content = content.replace("<think>", "Thinking...").replace("</think>", "\n\n")
            response += content
            yield response

chat_ui = gr.ChatInterface(
    chat_with_deepseek,
    chatbot=gr.Chatbot(height=400, label="Ollama Deepseek Chatbot"),
    textbox=gr.Textbox(placeholder="Type your message and press Enter", container=False),
    title="Deepseek Chatbot (Ollama)",
    description="Chat with Deepseek-R1 via Ollama. Responses are streamed in real-time."
)

chat_ui.queue()  # Enable queue for streaming/yield
chat_ui.launch()

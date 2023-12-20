import gradio as gr

# Gradio application setup
def create_demo():
    with gr.Blocks(title= "RAG Chatbot Q&A",
        theme = "Soft"
        ) as demo:
        with gr.Column():
            with gr.Row():
                chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=680)
                show_img = gr.Image(label='Overview', height=680)

        with gr.Row():
            with gr.Column(scale=0.60):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type here to ask your PDF",
                container=False)

            with gr.Column(scale=0.20):
                submit_button = gr.Button('Send')

            with gr.Column(scale=0.20):
                uploaded_pdf = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
                

        return demo, chat_history, show_img, text_input, submit_button, uploaded_pdf

if __name__ == '__main__':
    demo, chatbot, show_img, text_input, submit_button, uploaded_pdf = create_demo()
    demo.queue()
    demo.launch()

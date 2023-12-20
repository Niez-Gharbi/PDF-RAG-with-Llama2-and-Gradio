from interface import create_demo
from pdfchatbot import PDFChatBot

# Create Gradio interface
demo, chat_history, show_img, txt, submit_button, uploaded_pdf = create_demo()

# Create PDFChatBot instance
pdf_chatbot = PDFChatBot()

# Set up event handlers
with demo:
    # Event handler for uploading a PDF
    uploaded_pdf.upload(pdf_chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img])

    # Event handler for submitting text and generating response
    submit_button.click(pdf_chatbot.add_text, inputs=[chat_history, txt], outputs=[chat_history], queue=False).\
        success(pdf_chatbot.generate_response, inputs=[chat_history, txt, uploaded_pdf], outputs=[chat_history, txt]).\
        success(pdf_chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img])

if __name__ == "__main__":
    demo.launch()
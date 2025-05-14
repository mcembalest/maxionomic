from gpt4all import GPT4All
import gradio as gr
import json
import requests
import os

# setup Nomic Atlas parameters
atlas_dataset_name = "nomic/example-text-dataset-news"
atlas_dataset_id = "4576c3a3-773d-4ba1-bf51-ff60734e4e00"

# setup LLM parameters
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(model_name)
system_prompt = "You are a helpful assistant. Use the following context to answer the user's question:"

from nomic import AtlasDataset
df = AtlasDataset(atlas_dataset_name).maps[0].data.df

def retrieve(query, top_k=5):
    """Uses the Nomic Atlas API to retrieve data most similar to the query"""
    rag_request_payload = {
        "projection_id": atlas_dataset_id,
        "k": top_k,
        "query": query,
        "selection": {
            "polarity": True,
            "method": "composition",
            "conjunctor": "ALL",
            "filters": [{"method": "search", "query": " ", "field": "text"}]
        }
    }
    
    rag_response = requests.post(
        "https://api-atlas.nomic.ai/v1/query/topk", 
        data=json.dumps(rag_request_payload), 
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['NOMIC_API_KEY']}"}
    )
    
    results = rag_response.json()
    formatted_results = ""
    for idx, data_id in enumerate(results['data'], 1):
        matching_rows = df[df['id_'] == data_id['id_']]
        for _, row in matching_rows.iterrows():
            formatted_results += f"Result {idx} (Atlas ID: {data_id}):\n{row.text}\n\n"
    return formatted_results

with gr.Blocks() as demo:
    gr.Markdown("# RAG using the Atlas API for retrieval and the GPT4ALL Python SDK for generation")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(f"### Response from {model_name}")
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox()
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gr.Markdown("### Retrieved data from Atlas")
            context_display = gr.Textbox(
                label="Latest retrieved data from Atlas",
                interactive=False,
                lines=20
            )

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    def get_context(history: list):
        last_user_message = history[-1]['content']
        context = retrieve(last_user_message)
        return context
    
    def bot(history: list, context: str):
        formatted_messages = [
            {
                'role': 'system', 
                'content': f"{system_prompt}\n\n" + context
            }
        ]

        for msg in history:
            formatted_messages.append({'role': msg['role'], 'content': msg['content']})

        full_prompt = "\n".join([m['content'] for m in formatted_messages])
        
        history.append({"role": "assistant", "content": ""})
        with model.chat_session():
            response = model.generate(
                full_prompt,
                max_tokens=1024,
                streaming=True
            )
            for chunk in response:
                history[-1]['content'] += chunk
                yield history

    msg.submit(
    user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )

    submit.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )
    clear.click(lambda: (None, ""), None, [chatbot, context_display], queue=False)

demo.launch()
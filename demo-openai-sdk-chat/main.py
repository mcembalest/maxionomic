from datetime import datetime
from nomic import AtlasDataset
import uuid
import tiktoken
try:
    dataset = AtlasDataset(
        identifier="chat-demo"
    )
except ValueError:
    print("* creating new AtlasDataset *")
    dataset = AtlasDataset(
        identifier="chat-demo", 
        unique_id_field="id", 
        description="logging chat application", 
        is_public=False
    )
print("atlas dataset loaded")

from openai import OpenAI
client = OpenAI()
model = "gpt-4o"

tokenizer = tiktoken.encoding_for_model(model)

# dollars per million tokens
openai_cost = {
    "in" : 2.5,
    "out" : 10.0
}

messages = [
    {"role": "developer", "content": "You are a helpful AI assistant."},
]

while True:
    user_message = input(" > chat or type 'q' to quit > ")
    if user_message == 'q':
        dataset.create_index(
            f"chat-{datetime.now()}",
            indexed_field="content",
        )
        break
    else:
        messages.append({"role": "user", "content": user_message})
        new_uuid = uuid.uuid4()
        user_tokens = tokenizer.encode(user_message)
        user_cost = len(user_tokens) * openai_cost["in"] / 1000000
        dataset.add_data(
            data=[{
                "id": str(new_uuid),
                "role": "user",
                "content": user_message,
                "num_tokens": len(user_tokens),
                "cost": user_cost
            }]
        )
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        model_response = ""
        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                model_response += chunk_content
                print(chunk_content, end="", flush=True)
        messages.append({"role": "assistant", "content": model_response})
        assistant_tokens = tokenizer.encode(model_response)
        assistant_cost = len(assistant_tokens) * openai_cost["out"] / 1000000
        # Atlas IDs can only have up to 36 characters, so we remove some characters from the openai completion ID
        id_truncated = chunk.id.replace("chatcmpl-", "")
        dataset.add_data(
            data=[{
                "id": id_truncated,
                "role": "assistant",
                "content": model_response,
                "num_tokens": len(assistant_tokens),
                "cost": user_cost
            }]
        )
        print("\n******************")
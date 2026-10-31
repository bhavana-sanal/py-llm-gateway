"""Talk to llm-gateway with the official OpenAI SDK.

Start the gateway first:  llm-gateway --config gateway.example.toml
Then run:                 python examples/openai_sdk.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-gateway-dev")

print("models:", [m.id for m in client.models.list()])

reply = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "Hello from the OpenAI SDK"}],
    temperature=0,
)
print("reply:", reply.choices[0].message.content)
print("usage:", reply.usage)

print("stream: ", end="")
stream = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "Stream this back"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

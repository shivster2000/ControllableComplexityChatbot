from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from controllable_dialogpt import ControllableDialoGPT

from generation_utils import Reranker, Wordlist, cefr_to_int, load_wordlist

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
vocab = load_wordlist("data/sample_wordlist.txt")

# Replace agent_opt JSON with dict config
config = {
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "top_k": 40,
    "top_p": 0.95,
    "num_return_sequences": 20,  # for reranking
    "reranker": Reranker(
        model="complexity_model",
        tokenizer="distilroberta-base",
        device="cuda",
        cefr=3
    ),   
    "wordlist": Wordlist(vocab, tokenizer)

}

agent = ControllableDialoGPT(config)

print("Start chatting (type 'exit' to stop):")
while True:
    user_input = input(">> User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.generate_response(user_input)
    print(f"🤖 Bot: {response}")
agent.set_interactive_mode(True)

import gradio as gr
import os
import json
from parlai.core.opt import Opt
from parlai.zoo.blender.blender_3B import download
from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import DialogPartnerWorld
from controllable_blender import ControllableBlender
from huggingface_hub import snapshot_download

snapshot_download(repo_id="shivansarora/ControllableBlender", local_dir="ParlAI/data/models/blender/blender_3B")

# Load options
agent_opt = json.load(open("blender_3B.opt", 'r'))
download(agent_opt["datapath"])
conversation_state = {"world": None, "human_agent": None}

class GradioHumanAgent(Agent):
    def __init__(self, opt):
        super().__init__(opt)
        self.msg = None

    def observe(self, msg):
        return msg

    def act(self):
        return {"text": self.msg, "episode_done": False}


def init_world(cefr, inference_type):
    opt = agent_opt.copy()
    opt["rerank_cefr"] = cefr
    opt["inference"] = inference_type

    # Settings for rerank methods (not used if "inference" == "vocab")
    opt["rerank_tokenizer"] = "distilroberta-base"        # Tokenizer from Huggingface Transformers. Must be compatible with "rerank_model"
    opt["rerank_model"] = "complexity_model"              # Model fine-tuned on complexity data
    opt["rerank_model_device"] = "cuda"                   # Device for complexity model
    opt["penalty_stddev"] = 2                             # Controls how harshly sub-tokens are penalised (lower = harsher). Use -1 to remove penalties
    opt["filter_path"] = "data/filter.txt"                # Path to list of English words to ensure OOV words are not generated. Capitalised words are ignored. Use empty string to remove filter

    # Settings for vocab methods (not used if "inference" == "rerank")
    opt["wordlist_path"] = "data/sample_wordlist.txt"     # Path to list of vocab the chatbot is restricted to    
   
    # Same top-k sampling configs for all settings described in the paper
    opt["beam_size"] = 20
    opt["topk"] = 40

    human_agent = GradioHumanAgent(opt)
    model_agent = ControllableBlender(opt)
    world = DialogPartnerWorld(opt, [human_agent, model_agent])

    return human_agent, world

def chat(user_input, cefr, inference_type, history):
    if conversation_state["world"] is None:

        human_agent, world = init_world(cefr, inference_type)
        conversation_state["world"] = world
        conversation_state["human_agent"] = human_agent

    conversation_state["human_agent"].msg = user_input

    conversation_state["world"].parley()

    bot_reply = conversation_state["world"].acts[1].get("text", "")
    history.append([user_input, bot_reply.strip()])
    return history, history

def reset_chat():
    conversation_state["world"] = None
    conversation_state["human_agent"] = None
    return []

with gr.Blocks() as demo:
    cefr = gr.Dropdown(["A1", "A2", "B1", "B2", "C1", "C2"], label="CEFR", value="B2")
    inference_type = gr.Dropdown(["rerank", "vocab"], label="Inference", value="rerank")
    user_input = gr.Textbox(label="your message")
    chatbot = gr.Chatbot(label="Controllable Complexity Chatbot")
    send_btn = gr.Button("Send")

    state = gr.State([])

    def user_chat(message, cefr_level, infer_type, history):
        # call your chat function here
        new_history, _ = chat(message, cefr_level, infer_type, history)
        return new_history, new_history

    send_btn.click(
        fn=user_chat,
        inputs=[user_input, cefr, inference_type, state],
        outputs=[chatbot, state]
    )

demo.launch(share=True)

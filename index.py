from typing import Tuple
import json

from flask import Flask, abort, jsonify, request
from flask_cors import CORS
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

MAX_LENGTH = int(10000)  # max length for generated text to avoid infinte loops
MODEL_CLASSES = {"gpt2": (GPT2LMHeadModel, GPT2Tokenizer)}

with open("config.json") as json_file:
    model_config = json.load(json_file)["model"]


def set_seed(args):
    seed = model_config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_config["n_gpu"] > 0:
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length: int, max_sequence_length: int) -> int:
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def make_model() -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    model_config["device"] = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model_config["n_gpu"] = torch.cuda.device_count()

    set_seed(model_config["seed"])

    # Initialize the model and tokenizer
    try:
        model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained("gpt2")
    model = model_class.from_pretrained("gpt2")
    model.to(model_config["device"])

    model_config["length"] = adjust_length_to_model(
        model_config["length"], max_sequence_length=model.config.max_position_embeddings
    )

    return model, tokenizer


@app.route("/generate", methods=["POST"])
def get_generated_text():
    data = request.get_json()
    print(data)
    user_config = {}
    if "context" not in data or len(data["context"]) == 0:
        abort(400)
    else:
        context = data["context"]

    if "config" in data:
        user_config = data["config"]

    encoded_prompt = tokenizer.encode(
        context, add_special_tokens=False, return_tensors="pt"
    )
    encoded_prompt = encoded_prompt.to(model_config["device"])
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=user_config.get("length", model_config["length"]),
        temperature=user_config.get("temperature", model_config["temperature"]),
        top_k=user_config.get("k", model_config["k"]),
        top_p=user_config.get("p", model_config["p"]),
        repetition_penalty=user_config.get(
            "repetition_penalty", model_config["repetition_penalty"]
        ),
        do_sample=True,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(user_config.get("stop_token", model_config["stop_token"]))]

    return jsonify({"text": text})


model, tokenizer = make_model()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, host="0.0.0.0")

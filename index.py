from flask import Flask, abort, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
import argparse
import torch
from transformers import (
  GPT2LMHeadModel,
  GPT2Tokenizer
)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

MAX_LENGTH = int(10000)

MODEL_CLASSES = {
"gpt2": (GPT2LMHeadModel, GPT2Tokenizer)
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def make_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")


    tokenizer = tokenizer_class.from_pretrained("gpt2")
    model = model_class.from_pretrained("gpt2")
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

    return tokenizer, model, args.device

@app.route("/generate", methods=['POST'])
def get_generated_text():
    data = request.get_json()
    print(data)

    if 'context' not in data or len(data['context']) == 0:
        abort(400)
    else:
        context = data['context']
    encoded_prompt = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=200,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    # text = text[: text.find('.') if args.stop_token else None]

    # print(text)

    return jsonify({"text": text})

tokenizer, model, device = make_model()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host="0.0.0.0")
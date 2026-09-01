import os
import argparse
import torch
import onnx
from transformers import VitsModel, VitsTokenizer
from wrappers import SherpaVitsWrapper

def export_hf_vits_to_sherpa(model_id: str, output_dir: str):
    # Ensure target path structure exists safely
    os.makedirs(output_dir, exist_ok=True)
    
    onnx_filename = os.path.join(output_dir, "model.onnx")
    tokens_filename = os.path.join(output_dir, "tokens.txt")

    print(f"Loading model and tokenizer from Hugging Face: {model_id}...")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model.eval()

    num_speakers = getattr(model.config, "num_speakers", 1)
    is_multi_speaker = num_speakers > 1
    print(f"Detected speaker configuration count: {num_speakers} (Multi-Speaker: {is_multi_speaker})")

    wrapper = SherpaVitsWrapper(model, is_multi_speaker=is_multi_speaker)

    # Setup dummy tensors for graph tracing
    dummy_x = torch.randint(0, len(tokenizer), (1, 10), dtype=torch.long)
    dummy_x_lengths = torch.tensor([10], dtype=torch.long)
    dummy_noise_scale = torch.tensor([0.667], dtype=torch.float32)
    dummy_length_scale = torch.tensor([1.0], dtype=torch.float32)
    dummy_noise_scale_w = torch.tensor([0.8], dtype=torch.float32)

    if is_multi_speaker:
        dummy_sid = torch.tensor([0], dtype=torch.long)
        dummy_inputs = (dummy_x, dummy_x_lengths, dummy_noise_scale, dummy_length_scale, dummy_noise_scale_w, dummy_sid)
        input_names = ["x", "x_lengths", "noise_scale", "length_scale", "noise_scale_w", "sid"]
    else:
        dummy_inputs = (dummy_x, dummy_x_lengths, dummy_noise_scale, dummy_length_scale, dummy_noise_scale_w)
        input_names = ["x", "x_lengths", "noise_scale", "length_scale", "noise_scale_w"]

    output_names = ["y"]

    dynamic_axes = {
        "x": {0: "batch_size", 1: "seq_len"},
        "x_lengths": {0: "batch_size"},
        "y": {0: "batch_size", 1: "num_samples"}
    }
    if is_multi_speaker:
        dynamic_axes["sid"] = {0: "batch_size"}

    print(f"Tracing execution graph and exporting to ONNX (Opset 17)...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_filename,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Injecting runtime metadata fields into ONNX graph headers...")
    onnx_model = onnx.load(onnx_filename)
    
    add_blank = str(int(getattr(model.config, "add_blank", True)))
    sample_rate = str(model.config.sampling_rate)

    metadata = {
        "model_type": "vits",
        "comment": "vits-bam",
        "frontend": "characters",
        "language": "bam",
        "has_espeak": "0",
        "sample_rate": sample_rate,
        "n_speakers": str(num_speakers),
        "add_blank": add_blank
    }

    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(onnx_model, onnx_filename)
    print(f"Successfully wrote optimized graph to: {onnx_filename}")

    print("Writing structural tokens.txt file map...")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    with open(tokens_filename, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            if token == "<unk>":
                continue
            f.write(f"{token} {idx}\n")
            
    print(f"Successfully generated: {tokens_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hugging Face VITS checkpoints cleanly to sherpa-onnx formats.")
    
    parser.add_argument(
        "--model-id", "-m",
        type=str,
        default="RobotsMali/bam-vits-fintech",
        help="The Hugging Face repo path identifier or local checkpoint folder route (Default: RobotsMali/bam-vits-fintech)."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="assets/vits",
        help="Target output directory path where model.onnx and tokens.txt will be saved (Default: assets/vits)."
    )

    args = parser.parse_args()
    export_hf_vits_to_sherpa(model_id=args.model_id, output_dir=args.output_dir)

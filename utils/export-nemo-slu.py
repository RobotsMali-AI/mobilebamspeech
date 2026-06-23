import os
import argparse
import torch
import nemo.collections.asr as nemo_asr

def export_nemo_slu_to_onnx(model_id: str, output_dir: str):
    # Ensure target path structure exists safely
    os.makedirs(output_dir, exist_ok=True)

    # Resolve all 4 target file names dynamically
    encoder_path = os.path.join(output_dir, "slu_encoder.onnx")
    classifier_path = os.path.join(output_dir, "slu_classifier.onnx")
    embedding_path = os.path.join(output_dir, "slu_embedding.onnx")
    decoder_path = os.path.join(output_dir, "slu_decoder.onnx")

    print(f"Loading NeMo SLU Intent-Slot model from: {model_id}...")
    model = nemo_asr.models.SLUIntentSlotBPEModel.from_pretrained(model_id)
    model.eval()

    # ---------------------------------------------------------
    # Native NeMo Exports: Encoder & Classifier
    # ---------------------------------------------------------
    print(f"Exporting NeMo Native Encoder block to: {encoder_path}...")
    model.encoder.export(encoder_path)

    print(f"Exporting NeMo Native Classifier block to: {classifier_path}...")
    model.classifier.export(classifier_path)

    # Re-verify evaluation flags for direct submodule tracing passes
    model.eval()

    # Extract sub-modules
    embedding_module = model.embedding
    decoder_module = model.decoder

    # Determine hidden size configurations from graph state
    hidden_size = model.cfg.decoder.hidden_size
    device = next(model.parameters()).device

    # ---------------------------------------------------------
    # Export 1: TransformerEmbedding (Manual Torch Pass)
    # ---------------------------------------------------------
    print(f"Exporting Embedding submodule to: {embedding_path}...")
    dummy_input_ids = torch.zeros((1, 5), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros_like(dummy_input_ids)
    dummy_start_pos = torch.tensor(0, dtype=torch.long, device=device)

    torch.onnx.export(
        embedding_module,
        args=(dummy_input_ids, dummy_token_type_ids, dummy_start_pos),
        f=embedding_path,
        input_names=["input_ids", "token_type_ids", "start_pos"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "token_type_ids": {0: "batch_size", 1: "seq_length"},
            "embeddings": {0: "batch_size", 1: "seq_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # ---------------------------------------------------------
    # Export 2: TransformerDecoder (Manual Torch Pass)
    # ---------------------------------------------------------
    print(f"Exporting Decoder submodule to: {decoder_path}...")
    dummy_decoder_states = torch.randn(1, 5, hidden_size, device=device)
    dummy_decoder_mask = torch.ones(1, 5, dtype=torch.float32, device=device)
    dummy_encoder_states = torch.randn(1, 40, hidden_size, device=device)
    dummy_encoder_mask = torch.ones(1, 40, dtype=torch.float32, device=device)

    torch.onnx.export(
        decoder_module,
        args=(
            dummy_decoder_states,
            dummy_decoder_mask,
            dummy_encoder_states,
            dummy_encoder_mask,
        ),
        f=decoder_path,
        input_names=[
            "decoder_states",
            "decoder_mask",
            "encoder_states",
            "encoder_mask",
        ],
        output_names=["decoded_states"],
        dynamic_axes={
            "decoder_states": {0: "batch_size", 1: "target_seq_len"},
            "decoder_mask": {0: "batch_size", 1: "target_seq_len"},
            "encoder_states": {0: "batch_size", 1: "acoustic_seq_len"},
            "encoder_mask": {0: "batch_size", 1: "acoustic_seq_len"},
            "decoded_states": {0: "batch_size", 1: "target_seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"\n[SUCCESS] All SLU subcomponents compiled and exported to folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible ONNX Extraction Suite for NeMo SLU Intent-Slot Models.")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="RobotsMali/mobile-bamking-slu-intent-v0",
        help="Pretrained .nemo model string identifier or absolute local path reference (Default: RobotsMali/mobile-bamking-slu-intent-v0)."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="assets/slurp",
        help="Target destination directory for the 4 ONNX graphs (Default: assets/slurp)."
    )

    args = parser.parse_args()
    export_nemo_slu_to_onnx(model_id=args.model, output_dir=args.output_dir)
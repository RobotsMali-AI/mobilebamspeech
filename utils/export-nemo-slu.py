import os
import argparse

import numpy as np
import torch
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from wrappers import (
    CachedDecoderWrapper,
    EncoderWrapper,
    InferenceEmbeddingWrapper,
)

# ==============================================================================
# METRICS / VERIFICATION
# ==============================================================================

def _metrics(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    diff = a - b
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0

    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    cosine = float(np.dot(af, bf) / denom) if denom > 0 else 1.0

    return max_abs, mean_abs, rmse, cosine


def _print_metrics(name, a, b):
    max_abs, mean_abs, rmse, cosine = _metrics(a, b)
    print(
        f"[VERIFY] {name:<38} "
        f"max={max_abs:.8f}  mean={mean_abs:.8f}  "
        f"rmse={rmse:.8f}  cosine={cosine:.8f}"
    )
    return max_abs, mean_abs, rmse, cosine


def _assert_close(name, a, b, max_abs_threshold=1e-3):
    max_abs, _, _, cosine = _print_metrics(name, a, b)

    if max_abs > max_abs_threshold:
        raise RuntimeError(
            f"{name} verification FAILED: max abs diff {max_abs:.8f} "
            f"> threshold {max_abs_threshold:.8f}; cosine={cosine:.8f}"
        )


def _probe_cache(decoder: torch.nn.Module, hidden_size: int, device: torch.device):
    """
    Determine cache level count from the actual NeMo decoder rather than
    hard-coding num_layers + 1.
    """
    with torch.inference_mode():
        decoder_states = torch.randn(
            1, 1, hidden_size, dtype=torch.float32, device=device
        )
        decoder_mask = torch.ones(
            1, 1, dtype=torch.float32, device=device
        )
        encoder_states = torch.randn(
            1, 4, hidden_size, dtype=torch.float32, device=device
        )
        encoder_mask = torch.ones(
            1, 4, dtype=torch.float32, device=device
        )

        mems = decoder(
            decoder_states=decoder_states,
            decoder_mask=decoder_mask,
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            decoder_mems_list=None,
            return_mems=True,
            return_mems_as_list=False,
        )

    print(f"[+] Decoder cache probe: {tuple(mems.shape)}")
    print("    layout = [cache_levels, batch, cached_sequence_length, hidden_size]")

    return int(mems.shape[0])


# ==============================================================================
# EXPORT
# ==============================================================================

def export_nemo_slu_to_onnx(
    model_id: str,
    output_dir: str,
    prefix: str,
    verify_threshold: float = 1e-3,
):
    os.makedirs(output_dir, exist_ok=True)

    encoder_path = os.path.join(output_dir, f"{prefix}-encoder.onnx")
    classifier_path = os.path.join(output_dir, f"{prefix}-classifier.onnx")
    embedding_path = os.path.join(output_dir, f"{prefix}-embedding.onnx")
    decoder_path = os.path.join(output_dir, f"{prefix}-decoder.onnx")

    print(f"[+] Loading NeMo SLU model: {model_id}")
    model = nemo_asr.models.SLUIntentSlotBPEModel.from_pretrained(model_id)

    # Keeping export + verification on CPU removes device-to-device numerical
    # differences from the validation itself.
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    hidden_size = int(model.cfg.decoder.hidden_size)

    # --------------------------------------------------------------------------
    # 1. ENCODER — exact runtime forward path
    # --------------------------------------------------------------------------
    encoder_wrapper = EncoderWrapper(model.encoder).to(device).eval()

    # Mimic the real NeMo preprocessor shape observed for the test audio:
    # tensor T=313, actual valid length=312.
    torch.manual_seed(1234)
    dummy_audio_signal = torch.randn(
        1, 80, 313, dtype=torch.float32, device=device
    )
    dummy_audio_length = torch.tensor(
        [312], dtype=torch.long, device=device
    )

    with torch.inference_mode():
        pt_encoder_out, pt_encoder_len = encoder_wrapper(
            dummy_audio_signal,
            dummy_audio_length,
        )

    print(f"[+] Exporting encoder -> {encoder_path}")
    torch.onnx.export(
        encoder_wrapper,
        args=(dummy_audio_signal, dummy_audio_length),
        f=encoder_path,
        input_names=["audio_signal", "length"],
        output_names=["encoded", "encoded_length"],
        dynamic_axes={
            "audio_signal": {0: "batch_size", 2: "mel_time"},
            "length": {0: "batch_size"},
            "encoded": {0: "batch_size", 2: "encoded_time"},
            "encoded_length": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify immediately in the SAME process against the SAME tensors.
    enc_sess = ort.InferenceSession(
        encoder_path,
        providers=["CPUExecutionProvider"],
    )

    onnx_encoder_out, onnx_encoder_len = enc_sess.run(
        None,
        {
            "audio_signal": dummy_audio_signal.cpu().numpy(),
            "length": dummy_audio_length.cpu().numpy().astype(np.int64),
        },
    )

    pt_encoder_out_np = pt_encoder_out.cpu().numpy()
    pt_encoder_len_np = pt_encoder_len.cpu().numpy().astype(np.int64)

    print(
        f"[VERIFY] Encoder lengths                     "
        f"PT={pt_encoder_len_np.tolist()} "
        f"ONNX={np.asarray(onnx_encoder_len).astype(np.int64).tolist()}"
    )

    if not np.array_equal(
        pt_encoder_len_np,
        np.asarray(onnx_encoder_len).astype(np.int64),
    ):
        raise RuntimeError(
            "Encoder verification FAILED: encoded lengths differ."
        )

    # Compare the complete exported tensor. For this deterministic dummy input
    # both graphs should have identical output shapes.
    _assert_close(
        "Encoder output",
        pt_encoder_out_np,
        onnx_encoder_out,
        max_abs_threshold=verify_threshold,
    )

    # --------------------------------------------------------------------------
    # 2. CLASSIFIER
    # --------------------------------------------------------------------------
    print(f"[+] Exporting classifier -> {classifier_path}")
    model.classifier.export(classifier_path)

    model.eval()

    # --------------------------------------------------------------------------
    # 3. EMBEDDING — EXACT NeMo inference semantics (NO token_type_ids)
    # --------------------------------------------------------------------------
    embedding_wrapper = InferenceEmbeddingWrapper(
        model.embedding
    ).to(device).eval()

    dummy_input_ids = torch.tensor(
        [[1]],  # BOS
        dtype=torch.long,
        device=device,
    )
    dummy_start_pos = torch.tensor(
        0,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        pt_embedding_out = embedding_wrapper(
            dummy_input_ids,
            dummy_start_pos,
        )

    print(f"[+] Exporting inference embedding -> {embedding_path}")
    torch.onnx.export(
        embedding_wrapper,
        args=(dummy_input_ids, dummy_start_pos),
        f=embedding_path,
        input_names=["input_ids", "start_pos"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "token_seq_len"},
            "embeddings": {0: "batch_size", 1: "token_seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    emb_sess = ort.InferenceSession(
        embedding_path,
        providers=["CPUExecutionProvider"],
    )

    onnx_embedding_out = emb_sess.run(
        None,
        {
            "input_ids": dummy_input_ids.cpu().numpy(),
            "start_pos": np.array(0, dtype=np.int64),
        },
    )[0]

    _assert_close(
        "Embedding BOS / start_pos=0",
        pt_embedding_out.cpu().numpy(),
        onnx_embedding_out,
        max_abs_threshold=verify_threshold,
    )

    # Verify a nonzero positional offset too. This catches accidental constant
    # folding / incorrect handling of start_pos.
    later_input_ids = torch.tensor(
        [[13]],
        dtype=torch.long,
        device=device,
    )
    later_start_pos = torch.tensor(
        9,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        pt_embedding_later = embedding_wrapper(
            later_input_ids,
            later_start_pos,
        )

    onnx_embedding_later = emb_sess.run(
        None,
        {
            "input_ids": later_input_ids.cpu().numpy(),
            "start_pos": np.array(9, dtype=np.int64),
        },
    )[0]

    _assert_close(
        "Embedding token / start_pos=9",
        pt_embedding_later.cpu().numpy(),
        onnx_embedding_later,
        max_abs_threshold=verify_threshold,
    )

    # --------------------------------------------------------------------------
    # 4. SINGLE CACHED DECODER
    # --------------------------------------------------------------------------
    decoder = model.decoder.to(device).eval()
    cache_levels = _probe_cache(
        decoder,
        hidden_size,
        device,
    )

    decoder_wrapper = CachedDecoderWrapper(
        decoder
    ).to(device).eval()

    dummy_decoder_states = torch.randn(
        1, 1, hidden_size, dtype=torch.float32, device=device
    )
    dummy_decoder_mask = torch.ones(
        1, 1, dtype=torch.float32, device=device
    )
    dummy_encoder_states = torch.randn(
        1, 40, hidden_size, dtype=torch.float32, device=device
    )
    dummy_encoder_mask = torch.ones(
        1, 40, dtype=torch.float32, device=device
    )

    # Trace with a non-empty cache. The cache length axis is dynamic and the
    # same graph will also receive a zero-length cache for generation step 0.
    dummy_past_mems = torch.randn(
        cache_levels,
        1,
        2,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )

    print(f"[+] Exporting decoder -> {decoder_path}")
    torch.onnx.export(
        decoder_wrapper,
        args=(
            dummy_decoder_states,
            dummy_decoder_mask,
            dummy_encoder_states,
            dummy_encoder_mask,
            dummy_past_mems,
        ),
        f=decoder_path,
        input_names=[
            "decoder_states",
            "decoder_mask",
            "encoder_states",
            "encoder_mask",
            "past_mems",
        ],
        output_names=[
            "last_hidden",
            "present_mems",
        ],
        dynamic_axes={
            "decoder_states": {
                0: "batch_size",
                1: "token_seq_len",
            },
            "decoder_mask": {
                0: "batch_size",
                1: "token_seq_len",
            },
            "encoder_states": {
                0: "batch_size",
                1: "acoustic_seq_len",
            },
            "encoder_mask": {
                0: "batch_size",
                1: "acoustic_seq_len",
            },
            # [cache_levels, B, past_len, H]
            "past_mems": {
                1: "batch_size",
                2: "past_cache_len",
            },
            "last_hidden": {
                0: "batch_size",
                1: "last_hidden_seq_len",
            },
            # [cache_levels, B, present_len, H]
            "present_mems": {
                1: "batch_size",
                2: "present_cache_len",
            },
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # --------------------------------------------------------------------------
    # 5. DECODER SANITY CHECK — especially zero-length first-step cache
    # --------------------------------------------------------------------------
    dec_sess = ort.InferenceSession(
        decoder_path,
        providers=["CPUExecutionProvider"],
    )

    # First autoregressive step: empty history.
    empty_past = torch.empty(
        cache_levels,
        1,
        0,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        pt_last_empty, pt_present_empty = decoder_wrapper(
            dummy_decoder_states,
            dummy_decoder_mask,
            dummy_encoder_states,
            dummy_encoder_mask,
            empty_past,
        )

    onnx_last_empty, onnx_present_empty = dec_sess.run(
        None,
        {
            "decoder_states": dummy_decoder_states.cpu().numpy(),
            "decoder_mask": dummy_decoder_mask.cpu().numpy(),
            "encoder_states": dummy_encoder_states.cpu().numpy(),
            "encoder_mask": dummy_encoder_mask.cpu().numpy(),
            "past_mems": empty_past.cpu().numpy(),
        },
    )

    _assert_close(
        "Decoder empty-cache last_hidden",
        pt_last_empty.cpu().numpy(),
        onnx_last_empty,
        max_abs_threshold=verify_threshold,
    )
    _assert_close(
        "Decoder empty-cache present_mems",
        pt_present_empty.cpu().numpy(),
        onnx_present_empty,
        max_abs_threshold=verify_threshold,
    )

    # Subsequent autoregressive step: real cache from previous step.
    next_decoder_states = torch.randn(
        1, 1, hidden_size, dtype=torch.float32, device=device
    )

    with torch.inference_mode():
        pt_last_cached, pt_present_cached = decoder_wrapper(
            next_decoder_states,
            dummy_decoder_mask,
            dummy_encoder_states,
            dummy_encoder_mask,
            pt_present_empty,
        )

    onnx_last_cached, onnx_present_cached = dec_sess.run(
        None,
        {
            "decoder_states": next_decoder_states.cpu().numpy(),
            "decoder_mask": dummy_decoder_mask.cpu().numpy(),
            "encoder_states": dummy_encoder_states.cpu().numpy(),
            "encoder_mask": dummy_encoder_mask.cpu().numpy(),
            "past_mems": onnx_present_empty,
        },
    )

    _assert_close(
        "Decoder cached last_hidden",
        pt_last_cached.cpu().numpy(),
        onnx_last_cached,
        max_abs_threshold=verify_threshold,
    )
    _assert_close(
        "Decoder cached present_mems",
        pt_present_cached.cpu().numpy(),
        onnx_present_cached,
        max_abs_threshold=verify_threshold,
    )

    # --------------------------------------------------------------------------
    # DONE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[SUCCESS] Export + immediate verification completed.")
    print("=" * 80)
    print(f"  Encoder    : {encoder_path}")
    print(f"  Embedding  : {embedding_path}")
    print(f"  Decoder    : {decoder_path}")
    print(f"  Classifier : {classifier_path}")
    print()
    print(f"  Decoder cache levels : {cache_levels}")
    print(f"  Hidden size          : {hidden_size}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export NeMo SLU to ONNX with exact inference embedding semantics, "
            "one cached decoder, and immediate PT-vs-ONNX verification."
        )
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="RobotsMali/soloni-ic-slot-fintech-v0",
        help="NeMo/Hugging Face model identifier or local model path.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="assets/slurp",
        help="Directory for exported ONNX files.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="soloni-ic-slot-fintech-v0",
        help="Filename prefix for exported ONNX files.",
    )

    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=1e-3,
        help="Maximum allowed absolute PT-vs-ONNX difference during verification.",
    )

    args = parser.parse_args()

    export_nemo_slu_to_onnx(
        model_id=args.model,
        output_dir=args.output_dir,
        prefix=args.prefix,
        verify_threshold=args.verify_threshold,
    )

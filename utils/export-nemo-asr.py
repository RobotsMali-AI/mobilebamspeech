import argparse
import nemo.collections.asr as nemo_asr
import sys


def export_model(hf_model_id: str, out_path: str):
    # Try loading as a CTC model first (e.g., quartznet)
    try:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(hf_model_id)
        model_type = 'ctc'
    except Exception:
        # Fallback to RNNT with BPE (e.g., soloni)
        try:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(hf_model_id)
            model_type = 'rnnt-bpe'
        except Exception as e:
            print(f"Failed to load model '{hf_model_id}' as known NeMo ASR model types: {e}")
            return 1

    model.eval()

    if model_type == 'rnnt-bpe':
        # If model has an auxiliary CTC head, switch to CTC decoding for export
        if hasattr(model.cfg, 'aux_ctc') and hasattr(model.cfg.aux_ctc, 'decoding'):
            ctc_decoding_cfg = model.cfg.aux_ctc.decoding
            model.change_decoding_strategy(decoder_type='ctc', decoding_cfg=ctc_decoding_cfg)

    # Export to ONNX
    try:
        model.export(out_path)
    except Exception as e:
        print(f"Failed to export model to '{out_path}': {e}")
        return 1

    print(f"Exported {hf_model_id} ({model_type}) to {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Export NeMo ASR model to ONNX')

    parser.add_argument(
        'hf_model_id', 
        default="RobotsMali/soloni-114m-tdt-ctc-v2",
        help="The Hugging Face repo path identifier or local checkpoint folder route (Default: RobotsMali/soloni-114m-tdt-ctc-v2)."
    )
    parser.add_argument(
        'out_path', 
        default="assets/asr/soloni-ctc.onnx",
        help='Output path for exported model (Default: assets/asr/soloni-ctc.onnx).'
    )
    args = parser.parse_args()

    rc = export_model(args.hf_model_id, args.out_path)
    sys.exit(rc)


if __name__ == '__main__':
    main()
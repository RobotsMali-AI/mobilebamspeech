"""PyTorch wrappers used by the custom ONNX exporters."""

import torch.nn as nn


class EncoderWrapper(nn.Module):
    """Expose the SLU encoder's runtime forward signature for ONNX export."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length):
        encoded, encoded_len = self.encoder(
            audio_signal=audio_signal,
            length=length,
        )
        return encoded, encoded_len


class InferenceEmbeddingWrapper(nn.Module):
    """Match NeMo inference embeddings without supplying token_type_ids."""

    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, start_pos):
        return self.embedding.forward(
            input_ids,
            start_pos=start_pos,
        )


class CachedDecoderWrapper(nn.Module):
    """Expose one SLU decoder graph for empty and populated memory caches."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        decoder_states,
        decoder_mask,
        encoder_states,
        encoder_mask,
        past_mems,
    ):
        present_mems = self.decoder(
            decoder_states=decoder_states,
            decoder_mask=decoder_mask,
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            decoder_mems_list=past_mems,
            return_mems=True,
            return_mems_as_list=False,
        )
        last_hidden = present_mems[-1, :, -1:, :]
        return last_hidden, present_mems


class SherpaVitsWrapper(nn.Module):
    """Adapt Hugging Face VITS inputs to sherpa-onnx's graph contract."""

    def __init__(self, hf_model, is_multi_speaker=False):
        super().__init__()
        self.hf_model = hf_model
        self.is_multi_speaker = is_multi_speaker

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8,
        sid=None,
    ):
        forward_kwargs = {"input_ids": x}
        if self.is_multi_speaker and sid is not None:
            forward_kwargs["speaker_id"] = sid

        waveform = self.hf_model(**forward_kwargs).waveform

        # Keep the structural sherpa-onnx inputs in the traced graph.
        dummy_zero = (
            noise_scale.sum() * 0.0
            + length_scale.sum() * 0.0
            + noise_scale_w.sum() * 0.0
            + x_lengths.sum() * 0.0
        )
        return waveform + dummy_zero.to(waveform.dtype)

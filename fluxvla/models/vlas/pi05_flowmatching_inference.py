import math

import torch

from fluxvla.engines import VLAS
from .pi05_flowmatching import PI05FlowMatching


@VLAS.register_module()
class PI05FlowMatchingInference(PI05FlowMatching):
    """Inference variant of PI05FlowMatching with Triton acceleration.

    Args:
        num_views (int): Number of camera views. Default: 2.
        *args: Forwarded to :class:`PI05FlowMatching`.
        **kwargs: Forwarded to :class:`PI05FlowMatching`.
    """

    def __init__(self, num_views=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self._triton_ready = False

    def predict_action(self,
                       images,
                       lang_tokens,
                       states,
                       img_masks=None,
                       lang_masks=None,
                       past_key_values=None,
                       noise=None,
                       *args,
                       **kwargs):
        if not self._triton_ready:
            self.prepare_triton_inference(
                num_views=self.num_views,
                max_prompt_len=getattr(self, 'triton_max_prompt_len', 48),
                chunk_size=self.n_action_steps,
                num_steps=getattr(self, 'num_steps', 10))
            self._triton_ready = True

        pixel_values = images.unflatten(1, (-1, 3))[0]
        images_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous().bfloat16()

        prompt_len = (
            int(lang_masks[0].sum().item())
            if lang_masks is not None else lang_tokens.shape[1])
        lang_emb = self.llm_backbone.embed_tokens(lang_tokens[0, :prompt_len])
        lang_emb = (lang_emb * math.sqrt(lang_emb.shape[-1])).bfloat16()

        chunk_size = self.n_action_steps
        if noise is None:
            noise_t = torch.randn(
                chunk_size,
                self.max_action_dim,
                dtype=torch.bfloat16,
                device=states.device)
        else:
            noise_t = noise[0].to(dtype=torch.bfloat16)
        if noise_t.shape[-1] < 32:
            pad = torch.zeros(
                noise_t.shape[0],
                32 - noise_t.shape[-1],
                dtype=torch.bfloat16,
                device=noise_t.device)
            noise_t = torch.cat([noise_t, pad], dim=-1)

        denoised = self._pi05_inference.forward(images_nhwc, lang_emb,
                                                prompt_len, noise_t)
        result = denoised[:, :self.max_action_dim].unsqueeze(0).float()

        return result

    def _prepare_adarms_cond(self, num_steps):
        """Pre-compute sinusoidal time embeddings for each step."""
        dt = -1.0 / num_steps
        time_val = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        min_period = 4e-3
        max_period = 4.0
        embedding_dim = 1024
        fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device='cuda')
        period = min_period * (max_period / min_period)**fraction
        time_embs = []
        for _ in range(num_steps):
            sinusoid_input = (
                time_val.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 *
                math.pi)
            emb = torch.cat(
                [torch.sin(sinusoid_input),
                 torch.cos(sinusoid_input)], dim=-1)
            time_embs.append(emb.to(torch.bfloat16))
            time_val = time_val + dt
        return torch.cat(time_embs, dim=0)

    def _prepare_action_time_triton(self) -> dict:
        weights = {}
        weights.update(
            self.action_in_proj.prepare_triton('decoder_action_in_proj'))
        weights.update(
            self.action_out_proj.prepare_triton('decoder_action_out_proj'))
        weights.update(self.time_mlp_in.prepare_triton('decoder_time_mlp_in'))
        weights.update(
            self.time_mlp_out.prepare_triton('decoder_time_mlp_out'))
        return weights

    def prepare_triton_inference(self, num_views, max_prompt_len, chunk_size,
                                 num_steps):
        """Collect weights and build the Triton pipeline.

        Args:
            num_views (int): The number of views.
            max_prompt_len (int): The maximum prompt length.
            chunk_size (int): The chunk size.
            num_steps (int): Denoising steps.
        """
        weights = {}
        weights.update(self.vision_backbone.prepare_triton())
        weights.update(self.llm_backbone.prepare_triton(role='llm'))
        weights.update(self.llm_expert.prepare_triton(role='expert'))
        weights.update(
            self.projector.prepare_triton(
                prefix='encoder_multi_modal_projector'))
        weights.update(self._prepare_action_time_triton())
        weights.update(
            {'decoder_time_embeds': self._prepare_adarms_cond(num_steps)})

        from fluxvla.ops.triton.pi05_inference import PI05Inference
        self._pi05_inference = PI05Inference.from_weights(
            weights, num_views, max_prompt_len, chunk_size, num_steps)

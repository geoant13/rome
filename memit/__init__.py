"""MEMIT (Meng et al., 2023), from kmeng01/memit at commit 80426fd9316c (MIT licence, see
LICENSE in this folder), with the patches needed for Qwen3 under transformers 5, each
marked "Patched (phd-constants)" where it is made:
  - decoder layers return a tensor, not a tuple (compute_z: untuple before indexing);
  - tied input/output embeddings: the output head's weight is read from the module,
    not looked up by parameter name (compute_z);
  - hidden size from config.hidden_size when there is no n_embd (compute_z);
  - get_cov calls this fork's layer_stats, which has no force_recompute argument;
  - weight updates accumulated in fp32 and cast back (memit_main), as in rome_main.
Context templates are set by the caller (CONTEXT_TEMPLATES_CACHE): MEMIT's own
generate_fast is broken under transformers 5 (see phd-constants exp02 Stage B).
"""
from .memit_main import MEMITHyperParams, apply_memit_to_model
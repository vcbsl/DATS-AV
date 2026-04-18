from typing import Optional, Dict, Tuple, List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

# ====== Shared blocks (as before) ======
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.0, init_std: float = 0.02):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=init_std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pe = self.pe
        if T != pe.size(1):
            pe = F.interpolate(pe.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        return self.dropout(x + pe[:, :T, :])

class AudioFeatureEncoder(nn.Module):
    def __init__(self, stft_bins: int, hidden_dim: int = 128, nhead: int = 4, dropout: float = 0.1,
                 norm_inputs: bool = True, num_layers: int = 1, pe_max_len: int = 2000, pe_dropout: float = 0.0):
        super().__init__()
        self.norm_inputs = norm_inputs
        self.expected_bins = int(stft_bins)
        self.input_proj = nn.Linear(self.expected_bins, hidden_dim)
        self.pe = LearnedPositionalEncoding(hidden_dim, max_len=pe_max_len, dropout=pe_dropout)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
    def forward(self, x_stft: torch.Tensor) -> torch.Tensor:
        if x_stft.ndim != 3:
            raise ValueError(f"expected [B,F,S] or [B,S,F], got {tuple(x_stft.shape)}")
        if x_stft.shape[-1] == self.expected_bins:
            x = x_stft
        elif x_stft.shape[-2] == self.expected_bins:
            x = x_stft.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"STFT bins mismatch, expected one dim == {self.expected_bins}, got {tuple(x_stft.shape)}")
        if self.norm_inputs:
            x = torch.log1p(x)
            x = x - x.mean(dim=(1, 2), keepdim=True)
            x = x / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        x = self.input_proj(x)
        x = self.pe(x)
        x = self.encoder(x)
        return x

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ====== Visual encoder (single frame; supports [B,T,3,H,W] by picking mid T) ======
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(2048, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        if faces.ndim == 5:  # [B,T,3,H,W] -> mid frame
            T = faces.size(1)
            faces = faces[:, T // 2, ...]
        if faces.ndim != 4 or faces.size(1) != 3:
            raise ValueError(f"faces must be [B,3,H,W] or [B,T,3,H,W], got {tuple(faces.shape)}")
        x = (faces - self.mean) / self.std
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        return x

class AudioOnlyModel(nn.Module):
    def __init__(self, stft_bins: int = 201, enc_dim: int = 128, enc_heads: int = 4, enc_layers: int = 1,
                 enc_dropout: float = 0.1, norm_inputs: bool = True, audio75_dim: int = 75,
                 mlp_hidden: int = 256, cls_dropout: float = 0.3, pe_max_len: int = 2000, pe_dropout: float = 0.0):
        super().__init__()
        self.enc_dim = enc_dim
        self.audio_encoder = AudioFeatureEncoder(stft_bins, enc_dim, enc_heads, enc_dropout,
                                                 norm_inputs, enc_layers, pe_max_len, pe_dropout)
        self.audio75_proj = MLPProjector(audio75_dim, enc_dim, p_drop=0.1)
        self.classifier = nn.Sequential(nn.Linear(enc_dim * 2, mlp_hidden), nn.ReLU(True),
                                        nn.Dropout(cls_dropout), nn.Linear(mlp_hidden, 1))
    def forward(self, audio_stft: torch.Tensor, audio75: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = audio_stft.size(0); device = audio_stft.device
        seq = self.audio_encoder(audio_stft)
        pooled = seq.mean(dim=1)
        aux = self.audio75_proj(audio75) if audio75 is not None else None
        if aux is None:
            aux = torch.zeros(B, self.enc_dim, device=device, dtype=pooled.dtype)
        z = self.classifier(torch.cat([pooled, aux], dim=-1)).squeeze(-1)
        return z

RECOMMENDED_SETS_VIS = {
    "one": [39],
    "small": [39, 40, 53, 36, 65, 37, 33, 34, 0, 4],
    "bins9_11_pack": [33, 34, 36, 37, 39, 40, 51, 52, 53, 63, 64, 65, 0, 4],
}

class VisualOnlyModel(nn.Module):
    def __init__(self, *, hidden_dim: int = 256, nhead: int = 4, dropout: float = 0.3,
                 pretrained_backbone: bool = True, freeze_backbone: bool = False,
                 fusion: Literal["xattn", "gated", "concat"] = "xattn",
                 feature_set: Literal["one", "small", "bins9_11_pack"] = "bins9_11_pack",
                 feature_indices: Optional[List[int]] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion = fusion
        if feature_indices is not None and len(feature_indices) > 0:
            sel = sorted(set(int(i) for i in feature_indices))
        else:
            sel = sorted(RECOMMENDED_SETS_VIS[feature_set])
        self.register_buffer("sel_idx", torch.tensor(sel, dtype=torch.long), persistent=False)
        self.visual_in_dim = len(sel)
        self.face_enc = ImageEncoder(out_dim=hidden_dim, pretrained=pretrained_backbone, freeze_backbone=freeze_backbone)
        self.visual_proj = MLPProjector(self.visual_in_dim, hidden_dim, p_drop=0.1)
        if fusion == "concat":
            self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, 1))
        elif fusion == "gated":
            self.gate_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 256),
                                      nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, 1))
        elif fusion == "xattn":
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.xattn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True, dropout=dropout)
            self.xattn_norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, 1))
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")
    def _slice_or_pass(self, visual_feats: torch.Tensor) -> torch.Tensor:
        if visual_feats.ndim != 2:
            raise ValueError(f"visual_feats must be [B,D], got {tuple(visual_feats.shape)}")
        D = visual_feats.size(1)
        if D == 75:
            return visual_feats.index_select(dim=1, index=self.sel_idx)
        if D == self.visual_in_dim:
            return visual_feats
        raise ValueError(f"Unexpected visual dim {D}. Expected 75 or {self.visual_in_dim}.")
    def _fuse(self, face_emb: torch.Tensor, vproj: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([face_emb, vproj], dim=-1)
        if self.fusion == "gated":
            gate = self.gate_mlp(torch.cat([face_emb, vproj], dim=-1))
            return gate * face_emb + (1.0 - gate) * vproj
        q = self.q_proj(vproj).unsqueeze(1)
        k = self.k_proj(face_emb).unsqueeze(1)
        v = self.v_proj(face_emb).unsqueeze(1)
        z, _ = self.xattn(query=q, key=k, value=v)
        z = self.xattn_norm(z.squeeze(1))
        return z
    def forward(self, faces: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        face_emb = self.face_enc(faces)
        x_sel = self._slice_or_pass(visual_feats)
        vproj = self.visual_proj(x_sel)
        fused = self._fuse(face_emb, vproj)
        logits = self.head(fused).squeeze(-1)
        return logits

# ====== AV dissonance (main) ======
class DissonanceExpert(nn.Module):
    def __init__(self, vis_dim: int, aud_dim: int = 75, stft_bins: int = 201, emb_dim: int = 128, hidden: int = 256,
                 enc_heads: int = 4, enc_layers: int = 1, enc_dropout: float = 0.1,
                 pe_max_len: int = 2000, pe_dropout: float = 0.0, cls_dropout: float = 0.3):
        super().__init__()
        self.aud_proj = MLPProjector(aud_dim, emb_dim, p_drop=0.1)
        self.vis_proj = MLPProjector(vis_dim, emb_dim, p_drop=0.1)
        self.stft_enc = AudioFeatureEncoder(stft_bins=stft_bins, hidden_dim=emb_dim, nhead=enc_heads,
                                            dropout=enc_dropout, norm_inputs=True, num_layers=enc_layers,
                                            pe_max_len=pe_max_len, pe_dropout=pe_dropout)
        av_in = emb_dim * 2 + emb_dim
        self.cls = nn.Sequential(nn.Linear(av_in, hidden), nn.ReLU(True), nn.Dropout(cls_dropout), nn.Linear(hidden, 1))
        self.diss = nn.Sequential(nn.Linear(emb_dim * 2, hidden // 2), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(hidden // 2, 1))
    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
    def forward(self, x_vis: torch.Tensor, x_aud: torch.Tensor, stft: torch.Tensor) -> Dict[str, torch.Tensor]:
        a = self.aud_proj(x_aud)
        v = self.vis_proj(x_vis)
        s_seq = self.stft_enc(stft)
        s_tok = s_seq.mean(dim=1)
        a_n, v_n = self._l2n(a), self._l2n(v)
        absdiff = torch.abs(a_n - v_n)
        prod = a_n * v_n
        logits = self.cls(torch.cat([absdiff, prod, s_tok], dim=-1)).squeeze(-1)
        diss_logit = self.diss(torch.cat([absdiff, prod], dim=-1)).squeeze(-1)
        diss = torch.sigmoid(diss_logit)
        return {"logits": logits, "dissonance_logit": diss_logit, "dissonance": diss}

# ====== AUX ensemble (gate over audio/visual logits) ======
class AuxEnsembleExpert(nn.Module):
    def __init__(self, visual_only: VisualOnlyModel, audio_only: AudioOnlyModel,
                 gate_in_dim: int = 6, gate_hidden: int = 32, gate_dropout: float = 0.0):
        super().__init__()
        self.v = visual_only
        self.a = audio_only
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(True),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
    @staticmethod
    def _build_gate_feats(*, stft: torch.Tensor, x_vis: torch.Tensor, diss_prob: torch.Tensor) -> torch.Tensor:
        B = stft.size(0); F = stft.size(1)
        device = stft.device; dtype = stft.dtype
        p = diss_prob.clamp(0, 1)
        conf = (p - 0.5).abs() * 2.0
        F_norm = torch.tanh(torch.full((B,), float(F) / 300.0, device=device, dtype=dtype))
        a_energy = torch.log1p(stft).mean(dim=(1, 2))
        v_energy = x_vis.abs().mean(dim=1)
        a_std = torch.log1p(stft).std(dim=(1, 2))
        feats = torch.stack([p, conf, F_norm, a_energy, v_energy, a_std], dim=-1)
        return feats.to(device)
    def forward(self, faces: torch.Tensor, x_vis: torch.Tensor, stft: torch.Tensor, x_aud: torch.Tensor,
                diss_prob: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_v = self.v(faces, x_vis)            # visual-only logit (train with y_v)
        z_a = self.a(stft, x_aud)             # audio-only  logit (train with y_a)
        gate_feats = self._build_gate_feats(stft=stft, x_vis=x_vis, diss_prob=diss_prob)
        g = self.gate(gate_feats).squeeze(-1) # gating weight in [0,1]
        z = g * z_a + (1.0 - g) * z_v         # ensemble logit (train with y_mm)
        return {"logits": z, "z_a": z_a, "z_v": z_v, "gate": g, "gate_feats": gate_feats}

# ====== Full dual model with routing ======
class DissonanceDualModel(nn.Module):
    """
    Train:
      - AV main    : outputs['diss_logits'] vs y_mm
      - AUX gate   : outputs['aux_logits']  vs y_mm
      - Visual-only: outputs['aux_v_logits'] vs y_v
      - Audio-only : outputs['aux_a_logits'] vs y_a

    Inference:
      - Compute p_diss = σ(dissonance_logit), conf = 2|p_diss-0.5|
      - If conf < tau -> use AUX ensemble logit; else use AV main logit
    """
    def __init__(self, *, vis_dim: int, aud_dim: int = 75, stft_bins: int = 201,
                 emb_dim_audio: int = 128, emb_dim_visface: int = 256, hidden_audio: int = 256,
                 enc_heads: int = 4, enc_layers: int = 1, enc_dropout: float = 0.1,
                 pe_max_len: int = 2000, pe_dropout: float = 0.0, cls_dropout_audio: float = 0.3,
                 fusion_mode: Literal["gated", "concat", "xattn"] = "gated",
                 face_pretrained: bool = True, face_freeze_backbone: bool = False, switch_threshold: float = 0.25,
                 feature_set: Literal["one", "small", "bins9_11_pack"] = "bins9_11_pack",
                 feature_indices: Optional[List[int]] = None):
        super().__init__()
        self.switch_threshold = float(switch_threshold)
        self.diss_expert = DissonanceExpert(
            vis_dim=vis_dim, aud_dim=aud_dim, stft_bins=stft_bins, emb_dim=emb_dim_audio, hidden=hidden_audio,
            enc_heads=enc_heads, enc_layers=enc_layers, enc_dropout=enc_dropout,
            pe_max_len=pe_max_len, pe_dropout=pe_dropout, cls_dropout=cls_dropout_audio,
        )
        visual_only = VisualOnlyModel(
            hidden_dim=emb_dim_visface, nhead=enc_heads, dropout=0.3, pretrained_backbone=face_pretrained,
            freeze_backbone=face_freeze_backbone, fusion=fusion_mode, feature_set=feature_set,
            feature_indices=feature_indices,
        )
        audio_only = AudioOnlyModel(
            stft_bins=stft_bins, enc_dim=emb_dim_audio, enc_heads=enc_heads, enc_layers=enc_layers,
            enc_dropout=enc_dropout, norm_inputs=True, audio75_dim=aud_dim,
            mlp_hidden=hidden_audio, cls_dropout=cls_dropout_audio, pe_max_len=pe_max_len, pe_dropout=pe_dropout,
        )
        self.aux_expert = AuxEnsembleExpert(visual_only=visual_only, audio_only=audio_only)

    def forward(self, *, x_vis20: torch.Tensor, x_vis75: torch.Tensor, face: torch.Tensor,
                stft: torch.Tensor, x_aud: torch.Tensor, infer_switch: bool = False,
                switch_threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
        # AV main (dissonance)
        av = self.diss_expert(x_vis=x_vis20, x_aud=x_aud, stft=stft)
        p_diss = torch.sigmoid(av["dissonance_logit"]).detach()

        # AUX ensemble (train-time logits for heads, and mm-ensemble)
        aux = self.aux_expert(faces=face, x_vis=x_vis75, stft=stft, x_aud=x_aud, diss_prob=p_diss)

        # Optional routed logit for eval
        if infer_switch:
            thr = float(self.switch_threshold if switch_threshold is None else switch_threshold)
            conf = (p_diss - 0.5).abs() * 2.0
            pick_aux = (conf < thr).float()
            logits_switch = (1.0 - pick_aux) * av["logits"] + pick_aux * aux["logits"]
        else:
            logits_switch = None

        return {
            # Routed inference head
            "logits_switch": logits_switch,
            # AV main
            "diss_logits": av["logits"],
            "dissonance": torch.sigmoid(av["dissonance_logit"]),
            "dissonance_logit": av["dissonance_logit"],
            # AUX ensemble outputs
            "aux_logits": aux["logits"],        # train with y_mm (multimodal)
            "aux_a_logits": aux["z_a"],         # train with y_a  (audio-only label)
            "aux_v_logits": aux["z_v"],         # train with y_v  (visual-only label)
            "aux_gate": aux["gate"],
        }

# ====== Criterion (supports per-modality labels; backward compatible) ======
class DualCriterion(nn.Module):
    def __init__(self, lambda_aux: float = 0.25, lambda_total_balancing: float = 1.0,
                 use_switch_loss: bool = True, lambda_a: float = 1.0, lambda_v: float = 1.0,
                 lambda_aux_mm: float = 1.0):
        super().__init__()
        self.lambda_aux = float(lambda_aux)                 # scales whole AUX block
        self.lambda_total = float(lambda_total_balancing)
        self.use_switch_loss = bool(use_switch_loss)
        self.lambda_a = float(lambda_a)                     # audio-only head
        self.lambda_v = float(lambda_v)                     # visual-only head
        self.lambda_aux_mm = float(lambda_aux_mm)           # mm label for ensemble logit
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], labels, infer_switch: bool = False) -> Dict[str, torch.Tensor]:
        """
        labels can be:
          - Tensor y_mm                      (backward compatible)
          - Dict {'y_mm': Tensor, 'y_a': Tensor, 'y_v': Tensor}
        """
        if isinstance(labels, dict):
            y_mm = labels["y_mm"].float()
            y_a  = labels.get("y_a")
            y_v  = labels.get("y_v")
            y_a = y_a.float() if y_a is not None else None
            y_v = y_v.float() if y_v is not None else None
        else:
            y_mm = labels.float()
            y_a = y_v = None

        # AV main loss
        loss_diss = self.bce(outputs["diss_logits"], y_mm)

        # AUX losses (only if logits exist)
        loss_aux_mm = self.bce(outputs["aux_logits"], y_mm) if ("aux_logits" in outputs) else torch.tensor(0.0, device=outputs["diss_logits"].device)
        loss_a = self.bce(outputs["aux_a_logits"], y_a) if (y_a is not None and "aux_a_logits" in outputs) else torch.tensor(0.0, device=outputs["diss_logits"].device)
        loss_v = self.bce(outputs["aux_v_logits"], y_v) if (y_v is not None and "aux_v_logits" in outputs) else torch.tensor(0.0, device=outputs["diss_logits"].device)

        # Weighted total
        aux_block = self.lambda_aux_mm * loss_aux_mm + self.lambda_a * loss_a + self.lambda_v * loss_v
        loss_total = self.lambda_total * (loss_diss + self.lambda_aux * aux_block)

        # Optional switch loss (only if you compute routed logits during training)
        if infer_switch and self.use_switch_loss and outputs.get("logits_switch") is not None:
            loss_sw = self.bce(outputs["logits_switch"], y_mm)
            loss_total = loss_total + loss_sw
        else:
            loss_sw = torch.tensor(0.0, device=outputs["diss_logits"].device)

        with torch.no_grad():
            acc_diss = ((torch.sigmoid(outputs["diss_logits"]) >= 0.5) == (y_mm >= 0.5)).float().mean()
            acc_aux  = ((torch.sigmoid(outputs.get("aux_logits", torch.zeros_like(outputs["diss_logits"]))) >= 0.5) == (y_mm >= 0.5)).float().mean()

        return {
            "loss": loss_total,
            "loss_diss": loss_diss.detach(),
            "loss_aux_mm": loss_aux_mm.detach(),
            "loss_a": loss_a.detach(),
            "loss_v": loss_v.detach(),
            "loss_sw": loss_sw.detach(),
            "acc_diss": acc_diss.detach(),
            "acc_aux": acc_aux.detach(),
        }

def build_dissonance_dual_model(
    *,
    vis_dim: int,
    aud_dim: int = 75,
    stft_bins: int = 201,
    emb_dim_audio: int = 128,
    emb_dim_visface: int = 256,
    hidden_audio: int = 256,
    enc_heads: int = 4,
    enc_layers: int = 1,
    enc_dropout: float = 0.1,
    pe_max_len: int = 2000,
    pe_dropout: float = 0.0,
    cls_dropout_audio: float = 0.3,
    fusion_mode: Literal["gated", "concat", "xattn"] = "gated",
    face_pretrained: bool = True,
    face_freeze_backbone: bool = False,
    switch_threshold: float = 0.25,
    lambda_aux: float = 0.25,
    lambda_total_balancing: float = 1.0,
    feature_set: Literal["one", "small", "bins9_11_pack"] = "bins9_11_pack",
    feature_indices: Optional[List[int]] = None,
) -> Tuple[DissonanceDualModel, DualCriterion]:
    model = DissonanceDualModel(
        vis_dim=vis_dim, aud_dim=aud_dim, stft_bins=stft_bins,
        emb_dim_audio=emb_dim_audio, emb_dim_visface=emb_dim_visface, hidden_audio=hidden_audio,
        enc_heads=enc_heads, enc_layers=enc_layers, enc_dropout=enc_dropout,
        pe_max_len=pe_max_len, pe_dropout=pe_dropout, cls_dropout_audio=cls_dropout_audio,
        fusion_mode=fusion_mode, face_pretrained=face_pretrained, face_freeze_backbone=face_freeze_backbone,
        switch_threshold=switch_threshold, feature_set=feature_set, feature_indices=feature_indices,
    )
    crit = DualCriterion(lambda_aux=lambda_aux, lambda_total_balancing=lambda_total_balancing, use_switch_loss=True)
    return model, crit

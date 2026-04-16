# Integration diffs — wiring the remediation modules into existing code

These are the one-line edits in the existing `v3-fixes` tree that make the new
modules take effect. Apply after copying `resistancemap/` from
`ResistanceMap_v3_remediation/` over your checkout.

---

## 1. `resistancemap/data/preprocessors.py` — use hierarchical splits

```diff
-from sklearn.model_selection import GroupShuffleSplit
+from resistancemap.data.splits import make_split, assert_folds_disjoint

-def split(meta, seed=0):
-    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
-    tr, te = next(gss.split(meta, groups=meta["patient_id"]))
-    return tr, te
+def split(meta, kind="patient", n_folds=5, seed=0):
+    folds, manifest = make_split(meta, kind=kind, n_folds=n_folds, seed=seed)
+    assert_folds_disjoint(folds, meta,
+        key={"patient": "patient_id",
+             "lineage": "lineage_subtype",
+             "drug_holdout": "drug"}[kind])
+    return folds, manifest
```

---

## 2. `resistancemap/main.py` — disjointness assertion before VAE finetune

```diff
 # After `data_prep` and before `vae_finetune`:
+from resistancemap.data.splits import assert_no_overlap
+assert_no_overlap(
+    pretrain_cohort.patient_ids,
+    finetune_cohort.patient_ids,
+    key="patient_id", label_a="pretrain", label_b="finetune",
+)
+assert_no_overlap(
+    pretrain_cohort.lineage_ids,
+    eval_cohort.lineage_ids,
+    key="lineage", label_a="pretrain", label_b="eval",
+)
```

---

## 3. `resistancemap/data/loaders.py` — gate CoMMpass

```diff
-def load_commpass(path):
-    return pd.read_csv(path)
+def load_commpass(path, dar_id: str | None = None):
+    import os
+    if dar_id is None:
+        dar_id = os.environ.get("RM_COMMPASS_DAR")
+    if dar_id is None:
+        raise RuntimeError(
+            "CoMMpass access requires a dbGaP Data Access Request. "
+            "Set env RM_COMMPASS_DAR=... or pass dar_id=. "
+            "Public checkpoints/metrics must NOT depend on this cohort."
+        )
+    df = pd.read_csv(path)
+    df.attrs["controlled_access"] = True
+    df.attrs["dar_id"] = dar_id
+    return df
```

---

## 4. `resistancemap/models/rl_treatment_optimization.py` — OPE gate

```diff
+from resistancemap.models.rl.ope import evaluate as ope_evaluate
+
 class RLTreatmentOptimizer:
     def __init__(self, ...):
         ...
+        self._ope_passed = False
+        self._ope_result = None
+
+    def calibrate_off_policy(self, behavior_logp, target_logp, rewards, q_hat, v_hat):
+        self._ope_result = ope_evaluate(behavior_logp, target_logp, rewards, q_hat, v_hat)
+        self._ope_passed = self._ope_result.passed
+        return self._ope_result

     def recommend(self, state):
+        if not self._ope_passed:
+            raise RuntimeError(
+                "RL recommendations locked: call calibrate_off_policy() first "
+                "and confirm positivity + WIS/DR agreement."
+            )
         ...
```

---

## 5. `resistancemap/inference/pipeline.py` — adjudicate outputs

```diff
+from resistancemap.infrastructure.guardrails import adjudicate
+
 def predict(self, inputs):
     outputs = self._forward(inputs)
+    failed = adjudicate(outputs)
+    if failed:
+        # FastAPI layer translates to HTTP 422
+        raise GuardrailFailure(failed)
     return outputs
```

```diff
+class GuardrailFailure(ValueError):
+    def __init__(self, failed_rules: list[str]):
+        self.failed_rules = failed_rules
+        super().__init__(f"Guardrails failed: {failed_rules}")
```

FastAPI layer:

```diff
+from resistancemap.inference.pipeline import GuardrailFailure
+
 @app.exception_handler(GuardrailFailure)
 def _guardrail(request, exc):
     return JSONResponse(status_code=422,
         content={"error": "guardrail_failure", "rules": exc.failed_rules})
```

---

## 6. `resistancemap/models/protein_network.py` — GraphSAGE + sampling

```diff
-from torch_geometric.nn import GATConv
+from torch_geometric.nn import SAGEConv, GATConv

 class PPIGraphNetwork(nn.Module):
-    def __init__(self, in_dim=1280, hidden_dim=256, n_layers=4, n_heads=8, ...):
+    def __init__(self, in_dim=1280, hidden_dim=256, n_layers=4, n_heads=8,
+                 conv: str = "sage",
+                 num_neighbors: tuple[int, ...] = (25, 10, 10, 5),
+                 ...):
         ...
-        self.convs = nn.ModuleList([
-            GATConv(in_dim if i == 0 else hidden_dim * n_heads,
-                    hidden_dim, heads=n_heads, dropout=dropout)
-            for i in range(n_layers)
-        ])
+        if conv == "sage":
+            self.convs = nn.ModuleList([
+                SAGEConv(in_dim if i == 0 else hidden_dim, hidden_dim)
+                for i in range(n_layers)
+            ])
+            self.num_neighbors = num_neighbors
+        elif conv == "gat":
+            self.convs = nn.ModuleList([
+                GATConv(in_dim if i == 0 else hidden_dim * n_heads,
+                        hidden_dim, heads=n_heads, dropout=dropout)
+                for i in range(n_layers)
+            ])
+            self.num_neighbors = None
+        else:
+            raise ValueError(f"conv must be 'sage' or 'gat'; got {conv!r}")
```

Training loop uses `torch_geometric.loader.NeighborLoader` with
`self.num_neighbors` when `conv == "sage"`.

---

## 7. `resistancemap/models/fusion.py` — parameter-budget warning

```diff
+import warnings
+from resistancemap._facts import FACTS
+
 class MultiModalFusion(nn.Module):
-    def __init__(self, ...):
+    def __init__(self, ..., n_train: int | None = None,
+                 max_params_per_sample: float = 0.1):
         super().__init__()
         ...
+        n_params = sum(p.numel() for p in self.parameters())
+        if n_train is not None and n_params > max_params_per_sample * n_train:
+            warnings.warn(
+                f"Fusion has {n_params} params for {n_train} train samples "
+                f"({n_params/n_train:.1f}/sample). Consider LateFusionLogistic "
+                f"(resistancemap/evaluation/baselines.py).",
+                stacklevel=2,
+            )
```

---

## 8. `resistancemap/evaluation/metrics/__init__.py` — enforce contract

```diff
+from .per_drug import (
+    per_drug_metrics, pooled_with_ci, require_per_drug_with_pooled,
+)
+
+def emit_headline(y_true, y_score, drug, target_key: str):
+    """Public metric writer. Enforces per-drug + pooled contract."""
+    from resistancemap.data.targets import require_target
+    require_target(target_key)          # refuse unknown targets
+    return require_per_drug_with_pooled(y_true, y_score, drug)
```

All call sites that previously wrote a pooled AUROC to a report now go through
`emit_headline` and cannot bypass the per-drug table.

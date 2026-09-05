"""Prefill-only GenRec serving on Modal.

    modal deploy modal_serve.py      # -> a live HTTPS endpoint

Serving GenRec is NOT LLM text serving. There is no decode loop, so none of the
usual generation machinery applies: one forward pass produces a pooled hidden
state, and one matmul against the item table produces the scores. The endpoints
below expose that, plus the two things that only become visible at serving time:

  * /bench   the blog's cost claim, measured. "Serving cost is approximately
             proportional to context length" and "context to roughly one-third
             with negligible degradation" are testable on our own model.
  * /catalog/add   continuous catalogue growth with NO retraining, which is the
             v2 cold-start path. The 5-core benchmark structurally cannot test
             this, so serving is the only place it can be demonstrated.
"""
import modal

app = modal.App("genrec-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.44", "numpy",
                 "fastapi[standard]", "prometheus-client", "httpx")
    .add_local_dir("src", remote_path="/root/src")
)

out_vol = modal.Volume.from_name("genrec-out", create_if_missing=True)
MODEL_DIR = "/out/v2/model"


@app.cls(image=image, gpu="A10G", volumes={"/out": out_vol},
         scaledown_window=300, timeout=600)
class Server:
    @modal.enter()
    def load(self):
        import sys, time
        sys.path.insert(0, "/root/src")
        from serve import GenRecService
        t0 = time.time()
        self.svc = GenRecService(MODEL_DIR, device="cuda", backend="torch",
                                 domain="product")
        self.load_s = time.time() - t0
        print(f"[serve] model ready in {self.load_s:.1f}s")

    @modal.asgi_app()
    def web(self):
        from api import create_app
        return create_app(self.svc, load_seconds=self.load_s)

"""Benchmark train step time for encoder configs on synthetic studies (A100)."""
import sys, time, os, itertools, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KneeNet, gpu_augment
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda')
cfgs = [a.split(':') for a in sys.argv[1:]]  # backbone:img:k:ckpt:compile
for bb, img, k, ck, comp in cfgs:
    img, k, ck, comp = int(img), int(k), ck == '1', comp == '1'
    m = KneeNet(bb, pretrained=False, grad_ckpt=ck).to(dev)
    m.enc.to(memory_format=torch.channels_last)
    if comp: m.enc = torch.compile(m.enc)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    S = 6
    x = torch.randint(0, 255, (1, S, k, 3, img, img), dtype=torch.uint8, device=dev)
    types = torch.zeros(1, S, dtype=torch.long, device=dev); pos = torch.rand(1, S, k, 2, device=dev)
    present = torch.ones(1, S, device=dev); present[0, 5] = 0
    y = torch.rand(1, 12, device=dev)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for it in range(25):
        t0 = time.time()
        xa = gpu_augment(x, present)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = m(xa, types, pos, present)
        loss = F.binary_cross_entropy_with_logits(lg.float(), y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    t = sum(ts[10:]) / len(ts[10:])
    print(f"{bb} img{img} k{k} ckpt{int(ck)} compile{int(comp)}: {t:.3f}s/study  {5*k/t:.0f} img/s  peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
    del m, opt; torch.cuda.empty_cache()

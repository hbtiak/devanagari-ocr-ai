import torch
from .model import CRNN

CHARS = [""] + list("अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह")
IDX2CHAR = {i:c for i,c in enumerate(CHARS)}

def greedy_decode(output):
    probs = torch.softmax(output, dim=2)
    best = probs.argmax(2)
    conf = probs.max(2)[0].mean().item()

    prev = -1
    text = ""
    for t in best[0]:
        if t != prev and t != 0:
            text += IDX2CHAR[t.item()]
        prev = t

    return text, conf

def run_ocr(image):
    model = CRNN(num_classes=len(CHARS))
    model.eval()

    image = image / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        out = model(image)

    return greedy_decode(out)

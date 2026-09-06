# Mine AI

**Apna khud ka chat AI — scratch se. Koi API nahi, koi ML library nahi.**

Ye ek web app hai jis mein ek chhota transformer language model chalta hai jo
**poori tarah is repo ke andar** likha gaya hai: apna tokenizer, apna autograd,
apna attention, apna training loop. `package.json` mein **zero dependencies** hain —
sirf Node ke built-in modules.

```
No OpenAI. No Anthropic. No HuggingFace. No TensorFlow. No PyTorch. No npm packages.
```

---

## Fauran shuru karen

```bash
git clone https://github.com/mubashirsafeer7/mine-ai.git
cd mine-ai

npm test          # sab kuch theek hai ya nahi (gradient checks bhi)
npm run train     # apna model train karen  (~30 minute)
npm start         # http://localhost:3000
```

Terminal mein baat karni ho to:

```bash
npm run chat
```

> Repo mein ek trained model (`models/mine-ai.json`) pehle se maujood hai,
> is liye `npm start` bina training ke bhi chal jayega. Apna data daal kar
> `npm run train` chalayen to model dobara aap ke data par train ho jayega.

---

## Ye kaam kaise karta hai

```
data/corpus.txt
      │
      │  parseCorpus()          <|user|> … <|bot|> … <|end|>
      ▼
  BPE Tokenizer                 src/core/tokenizer.js
      │  text → token ids
      ▼
  Transformer                   src/core/model.js
      │  embedding → 4 × (attention + MLP) → logits
      ▼
  Training loop                 src/scripts/train.js
      │  cross-entropy → autograd → AdamW
      ▼
  models/mine-ai.json           weights (base64 float32) + tokenizer
      │
      ▼
  Brain (model + memory)        src/core/brain.js
      │
      ▼
  Web app / CLI                 src/server.js · web/
```

### 1. Tokenizer (BPE, scratch se)

`src/core/tokenizer.js` — Byte Pair Encoding khud implement kiya gaya hai:

- Text words mein tootta hai, leading space `▁` ban jati hai.
- Har word pehle characters ki list hoti hai.
- Sab se zyada aane wale adjacent pair ko baar baar merge karte hain jab tak
  vocab size poora na ho jaye.
- `<|user|>`, `<|bot|>`, `<|end|>` jaise special tokens kabhi nahi tootte.

### 2. Autograd (scratch se)

`src/core/tensor.js` — reverse-mode automatic differentiation. Har operation
apna `_backward()` rakhta hai, aur `loss.backward()` topological order mein
poore graph par gradients chala deta hai.

Ops: `matmul`, `matmulNT`, `add`, `addBias`, `scale`, `gelu`, `layerNorm`,
`causalSoftmax`, `sliceCols`, `concatCols`, `embed`, `softmaxCrossEntropy`.

Har gradient **finite differences se verify** kiya gaya hai — `npm test` dekhen.

### 3. Model

`src/core/model.js` — GPT jaisa decoder-only transformer:

| Cheez | Value (default) |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| Embedding size | 128 |
| Context (block size) | 64 tokens |
| Vocab | ~890 BPE pieces |
| Parameters | ~915,000 |

Har block: `LayerNorm → Causal Multi-Head Self-Attention → residual`, phir
`LayerNorm → MLP (4× hidden, GELU) → residual`. Aakhir mein final LayerNorm aur
**tied lm-head** (token embedding ka hi transpose use hota hai).

### 4. Training

`src/scripts/train.js` — AdamW optimizer (scratch se), gradient clipping,
warmup + cosine learning-rate decay.

Ahem baat: **loss sirf bot ke tokens par lagta hai**. User ke tokens sirf
context hain (masked). Isi liye model sawal repeat karne ke bajaye jawab dena
seekhta hai.

### 5. Brain — model + memory

Model chhota hai (~900k params, ~200 conversations). Itne chhote model se har
sawal ka saaf jawab mushkil hai, is liye `src/core/brain.js` do cheezen jodta hai:

- **Neural** — transformer khud jawab generate karta hai.
- **Memory** — TF-IDF retrieval (`src/core/memory.js`, ye bhi scratch se: words +
  character trigrams + cosine similarity) corpus mein se sab se milta julta
  sawal dhoondti hai.

Teen modes hain, UI mein switch kar sakte hain:

| Mode | Kya karta hai |
|---|---|
| `auto` | Match bohat pakka (≥0.62) ho to memory, warna neural. Neural output bigar jaye to memory bacha leti hai. |
| `neural` | **Hamesha** model se generate — apni training ka asli test yahi hai. |
| `memory` | Sirf retrieval, model bilkul use nahi hota. |

Har jawab ke neeche badge batata hai wo kahan se aaya: `neural`, `memory`,
`memory rescue`, ya `fallback`.

---

## Interface

Saada rakha gaya hai. Light theme default hai; dark theme toggle mein hai.
Koi font download nahi hota, koi CDN nahi — system fonts aur inline SVG.

- Chat history browser mein rehti hai (`localStorage`) — search, delete,
  din ke hisab se grouping.
- Har jawab ke neeche uska source: `neural` / `memory` / `memory rescue` /
  `fallback`, sath mein waqt.
- **Model** panel: parameters, layers, heads, vocab aur context meter.
- Mode composer mein: Auto · Neural · Memory.
- Jawab ke beech mein stop, phir retry aur copy.
- Shortcuts: `Ctrl/⌘+K` nayi chat, `Ctrl/⌘+B` sidebar, `Esc` band karein.

UI English mein hai; jawab ki zaban aap ke corpus par hai — abhi wo
Roman Urdu hai.

---

## Apne AI ko khud train karen

Ye poora point hai — model **aap ka** hai.

### 1. Apni baat cheet likhen

`data/corpus.txt` (ya `data/` mein koi bhi naya `.txt`) mein is format mein:

```
<|user|> tumhe cricket pasand hai
<|bot|> Ji haan, cricket dekhna achha lagta hai. Aap ka pasandida khilari kaun hai?
<|end|>
<|user|> mera naam ahmad hai
<|bot|> Mil kar khushi hui Ahmad! Main Mine AI hoon.
<|end|>
```

Multi-turn bhi chalta hai — ek `<|end|>` se pehle jitne `<|user|>`/`<|bot|>`
turns chahen likh dein.

**Mashwara:** ek hi baat ke 2-3 alag alag andaz likhen (`salam`, `assalam o alaikum`,
`hello`). Chhota model isi tarah generalize karna seekhta hai.

### 2. Train karen

```bash
npm run train
```

Options:

```bash
npm run train -- --steps 3000 --batch 16 --lr 0.003
npm run train -- --layers 6 --heads 6 --embd 192 --block 96   # bara model
npm run train -- --vocab 1500                                  # bara vocab
```

| Flag | Default | Matlab |
|---|---|---|
| `--steps` | 1500 | kitne training steps |
| `--batch` | 12 | ek step mein kitni misalen |
| `--lr` | 0.003 | learning rate |
| `--layers` `--heads` `--embd` | 4 / 4 / 128 | model ka size |
| `--block` | 64 | context length (tokens) |
| `--vocab` | 900 | tokenizer vocab size |
| `--seed` | 1337 | reproducibility |
| `--out` | `models/mine-ai.json` | checkpoint kahan save ho |

### 3. Test karen

```bash
npm run chat -- --mode neural    # sirf neural output dekhen
npm start                        # web app
```

Web app mein **Mode = Neural only** kar ke dekhen ke aap ka model asal mein
kitna seekha hai. `auto` mode rozana istemal ke liye behtar hai.

---

## Repo ka naqsha

```
mine-ai/
├── data/
│   └── corpus.txt              training data (aap yahan likhte hain)
├── models/
│   └── mine-ai.json            trained weights + tokenizer
├── src/
│   ├── core/
│   │   ├── tensor.js           autograd engine + matrix ops
│   │   ├── model.js            transformer (attention, MLP, blocks)
│   │   ├── tokenizer.js        BPE tokenizer
│   │   ├── optimizer.js        AdamW + gradient clipping + LR schedule
│   │   ├── sampler.js          temperature / top-k / top-p sampling
│   │   ├── dataset.js          corpus parsing + masked training examples
│   │   ├── memory.js           TF-IDF retrieval
│   │   ├── checkpoint.js       model save / load
│   │   └── brain.js            model + memory ko jodta hai
│   ├── scripts/
│   │   ├── train.js            training
│   │   ├── chat.js             terminal chat
│   │   └── build-tokenizer.js  tokenizer inspect karne ke liye
│   └── server.js               HTTP server + SSE streaming
├── web/
│   ├── index.html              chat UI
│   ├── style.css               dark + light theme
│   └── app.js                  frontend (koi framework nahi)
└── test/
    └── run.js                  gradient checks + tokenizer + training tests
```

---

## Tests

```bash
npm test
```

Kya check hota hai:

- `matmul` / `matmulNT` hand-computed jawab se match karte hain
- **Finite-difference gradient checks** — `layerNorm`, `gelu`, `causalSoftmax`,
  `embed`/`slice`/`concat`, aur poore model par
- Causal mask waqai future tokens block karta hai, har row ka sum 1 hai
- Tokenizer round-trip aur special tokens
- Dataset user tokens ko mask karta hai
- Model ek chhote dataset par overfit kar leta hai (training loop kaam karta hai)
- Checkpoint save/load bilkul wahi weights wapas deta hai

---

## Imaandar baat — limits

Ye asli, kaam karta hua transformer hai, lekin **chhota** hai:

- ~915k parameters (GPT-3 mein 175 **billion** hain).
- ~200 conversations ka training data.
- 64 token ka context — lambi baat cheet yaad nahi rehti.
- Val loss train loss se kaafi zyada rehta hai, kyunki data chhota hai aur model
  usay kaafi had tak **yaad** kar leta hai. Ye is scale par normal hai.
- Naye moze par `neural` mode kabhi kabhi ajeeb jumla bana dega — isi liye
  `auto` mode mein memory rescue lagi hui hai.

Behtar banana hai? **Zyada data likhen** (sab se zyada farq isi se parta hai),
phir `--steps` aur model size barhayen.

---

## Privacy

Sab kuch aap ke apne computer par chalta hai. Server sirf `localhost` par
sunta hai, koi network request bahar nahi jati, koi API key nahi chahiye,
koi bill nahi aata.

---

## License

MIT

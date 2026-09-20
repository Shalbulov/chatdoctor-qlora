# QLoRA fine-tune of Qwen2.5-7B-Instruct on doctor replies

[Русская версия](README.ru.md)

Teaching a 7B instruct model to answer a patient the way a doctor on a consultation
site does: one paragraph, straight to the assessment, no bullet lists and no
disclaimers. The base model answers the same question with a structured essay. That gap
is the thing being trained away, and it is what the evaluation measures.

Everything runs on free Kaggle GPUs: 27 minutes of training, and the adapter is 81 MB.

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: `lavita/ChatDoctor-HealthCareMagic-100k`
- Method: QLoRA, 4-bit nf4 base, LoRA r=16 on attention and MLP
- Adapter only, 15 GB of base weights never leave the Hub

## Results

One epoch over the 1,000 training examples, 63 optimiser steps, 27 minutes of training
on two Kaggle T4s. The adapter here is the checkpoint with the lowest validation loss,
which in this run is the final step.

![loss curve](reports/loss_curve.png)

Validation falls from 4.806 to 4.663 and is flattening rather than turning up. The three
epoch attempt that came first is in "What did not work" below.

| | base | base + adapter |
|---|---|---|
| ROUGE-L F1 | 0.119 | **0.168** |
| BERTScore F1 (rescaled) | -0.037 | **0.125** |
| wins on ROUGE-L, out of 20 | 3 | **17** |
| mean words | 303 | 75 |
| mean characters | 2049 | 436 |
| answers with a bullet list | 95% | 0% |
| answers with bold text | 95% | 0% |
| answers with a heading | 0% | 0% |
| answers with a disclaimer | 60% | 20% |

Reference answers, for comparison: 102 words, 0% with a bullet list, 15% with a
disclaimer.

Both metrics improve, and the paired count says it is not one outlier carrying the mean:
the adapter wins 17 of 20 examples. The style markers move further than the metrics do.
The base model opens a bullet list in 19 of 20 answers and the fine-tuned one in none,
against none in the references. Length drops from 303 words to 75, where the doctors
average 102, so the adapter slightly overshoots into terseness.

The metrics deserve to be read in that order. ROUGE-L rising from 0.119 to 0.168 is
partly the length collapse doing the work: a 300 word essay cannot overlap much with a
100 word paragraph however good it is. BERTScore crossing from below zero to 0.125 is
the stronger signal, since rescaling puts 0 at the similarity of an unrelated pair of
sentences. The honest summary is that the model learned the register convincingly, and
that neither metric can say whether it learned better medicine.

### One example

**Patient:** "i am having serious issues controling my bp. i take 10mg amlodopine, 40mg
lisinopril, 100mg antenolol and 2.5 mg bendrofluazide daily, i had it checked and it was
162/108 [...] and at 32 i am getting a bit fed up of this. any ides?"

**Base model**, 1 of 20 answers without a bullet list:

> It sounds like you're doing everything right in terms of lifestyle modifications,
> which is great. However, managing high blood pressure (hypertension) can sometimes be
> challenging despite these efforts. Here are some suggestions and considerations:
>
> 1. **Consult Your Doctor**: Given that your blood pressure remains high despite taking
> multiple medications, it would be wise to consult your healthcare provider [...]

**Base + adapter:**

> You are taking multiple medicines for hypertension. You should consult your doctor and
> get evaluated. You may need to change your medicines or increase the dose of your
> medicines. You can also get done 24 hours blood pressure monitoring to know about your
> blood pressure pattern. You should avoid stress and anxiety as these can increase your
> blood pressure.

**The doctor's actual reply:**

> You will first evaluation for cause of blood pressure. You should undergo test like
> kidney function test, renal artery Doppler, Sr cortisol and 24 hr Urinary
> Metanephrines [...]

The register is right and the specificity is not. The reference names the tests; the
fine-tuned model produces the shape of a doctor's reply around thinner content. That gap
is the honest limit of 1,000 examples of style transfer.

## Why this dataset

The task asks for a specific answer style, so the dataset has to differ from the base
model in a way a reader can see without squinting. HealthCareMagic does: real replies
from a consultation site, written as a single dense paragraph that commits to an
assessment. Ask Qwen2.5-7B-Instruct the same question and you get headings, bullets, a
differential, and a closing line telling you to see a doctor. Two clearly different
registers, which makes both the training signal and the evaluation legible.

The instruction column carries no information. It is the same sentence in all 112,165
rows, so it lives in `src/prompt.py` as a system prompt instead of being repeated
inside every example:

> If you are a doctor, please answer the medical questions based on the patient's
> description.

## Data

`src/explore.py` is a one-off look at the raw corpus that set the thresholds. It is in
the repository so the numbers below can be re-derived rather than taken on trust. What
it found: the answers are scraped and damaged in two different ways. Template noise the
scraper added is repairable, word substitution inside the medical text is not.

```
"Chat Doctor" dropped into the middle of a sentence     36,248 rows
answers with no terminal punctuation, cut mid-sentence  21,286 rows
"Phoenix nerve" for phrenic nerve, "Cellophane You" for Thank You
```

The second family is why the pipeline drops rows rather than patching them. A
substitution that lands on a medical term produces fluent, confident, wrong text, and
there is no way to catch it without a medical lexicon. Anything carrying a marker of
that damage is removed.

`src/clean.py` holds one repair rule per noise family; `src/prepare_data.py` runs the
funnel and writes the splits. Cheap row-level rules run over all 112k rows first, then
a fixed-seed pool of 6,000 is drawn and the O(n^2) near-duplicate pass runs inside it.
Near-dup over the full corpus would be 112k x 112k similarities to recover about 0.2%
of rows.

```
stage                                   kept   dropped
------------------------------------------------------
loaded                                112165
exact duplicate question              111964       201
'Chat Doctor' left mid-answer          75716     36248
corrupted opening word                 71972      3744
'.*' corruption marker                 71543       429
markdown or enumerated list            67178      4365
answer starts mid-word                 57282      9896
answer cut off mid-sentence            35996     21286
length outside bounds                  34566      1430
sampled candidate pool (seed 42)        6000     28566
near-duplicate question (cos>=0.85)     6000
```

Near-duplicate removal found nothing inside the pool, which is the expected outcome
after exact-duplicate removal on a corpus this sparse; it stays in the pipeline because
it is cheap at pool size and the assignment asks for it.

Of 112,165 rows, 34,566 survive. Splits are 1,000 train, 100 validation, 20 held out,
all seed 42, JSONL with `instruction` and `response`. Ten random training rows are
rendered in `reports/sample_train.md` for manual review, because a funnel table says
nothing about whether the surviving text is any good.

Training on 1,000 of the 34,566 is deliberate. Style transfer of this kind saturates
early, and a smaller set buys three epochs inside one free GPU session with room for a
failed attempt.

## Training

`src/train_lora.py`. The card drives most of the configuration: a T4 is compute
capability 7.5, so no bf16 and no flash-attention, which means fp16 throughout, fp16 as
the 4-bit compute dtype, and sdpa as the attention backend.

| | |
|---|---|
| quantisation | nf4, double quant, fp16 compute |
| LoRA | r=16, alpha=32, dropout 0.05 |
| target modules | q,k,v,o + gate,up,down |
| sequence length | 640 (longest cleaned example is 524 tokens) |
| batch | 2 per device x 2 cards x 4 accumulation = effective 16 |
| optimiser | paged AdamW 8-bit, lr 2e-4, cosine, 3% warmup |
| epochs | 1, 63 optimiser steps, best checkpoint by validation loss |
| other | gradient checkpointing, max grad norm 0.3, adapter saved in fp16 |

Two choices worth defending.

**Loss covers the answer only.** Prompt tokens are set to -100 in `src/sft_data.py`.
Without the mask the model spends half its capacity learning to reproduce patient
complaints, which is not the task.

**LoRA targets the MLP blocks too, not just attention.** Attention-only adapters train
faster and adapt style less. The verbosity being trained away lives in the feed-forward
blocks, so they are included.

The prompt and the answer are tokenised separately rather than tokenising the whole
conversation and slicing it. The chat template can merge tokens across the boundary,
and an off-by-one there silently shifts the entire label mask.

## Evaluation

`src/generate.py` answers the 20 held-out questions twice and `src/metrics.py` scores
both columns against the doctor's real reply.

Both columns come off one loaded model. The adapter is attached once and switched off
for the base pass through `disable_adapter()`, so the comparison is against exactly the
weights the adapter sits on, with identical 4-bit quantisation on both sides. Loading an
unquantised base separately would have made quantisation a second difference between the
columns. Decoding is greedy: sampling would put run-to-run noise on top of a 20 example
comparison.

Three groups of numbers, because no one of them settles the question.

**ROUGE-L** is lexical overlap with the reference. On free-form medical advice it is a
weak proxy, since two correct answers can share almost no wording. It is reported
because the task asks for a standard metric and because a large move in it is still
informative. A paired win count sits next to the mean, since a mean over 20 examples can
be carried by one outlier.

**BERTScore** is embedding overlap, less sensitive to paraphrase, rescaled against the
package baseline. Without rescaling every score sits near 0.85 and differences are
invisible.

**Style markers** are length, bullet lists, bold, headings and disclaimers. This is what
the fine-tune was aimed at, so it is measured directly rather than inferred from a
metric that only sees it sideways. The reference answers are scored on the same markers,
so it is clear which pole each column sits closer to.

## Running it

```bash
pip install -r requirements-data.txt
python src/prepare_data.py                 # writes data/*.jsonl and data/funnel.txt

# training needs a GPU, see notebooks/kaggle_train.ipynb or notebooks/colab_train.ipynb
PYTHONPATH=src python src/train_lora.py --out adapter

pip install -r requirements-eval.txt
PYTHONPATH=src python src/generate.py --adapter adapter   # needs a GPU
python src/metrics.py                                     # CPU, about a minute
```

The notebooks are thin on purpose: they orchestrate, and the code they call is the same
code that sits in `src/`.

## What did not work

**Three epochs was two too many.** The first run trained 3 epochs, 189 steps. Training
loss fell from 5.6 to 3.5 while validation bottomed at 4.663 near step 50 and then
climbed back to 4.831. Only the final adapter was saved, so that run shipped its own
worst checkpoint. The picture is in `reports/loss_curve_3epochs.png`. The fix is the
`--load-best` flag: validate every 8 steps and let the Trainer restore the best
checkpoint at the end. One epoch reaches the same validation minimum and stops there.

**The step arithmetic was wrong by a factor of two.** Kaggle hands out two T4s and the
Trainer uses both, so a per-device batch of 2 with 4 accumulation steps is an effective
batch of 16, not the 8 that single-card arithmetic assumed. The first run took 189 steps
where the notebook predicted 375. The estimate now multiplies by the visible device
count, and the script prints what it resolved before training starts.

**A Kaggle dataset that is still processing mounts as nothing at all.** The first kernel
was pushed seconds after the code dataset was created, so `/kaggle/input` was empty.
Since `!python missing_file.py` does not raise inside a notebook, the run sailed through
both training cells and failed 40 minutes later at the plotting cell, with a GPU session
spent on nothing. The staging cell now prints the mount and asserts on it, and finds the
code by filename rather than by path: the real mount point is
`/kaggle/input/datasets/<owner>/<slug>`, not the `/kaggle/input/<slug>` the docs
describe.

**Near-duplicate removal earned no rows.** Cosine over character n-grams at 0.85 dropped
0 of the 6,000 pooled rows, because exact duplicate removal had already taken the 201
worth taking. It stays in the pipeline since it costs nothing at pool size, but it is
reported here rather than quietly presented as if it had worked.

**W&B runs offline.** Live logging needs an API key inside the kernel, and the only
key-free way to get one there on Kaggle is a secret created through the web UI. The run
is logged offline into `wandb/` and becomes a normal run with
`wandb sync wandb/offline-run-*`.

## Limitations

The word-substitution damage described above is detectable only when it produces a
recognisable marker. Rows where the substitution landed cleanly are still in the
training data, so the model is learning from some quietly wrong medical text. This is a
property of the corpus, not something the filters can fix.

Twenty held-out examples is what the task asks for and it is enough to see a style
change, but it is too few to separate two close models on ROUGE. The paired win count is
reported for that reason and should be read as a direction, not a measurement.

Nothing here is medical advice, and the fine-tune makes the model more confident in
tone, not more correct. Training a model to drop its disclaimers is a legitimate style
exercise and a bad idea in production.

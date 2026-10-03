# ZERO LLM

**ZERO LLM**, sıfırdan oluşturulan, başlangıçta herhangi bir dil bilgisi veya pretrained model ağırlığı içermeyen bir **decoder-only Transformer dil modeli çekirdeğidir**.

Projenin amacı hazır bir LLM'i kullanmak veya fine-tune etmek değil; model mimarisini, tokenizer'ı ve eğitim altyapısını kontrol ederek **temelden eğitilebilir bağımsız bir dil modeli** oluşturmaktır.

> **ZERO = Zero Knowledge Initialization**

Model başlangıçta yalnızca mimariden ve rastgele initialize edilmiş ağırlıklardan oluşur. Dil bilgisi eğitim sürecinde öğrenilir.

---

## Özellikler

* Decoder-only Transformer
* Causal Self-Attention
* Rotary Positional Embedding (RoPE)
* RMSNorm
* SwiGLU tabanlı Feed Forward Network
* Residual Connections
* Configurable Multi-Head / Grouped-Query Attention
* Configurable vocabulary size
* Configurable context length
* Configurable model depth and width
* Next-token prediction
* Cross-Entropy training objective
* Gradient accumulation
* Mixed precision
* Gradient clipping
* Checkpoint / resume
* Validation loss tracking
* Token generation
* Greedy decoding
* Temperature sampling
* Top-k sampling
* Top-p sampling
* CPU/GPU desteği

---

## Mimari

ZERO LLM'nin temel veri akışı:

```text
                    ┌──────────────┐
                    │     Text     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Tokenizer  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Token IDs    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Embedding  │
                    └──────┬───────┘
                           │
                           ▼
             ┌────────────────────────────┐
             │    Transformer Block × N   │
             │                            │
             │  RMSNorm                   │
             │      ↓                     │
             │  Causal Self-Attention     │
             │      ↓                     │
             │  Residual                  │
             │      ↓                     │
             │  RMSNorm                   │
             │      ↓                     │
             │  SwiGLU / FFN              │
             │      ↓                     │
             │  Residual                  │
             └────────────┬───────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │  Final RMSNorm│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    LM Head   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Logits    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Next Token   │
                    └──────────────┘
```

---

## Transformer Block

Her Transformer bloğu Pre-Norm yapısı kullanır:

```text
x
│
├── RMSNorm
│
├── Causal Self-Attention
│
└── Residual Add
        │
        ▼
     RMSNorm
        │
        ▼
      SwiGLU
        │
        ▼
   Residual Add
```

Matematiksel olarak:

```python
x = x + attention(norm_1(x))
x = x + mlp(norm_2(x))
```

---

## Attention

Attention mekanizması causal'dır.

Model yalnızca mevcut token ve kendisinden önce gelen tokenleri görebilir.

Temel attention işlemi:

```text
Attention(Q, K, V)
    =
softmax((QKᵀ / √d) + causal_mask)V
```

Mimari, `num_attention_heads` ve `num_kv_heads` parametrelerini ayrı tutar.

Bu sayede Multi-Head Attention veya Grouped-Query Attention yapılandırmaları desteklenebilir.

---

## Positional Encoding

ZERO LLM, token sırasını temsil etmek için **Rotary Positional Embedding (RoPE)** kullanır.

Pozisyon bilgisi attention mekanizması içerisinde uygulanır.

---

## Feed Forward Network

Transformer bloklarında gated feed-forward yapı kullanılır.

Varsayılan yaklaşım:

```text
Input
  │
  ├───────────────┐
  │               │
  ▼               ▼
Linear          Linear
  │               │
  ▼               │
SiLU              │
  │               │
  └───────×───────┘
          │
          ▼
       Linear
          │
          ▼
        Output
```

Bu yapı SwiGLU tabanlıdır.

---

## Model Yapılandırması

Model parametreleri merkezi bir configuration sistemi üzerinden yönetilir.

Örnek:

```python
config = {
    "vocab_size": 32000,
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "num_kv_heads": 12,
    "intermediate_size": 2048,
    "max_seq_len": 2048,
    "rope_theta": 10000,
    "rms_norm_eps": 1e-6,
}
```

Model boyutu mimari değiştirilmeden config üzerinden değiştirilebilir.

Örneğin:

```text
Tiny
100M
300M
500M
1B
```

gibi farklı ölçeklerde modeller oluşturulabilir.

---

## Tokenizer

Tokenizer modelden bağımsızdır.

ZERO LLM'nin hedef kullanım alanlarından biri Türkçe dil modeli eğitimi olduğundan UTF-8 ve Türkçe karakter desteği önemlidir.

Desteklenmesi gereken karakterler:

```text
ç Ç
ğ Ğ
ı İ
ö Ö
ş Ş
ü Ü
```

Tokenizer'ın vocabulary'si modelden bağımsız şekilde oluşturulabilir.

---

## Eğitim

ZERO LLM, **next-token prediction** yaklaşımıyla eğitilir.

Örneğin:

```text
Input:
Ben bugün ev

Target:
bugün ev e gidiyorum
```

Gerçek eğitimde her pozisyon için bir sonraki token tahmin edilir.

Loss fonksiyonu:

```text
Cross Entropy Loss
```

Training sistemi aşağıdaki özellikleri destekleyecek şekilde tasarlanmıştır:

* Gradient accumulation
* Mixed precision
* Gradient clipping
* Optimizer
* Learning-rate scheduler
* Checkpointing
* Resume training
* Validation
* Loss tracking
* Tokens/sec
* GPU memory monitoring

---

## Proje Yapısı

```text
ZERO_LLM/
│
├── config.py
├── model.py
├── tokenizer.py
├── embeddings.py
├── attention.py
├── rope.py
├── rmsnorm.py
├── mlp.py
├── transformer_block.py
├── lm_head.py
│
├── dataset.py
├── train.py
├── inference.py
├── generate.py
├── checkpoint.py
├── utils.py
│
├── tests/
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   ├── test_training.py
│   └── test_generation.py
│
└── README.md
```

---

## Sıfırdan Başlangıç

ZERO LLM herhangi bir pretrained checkpoint ile başlamaz.

Başlangıç durumu:

```text
Architecture
     +
Random Initialization
     =
ZERO LLM
```

Modelin ilk oluşturulduğu anda dil bilgisi bulunmaz.

Öğrenme, eğitim corpus'u üzerinde gerçekleştirilen gradient-based optimization sonucunda oluşur.

---

## Pretrained Model Kullanımı Yok

ZERO LLM'nin temel prensiplerinden biri model ağırlıklarının dışarıdan alınmamasıdır.

Proje aşağıdakileri kullanmaz:

* Pretrained LLM
* Fine-tuning
* LoRA
* Adapter
* Knowledge distillation
* Model weight transfer
* Harici LLM API
* Hazır LLM inference engine

Amaç, model çekirdeğinin tamamen bağımsız olarak oluşturulmasıdır.

---

## Test

İlk geliştirme aşamasında büyük model veya gerçek corpus ile eğitim yapılmaz.

Öncelikle küçük bir model kullanılır:

```text
vocab_size:          1000
hidden_size:          128
num_layers:             4
num_attention_heads:   4
num_kv_heads:          4
```

Ardından:

```text
Model Initialization
        ↓
Forward Pass
        ↓
Loss Calculation
        ↓
Backward Pass
        ↓
Optimizer Update
        ↓
Checkpoint
        ↓
Checkpoint Reload
        ↓
Generation
```

test edilir.

---

## Doğrulama

İlk sürüm aşağıdaki kontrollerden geçmelidir:

* [ ] Model initialize oluyor
* [ ] Parameter count doğru
* [ ] Forward pass çalışıyor
* [ ] Causal mask doğru
* [ ] RoPE çalışıyor
* [ ] RMSNorm çalışıyor
* [ ] SwiGLU çalışıyor
* [ ] Residual connections çalışıyor
* [ ] Loss hesaplanıyor
* [ ] Backpropagation çalışıyor
* [ ] Optimizer ağırlıkları güncelliyor
* [ ] Loss eğitim sırasında düşüyor
* [ ] Checkpoint oluşturuluyor
* [ ] Checkpoint geri yükleniyor
* [ ] Inference çalışıyor
* [ ] Token generation çalışıyor

---

## Teknoloji

| Bileşen       | Teknoloji                |
| ------------- | ------------------------ |
| Language      | Python                   |
| Deep Learning | PyTorch                  |
| Architecture  | Decoder-only Transformer |
| Attention     | Causal Self-Attention    |
| Position      | RoPE                     |
| Normalization | RMSNorm                  |
| FFN           | SwiGLU                   |
| Objective     | Next-Token Prediction    |
| Loss          | Cross Entropy            |

---

## Projenin Durumu

ZERO LLM şu aşamada **model çekirdeğinin oluşturulması ve doğrulanması** aşamasındadır.

Öncelik:

```text
Doğru mimari
      ↓
Çalışan çekirdek
      ↓
Doğrulanabilir eğitim
      ↓
Tokenizer
      ↓
Dataset
      ↓
Model eğitimi
```

---

## Lisans

Lisans bilgisi henüz belirlenmemiştir.

---

## ZERO LLM

> **Start from zero. Learn from data.**

# ZERO LLM

**ZERO LLM, rastgele ağırlıklarla başlayan, sıfırdan eğitilebilir bir decoder-only
Transformer çekirdeğidir.** Hazır model ağırlığı, harici LLM API'si, LoRA veya bilgi
aktarımı kullanmaz. **ZERO = Zero Knowledge Initialization.**

Bu depo şu anda model çekirdeği, küçük eğitim deneyleri ve kesinti sonrası toparlanma
altyapısını içerir. **Eğitilmiş, dil yeteneği kanıtlanmış bir Türkçe LLM yayımlanmamıştır.**
Rastgele modelin veya birkaç adımlık denemenin anlamsız çıktı vermesi beklenir.

## Güncel durum ve doğrulama

2026-11-24 tarihinde Linux / Python 3.13.5 / PyTorch 2.15.0+cpu ortamında
**74 test geçti** (22,13 saniye).
Testleri yeniden çalıştırmak için aşağıdaki komutları kullanın; ayrıntılı kapsam
[tests/](tests/) ve [CHANGELOG.md](CHANGELOG.md) içinde yer alır.

| Alan | Durum |
| --- | --- |
| Forward, backward, optimizer, causal attention | CPU testleri var |
| RoPE, RMSNorm, SwiGLU, GQA | Çekirdek ve matematiksel referans testleri var |
| Eğitim → checkpoint → yalnız checkpoint ile üretim | Uçtan uca test edildi |
| Epoch ortasında devam, veri sırası ve dropout RNG | Kesintisiz eğitimle bit düzeyinde eşitlik CPU'da test edildi |
| SIGINT, SIGTERM, SIGKILL sonrası toparlanma | Gerçek alt süreçler durdurularak test edildi |
| Bozuk/yetersiz son checkpoint, yarım yayın, fsync hatası | Hata enjeksiyonu ve geri dönüş testleri var |
| Gerçek elektrik kesintisi / donanım arızası | **Fiziksel olarak test edilmedi; mutlak veri kaybı garantisi yok** |
| CUDA, gerçek AMP, büyük model eğitimi | Kod yolu mevcut; bu ortamda **doğrulanmadı** |
| Windows, dağıtık eğitim, çoklu DataLoader worker | Eğitim kurtarma akışı için destek/doğrulama yok |

Desteklenen eğitim/kurtarma hedefi Linux/POSIX üzerinde **tek eğitim süreci,
`num_workers=0`** çalışmasıdır. CI, Ubuntu üzerinde CPU testlerini çalıştırır.

## Kurulum

Python 3.11 veya üzeri önerilir. Linux CPU kurulumu:

```sh
git clone https://github.com/ulusluyz/ZERO-LLM.git
cd ZERO-LLM
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

CUDA kurulumu için sürücünüze uygun paketi [PyTorch'un resmi kurulum sayfasından](https://pytorch.org/get-started/locally/)
seçin. Depo GPU sürücüsü kurmaz. CPU'da iş parçacığı sınırı küçük modellerde daha
verimli olabilir; cihazınıza göre ayarlayın.

## İlk çalıştırma

Eğitimsiz çekirdekle üretim:

```sh
.venv/bin/python generate.py --prompt "Merhaba" --max_new_tokens 20
```

Küçük mekanizma testi; kesintiden sonra **aynı komut yeniden çalıştırılabilir**:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash run_training.sh \
  --preset tiny --epochs 2 --seq_len 64 --checkpoint_dir checkpoints/demo
```

`run_training.sh`, `train.py --resume_from auto` çalıştırır. Klasör boşsa sıfırdan
başlar; kayıt varsa en ileri **sağlam** kayıttan devam eder. Plan zaten tamamlanmışsa
optimizer adımlarını tekrarlamaz. Betik `.venv/bin/python` kullanır; başka ortam için
`ZERO_PYTHON=/tam/yol/python` tanımlanabilir.

Bu betik bilgisayar açılışında kendiliğinden başlamaz. Kesinti sonrası komutu yeniden
çalıştırmanız gerekir; systemd/başlangıç servisi kurulmaz.

## Gerçek verinizle eğitim

UTF-8 dosyalarda boş olmayan her satır bir belgedir:

```sh
bash run_training.sh \
  --corpus_file /veri/train.txt --val_file /veri/val.txt \
  --preset tiny --epochs 10 --seq_len 512 \
  --batch_size 4 --grad_accum_steps 4 \
  --checkpoint_dir checkpoints/turkce \
  --save_every_steps 100 --save_every_seconds 300 --keep_checkpoints 3
```

* `--val_file` verilmezse son yaklaşık %10 belge doğrulamaya ayrılır; en az iki belge
  gerekir. Aynı belge iki kümede bulunursa eğitim reddedilir. Bu kontrol yakın kopya
  veya anlamsal veri sızıntısı analizi değildir; veri kalitesi hâlâ kullanıcının işidir.
* Corpus verilmezse ayrı küçük demo eğitim/doğrulama metinleri kullanılır.
  Bu deney dil yeterliliği veya benchmark sonucu sayılmaz.
* Tokenizer yalnızca eğitim metninin ilk `--tokenizer_sample_chars` karakterinde
  (varsayılan 1.000.000) eğitilir. Sözlük hedefi üst sınırdır; frekansı ikiden düşük
  çiftler birleştirilmez, birleşimler boşluk sınırlarını aşmaz. Yeniden eğitim için
  yeni tokenizer oluşturulur; mevcut token kimlikleri sessizce değiştirilmez.
* Corpus satır satır işlenir; int64 token dosyaları bellek eşlemeyle okunur. Örnekler
  ihtiyaç halinde oluşturulur; son eksik pencere atılmaz. BPE eğitim örneği, belge
  hash kümesi ve örnek permütasyonu yine bellek kullanır. Bu, sınırsız ölçeklenen
  dağıtık veri altyapısı değildir; çok uzun tek satırlar da bellek tüketebilir.
* BOS/EOS eklenir. Girdiler gerçek pad kimliğiyle doldurulur; yapay hedefler `-100`
  ile cross-entropy dışında tutulur. Loss ve accumulation gerçek hedef token sayısına
  göre ağırlıklandırılır. Son eksik accumulation grubu da doğru ölçeklenir.
* `--seq_len` gerçek eğitim bağlamını ve kaydedilen model bağlamını birlikte belirler;
  verilmezse preset maksimumu kullanılır. Gizli 64-token eğitim sınırı yoktur.
* Cosine plan toplam optimizer adımıyla belirlenir, sonunda sıfırda kalır. AMP'nin
  taşma nedeniyle atladığı optimizer adımı scheduler'ı ilerletmez.
* `--dropout` isteğe bağlıdır; varsayılan `0.0`. `--mixed_precision` yalnız CUDA'da
  etkinleşir; gerçek CUDA/AMP testi bu sürüm için yapılmamıştır.

## Elektrik kesilirse / süreç kapanırsa

### Ne kaydedilir?

İlk eğitim adımından **önce**, ardından varsayılan olarak **100 başarılı optimizer
adımı veya 300 saniye** aralıkla (hangisi önce dolarsa) ve her epoch sonunda kayıt
alınır. Süre kontrolü yalnız tamamlanmış accumulation gruplarında yapılır; uzun bir
grup veya kayıt işlemi nedeniyle 300 saniye kesin bir üst sınır değildir.

Checkpoint, tek başına şu bilgileri taşır:

* Model ağırlıkları, tam config, tokenizer ve tokenizer SHA-256 kimliği.
* AdamW optimizer, scheduler ve AMP scaler durumları.
* Global optimizer adımı, mevcut epoch, sıradaki **batch konumu**, epoch loss/token
  sayaçları. Batch konumu modelden bağımsız deterministik epoch permütasyonuyla
  yeniden oluşturulur; önceki batch'ler yeniden eğitilmez.
* Python, NumPy, PyTorch CPU ve mevcutsa CUDA RNG durumları; dropout akışı da korunur.
* Eğitim/doğrulama token hash'leri, eğitim planı, sürüm/cihaz kontrolleri ve veri
  dosyalarının mutlak yolları. **Corpus içeriği checkpoint içine kopyalanmaz.**

### Nasıl devam edilir?

```sh
# Demo veya daha önce kaydedilen veri yolları ile:
bash run_training.sh --checkpoint_dir checkpoints/turkce

# Aynı işlemin açık Python karşılığı:
.venv/bin/python train.py --checkpoint_dir checkpoints/turkce --resume_from auto

# Belirli bir dosyadan:
.venv/bin/python train.py --checkpoint_dir checkpoints/turkce \
  --resume_from checkpoints/turkce/checkpoint_e000000_b00000400_s000000000100_ORNEK.pt
```

Örnekteki dosya adı yer tutucudur; eğitimde yazdırılan gerçek adı kullanın.
Verilmeyen matematiksel eğitim ayarları, config ve veri yolları kayıttan alınır.
Veri taşındıysa yeni `--corpus_file` / `--val_file` yolları verilebilir; token hash'leri
aynı olmalıdır. Açıkça farklı verilen plan/ayarlar veya değişen veri reddedilir.

`--epochs`, **ek epoch sayısı değil toplam planlanan epoch sayısıdır**. Kayıtlı
cosine planı değiştirerek yeni bir eğitim evresi başlatma bu resume akışının amacı
değildir. Aynı PyTorch/Python sürümü ve cihaz türü kontrol edilir; farklı GPU,
platform veya kernel arasında bit düzeyinde eşitlik garantisi verilmez.

### Kaydın yarıda kalması nasıl ele alınır?

1. Kayıt aynı klasörde geçici dosyaya yazılır; `flush` ve dosya `fsync` uygulanır.
2. SHA-256 yan dosyası da diske senkronlanır. İki dosya benzersiz son isimlerine
   `os.replace` ile taşınır; POSIX'te klasöre de `fsync` uygulanır.
3. Bunlar tek bir iki-dosya atomik işlem değildir. Arada kesinti olursa kayıp veya
   uyuşmayan checksum nedeniyle yeni kayıt reddedilir; eski benzersiz kayıt korunur.
4. `auto`, epoch/batch konumuna göre en ileri kaydı doğrular. Bozuk veya yarım kaydı
   uyarı vererek atlar, önceki sağlam kayda döner. Hepsi bozuksa **hata verir;
   sessizce rastgele ağırlıklardan başlamaz**. `.tmp` dosyaları kayıt sayılmaz.
5. Başarılı yeni kayıt doğrulandıktan sonra varsayılan son üç checksum-doğrulanmış
   kayıt tutulur. `--keep_checkpoints` en az ikidir. Bozuk/eksik dosyalar inceleme
   için yerinde kalabilir; bunlar sağlam yedek kotasına dahil değildir.
6. Aynı kayıt klasörünü iki eğitim sürecinin birlikte kullanması OS kilidiyle
   engellenir. Süreç zorla öldürülürse kilit OS tarafından bırakılır.

Checksum, kazara bozulmayı saptar; imza veya güvenilmeyen dosyalar için güvenlik
sınırı değildir. Checkpoint'i ve yanındaki `.pt.sha256` dosyasını **birlikte** taşıyın.

### Açık sınırlar

* Ani elektrik kesintisi / `SIGKILL` yeni kayıt alma fırsatı vermez. **Son sağlam
  checkpoint'ten sonraki, diske kaydedilmemiş adımlar kaybolur.** Devam sırasında
  bunlar yeniden hesaplanır. Accumulation ortasındaki kısmi gradyanlar saklanmaz.
* `Ctrl+C` / `SIGTERM`, ilk kez yakalandıktan sonra sıradaki tamamlanmış accumulation
  grubunda güvenli kayıt ve çıkış ister; devam eden uzun işlem bitmeden anında çıkmaz.
* En az ilerleme kaybı için `--save_every_steps 1` seçilebilir; her adımda tam model
  ve optimizer yazmak özellikle büyük modellerde ciddi IO ve disk maliyeti getirir.
* `fsync` ve önceki kayıtlar riski azaltır; disk/denetleyici arızası, dosya sistemi
  sorunu veya diskin tamamının kaybına karşı mutlak garanti sağlamaz. Aynı diskteki
  üç kayıt ayrı fiziksel yedek değildir. Fiziksel elektrik kesintisi testi yapılmadı.
* Disk dolması / IO hatası kayıt hatası olarak görünür; başarı mesajı verilmez ve
  eski kayıtlar yeni kayıt başarıyla tamamlanmadan silinmez.
* Zorla öldürme geride `.checkpoint-*.tmp` / `.tokens-*` bırakabilir. Eğitim durmuşken
  bu geçici dosyalar temizlenebilir; `.pt` ve `.pt.sha256` çiftlerini silmeyin.

## Üretim

```sh
.venv/bin/python generate.py --checkpoint /tam/yol/checkpoint.pt \
  --prompt "Merhaba" --max_new_tokens 30 --temperature 0.7 --top_k 40 --top_p 0.9
```

Config ve tokenizer checkpoint'ten yüklenir. Ayrı tokenizer verilirse yalnız boyutu
değil tam kimliği doğrulanır. `temperature=0` greedy üretimdir. EOS üretimi durdurur;
boş prompt CLI'da BOS ile başlar. Geçersiz parametreler ve dosya yolları hata verir.

SDPA ve GQA uyumlu KV cache kullanılır; cache yalnız KV head'lerini saklar.
`--no_cache` referans yolu sağlar. Pencere dolunca kaydırılan bağlam yeniden
hesaplanır; böylece RoPE konumları önbelleksiz yolla aynı kalır. Bu noktadan sonra
her token için tam pencere maliyeti geri gelir; sınırsız bağlam desteği yoktur.

## Mimari ve gerçek boyutlar

```text
Tokenizer → token embedding → N × TransformerBlock → RMSNorm → LM head

Her blok:
x = x + causal_attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Attention: `softmax(QKᵀ / sqrt(head_dim) + causal_mask) V`.
RoPE, Q/K çiftlerini konuma bağlı döndürür. SwiGLU:
`down_proj(SiLU(gate_proj(x)) * up_proj(x))`. Residual projeksiyonlar bir kez
`0.02 / sqrt(2 * num_layers)` standart sapmasıyla başlatılır.

Preset adları tarihsel takma adlardır; tam parametre sayısı oldukları iddia edilmez.
Tanımlı vocabulary ve ayrı embedding/LM head için:

| Preset | Parametre | Maksimum bağlam |
| --- | ---: | ---: |
| tiny | 1.047.680 | 512 |
| 100m | 134.105.856 | 2.048 |
| 300m | 373.867.520 | 2.048 |
| 500m | 777.856.512 | 2.048 |
| 1b | 1.574.045.696 | 2.048 |

Tokenizer vocabulary'si ve `tie_embeddings` gerçek sayıyı değiştirir. Eğitim başında
parametre sayısı ve gerçek bağlam yazdırılır. Büyük preset'ler bu ortamda eğitilmedi;
bellek veya performans taahhüdü verilmez. Özellikle optimizer ve çoklu checkpoint
kopyaları için yeterli disk alanı gerekir.

## Dosyalar ve uyumluluk

| Dosya | Görev |
| --- | --- |
| `config.py`, `model.py`, `transformer_block.py` | Config doğrulama ve model |
| `attention.py`, `rope.py`, `rmsnorm.py`, `mlp.py` | Transformer matematiği |
| `tokenizer.py`, `dataset.py` | Byte-BPE, özel token'lar ve maskeli hedefler |
| `train.py`, `recovery.py` | Eğitim, deterministik batch konumu, sinyal ve kilit |
| `checkpoint.py` | Dayanıklı kayıt, checksum, otomatik bulma ve saklama |
| `inference.py`, `generate.py` | Örnekleme ve KV cache'li üretim |
| `run_training.sh` | Aynı komutla otomatik devam |
| `tests/`, `.github/workflows/tests.yml` | Yerel testler ve CPU CI |
| `requirements.txt` | Bağımlılık aralıkları |

Checkpoint v3, batch konumundan devam etmek için gereken yeni biçimdir. Önceki yerel
v2 paketleri config/tokenizer içeriyorsa **üretimde** okunabilir, fakat yeni veri
sırası sözleşmesi nedeniyle eğitim resume için reddedilir. İlk depodaki config ve
tokenizer içermeyen eski kayıtlar üretim CLI'sında da açık hatayla reddedilir.
Yeniden eğitim sıfırdan başlar; eski kayıtlarda olmayan bilgiler uydurulmaz.

## Testler ve bilinen eksikler

CPU testleri; çekirdek matematiği, padding, son eksik accumulation grubu, scheduler,
RNG/optimizer aktarımı, epoch ve batch ortası resume, dropout eşdeğerliği, gerçek
SIGINT/SIGTERM/SIGKILL sonrası toparlanma, bozuk kayda geri dönüş, disk hatası
simülasyonu, KV cache ve bağlam sınırını kapsar. Kontrol amaçlı güvenli durdurma:

```sh
bash run_training.sh --epochs 2 --seq_len 64 --checkpoint_dir checkpoints/kesinti \
  --stop_after_step 2
bash run_training.sh --checkpoint_dir checkpoints/kesinti
```

Büyük Türkçe corpus ile dil kalitesi, perplexity/benchmark karşılaştırmaları,
dağıtık eğitim, çoklu worker kurtarma, Windows kurtarma, gerçek GPU/AMP ve fiziksel
kesinti davranışı doğrulanmış değildir. Eğitim verisinin lisansı, kalite kontrolü,
yakın kopya temizliği ve bağımsız benchmark hazırlığı ayrıca ele alınmalıdır.

## Lisans

[MIT](LICENSE.md).

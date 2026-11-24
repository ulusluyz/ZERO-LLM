# Değişiklik kaydı

## 2026-11-24 — Çekirdek doğrulama ve kesintiye dayanıklı eğitim

Bu güncelleme ilk depodaki çekirdeğin doğruluk sorunlarını ve eğitim/kayıt akışını
düzeltir. Pretrained ağırlık veya eğitilmiş Türkçe model yayımlamaz.

### Düzeltilenler

- Küçük corpus'un tek tokene dönüşmesi; düşük frekanslı BPE birleşimleri ve
  boşluklar arasında sınırsız birleşme engellendi.
- `<unk>` ile padding ve padding hedefleri üzerinde loss; gerçek pad kimliği,
  `ignore_index=-100`, korunan son pencere ve token ağırlıklı metrikler eklendi.
- Eğitim/doğrulama için aynı örneğin kullanılması kaldırıldı; belge bazında ayırma,
  ayrı doğrulama dosyası ve tam belge kopyası kontrolü eklendi.
- Eksik son accumulation grubunun ölçeklenmesi, sabit 100 adımlık cosine plan ve
  AMP'nin atladığı adımlarda scheduler'ın ilerlemesi düzeltildi.
- Residual projeksiyonların her alt modülde yeniden başlatılması kaldırıldı.
- Yanlış corpus/tokenizer/checkpoint yolunda sessiz fallback kaldırıldı.
- Eğitimde gizli 64-token sınırı kaldırıldı; gerçek bağlam config'e kaydediliyor.
- Model boyutu etiketlerinin yaklaşık adlar olduğu ve gerçek sayılar belgelendi.

### Eklenenler

- Config/tokenizer/hash, optimizer, scheduler, scaler, RNG ve veri kimliği içeren
  bağımsız checkpoint paketleri; checkpoint-only üretim ve tokenizer uyuşmazlık hatası.
- v3 checkpoint: epoch ortası batch konumu, deterministik veri sırası, loss/token
  sayaçları ve global optimizer adımıyla devam.
- İlk adım öncesi, adım/süre bazında, epoch sonunda ve kontrollü duruşta kayıt.
- Dosya/klasör fsync, benzersiz dosya isimleri, SHA-256 yan dosyası, yarım/bozuk
  kayıttan önceki sağlam kayda dönüş ve yapılandırılabilir en az iki sağlam yedek.
- SIGINT/SIGTERM için accumulation sınırında kayıt; SIGKILL sonrası önceki kayıttan
  devam ve otomatik bırakılan OS eğitim kilidi.
- `--resume_from auto`, ayar/veri yolu geri yükleme, tamamlanan planı tekrar etmeme,
  `run_training.sh`, `--stop_after_step` ve `--stop_after_epoch`.
- Akış halinde corpus tokenizasyonu, bellek eşlemeli token'lar ve lazy dataset.
- SDPA, GQA uyumlu KV cache, cache'siz referans yoluyla eşdeğer pencere davranışı.
- EOS, boş prompt ve config/RoPE/örnekleme parametre kontrolleri.
- Bağımlılık dosyası, Ubuntu CPU CI ve kesinti/hata enjeksiyonu testleri.

### Yerel doğrulama

Linux / Python 3.13.5 / PyTorch 2.15.0+cpu ile 74 test geçti (22,13 saniye).
SIGINT/SIGTERM/SIGKILL gerçek alt süreçlerde uygulandı. Bu sonuç fiziksel elektrik
kesintisi veya gerçek GPU/AMP testi anlamına gelmez.

### Açık sınırlar

Son sağlam kayıttan sonraki yazılmamış ilerleme ani kesintide kaybolur. Checkpoint
iki dosyasıyla birlikte taşınır; aynı diskteki yedekler fiziksel disk yedeği değildir.
Fiziksel elektrik kesintisi, Windows, gerçek GPU/AMP ve büyük ölçekli dil kalitesi
doğrulanmadı. Yeni eğitim resume v3 ister; v2 paketleri yalnız yeterli config/tokenizer
varsa üretimde kullanılabilir. Ayrıntılar [README.md](README.md) içinde açıklanır.

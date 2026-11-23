> وضعیت موتور Studio: تولید جدید با **Wan2.2-Animate-14B** در حالت Replacement انجام می‌شود.
> Qwen تصویر لباس را می‌سازد؛ SAM3 ماسک بدن، صورت و مو را جداگانه استخراج می‌کند؛ ViTPose حرکت و برش متحرک صورت را برای Animate آماده می‌کند.
> در مرحلهٔ انتخاب بازیگر، دکمه‌های بدن/صورت/مو ماسک هر بخش را نمایش می‌دهند. برای پروژه‌های قدیمی ردیابی را تکرار کنید. نمایش این ماسک‌ها به معنی چسباندن ثابت صورت روی خروجی نیست.
> اجرای Animate از BF16 با حفظ بخش‌های FP32 استفاده می‌کند؛ کوانتیزه‌سازی ۸بیتی و افزایش کیفیت SeedVR2 در این مسیر فعال نیستند. T5 و VAE موجود دوباره استفاده می‌شوند.
> تغییر مدل حفظ کامل هویت را تضمین نمی‌کند. نخست یک بخش کوتاه را بررسی کنید. اطلاعات SCAIL-2 در ادامه، مربوط به موتور قدیمی است.

# video_cloth_swap

تعویض لباس بازیگر در ویدیو، مخصوص فیلم‌های واقعی و صحنه‌های اکشن (نه فقط شرایط استودیویی).

| مرحله | مدل | کار |
|---|---|---|
| `masks` | **SAM3** (transformers) | ردیابی بازیگر در کل کلیپ، با پیوند دوبارهٔ ID بعد از اکلوژن، به‌اضافهٔ ماسک صورت (و در صورت نیاز مو و لباس) |
| `keyframe` | **Qwen-Image-Edit-2511** | انتخاب بهترین فریم (بزرگ، واضح، بریده‌نشده، با صورت) و پوشاندن لباس جدید در آن فریم |
| `scail` | **SCAIL-2** (حالت replacement) | بازسازی بازیگر در کل ویدیو با لباس جدید، بدون نیاز به اسکلت |
| `upscale` | Lanczos یا **SeedVR2** | بردن خروجی از رزولوشن کاری به رزولوشن اصلی |
| `composite` | — | گذاشتن ناحیهٔ بازیگر روی فریم‌های اصلی، نگه داشتن صورت و پس‌زمینهٔ اصلی، و اضافه کردن صدا |

خروجی هر مرحله در پوشهٔ کاری ذخیره می‌شود. پس می‌شود از هر مرحله دوباره شروع کرد؛ مثلاً SCAIL-2 را با تنظیمات دیگر اجرا کرد بدون اینکه ماسک‌ها و فریم کلیدی دوباره ساخته شوند.

## استودیو (رابط گرافیکی)

برای کار راحت، یک نرم‌افزار گرافیکی تحت وب هم هست. همهٔ مراحل را قدم‌به‌قدم جلوی کاربر انجام می‌دهد و هر مرحله قابل ویرایش است. اجرایش با یک دوبار کلیک روی `studio.bat` است، یا با:

```bash
.venv/Scripts/python.exe -m cloth_swap.studio
```

بعد مرورگر روی `http://127.0.0.1:7860` باز می‌شود. رابط به‌صورت پیش‌فرض فارسی و راست‌چین است و با یک دکمه انگلیسی می‌شود.

| مرحله | کاربر چه کار می‌تواند بکند |
|---|---|
| ۱. ویدیو | دیدن کلیپ و انتخاب بازهٔ کار با دستگیره‌های تایم‌لاین |
| ۲. بازیگر | تشخیص خودکار افراد و انتخاب با یک کلیک؛ یا کلیک «افزودن/حذف» و کشیدن کادر. ماسک در لحظه دیده می‌شود (حدود ۰.۲ ثانیه). بعد از ردیابی، روی هر فریمی که ماسک اشتباه است اصلاح و دوباره ردیابی |
| ۳. لباس | آپلود عکس لباس (جلو/پشت) یا فقط توضیح متنی؛ توضیح بازیگر و صحنه |
| ۴. فریم مرجع | فریم‌های پیشنهادی هوشمند یا هر فریم دلخواه؛ ساخت ۱ تا ۴ نسخه؛ مقایسهٔ قبل/بعد با اسلایدر؛ ویرایش دستور و seed؛ یا آپلود فریم ویرایش‌شدهٔ خود کاربر |
| ۵. ساخت ویدیو | انتخاب کیفیت (پیش‌نمایش/استاندارد/بالا) با زمان تخمینی؛ Relight؛ پیشرفت زنده و امکان لغو |
| ۶. بازبینی | پخش هم‌زمان قبل/بعد با اسلایدر؛ تنظیم حفظ صورت/مو، نرمی لبه و گسترش ناحیه و اعمال سریع؛ دانلود |

پروژه‌ها در `studio_data/` ذخیره می‌شوند و هر وقت دوباره باز شوند از همان مرحله ادامه پیدا می‌کنند.

معماری:

- **بک‌اند:** FastAPI در `cloth_swap/studio/`.
- **فرانت‌اند:** React + TypeScript + Tailwind در `studio-ui/`.
- **مدل سبک (SAM3 کلیکی):** داخل سرور می‌ماند تا هر کلیک کمتر از ۰.۲ ثانیه طول بکشد.
- **مدل‌های سنگین (ردیابی ویدیو و Qwen):** در یک پروسهٔ worker جدا اجرا می‌شوند. این worker قبل از SCAIL-2 بسته می‌شود تا همهٔ RAM و VRAM برگردد. بدون این کار، روی سیستم ۳۲ گیگی SCAIL-2 به swap می‌افتاد.
- **cache پرامپت:** encoding پرامپت با T5 در `.cache/t5` ذخیره می‌شود. ساخت دوباره با همان پرامپت دیگر T5 را بارگذاری نمی‌کند (روی هارد مکانیکی ۶ دقیقه صرفه‌جویی).

برای توسعهٔ رابط، `npm run dev` را در `studio-ui/` اجرا کنید (پورت ۵۱۷۳؛ درخواست‌های `/api` به سرور ۷۸۶۰ می‌روند) و بعد از تغییرات `npm run build`.

## نصب

> SCAIL-2 و SeedVR2 روی لینوکس توسعه داده شده‌اند. روی ویندوز، **WSL2** توصیه می‌شود.
> کارت‌های سری RTX 50 به PyTorch با CUDA 12.8 یا بالاتر نیاز دارند.

### ۱. محیط اصلی (همین پروژه)

```bash
python -m venv .venv
```

PyTorch را مطابق CUDA خودتان نصب کنید، بعد بقیهٔ وابستگی‌ها را:

```bash
pip install -r requirements.txt
```

SAM3 روی Hugging Face gated است. یک بار لایسنس `facebook/sam3` را در سایت قبول کنید و بعد وارد شوید:

```bash
hf auth login
```

### ۲. محیط SCAIL-2 (جداگانه)

```bash
git clone -b wan-scail2 https://github.com/zai-org/SCAIL-2
```

```bash
cd SCAIL-2 && pip install -r requirements.txt
```

```bash
hf download zai-org/SCAIL-2 --local-dir /models/SCAIL-2
```

```bash
python convert.py --scail-dir /models/SCAIL-2 --save-path /models/SCAIL-2.safetensors
```

LoRA سرعت lightx2v (فایل `lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors` از `Kijai/WanVideo_comfy`) را دانلود کنید.

LoRA نور (relighting) هم اختیاری است. فایل `model/relighting-lora.pt` را از `zai-org/SCAIL-2` بگیرید و تبدیلش کنید:

```bash
python convert_lora.py --input relighting-lora.pt --output relighting_lora.safetensors
```

### ۲-ب. سیستم با RAM و VRAM حدود ۳۲ گیگ (مثل RTX 5090)

checkpoint رسمی SCAIL-2 در fp32 است (۶۶ گیگ) و loader خود repo کل مدل را در RAM می‌سازد. روی چنین سیستمی جا نمی‌شود. راه‌حل این پروژه:

- **وزن DiT:** فایل `wan2.1_14B_SCAIL_2_fp8_scaled.safetensors` از `Comfy-Org/SCAIL-2` (۱۸ گیگ). نام و شکل tensorهایش دقیقاً با خروجی `convert.py` یکی است، پس تبدیل لازم نیست.
- **`scail.weights: fp8`:** runner وزن‌ها را tensor به tensor مستقیم روی GPU می‌برد. لایه‌های خطی ۴۰ بلوک به‌صورت float8 با scale جداگانه برای هر ردیف می‌مانند (حدود ۱۶ گیگ VRAM)، و LoRAها در همین مرحله ادغام می‌شوند.
- **بدون flash-attn (ویندوز):** attention خودکار به `torch.scaled_dot_product_attention` برمی‌گردد.
- **Qwen-Image-Edit-2511:** نسخهٔ از قبل ۴ بیتی‌شدهٔ `ovedrive/Qwen-Image-Edit-2511-4bit` (۱۷ گیگ به‌جای ۵۸ گیگ)، با `qwen.quantization: prequantized`.

همهٔ این مدل‌ها (حدود ۵۵ گیگ) با یک اسکریپت دانلود می‌شوند:

```bash
bash scripts/download_models.sh D:/models
```

نمونهٔ تنظیمات کامل این حالت در `configs/local.yaml` است.

### ۳. فایل تنظیمات خودتان

یک فایل `my.yaml` بسازید. فقط کلیدهایی که تغییر می‌دهید لازم است؛ بقیه از `configs/default.yaml` می‌آیند.

```yaml
scail:
  python: /home/me/envs/scail2/bin/python
  repo_dir: /home/me/SCAIL-2
  ckpt_dir: /models/SCAIL-2
  scail_path: /models/SCAIL-2.safetensors
  loras:
    - {path: /models/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors, alpha: 1.0}
    - {path: /models/relighting_lora.safetensors, alpha: 1.0}
```

## اجرا

حالت پایه: بزرگ‌ترین آدم صحنه انتخاب می‌شود و لباسش با لباس عکس عوض می‌شود.

```bash
python -m cloth_swap --config my.yaml --video fight.mp4 --garment jacket.png --garment-desc "a black leather biker jacket" --scene-desc "He fights two men in a dark warehouse, fast handheld camera." --out result.mp4
```

وقتی چند نفر در صحنه هستند، روی بازیگر کلیک کنید (مختصات پیکسل در فریم `--select-frame`):

```bash
python -m cloth_swap --config my.yaml --video fight.mp4 --garment jacket.png --garment-desc "a black leather biker jacket" --point 812 430 --select-frame 0 --out result.mp4
```

برای بررسی فریم کلیدی قبل از مرحلهٔ سنگین SCAIL-2، بعد از مرحلهٔ keyframe متوقف شوید. خروجی در `work/fight/02_keyframe/` ذخیره می‌شود.

```bash
python -m cloth_swap --config my.yaml --video fight.mp4 --garment jacket.png --garment-desc "a black leather biker jacket" --stop-after keyframe --out result.mp4
```

اگر فریم کلیدی را خودتان با ابزار دیگری ساخته‌اید (مثلاً Nano Banana)، همان فریم را بدهید تا Qwen اجرا نشود:

```bash
python -m cloth_swap --config my.yaml --video fight.mp4 --keyframe-image edited_frame_40.png --keyframe-frame 40 --garment-desc "a black leather biker jacket" --out result.mp4
```

برای اجرای دوبارهٔ SCAIL-2 با کیفیت بالاتر (بدون lightx2v، ۴۰ قدم)، بدون ساختن دوبارهٔ ماسک‌ها و فریم کلیدی:

```bash
python -m cloth_swap --config my.yaml --video fight.mp4 --garment jacket.png --from-stage scail --set "scail.loras=[]" --set scail.steps=40 --set scail.shift=3.0 --set scail.guide_scale=5.0 --out result_hq.mp4
```

## تنظیمات مهم

| کلید | توضیح |
|---|---|
| `scail.steps` / `shift` / `guide_scale` | با lightx2v: `8 / 1.0 / 1.0` (سریع). بدون آن: `40 / 3.0 / 5.0` (کیفیت بالاتر) |
| `scail.target_area` | تعداد پیکسل‌های رزولوشن کاری؛ پیش‌فرض 704×1280. برای 1920×800 می‌شود 1472×608 |
| `scail.replace_dilate` / `replace_temporal` | بزرگ‌تر کردن ماسک برای لباس‌های گشادتر و حرکت‌های سریع (motion blur) |
| `composite.mode` | `actor`: کل بازیگر عوض می‌شود. `garment`: فقط ناحیهٔ لباس قدیم و جدید (با `sam3.old_garment_prompt` و `sam3.new_garment_prompt`). `full`: کل فریم SCAIL-2 |
| `composite.keep_face` | صورت اصلی بازیگر دقیقاً حفظ می‌شود |
| `keyframe.num_extra` | فریم‌های کلیدی اضافه (مثلاً از پشت) به‌عنوان مرجع. در SCAIL-2 آزمایشی است |
| `qwen.quantization` | `nf4` برای کارت‌های ۲۴ تا ۳۲ گیگابایتی |
| `upscale.backend` | `seedvr2` برای بازسازی جزئیات. دستور اجرایش در `upscale.seedvr2.command` قابل تنظیم است |
| `scail.prompt_enhancer` | پرامپت را با اسکریپت خود SCAIL-2 بازنویسی می‌کند. **فریم‌ها را به Gemini می‌فرستد** و `GEMINI_API_KEY` لازم دارد |

SCAIL-2 با پرامپت‌های بلند و دقیق آموزش دیده است. هر چه `--subject-desc`، `--garment-desc` و `--scene-desc` دقیق‌تر باشند، نتیجه بهتر است. یا کل پرامپت را با `--prompt` بدهید.

## پوشهٔ کاری (`work/<نام ویدیو>/`)

```
01_masks/     actor.npz, face.npz, preview.mp4
02_keyframe/  original_0.png, edited_raw_*.png, keyframe_0.png, ref_mask_0.npz
03_scail/     driving.mp4, replace_mask.mp4, ref_0.png, ref_mask_0.png, prompt.txt, scail_out.mp4
04_upscale/   generated_full.mp4
05_composite/ composite_silent.mp4, side_by_side.mp4
meta.json, config_used.yaml, <stage>.done
```

## نکته‌ها برای صحنه‌های اکشن

- کلیپ‌های طولانی را به برش‌های ۵ تا ۲۰ ثانیه‌ای تقسیم کنید (`--start-frame` / `--max-frames`). SCAIL-2 داخل هر اجرا تکه‌های ۸۱ فریمی با همپوشانی ۵ فریم می‌سازد.
- فایل `01_masks/preview.mp4` را چک کنید. اگر ماسک بازیگر به نفر دیگری پرید، از `--point` یا `actor.mode=track` استفاده کنید.
- فریم کلیدی باید لباس را کامل و واضح نشان دهد. اگر فریم خودکار مناسب نبود، با `--keyframe-frame` عوضش کنید.
- اگر لباس جدید از قبلی بزرگ‌تر است (مثلاً پالتو به جای تی‌شرت)، `scail.replace_dilate` و `composite.dilate` را بیشتر کنید.

## نتیجهٔ تست روی این سیستم

سخت‌افزار: RTX 5090 (۳۲ گیگ)، ۳۲ گیگ RAM، مدل‌ها روی هارد مکانیکی.

کلیپ: نمونهٔ replacement خود SCAIL-2 (`test_data/violin.mp4`)، ۸۱ فریم، 1280×720.

لباس جدید: «کت چرمی قرمز روی تی‌شرت سفید» (فقط متن). رزولوشن کاری 896×512، lightx2v با ۸ قدم.

| مرحله | زمان |
|---|---|
| ماسک (SAM3، با بارگذاری مدل) | ۱۱۰ ثانیه |
| فریم کلیدی (Qwen-Image-Edit-2511 چهاربیتی) | ۵ دقیقه |
| SCAIL-2: T5 | ۸۰ ثانیه |
| SCAIL-2: بارگذاری DiT از هارد | حدود ۶ دقیقه |
| SCAIL-2: sampling (۳۱ ثانیه در هر قدم) | ۴ دقیقه |
| SCAIL-2: decode | چند ثانیه |
| upscale (Lanczos) و composite | ۱۱ ثانیه |

اوج مصرف GPU در SCAIL-2: ۲۵.۸ گیگ.

خروجی‌ها: `work/violin/result.mp4` و مقایسهٔ کنار‌به‌کنار در `work/violin/05_composite/side_by_side.mp4`.

## نکته‌های اجرا روی ویندوز و کارت ۳۲ گیگی

- **سرریز حافظهٔ GPU:** درایور NVIDIA روی ویندوز وقتی VRAM پر شود، به‌جای خطای OOM حافظه را به RAM سیستم می‌برد (sysmem fallback). نتیجه فقط این است که اجرا ده‌ها برابر کند می‌شود. برای جلوگیری، runner این کارها را می‌کند:
  - RoPE را تکه‌تکه و در complex64 محاسبه می‌کند.
  - FFN را chunked اجرا می‌کند.
  - T5 را قبل از DiT اجرا و آزاد می‌کند.
  - CLIP را بعد از استفاده و DiT را موقع decode در VAE به CPU می‌برد.
- **اگر باز هم کند بود:** `scail.target_area` را کم کنید. سرریز را در Task Manager یا با شمارندهٔ `GPU Process Memory\Shared Usage` می‌شود دید.
- **ترتیب import:** روی ویندوز `decord` نباید قبل از `torch` import شود، وگرنه بارگذاری `c10.dll` خطا می‌دهد. runner این ترتیب را رعایت می‌کند.
- **سرعت دیسک:** خواندن ۱۸ گیگ وزن DiT از هارد مکانیکی چند دقیقه طول می‌کشد. اگر جا دارید، `D:/models` را روی SSD بگذارید.

## تست‌ها

تست‌ها بدون GPU و بدون مدل اجرا می‌شوند. منطق ماسک، طول تکه‌های SCAIL-2، رنگ ماسک‌ها بعد از انکود، composite و کل پایپ‌لاین را با مدل‌های جعلی چک می‌کنند:

```bash
python -m pytest -q tests
```

## لایسنس مدل‌ها

| مدل | لایسنس |
|---|---|
| SCAIL-2 | کد Apache-2.0، کارت مدل MIT |
| Qwen-Image-Edit-2511 | Apache-2.0 |
| SeedVR2 | Apache-2.0 |
| SAM3 | لایسنس اختصاصی Meta؛ قبل از استفادهٔ تجاری بخوانید |
| lightx2v | کارت مدل را چک کنید |

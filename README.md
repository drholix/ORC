# OCR Pipeline (PaddleOCR + OpenCV)

A modular Python toolkit that turns scanned documents, receipts, identity cards, and UI screenshots into structured text. The solution uses PaddleOCR for detection/recognition combined with an OpenCV-based preprocessing pipeline optimised for laptops.

## What you get (fitur utama)

- **Snipping OCR super ringan** – jalankan `python -m app.cli snip`, seleksi area layar seperti screenshot tool, teks langsung tampil dan tersalin ke clipboard. Ini inti kebutuhan awal proyek (image ➜ text secepat mungkin).
- **Perpustakaan modular** – folder `app/` berisi modul kecil (preprocess, inference, postprocess, pdf, tables, cache) sehingga mudah dipahami pemula dan gampang diutak-atik jika ingin belajar.
- **CLI serbaguna** – `python -m app.cli run --input ...` untuk memproses file tunggal, folder, URL, atau PDF multi-halaman tanpa harus menulis kode Python.
- **REST API siap pakai** – FastAPI menghadirkan endpoint `/ocr`, `/healthz`, dan `/metrics` sehingga bisa diintegrasikan ke aplikasi lain (mis. dashboard internal atau bot).
- **Batch & caching otomatis** – worker pool + SQLite cache memastikan pemrosesan massal tetap cepat, tidak mengulang OCR untuk file yang sama.
- **Dukungan dokumen kompleks** – mode tabel sederhana, ekstraksi teks PDF, dan flag handwriting (memberi peringatan kalau model khusus belum tersedia).
- **Peralatan tuning** – skrip benchmarking, evaluasi CER/WER, serta profil performa untuk membantu belajar optimasi OCR.
- **Konfigurasi mudah** – semua parameter penting (bahasa, GPU, ukuran gambar, batas memori) tersimpan di `config.yaml`.
- **Model PaddleOCR 3.3 terbaru** – langsung memakai PP-OCRv5 multilingual (akurasi +40% di beberapa skenario) dan siap di-upgrade ke PaddleOCR-VL untuk parsing dokumen kompleks.

## Step-by-Step: Getting Productive Fast

The goal is to get a working OCR pipeline on a laptop without needing paid services.

### 1. Siapkan lingkungan kerja

```bash
git clone https://github.com/your-user/ocr-pipeline.git
cd ocr-pipeline
python -m venv .venv           # Python 3.10+ sangat dianjurkan
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Jalankan tes cepat (opsional tetapi direkomendasikan)

```bash
pytest
```

### 3. Siapkan contoh gambar (opsional tapi praktis)

Repo ini tidak menyertakan file biner. Untuk membuat gambar contoh lokal jalankan:

```bash
python scripts/generate_sample_image.py
```

Perintah tersebut menulis `sample_data/sample_text.png` menggunakan OpenCV sehingga Anda punya bahan uji tanpa perlu mengunduh apa pun.

### 4. Uji pipeline dengan contoh data

```bash
python -m app.cli run \
  --input sample_data/sample_text.png \
  --output result.json \
  --lang id,en \
  --min-conf 0.6

cat result.json
```

`result.json` akan memuat teks terdeteksi, bounding box, confidence, serta metadata durasi.

### 4. OCR Cepat dari layar (Snipping tool)

Ingin hasil OCR dari bagian layar apa pun seperti alat snipping bawaan OS? Ikuti panduan berikut.

#### 4.1. Langkah kilat lintas platform

```bash
pip install mss pyperclip  # jika belum ada
# CPU build PaddlePaddle (wajib untuk OCR asli)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/simple
python -m app.cli snip
```

Langkah-langkahnya:

1. Layar akan menjadi gelap semi-transparan. Drag area yang ingin di-OCR.
2. Lepas mouse → proses OCR berlangsung offline menggunakan PaddleOCR CPU.
3. Hasil teks tampil di jendela kecil dan otomatis masuk ke clipboard.

#### 4.2. Panduan rinci untuk Kali Linux (XFCE)

Langkah berikut diuji pada **Kali Linux 2024.x (rolling) dengan desktop XFCE** yang baru terpasang. Jika Anda memakai flavour lain (GNOME/Wayland), lihat catatan di akhir sub-bab.

**Gambaran cepat alurnya:**

| Tahap | Apa yang dilakukan | Perintah utama |
|-------|--------------------|----------------|
| 1 | Siapkan dependensi OS (Python, Tkinter, clipboard) | `sudo apt install ...` |
| 2 | Kloning repo & aktifkan virtualenv | `git clone`, `python3 -m venv`, `source .venv/bin/activate` |
| 3 | Pasang dependensi Python & add-on snipping | `pip install -r requirements.txt mss pyperclip` |
| 4 | Jalankan snip dan pilih area layar | `python -m app.cli snip` |
| 5 | Cek clipboard & lakukan troubleshooting ringan | `Ctrl+V`, cek `xfce4-clipman` |

Ikuti detail berikut bila Anda benar-benar mulai dari nol:

1. **Perbarui paket dan pasang prasyarat grafis.** Kali biasanya sudah punya Python, tetapi kita memastikan semua komponen Tk/clipboard tersedia:

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip python3-tk \
     xclip xfce4-clipman-plugin
   ```

   - `python3-tk` memunculkan jendela Tkinter (tanpa ini, snipping tidak tampil).
   - `xclip` adalah backend clipboard standar X11; `xfce4-clipman-plugin` menahan isi clipboard agar tidak hilang saat aplikasi ditutup.
   - Jika Anda sudah menggunakan Wayland, tambahkan `sudo apt install wl-clipboard`.

2. **Aktifkan Clipboard Manager di panel XFCE.** Klik kanan panel → `Panel → Add New Items...` → cari *Clipboard Manager* → `Add`. Pastikan statusnya *Running* (klik ikon gunting dua kali bila perlu). Ini membantu pemula yang belum pernah mengelola clipboard manual.

3. **Siapkan project di folder kerja Anda.**

   ```bash
   mkdir -p ~/projects && cd ~/projects
   git clone https://github.com/your-user/ocr-pipeline.git
   cd ocr-pipeline

   python3 -m venv .venv
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   pip install mss pyperclip
   # wajib: PaddleOCR membutuhkan paket paddlepaddle inti
   pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/simple
   # alternatif resmi (CPU):
   python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   ```

   Tips pemula:
   - Jika shell menampilkan “command not found: python3”, ulangi langkah 1 (Python belum terinstal).
   - Simpan perintah `source .venv/bin/activate` di akhir file `~/.bashrc` bila ingin virtualenv aktif otomatis setiap buka terminal proyek.

4. **Verifikasi akses layar.**
   - Jalankan `echo $XDG_SESSION_TYPE`. Bila hasilnya `x11`, Anda aman menggunakan `mss` langsung. Bila `wayland`, baca catatan di bawah untuk tambahan konfigurasi.
   - Jika menggunakan driver NVIDIA proprietary, buka *NVIDIA Settings → OpenGL Settings* dan centang *Allow Flipping* untuk mencegah layar hitam saat discreenshot.
   - Pastikan Anda menjalankan perintah dari terminal grafis (mis. `xfce4-terminal`) dan bukan TTY murni.

5. **Mulai OCR snipping.**

   ```bash
   python -m app.cli snip
   ```

   Urutan yang terjadi:
   1. Layar redup semi-transparan.
   2. Klik–drag area teks yang ingin dipanen.
   3. Lepas mouse → pipeline preprocessing + PaddleOCR berjalan di CPU.
   4. Jendela hasil muncul menampilkan teks; isi clipboard langsung diperbarui.

   > **Tip:** Jika Anda lupa memasang `paddlepaddle`, aplikasi tetap jalan namun memakai *dummy engine* sebagai placeholder dan akan menampilkan teks contoh beserta log peringatan. Pasang paket tersebut lalu jalankan ulang agar hasil OCR nyata keluar.

6. **Tes clipboard Anda.** Buka aplikasi teks ringan, misalnya *Mousepad* (`mousepad &`). Tekan `Ctrl+V`. Jika teks muncul, workflow selesai. Bila clipboard kosong:

   ```bash
   export DISPLAY=:0
   export XAUTHORITY=~/.Xauthority
   ```

   Jalankan ulang perintah `python -m app.cli snip` di terminal yang sama. Pastikan juga ikon *Clipboard Manager* berwarna (menandakan aktif).

7. **Troubleshooting khusus Kali/XFCE.**
   - **Jendela tidak muncul:** pastikan paket `python3-tk` terinstal dan Anda tidak menjalankan perintah via SSH tanpa forwarding X11.
   - **Hasil screenshot hitam:** buka `Settings Manager → Window Manager Tweaks → Compositor` dan aktifkan *Display compositing*. Jika masih hitam, matikan dulu compositing lalu hidupkan lagi.
   - **Clipboard hilang setelah aplikasi tertutup:** pastikan `xfce4-clipman` berjalan (ikon gunting aktif). Anda bisa menandai opsi “Persist Primary Selection” agar isi clipboard bertahan.
   - **Wayland session:** instal `wl-clipboard`, lalu sebelum menjalankan snip set `export WAYLAND_DISPLAY=wayland-0` dan jalankan aplikasi dari terminal Wayland (misal `kgx`).

**Catatan GNOME/Wayland:** Langkah 1–3 tetap berlaku, namun tambahkan `sudo apt install wl-clipboard gnome-screenshot` dan `pip install pillow>=10.0`. Pastikan variabel `XDG_SESSION_TYPE=wayland` terpasang dan jalankan terminal yang mendukung pipe ke Wayland (mis. *GNOME Console*). Jika `mss` masih gagal, Anda dapat sementara masuk ke sesi X11 dari layar login GNOME dengan memilih opsi “GNOME on Xorg”.

#### 4.3. Catatan tambahan

- Pada Windows/macOS, Tkinter sudah termasuk paket Python standar sehingga tidak perlu instal `python3-tk`.
- Jika ingin menjadikan perintah ini pintasan desktop, buat file `.desktop` yang menjalankan `python -m app.cli snip` di dalam virtualenv.

### 5. Jalankan sebagai REST API (FastAPI)

```bash
uvicorn app.api:create_app --factory --reload
```

Kemudian buka `http://127.0.0.1:8000/docs` untuk mencoba endpoint `/ocr` melalui Swagger UI.

Contoh permintaan `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
-F "file=@sample_data/sample_text.png" \
  -F "languages=id,en" \
  -F "min_confidence=0.6" \
  -o result.json
```

### 6. Jalankan batch folder atau PDF

Letakkan beberapa gambar/PDF di sebuah folder (mis. `sample_data/batch_inputs/`).

```bash
mkdir -p sample_data/batch_inputs
python scripts/generate_sample_image.py
cp sample_data/sample_text.png sample_data/batch_inputs/struk.png

python -m app.cli run \
  --input sample_data/batch_inputs \
  --output batch_results.json \
  --threads 4 \
  --pdf-text         # ekstrak teks embedded sebelum OCR
```

File keluaran akan berisi daftar hasil OCR untuk setiap halaman atau gambar.

### 7. Kontainer Docker (CPU)

Ingin mencoba tanpa mengotak-atik Python di laptop? Jalankan image Docker bawaan ini.

```bash
docker build -t ocr-app .
docker run -p 8000:8000 ocr-app
```

Setelah kontainer aktif:

- **Apa yang berjalan:** perintah `uvicorn app.api:create_app` otomatis menyalakan REST API FastAPI.
- **Isi image:** dependensi utama seperti PaddleOCR, OpenCV, FastAPI, PyPDF/pdf2image, serta utilitas `poppler-utils` sudah dipasang pada base image `python:3.11-slim`.
- **Cara mengakses:** buka `http://127.0.0.1:8000/docs` untuk Swagger UI, atau pakai `curl` contoh di atas.
- **Catatan GPU:** di akhir Dockerfile ada komentar cara mengganti base image NVIDIA CUDA jika nanti ingin akselerasi GPU tanpa mengubah struktur aplikasi.

### 8. Mode lanjutan: caching & konfigurasi

- Simpan perubahan Anda di `config.yaml` untuk mengatur bahasa, ukuran maksimum gambar, jumlah worker, hingga pemakaian GPU.
- Jalankan `python -m app.cli cache-warm --folder sample_data` untuk mengisi cache SQLite dan mempercepat proses berikutnya.
- Gunakan `scripts/benchmark.py` untuk mengukur latency p50/p95 dan throughput pada jumlah worker berbeda.
- Jalankan `scripts/evaluate_accuracy.py --pred result.json --truth sample_data/sample_text_gt.txt` untuk menghitung CER/WER jika memiliki ground-truth.

### 9. Memilih model PaddleOCR 3.3.0

Rilis PaddleOCR 3.3.0 (Oktober 2025) membawa dua upgrade besar: **PP-OCRv5** untuk pengenalan teks multilingual ultra-ringkas dan **PaddleOCR-VL** (Vision-Language) untuk parsing halaman kompleks (tabel, formula, diagram).

- **Default proyek ini** memakai `ocr_version: "PP-OCRv5"` sehingga langsung mendapatkan peningkatan akurasi Latin/Cyrillic/Arabic tanpa konfigurasi tambahan.
- Untuk mengganti model langsung dari CLI:

  ```bash
  python -m app.cli run \
    --input sample_data/sample_text.png \
    --ocr-version PP-OCRv5 \
    --lang id,en
  ```

- Jika Anda ingin mencoba PaddleOCR-VL:

  1. Pasang dependensi terbaru: `pip install --upgrade paddleocr>=3.3.0 paddlenlp`.
  2. Unduh bobot **PaddleOCR-VL-0.9B** dari HuggingFace lalu isi `config.yaml` dengan lokasi `det_model_dir`, `rec_model_dir`, dan `structure_version: PaddleOCR-VL`.
  3. Jalankan CLI atau snipping tool dengan opsi `--engine paddle --ocr-version PP-OCRv5` (untuk fallback) atau sesuaikan sesuai eksperimen.
  4. Mode ini cocok untuk dokumen penuh elemen (tabel/formula) sambil tetap ringan untuk laptop berkat arsitektur NaViT + ERNIE 4.5 mini.

- Gunakan flag `--engine dummy` jika hanya ingin menguji pipeline tanpa memuat model berat (berguna saat CI).

## Configuration

Edit `config.yaml` (auto-generated on first run) to tweak thresholds, GPU usage, language hints, and resource limits. Key options:

| Option | Description |
| ------ | ----------- |
| `languages` | Priority languages (e.g. `['id', 'en']`). |
| `min_confidence` | Filter out low-confidence lines. |
| `enable_gpu` | Toggle PaddleOCR GPU inference (auto falls back to CPU). |
| `ocr_version` | Pilih versi model PaddleOCR (`PP-OCRv5`, `PP-OCRv4`, dsb). |
| `det_model_dir` / `rec_model_dir` | Path custom weight (mis. PaddleOCR-VL dari HuggingFace). |
| `structure_version` | Aktifkan parser struktur dokumen (isi `PaddleOCR-VL` untuk VLM). |
| `threads` | Worker pool size for batch processing. |
| `max_image_size` | Longest edge allowed before downscaling. |
| `table_mode` | Enable table detection/cell cropping metadata. |
| `handwriting_mode` | Adds warnings and allows swapping to handwriting-capable engines. |

## Accuracy & Performance

- Measure **CER/WER** with your own labelled data and record results in `reports/ocr_eval.md`.
- Use `scripts/benchmark.py` to estimate latency (`p50`, `p95`) and throughput at different worker counts.
- Profile slow scenarios using `python profiling/profile_example.py sample_data/sample_text.png` (generates `ocr.prof`).

## Extending the Engine

`app/inference.py` exposes a `BaseOCREngine` protocol. To swap engines (e.g., EasyOCR, Tesseract), implement `.infer(image, lang)` and inject it into `OCRService`.

## Tips for High Quality OCR

- Scan at ≥300 DPI, avoid motion blur, and crop redundant background.
- Ensure even lighting; deskew images before OCR when possible.
- Prefer PNG/TIFF for lossless text rendering; for camera photos enable auto-rotate.
- Tweak preprocessing parameters in `app/preprocess.py` for specific document types.

## Troubleshooting

- **Slow startup**: PaddleOCR downloads weights on the first run. Cache them in `~/.paddleocr` for reuse.
- **Memory issues**: lower `max_image_size`, enable `downscale_on_oom`, or increase swap.
- **PDF errors**: install `poppler` system utilities required by `pdf2image`.
- **GPU usage**: install PaddleOCR GPU wheels and update `config.yaml` (`enable_gpu: true`).

## License

MIT License. See `LICENSE` for details.

# Fixing CUDA Out Of Memory _ Google AI Studio

> **المصدر:** https://aistudio.google.com/prompts/1OZZHSDk0vRMNHkdrKwkKE_0ghGqmmkZP
> **تاريخ التصدير:** 10/25/2026, 2:58:31 AM
> **إجمالي الفقرات/الرسائل:** 87

---

Fixing CUDA Out Of Memory

105,111 tokens

more_vert
User 1:21 AM
docs
docs
Generated File October 25, 2026 - 2:40AM.plain
15,178 tokens

more_vert
Model 1:21 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

The error CUDA out of memory occurs because the model weights alone exceed your GPU’s VRAM before the engine can even finish initializing.

Root Cause Analysis

Model Size vs. GPU VRAM:

GPU: 16 GB VRAM (~15.48 GiB usable).

Model: LLaMA-3.1-70B. Even at 2-bit quantization, 70 billion parameters require at least:

70
×
10
9
×
2
 bits
8
×
10
9
≈
17.5
 GB
8×10
9
70×10
9
×2 bits
	​

≈17.5 GB

Adding the token embeddings (embed_tokens) and output layer (lm_head), which are typically kept at 16-bit or 8-bit, along with quantization scales/offsets, the total weight footprint is ~18–22 GB.

Where it crashed: In create_weights inside humming.py while creating PyTorch weight tensors. PyTorch already had allocated 14.74 GiB of your 15.48 GiB and ran out of room while still assembling layers.

KV Cache & Context Window:

vLLM defaulted to --max-model-len 131072 (128k context). Even if the weights fit, 128k context requires gigabytes of additional VRAM for the KV cache.

How to Fix / Work Around It
Option 1: Run with llama.cpp (Recommended for single 16 GB GPUs)

Given your directory path is GSQ-RCO-GGUF, if you have a GGUF version of the model, llama.cpp is specifically built to offload part of the model to CPU system RAM:

code
Bash
download
content_copy
expand_less
# Example using llama-server (llama.cpp)
./llama-server -m /path/to/model.gguf -ngl 50 -c 4096

(Adjust -ngl to offload as many layers to your 16 GB GPU as will fit, while keeping the remaining layers in system RAM).

Option 2: Use CPU Offloading in vLLM

If your build/fork of vLLM supports CPU offloading, offload part of the weights to system RAM, drastically reduce the context window, and enable eager mode to eliminate CUDA graph overhead:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 8 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.95

Note: CPU offloading in vLLM will impact token generation speeds considerably.

Option 3: Multi-GPU (Tensor Parallelism)

If you have a second GPU (e.g., 2 × 16 GB GPUs):

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --tensor-parallel-size 2 \
  --max-model-len 8192
Option 4: Switch to a Smaller Model

If you are constrained to a single 16 GB GPU without CPU offloading:

LLaMA-3.1-8B (can run comfortably at 4-bit, 8-bit, or even 16-bit FP16).

Mistral-Nemo-12B or Qwen2.5-14B / 32B (quantized).

more_vert

more_vert
Model 1:26 AM
Thoughts

Expand to view model thoughts

chevron_right

more_vert

The previous Out-of-Memory (OOM) problem is resolved: the CPU offloading worked, successfully offloading ~16.94 GB of parameters to system RAM and leaving the model using only ~3.95 GiB of GPU VRAM.

However, the startup failed right after loading during the warm-up run (profile_run) due to:

code
Text
download
content_copy
expand_less
File ".../site-packages/humming/tune/__init__.py", line 47, in get_heuristics_class
    return heuristics_map[sm_version]
KeyError: 120
Why this happens

sm_version: 120: Your GPU is an NVIDIA RTX 50-series card (Blackwell consumer architecture), which has Compute Capability 12.0 (
12
×
10
+
0
=
120
12×10+0=120
).

The Root Cause: The humming quantization kernel library calculates your GPU architecture as 120, but its internal tuning table (heuristics_map) only defines entries for older architectures (such as SM 75, 80, 86, 89, 90, or 100) and does not have a key for 120[1][2][3].

Solution 1: Patch humming to recognize SM 120 (Recommended)

Aliasing 120 to fallback to SM 100 (Blackwell) or SM 89 (Ada) allows the JIT GEMM kernels to execute[1][3].

Open the file in an editor:

code
Bash
download
content_copy
expand_less
nano /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

Locate the function around line 45–47:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    ...
    return heuristics_map[sm_version]

Look at heuristics_map in that file to see what the highest supported SM class is (usually Sm100Heuristics or Sm89Heuristics)[3].

Add a fallback or map 120 right before the return statement. For example:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # ... existing logic calculating sm_version ...

    # --- Add this patch ---
    if sm_version == 120 or sm_version not in heuristics_map:
        # Fall back to 100 (Blackwell) or 89 (Ada)
        fallback_sm = 100 if 100 in heuristics_map else 89
        return heuristics_map[fallback_sm]
    # -----------------------

    return heuristics_map[sm_version]

Alternatively, find where heuristics_map = { ... } is defined in that file and add:

code
Python
download
content_copy
expand_less
# If Sm100Heuristics exists:
120: Sm100Heuristics,
# Or if only Sm89Heuristics exists:
120: Sm89Heuristics,

Save the file (Ctrl+O, Enter, Ctrl+X in nano) and re-run your vllm serve command.

Solution 2: Update the humming package

If you installed humming / humming-kernels from PyPI, a newer build may have already added native SM 120 support[4]:

code
Bash
download
content_copy
expand_less
source ~/Desktop/GSQ-RCO-GGUF/GSQ/.venv/bin/activate
pip install --upgrade humming humming-kernels

If it updates, try starting the server again without manual file edits.

Sources
help

vllm.ai
github.com
github.com
huggingface.co
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"humming" "KeyError" sm_version
"humming" quantization vllm
"heuristics_map" "humming" "DeviceHeuristics"

edit
more_vert

more_vert
Model 1:27 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

That hardware configuration explains the log output and gives you plenty of capacity to run this model:

GPU: RTX 5060 Ti (16 GB VRAM)

Built on NVIDIA's Blackwell architecture, which reports Compute Capability 12.0 (sm_version = 120).

This confirms why the humming quantization library threw KeyError: 120—the package was written prior to the RTX 50-series and lacks a tuning profile entry for SM 120.

System RAM: 96 GB DDR4

You have ample system memory (the 2-bit model checkpoint is ~21 GB, requiring ~17 GB offload).

Step 1: Fix the KeyError: 120 for your RTX 5060 Ti

Apply the patch to map your RTX 5060 Ti to the closest existing heuristics profile.

Open the file in your virtual environment:

code
Bash
download
content_copy
expand_less
nano ~/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

Look at the imports and the heuristics_map dictionary (around line 30–50). You will likely see something like:

code
Python
download
content_copy
expand_less
heuristics_map = {
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    89: Sm89Heuristics,
    # 100: Sm100Heuristics (if present)
}

Map 120 to the highest available entry (use Sm100Heuristics if defined, or Sm89Heuristics otherwise):

code
Python
download
content_copy
expand_less
# Add this line into the heuristics_map dict:
120: Sm89Heuristics,  # or Sm100Heuristics if available

Save and exit (Ctrl+O, Enter, Ctrl+X).

Step 2: Optimal Execution Command

Since you have 96 GB of RAM and 16 GB of VRAM, you can run:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 18 \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.90
Performance Tip: HP Z440 Workstation & PCIe 3.0

Because your machine is an HP Z440 Workstation, its motherboard runs PCIe Gen 3.0:

vLLM UVA Offloader streams parameters dynamically across the PCIe bus during every forward pass. On PCIe 3.0, this can become a major bottleneck and limit generation speed to ~1–3 tokens/sec.

Alternative (llama.cpp): If generation in vLLM feels too slow, llama.cpp handles hybrid GPU/CPU setups differently: it places the first 
𝑁
N
 layers permanently in GPU VRAM and runs the rest in system RAM, transferring only the intermediate layer activations over PCIe rather than full layer weights.

edit
more_vert
User 1:28 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ pip install --upgrade humming humming-kernels
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
python3-xyz, where xyz is the package you are trying to
install.

code
Code
download
content_copy
expand_less
If you wish to install a non-Debian-packaged Python package,
create a virtual environment using python3 -m venv path/to/venv.
Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
sure you have python3-full installed.

If you wish to install a non-Debian packaged Python application,
it may be easiest to use pipx install xyz, which will manage a
virtual environment for you. Make sure you have pipx installed.

See /usr/share/doc/python3.14/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ pip install humming humming-kernels
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
python3-xyz, where xyz is the package you are trying to
install.

code
Code
download
content_copy
expand_less
If you wish to install a non-Debian-packaged Python package,
create a virtual environment using python3 -m venv path/to/venv.
Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
sure you have python3-full installed.

If you wish to install a non-Debian packaged Python application,
it may be easiest to use pipx install xyz, which will manage a
virtual environment for you. Make sure you have pipx installed.

See /usr/share/doc/python3.14/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$

more_vert
Model 1:28 AM
Thoughts

Expand to view model thoughts

chevron_right

more_vert

This error happens because your shell's pip command is pointing to your system Python, where Debian/Ubuntu blocks direct package installations (PEP 668).

Your vLLM environment was created with uv and lives in:
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv (Python 3.11).

Method 1: Install using the Virtual Environment (Fixes pip)

The package name on PyPI is humming-kernels[1]. Target the .venv using one of the following commands:

Option A: Using uv (Fastest, since your venv was built with uv)
code
Bash
download
content_copy
expand_less
uv pip install --upgrade humming-kernels
Option B: Calling the .venv directly
code
Bash
download
content_copy
expand_less
./.venv/bin/python -m pip install --upgrade humming-kernels
Method 2: Patch KeyError: 120 Directly (Recommended)

Even the newest release of humming-kernels may still only contain tuning tables up to SM 100 or SM 89. You can fix the KeyError: 120 immediately by adding an alias for SM 120 directly in the installed source code[2]:

Check what architectures exist in the file:

code
Bash
download
content_copy
expand_less
grep -n "Sm" /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

Open the file in an editor:

code
Bash
download
content_copy
expand_less
nano /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

Around line 40–50, find the get_heuristics_class function:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # ...
    return heuristics_map[sm_version]

Change it to automatically fall back if sm_version == 120:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # ...
    if sm_version == 120 or sm_version not in heuristics_map:
        # Fall back to SM 100 (Blackwell) or SM 89 (Ada)
        fallback = 100 if 100 in heuristics_map else 89
        return heuristics_map[fallback]

    return heuristics_map[sm_version]

Save and exit (Ctrl+O 
→
→
 Enter 
→
→
 Ctrl+X).

Relaunch the server:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 18 \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.90

Sources
help

pypi.org
vllm.ai
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"humming" "humming-kernels" pip

more_vert
Model 1:31 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

Virtual environments created with uv omit standard pip by default to remain lightweight.

1. If you want to use the package manager: use uv

Use uv directly from your terminal:

code
Bash
download
content_copy
expand_less
uv pip install --upgrade humming-kernels

(Or if you ever need traditional pip inside the venv: uv pip install pip)

2. Patching the file directly (The actual fix for KeyError: 120)

Updating humming-kernels might still fail because RTX 50-series (SM 120) support is often not yet in the released tuning tables.

You can patch the installed file directly in one step.

Step A: Inspect the file around line 47

Run this command to check how heuristics_map is structured:

code
Bash
download
content_copy
expand_less
sed -n '35,55p' /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py
Step B: Edit the file

Open it with nano:

code
Bash
download
content_copy
expand_less
nano /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

Locate the get_heuristics_class function (around line 45) and update it to handle SM 120:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # (leave any code above this line untouched)

    # Add fallback for SM 120 / RTX 50-series
    if sm_version == 120 or sm_version not in heuristics_map:
        fallback = 100 if 100 in heuristics_map else 89
        return heuristics_map[fallback]

    return heuristics_map[sm_version]

Save and exit with Ctrl+O 
→
→
 Enter 
→
→
 Ctrl+X.

3. Re-run vLLM

After patching, start your server:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 18 \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.90

more_vert
Model 1:43 AM
Thoughts

Expand to view model thoughts

chevron_right

more_vert

حزمة humming-kernels متثبتة بالفعل عندك وموجودة داخل بيئة العمل، والمشكلة مش إنها ناقصة أو محتاجة تنزيل من الصفر[1][2].

الدليل إن الخطأ اللي ظهر عندك طالع من داخل الملف بتاعها نفسه:

code
Text
download
content_copy
expand_less
File ".../site-packages/humming/tune/__init__.py", line 47, in get_heuristics_class
    return heuristics_map[sm_version]
KeyError: 120
إيه سبب المشكلة بالضبط؟

كرتك RTX 5060 Ti مبني على معمارية Blackwell الحديثة، وكود المعمارية بتاعته في بايثون هو 120 (Compute Capability 12.0).
مكتبة humming لما بتيجي تقرأ الكرت عشان تشغل الكيرنلز (Kernels)، بتدور على إعدادات كرت رقمه 120 ومش بتلاقيه في الجدول بتاعها (الجدول آخره كروت جيل 89 و 100)، فبتقف وتطلع خطأ KeyError: 120.

الحل في خطوة واحدة (تعديل سطر في الملف المتثبت):

عشان تخليها تقبل الكرت وتشتغل فوراً، افتح الملف ده وعدل فيه سطر يخلي كرتك يتعامل كـ 100 أو 89:

افتح الملف بأمر nano:

code
Bash
download
content_copy
expand_less
nano /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

انزل عند السطر 45 تقريباً عند دالة get_heuristics_class، هتلاقيها بالشكل ده:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # ...
    return heuristics_map[sm_version]

غيرها وخليها كده:

code
Python
download
content_copy
expand_less
def get_heuristics_class(sm_version=None):
    # ...
    if sm_version == 120 or sm_version not in heuristics_map:
        fallback = 100 if 100 in heuristics_map else 89
        return heuristics_map[fallback]

    return heuristics_map[sm_version]

اضغط للحفظ:

Ctrl + O ثم اضغط Enter

بعدين Ctrl + X عشان تخرج.

لو حابب تثبت أحدث نسخة من humming-kernels باستخدام uv:

لو عايز تنزل أحدث إصدار من جيت هاب أو تحدثه بدون مشاكل pip و externally-managed-environment، استخدم أمر uv (لأن بيئتك مبنية بيه)[1]:

code
Bash
download
content_copy
expand_less
uv pip install --upgrade "humming-kernels[cu12]"

أو من السورس مباشرة[1]:

code
Bash
download
content_copy
expand_less
uv pip install git+https://github.com/vllm-project/humming.git
بعد التعديل، شغل السيرفر بالأمر ده:
code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 18 \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.90

Sources
help

github.com
vllm.ai
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"humming-kernels"

more_vert
Model 1:44 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

سبب هذا الخطأ هو عدم توافق بين إصدار PyTorch وإصدار vLLM المثبتين في البيئة.

عند تشغيل أو تحديث الحزم، تم تثبيت/تحديث نسخة جديدة من torch (تظهر في التحذيرات الأولى deprecated)، في حين أن مكتبة C++ الخاصة بـ vLLM (vllm._C.abi3.so) مبنية ومترجمة لنسخة محددة أخرى من PyTorch، مما تسبب في خطأ:

code
Text
download
content_copy
expand_less
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
الحل: إعادة تثبيت إصدار PyTorch المتوافق مع vLLM

يجب تثبيت نسخة PyTorch المتوافقة مع إصدار vLLM وCUDA لديك باستخدام uv:

1. معرفة الإصدار المناسب

إصدار vLLM لديك هو 0.20.2 (أو نسخة مخصصة/Fork لـ GSQ/Humming). في الغالب يحتاج PyTorch 2.5 أو 2.6 مع CUDA 12.4 أو 12.6.

قم بتشغيل الأمر التالي لإعادة تثبيت PyTorch متوافق مع CUDA 12:

code
Bash
download
content_copy
expand_less
uv pip install --upgrade --reinstall "torch==2.5.1" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu124

ملاحظة: إذا كان مشروع GSQ الذي تستخدمه يعتمد على PyTorch 2.4، يمكنك استبدال 2.5.1 بـ 2.4.0:

code
Bash
download
content_copy
expand_less
uv pip install --upgrade --reinstall "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu124
2. التأكد من حل مشكلة الاستيراد

بعد انتهاء التثبيت، اختبر عمل C++ extension عبر هذا الأمر:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('PyTorch and vLLM C++ extensions loaded successfully!')"

إذا طبعت العبارة بدون أي أخطاء ImportError، تكون المشكلة قد حُلّت تماماً.

3. إعادة تشغيل السيرفر

تأكد من تعديل ملف humming/tune/__init__.py لتخطي مشكلة KeyError: 120 كما تم توضيحه سابقاً، ثم شغّل:

code
Bash
download
content_copy
expand_less
vllm serve ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

(تم ضبط --cpu-offload-gb 20 لأن حجم النموذج 2-bit بالكامل حوالي 21 GB، مع حجز الباقي في الـ VRAM).

edit
more_vert

يمكنك تنفيذ هذا الأمر مباشرة في الطرفية (Terminal)، حيث يقوم بإنشاء نسخة احتياطية للملف (.bak) وتعديل الكود تلقائياً ليعمل كرت RTX 5060 Ti بأعلى معمارية مدعومة داخل المكتبة:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c '
file_path = "/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py"
with open(file_path, "r") as f:
    code = f.read()

target = "return heuristics_map[sm_version]"
patch = """if sm_version not in heuristics_map:
        return heuristics_map[max(heuristics_map.keys())]
    return heuristics_map[sm_version]"""

if target in code:
    with open(file_path + ".bak", "w") as f_bak:
        f_bak.write(code)
    with open(file_path, "w") as f_out:
        f_out.write(code.replace(target, patch, 1))
    print("تم تعديل الملف بنجاح! جاهز للتشغيل.")
else:
    print("الملف معدل بالفعل مسبقاً أو السطر غير موجود.")
'
للتأكد من نجاح التعديل:

شغّل هذا الأمر لعرض الأسطر المعدلة:

code
Bash
download
content_copy
expand_less
grep -C 3 "max(heuristics_map.keys())" /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

إذا ظهرت لك أسطر الكود المعدلة، يمكنك بعدها تشغيل vllm serve مباشرة.

more_vert
Model 1:45 AM
Thoughts

Expand to view model thoughts

chevron_right

more_vert

هناك سببان لما حدث في الأمر الأخير:

1. سبب الـ SyntaxError (خطأ في كتابة علامات التنصيص)

في الأمر الذي قمت بتشغيله:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('...loaded successfully' 2>&1 | tee ~/gsq_smoke.log)"

تم وضع كود الطرفية الخاص بالتسجيل 2>&1 | tee ... داخل علامات تنصيص بايثون المزدوجة "..."، فاعتبرها بايثون كوداً برمجياً خاطئاً.

الصياغة الصحيحة للأمر (إغلاق علامة التنصيص قبل الـ pipe |):

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('Success!')" 2>&1 | tee ~/gsq_smoke.log
2. حل خطأ ImportError: undefined symbol

الخطأ:

code
Text
download
content_copy
expand_less
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...

يعني أن نسخة torch الحالية غير متوافقة ثنائياً (ABI mismatch) مع حزمة vllm المترجمة في مشروع GSQ[1][2].

الخطوة 1: فحص الإصدار الحالي

شغّل هذا الأمر لمعرفة إصدار PyTorch وCUDA الموجودين حالياً:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; print('Torch Version:', torch.__version__, '| CUDA:', torch.version.cuda)"
الخطوة 2: استعادة إصدار PyTorch المتوافق

مشروع GSQ ونسخة vLLM المثبتة لديك مترجمة أساساً للعمل مع torch==2.5.1 (أو 2.4.0) مع CUDA 12.4.

قم بتنفيذ أمر التثبيت المتوافق باستخدام uv:

code
Bash
download
content_copy
expand_less
uv pip install --upgrade --reinstall "torch==2.5.1" torchvision --index-url https://download.pytorch.org/whl/cu124

(إذا استمر نفس الخطأ مع 2.5.1، استبدلها بـ torch==2.4.0):

code
Bash
download
content_copy
expand_less
uv pip install --upgrade --reinstall "torch==2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu124
الخطوة 3: التحقق

بعد اكتمال التثبيت، شغّل الأمر الصحيح:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('PyTorch and vLLM C++ extensions loaded successfully!')"

بمجرد أن يطبع successfully، سيكون بإمكانك تشغيل السيرفر مباشرة.

Sources
help

github.com
github.com
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation"
vllm "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation"

edit
more_vert
User 1:45 AM

تم تعديل الملف بنجاح! جاهز للتشغيل.
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ grep -C 3 "max(heuristics_map.keys())" /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/init.py

code
Code
download
content_copy
expand_less
if sm_version in heuristics_map:
    if sm_version not in heuristics_map:
    return heuristics_map[max(heuristics_map.keys())]
return heuristics_map[sm_version]

sm_version_base = sm_version // 10 * 10

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$

edit
more_vert

انتبه للناتج الذي ظهر لك من أمر grep:

code
Python
download
content_copy
expand_less
if sm_version in heuristics_map:
        if sm_version not in heuristics_map:
        return heuristics_map[max(heuristics_map.keys())]
    return heuristics_map[sm_version]

الاستبدال تم في المكان الخاطئ وأحدث خطأين:

خطأ مسافات (IndentationError) في بايثون، لأن سطر return ليس بمحاذاة الـ if.

تناقض منطقي: الشرط الأول يقول إذا كان الكرت موجوداً وبداخله مباشرة إذا كان الكرت غير موجود، بالتالي لن يُنفذ أبداً.

الحل الصحيح والمضمون (في 3 أوامر سريعة)

أفضل طريقة وأبسطها هي إضافة رقم 120 مباشرة إلى القاموس heuristics_map دون التعديل داخل الدوال، حتى نتجنب أي أخطاء في المسافات.

1. استعادة الملف الأصلي من النسخة الاحتياطية:
code
Bash
download
content_copy
expand_less
cp /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py.bak /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py
2. إضافة كرت 120 بنهاية الملف مباشرة:

نفّذ هذا الأمر لإضافة السطر في آخر الملف:

code
Bash
download
content_copy
expand_less
echo -e "\n# RTX 50-series support\nheuristics_map[120] = heuristics_map[max(heuristics_map.keys())]" >> /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py
3. اختبار نجاح التعديل فوراً:

شغّل هذا الأمر لاختبار الدالة مع كود معمارية كرتك (120):

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "from humming.tune import get_heuristics_class; print('Success! Loaded heuristics:', get_heuristics_class(120))"

إذا طبع لك:

code
Text
download
content_copy
expand_less
Success! Loaded heuristics: <class '...'>

هذا يؤكد حل مشكلة KeyError: 120 نهائياً.

edit
more_vert
Model 1:46 AM

نفّذ هذا الأمر الآن في الطرفية واكتب لي الناتج الذي سيظهر لك (Output):

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; print('Torch Version:', torch.__version__, '| CUDA:', torch.version.cuda)"

بمجرد أن تعطيني النتيجة، سنعرف بالضبط ما هو إصدار PyTorch المثبت حالياً، لنحل تعارض ImportError: undefined symbol ونعيد تشغيل السيرفر مباشرة.

more_vert
Model 1:46 AM
Thoughts

Expand to view model thoughts

chevron_right

more_vert

النتيجة ممتازة ووضّحت أصل المشكلة بالكامل:

code
Text
download
content_copy
expand_less
Torch Version: 2.14.1+cu130 | CUDA: 13.0
سبب المشكلة:

نسخة PyTorch المثبتة حالياً عندك هي 2.14.1+cu130 (CUDA 13.0).
في إصدارات PyTorch الحديثة (مع CUDA 13)، قامت PyTorch بتغيير توقيع الدالة البرمجية (C++ ABI) لـ:
c10::cuda::c10_cuda_check_implementation (تغير نوع أحد المعاملات من int إلى unsigned int)[1].
بينما ملفات الـ C++ المجمعة لمشروع vLLM (vllm._C.abi3.so) كانت مبنية ومترجمة لإصدار PyTorch أقدم[2][3]، ولذلك عند استدعائها تبحث عن التوقيع القديم ولا تجده، فيظهر لك:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...[1][2]

لمعرفة النسخة الأصلية الصحيحة التي يحتاجها vLLM لديك:

نفّذ هذا الأمر لعرض الحزم وحزمة vLLM لمعرفة كيفية تثبيتها:

code
Bash
download
content_copy
expand_less
uv pip list | grep -E "torch|vllm|humming"

انسخ الناتج هنا لنرى ما هي النسخ المرتبطة ببعضها بدقة ونصلح التعارض في خطوة واحدة.

Sources
help

github.com
github.com
github.com
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"_ZN3c104cuda29c10_cuda_check_implementation"
"c10_cuda_check_implementation" vllm undefined symbol

more_vert
Model 1:47 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

الخطأ هنا بسيط جداً ومُبشّر:

الدالة get_heuristics_class() المعامل (Argument) الخاص بها هو رقم كرت الشاشة (CUDA Device Index) مثل 0 أو 1، وليس رقم المعمارية!
عندما مررنا 120، ظنت بايثون أنك تسأل عن الكرت رقم 120 في الجهاز (invalid CUDA device index 120) وأنت تملك فقط الكرت رقم 0.

الاختبار الصحيح:

استدعِ الدالة بدون أي أرقام بين القوسين ()؛ ستقوم تلقائياً بفحص كرتك (GPU 0: RTX 5060 Ti)، وستقرأ أن معماريته هي 120، وستأخذ القيمة من السطر الذي أضفناه بنجاح:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "from humming.tune import get_heuristics_class; print('Success! Loaded heuristics:', get_heuristics_class())"

شغّل هذا الأمر الآن، وسيقرأ الكرت رقم 0 ويعيد لك Success! بنجاح.

edit
more_vert
Model 1:47 AM

النتيجة واضحة: إصدار PyTorch المثبت لديك حالياً هو 2.14.1+cu130 مع CUDA 13.0.

هذا الإصدار هو المسؤول المباشر عن الخطأ:

code
Text
download
content_copy
expand_less
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...

لأن حزمة vllm المجمعة مسبقاً في مشروع GSQ بُنيت بالأساس على توافقية C++ ABI تسبق CUDA 13.0 (تحديداً إصدارات PyTorch 2.5 أو 2.6 مع CUDA 12.4 / 12.6).

الخطوة 1: فحص حزم البيئة

لنرى النسخ المثبتة حالياً داخل البيئة:

code
Bash
download
content_copy
expand_less
uv pip list | grep -E "torch|vllm|humming"
الخطوة 2: اختبار استدعاء C++ الحالي

تأكد مما إذا كان vllm._C يعمل أم يتعطل مع الإصدار الحالي:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('vLLM C++ module is OK!')"

انسخ ناتج هذين الأمرين، وإذا ظهر خطأ undefined symbol، سنقوم بإرجاع PyTorch إلى الإصدار المتوافق مع vLLM فوراً عبر uv.

more_vert
Model 1:48 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

قائمة الحزم وضحت تفاصيل البيئة:

torch: 2.14.1 (مع CUDA 13.0)

vllm: 0.20.2

humming-kernels: 0.1.18

مشكلة ImportError تحدث لأن ملف vllm._C.abi3.so يبحث عن دالة C++ بتوقيع محدد داخل مكتبة libtorch_cuda.so التابعة لـ PyTorch، لكن الإصدار 2.14.1+cu130 يحتوي على توقيع مختلف قليلاً.

خطوة سريعة: فحص التوقيع والتحقق من الاستيراد

نفّذ هذا الأمر أولاً للتحقق مما إذا كان vllm._C لا يزال يواجه مشكلة الاستيراد، والبحث عن اسم الدالة داخل مكتبة PyTorch الحالية:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "
import torch
print('1. Torch path:', torch.__file__)
try:
    import vllm._C
    print('2. vLLM C++ extension: SUCCESS')
except Exception as e:
    print('2. vLLM C++ extension FAILED:', e)
"

انسخ لي الناتج الذي سيظهر لك، وبناءً عليه سنحدد ما إذا كان الحل يتطلب:

تعديل خفيف في مسار التحميل التلقائي للمكتبة (Preload لـ libc10_cuda.so).

أو تثبيت بناء vLLM المتوافق تماماً مع إصدار PyTorch 2.14 وCUDA 13 لديك.

edit
more_vert
Model 1:48 AM

هذه نتيجة ممتازة وخطوة أساسية تم حلها بنجاح:

code
Text
download
content_copy
expand_less
Success! Loaded heuristics: <class 'humming.tune.sm121.Sm121Heuristics'>

مكتبة humming قرأت كرتك RTX 5060 Ti وتعرفت عليه، وربطته بفئة معمارية Blackwell الحديثة (Sm121Heuristics) بدون أي أخطاء.

خطأ KeyError: 120 انتهى تماماً ولم يعد موجوداً.

الخطوة التالية والأخيرة:

الآن لنختبر استيراد محرك vllm المكتوب بـ C++ للتأكد من جاهزيته قبل تشغيل السيرفر:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "
import torch
try:
    import vllm._C
    print('vLLM C++ extension: SUCCESS')
except Exception as e:
    print('vLLM C++ extension FAILED:', e)
"

إذا طبع لك SUCCESS: سنقوم بتشغيل سيرفر vllm serve مباشرة الآن.

إذا طبع FAILED: سنعالج استدعاء _C.abi3.so في خطوة واحدة.

more_vert
Model 1:49 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

المشكلة نفسها تتكرر بسبب مكان علامة التنصيص المزدوجة ":

أوامر اللينكس 2>&1 | tee ~/gsq_smoke.log مكتوبة داخل كود بايثون، وبايثون لا يفهم أوامر الطرفية داخل نصوصه فيعطيك SyntaxError.

انسخ هذا الأمر كما هو بالضبط (بدون أي تعديل):
code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('vLLM C++ module is OK!')"

(ملاحظة: إذا كنت تريد تسجيل النتيجة في ملف gsq_smoke.log، يجب أن تُغلق علامة التنصيص " قبل علامة الـ pipe | هكذا:)

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; print('vLLM C++ module is OK!')" 2>&1 | tee ~/gsq_smoke.log

شغّل الأمر الأول البسيط مباشرة، واكتب لي ما الذي سيظهر في الشاشة.

more_vert

السبب الحقيقي والنهائي لتعطل vllm._C اتضح بنسبة 100%[1]:

code
Text
download
content_copy
expand_less
vLLM C++ extension FAILED: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb

مشروع GSQ مبني ومثبت عبر ملف قفل محدد مسبقاً (uv.lock)[2].

عندما تم تنفيذ أوامر ترقية وتحديث سابقة، قامت أداة التثبيت بجلب نسخة PyTorch 2.14.1 مع CUDA 13.0 الحديثة، والتي غيّرت توقيع دوال C++ الداخلية الخاصة بـ CUDA (c10_cuda_check_implementation)[1]، بينما ملف vllm._C المجمّع مسبقاً يحتاج الإصدار الأصلي المحدد في مشروع GSQ[1].

الحل الجذري: مزامنة البيئة مع ملف المشروع الأصلي (uv sync)

ميزة استخدام uv في مشروع GSQ هي وجود ملف uv.lock الذي يحتوي على النسخ المتوافقة حرفياً بدون أي تضارب[2]:

1. استعادة الحزم المتوافقة عبر uv sync:

شغّل هذا الأمر وأنت داخل مجلد المشروع:

code
Bash
download
content_copy
expand_less
cd ~/Desktop/GSQ-RCO-GGUF/GSQ
uv sync

(سيقوم uv خلال ثوانٍ معدودة بإعادة ضبط إصدارات PyTorch وvLLM إلى النسخ المتطابقة المتوافقة مع بعضها تماماً)[2].

2. إعادة تفعيل كرت RTX 5060 Ti:

(لأن uv sync قد يعيد تثبيت حزمة humming-kernels على وضعها الأصلي)، نفّذ هذا الأمر لضمان استمرار تعريف الكرت[2]:

code
Bash
download
content_copy
expand_less
echo -e "\n# RTX 50-series support\nheuristics_map[120] = heuristics_map[max(heuristics_map.keys())]" >> /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py
3. الفحص النهائي:

تأكد من أن كلاً من torch و vllm._C و humming يعملون بدون أدنى خطأ:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "import torch; import vllm._C; from humming.tune import get_heuristics_class; print('ALL SYSTEMS READY! GPU class:', get_heuristics_class())"

بمجرد ظهور ALL SYSTEMS READY!، يمكنك فوراً تشغيل السيرفر.

Sources
help

github.com
github.com
Google Search Suggestions
Display of Search Suggestions is required when using Grounding with Google Search. Learn more
"c10_cuda_check_implementation" undefined symbol vllm
"ISTA-DASLab/GSQ" setup_env.sh

more_vert
Model 1:50 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

الآن اتضح سر تكرار هذا الخطأ الغريب!

سبب المشكلة:

في موجه أوامر لينكس (Bash)، علامة التعجب ! داخل علامات التنصيص المزدوجة "..." ليست حرفاً عادياً، بل هي اختصار يسمى History Expansion.
الطرفية بمجرد أن رأت ! استرجعت الأمر القديم الخاطئ من ذاكرة الأوامر السابقة واستبدلته تلقائياً بالخطأ قبل أن يُرسل إلى بايثون!

الحل: إلغاء هذه الخاصية واستخدام أمر بدون علامة تعجب

نفّذ هذا الأمر أولاً لتعطيل استرجاع الأوامر بعلامة التعجب نهائياً في جلستك الحالية:

code
Bash
download
content_copy
expand_less
set +H

ثم شغّل هذا الأمر (تم حذف علامة التعجب واستخدام علامات تنصيص مفردة '...' لمنع اللينكس من التعديل عليه):

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c 'import torch; import vllm._C; print("vLLM C++ module is OK")'

شغّل هذين الأمرين، ولن يظهر أي SyntaxError بعد الآن.

edit
more_vert
Model 1:51 AM

أداء رائع جداً! مخرجات uv sync توضح تماماً ما حدث:

تمت إعادة torch إلى الإصدار الأصلي المتوافق 2.11.0+cu130 الذي بُنيت عليه حزمة vllm.

تمت إعادة ضبط مكتبات triton و cuda-bindings و humming-kernels للإصدارات المحددة بدقة في uv.lock.

تم تطبيق تعديل دعم كرت RTX 5060 Ti بنجاح عبر سطر heuristics_map[120].

الفحص النهائي السريع:

تأكد من أن جميع المكونات تعمل وتتصل بالكرت بدون أي اعتراض:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c 'import torch; import vllm._C; from humming.tune import get_heuristics_class; print("READY: GPU class:", get_heuristics_class())'
تشغيل السيرفر مباشرة:

إذا طبع الأمر السابق READY، شغّل السيرفر فوراً بالأمر التالي المخصص لمواصفات جهازك (RTX 5060 Ti 16GB + 96GB RAM):

code
Bash
download
content_copy
expand_less
vllm serve ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

(أو إذا كنت تشغله من المجلد المحلي المحمل على سطح المكتب):

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

edit
more_vert
Model 1:52 AM

البيئة الآن جاهزة تماماً 100%!

code
Text
download
content_copy
expand_less
READY: GPU class: <class 'humming.tune.sm100.Sm100Heuristics'>

تم حل تضارب C++ ABI بين PyTorch و vLLM.

تم تفعيل دعم كرتك RTX 5060 Ti وربطه بمعمارية Sm100Heuristics.

أمر التشغيل النهائي:

شغّل السيرفر الآن باستخدام النموذج المحمّل محلياً على جهازك:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

انتظر حتى ينتهي من تحميل طبقات النموذج وعمل الاختبار التمهيدي؛ سيفتح السيرفر ويعطيك عنوان الاستدعاء (غالباً http://0.0.0.0:8000).

more_vert
Model 1:54 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

السبب هنا بسيط جداً:

أنت كتبت في الأمر اسم المستودع أونلاين: ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ، لذلك يحاول vLLM الاتصال بموقع HuggingFace وتحميل النموذج من الإنترنت، ولكن الاتصال بموقع HuggingFace واجه انقطاعاً أو بطئاً (handshake timed out).

بينما النموذج محمّل عندك بالفعل محلياً على جهازك في المسار الموجود على سطح المكتب!

الحل:

اضغط Ctrl + C لإيقاف العملية الحالية.

شغّل الأمر مستخدماً المسار المحلي للنموذج المحمّل عندك مباشرة (فلن يحتاج للإنترنت نهائياً):

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

more_vert
Model 1:56 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

التقدم ممتاز جداً! النموذج اجتاز كل العقبات الصعبة بنجاح:

PyTorch و vLLM C++ extensions تعملان بدون أي تضارب.

تم تقسيم النموذج بنجاح: 16.94 GiB تم نقلها للرامات (CPU)، وحجز 3.95 GiB فقط في كرت الشاشة.

تم تحميل جميع ملفات النموذج الـ 8 في 3.4 ثوانٍ فقط.

الخطوة الوحيدة المتبقية حدثت أثناء حساب سرعة الأنوية (Tensor Cores) في ملف device.py السطر 49:

code
Text
download
content_copy
expand_less
File ".../humming/utils/device.py", line 49, in estimate_tensorcore_max_tops
    ops_per_clock = ops_map[sm_version]
KeyError: 120

حيث يوجد جدول أداء اسمه ops_map لم يكن مضافاً إليه أيضاً رقم المعمارية 120.

الحل: تعديل ops_map ليقبل كرت 120 تلقائياً

نفّذ هذا الأمر في الطرفية، حيث سيقوم بتعديل السطر بأمان تام ليرجع أعلى قيمة متاحة في الجدول إذا لم يجد 120، ولن يحدث KeyError بعدها أبداً:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c '
file_path = "/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/utils/device.py"
with open(file_path, "r") as f:
    code = f.read()

target = "ops_per_clock = ops_map[sm_version]"
patch = "ops_per_clock = ops_map.get(sm_version, ops_map[max(ops_map.keys())])"

if target in code:
    with open(file_path + ".bak", "w") as f_bak:
        f_bak.write(code)
    with open(file_path, "w") as f_out:
        f_out.write(code.replace(target, patch))
    print("تم تعديل ops_map بنجاح!")
else:
    print("السطر غير موجود أو تم تعديله مسبقاً.")
'
اختبار سريع للتحقق من عمل الدالة:

تأكد من أن حسابات الكرت تعمل الآن بدون KeyError: 120:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c "from humming.utils.device import estimate_tensorcore_max_tops; print('TensorCore Tops:', estimate_tensorcore_max_tops())"
تشغيل السيرفر:

بمجرد أن يطبع رقم الأداء بدون أخطاء، أعد تشغيل السيرفر:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

edit
more_vert
Model 1:57 AM

مبروك! النتيجة مثالية وتم حل المشكلة بالكامل:

code
Text
download
content_copy
expand_less
TensorCore Tops: 828.1128960000001

قامت مكتبة humming بقراءة كرتك RTX 5060 Ti وحساب أداء الـ Tensor Cores بنجاح تام، وبذلك انتهت كل أخطاء KeyError: 120.

شغّل السيرفر الآن:
code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager

انتظر لحظات حتى يكمل مرحلة الـ Warm-up/Profiling، وسيفتح السيرفر ويعطيك عنوان الاتصال (http://0.0.0.0:8000) ليكون النموذج جاهزاً للاستخدام واستقبال الطلبات فوراً.

edit
more_vert
Model 2:03 AM

بمجرد أن يبدأ السيرفر ويعطيك رسالة مثل:
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

افتح نافذة طرفية جديدة (Terminal جديد) وجرب إرسال الطلب بإحدى الطرق التالية:

1. باستخدام أمر curl (الأسرع من الطرفية)
code
Bash
download
content_copy
expand_less
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    "messages": [
      {"role": "user", "content": "مرحباً، عرفني بنفسك في سطرين."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
2. باستخدام بايثون (كود بسيط)

إذا كنت تفضل إرسال الطلب عبر سكربت بايثون:

code
Bash
download
content_copy
expand_less
./.venv/bin/python -c '
import urllib.request
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    "messages": [{"role": "user", "content": "مرحباً، ما هي عاصمة فرنسا؟"}],
    "max_tokens": 50
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))
    print("\nرد النموذج:\n" + result["choices"][0]["message"]["content"])
'
3. باستخدام مكتبة OpenAI الرسمية:

إذا كان لديك تطبيق يستخدم مكتبة openai:

code
Python
download
content_copy
expand_less
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-needed",  # vLLM لا يطلب مفتاح افتراضياً
)

response = client.chat.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    messages=[{"role": "user", "content": "اكتب لي كود بايثون لطباعة الأرقام الفردية من 1 إلى 10"}],
    max_tokens=150,
)

print(response.choices[0].message.content)

جرب إرسال أمر curl الأول وأخبرني بالنتيجة التي يعيدها لك!

edit
more_vert
Model 2:05 AM

الخبر الرائع: السيرفر شغال بنجاح 100% والنموذج استقبل الطلب بالفعل!

سبب هذا الخطأ فقط:

code
Text
download
content_copy
expand_less
As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one.

مجلد النموذج المحلي تنقصه إعدادات قالب المحادثة الافتراضي (chat_template) داخل ملف tokenizer_config.json.

لديك حلّان بسيطان جداً:

الحل 1: استخدام endpoint إكمال النصوص العادي (/v1/completions) فوراً

لا يحتاج هذا المسار لقالب محادثة (Chat Template) ويعمل في هذه اللحظة بدون إعادة تشغيل السيرفر:

code
Bash
download
content_copy
expand_less
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    "prompt": "مرحباً، عرفني بنفسك في سطرين:\n",
    "max_tokens": 100,
    "temperature": 0.7
  }'
الحل 2: تفعيل قالب Llama-3 للمحادثات (/v1/chat/completions)

إذا أردت استخدام واجهة الشات الطبيعية (messages مع role: user / assistant)، فقط أضف خيار --chat-template عند تشغيل السيرفر.

أوقف السيرفر الحالي بـ Ctrl + C.

أعد تشغيله بإضافة قالب Llama 3 المدمج في vLLM:

code
Bash
download
content_copy
expand_less
vllm serve /home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ \
  --cpu-offload-gb 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --enforce-eager \
  --chat-template "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

بعدها سيعمل أمر curl الخاص بالـ chat/completions الذي أرسلته سابقاً بدون أي مشاكل!

Citations

https://github.com/A-transformer/MambaInLlama

edit
more_vert
User 2:07 AM

m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF
𝑝
𝑦
𝑡
ℎ
𝑜
𝑛
3
𝑎
.
𝑝
𝑦
𝑇
𝑟
𝑎
𝑐
𝑒
𝑏
𝑎
𝑐
𝑘
(
𝑚
𝑜
𝑠
𝑡
𝑟
𝑒
𝑐
𝑒
𝑛
𝑡
𝑐
𝑎
𝑙
𝑙
𝑙
𝑎
𝑠
𝑡
)
:
𝐹
𝑖
𝑙
𝑒
"
/
ℎ
𝑜
𝑚
𝑒
/
𝑚
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
/
𝑎
.
𝑝
𝑦
"
,
𝑙
𝑖
𝑛
𝑒
1
,
𝑖
𝑛
<
𝑚
𝑜
𝑑
𝑢
𝑙
𝑒
>
𝑓
𝑟
𝑜
𝑚
𝑜
𝑝
𝑒
𝑛
𝑎
𝑖
𝑖
𝑚
𝑝
𝑜
𝑟
𝑡
𝑂
𝑝
𝑒
𝑛
𝐴
𝐼
𝑀
𝑜
𝑑
𝑢
𝑙
𝑒
𝑁
𝑜
𝑡
𝐹
𝑜
𝑢
𝑛
𝑑
𝐸
𝑟
𝑟
𝑜
𝑟
:
𝑁
𝑜
𝑚
𝑜
𝑑
𝑢
𝑙
𝑒
𝑛
𝑎
𝑚
𝑒
𝑑
′
𝑜
𝑝
𝑒
𝑛
𝑎
𝑖
′
𝑚
@
𝑚
−
𝐻
𝑃
−
𝑍
440
−
𝑊
𝑜
𝑟
𝑘
𝑠
𝑡
𝑎
𝑡
𝑖
𝑜
𝑛
:
 
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
python3a.pyTraceback(mostrecentcalllast):File"/home/m/Desktop/GSQ−RCO−GGUF/a.py",line1,in<module>fromopenaiimportOpenAIModuleNotFoundError:Nomodulenamed
′
openai
′
m@m−HP−Z440−Workstation: /Desktop/GSQ−RCO−GGUF
 pip install openai
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
python3-xyz, where xyz is the package you are trying to
install.

code
Code
download
content_copy
expand_less
If you wish to install a non-Debian-packaged Python package,
create a virtual environment using python3 -m venv path/to/venv.
Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
sure you have python3-full installed.

If you wish to install a non-Debian packaged Python application,
it may be easiest to use pipx install xyz, which will manage a
virtual environment for you. Make sure you have pipx installed.

See /usr/share/doc/python3.14/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$

edit
more_vert
Model 2:07 AM

السبب هو أنك قمت بتشغيل السكربت وتثبيت الحزمة باستخدام بايثون الخاص بالنظام (python3 و pip) خارج البيئة الافتراضية، بينما بيئة العمل التي تحتوي على حزم المشروع هي .venv الموجودة داخل مجلد GSQ.

الحل في خطوتين:
1. تثبيت مكتبة openai داخل البيئة الافتراضية (باستخدام uv):

من الطرفية، شغّل هذا الأمر لتثبيتها في البيئة المناسبة فوراً دون أخطاء:

code
Bash
download
content_copy
expand_less
uv pip --python /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/bin/python install openai
2. تشغيل الملف a.py باستخدام بايثون الخاص بالبيئة الافتراضية:
code
Bash
download
content_copy
expand_less
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/bin/python a.py
ملاحظة هامة بخصوص الكود داخل a.py:

إذا كان سيرفر vLLM لا يزال يعمل في الخلفية بالتشغيل السابق (بدون --chat-template)، تأكد أن تستخدم client.completions.create بدلاً من chat.completions لتجنب خطأ chat template:

code
Python
download
content_copy
expand_less
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt="مرحباً، عرفني بنفسك في سطرين:\n",
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].text)

edit
more_vert
Model 2:08 AM

بما أنك قمت بتفعيل البيئة بالفعل وظهرت لك علامة (gsq)، فلست بحاجة لتمرير مسار بايثون؛ يمكنك التثبيت مباشرة بأمر بسيط جداً:

code
Bash
download
content_copy
expand_less
uv pip install openai

بعد اكتمال التثبيت، شغّل ملفك مباشرة:

code
Bash
download
content_copy
expand_less
python ~/Desktop/GSQ-RCO-GGUF/a.py

edit
more_vert
Model 2:13 AM

النموذج يعمل ويولّد نصوصاً بالفعل الآن! (Running: 1 reqs).

لكن تلاحظ أن سرعة التوليد بطيئة جداً:

code
Text
download
content_copy
expand_less
Avg generation throughput: 0.3 to 0.4 tokens/s

أي توليد توكن واحد كل 3 ثوانٍ تقريباً.

لماذا السرعة 0.3 - 0.4 tokens/s؟

طبيعة معمارية vLLM مع الـ CPU Offload:
محرك vLLM مصمم أصلاً ليعمل بكامل طبقاته داخل ذاكرة كرت الشاشة (VRAM). عند تفعيل --cpu-offload-gb 20، يقوم المحرك بنقل أوزان الطبقات جيئة وذهاباً بين الرامات وكرت الشاشة أثناء كل توكن يتم توليده عبر منفذ الـ PCIe (UVAOffloader).

منفذ PCIe 3.0 في جهاز HP Z440:
الجهاز يعتمد على الجيل الثالث (PCIe 3.0 x16)، وسرعة نقل البيانات عبره حوالي 12-14 GB/s كحد أقصى، مما يجعل نقل أكثر من 16 جيجابايت لكل توكن بمثابة عنق زجاجة خانق للأداء.

كيف ترفع السرعة بشكل كبير؟

إذا أردت سرعة توليد عملية (مثلاً 10 إلى 20 tokens/sec على نفس جهازك وبدون شراء أي قطع جديدة):

الخيار الأفضل: استخدام llama.cpp (المخصص للـ CPU/GPU Hybrid)

بما أن مجلدك اسمه GSQ-RCO-GGUF، فإذا كان لديك ملف النموذج بصيغة .gguf، فإن llama.cpp يتعامل مع الرامات بشكل مختلف كلياً وذكي:

يضع جزءاً ثابتاً من الطبقات داخل VRAM الكرت (مثلاً 40 طبقة على الـ RTX 5060 Ti).

ويترك باقي الطبقات ثابتة في الرامات (96 GB RAM).

لا ينقل أوزان الطبقات عبر الـ PCIe أبداً، بل ينقل فقط التنشيطات (Activations) بحجم بضعة كيلوبايت، مما يلغي عنق الزجاجة للـ PCIe 3.0 بالكامل!

أمر تشغيله عبر llama-server:

code
Bash
download
content_copy
expand_less
./llama-server -m /path/to/Llama-3.1-70B-Q2_K.gguf -ngl 40 -c 2048 --port 8000
إذا أردت البقاء على vLLM بأعلى سرعة ممكنة:

استخدم نموذج بحجم 8B أو 14B أو 32B:
كرتك RTX 5060 Ti (16 GB) قادر على تشغيل Llama-3.1-8B بالكامل داخل الـ VRAM بدون أي أوفلود، وستحصل على سرعة هائلة تصل إلى 50 - 80 tokens/sec!

كما يمكنه تشغيل Qwen2.5-14B (4-bit) بالكامل داخل الـ VRAM بسرعة 30 - 45 tokens/sec.

more_vert
Model 2:16 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

سبب عدم عمل هذا المجلد مع llama.cpp هو أن:

النموذج الموجود داخل المجلد الحالي (Llama-3.1-70B-Instruct-2Bit-GSQ) ليس بصيغة GGUF؛ بل هو ملفات safetensors مكممة بتقنية بحثية خاصة اسمها GSQ / Humming طورها فريق (ISTA-DASLab)، وهذه التقنية لا تملك كيرنلز تنفيذية إلا داخل مكتبة vLLM فقط.

بينما llama.cpp يحتاج إلى ملف ينتهي بامتداد .gguf.

كيف تشغل Llama-3.1-70B على llama.cpp بسرعة حقيقية؟

إذا أردت الحصول على سرعة ممتازة للـ 70B على رامات جهازك (96GB) وكرتك (16GB)، تحتاج إلى تحميل ملف GGUF قياسي 2-bit (مثل IQ2_M أو Q2_K)، فهذه الصيغ مدعومة أصلياً داخل llama.cpp.

1. تحميل ملف GGUF بحجم 2-bit (~25 GB):

يمكنك تحميل نسخة 2-bit الرسمية بصيغة GGUF (مثلاً من مستودع bartowski الشهير):

code
Bash
download
content_copy
expand_less
# تثبيت huggingface-cli إذا لم يكن موجوداً
uv pip install huggingface-hub

# تحميل ملف Q2_K بصيغة GGUF مباشرة داخل مجلدك
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  Meta-Llama-3.1-70B-Instruct-Q2_K.gguf \
  --local-dir /home/m/Desktop/GSQ-RCO-GGUF/
2. تشغيله بـ llama-cli مع تسريع الكرت والرامات:

بعد اكتمال التحميل، شغّله بالأمر التالي:

code
Bash
download
content_copy
expand_less
/home/m/Desktop/llama.cpp-latest/build/bin/llama-cli \
  -m /home/m/Desktop/GSQ-RCO-GGUF/Meta-Llama-3.1-70B-Instruct-Q2_K.gguf \
  -ngl 38 \
  -c 4096 \
  -t 8 \
  -p "مرحباً، عرفني بنفسك في سطرين:"

-ngl 38: يضع حوالي 38 طبقة داخل الـ 16GB VRAM لكرتك RTX 5060 Ti، والطبقات المتبقية في الرامات (96GB).

-t 8: عدد أنوية المعالج (يمكنك زيادتها حسب عدد أنوية الـ Xeon في HP Z440).

النتيجة المتوقعة: ستتحول السرعة من 0.3 tokens/s في vLLM إلى ما بين 6 و 12 tokens/s على الأقل!

edit
more_vert

الخطوة الكبرى تحققت: الاتصال يعمل 100%، والسيرفر استجاب واستقبل بايثون الرد وطبعه بنجاح!

لماذا ظهر الرد بهذا الشكل الغريب وتكررت الكلمات؟

طبيعة نماذج المحادثة (Instruct Models):
هذا النموذج هو Llama-3.1-Instruct. هذه النماذج لا تفهم النصوص العادية كـ "سؤال وجواب" إلا إذا وُضعت داخل قالب وسوم Llama-3 الخاصة (مثل <|start_header_id|>user...).
عندما أرسلنا له نصاً عادياً، اعتبر نفسه يكمل ملفاً نصياً خاماً أو استبياناً من الإنترنت (مثل: "املأ الفراغات: أنا [الاسم] من [البلد]...").

غياب وسم التوقف (Stop Tokens):
لم نحدد له أين ينتهي الكلام، فاستمر في توليد التوكنات وتكرار نفسه حتى انتهى الحد الأقصى (max_tokens).

التعديل في a.py لإعطائه الوسوم الصحيحة:

افتح ملف a.py وضع فيه الكود التالي مباشرة (بدون الحاجة لإعادة تشغيل السيرفر):

code
Python
download
content_copy
expand_less
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

# صياغة النص بقالب Llama-3.1 الرسمي
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

مرحباً، عرفني بنفسك في سطرين باختصار.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt=prompt,
    max_tokens=100,
    temperature=0.6,
    stop=["<|eot_id|>", "<|end_of_text|>"],  # للتوقف فور إتمام الجواب
    frequency_penalty=0.5,  # لمنع التكرار
)

print("\nرد النموذج:\n")
print(response.choices[0].text.strip())

شغّل الآن:

code
Bash
download
content_copy
expand_less
python a.py

وسيعطيك إجابة واعية ومنظمة تفيدك مباشرة بأنه المساعد الذكي Llama.

edit
more_vert
Model 2:19 AM

إليك كود a.py معدلاً بالكامل باللغة الإنجليزية بدون أي أحرف عربية:

code
Python
download
content_copy
expand_less
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello! Introduce yourself briefly in two sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt=prompt,
    max_tokens=100,
    temperature=0.6,
    stop=["<|eot_id|>", "<|end_of_text|>"],
    frequency_penalty=0.5,
)

print("Model Response:\n")
print(response.choices[0].text.strip())

شغله الآن عبر الطرفية:

code
Bash
download
content_copy
expand_less
python a.py

more_vert
Model 2:23 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

نعم بالتأكيد! إليك كود شات تفاعلي مستمر (Interactive Chat Loop) في ملف بايثون.

هذا الكود يتميز بـ:

حفظ سياق المحادثة (History): يتذكر ما قلته له في الأسئلة السابقة.

الكتابة الحية (Streaming): يطبع الكلمات حرفاً بحرف وتوكناً بتوكن فور توليدها مباشرة حتى لا تنتظر انتهاء الرد كاملاً.

متوافق مع قالب Llama-3.1 وسيرفر vLLM الحالي.

أنشئ ملفاً باسم chat.py:
code
Bash
download
content_copy
expand_less
nano ~/Desktop/GSQ-RCO-GGUF/chat.py

وضع بداخله الكود التالي:

code
Python
download
content_copy
expand_less
from openai import OpenAI
import sys

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

MODEL_NAME = "/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ"

# Conversation history storage
history = []

def build_llama3_prompt(history):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful and concise AI assistant.<|eot_id|>"
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

print("=" * 50)
print("Chat session started! Type 'exit' or 'quit' to end.")
print("=" * 50)

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat. Goodbye!")
        break

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Build the full formatted prompt with history
    full_prompt = build_llama3_prompt(history)

    print("\nAssistant: ", end="", flush=True)

    try:
        response_stream = client.completions.create(
            model=MODEL_NAME,
            prompt=full_prompt,
            max_tokens=256,
            temperature=0.6,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            frequency_penalty=0.5,
            stream=True,  # Stream tokens live as they are generated
        )

        assistant_reply = ""
        for chunk in response_stream:
            token = chunk.choices[0].text
            assistant_reply += token
            print(token, end="", flush=True)
        print()

        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_reply.strip()})

    except Exception as e:
        print(f"\n[Error]: {e}")

احفظ الملف بـ Ctrl + O ثم Enter ثم Ctrl + X.

للتشغيل:
code
Bash
download
content_copy
expand_less
python ~/Desktop/GSQ-RCO-GGUF/chat.py

يمكنك التحدث معه وسؤاله باستمرار، وسيرد عليك كلمة بكلمة مع تذكر الحوار بالكامل. ولإنهاء المحادثة اكتب exit.

more_vert
Model 2:38 AM
Thoughts

Expand to view model thoughts

chevron_right

edit
more_vert

العفو، وسعيد جداً أن كل شيء يعمل معك الآن بسلاسة!

نعم، إليك حصر دقيق لكل الملفات التي تم تعديلها والملفات التي تم إنشاؤها أثناء حل المشكلات:

أولاً: ملفات المكتبة (داخل بيئة العمل .venv لدعم كرت RTX 5060 Ti):

ملف ضبط كيرنلز المعمارية (Heuristics):

المسار:
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

التعديل: أضفنا في نهاية الملف دعم المعمارية 120 للتعامل مع كروت RTX 50-series:

code
Python
download
content_copy
expand_less
heuristics_map[120] = heuristics_map[max(heuristics_map.keys())]

(توجد بجانبه نسخة احتياطية باسم __init__.py.bak).

ملف حساب أداء أنوية Tensor Cores:

المسار:
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/utils/device.py

التعديل: في السطر 49 داخل دالة estimate_tensorcore_max_tops:

كان سابقاً:

code
Python
download
content_copy
expand_less
ops_per_clock = ops_map[sm_version]

أصبح:

code
Python
download
content_copy
expand_less
ops_per_clock = ops_map.get(sm_version, ops_map[max(ops_map.keys())])

(توجد بجانبه نسخة احتياطية باسم device.py.bak).

ثانياً: ضبط الحزم والبيئة الافتراضية:

تم تشغيل uv sync لإعادة مواءمة إصدار PyTorch المتوافق (2.11.0+cu130) مع ملفات C++ لـ vLLM.

تم تثبيت مكتبة openai داخل البيئة للتواصل مع السيرفر عبر بايثون.

ثالثاً: السكربتات التي قمت بإنشائها لتشغيل النموذج:

a.py

المسار: /home/m/Desktop/GSQ-RCO-GGUF/a.py

ملف الاختبار المفرد (إرسال سؤال واحد وتلقي الجواب بقالب Llama-3).

chat.py

المسار: /home/m/Desktop/GSQ-RCO-GGUF/chat.py

سكربت الشات التفاعلي الكامل (يحفظ سجل الحوار ويطبع الرد مباشرة بنظام الـ Streaming).

نصيحة للمستقبل: إذا قمت يوماً ما بحذف مجلد .venv أو إعادة إنشائه من الصفر، ستحتاج فقط لتطبيق تعديلي ملفي humming المذكورين أعلاه ليتعرف على كرتك من جديد.

thumb_up
thumb_down

more_vert
User 1:26 AM
docs
Paste October 25, 2026 - 1:26AM
17,255 tokens

edit
more_vert
User 1:27 AM

rtx5060ti16gb ram 96gb

edit
more_vert
User 1:31 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ
.
/
.
𝑣
𝑒
𝑛
𝑣
/
𝑏
𝑖
𝑛
/
𝑝
𝑦
𝑡
ℎ
𝑜
𝑛
−
𝑚
𝑝
𝑖
𝑝
𝑖
𝑛
𝑠
𝑡
𝑎
𝑙
𝑙
−
−
𝑢
𝑝
𝑔
𝑟
𝑎
𝑑
𝑒
ℎ
𝑢
𝑚
𝑚
𝑖
𝑛
𝑔
−
𝑘
𝑒
𝑟
𝑛
𝑒
𝑙
𝑠
/
ℎ
𝑜
𝑚
𝑒
/
𝑚
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
/
𝐺
𝑆
𝑄
/
.
𝑣
𝑒
𝑛
𝑣
/
𝑏
𝑖
𝑛
/
𝑝
𝑦
𝑡
ℎ
𝑜
𝑛
:
𝑁
𝑜
𝑚
𝑜
𝑑
𝑢
𝑙
𝑒
𝑛
𝑎
𝑚
𝑒
𝑑
𝑝
𝑖
𝑝
(
𝑔
𝑠
𝑞
)
𝑚
@
𝑚
−
𝐻
𝑃
−
𝑍
440
−
𝑊
𝑜
𝑟
𝑘
𝑠
𝑡
𝑎
𝑡
𝑖
𝑜
𝑛
:
 
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
/
𝐺
𝑆
𝑄
./.venv/bin/python−mpipinstall−−upgradehumming−kernels/home/m/Desktop/GSQ−RCO−GGUF/GSQ/.venv/bin/python:Nomodulenamedpip(gsq)m@m−HP−Z440−Workstation: /Desktop/GSQ−RCO−GGUF/GSQ

edit
more_vert
User 1:43 AM

بيستخدم كرت الشاشة المشكلة بنزل وثبت humming-kernels

edit
more_vert
User 1:47 AM

(gsq) m@m-HP-Z440-Workstation:/Desktop/GSQ-RCO-GGUF/GSQ
 echo -e "\n# RTX 50-series support\nheuristics_map[120] = heuristics_map[max(heuristics_map.keys())]" >> /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/init.py
(gsq) m@m-HP-Z440-Workstation:/Desktop/GSQ-RCO-GGUF/GSQ$ ./.venv/bin/python -c "from humming.tune import get_heuristics_class; print('Success! Loaded heuristics:', get_heuristics_class(120))"
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/init.py", line 45, in get_heuristics_class
info = DeviceInfo(device)
^^^^^^^^^^^^^^^^^^
File "/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/device.py", line 40, in new
instance._value = _get_device_info_extension()._DeviceInfo(index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: invalid CUDA device index 120

edit
more_vert
User 1:47 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ ./.venv/bin/python -c "import torch; print('Torch Version:', torch.version, '| CUDA:', torch.version.cuda)"
Torch Version: 2.14.1+cu130 | CUDA: 13.0

edit
more_vert
User 1:48 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ
𝑢
𝑣
𝑝
𝑖
𝑝
𝑙
𝑖
𝑠
𝑡
∣
𝑔
𝑟
𝑒
𝑝
−
𝐸
"
𝑡
𝑜
𝑟
𝑐
ℎ
∣
𝑣
𝑙
𝑙
𝑚
∣
ℎ
𝑢
𝑚
𝑚
𝑖
𝑛
𝑔
"
ℎ
𝑢
𝑚
𝑚
𝑖
𝑛
𝑔
−
𝑘
𝑒
𝑟
𝑛
𝑒
𝑙
𝑠
0.1.18
𝑙
𝑖
𝑜
𝑛
−
𝑝
𝑦
𝑡
𝑜
𝑟
𝑐
ℎ
0.2.4
𝑡
𝑜
𝑟
𝑐
ℎ
2.14.1
𝑡
𝑜
𝑟
𝑐
ℎ
−
𝑐
−
𝑑
𝑙
𝑝
𝑎
𝑐
𝑘
−
𝑒
𝑥
𝑡
0.1.5
𝑡
𝑜
𝑟
𝑐
ℎ
𝑎
𝑢
𝑑
𝑖
𝑜
2.11.0
𝑡
𝑜
𝑟
𝑐
ℎ
𝑣
𝑖
𝑠
𝑖
𝑜
𝑛
0.26.0
+
𝑐
𝑢
130
𝑣
𝑙
𝑙
𝑚
0.20.2
(
𝑔
𝑠
𝑞
)
𝑚
@
𝑚
−
𝐻
𝑃
−
𝑍
440
−
𝑊
𝑜
𝑟
𝑘
𝑠
𝑡
𝑎
𝑡
𝑖
𝑜
𝑛
:
 
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
/
𝐺
𝑆
𝑄
uvpiplist∣grep−E"torch∣vllm∣humming"humming−kernels0.1.18lion−pytorch0.2.4torch2.14.1torch−c−dlpack−ext0.1.5torchaudio2.11.0torchvision0.26.0+cu130vllm0.20.2(gsq)m@m−HP−Z440−Workstation: /Desktop/GSQ−RCO−GGUF/GSQ

edit
more_vert
User 1:48 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ ./.venv/bin/python -c "from humming.tune import get_heuristics_class; print('Success! Loaded heuristics:', get_heuristics_class())"
Success! Loaded heuristics: <class 'humming.tune.sm121.Sm121Heuristics'>

edit
more_vert
User 1:49 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ

edit
more_vert
User 1:49 AM

Torch path: /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/torch/init.py
W1025 01:49:21.466000 63209 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1025 01:49:21.466000 63209 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1025 01:49:21.466000 63209 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead

vLLM C++ extension FAILED: /home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb

edit
more_vert
User 1:52 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ ./.venv/bin/python -c 'import torch; import vllm._C; from humming.tune import get_heuristics_class; print("READY: GPU class:", get_heuristics_class())'
READY: GPU class: <class 'humming.tune.sm100.Sm100Heuristics'>

edit
more_vert
User 1:54 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ vllm serve ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ 
--cpu-offload-gb 20 
--gpu-memory-utilization 0.90 
--max-model-len 2048 
--enforce-eager
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.20.2
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]   █▄█▀ █     █     █     █  model   ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:299]
(APIServer pid=69732) INFO 10-25 01:52:18 [utils.py:233] non-default args: {'model_tag': 'ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ', 'model': 'ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ', 'max_model_len': 2048, 'enforce_eager': True, 'gpu_memory_utilization': 0.9, 'cpu_offload_gb': 20.0}
(APIServer pid=69732) '_ssl.c:999: The handshake operation timed out' thrown while requesting HEAD https://huggingface.co/ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ/resolve/main/config.json
(APIServer pid=69732) Retrying in 1s [Retry 1/5].

more_vert
User 1:56 AM
docs
docs
Generated File October 25, 2026 - 2:50AM.plain
18,513 tokens

edit
more_vert
User 1:57 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF/GSQ$ ./.venv/bin/python -c "from humming.utils.device import estimate_tensorcore_max_tops; print('TensorCore Tops:', estimate_tensorcore_max_tops())"
TensorCore Tops: 828.1128960000001

edit
more_vert
User 2:03 AM

ارسال  طلب

edit
more_vert
User 2:13 AM

/usr/include/features.h:319:10: note: this is the location of the previous definition
319 | # define _POSIX_C_SOURCE        202405L
|          ^~~~~~~~~~~~~~~
(APIServer pid=77339) INFO 10-25 02:10:26 [loggers.py:271] Engine 000: Avg prompt throughput: 1.6 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:10:36 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:10:46 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:10:56 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:06 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:16 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:26 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:36 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:46 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:11:56 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:06 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:16 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:26 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:36 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:46 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:12:56 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:13:06 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=77339) INFO 10-25 02:13:16 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%

edit
more_vert
Model 2:13 AM

النموذج يعمل ويولّد نصوصاً بالفعل الآن! (Running: 1 reqs).

لكن تلاحظ أن سرعة التوليد بطيئة جداً:

code
Text
download
content_copy
expand_less
Avg generation throughput: 0.3 to 0.4 tokens/s

أي توليد توكن واحد كل 3 ثوانٍ تقريباً.

لماذا السرعة 0.3 - 0.4 tokens/s؟

طبيعة معمارية vLLM مع الـ CPU Offload:
محرك vLLM مصمم أصلاً ليعمل بكامل طبقاته داخل ذاكرة كرت الشاشة (VRAM). عند تفعيل --cpu-offload-gb 20، يقوم المحرك بنقل أوزان الطبقات جيئة وذهاباً بين الرامات وكرت الشاشة أثناء كل توكن يتم توليده عبر منفذ الـ PCIe (UVAOffloader).

منفذ PCIe 3.0 في جهاز HP Z440:
الجهاز يعتمد على الجيل الثالث (PCIe 3.0 x16)، وسرعة نقل البيانات عبره حوالي 12-14 GB/s كحد أقصى، مما يجعل نقل أكثر من 16 جيجابايت لكل توكن بمثابة عنق زجاجة خانق للأداء.

كيف ترفع السرعة بشكل كبير؟

إذا أردت سرعة توليد عملية (مثلاً 10 إلى 20 tokens/sec على نفس جهازك وبدون شراء أي قطع جديدة):

الخيار الأفضل: استخدام llama.cpp (المخصص للـ CPU/GPU Hybrid)

بما أن مجلدك اسمه GSQ-RCO-GGUF، فإذا كان لديك ملف النموذج بصيغة .gguf، فإن llama.cpp يتعامل مع الرامات بشكل مختلف كلياً وذكي:

يضع جزءاً ثابتاً من الطبقات داخل VRAM الكرت (مثلاً 40 طبقة على الـ RTX 5060 Ti).

ويترك باقي الطبقات ثابتة في الرامات (96 GB RAM).

لا ينقل أوزان الطبقات عبر الـ PCIe أبداً، بل ينقل فقط التنشيطات (Activations) بحجم بضعة كيلوبايت، مما يلغي عنق الزجاجة للـ PCIe 3.0 بالكامل!

أمر تشغيله عبر llama-server:

code
Bash
download
content_copy
expand_less
./llama-server -m /path/to/Llama-3.1-70B-Q2_K.gguf -ngl 40 -c 2048 --port 8000
إذا أردت البقاء على vLLM بأعلى سرعة ممكنة:

استخدم نموذج بحجم 8B أو 14B أو 32B:
كرتك RTX 5060 Ti (16 GB) قادر على تشغيل Llama-3.1-8B بالكامل داخل الـ VRAM بدون أي أوفلود، وستحصل على سرعة هائلة تصل إلى 50 - 80 tokens/sec!

كما يمكنه تشغيل Qwen2.5-14B (4-bit) بالكامل داخل الـ VRAM بسرعة 30 - 45 tokens/sec.

9.9s

edit
more_vert
User 2:16 AM

/home/m/Desktop/llama.cpp-latest/build/bin/llama-cli
[vجرب ولكن لا يعمل الا مع vllm

more_vert
Model 2:16 AM
Thoughts

Expand to view model thoughts

chevron_right
6.7s

edit
more_vert
User 2:16 AM
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$ python a.py
"أنا [الاسم]، من [البلد]، و [الصفة]، و [الكلية]، و [الرغب]، و [الخط]، و [الذوق]، و [اللون]، و [النسك]، و [الرغب]، و [الخط]، و [الذوق]، و [اللون]، و [النسك]، و
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$

edit
more_vert

سبب عدم عمل هذا المجلد مع llama.cpp هو أن:

النموذج الموجود داخل المجلد الحالي (Llama-3.1-70B-Instruct-2Bit-GSQ) ليس بصيغة GGUF؛ بل هو ملفات safetensors مكممة بتقنية بحثية خاصة اسمها GSQ / Humming طورها فريق (ISTA-DASLab)، وهذه التقنية لا تملك كيرنلز تنفيذية إلا داخل مكتبة vLLM فقط.

بينما llama.cpp يحتاج إلى ملف ينتهي بامتداد .gguf.

كيف تشغل Llama-3.1-70B على llama.cpp بسرعة حقيقية؟

إذا أردت الحصول على سرعة ممتازة للـ 70B على رامات جهازك (96GB) وكرتك (16GB)، تحتاج إلى تحميل ملف GGUF قياسي 2-bit (مثل IQ2_M أو Q2_K)، فهذه الصيغ مدعومة أصلياً داخل llama.cpp.

1. تحميل ملف GGUF بحجم 2-bit (~25 GB):

يمكنك تحميل نسخة 2-bit الرسمية بصيغة GGUF (مثلاً من مستودع bartowski الشهير):

code
Bash
download
content_copy
expand_less
# تثبيت huggingface-cli إذا لم يكن موجوداً
uv pip install huggingface-hub

# تحميل ملف Q2_K بصيغة GGUF مباشرة داخل مجلدك
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  Meta-Llama-3.1-70B-Instruct-Q2_K.gguf \
  --local-dir /home/m/Desktop/GSQ-RCO-GGUF/
2. تشغيله بـ llama-cli مع تسريع الكرت والرامات:

بعد اكتمال التحميل، شغّله بالأمر التالي:

code
Bash
download
content_copy
expand_less
/home/m/Desktop/llama.cpp-latest/build/bin/llama-cli \
  -m /home/m/Desktop/GSQ-RCO-GGUF/Meta-Llama-3.1-70B-Instruct-Q2_K.gguf \
  -ngl 38 \
  -c 4096 \
  -t 8 \
  -p "مرحباً، عرفني بنفسك في سطرين:"

-ngl 38: يضع حوالي 38 طبقة داخل الـ 16GB VRAM لكرتك RTX 5060 Ti، والطبقات المتبقية في الرامات (96GB).

-t 8: عدد أنوية المعالج (يمكنك زيادتها حسب عدد أنوية الـ Xeon في HP Z440).

النتيجة المتوقعة: ستتحول السرعة من 0.3 tokens/s في vLLM إلى ما بين 6 و 12 tokens/s على الأقل!

12.1s

edit
more_vert
User 2:16 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF
𝑝
𝑦
𝑡
ℎ
𝑜
𝑛
𝑎
.
𝑝
𝑦
"
أنا
[
الاسم
]
،من
[
البلد
]
،و
[
الصفة
]
،و
[
الكلية
]
،و
[
الرغب
]
،و
[
الخط
]
،و
[
الذوق
]
،و
[
اللون
]
،و
[
النسك
]
،و
[
الرغب
]
،و
[
الخط
]
،و
[
الذوق
]
،و
[
اللون
]
،و
[
النسك
]
،و
(
𝑔
𝑠
𝑞
)
𝑚
@
𝑚
−
𝐻
𝑃
−
𝑍
440
−
𝑊
𝑜
𝑟
𝑘
𝑠
𝑡
𝑎
𝑡
𝑖
𝑜
𝑛
:
 
/
𝐷
𝑒
𝑠
𝑘
𝑡
𝑜
𝑝
/
𝐺
𝑆
𝑄
−
𝑅
𝐶
𝑂
−
𝐺
𝐺
𝑈
𝐹
pythona.py"أنا[الاسم]،من[البلد]،و[الصفة]،و[الكلية]،و[الرغب]،و[الخط]،و[الذوق]،و[اللون]،و[النسك]،و[الرغب]،و[الخط]،و[الذوق]،و[اللون]،و[النسك]،و(gsq)m@m−HP−Z440−Workstation: /Desktop/GSQ−RCO−GGUF

more_vert
Model 2:16 AM
Thoughts

Expand to view model thoughts

chevron_right
14.3s

edit
more_vert

الخطوة الكبرى تحققت: الاتصال يعمل 100%، والسيرفر استجاب واستقبل بايثون الرد وطبعه بنجاح!

لماذا ظهر الرد بهذا الشكل الغريب وتكررت الكلمات؟

طبيعة نماذج المحادثة (Instruct Models):
هذا النموذج هو Llama-3.1-Instruct. هذه النماذج لا تفهم النصوص العادية كـ "سؤال وجواب" إلا إذا وُضعت داخل قالب وسوم Llama-3 الخاصة (مثل <|start_header_id|>user...).
عندما أرسلنا له نصاً عادياً، اعتبر نفسه يكمل ملفاً نصياً خاماً أو استبياناً من الإنترنت (مثل: "املأ الفراغات: أنا [الاسم] من [البلد]...").

غياب وسم التوقف (Stop Tokens):
لم نحدد له أين ينتهي الكلام، فاستمر في توليد التوكنات وتكرار نفسه حتى انتهى الحد الأقصى (max_tokens).

التعديل في a.py لإعطائه الوسوم الصحيحة:

افتح ملف a.py وضع فيه الكود التالي مباشرة (بدون الحاجة لإعادة تشغيل السيرفر):

code
Python
download
content_copy
expand_less
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

# صياغة النص بقالب Llama-3.1 الرسمي
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

مرحباً، عرفني بنفسك في سطرين باختصار.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt=prompt,
    max_tokens=100,
    temperature=0.6,
    stop=["<|eot_id|>", "<|end_of_text|>"],  # للتوقف فور إتمام الجواب
    frequency_penalty=0.5,  # لمنع التكرار
)

print("\nرد النموذج:\n")
print(response.choices[0].text.strip())

شغّل الآن:

code
Bash
download
content_copy
expand_less
python a.py

وسيعطيك إجابة واعية ومنظمة تفيدك مباشرة بأنه المساعد الذكي Llama.

17.9s

edit
more_vert
User 2:19 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$ python a.py

رد النموذج:

أنا مساعد الذكي، و ساعدي في كل ما يخص اللغة.
(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$

u]g الكود بدون عربى

edit
more_vert
User 2:23 AM

(gsq) m@m-HP-Z440-Workstation:~/Desktop/GSQ-RCO-GGUF$ python a.py

رد النموذج:

I'm just a computer program, so I don't have feelings or emotions like humans do. However, I'm functioning properly and ready to assist you with any questions or tasks you may have! How can I help you today?

iهل يمكن كود شات ديالوج مع النموذج

edit
more_vert

العفو، وسعيد جداً أن كل شيء يعمل معك الآن بسلاسة!

نعم، إليك حصر دقيق لكل الملفات التي تم تعديلها والملفات التي تم إنشاؤها أثناء حل المشكلات:

أولاً: ملفات المكتبة (داخل بيئة العمل .venv لدعم كرت RTX 5060 Ti):

ملف ضبط كيرنلز المعمارية (Heuristics):

المسار:
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/tune/__init__.py

التعديل: أضفنا في نهاية الملف دعم المعمارية 120 للتعامل مع كروت RTX 50-series:

code
Python
download
content_copy
expand_less
heuristics_map[120] = heuristics_map[max(heuristics_map.keys())]

(توجد بجانبه نسخة احتياطية باسم __init__.py.bak).

ملف حساب أداء أنوية Tensor Cores:

المسار:
/home/m/Desktop/GSQ-RCO-GGUF/GSQ/.venv/lib/python3.11/site-packages/humming/utils/device.py

التعديل: في السطر 49 داخل دالة estimate_tensorcore_max_tops:

كان سابقاً:

code
Python
download
content_copy
expand_less
ops_per_clock = ops_map[sm_version]

أصبح:

code
Python
download
content_copy
expand_less
ops_per_clock = ops_map.get(sm_version, ops_map[max(ops_map.keys())])

(توجد بجانبه نسخة احتياطية باسم device.py.bak).

ثانياً: ضبط الحزم والبيئة الافتراضية:

تم تشغيل uv sync لإعادة مواءمة إصدار PyTorch المتوافق (2.11.0+cu130) مع ملفات C++ لـ vLLM.

تم تثبيت مكتبة openai داخل البيئة للتواصل مع السيرفر عبر بايثون.

ثالثاً: السكربتات التي قمت بإنشائها لتشغيل النموذج:

a.py

المسار: /home/m/Desktop/GSQ-RCO-GGUF/a.py

ملف الاختبار المفرد (إرسال سؤال واحد وتلقي الجواب بقالب Llama-3).

chat.py

المسار: /home/m/Desktop/GSQ-RCO-GGUF/chat.py

سكربت الشات التفاعلي الكامل (يحفظ سجل الحوار ويطبع الرد مباشرة بنظام الـ Streaming).

نصيحة للمستقبل: إذا قمت يوماً ما بحذف مجلد .venv أو إعادة إنشائه من الصفر، ستحتاج فقط لتطبيق تعديلي ملفي humming المذكورين أعلاه ليتعرف على كرتك من جديد.

thumb_up
thumb_down
14.4s


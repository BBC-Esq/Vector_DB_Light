# constants.py

priority_libs = {
    "cp311": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=70b3d8bfe04438006ec880c162b0e3aaac90c48b759aa41638dd714c732b182c",
            "https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=7a1eb6154e05b8056b34c7a41495e09d57f79eb0180eb4e7f3bb2a61845ca8ea",
            "triton-windows==3.4.0.post20",
            "xformers==0.0.32.post2",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
            "nvidia-ml-py==13.580.82",
        ],
        "CPU": [
            # "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-win_amd64.whl",
            # "https://download.pytorch.org/whl/cpu/torchvision-0.23.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=51603eb071d0681abc4db98b10ff394ace31f425852e8de249b91c09c60eb19a",
            # "https://download.pytorch.org/whl/cpu/torchaudio-2.8.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=db37df7eee906f8fe0a639fdc673f3541cb2e173169b16d4133447eb922d1938"
        ],
        "COMMON": [
            # "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp311-cp311-win_amd64.whl",
        ],
    },
    "cp312": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=20fa9c7362a006776630b00b8a01919fedcf504a202b81358d32c5aef39956fe",
            "https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=cce3a60cd9a97f7360c8f95504ac349311fb7d6b9b826135936764f4de5f782d",
            "triton-windows==3.4.0.post20",
            "xformers==0.0.32.post2",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
            "nvidia-ml-py==13.580.82",
        ],
        "CPU": [
            # "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl",
            # "https://download.pytorch.org/whl/cpu/torchvision-0.23.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=a651ccc540cf4c87eb988730c59c2220c52b57adc276f044e7efb9830fa65a1d",
            # "https://download.pytorch.org/whl/cpu/torchaudio-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=9b302192b570657c1cc787a4d487ae4bbb7f2aab1c01b1fcc46757e7f86f391e"
        ],
        "COMMON": [
            # "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp312-cp312-win_amd64.whl",
        ]
    }
}

libs = [
    "accelerate==1.11.0",
    "aiofiles==25.1.0",
    "aiohappyeyeballs==2.6.1",
    "aiohttp==3.13.2", # langchain libraries require <4
    "aiosignal==1.4.0", #aiohttp requires >=1.4.0
    "anndata==0.12.5",
    "annotated-types==0.7.0",
    "anyio==4.11.0",
    "array_api_compat==1.12.0", # only anndata requires
    "async-timeout==5.0.1",
    "attrs==25.4.0",
    "av==16.0.1",
    "backoff==2.2.1",
    "beautifulsoup4==4.14.2",
    "bitsandbytes==0.48.2",
    "braceexpand==0.1.7",
    "certifi==2025.10.5",
    "cffi==2.0.0",
    "chardet==5.2.0",
    "charset-normalizer==3.4.4", # requests requires <4
    "click==8.3.0",
    "cloudpickle==3.1.2", # only required by tiledb-cloud and 3+ is only supported by tiledb-cloud 0.13+
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "cryptography==46.0.3", # only required by unstructured and pdfminer.six
    "dataclasses-json==0.6.7",
    "datasets==4.3.0",
    "deepdiff==8.6.1", # required by unstructured
    "diffusers==0.35.2", # required by chatterbox-lite
    "dill==0.3.8", # datasets requires <0.3.9; multiprocess requires >=0.3.8
    "distro==1.9.0",
    "docx2txt==0.9",
    "donfig==0.8.1.post1", # only required by zarr
    "einops==0.8.1",
    "einx==0.3.0",
    "emoji==2.15.0",
    "encodec==0.1.1",
    "et-xmlfile==2.0.0", # openpyxl requires; caution...openpyxl 3.1.5 (6/28/2024) predates et-xmlfile 2.0.0 (10/25/2024)
    "eval-type-backport==0.2.2", # only required by unstructured
    "filetype==1.2.0",
    "filelock==3.20.0",
    "frozendict==2.4.6",
    "frozenlist==1.8.0",
    "fsspec[http]==2025.9.0", # datasets requires...
    "greenlet==3.2.4",
    "gTTS==2.5.4",
    "h11==0.16.0",
    "h5py==3.15.1",
    "hf-xet==1.2.0",
    "html5lib==1.1", # only required by unstructured
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "httpx-sse==0.4.3",
    "huggingface-hub==0.36.0", # tokenizers requires <1.0
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.2",
    "idna==3.11",
    "img2pdf==0.6.1",
    "importlib_metadata==8.7.0",
    "Jinja2==3.1.6",
    "jiter==0.11.1", # required by openai newer versions
    "joblib==1.5.2",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "jsonschema", # only required by tiledb-cloud
    "langchain==0.3.27",
    "langchain-community==0.3.31",
    "langchain-core==0.3.79",
    "langchain-huggingface==0.3.1",
    "langchain-text-splitters==0.3.11",
    "langdetect==1.0.9",
    "langsmith==0.4.37",
    "legacy-api-wrap==1.5", # only required by anndata
    "lxml==6.0.2",
    "Markdown==3.9",
    "markdown-it-py==4.0.0",
    "MarkupSafe==3.0.3",
    "marshmallow==3.26.1", # dataclasses-json requires <4.0.0
    "mdurl==0.1.2",
    "more-itertools==10.8.0",
    "mpmath==1.3.0", # sympy 1.13.1 requires <1.4
    "msg-parser==1.2.0",
    "multidict==6.7.0",
    "multiprocess==0.70.16", # datasets requires <0.70.17
    "mypy-extensions==1.1.0",
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.5",
    "nltk==3.9.2", # not higher; gives unexplained error
    "numcodecs==0.16.3", # only required by zarr
    "numpy==2.3.4", # numba 0.61.2 requires <2.3
    "olefile==0.47",
    "openai==2.6.1", # only required by chat_lm_studio.py script and whispers2t (if using openai vanilla backend)
    "openpyxl==3.1.5",
    "optimum==2.0.0",
    "ordered-set==4.1.0",
    "orderly-set==5.5.0", # deepdiff 8.2.0 requires >=5.3.0,<6
    "orjson==3.11.4",
    "packaging==25.0",
    "pandas==2.3.3",
    "pillow==12.0.0",
    "pipdeptree",
    "platformdirs==4.5.0",
    "propcache==0.4.1",
    "protobuf==6.33.0",
    "psutil==7.1.3",
    "pyarrow==22.0.0",
    "pycparser==2.23",
    "pydantic==2.12.3",
    "pydantic_core==2.41.4", # pydantic 2.11.7 requires ==2.37.2; CAUTION, package checker is incorrect, check repo instead
    "pydantic-settings==2.11.0", # langchain-community requires >=2.4.0,<3.0.0
    "Pygments==2.19.2",
    "pypandoc==1.15",
    "pypdf==6.1.3",
    "pyreadline3==3.5.4",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.2.0",
    "python-dotenv==1.2.1",
    "python-iso639==2025.2.18",
    "python-magic==0.4.27",
    "python-oxmsg==0.0.2", # only required by unstructured library
    "pytz==2025.2",
    "PyYAML==6.0.3",
    "rapidfuzz==3.14.3",
    "regex==2025.10.23",
    "requests==2.32.5",
    "requests-toolbelt==1.0.0",
    "rich==14.2.0",
    "ruamel.yaml==0.18.16",
    "ruamel.yaml.clib==0.2.14",
    "safetensors==0.6.2",
    "scikit-learn==1.7.2",
    "scipy==1.16.3",
    "sentence-transformers==5.1.2",
    "sentencepiece==0.2.1",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sounddevice==0.5.3",
    "soundfile==0.13.1",
    "soupsieve==2.8",
    "speechbrain==1.0.3",
    "SQLAlchemy==2.0.44", # langchain and langchain-community require <3.0.0
    "sympy==1.13.3", # torch 2.8.0 requires 1.13.3
    "tabulate2==1.10.2",
    "tblib==3.2.1", # only required by tiledb-cloud
    "tenacity==9.1.2",
    "termcolor==3.2.0",
    "threadpoolctl==3.6.0",
    "tiktoken==0.12.0",
    "tiledb==0.35.2", # requires numpy>=1.25
    "tiledb-cloud==0.14.3", # requires attrs>=21.4.0, tblib>=1.7, tiledb~=0.30,!=0.33.1,!=0.33.2
    "tiledb-vector-search==0.15.0", # requires tiledb-cloud>=0.11, tiledb>=0.35.1, numpy>=1.25.0
    "timm==1.0.21",
    "tokenizers==0.22.1",
    "tqdm==4.67.1",
    "transformers==4.57.1",
    "typing-inspection==0.4.2", # required by pydantic and pydantic-settings
    "typing_extensions==4.15.0", # unstructured 0.18.15 requires 4.15.0
    "unstructured-client==0.42.3",
    "tzdata==2025.2",
    "urllib3==2.5.0", # requests requires <3
    "vector-quantize-pytorch==1.24.2",
    "vocos==0.1.0",
    "watchdog==6.0.0",
    "wcwidth==0.2.14", # only required by tabulate2
    "webencodings==0.5.1", # only required by html5lib
    "wrapt==1.17.3", # unstructured 0.18.15 requires 1.17.3
    "xlrd==2.0.2",
    "xxhash==3.6.0",
    "yarl==1.22.0", # aiohttp requires <2
    "zarr==3.1.3", # only required by anndata
    "zipp==3.23.0",
    "zstandard==0.25.0" # only required by langsmith 3+
]

full_install_libs = [
    "PySide6==6.10.0",
    "pymupdf==1.26.5",
    "unstructured==0.18.15"
]

VECTOR_MODELS = {
    'BAAI': [
        {
            'name': 'bge-small-en-v1.5',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'BAAI/bge-small-en-v1.5',
            'cache_dir': 'BAAI--bge-small-en-v1.5',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32',
            'license': 'mit',
        },
        {
            'name': 'bge-base-en-v1.5',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'BAAI/bge-base-en-v1.5',
            'cache_dir': 'BAAI--bge-base-en-v1.5',
            'type': 'vector',
            'parameters': '109m',
            'precision': 'float32',
            'license': 'mit',
        },
        {
            'name': 'bge-large-en-v1.5',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'BAAI/bge-large-en-v1.5',
            'cache_dir': 'BAAI--bge-large-en-v1.5',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32',
            'license': 'mit',
        },
    ],
    'Qwen': [
        {
            'name': 'Qwen3-Embedding-0.6B',
            'dimensions': 1024,
            'max_sequence':8192,
            'size_mb': 1190,
            'repo_id': 'Qwen/Qwen3-Embedding-0.6B',
            'cache_dir': 'Qwen--Qwen3-Embedding-0.6B',
            'type': 'vector',
            'parameters': '596m',
            'precision': 'bfloat16',
            'license': 'apache-2.0',
        },
    ],
}

DOCUMENT_LOADERS = {
    # ".pdf": "PyMuPDFLoader",
    ".pdf": "CustomPyMuPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".enex": "EverNoteLoader",
    ".epub": "UnstructuredEPubLoader",
    ".eml": "UnstructuredEmailLoader",
    ".msg": "UnstructuredEmailLoader",
    ".csv": "CSVLoader",
    ".xls": "UnstructuredExcelLoader",
    ".xlsx": "UnstructuredExcelLoader",
    ".xlsm": "UnstructuredExcelLoader",
    ".rtf": "UnstructuredRTFLoader",
    ".odt": "UnstructuredODTLoader",
    ".md": "UnstructuredMarkdownLoader",
    ".html": "BSHTMLLoader",
}

TOOLTIPS = {
    "CHOOSE_FILES": "Select documents to add to the database. Remember to transcribe audio files in the Tools tab first.",
    "CHUNK_OVERLAP": "Characters shared between chunks. Set to 25-50% of chunk size.",
    "CHUNK_SIZE": (
        "<html><body>"
        "Upper limit (in characters, not tokens) that a chunk can be after being split.  Make sure that it falls within"
        "the Max Sequence of the embedding model being used, which is measured in tokens (not characters), remembering that"
        "approximately 3-4 characters = 1 token."
        "</body></html>"
    ),
    "CHUNKS_ONLY": "Solely query the vector database and get relevant chunks. Very useful to test the chunk size/overlap settings.",
    "CONTEXTS": "Maximum number of chunks (aka contexts) to return.",
    "COPY_RESPONSE": "Copy the chunks (if chunks only is checked) or model's response to the clipboard.",
    "CREATE_DEVICE_DB": "Choose 'cpu' or 'cuda'. Use 'cuda' if available.",
    "CREATE_DEVICE_QUERY": "Choose 'cpu' or 'cuda'. 'cpu' recommended to conserve VRAM.",
    "CREATE_VECTOR_DB": "Creates a new vector database.",
    "DATABASE_NAME_INPUT": "Enter a unique database name. Use only lowercase letters, numbers, underscores, and hyphens.",
    "DATABASE_SELECT": "Vector database that will be queried.",
    "DOWNLOAD_MODEL": "Download the selected vector model.",
    "FILE_TYPE_FILTER": "Only allows chunks that originate from certain file types.",
    "HALF_PRECISION": "Uses bfloat16/float16 for 2x speedup. Requires a GPU.",
    "MODEL_BACKEND_SELECT": "Choose the backend for the large language model response.",
    "PORT": "Must match the port used in LM Studio.",
    "QUESTION_INPUT": "Type your question here or use the voice recorder.",
    "SEARCH_TERM_FILTER": "Removes chunks without exact term. Case-insensitive.",
    "SELECT_VECTOR_MODEL": "Choose the vector model for text embedding.",
    "SIMILARITY": "Relevance threshold for chunks. 0-1, higher returns more. Don't use 1.",
    "VECTOR_MODEL_DIMENSIONS": "Higher dimensions captures more nuance but requires more processing time.",
    "VECTOR_MODEL_DOWNLOADED": "Whether the model has been downloaded.",
    "VECTOR_MODEL_LINK": "Huggingface link.",
    "VECTOR_MODEL_MAX_SEQUENCE": "Number of tokens the model can process at once. Different from the Chunk Size setting, which is in characters.",
    "VECTOR_MODEL_NAME": "The name of the vector model.",
    "VECTOR_MODEL_PARAMETERS": "The number of internal weights and biases that the model learns and adjusts during training.",
    "VECTOR_MODEL_PRECISION": (
        "<html>"
        "<body>"
        "<p style='font-size: 14px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 10px;'>"
        "<b>The precision ultimately used depends on your setup:</b></p>"
        "<table style='border-collapse: collapse; width: 100%; font-size: 12px; color: #34495e;'>"
        "<thead>"
        "<tr style='background-color: #ecf0f1; text-align: left;'>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Compute Device</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Embedding Model Precision</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>'Half' Checked?</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Precision Ultimately Used</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CPU</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Any</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Either</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float16</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>bfloat16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.0) or <code>float16</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>No</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.0) or <code>float16</code>"
        "</td>"
        "</tr>"
        "</tbody>"
        "</table>"
        "</body>"
        "</html>"
    ),
    "VECTOR_MODEL_SELECT": "Choose a vector model to download.",
    "VECTOR_MODEL_SIZE": "Size on disk.",
}

system_message = "You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."
rag_string = "Here are the contexts to base your answer on.  However, I need to reiterate that I only want you to base your response on these contexts and do not use outside knowledge that you may have been trained with."
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json
# GeVLM

## Configure LLM

1. **Download the Model**:
   - Visit [https://huggingface.co/lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and download the model files.

2. **Update `scripts/config.py`**:
   - Open the `scripts/config.py` file.
   - Locate the line where the model path is defined and update it to point to the downloaded. For example:
     ```python
     llama_model_path = "/path/to/vicuna-7b-v1.5"
     ```
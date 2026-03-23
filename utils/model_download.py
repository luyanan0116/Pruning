from modelscope import snapshot_download

# 指定自定义下载路径 (例如：/data/models 或 D:\AI\Models)
custom_cache_dir = 'D:\WorkSpace\Pruning\models' 

model_dir = snapshot_download(
    'shakechen/Llama-2-7b-hf',
    cache_dir=custom_cache_dir
)

print(f"模型已下载到: {model_dir}")
```python
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-pokemon-diffusers", torch_dtype=torch.float16)  
pipe = pipe.to("cuda")
```

    /home/team08/.local/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Fetching 15 files: 100%|██████████| 15/15 [00:00<00:00, 91180.52it/s]
    /home/team08/.local/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



```python
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

```


```python
prompt = "Beyonce"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:14<00:00,  3.64it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_2_1.png)
    




```python
prompt = "Dean Lewis"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.83it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_3_1.png)
    




```python
prompt = "Nicki Minaj"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.83it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_4_1.png)
    




```python

```


```python
prompt = "Elvis"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.82it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_6_1.png)
    




```python
prompt = "Slipknot"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.82it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_7_1.png)
    




```python
prompt = "Ed Sheeran"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.82it/s]
    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.





    
![png](output_8_1.png)
    




```python
prompt = "Lana Del Rey"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.82it/s]





    
![png](output_9_1.png)
    




```python
prompt = "Lady Gaga"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.80it/s]





    
![png](output_10_1.png)
    




```python
prompt = "Tesla"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, True
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

grid = image_grid(images, rows=2, cols=2)
grid
```

    100%|██████████| 51/51 [00:13<00:00,  3.83it/s]





    
![png](output_11_1.png)
    




```python

```

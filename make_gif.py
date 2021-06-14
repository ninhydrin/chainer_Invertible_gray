from PIL import Image
from pathlib import Path
from tqdm import tqdm

imgs_path = Path("result/preview")
gif = []
for i in tqdm(range(125)):
    name = f"image000{i * 400 + 10:05}.png"
    img_path = imgs_path / name

    if img_path.exists():
        # gif.append(Image.open(img_path).resize((1008, 336)))
        gif.append(Image.open(img_path).resize((1008, 336)))
gif[0].save("invertible_gray.gif", save_all=True, append_images=gif[1:])

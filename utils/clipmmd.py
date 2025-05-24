import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPMMD:
    def __init__(self, device, reference_images_dir, cfg):
        self.device = device
        self.cfg = cfg
        self.reference_images_dir = reference_images_dir
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.reference_features = self._load_or_compute_reference_features()

    def _load_or_compute_reference_features(self):
        """
        Load cached reference features if available, otherwise compute and save them in batches.
        """
        cache_path = os.path.join(self.reference_images_dir, "reference_features.pt")

        # Check if cached features exist
        if os.path.exists(cache_path):
            print(f"Loading cached reference features from {cache_path}")
            return torch.load(cache_path, map_location=self.device)

        # Compute reference features if cache is not available
        print(f"Computing reference features in batches and saving to {cache_path}")
        reference_images = []
        num_images = 0
        for filename in os.listdir(self.reference_images_dir):
            if num_images == self.cfg.training.sample_grid_dimension**2:
                break
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.reference_images_dir, filename)
                reference_images.append(Image.open(image_path).convert("RGB"))
                num_images += 1

        if not reference_images:
            raise ValueError(f"No valid images found in directory: {self.reference_images_dir}")

        batch_size = 64  # Adjust batch size based on your GPU memory
        all_features = []

        for i in range(0, len(reference_images), batch_size):
            batch_images = reference_images[i:i + batch_size]
            inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                batch_features = self.clip_model.get_image_features(**inputs)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features)

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)

        # Save computed features to cache
        torch.save(all_features, cache_path)
        return all_features

    def compute_mmd(self, generated_images):
        """
        Compute Maximum Mean Discrepancy (MMD) between generated and reference images.
        """
        inputs = self.clip_processor(images=generated_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_features = self.clip_model.get_image_features(**inputs)
        generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)

        # Compute MMD (L2 distance between generated and reference features)
        mmd = torch.mean((generated_features - self.reference_features) ** 2).item()
        return mmd
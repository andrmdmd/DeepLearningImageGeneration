import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPMMD:
    def __init__(self, device, reference_images_dir):
        self.device = device
        self.reference_images_dir = reference_images_dir
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.reference_features = self._load_or_compute_reference_features()

    def _load_or_compute_reference_features(self):
        """
        Load cached reference features if available, otherwise compute and save them.
        """
        cache_path = os.path.join(self.reference_images_dir, "reference_features.pt")

        # Check if cached features exist
        if os.path.exists(cache_path):
            print(f"Loading cached reference features from {cache_path}")
            return torch.load(cache_path, map_location=self.device)

        # Compute reference features if cache is not available
        print(f"Computing reference features and saving to {cache_path}")
        reference_images = []
        for filename in os.listdir(self.reference_images_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.reference_images_dir, filename)
                reference_images.append(Image.open(image_path).convert("RGB"))

        inputs = self.clip_processor(images=reference_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

        # Save computed features to cache
        torch.save(features, cache_path)
        return features

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
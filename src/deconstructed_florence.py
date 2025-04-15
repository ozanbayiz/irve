import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
import sys # For exiting cleanly
import os

class FlorenceVisionEncoder(nn.Module):
    """
    Encapsulates the vision encoding part of Florence-2, allowing for intermediate feature extraction.
    Assumes single-frame input (T=1) based on the reference _encode_image method.
    """
    def __init__(
        self,
        vision_tower,
        image_pos_embed,
        visual_temporal_embed,
        image_projection,
        image_proj_norm,
        vision_config,
        device,
        dtype
    ):
        super().__init__()
        self.conv_embeds = vision_tower.convs
        self.blocks = vision_tower.blocks
        self.pos_embed = image_pos_embed
        self.temporal_embed = visual_temporal_embed # Kept for completeness, but logic assumes T=1
        self.projection_layer = image_projection
        self.proj_norm = image_proj_norm
        self.config = vision_config
        self.device = device
        self.dtype = dtype
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Internal encoding logic, replicating Florence2ForConditionalGeneration._encode_image.
        """
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D (B, C, H, W), got {pixel_values.ndim}D")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        batch_size, _, H, W = pixel_values.shape
        T = 1 # Hardcoded assumption based on reference _encode_image
        current_size = (H, W)
        x = pixel_values

        # Pass through convolutional embeddings and blocks
        for i in range(len(self.blocks)):
            x, current_size = self.conv_embeds[i](x, current_size)
            x, current_size = self.blocks[i](x, current_size)

        # Apply positional embeddings
        if self.pos_embed is not None:
            # Reshape for positional embedding which expects (B*T, H, W, C) after view
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h = w = int(num_tokens ** 0.5) # Assumes square feature map
            assert h * w == num_tokens, 'Only square feature maps supported for pos embed in this impl.'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1]) # Reshape back to (B, T*num_tokens, C)

        # Apply temporal embeddings (only relevant if T > 1, which is not assumed here)
        if self.temporal_embed is not None:
            visual_temporal_embed = self.temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        # Feature selection based on config (mirroring original _encode_image)
        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for source in self.config.image_feature_source:
            if source not in x_feat_dict:
                raise ValueError(f"Invalid image feature source: {source}")
            feature_tensor = x_feat_dict[source]
            new_x.append(feature_tensor)

        # Concatenate selected features along the sequence dimension
        x = torch.cat(new_x, dim=1)

        # Apply projection and normalization
        x = x @ self.projection_layer
        image_features = self.proj_norm(x)

        # Return final features and optionally the intermediate ones collected earlier
        return image_features
    

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input pixel values into image features.
        """
        final_features, _ = self._encode_internal(pixel_values, return_intermediates=False)
        return final_features

    def encode_with_intermediate_outputs(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encodes the input pixel values and returns both final features and intermediate block outputs.
        """
        return self._encode_internal(pixel_values, return_intermediates=True)

    def get_latent_representations(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns the outputs of all blocks in the vision tower.
        """
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D (B, C, H, W), got {pixel_values.ndim}D")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        batch_size, _, H, W = pixel_values.shape
        current_size = (H, W)
        x = pixel_values
        intermediate_outputs = []

        # Pass through convolutional embeddings and blocks
        for i in range(len(self.blocks)):
            x, current_size = self.conv_embeds[i](x, current_size)
            x, current_size = self.blocks[i](x, current_size)
            intermediate_outputs.append(x.clone())

        return intermediate_outputs

    def _encode_internal(
        self,
        pixel_values: torch.Tensor,
        return_intermediates: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Internal encoding logic, replicating Florence2ForConditionalGeneration._encode_image.
        """
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D (B, C, H, W), got {pixel_values.ndim}D")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        batch_size, _, H, W = pixel_values.shape
        T = 1 # Hardcoded assumption based on reference _encode_image
        current_size = (H, W)
        x = pixel_values
        intermediate_outputs = []

        # Pass through convolutional embeddings and blocks
        for i in range(len(self.blocks)):
            x, current_size = self.conv_embeds[i](x, current_size)
            x, current_size = self.blocks[i](x, current_size)
            if return_intermediates:
                intermediate_outputs.append(x.clone())

        # Apply positional embeddings
        if self.pos_embed is not None:
            # Reshape for positional embedding which expects (B*T, H, W, C) after view
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h = w = int(num_tokens ** 0.5) # Assumes square feature map
            assert h * w == num_tokens, 'Only square feature maps supported for pos embed in this impl.'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1]) # Reshape back to (B, T*num_tokens, C)

        # Apply temporal embeddings (only relevant if T > 1, which is not assumed here)
        if self.temporal_embed is not None:
            visual_temporal_embed = self.temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        # Feature selection based on config (mirroring original _encode_image)
        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for source in self.config.image_feature_source:
            if source not in x_feat_dict:
                raise ValueError(f"Invalid image feature source: {source}")
            feature_tensor = x_feat_dict[source]
            new_x.append(feature_tensor)

        # Concatenate selected features along the sequence dimension
        x = torch.cat(new_x, dim=1)

        # Apply projection and normalization
        x = x @ self.projection_layer
        image_features = self.proj_norm(x)

        # Return final features and optionally the intermediate ones collected earlier
        return image_features, intermediate_outputs


class DeconstructedFlorence2:
    """
    Orchestrator class that loads Florence-2 (using AutoModelForCausalLM as per user)
    and allows explicit vision encoding and generation.
    """
    def __init__(self, model_name_or_path, device='cuda', dtype=torch.bfloat16, trust_remote_code=True):
        # User constraint: Load using AutoModelForCausalLM.
        # This assumes the loaded model has the necessary Florence-2 attributes.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ).to(device).to(dtype)

        self.model.eval()

        # Assume these attributes exist on the loaded model
        self.vision_config = self.model.config.vision_config
        self.text_config = self.model.config.text_config
        self.device = device
        self.dtype = dtype

        # Instantiate the vision encoder using attributes assumed to be present
        self.vision_encoder = FlorenceVisionEncoder(
            vision_tower=self.model.vision_tower,
            image_pos_embed=getattr(self.model, 'image_pos_embed', None), # Use getattr for safer access
            visual_temporal_embed=getattr(self.model, 'visual_temporal_embed', None),
            image_projection=self.model.image_projection,
            image_proj_norm=self.model.image_proj_norm,
            vision_config=self.vision_config,
            device=self.device,
            dtype=self.dtype
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes image using the decoupled vision encoder. Handles device/dtype placement.
        """
        with torch.no_grad():
             pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
             image_features = self.vision_encoder.encode(pixel_values)
        return image_features

    def encode_image_with_intermediates(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encodes image using the decoupled vision encoder and returns intermediates. Handles device/dtype placement.
        """
        with torch.no_grad():
             pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
             final_features, intermediate_outputs = self.vision_encoder.encode_with_intermediate_outputs(pixel_values)
        return final_features, intermediate_outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        **generation_kwargs
    ):
        """
        Generates text conditioned on the image, performing vision encoding explicitly first.
        Relies on self.model having _merge_input_ids_with_image_features and language_model.generate.
        """
        with torch.no_grad():
            # 1. Encode image explicitly (handles device/dtype inside)
            image_features = self.encode_image(pixel_values) # (B, img_seq_len, D)

            # 2. Embed text prompt
            input_ids = input_ids.to(self.device)
            # Assumes get_input_embeddings method exists
            text_embeds = self.model.get_input_embeddings()(input_ids) # (B, text_seq_len, D)
            text_embeds = text_embeds.to(self.dtype) # Ensure dtype match

            # 3. Prepare combined inputs using the model's internal merging method
            # Assumes _merge_input_ids_with_image_features method exists and works correctly
            inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(image_features, text_embeds)
            attention_mask = attention_mask.to(self.device) # Ensure mask is on device

        # 4. Call language model generate
        # Assumes language_model attribute exists and has a generate method
        generated_ids = self.model.language_model.generate(
            input_ids=None, # We provide embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        return generated_ids

# --- Example Usage and Testing ---
if __name__ == '__main__':
    # Basic setup
    model_id = "microsoft/Florence-2-base"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using device: {device}, dtype: {compute_dtype}")

    # Load models and processor
    try:
        deconstructed_model = DeconstructedFlorence2(model_id, device=device, dtype=compute_dtype, trust_remote_code=True)
        # Load original model class for comparison *if* needed, otherwise use deconstructed_model.model
        # For this test, we strictly compare against the model loaded inside DeconstructedFlorence2
        original_model_ref = deconstructed_model.model
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Models and processor loaded.")
    except Exception as e:
        print(f"Error loading models or processor: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare inputs
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        print("Inputs prepared.")
    except Exception as e:
        print(f"Error preparing inputs: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Test Vision Encoder Output ---
    print("\nTesting Vision Encoder equivalence...")
    try:
        deconstructed_features = deconstructed_model.encode_image(pixel_values)
        with torch.no_grad():
             # Call internal method on the reference model instance
             original_features = original_model_ref._encode_image(
                 pixel_values.to(device=device, dtype=compute_dtype)
             )

        # Compare outputs
        atol = 1e-3 if compute_dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if compute_dtype == torch.bfloat16 else 1e-4
        if torch.allclose(deconstructed_features, original_features, atol=atol, rtol=rtol):
            print(f"  SUCCESS: Vision encoder outputs match within tolerance (atol={atol}, rtol={rtol}).")
        else:
            print(f"  FAILURE: Vision encoder outputs DO NOT match within tolerance (atol={atol}, rtol={rtol}).")
            diff = torch.abs(deconstructed_features - original_features)
            print(f"    Max absolute difference: {diff.max().item():.4f}")
            print(f"    Mean absolute difference: {diff.mean().item():.4f}")

    except AttributeError as ae:
         print(f"  Error during vision comparison: {ae}", file=sys.stderr)
         print("  Hint: This might indicate the model loaded by AutoModelForCausalLM is missing the '_encode_image' method.", file=sys.stderr)
    except Exception as e:
        print(f"  Error during vision encoder comparison: {e}", file=sys.stderr)

    # --- Test Generation Equivalence ---
    print("\nTesting Generation equivalence...")
    generation_kwargs = {
        "max_new_tokens": 50,
        "num_beams": 3,
        "do_sample": False
    }

    deconstructed_decoded_text = "[Deconstructed generation failed]"
    original_decoded_text = "[Original generation failed]"
    deconstructed_ids = None
    original_ids = None

    try:
        # Deconstructed generation
        deconstructed_ids = deconstructed_model.generate(
            input_ids=input_ids.clone(), # Use clone to avoid potential side effects
            pixel_values=pixel_values.clone(),
            **generation_kwargs
        )
        deconstructed_decoded_text = processor.batch_decode(deconstructed_ids, skip_special_tokens=True)[0]
        print(f"Deconstructed Generation Output: '{deconstructed_decoded_text}'")
    except AttributeError as ae:
         print(f"Error during deconstructed generation: {ae}", file=sys.stderr)
         print("  Hint: This might indicate missing 'get_input_embeddings', '_merge_input_ids_with_image_features', or 'language_model' on the loaded model.", file=sys.stderr)
    except Exception as e:
        print(f"Error during deconstructed generation: {e}", file=sys.stderr)

    try:
        # Original model generation (using the reference instance)
        hf_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=device, dtype=compute_dtype)
        with torch.no_grad():
            original_ids = original_model_ref.generate(**hf_inputs, **generation_kwargs)
        original_decoded_text = processor.batch_decode(original_ids, skip_special_tokens=True)[0]
        print(f"Original Generation Output:      '{original_decoded_text}'")
    except AttributeError as ae:
         print(f"Error during original generation: {ae}", file=sys.stderr)
         print("  Hint: This likely confirms attributes are missing from the model loaded by AutoModelForCausalLM.", file=sys.stderr)
    except Exception as e:
        print(f"Error during original model generation: {e}", file=sys.stderr)

    # Final Comparison
    if deconstructed_ids is not None and original_ids is not None:
        if torch.equal(deconstructed_ids, original_ids):
            print("\nSUCCESS: Generated token IDs match!")
        else:
            print("\nWARNING: Generated token IDs do NOT match.")
            if deconstructed_decoded_text == original_decoded_text:
                 print("However, the decoded text outputs are identical.")
            else:
                 print("And the decoded text outputs also differ.")
    else:
         print("\nGeneration comparison skipped as one or both failed.")
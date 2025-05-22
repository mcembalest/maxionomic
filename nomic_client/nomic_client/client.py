import os
import io
import requests
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from PIL import Image
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from urllib.parse import urlparse

from .dataset import AtlasDataset

BASE_URL = "https://api-atlas.nomic.ai/v1"
STAGING_URL = "https://staging-api-atlas.nomic.ai/v1"


class NomicClient:
    """Interact with the Nomic API and Atlas platform."""

    def __init__(
        self, api_key: Optional[str] = None, environment: Optional[str] = None
    ):
        """
        Connects to Nomic using an API key.

        Args:
            api_key: Your Nomic API key. If None, attempts to read from NOMIC_API_KEY environment variable.
            environment: Optional environment, use "staging" to connect to staging environment.
        """
        if api_key is None:
            api_key = os.getenv("NOMIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Pass it as an argument or set the NOMIC_API_KEY environment variable."
            )
        self.api_key = api_key
        if environment == "staging":
            self.base_url = STAGING_URL
        else:
            self.base_url = BASE_URL
        self.embedding_base_url = BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self._user_info = self._get("user")

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Helper method to send GET requests and handle common errors."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            response_text = (
                e.response.text
                if hasattr(e, "response") and e.response
                else "No response"
            )
            raise RuntimeError(
                f"API request to {endpoint} failed: {e}. Response: {response_text}"
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to parse API response from {endpoint}: {e}"
            ) from e

    def _post(
        self,
        endpoint: str,
        json_payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        files: Optional[List[tuple]] = None,
        data: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Helper method to send POST requests and handle common errors.
        Optionally specify a base_url and files for multipart requests.
        """
        url = f"{base_url or self.base_url}/{endpoint}"

        try:
            if files:
                files_dict = {}
                for idx, (name, content) in enumerate(files):
                    filename = f"image_{idx}.jpg"
                    content_type = "image/jpeg"
                    files_dict[name] = (filename, content, content_type)
                response = requests.post(
                    url,
                    headers=custom_headers or self.headers,
                    files=files_dict,
                    data=data,
                    timeout=timeout,
                )
            else:
                if data:
                    response = requests.post(
                        url,
                        headers=custom_headers or self.headers,
                        data=data,
                        timeout=timeout,
                    )
                elif json_payload:
                    response = requests.post(
                        url,
                        headers=custom_headers or self.headers,
                        json=json_payload,
                        timeout=timeout,
                    )
                else:
                    response = requests.post(
                        url,
                        headers=custom_headers or self.headers,
                        timeout=timeout,
                    )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            response_text = (
                e.response.text
                if hasattr(e, "response") and e.response
                else "No response"
            )
            raise RuntimeError(
                f"API request to {endpoint} failed: Response: {response_text}"
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to parse API response from {endpoint}: {e}"
            ) from e

    @property
    def user(self) -> Dict[str, Any]:
        """Get the current user's information."""

        class NomicUser(dict):
            def __repr__(self) -> str:
                name = self.get("name", "N/A")
                email = self.get("email", "N/A")
                nickname = self.get("nickname", "N/A")
                organizations = self.get("organizations", [])
                org_names = []
                for org in organizations:
                    role = org.get("access_role", "")
                    slug = org.get("slug", "N/A")
                    org_names.append(f"{slug} ({role})")
                org_display = ", ".join(org_names) if org_names else "N/A"
                lines = [
                    f"┌ Nomic User ─────────────────────────────────",
                    f"│ Name: {name}",
                    f"│ Email: {email}",
                    f"│ Nickname: {nickname}",
                    f"│ Organizations: {org_display}",
                    f"└────────────────────────────────────────────",
                ]
                return "\n".join(lines)

        return NomicUser(self._user_info)

    def embed_text(
        self,
        texts: Sequence[str],
        model: str = "nomic-embed-text-v1.5",
        task_type: str = "search_document",
        dimensionality: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Embed texts using a Nomic text embedding model."""
        payload = {
            "texts": list(texts),
            "model": model,
            "task_type": task_type,
        }
        if dimensionality is not None:
            payload["dimensionality"] = dimensionality

        return self._post("embedding/text", payload, base_url=self.embedding_base_url)

    def _is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL."""
        if not isinstance(url, str):
            return False
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme and parsed_url.netloc)
        
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize an image if it exceeds the maximum dimensions."""
        width, height = img.size
        max_width = 512
        max_height = 512
        if width > max_width or height > max_height:
            downsize_factor = max(width // max_width, height // max_height)
            img = img.resize((width // downsize_factor, height // downsize_factor))
        return img

    def embed_image(
        self,
        images: Sequence[Union[str, Image.Image]],
        model: str = "nomic-embed-vision-v1.5",
    ) -> Dict[str, Any]:
        """Embed images using a Nomic image embedding model.

        Args:
            images: List of images to embed. Can be:
                   - Local file paths (str)
                   - URLs (str)
                   - PIL Image objects
            model: The model to use for embedding

        Returns:
            Dictionary containing embeddings and metadata

        Raises:
            ValueError: If no valid images were provided
        """
        if isinstance(images, str):
            raise TypeError("'images' parameter must be a list of strings or PIL images, not str")
        
        # Determine if we're dealing with URLs only by checking the first item
        all_urls = (images and isinstance(images[0], str) and self._is_valid_url(images[0]))
        urls = []
        image_files = []
        for img_input in images:
            if all_urls:
                if isinstance(img_input, str) and self._is_valid_url(img_input):
                    urls.append(img_input)
                else:
                    raise ValueError("When using URLs, all items must be URL strings")
                continue
            try:
                if isinstance(img_input, str):
                    if os.path.exists(img_input):
                        with open(img_input, "rb") as f:
                            file_bytes = f.read()
                            image_files.append(("images", file_bytes))
                    elif self._is_valid_url(img_input):
                        raise ValueError("Cannot mix URLs and local files. Use all URLs or all local files.")
                    else:
                        raise ValueError(f"Invalid image path: {img_input}")
                elif isinstance(img_input, Image.Image):
                    img = img_input.convert("RGB")
                    img = self._resize_image(img)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    buf_value = buf.getvalue()
                    image_files.append(("images", buf_value))
                else:
                    raise ValueError(f"Not a valid image: {img_input}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process image input ({type(img_input)}): {e}"
                ) from e

        if all_urls:
            return self._post(
                "embedding/image",
                data={"model": model, "urls": urls},
                base_url=self.embedding_base_url,
            )
        else:
            if not image_files:
                raise ValueError("No valid images provided for embedding")                
            return self._post(
                "embedding/image",
                files=image_files,
                data={"model": model},
                base_url=self.embedding_base_url,
            )

    def create_dataset(
        self,
        dataset_name: str,
        is_public: bool = True,
        description: str = "",
    ) -> AtlasDataset:
        """Creates a new Atlas dataset.

        Args:
            dataset_name: The name for the new dataset.
            is_public: Whether the dataset should be publicly accessible.
            description: A description for the dataset.

        Returns:
            An AtlasDataset instance for the created dataset.

        Raises:
            RuntimeError: If dataset creation fails or no organization is available.
        """
        org_identifier = self._user_info.get("organizations")[0].get("organization_id")
        payload = {
            "organization_id": org_identifier,
            "project_name": dataset_name,
            "is_public": is_public,
            "description": description,
        }
        creation_response = self._post("project/create", payload)
        dataset_info = self._get_dataset_by_id(creation_response["project_id"])            
        return AtlasDataset._create(self, dataset_info)

    def load_dataset(self, dataset_name: str) -> AtlasDataset:
        """
        Loads an existing Atlas dataset by its name.

        Args:
            dataset_name: The name of the dataset to load.

        Returns:
            An AtlasDataset instance for the loaded dataset.

        Raises:
            ValueError: If the dataset cannot be found or accessed.
        """
        try:
            dataset_info = self._get_dataset_by_name(dataset_name)
            return AtlasDataset._create(self, dataset_info)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}': {e}") from e

    def _get_organization(self, org_id_or_slug: str) -> Dict[str, Any]:
        """(Private) Fetch details for a specific organization."""
        return self._get(f"organization/{org_id_or_slug}")

    def _get_dataset_by_id(self, dataset_id: str) -> Dict[str, Any]:
        """(Private) Fetch details for a dataset using its ID."""
        return self._get(f"dataset/{dataset_id}")
        
    def _get_dataset_by_name(self, dataset_name: str) -> Dict[str, Any]:
        """(Private) Fetch details for a dataset using org slug and dataset name."""
        org_slug = self.user.get('organizations')[0].get('slug')
        return self._get(f"dataset/{org_slug}/{dataset_name}")
    
    # This wrapper can be used for backward compatibility
    def _get_dataset(self, dataset_name_or_id: str) -> Dict[str, Any]:
        """(Private) Fetch details for a specific dataset.
        
        This is a legacy wrapper that will be removed in a future version.
        Use _get_dataset_by_id or _get_dataset_by_name directly instead.
        """
        try:
            return self._get_dataset_by_name(dataset_name_or_id)
        except Exception:
            return self._get_dataset_by_id(dataset_name_or_id)
    
    def _delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """(Private) Sends a request to delete a dataset."""
        payload = {"project_id": dataset_id}
        result = self._post("project/remove", payload)
        return result

    # def _add_arrow_data(self, dataset_id: str, arrow_data: bytes) -> Dict[str, Any]:
    #     """(Private) Uploads Arrow data to a specific dataset."""
    #     custom_headers = self.headers.copy()
    #     custom_headers["Content-Type"] = "application/octet-stream"
    #     return self._post(
    #         "project/data/add/arrow",
    #         base_url=self.base_url,
    #         data=arrow_data,
    #         timeout=300,
    #         files=None,
    #         json_payload=None,
    #         custom_headers=custom_headers,
    #     )

    # def _add_blobs(
    #     self, dataset_id: str, files: List[Tuple[str, bytes]]
    # ) -> Dict[str, Any]:
    #     """(Private) Uploads blob files (like images) to a specific dataset."""
    #     blob_files = [("blobs", blob_bytes) for _, blob_bytes in files]
    #     return self._post(
    #         "project/data/add/blobs",
    #         base_url=self.base_url,
    #         files=blob_files,
    #         data={"dataset_id": dataset_id},
    #         timeout=60,
    #     )

    # def _create_index(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    #     """(Private) Sends a request to create a new map index."""
    #     return self._post("project/index/create", payload)

    # def _get_index_job_status(self, job_id: str) -> Dict[str, Any]:
    #     """(Private) Fetches the status of an index creation job."""
    #     return self._get(f"project/index/job/{job_id}", timeout=10)

    # def _process_and_upload_images_for_dataset(
    #     self, dataset_id: str, table: pd.DataFrame, image_col_names: List[str]
    # ) -> Dict[int, str]:
    #     """(Private) Processes image sources from table columns and uploads them concurrently."""
    #     images_to_upload: List[Tuple[int, bytes]] = []
    #     total_rows = len(table)
    #     with tqdm(total=total_rows * 2, leave=False) as pbar:
    #         image_data_arrays = {
    #             name: table[name].to_pylist() for name in image_col_names
    #         }
    #         processed_indices_count = 0
    #         for i in range(total_rows):
    #             image_processed_for_row = False
    #             for img_col_name in image_col_names:
    #                 img_source = image_data_arrays[img_col_name][i]
    #                 if img_source is not None:
    #                     try:
    #                         img_bytes: Optional[bytes] = None
    #                         if isinstance(img_source, str) and os.path.exists(
    #                             img_source
    #                         ):
    #                             image = Image.open(img_source)
    #                         elif isinstance(img_source, bytes):
    #                             image = Image.open(io.BytesIO(img_source))
    #                         elif isinstance(img_source, Image.Image):
    #                             image = img_source  # Already a PIL Image
    #                         else:
    #                             raise ValueError(
    #                                 f"invalid image type at index {i}: type {type(img_source)}"
    #                             )
    #                             continue
    #                         image = image.convert("RGB")
    #                         max_dim = 512
    #                         if image.height > max_dim or image.width > max_dim:
    #                             image.thumbnail((max_dim, max_dim))
    #                         buffered = io.BytesIO()
    #                         image.save(buffered, format="JPEG", quality=90)
    #                         img_bytes = buffered.getvalue()
    #                         if img_bytes:
    #                             images_to_upload.append((i, img_bytes))
    #                             image_processed_for_row = True
    #                             break
    #                     except Exception as e:
    #                         print(
    #                             f"Warning: Failed image for row index {i} (column: {img_col_name}): {e}"
    #                         )
    #             pbar.update(1)
    #             if image_processed_for_row:
    #                 processed_indices_count += 1

    #         pbar.set_description("Image processing complete")

    #         if not images_to_upload:
    #             print("No valid images found to process and upload.")
    #             return {}

    #         print(f"Processed {processed_indices_count} images. Starting upload...")
    #         pbar.reset(total=len(images_to_upload))
    #         pbar.set_description("Uploading images")
    #         index_to_hash = self._upload_image_batches_concurrently(
    #             dataset_id=dataset_id, images_to_upload=images_to_upload, pbar=pbar
    #         )

    #     print(f"Successfully uploaded {len(index_to_hash)} images.")
    #     return index_to_hash

    # def _upload_image_batches_concurrently(
    #     self, dataset_id: str, images_to_upload: List[Tuple[int, bytes]], pbar: tqdm
    # ) -> Dict[int, str]:
    #     """(Private) Handles concurrent upload of image byte batches via _add_blobs."""
    #     batch_size = 40
    #     num_workers = 10
    #     index_to_hash_results: Dict[int, str] = {}

    #     def send_batch_request(batch: List[Tuple[int, bytes]]):
    #         indices_in_batch = [idx for idx, _ in batch]
    #         try:
    #             response_json = self._add_blobs(dataset_id, batch)
    #             returned_hashes = response_json.get("hashes", [])
    #             if len(returned_hashes) == len(indices_in_batch):
    #                 return dict(zip(indices_in_batch, returned_hashes))
    #             else:
    #                 raise ValueError(
    #                     f"Upload miscount: expected {len(indices_in_batch)}, got {len(returned_hashes)}."
    #                 )
    #         except RuntimeError as e:
    #             raise RuntimeError(f"Error uploading image batch: {e}") from e
    #             return {}

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = {}
    #         for i in range(0, len(images_to_upload), batch_size):
    #             batch = images_to_upload[i : i + batch_size]
    #             future = executor.submit(send_batch_request, batch)
    #             futures[future] = i
    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 batch_result = future.result()
    #                 if batch_result:
    #                     index_to_hash_results.update(batch_result)
    #             except Exception as e:
    #                 print(f"Error processing upload future result: {e}")
    #             finally:
    #                 if pbar:
    #                     pbar.update(1)

    #     return index_to_hash_results

    

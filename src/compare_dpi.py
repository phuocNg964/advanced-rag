import time
import os
import shutil
from unstructured.partition.pdf import partition_pdf

pdf_path = "/app/data/raw/Papers/Attention is all you need.pdf"
output_dir = "/app/data/processed/compare_dpi_100"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

dpi = 100
print(f"Extracting with DPI={dpi}...")
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    pdf_image_dpi=dpi,
    ocr_mode="individual_blocks",
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=False,
    extract_image_block_output_dir=output_dir,
)

print(f"Extraction complete. Images saved to {output_dir}")

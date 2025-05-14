import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

inter_font_path = '/Users/maxcembalest/Library/Fonts/Inter/Inter-VariableFont_opsz,wght.ttf'
crimson_pro_font_path = '/Users/maxcembalest/Library/Fonts/Crimson_Pro/CrimsonPro-VariableFont_wght.ttf'
fm.fontManager.addfont(inter_font_path)
fm.fontManager.addfont(crimson_pro_font_path)

# Extracted style settings
plt.rcParams.update({
    'font.family': 'Inter',
    'font.size': 12,  # Increased from 10 for better readability
    'figure.titlesize': 20,  # Increased from 16
    'figure.titleweight': 'bold',
    'axes.titlesize': 18,  # Increased from 14
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,  # Added explicit label size
    'figure.facecolor': '#FEFBF6',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#0D4A34',
    'axes.labelcolor': '#0D4A34',
    'xtick.color': '#0D4A34',
    'ytick.color': '#0D4A34',
    'grid.color': '#0D4A34',
    'grid.alpha': 0.1,
})

data = """Date,Provider,Pricing Model,Cluster_Size_GPUs,USD_per_Hour
2023-07-26,AWS,On-Demand (p5.48xlarge),8,98.32
2023-07-26,AWS,1-Year Reserved (p5.48xlarge),8,57.63
2023-07-26,AWS,3-Year Reserved (p5.48xlarge),8,43.16
2023-08-02,Lambda Labs,On-Demand (8xH100 SXM),8,20.72
2024-02-01,CoreWeave,On-Demand (8xHGX H100),8,27.92
2024-12-01,RunPod,Community Cloud On-Demand (H100 80GB PCIe),8,15.92
2024-06-25,RunPod,On-Demand (8xH100 80GB SXM5),8,37.52
2025-01-01,Google Cloud,On-Demand (A3-highgpu-8g),8,88.49
2025-04-25,CoreWeave,On-Demand (gd-8xh100ib-i128),8,49.24
2025-05-12,Google Cloud,1-Year Committed Use (A3-highgpu-8g),8,61.38
2025-05-12,Google Cloud,3-Year Committed Use (A3-highgpu-8g),8,38.87
"""
df = pd.read_csv(StringIO(data))
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Label'] = df['Provider'] + ' - ' + df['Pricing Model']



# Define line styles by pricing model
model_styles = {
    'On-Demand': '-',  # solid
    'On-Demand (A3-highgpu-8g)': '-',  # solid
    'Community Cloud On-Demand': '--',  # dashed
    'Secure Cloud On-Demand': '-.',  # dash-dot
    '1-Year Reserved': '--',  # dashed
    '1-Year Committed Use (CUD)': '--',  # dashed
    '3-Year Reserved': ':',  # dotted
    '3-Year Committed Use (CUD)': ':'  # dotted
}

# Define marker style by provider
provider_markers = {
    'AWS': 'o',            # Circle
    'Google Cloud': 's',   # Square
    'Lambda Labs': '^',    # Triangle
    'RunPod': 'D',         # Diamond,
    'CoreWeave': 'o'
}

# Map provider names to image filenames
provider_images = {
    'AWS': 'aws.png',
    'Google Cloud': 'gcp.png',
    'Lambda Labs': 'lambda.png',
    'RunPod': 'runpod.png',
    'CoreWeave': 'coreweave.png'
}

# Define provider-specific zoom factors (AWS needs to be smaller)
provider_zoom_factors = {
    'AWS': 0.012,
    'Google Cloud': 0.25,
    'Lambda Labs': 0.15,
    'RunPod': 0.3,
    'CoreWeave': 0.03,
}

# Function to load and create an OffsetImage with provider-specific zoom
def get_image(provider_name):
    zoom = provider_zoom_factors.get(provider_name) # Get specific zoom or default
    # Get the current directory and construct absolute path to assets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'assets')
    
    path = os.path.join(assets_dir, provider_images[provider_name])
    try:
        img = plt.imread(path)
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception as e:
        print(f"Error loading image for {provider_name} at {path}: {e}")
        color = "#04281B"
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        color_box = np.ones((10, 10, 4))
        color_box[:, :, 0] = r
        color_box[:, :, 1] = g
        color_box[:, :, 2] = b
        color_box[:, :, 3] = 1.0
        return OffsetImage(color_box, zoom=zoom)

# Create plot with larger figure size
fig, ax = plt.subplots(figsize=(14, 8))

# Sort dataframe by provider and date to ensure consistent plotting
df = df.sort_values(['Provider', 'Date'])

# Plot lines first (so they appear behind the markers)
providers = df['Provider'].unique()
for provider in providers:
    provider_data = df[df['Provider'] == provider]
    
    # Connect points for the same provider with lines
    for model in provider_data['Pricing Model'].unique():
        model_data = provider_data[provider_data['Pricing Model'] == model]
        
        # Add small circular markers just for position reference
        ax.scatter(
            model_data['Date'],
            model_data['USD_per_Hour'],
            s=10,  # Reduced from 60 for "super tiny" markers
            color="#04281B",
            marker='o',  # Simple circles for all
            zorder=10,
            edgecolor='white',
            linewidth=1,
            alpha=0.7
        )
        
# Now add logo images for each data point (with strict size controls)
runpod_logo_x_offset_count = {} # Key: original date, Value: count of logos placed for this date for RunPod
runpod_point_counter = 0 # To distinguish between the first and second RunPod points for text annotation
coreweave_reserved_annotated_count = 0 # To track the first CoreWeave Reserved point for annotation

for provider in providers:
    provider_data = df[df['Provider'] == provider]

    for _, row in provider_data.iterrows():
        original_date = row['Date']
        date_for_logo_x_position = original_date # Default to original date

        if provider == 'RunPod':
            current_count_for_date = runpod_logo_x_offset_count.get(original_date, -1) + 1
            runpod_logo_x_offset_count[original_date] = current_count_for_date
            offset_multiplier = current_count_for_date
            if offset_multiplier > 0:
                days_to_shift = 15
                date_for_logo_x_position = original_date + pd.Timedelta(days=days_to_shift * offset_multiplier)
        
        image_box = get_image(row['Provider'])
        
        annotation_box = AnnotationBbox(
            image_box,
            (date_for_logo_x_position, row['USD_per_Hour']),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            bboxprops=dict(alpha=0.0),
            zorder=11
        )
        ax.add_artist(annotation_box)

        # Add text annotations
        text_label = row['Provider'] + '\n' + row['Pricing Model'] # Use full pricing model string

        # Defaults for annotation alignment and offset
        ha_align = 'left'
        va_align = 'center'
        text_x_offset = 35  # Default horizontal offset (to the right)
        text_y_offset = 0   # Default vertical offset

        if provider == 'CoreWeave':
            if 'Reserved' in row['Pricing Model']:
                if coreweave_reserved_annotated_count == 0:
                    ha_align = 'center'
                    va_align = 'top'    # Anchor text box from its top, place below point
                    text_x_offset = 0
                    text_y_offset = -25 # Move text down (below the point)
                    coreweave_reserved_annotated_count += 1
                else: # Subsequent CoreWeave Reserved points
                    if row['Date'].year == 2025:
                        ha_align = 'right'
                        text_x_offset = -35
                    # else defaults (left, +35 for other years Reserved)
            else: # CoreWeave On-Demand
                if row['Date'].year == 2025:
                    ha_align = 'right'
                    text_x_offset = -35
                # else defaults (left, +35 for 2023/other years On-Demand)
        
        elif provider == 'Google Cloud':
            ha_align = 'right'
            text_x_offset = -35 # Points to the left
        
        elif provider == 'RunPod':
            # runpod_point_counter is incremented before this block if provider is RunPod
            if runpod_point_counter == 1: # First RunPod point
                ha_align = 'right'
                text_x_offset = -35 # Points to the left
            # Else (second RunPod point, etc.), defaults to 'left', text_x_offset = 35

        ax.annotate(text_label,
                    (date_for_logo_x_position, row['USD_per_Hour']),
                    textcoords="offset points",
                    xytext=(text_x_offset, text_y_offset), # Use y_offset
                    ha=ha_align,
                    va=va_align, # Use va_align
                    fontsize=8.5,
                    color="#04281B",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# Formatting
ax.set_title('Cost of 8xH100 GPUs', fontfamily='Crimson Pro', fontweight='bold', fontsize=24)
ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Cost (USD per Hour)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Set y-axis to start at 0
ax.set_ylim(bottom=0)

# Improve date formatting and set x-axis limits explicitly
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b\n%Y'))
# Set x-axis limit to show only the date range we have data for
ax.set_xlim(
    pd.Timestamp('2023-06-01'),  # Start a bit before first data point
    pd.Timestamp('2025-07-01')   # End a bit after last data point
)
plt.tight_layout()
plt.savefig('h100s.png')

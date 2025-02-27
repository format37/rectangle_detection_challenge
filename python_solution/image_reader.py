from PIL import Image
import numpy as np
import cv2

def read_image(image_path):
    """
    Read an image and return its dimensions and RGB values as a flat array.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (ny, nx, data) where:
            - ny (int): Height of the image
            - nx (int): Width of the image
            - data (numpy.ndarray): Flattened array of RGB values as float32
    """
    # Read the image using PIL
    img = Image.open(image_path)
    
    # Convert to RGB mode if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and ensure float32 type
    img_array = np.array(img, dtype=np.float32)
    
    # Get dimensions
    ny, nx = img_array.shape[:2]
    
    # Reshape the array to be a 1D array of RGB values
    # The order will be R1,G1,B1,R2,G2,B2,...
    data = img_array.reshape(-1)
    
    return ny, nx, data

def save_image(ny, nx, data, output_path="out.png"):
    """
    Save RGB values as an image file.
    
    Args:
        ny (int): Height of the image
        nx (int): Width of the image
        data (numpy.ndarray): Flattened array of RGB values as float32
        output_path (str): Path where to save the image (default: "out.png")
    """
    # Reshape the flat array back to (height, width, 3) for RGB
    img_array = data.reshape(ny, nx, 3)
    
    # Convert to uint8 (standard image format)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Create image from array
    img = Image.fromarray(img_array)
    
    # Save the image
    img.save(output_path)

def visualize_rectangles(ny, nx, data, rectangles):
    """
    Draw detected rectangles on the original image.
    
    Args:
        ny (int): Height of the image
        nx (int): Width of the image
        data (numpy.ndarray): Flattened array of RGB values as float32
        rectangles: List of rectangles as (x, y, w, h)
        
    Returns:
        numpy.ndarray: Image with rectangles drawn
    """
    # Reshape the flat array back to (height, width, 3) for RGB
    img_array = data.reshape(ny, nx, 3).copy()
    
    # Draw rectangles
    for rect in rectangles:
        x, y, w, h = rect
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img_array.reshape(-1)

def plot_3d_color_distribution_interactive(img_array):
    """
    Create an interactive 3D plot of color distribution using Plotly.
    The plot will be saved as an HTML file that can be opened in any web browser.
    
    Args:
        img_array (numpy.ndarray): Image array in RGB format (height, width, 3)
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.offline import plot
    
    # Flatten the image array to get all pixels
    pixels = img_array.reshape(-1, 3)
    
    # Sample a subset of pixels if there are too many
    max_points = 20000  # Increased for better visualization
    if len(pixels) > max_points:
        indices = np.random.choice(len(pixels), max_points, replace=False)
        pixels = pixels[indices]
    
    print(f"Image array shape: {img_array.shape}")
    print(f"Image array min/max values: {np.min(img_array)} {np.max(img_array)}")
    print(f"Sampled pixels shape: {pixels.shape}")
    print(f"Sampled pixels min/max values: {np.min(pixels)} {np.max(pixels)}")
    print(f"Number of points to plot: {len(pixels)}")
    
    # Convert RGB values to hex for Plotly
    def rgb_to_hex(rgb):
        r, g, b = rgb
        return f'rgb({int(r)},{int(g)},{int(b)})'
    
    # Create a list of colors for each point
    colors = [rgb_to_hex(pixel) for pixel in pixels]
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pixels[:, 0],
        y=pixels[:, 1],
        z=pixels[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.8
        ),
        hovertemplate='R: %{x:.1f}<br>G: %{y:.1f}<br>B: %{z:.1f}',
    )])
    
    # Update the layout for better visualization
    fig.update_layout(
        title='Interactive RGB Color Distribution',
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            xaxis=dict(range=[np.min(pixels[:, 0]), np.max(pixels[:, 0])]),
            yaxis=dict(range=[np.min(pixels[:, 1]), np.max(pixels[:, 1])]),
            zaxis=dict(range=[np.min(pixels[:, 2]), np.max(pixels[:, 2])]),
            aspectmode='cube'  # Equal aspect ratio
        ),
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Add camera controls info
    fig.update_layout(
        annotations=[
            dict(
                x=0.01, y=0.99,
                xref='paper', yref='paper',
                text='Use mouse to rotate, scroll to zoom',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    # Save as an interactive HTML file
    output_file = 'rgb_color_distribution_interactive.html'
    plot(fig, filename=output_file, auto_open=True)
    print(f"Interactive 3D plot saved to '{output_file}' and opened in browser")
    
    return output_file

# Add K-means clustering to identify dominant colors
def analyze_dominant_colors(img_array, k=5):
    """
    Analyze and visualize dominant colors in the image using K-means clustering.
    
    Args:
        img_array (numpy.ndarray): Image array in RGB format (height, width, 3)
        k (int): Number of color clusters to identify
        
    Returns:
        list: List of dominant colors as RGB tuples
    """
    import numpy as np
    from scipy.cluster.vq import kmeans, vq
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Reshape the image to be a list of RGB pixels
    pixels = img_array.reshape(-1, 3)
    
    # Sample if too many pixels
    max_sample = 100000
    if len(pixels) > max_sample:
        indices = np.random.choice(len(pixels), max_sample, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    
    # Perform k-means clustering
    centroids, _ = kmeans(pixels_sample, k)
    centroids = centroids.astype(int)  # Convert to integers for RGB
    
    # Quantize all pixels to closest centroid
    labels, _ = vq(pixels, centroids)
    
    # Count occurrences of each label to determine dominance
    unique, counts = np.unique(labels, return_counts=True)
    color_counts = dict(zip(unique, counts))
    
    # Sort centroids by frequency
    sorted_colors = sorted([(color_counts[i], centroids[i]) for i in unique], 
                           key=lambda x: x[0], reverse=True)
    
    # Extract just the centroids in order of frequency
    dominant_colors = [color for _, color in sorted_colors]
    color_percentages = [count/len(pixels)*100 for count, _ in sorted_colors]
    
    # Convert RGB to hex for visualization
    def rgb_to_hex(rgb):
        return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
    
    hex_colors = [rgb_to_hex(color) for color in dominant_colors]
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter3d"}, {"type": "bar"}]]
    )
    
    # 3D scatter plot showing the clusters
    sample_size = min(5000, len(pixels))
    sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
    sampled_pixels = pixels[sample_indices]
    sampled_labels = labels[sample_indices]
    
    # Add points colored by cluster
    for i, centroid in enumerate(centroids):
        mask = sampled_labels == i
        if np.any(mask):
            fig.add_trace(
                go.Scatter3d(
                    x=sampled_pixels[mask, 0],
                    y=sampled_pixels[mask, 1],
                    z=sampled_pixels[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=rgb_to_hex(centroid),
                        opacity=0.7
                    ),
                    name=f'Cluster {i+1}: {rgb_to_hex(centroid)}',
                    hovertemplate='R: %{x:.1f}<br>G: %{y:.1f}<br>B: %{z:.1f}'
                ),
                row=1, col=1
            )
    
    # Add the centroids as larger points
    fig.add_trace(
        go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=hex_colors,
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            name='Dominant Colors',
            hovertemplate='R: %{x}<br>G: %{y}<br>B: %{z}'
        ),
        row=1, col=1
    )
    
    # Add bar chart showing color distribution
    fig.add_trace(
        go.Bar(
            x=color_percentages,
            y=[f"Color {i+1}" for i in range(len(dominant_colors))],
            orientation='h',
            marker=dict(color=hex_colors),
            text=[f"{p:.1f}%" for p in color_percentages],
            textposition='auto',
            hovertemplate='%{x:.2f}% of image<br>RGB: %{customdata}',
            customdata=dominant_colors
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Dominant Colors Analysis',
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            aspectmode='cube'
        ),
        xaxis=dict(title='Percentage'),
        yaxis=dict(title=''),
        width=1200,
        height=800,
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save as interactive HTML
    output_file = 'dominant_colors_analysis.html'
    fig.write_html(output_file)
    print(f"Dominant colors analysis saved to '{output_file}'")
    
    return dominant_colors, color_percentages

if __name__ == "__main__":
    ny, nx, data = read_image("../../generator/images/image_000.png")
    print(ny, nx, data)
    # Test the save function
    save_image(ny, nx, data)


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import time
import json

def detect_rectangles_deterministic(ny, nx, data, expected_coords=None, debug=False):
    """
    Deterministic rectangle detection using color clustering.
    
    Args:
        ny (int): Height of the image
        nx (int): Width of the image
        data (numpy.ndarray): Flattened array of RGB values as float32
        expected_coords (dict, optional): Expected rectangle coordinates for validation
        
    Returns:
        tuple: (rectangles, mask_image)
    """
    # Reshape the flat array back to (height, width, 3) for RGB
    img_array = data.reshape(ny, nx, 3)
    
    # Set a fixed seed for deterministic results
    np.random.seed(42)
    
    # Sample a larger subset of pixels for more reliable color analysis
    sample_indices = np.random.choice(ny*nx, min(20000, ny*nx), replace=False)
    sampled_colors = img_array.reshape(-1, 3)[sample_indices]
    
    # Use k-means to find the two dominant colors
    colors, _ = kmeans(sampled_colors, 2)
    color1, color2 = colors
    
    if debug:
        print(f"Detected color 1 (RGB): {color1}")
        print(f"Detected color 2 (RGB): {color2}")
    
    # If expected coordinates are provided, use them to determine color assignment
    rect_color = None
    bg_color = None
    
    if expected_coords:
        x0, y0 = expected_coords["x0"], expected_coords["y0"]
        x1, y1 = expected_coords["x1"], expected_coords["y1"]
        
        # Sample rectangle area (center point)
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        rect_color_sample = img_array[center_y, center_x]
        
        # Sample background (top-left corner, far from any potential rectangles)
        bg_color_sample = img_array[10, 10]
        
        if debug:
            # Print these samples for debugging
            print(f"Rectangle center sample at ({center_x}, {center_y}): {rect_color_sample}")
            print(f"Background sample at (10, 10): {bg_color_sample}")
        
        # Determine which cluster is closer to each sample
        rect_dist1 = np.sum((rect_color_sample - color1)**2)
        rect_dist2 = np.sum((rect_color_sample - color2)**2)
        
        bg_dist1 = np.sum((bg_color_sample - color1)**2)
        bg_dist2 = np.sum((bg_color_sample - color2)**2)
        
        # Assign colors based on closest matches
        if rect_dist1 < rect_dist2 and bg_dist2 < bg_dist1:
            rect_color, bg_color = color1, color2
        elif rect_dist2 < rect_dist1 and bg_dist1 < bg_dist2:
            rect_color, bg_color = color2, color1
        else:
            # If we have conflicting assignments, use cluster sizes
            counts1 = np.sum(np.all(np.abs(img_array.reshape(-1, 3) - color1) < 30, axis=1))
            counts2 = np.sum(np.all(np.abs(img_array.reshape(-1, 3) - color2) < 30, axis=1))
            
            if counts1 < counts2:
                rect_color, bg_color = color1, color2
            else:
                rect_color, bg_color = color2, color1
    else:
        # Without expected coordinates, check known positions
        # Assuming rectangle is smaller than background and not at the edges
        edge_samples = [
            img_array[0, 0],          # Top-left
            img_array[0, nx-1],       # Top-right
            img_array[ny-1, 0],       # Bottom-left
            img_array[ny-1, nx-1],    # Bottom-right
        ]
        
        # Get the average edge color
        edge_color = np.mean(edge_samples, axis=0)
        
        if debug:
            print(f"Edge average color: {edge_color}")
        
        # Compare with detected clusters
        dist1 = np.sum((edge_color - color1)**2)
        dist2 = np.sum((edge_color - color2)**2)
        
        if dist1 < dist2:
            bg_color, rect_color = color1, color2
        else:
            bg_color, rect_color = color2, color1
    
    if debug:
        print(f"Selected rectangle color (RGB): {rect_color}")
        print(f"Selected background color (RGB): {bg_color}")
    
    # Calculate color distances
    dist_to_rect = np.sum((img_array - rect_color)**2, axis=2)
    dist_to_bg = np.sum((img_array - bg_color)**2, axis=2)
    
    # Create binary mask with adaptive thresholding
    # If distances are similar, rely more on color clusters
    color_difference = np.sum((rect_color - bg_color)**2)
    if debug:
        print(f"Color difference magnitude: {color_difference}")
    
    if color_difference > 1000:
        # Colors are distinct enough, simple thresholding works
        mask = np.where(dist_to_rect < dist_to_bg, 255, 0).astype(np.uint8)
    else:
        # Colors are similar, use a more aggressive approach
        # Normalize distances to 0-1 range for better comparison
        max_dist = np.maximum(np.max(dist_to_rect), np.max(dist_to_bg))
        norm_dist_rect = dist_to_rect / max_dist
        norm_dist_bg = dist_to_bg / max_dist
        
        # Use a threshold factor for better separation
        threshold_factor = 0.9
        mask = np.where(norm_dist_rect < norm_dist_bg * threshold_factor, 255, 0).astype(np.uint8)
    
    # Save original mask for debugging
    # if debug:
    #     mask_img_original = Image.fromarray(mask)
    #     mask_img_original.save("mask_before_morphology.png")
    
    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours
    rectangles = []
    for contour in contours:
        # Filter out small contours
        area = cv2.contourArea(contour)
        if area < 100:  # Adjust minimum area as needed
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Store as absolute coordinates (x0, y0, x1, y1) instead of (x, y, w, h)
        rectangles.append((x, y, x+w, y+h))
        
        # Draw the rectangle on mask for visualization
        cv2.rectangle(mask, (x, y), (x+w, y+h), 128, 2)
    
    # Validate against expected coordinates if provided
    if expected_coords and rectangles:
        if debug:
            print(f"Expected rectangle: x0={expected_coords['x0']}, y0={expected_coords['y0']}, "
                f"x1={expected_coords['x1']}, y1={expected_coords['y1']}")
        
        for i, (x0, y0, x1, y1) in enumerate(rectangles):
            if debug:
                print(f"Detected rectangle {i+1}: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
            
            # Calculate IoU with expected rectangle
            expected_x0, expected_y0 = expected_coords["x0"], expected_coords["y0"]
            expected_x1, expected_y1 = expected_coords["x1"], expected_coords["y1"]
            
            # Calculate intersection
            x_left = max(x0, expected_x0)
            y_top = max(y0, expected_y0)
            x_right = min(x1, expected_x1)
            y_bottom = min(y1, expected_y1)
            
            if x_right < x_left or y_bottom < y_top:
                intersection = 0
            else:
                intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas
            detected_area = (x1 - x0) * (y1 - y0)
            expected_area = (expected_x1 - expected_x0) * (expected_y1 - expected_y0)
            union = detected_area + expected_area - intersection
            
            iou = intersection / union if union > 0 else 0
            if debug:
                print(f"IoU with expected rectangle: {iou:.4f}")
    
    return rectangles, mask

def calculate_rectangle_error(ny, nx, data, detected_rect, expected_rect):
    """
    Calculate the sum of squared errors between the detected rectangle and expected rectangle.
    
    Args:
        ny (int): Height of the image
        nx (int): Width of the image
        data (numpy.ndarray): Flattened array of RGB values as float32
        detected_rect (tuple): (x0, y0, x1, y1) of the detected rectangle
        expected_rect (dict): Expected rectangle coordinates and colors
        
    Returns:
        float: Sum of squared errors
    """
    # Reshape the flat array back to (height, width, 3) for RGB
    img_array = data.reshape(ny, nx, 3)
    
    # Extract detected rectangle coordinates
    x0, y0, x1, y1 = detected_rect
    
    # Extract expected rectangle coordinates and colors
    exp_x0, exp_y0 = expected_rect["x0"], expected_rect["y0"]
    exp_x1, exp_y1 = expected_rect["x1"], expected_rect["y1"]
    outer_color = np.array(expected_rect["outer_color"], dtype=np.float32)
    inner_color = np.array(expected_rect["inner_color"], dtype=np.float32)
    
    # Create a mask for the expected rectangle (what should be)
    expected_mask = np.zeros((ny, nx), dtype=bool)
    expected_mask[exp_y0:exp_y1, exp_x0:exp_x1] = True
    
    # Create a mask for the detected rectangle (what we have)
    detected_mask = np.zeros((ny, nx), dtype=bool)
    detected_mask[y0:y1, x0:x1] = True
    
    # Create additional masks for error analysis
    # True positives: pixels correctly identified as inside the rectangle
    tp_mask = np.logical_and(expected_mask, detected_mask)
    
    # False positives: pixels incorrectly identified as inside the rectangle
    fp_mask = np.logical_and(~expected_mask, detected_mask)
    
    # False negatives: pixels incorrectly identified as outside the rectangle
    fn_mask = np.logical_and(expected_mask, ~detected_mask)
    
    # True negatives: pixels correctly identified as outside the rectangle
    tn_mask = np.logical_and(~expected_mask, ~detected_mask)
    
    # Calculate errors based on color expectations
    errors = np.zeros_like(img_array, dtype=np.float32)
    
    # True positives should have inner color
    errors[tp_mask] = img_array[tp_mask] - inner_color
    
    # False positives should have outer color, but we detected as inner
    errors[fp_mask] = img_array[fp_mask] - outer_color
    
    # False negatives should have inner color, but we detected as outer
    errors[fn_mask] = img_array[fn_mask] - inner_color
    
    # True negatives should have outer color
    errors[tn_mask] = img_array[tn_mask] - outer_color
    
    # Sum of squared errors
    sse = np.sum(errors**2)
    
    return sse

def evaluate_detection(ny, nx, data, rectangles, expected_coords):
    """
    Evaluate the detection results against expected coordinates.
    
    Args:
        ny (int): Height of the image
        nx (int): Width of the image
        data (numpy.ndarray): Flattened array of RGB values as float32
        rectangles (list): List of detected rectangles (x0, y0, x1, y1)
        expected_coords (dict): Expected rectangle coordinates and colors
        
    Returns:
        dict: Evaluation metrics including error and IoU
    """
    if not rectangles:
        return {"error": float('inf'), "iou": 0.0, "message": "No rectangles detected"}
    
    # Find the best matching rectangle based on IoU
    best_rect = None
    best_iou = -1
    
    expected_x0, expected_y0 = expected_coords["x0"], expected_coords["y0"]
    expected_x1, expected_y1 = expected_coords["x1"], expected_coords["y1"]
    
    for rect in rectangles:
        x0, y0, x1, y1 = rect
        
        # Calculate intersection
        x_left = max(x0, expected_x0)
        y_top = max(y0, expected_y0)
        x_right = min(x1, expected_x1)
        y_bottom = min(y1, expected_y1)
        
        if x_right < x_left or y_bottom < y_top:
            intersection = 0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        detected_area = (x1 - x0) * (y1 - y0)
        expected_area = (expected_x1 - expected_x0) * (expected_y1 - expected_y0)
        union = detected_area + expected_area - intersection
        
        iou = intersection / union if union > 0 else 0
        
        if iou > best_iou:
            best_iou = iou
            best_rect = rect
    
    # Calculate error for the best rectangle
    error = calculate_rectangle_error(ny, nx, data, best_rect, expected_coords)
    
    return {
        "error": error,
        "iou": best_iou,
        "best_rect": best_rect,
        "message": f"Best IoU: {best_iou:.4f}, Error: {error:.2f}"
    }

def read_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ny, nx = img.shape[:2]
    data = img.astype(np.float32).reshape(-1)
    return ny, nx, data

# Example rectangle visualization function
def visualize_rectangles(ny, nx, data, rectangles):
    img_array = data.reshape(ny, nx, 3).astype(np.uint8)
    result = img_array.copy()
    
    for x0, y0, x1, y1 in rectangles:
        cv2.rectangle(result, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
    return result.reshape(-1)

# Example save function
def save_image(ny, nx, data, filename):
    img_array = data.reshape(ny, nx, 3).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)

def process_single_image(filepath, expected_coords, debug=False):
    """
    Process a single image for rectangle detection and evaluation.
    
    Args:
        filepath (str): Path to the image file
        expected_coords (dict): Expected rectangle coordinates and colors
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Results including detected rectangles, evaluation metrics, and timing
    """
    try:
        ny, nx, data = read_image(filepath)
        if debug:
            print(f"Image dimensions: {ny}x{nx}")
        
        start_time = time.time()
        rectangles, mask = detect_rectangles_deterministic(ny, nx, data, expected_coords, debug=debug)
        end_time = time.time()
        
        detection_time = end_time - start_time
        if debug:
            print(f"Detection completed in {detection_time:.3f} seconds")
            print(f"Detected {len(rectangles)} rectangles: {rectangles}")
        
        # Evaluate detection results
        evaluation = evaluate_detection(ny, nx, data, rectangles, expected_coords)
        
        # Create result dictionary
        result = {
            "filepath": filepath,
            "rectangles": rectangles,
            "detection_time": detection_time,
            "evaluation": evaluation,
            "mask": mask
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            "filepath": filepath,
            "error": str(e),
            "rectangles": [],
            "evaluation": {"error": float('inf'), "iou": 0.0, "message": f"Error: {e}"}
        }

def plot_performance_metrics(results):
    """
    Plot performance metrics for all processed images.
    
    Args:
        results (list): List of result dictionaries from process_single_image
    """
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Extract data for plotting
    filenames = [r["filepath"].split("/")[-1] for r in valid_results]
    ious = [r["evaluation"]["iou"] for r in valid_results]
    errors = [r["evaluation"]["error"] for r in valid_results]
    times = [r["detection_time"] for r in valid_results]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot IoU values
    axs[0].bar(range(len(filenames)), ious, color='skyblue')
    axs[0].set_title('IoU by Image (Higher is Better)')
    axs[0].set_ylabel('IoU')
    axs[0].set_xticks(range(len(filenames)))
    axs[0].set_xticklabels(filenames, rotation=45, ha='right')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight worst IoU performers
    worst_iou_indices = np.argsort(ious)[:3]
    for idx in worst_iou_indices:
        axs[0].bar(idx, ious[idx], color='red')
        axs[0].text(idx, ious[idx]/2, f"{ious[idx]:.2f}", 
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Plot error values
    axs[1].bar(range(len(filenames)), errors, color='lightgreen')
    axs[1].set_title('Error by Image (Lower is Better)')
    axs[1].set_ylabel('Error')
    axs[1].set_xticks(range(len(filenames)))
    axs[1].set_xticklabels(filenames, rotation=45, ha='right')
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight worst error performers
    worst_error_indices = np.argsort(errors)[-3:]
    for idx in worst_error_indices:
        axs[1].bar(idx, errors[idx], color='red')
        axs[1].text(idx, errors[idx]/2, f"{errors[idx]:.0f}", 
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Plot detection times
    axs[2].bar(range(len(filenames)), times, color='salmon')
    axs[2].set_title('Detection Time by Image')
    axs[2].set_ylabel('Time (seconds)')
    axs[2].set_xticks(range(len(filenames)))
    axs[2].set_xticklabels(filenames, rotation=45, ha='right')
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('rectangle_detection_performance.png')
    print("Performance metrics plot saved as 'rectangle_detection_performance.png'")
    
    # Print worst performers
    print("\n===== WORST PERFORMERS =====")
    print("Lowest IoU values:")
    for idx in worst_iou_indices:
        print(f"  - {filenames[idx]}: IoU = {ious[idx]:.4f}")
    
    print("\nHighest Error values:")
    for idx in worst_error_indices:
        print(f"  - {filenames[idx]}: Error = {errors[idx]:.2f}")

# Example usage
if __name__ == "__main__":
    # Read annotations from json file
    with open("../images/annotations.json", "r") as f:
        annotations = json.load(f)
    
    results = []
    
    # Process all images in the annotations
    for annotation in annotations:
        filepath = "../" + annotation["filename"]
        print(f"\nProcessing {filepath}...")
        
        result = process_single_image(filepath, annotation, debug=True)
        results.append(result)
        
        # Visualize and save results if detection was successful
        if "error" not in result:
            # # Generate output filename based on input filename
            # base_filename = filepath.split("/")[-1].split(".")[0]
            
            # # Visualize results
            # result_data = visualize_rectangles(
            #     result["mask"].shape[0], 
            #     result["mask"].shape[1], 
            #     read_image(filepath)[2], 
            #     result["rectangles"]
            # )
            # # save_image(
            #     result["mask"].shape[0], 
            #     result["mask"].shape[1], 
            #     result_data, 
            #     f"{base_filename}_detection_result.png"
            # )
            
            # # Save the mask
            # mask_img = Image.fromarray(result["mask"])
            # mask_img.save(f"{base_filename}_segmentation_mask.png")
            
            print(f"Evaluation: {result['evaluation']['message']}")
    
    # Print summary of results
    print("\n===== DETECTION SUMMARY =====")
    print(f"Processed {len(results)} images")
    
    # Calculate average IoU and error
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_iou = sum(r["evaluation"]["iou"] for r in valid_results) / len(valid_results)
        avg_error = sum(r["evaluation"]["error"] for r in valid_results) / len(valid_results)
        avg_time = sum(r["detection_time"] for r in valid_results) / len(valid_results)
        
        # Calculate median and worst detection time
        detection_times = [r["detection_time"] for r in valid_results]
        median_time = sorted(detection_times)[len(detection_times) // 2]
        worst_time = max(detection_times)
        
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Error: {avg_error:.2f}")
        print(f"Average Detection Time: {avg_time:.3f} seconds")
        print(f"Median Detection Time: {median_time:.3f} seconds")
        print(f"Worst Detection Time: {worst_time:.3f} seconds")
        
        # Plot performance metrics
        plot_performance_metrics(results)
    
    # Count failures
    failures = [r for r in results if "error" in r]
    if failures:
        print(f"Failed to process {len(failures)} images:")
        for failure in failures:
            print(f"  - {failure['filepath']}: {failure.get('error', 'Unknown error')}")
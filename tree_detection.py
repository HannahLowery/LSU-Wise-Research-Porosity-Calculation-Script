import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_porosity_from_silhouette(image_path, show_steps=True, manual_boxes=None, min_tree_area=100):
    """
    Calculate tree porosity from a black and white silhouette image.
    
    Parameters:
    -----------
    image_path : str
        Path to the black and white silhouette image
    show_steps : bool
        Whether to display visualization
    manual_boxes : list of tuples, optional
        List of manually defined boxes as [(x, y, width, height), ...]
        Example: [(100, 50, 200, 300), (400, 60, 180, 280)]
    min_tree_area : int
        Minimum area (in pixels) to consider as a tree (filters noise)
    
    Returns:
    --------
    dict : Analysis results including porosity for each tree
    """
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'")
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    print(f"Image loaded! Size: {gray.shape[1]}x{gray.shape[0]} pixels")
    
    height, width = gray.shape
    
    # Threshold to ensure pure black and white
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check if we should use manual boxes
    if manual_boxes is not None:
        print(f"\nUsing {len(manual_boxes)} manually defined box(es)")
        tree_boxes = manual_boxes
    else:
        # Automatic detection: Find separate tree regions
        print("\nAutomatic detection mode...")
        
        # Invert for contour detection (white objects on black background)
        inverted = cv2.bitwise_not(binary)
        
        # Find connected components (separate trees)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("Error: No black pixels found!")
            return None
        
        print(f"Found {len(contours)} connected region(s)")
        
        # Get bounding box for each contour
        tree_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter out very small regions (noise)
            if area >= min_tree_area:
                tree_boxes.append((x, y, w, h))
                print(f"  - Tree at ({x}, {y}) with size {w}x{h} (area: {area:,} px)")
            else:
                print(f"  - Skipped small region at ({x}, {y}) with area {area} px")
        
        if len(tree_boxes) == 0:
            print("Error: No significant trees found!")
            print(f"Try reducing min_tree_area (currently {min_tree_area})")
            return None
    
    # Analyze each tree
    tree_results = []
    img_with_boxes = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    colored_analysis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for idx, (x, y, w, h) in enumerate(tree_boxes):
        # Extract bounding box region
        bbox_region = binary[y:y+h, x:x+w]
        
        # Calculate porosity
        total_area = w * h
        black_pixels = np.sum(bbox_region == 0)
        white_pixels = np.sum(bbox_region == 255)
        porosity = white_pixels / total_area if total_area > 0 else 0
        density = black_pixels / total_area if total_area > 0 else 0
        
        # Choose color for this tree
        color = colors[idx % len(colors)]
        
        # Draw bounding box on visualization
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_with_boxes, f'Tree {idx+1}', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Color the analysis image
        cv2.rectangle(colored_analysis, (x, y), (x+w, y+h), color, 2)
        # Mark gaps in red within this box
        gap_mask = bbox_region == 255
        colored_analysis[y:y+h, x:x+w][gap_mask] = [0, 0, 255]  # Red for gaps
        
        tree_results.append({
            'id': idx + 1,
            'bbox': (x, y, w, h),
            'area': total_area,
            'black_pixels': black_pixels,
            'white_pixels': white_pixels,
            'porosity': porosity,
            'density': density
        })
    
    # Calculate overall statistics
    total_area = sum(t['area'] for t in tree_results)
    total_black = sum(t['black_pixels'] for t in tree_results)
    total_white = sum(t['white_pixels'] for t in tree_results)
    overall_porosity = total_white / total_area if total_area > 0 else 0
    overall_density = total_black / total_area if total_area > 0 else 0
    
    if show_steps:
        # Calculate grid layout
        num_trees = len(tree_results)
        num_cols = min(4, max(3, num_trees + 1))
        num_rows = max(2, (num_trees + 4) // num_cols + 1)
        
        fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
        
        # Original image
        ax1 = plt.subplot(num_rows, num_cols, 1)
        ax1.imshow(binary, cmap='gray')
        ax1.set_title('Input Silhouette\nBlack=Tree, White=Gaps', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # With bounding boxes
        ax2 = plt.subplot(num_rows, num_cols, 2)
        ax2.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detected {len(tree_results)} Tree(s)\nwith Bounding Boxes', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Gap analysis
        ax3 = plt.subplot(num_rows, num_cols, 3)
        ax3.imshow(cv2.cvtColor(colored_analysis, cv2.COLOR_BGR2RGB))
        ax3.set_title('Gap Analysis\nRed=Wind passes through', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Porosity breakdown bar chart
        ax4 = plt.subplot(num_rows, num_cols, 4)
        categories = ['Solid\n(Blocks Wind)', 'Gaps\n(Wind Passes)']
        values = [overall_density * 100, overall_porosity * 100]
        bar_colors = ['black', 'red']
        bars = ax4.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Percentage (%)', fontsize=11)
        ax4.set_title('Overall Porosity Breakdown', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Individual tree crops
        for i, tree in enumerate(tree_results):
            x, y, w, h = tree['bbox']
            tree_crop = binary[y:y+h, x:x+w]
            
            ax = plt.subplot(num_rows, num_cols, 5 + i)
            ax.imshow(tree_crop, cmap='gray')
            ax.set_title(f'Tree {i+1}\nPorosity: {tree["porosity"]:.1%}\nSize: {w}x{h}px', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Statistics text
        stats_position = 5 + len(tree_results)
        ax_stats = plt.subplot(num_rows, num_cols, stats_position)
        ax_stats.axis('off')
        
        stats_text = f"""POROSITY ANALYSIS

Trees Detected: {len(tree_results)}

OVERALL STATISTICS:
  Total Area: {total_area:,} px
  Solid: {total_black:,} px ({overall_density:.1%})
  Gaps:  {total_white:,} px ({overall_porosity:.1%})
  
  Overall Porosity: {overall_porosity:.1%}

INDIVIDUAL TREES:
"""
        for tree in tree_results:
            stats_text += f"\n  Tree {tree['id']}: {tree['porosity']:.1%}"
            stats_text += f" ({tree['bbox'][2]}×{tree['bbox'][3]}px)"
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center')
        ax_stats.set_title('Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Save results
    cv2.imwrite('detected_boxes.png', img_with_boxes)
    cv2.imwrite('gap_analysis.png', colored_analysis)
    
    print(f"\n{'='*70}")
    print(f"TREE POROSITY ANALYSIS")
    print(f"{'='*70}")
    print(f"\nNumber of trees: {len(tree_results)}")
    print(f"\nOVERALL RESULTS:")
    print(f"  Total area:     {total_area:,} pixels")
    print(f"  Solid (black):  {total_black:,} pixels ({overall_density:.1%})")
    print(f"  Gaps (white):   {total_white:,} pixels ({overall_porosity:.1%})")
    print(f"\n  OVERALL POROSITY: {overall_porosity:.2%}")
    print(f"  OVERALL DENSITY:  {overall_density:.2%}")
    print(f"\nINDIVIDUAL TREE RESULTS:")
    for tree in tree_results:
        print(f"  Tree {tree['id']}: {tree['porosity']:.2%} porosity")
        print(f"    Box: ({tree['bbox'][0]}, {tree['bbox'][1]}) size {tree['bbox'][2]}×{tree['bbox'][3]}px")
    print(f"{'='*70}")
    print("\nFiles saved:")
    print("  - detected_boxes.png")
    print("  - gap_analysis.png")
    
    return {
        'overall_porosity': overall_porosity,
        'overall_density': overall_density,
        'num_trees': len(tree_results),
        'trees': tree_results,
        'visualization': img_with_boxes,
        'gap_analysis': colored_analysis
    }


def draw_manual_boxes(image_path):
    """
    Interactive tool to manually draw bounding boxes on an image.
    
    Instructions:
    - Click and drag to draw a box
    - Press 'r' to reset and start over
    - Press 'q' or ESC to finish and return boxes
    
    Returns:
    --------
    list : List of boxes as [(x, y, width, height), ...]
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'")
        return None
    
    # Convert to grayscale for display
    if len(img.shape) == 3:
        display_img = img.copy()
    else:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    boxes = []
    drawing = False
    start_point = None
    current_rect = None
    
    def draw_all_boxes(img):
        """Draw all completed boxes and current box"""
        temp_img = img.copy()
        
        # Draw completed boxes in green
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(temp_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(temp_img, f'Tree {idx+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current box being drawn in yellow
        if drawing and start_point and current_rect:
            x, y, w, h = current_rect
            cv2.rectangle(temp_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        return temp_img
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_rect
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_rect = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                x1, y1 = start_point
                w = abs(x - x1)
                h = abs(y - y1)
                x_min = min(x, x1)
                y_min = min(y, y1)
                current_rect = (x_min, y_min, w, h)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing and start_point:
                x1, y1 = start_point
                w = abs(x - x1)
                h = abs(y - y1)
                x_min = min(x, x1)
                y_min = min(y, y1)
                
                if w > 10 and h > 10:  # Minimum size
                    boxes.append((x_min, y_min, w, h))
                    print(f"Box {len(boxes)} added: ({x_min}, {y_min}) size {w}×{h}px")
                
                drawing = False
                start_point = None
                current_rect = None
    
    # Create window and set mouse callback
    window_name = 'Draw Bounding Boxes (drag to draw, r=reset, q=quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "="*70)
    print("MANUAL BOX DRAWING MODE")
    print("="*70)
    print("\nInstructions:")
    print("  1. Click and drag to draw a box around each tree")
    print("  2. Press 'r' to reset and start over")
    print("  3. Press 'q' or ESC when finished")
    print("="*70 + "\n")
    
    while True:
        # Draw image with boxes
        display = draw_all_boxes(display_img)
        
        # Add instructions on image
        cv2.putText(display, f'Boxes: {len(boxes)} | r=reset | q=quit', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Reset boxes
        if key == ord('r'):
            boxes = []
            print("Reset - all boxes cleared")
        
        # Quit
        elif key == ord('q') or key == 27:  # 'q' or ESC
            break
    
    cv2.destroyAllWindows()
    
    if len(boxes) == 0:
        print("\nNo boxes drawn!")
        return None
    
    print(f"\n✓ Finished! Drew {len(boxes)} box(es)")
    for idx, (x, y, w, h) in enumerate(boxes):
        print(f"  Box {idx+1}: ({x}, {y}) size {w}×{h}px")
    
    return boxes


def draw_manual_lines(image_path):
    """
    Interactive tool to manually draw 4 lines (top, bottom, left, right) to define bounding boxes.
    
    Instructions:
    - Click to place the first point of a line
    - Click again to place the second point
    - Draw 4 lines in order: TOP, BOTTOM, LEFT, RIGHT
    - After 4 lines, the box is complete and you can start the next tree
    - Press 'u' to undo last line
    - Press 'r' to reset all
    - Press 'q' or ESC to finish
    
    Returns:
    --------
    list : List of boxes as [(x, y, width, height), ...]
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'")
        return None
    
    # Convert to display format
    if len(img.shape) == 3:
        display_img = img.copy()
    else:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    height, width = display_img.shape[:2]
    
    boxes = []
    current_lines = []  # Store lines for current tree: [top, bottom, left, right]
    current_point = None  # First point of line being drawn
    temp_point = None  # Mouse position for preview
    
    line_names = ["TOP", "BOTTOM", "LEFT", "RIGHT"]
    line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow
    
    def get_current_line_name():
        if len(current_lines) < 4:
            return line_names[len(current_lines)]
        return "COMPLETE"
    
    def draw_visualization(img):
        """Draw all completed boxes, current lines, and preview"""
        temp_img = img.copy()
        
        # Draw completed boxes
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(temp_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(temp_img, f'Tree {idx+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current lines
        for idx, line in enumerate(current_lines):
            p1, p2 = line
            color = line_colors[idx]
            cv2.line(temp_img, p1, p2, color, 2)
            # Label the line
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(temp_img, line_names[idx], (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw preview of current bounding box if we have all 4 lines
        if len(current_lines) == 4:
            top_line, bottom_line, left_line, right_line = current_lines
            
            # Get y coordinates from horizontal lines
            y_top = min(top_line[0][1], top_line[1][1])
            y_bottom = max(bottom_line[0][1], bottom_line[1][1])
            
            # Get x coordinates from vertical lines
            x_left = min(left_line[0][0], left_line[1][0])
            x_right = max(right_line[0][0], right_line[1][0])
            
            # Draw preview box
            cv2.rectangle(temp_img, (x_left, y_top), (x_right, y_bottom), (255, 0, 255), 2)
            cv2.putText(temp_img, 'PREVIEW (click to confirm)', (x_left, y_top-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw line being drawn
        if current_point and temp_point:
            color = line_colors[len(current_lines)] if len(current_lines) < 4 else (128, 128, 128)
            cv2.line(temp_img, current_point, temp_point, color, 2)
            cv2.circle(temp_img, current_point, 5, color, -1)
        
        # Draw point marker if we have first point
        if current_point and not temp_point:
            color = line_colors[len(current_lines)] if len(current_lines) < 4 else (128, 128, 128)
            cv2.circle(temp_img, current_point, 5, color, -1)
        
        return temp_img
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point, temp_point
        
        if event == cv2.EVENT_MOUSEMOVE:
            if current_point:
                temp_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(current_lines) >= 4:
                # We have 4 lines, clicking confirms and creates the box
                top_line, bottom_line, left_line, right_line = current_lines
                
                # Calculate box from lines
                y_top = min(top_line[0][1], top_line[1][1])
                y_bottom = max(bottom_line[0][1], bottom_line[1][1])
                x_left = min(left_line[0][0], left_line[1][0])
                x_right = max(right_line[0][0], right_line[1][0])
                
                box_width = x_right - x_left
                box_height = y_bottom - y_top
                
                if box_width > 10 and box_height > 10:
                    boxes.append((x_left, y_top, box_width, box_height))
                    print(f"\n✓ Tree {len(boxes)} completed: ({x_left}, {y_top}) size {box_width}×{box_height}px")
                    current_lines.clear()
                    current_point = None
                    temp_point = None
                else:
                    print("Box too small, try again")
                    current_lines.clear()
                    current_point = None
                    temp_point = None
            
            elif current_point is None:
                # Start new line
                current_point = (x, y)
                print(f"Drawing {get_current_line_name()} line - click second point")
            else:
                # Finish current line
                p2 = (x, y)
                current_lines.append((current_point, p2))
                print(f"✓ {line_names[len(current_lines)-1]} line added")
                
                current_point = None
                temp_point = None
                
                if len(current_lines) == 4:
                    print("\n4 lines complete! Click anywhere to confirm this tree box (or press 'u' to undo)")
                else:
                    print(f"Now draw {get_current_line_name()} line")
    
    # Create window
    window_name = 'Draw 4 Lines: TOP, BOTTOM, LEFT, RIGHT (u=undo, r=reset, q=quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "="*70)
    print("MANUAL LINE DRAWING MODE")
    print("="*70)
    print("\nInstructions:")
    print("  1. For each tree, draw 4 lines in order:")
    print("     - TOP line (horizontal)")
    print("     - BOTTOM line (horizontal)")
    print("     - LEFT line (vertical)")
    print("     - RIGHT line (vertical)")
    print("  2. Click two points to define each line")
    print("  3. After 4 lines, click to confirm the box")
    print("  4. Press 'u' to undo last line")
    print("  5. Press 'r' to reset all")
    print("  6. Press 'q' or ESC when finished")
    print("="*70)
    print(f"\nStart by drawing {get_current_line_name()} line\n")
    
    while True:
        display = draw_visualization(display_img)
        
        # Status text
        status = f'Trees: {len(boxes)} | Current: {get_current_line_name()} ({len(current_lines)}/4) | u=undo | r=reset | q=quit'
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Undo last line
        if key == ord('u'):
            if current_point:
                # Cancel current line being drawn
                current_point = None
                temp_point = None
                print("Cancelled current line")
            elif current_lines:
                # Remove last complete line
                removed_line = current_lines.pop()
                print(f"Undid {line_names[len(current_lines)]} line")
                if len(current_lines) < 4:
                    print(f"Now draw {get_current_line_name()} line")
            else:
                print("Nothing to undo")
        
        # Reset all
        elif key == ord('r'):
            boxes = []
            current_lines = []
            current_point = None
            temp_point = None
            print("\nReset - all boxes and lines cleared")
            print(f"Start by drawing {get_current_line_name()} line")
        
        # Quit
        elif key == ord('q') or key == 27:  # 'q' or ESC
            break
    
    cv2.destroyAllWindows()
    
    if len(boxes) == 0:
        print("\nNo boxes created!")
        return None
    
    print(f"\n✓ Finished! Created {len(boxes)} box(es) from lines")
    for idx, (x, y, w, h) in enumerate(boxes):
        print(f"  Box {idx+1}: ({x}, {y}) size {w}×{h}px")
    
    return boxes


# Main execution
if __name__ == "__main__":
    image_path = "treeWisee.png"  #Image changer
    
    print("\n" + "="*70)
    print("TREE POROSITY ANALYSIS - MODE SELECTION")
    print("="*70)
    print("\nChoose your mode:")
    print("1. Automatic detection")
    print("2. Manual box drawing (drag rectangles)")
    print("3. Manual line drawing (draw 4 lines: top, bottom, left, right)")
    print("="*70)
    
    mode = input("\nEnter 1, 2, or 3: ").strip()
    
    if mode == "2":
        # Manual box mode
        print("\n=== MANUAL BOX MODE ===")
        manual_boxes = draw_manual_boxes(image_path)
        
        if manual_boxes:
            result = calculate_porosity_from_silhouette(
                image_path, 
                show_steps=True, 
                manual_boxes=manual_boxes
            )
    
    elif mode == "3":
        # Manual line mode
        print("\n=== MANUAL LINE MODE ===")
        manual_boxes = draw_manual_lines(image_path)
        
        if manual_boxes:
            result = calculate_porosity_from_silhouette(
                image_path, 
                show_steps=True, 
                manual_boxes=manual_boxes
            )
    
    else:
        # Automatic mode
        print("\n=== AUTOMATIC MODE ===")
        result = calculate_porosity_from_silhouette(
            image_path, 
            show_steps=True,
            min_tree_area=500  # Adjust this if needed (lower = more sensitive)
        )
    
    if result:
        print(f"\n✓ Analysis complete!")
        print(f"Overall porosity: {result['overall_porosity']:.1%}")
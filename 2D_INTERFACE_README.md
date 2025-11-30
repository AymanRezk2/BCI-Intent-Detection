# NEUROWHEEL 2D Interface

A clean, modern 2D interface design for the NeuroWheel application, featuring a Google Maps-style map with a blurred background, animated elements, and a minimalistic design.

## Features

- **Google Maps-Style Map**: Fixed 2D map view (no zoom functionality) with grid pattern and road-like structures
- **Blurred Background**: Animated gradient background with blur effect underneath the map
- **Animated NEUROWHEEL Text**: Center text with subtle floating animation and soft white shadow
- **Brain Illustration**: Stylized SVG brain icon with pulse animation next to the text
- **Interactive Elements**: 
  - Animated buttons with hover effects
  - Info labels with fade animations
  - Parallax effect on map buildings
- **Soft Lighting**: Animated light effects for depth
- **Responsive Design**: Adapts to different screen sizes

## Usage

### As a Streamlit Page

The interface is available as a Streamlit page. To access it:

1. Run your Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the 2D Interface page from the sidebar, or access it directly at:
   ```
   http://localhost:8501/2d_interface
   ```

### As a Standalone HTML File

You can also open `interface_2d.html` directly in a web browser for a standalone view.

## Design Elements

### Color Scheme
- **Background**: Animated gradient (purple, pink, blue tones) with blur
- **Map**: Light green/white with subtle grid
- **Text**: Dark gray (#2c3e50) with white shadow
- **Buttons**: White with gradient, semi-transparent

### Animations
- **NEUROWHEEL Text**: Subtle floating motion (3s cycle)
- **Brain Icon**: Pulse and slight rotation (2s cycle)
- **Labels**: Fade in/out effect (2s cycle)
- **Light Effects**: Moving radial gradients (8s cycle)
- **Buildings**: Parallax effect on mouse movement

### Typography
- **Primary Font**: Poppins (for headings)
- **Secondary Font**: Inter (for UI elements)

## Customization

To customize the interface, edit `pages/2d_interface.py`:

- **Colors**: Modify the CSS gradient values in the `.blurred-background` class
- **Map Style**: Adjust the `.map` background patterns
- **Animations**: Modify keyframe animations (e.g., `subtleFloat`, `brainPulse`)
- **Layout**: Adjust spacing and sizing in the CSS classes

## Browser Compatibility

The interface uses modern CSS features:
- CSS Grid and Flexbox
- CSS Animations
- Backdrop filters
- CSS Gradients

Works best in:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Notes

- The map is fixed and does not support zoom/pan functionality as per design requirements
- All animations are subtle and performance-optimized
- The interface is designed to be visually appealing while maintaining clarity and usability


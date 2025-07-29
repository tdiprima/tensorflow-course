# Computer Vision with CNNs - Learning Guide

## What You Should Have Learned

### Convolutional Neural Networks (CNNs) Fundamentals

1. **Why CNNs for Images**
   - **Local pattern detection**: CNNs detect features in specific areas rather than globally
   - **Translation invariance**: Can find patterns regardless of their location in the image
   - **Dense vs Convolutional**: Dense layers look at entire image, CNNs examine local patches
   - **Feature hierarchy**: Early layers detect edges, later layers detect complex shapes

2. **Image Data Structure**
   - **3D tensors**: (height, width, color_channels)
   - **Color channels**: RGB = 3 channels, grayscale = 1 channel
   - **CIFAR-10 format**: 32x32x3 (32x32 pixels, 3 color channels)
   - **Batch dimensions**: (batch_size, height, width, channels)

### CNN Architecture Components

3. **Convolutional Layers**
   - **Filters/Kernels**: Small patterns (like 3x3) that slide across the image
   - **Feature maps**: Output showing where patterns were detected
   - **Multiple filters**: Each layer learns different patterns (edges, textures, shapes)
   - **Depth increase**: More filters = deeper feature maps (32 → 64 → 64)

4. **Pooling Layers**
   - **MaxPooling2D**: Takes maximum value from each 2x2 region
   - **Downsampling**: Reduces spatial dimensions while keeping important features
   - **Translation invariance**: Small shifts don't affect the pooled result
   - **Computational efficiency**: Fewer parameters to process

5. **Architecture Pattern**
   ```
   Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → Flatten → Dense → Dense
   ```
   - **Convolutional base**: Extracts features from images
   - **Classifier head**: Dense layers that make final predictions

### Practical Implementation

6. **CIFAR-10 Dataset**
   - **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   - **50,000 training + 10,000 test images**
   - **32x32 color images**: Small but good for learning CNN concepts
   - **Data normalization**: `/255.0` to scale pixels to 0-1 range

7. **Building CNNs with Keras**
   - **Sequential model**: Layers stacked one after another
   - **Conv2D parameters**: (filters, kernel_size, activation, input_shape)
   - **MaxPooling2D**: (pool_size) typically (2,2)
   - **Flatten**: Convert 2D feature maps to 1D for dense layers

8. **Training Process**
   - **Epochs**: 4 epochs for CIFAR-10 (adjust based on complexity)
   - **Loss function**: SparseCategoricalCrossentropy for multi-class
   - **Optimizer**: Adam (adaptive learning rate)
   - **Validation**: Monitor performance on test data during training

### Advanced Techniques

9. **Data Augmentation**
   - **Problem**: Limited training data leads to overfitting
   - **Solution**: Create variations of existing images
   - **Transformations**: rotation, shifting, shearing, zooming, flipping
   - **ImageDataGenerator**: Keras tool for automatic augmentation
   - **Benefits**: Better generalization, more robust models

10. **Transfer Learning**
    - **Pre-trained models**: Use models trained on millions of images (ImageNet)
    - **MobileNetV2**: Efficient CNN architecture from Google
    - **Feature extraction**: Use pre-trained convolutional base, add custom classifier
    - **Freezing**: Don't retrain the pre-trained layers
    - **Fine-tuning**: Optionally adjust final layers for your specific problem

### Transfer Learning Implementation

11. **Cats vs Dogs with Transfer Learning**
    - **Base model**: MobileNetV2 without top classification layer
    - **Image preprocessing**: Resize to 160x160, normalize to [-1, 1]
    - **Custom classifier**: GlobalAveragePooling2D + Dense(1) for binary classification
    - **Low learning rate**: 0.0001 to avoid destroying pre-trained features

12. **Data Pipeline**
    - **TensorFlow Datasets**: tfds.load() for downloading datasets
    - **Data splitting**: 80% train, 10% validation, 10% test
    - **Batching**: Process data in batches of 32
    - **Shuffling**: Randomize order to improve training

### Model Evaluation and Visualization

13. **Performance Monitoring**
    - **Training vs validation accuracy**: Watch for overfitting
    - **Loss curves**: Should decrease over time
    - **Accuracy curves**: Should increase over time
    - **Model checkpointing**: Save best models during training

14. **Visualization Techniques**
    - **Plotting images**: matplotlib for displaying datasets
    - **Subplot arrangements**: Organized grid display of multiple images
    - **Training history**: Plot accuracy and loss curves
    - **Figure sizing**: Control plot dimensions with figsize

### Key Computer Vision Concepts

15. **Feature Learning Hierarchy**
    - **Layer 1**: Edges, lines, simple patterns
    - **Layer 2**: Shapes, textures, corners
    - **Layer 3**: Complex objects, combinations of shapes
    - **Deeper layers**: More abstract, task-specific features

16. **Spatial Dimension Changes**
    - **Convolution**: Slight reduction (32x32 → 30x30 with 3x3 filter)
    - **Pooling**: Significant reduction (30x30 → 15x15 with 2x2 pooling)
    - **Depth increase**: More features learned (32 → 64 → 64 filters)
    - **Final flatten**: Convert to 1D for classification

### What's Next

This foundation prepares you for:
- **Object detection**: Finding and localizing objects in images
- **Semantic segmentation**: Pixel-level classification
- **Advanced CNN architectures**: ResNet, EfficientNet, Vision Transformers
- **Custom datasets**: Training CNNs on your own image data
- **Real-time applications**: Deploying models for live image processing

Understanding CNNs is crucial for any computer vision project. The concepts of local feature detection, hierarchical learning, and transfer learning are fundamental to modern AI vision systems.